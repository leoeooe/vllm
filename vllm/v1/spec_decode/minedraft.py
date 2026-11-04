# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-independent coordination of batch-parallel speculative decoding.

The controller owns accepted prefixes, not model or KV state. One drafter and
independent verifier replicas execute immutable work snapshots. Only verification
commits tokens; drafts for another batch can run while verification is in flight.
"""

from collections import deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

WorkKind = Literal["prefill", "draft", "verify"]


def _tokens(values: Sequence[int]) -> tuple[int, ...]:
    tokens = tuple(values)
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Token IDs must be nonnegative integers")
    return tokens


@dataclass(frozen=True)
class BatchRequest:
    """A committed prefix and its remaining output-token budget."""

    request_id: str
    token_ids: tuple[int, ...]
    max_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a nonempty string")
        object.__setattr__(self, "token_ids", _tokens(self.token_ids))
        if not self.token_ids:
            raise ValueError("A request needs at least one prompt token")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")


@dataclass(frozen=True)
class WorkItem:
    """One operation; proposal rows correspond to requests in the same order."""

    kind: WorkKind
    batch_id: int
    verifier_id: int
    epoch: int
    requests: tuple[BatchRequest, ...]
    draft_token_ids: tuple[tuple[int, ...], ...] = ()


@dataclass
class _Batch:
    verifier_id: int
    requests: dict[str, BatchRequest]
    kind: WorkKind = "prefill"
    epoch: int = 0
    drafts: dict[str, tuple[int, ...]] = field(default_factory=dict)


class MineDraftController:
    """Schedule fair, exclusive work for one drafter and multiple verifiers.

    Batches are pinned to ``batch_id % num_verifiers``. A caller launches each
    returned work item asynchronously, then calls ``complete`` as results arrive.
    An empty ``next_work`` result means waiting when ``inflight`` is nonempty.
    This controller is owned by one event-loop thread.
    """

    def __init__(
        self,
        batches: Sequence[Sequence[BatchRequest]],
        num_verifiers: int,
        num_speculative_tokens: int,
    ) -> None:
        if type(num_verifiers) is not int or num_verifiers < 1:
            raise ValueError("num_verifiers must be a positive integer")
        if type(num_speculative_tokens) is not int or num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be a positive integer")
        self.num_verifiers = num_verifiers
        self.num_speculative_tokens = num_speculative_tokens
        self._batches: dict[int, _Batch] = {}
        self._request_batches: dict[str, int] = {}
        self._outputs: dict[str, list[int]] = {}
        self._ready: deque[int] = deque()
        self._inflight: dict[int, WorkItem] = {}
        for batch_id, requests in enumerate(batches):
            rows: dict[str, BatchRequest] = {}
            for request in requests:
                if request.request_id in self._request_batches:
                    raise ValueError("Request IDs must be unique across batches")
                self._request_batches[request.request_id] = batch_id
                self._outputs[request.request_id] = []
                rows[request.request_id] = request
            self._batches[batch_id] = _Batch(batch_id % num_verifiers, rows)
            if rows:
                self._ready.append(batch_id)

    @property
    def done(self) -> bool:
        """Whether all requests and outstanding actor calls have drained."""
        return not self._ready and not self._inflight

    @property
    def inflight(self) -> tuple[WorkItem, ...]:
        return tuple(self._inflight.values())

    @property
    def output_token_ids(self) -> dict[str, list[int]]:
        """Committed output only, including partial outputs of cancelled requests."""
        return {
            request_id: list(tokens) for request_id, tokens in self._outputs.items()
        }

    def next_work(self) -> list[WorkItem]:
        """Reserve available actors in FIFO order without blocking on busy ones."""
        draft_busy = any(work.kind == "draft" for work in self._inflight.values())
        busy_verifiers = {
            work.verifier_id for work in self._inflight.values() if work.kind != "draft"
        }
        result = []
        for _ in range(len(self._ready)):
            batch_id = self._ready.popleft()
            batch = self._batches[batch_id]
            if (batch.kind == "draft" and draft_busy) or (
                batch.kind != "draft" and batch.verifier_id in busy_verifiers
            ):
                self._ready.append(batch_id)
                continue
            batch.epoch += 1
            requests = tuple(batch.requests.values())
            drafts = (
                tuple(batch.drafts[request.request_id] for request in requests)
                if batch.kind == "verify"
                else ()
            )
            work = WorkItem(
                batch.kind, batch_id, batch.verifier_id, batch.epoch, requests, drafts
            )
            self._inflight[batch_id] = work
            if batch.kind == "draft":
                draft_busy = True
            else:
                busy_verifiers.add(batch.verifier_id)
            result.append(work)
        return result

    def complete(
        self,
        work: WorkItem,
        token_ids: Mapping[str, Sequence[int]],
        finished: Collection[str] = (),
    ) -> None:
        """Apply proposals or newly committed tokens from a matching actor call.

        Verifiers return accepted draft tokens followed by their correction or
        bonus token, or a shorter terminal result. Tokens beyond the remaining
        output budget are discarded. Cancelled rows may be present or omitted.
        Invalid results leave both request state and actor reservations intact.
        """
        if self._inflight.get(work.batch_id) != work:
            raise ValueError("Stale, duplicate, or unexpected work completion")
        batch = self._batches[work.batch_id]
        finished_ids = set(finished)
        snapshot_ids = {request.request_id for request in work.requests}
        if set(token_ids) - snapshot_ids or finished_ids - snapshot_ids:
            raise ValueError("Completion contains an unexpected request ID")
        if work.kind == "draft" and finished_ids:
            raise ValueError("Drafting cannot finish a target request")
        if set(batch.requests) - (set(token_ids) | finished_ids):
            raise ValueError("Completion is missing active requests")

        normalized: dict[str, tuple[int, ...]] = {}
        for request_id, request in batch.requests.items():
            tokens = _tokens(token_ids.get(request_id, ()))
            if work.kind == "draft":
                limit = min(self.num_speculative_tokens, request.max_tokens)
            elif work.kind == "prefill":
                limit = 1
            else:
                drafts = batch.drafts[request_id]
                limit = len(drafts) + 1
                if tokens[:-1] != drafts[: max(0, len(tokens) - 1)]:
                    raise ValueError("Verifier changed an accepted draft prefix")
            if len(tokens) > limit:
                raise ValueError("Completion exceeds the work's token limit")
            if not tokens and work.kind != "draft" and request_id not in finished_ids:
                raise ValueError("A nonterminal verifier result must make progress")
            normalized[request_id] = tokens

        del self._inflight[work.batch_id]
        if work.kind == "draft":
            batch.drafts = normalized
            batch.kind = "verify"
        else:
            for request_id, request in tuple(batch.requests.items()):
                tokens = normalized[request_id][: request.max_tokens]
                self._outputs[request_id].extend(tokens)
                remaining = request.max_tokens - len(tokens)
                if remaining == 0 or request_id in finished_ids:
                    del batch.requests[request_id]
                else:
                    batch.requests[request_id] = replace(
                        request,
                        token_ids=request.token_ids + tokens,
                        max_tokens=remaining,
                    )
            batch.drafts.clear()
            batch.kind = "draft"
        if batch.requests:
            self._ready.append(work.batch_id)

    def cancel(self, request_id: str) -> bool:
        """Remove an active request, retaining reservations until calls return."""
        batch_id = self._request_batches.get(request_id)
        if batch_id is None:
            return False
        batch = self._batches[batch_id]
        if batch.requests.pop(request_id, None) is None:
            return False
        batch.drafts.pop(request_id, None)
        if not batch.requests and batch_id not in self._inflight:
            self._ready.remove(batch_id)
        return True

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental offline MineDraft with independently placed vLLM engines.

Run under an ``if __name__ == '__main__'`` guard. Each engine owns its device
group, KV cache and graph buffers. Only committed token IDs cross processes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import multiprocessing as mp
import os
import signal
import threading
import time
import traceback
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection, wait
from typing import TYPE_CHECKING, Any

from vllm.v1.spec_decode.minedraft import (
    BatchRequest,
    MineDraftController,
    WorkItem,
)

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import SamplingParams

# Process-lifetime serialization of MineDraft's temporary spawn environment.
# Each constructor restores the caller's environment before returning from start.
_ENGINE_START_LOCK = threading.Lock()
_ENGINE_SHUTDOWN_GRACE_SECONDS = 30


@dataclass(frozen=True)
class _EngineConfig:
    model: str
    devices: tuple[int, ...]
    device: str
    draft: bool
    num_speculative_tokens: int
    kwargs: dict[str, Any]
    probe_method: str | None


def _tokenizer_signature(tokenizer: Any) -> str:
    content = (sorted(tokenizer.get_vocab().items()), tokenizer.eos_token_id)
    return hashlib.sha256(json.dumps(content).encode()).hexdigest()


def _actor_environment(config: _EngineConfig) -> dict[str, str]:
    visible_devices = (
        "ASCEND_RT_VISIBLE_DEVICES"
        if config.device == "npu"
        else "CUDA_VISIBLE_DEVICES"
    )
    environment = {
        visible_devices: ",".join(map(str, config.devices)),
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    if config.device == "npu":
        environment["HCCL_NPU_SOCKET_PORT_RANGE"] = os.environ.get(
            "HCCL_NPU_SOCKET_PORT_RANGE", "auto"
        )
    return environment


def _engine_process(connection: Connection, config: _EngineConfig) -> None:
    # A process group includes nested TP workers and permits bounded teardown
    # after a failed collective, without touching another engine's workers.
    os.setsid()
    os.environ.update(_actor_environment(config))
    llm = None
    try:
        # Imports happen after device visibility and engine ownership are set.
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        from vllm.outputs import RequestOutput
        from vllm.platforms import current_platform
        from vllm.sampling_params import RequestOutputKind
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.engine.core_client import InprocClient

        if current_platform.device_type != config.device:
            raise RuntimeError(
                f"MineDraft requested {config.device} but the active vLLM platform "
                f"uses {current_platform.device_type}; select the matching plugin"
            )
        kwargs = dict(config.kwargs)
        kwargs.update(
            model=config.model,
            tensor_parallel_size=len(config.devices),
            distributed_executor_backend="mp",
            async_scheduling=False,
            disable_log_stats=False,
        )
        if not config.draft:
            kwargs["speculative_config"] = {
                "method": "minedraft",
                "num_speculative_tokens": config.num_speculative_tokens,
            }
        llm = LLM(**kwargs)
        engine = llm.llm_engine
        if not isinstance(engine.engine_core, InprocClient):
            raise RuntimeError("MineDraft requires an in-process EngineCore per actor")
        scheduler = engine.engine_core.engine_core.scheduler
        if not isinstance(scheduler, Scheduler):
            raise RuntimeError("MineDraft requires the native vLLM scheduler")
        tokenizer = llm.get_tokenizer()
        if config.probe_method:
            llm.collective_rpc(config.probe_method)
        connection.send(
            (
                "ready",
                _tokenizer_signature(tokenizer),
                max(tokenizer.get_vocab().values()),
            )
        )
        internal_ids: dict[str, str] = {}
        output_lengths: dict[str, int] = {}

        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "diagnostics":
                workers = (
                    llm.collective_rpc(config.probe_method)
                    if config.probe_method
                    else []
                )
                connection.send(
                    (
                        "diagnostics",
                        {
                            "workers": workers,
                            "metrics": [asdict(m) for m in llm.get_metrics()],
                        },
                    )
                )
                continue
            work, sampling_params = payload
            started = time.monotonic()
            result: dict[str, Any]
            if work.kind == "draft":
                if not config.draft:
                    raise RuntimeError("Draft job sent to a verifier")
                params = [
                    SamplingParams(
                        temperature=0,
                        max_tokens=min(config.num_speculative_tokens, req.max_tokens),
                        ignore_eos=True,
                        detokenize=False,
                    )
                    for req in work.requests
                ]
                outputs = llm.generate(
                    [
                        TokensPrompt(prompt_token_ids=list(req.token_ids))
                        for req in work.requests
                    ],
                    params,
                    use_tqdm=False,
                )
                tokens = {
                    req.request_id: list(output.outputs[0].token_ids)
                    for req, output in zip(work.requests, outputs, strict=True)
                }
                result = {"tokens": tokens, "finished": [], "outputs": []}
            else:
                if config.draft:
                    raise RuntimeError("Verification job sent to the drafter")
                if work.kind == "prefill":
                    for req in work.requests:
                        params = sampling_params.clone()
                        params.output_kind = RequestOutputKind.CUMULATIVE
                        internal_ids[req.request_id] = engine.add_request(
                            req.request_id,
                            {"prompt_token_ids": list(req.token_ids)},
                            params,
                        )
                        output_lengths[req.request_id] = 0
                pending = {req.request_id: req for req in work.requests}
                proposals = dict(
                    zip((r.request_id for r in work.requests), work.draft_token_ids)
                )
                tokens = {}
                finished = []
                outputs = []
                while pending:
                    selected = {internal_ids[req_id] for req_id in pending}
                    drafts = {
                        internal_ids[req_id]: (len(req.token_ids), proposals[req_id])
                        for req_id, req in pending.items()
                        if req_id in proposals
                    }
                    scheduler.set_minedraft_batch(selected, drafts)
                    step_outputs = engine.step()
                    for output in step_outputs:
                        if not isinstance(output, RequestOutput):
                            raise RuntimeError("MineDraft requires generation outputs")
                        req_id = output.request_id
                        if req_id not in pending:
                            raise RuntimeError(
                                "Verifier advanced an unselected request"
                            )
                        all_tokens = list(output.outputs[0].token_ids)
                        new_tokens = all_tokens[output_lengths[req_id] :]
                        if not new_tokens and not output.finished:
                            continue
                        output_lengths[req_id] = len(all_tokens)
                        tokens[req_id] = new_tokens
                        outputs.append(output)
                        pending.pop(req_id)
                        if output.finished:
                            finished.append(req_id)
                            internal_ids.pop(req_id)
                            output_lengths.pop(req_id)
                scheduler.set_minedraft_batch(set(), {})
                result = {"tokens": tokens, "finished": finished, "outputs": outputs}
            result["started"] = started
            result["ended"] = time.monotonic()
            connection.send(("result", result))
    except EOFError:
        pass
    except BaseException:
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            connection.send(("error", traceback.format_exc()))
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        connection.close()


class _Actor:
    def __init__(self, config: _EngineConfig):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_engine_process, args=(child, config))
        self._closed = False
        self._ready = False
        self._failed = False
        try:
            # Spawn imports vllm (and therefore torch) before invoking the target.
            # Device visibility must be inherited by that fresh interpreter.
            with _ENGINE_START_LOCK:
                environment = _actor_environment(config)
                previous = {name: os.environ.get(name) for name in environment}
                os.environ.update(environment)
                try:
                    self.process.start()
                finally:
                    for name, value in previous.items():
                        if value is None:
                            os.environ.pop(name, None)
                        else:
                            os.environ[name] = value
        except BaseException:
            self.connection.close()
            raise
        finally:
            child.close()

    def receive(self, timeout: float) -> tuple:
        ready = wait([self.connection, self.process.sentinel], timeout)
        if self.connection not in ready:
            self._failed = True
            if self.process.sentinel in ready:
                raise RuntimeError(f"MineDraft engine exited: {self.process.exitcode}")
            raise TimeoutError("MineDraft engine timed out")
        try:
            message = self.connection.recv()
        except EOFError as exc:
            self._failed = True
            raise RuntimeError("MineDraft engine closed its connection") from exc
        if message[0] == "error":
            self._failed = True
            raise RuntimeError(f"MineDraft engine failed:\n{message[1]}")
        if message[0] == "ready":
            self._ready = True
        return message

    def close(self, *, graceful: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        pid = self.process.pid
        assert pid is not None
        try:
            if self.process.is_alive():
                with contextlib.suppress(BrokenPipeError, OSError):
                    self.connection.send(("close", None))
            # The native executor itself waits for workers and then unlinks its
            # shared-memory queues. Let it finish before killing the whole group.
            grace = (
                _ENGINE_SHUTDOWN_GRACE_SECONDS
                if graceful and self._ready and not self._failed
                else 0
            )
            self.process.join(timeout=grace)
            # TP workers can outlive an already-exited actor. Its process group
            # must be cleaned up even when the leader is no longer alive.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGTERM)
            if self.process.is_alive():
                # Startup may have failed before the child reached setsid().
                self.process.terminate()
            self.process.join(timeout=5)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
            if self.process.is_alive():
                self.process.kill()
            self.process.join(timeout=5)
            if self.process.is_alive():
                raise RuntimeError("MineDraft engine could not be terminated")
        finally:
            self.connection.close()


class MineDraftLLM:
    """Offline batch-parallel speculative decoding with a shared greedy drafter.

    This experimental API accepts tokenized text prompts and ``n=1`` sampling.
    It is synchronous to its caller; draft and verifier actors run concurrently.
    ``num_batches`` is independent of the number of verifier replicas.
    Graph capture uses each engine's native compilation configuration.
    """

    def __init__(
        self,
        draft_model: str,
        target_model: str,
        draft_devices: Sequence[int],
        verifier_devices: Sequence[Sequence[int]],
        *,
        num_batches: int = 4,
        num_speculative_tokens: int = 3,
        device: str = "npu",
        enforce_eager: bool = False,
        timeout: float = 600,
        draft_engine_kwargs: dict[str, Any] | None = None,
        probe_method: str | None = None,
        **engine_kwargs: Any,
    ) -> None:
        groups = [tuple(draft_devices), *(tuple(g) for g in verifier_devices)]
        if device not in ("npu", "cuda"):
            raise ValueError("device must be npu or cuda")
        if not verifier_devices or any(not group for group in groups):
            raise ValueError("Each engine must have a nonempty device group")
        devices = [idx for group in groups for idx in group]
        if any(type(idx) is not int or idx < 0 for idx in devices):
            raise ValueError("Device IDs must be nonnegative integers")
        if len(set(devices)) != len(devices):
            raise ValueError("Draft and verifier device groups must be disjoint")
        if (
            type(num_batches) is not int
            or num_batches < len(verifier_devices)
            or num_batches < 2
        ):
            raise ValueError("Use at least two batches and one per verifier")
        if type(num_speculative_tokens) is not int or num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be a positive integer")
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be finite and positive")
        reserved = {
            "model",
            "tensor_parallel_size",
            "distributed_executor_backend",
            "async_scheduling",
            "speculative_config",
            "disable_log_stats",
            "pipeline_parallel_size",
            "data_parallel_size",
            "scheduler_cls",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        }
        for kwargs in (engine_kwargs, draft_engine_kwargs or {}):
            if forbidden := reserved.intersection(kwargs):
                raise ValueError(
                    f"MineDraft owns these engine options: {sorted(forbidden)}"
                )
            for name in ("max_model_len", "max_num_seqs"):
                if name in kwargs and (
                    type(kwargs[name]) is not int or kwargs[name] < 1
                ):
                    raise ValueError(f"{name} must be a positive integer")
            if "stream_interval" in kwargs and (
                type(kwargs["stream_interval"]) is not int
                or kwargs["stream_interval"] != 1
            ):
                raise ValueError("MineDraft requires stream_interval=1")
            if kwargs.get("runner", "auto") not in ("auto", "generate") or (
                kwargs.get("convert", "auto") not in ("auto", "none")
            ):
                raise ValueError("MineDraft engines must perform text generation")
            unsupported = {
                "enable_lora",
                "kv_transfer_config",
                "watermark_config",
                "enable_dbo",
            }
            if enabled := sorted(name for name in unsupported if kwargs.get(name)):
                raise ValueError(f"MineDraft does not support these options: {enabled}")
        self.num_batches = num_batches
        self.num_speculative_tokens = num_speculative_tokens
        self.timeout = timeout
        self.max_model_len = engine_kwargs.get("max_model_len", 2048)
        self.max_num_seqs = engine_kwargs.get("max_num_seqs", 64)
        self.timeline: list[dict[str, Any]] = []
        self._actors: list[_Actor] = []
        self._busy = False
        self._closed = False
        common = {
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "stream_interval": 1,
            "enable_prefix_caching": True,
            "enforce_eager": enforce_eager,
            **engine_kwargs,
        }
        if not enforce_eager and "compilation_config" not in common:
            common["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY"}
        draft_kwargs = {**common, **(draft_engine_kwargs or {})}
        if draft_kwargs.get("max_model_len") != self.max_model_len:
            raise ValueError("Draft and target must use the same max_model_len")
        try:
            for index, group in enumerate(groups):
                config = _EngineConfig(
                    model=draft_model if index == 0 else target_model,
                    devices=group,
                    device=device,
                    draft=index == 0,
                    num_speculative_tokens=num_speculative_tokens,
                    kwargs=draft_kwargs if index == 0 else common,
                    probe_method=probe_method,
                )
                self._actors.append(_Actor(config))
            ready = [actor.receive(timeout) for actor in self._actors]
            if any(message[0] != "ready" for message in ready):
                raise RuntimeError("Invalid MineDraft engine startup response")
            if len({message[1] for message in ready}) != 1:
                raise ValueError("Draft/target tokenizer mappings or EOS IDs differ")
            self._max_token_id = ready[0][2]
        except BaseException:
            self._close_actors(graceful=False)
            raise

    def generate(
        self, prompts: Sequence[Sequence[int]], sampling_params: SamplingParams
    ) -> list[RequestOutput]:
        """Generate outputs in prompt order, preserving target rejection sampling."""
        if self._closed or self._busy:
            raise RuntimeError("MineDraftLLM is closed or already generating")
        if sampling_params.n != 1:
            raise ValueError("MineDraft currently supports n=1")
        if sampling_params.max_tokens is None or sampling_params.max_tokens < 1:
            raise ValueError("Specify a positive max_tokens")
        if sampling_params.prompt_logprobs is not None:
            raise ValueError("MineDraft does not yet support prompt_logprobs")
        if sampling_params.structured_outputs is not None:
            raise ValueError("MineDraft does not yet support structured outputs")
        if sampling_params.stream_interval is not None and (
            type(sampling_params.stream_interval) is not int
            or sampling_params.stream_interval != 1
        ):
            raise ValueError("MineDraft requires sampling stream_interval=1")
        prompt_tokens = [tuple(prompt) for prompt in prompts]
        for prompt in prompt_tokens:
            if not prompt or any(
                type(token) is not int or not 0 <= token <= self._max_token_id
                for token in prompt
            ):
                raise ValueError(
                    "Prompts must be nonempty sequences of valid token IDs"
                )
            if len(prompt) + sampling_params.max_tokens > self.max_model_len:
                raise ValueError("Prompt plus max_tokens exceeds max_model_len")
        if not prompt_tokens:
            return []
        invocation = uuid.uuid4().hex
        requests = [
            BatchRequest(f"{invocation}-{idx}", prompt, sampling_params.max_tokens)
            for idx, prompt in enumerate(prompt_tokens)
        ]
        batches = [
            requests[index :: self.num_batches] for index in range(self.num_batches)
        ]
        resident_counts = [0] * (len(self._actors) - 1)
        for batch_id, batch in enumerate(batches):
            resident_counts[batch_id % len(resident_counts)] += len(batch)
        if max(resident_counts) > self.max_num_seqs:
            raise ValueError(
                "All requests assigned to a verifier must fit max_num_seqs; "
                "increase max_num_seqs or reduce the number of prompts"
            )
        controller = MineDraftController(
            batches, len(self._actors) - 1, self.num_speculative_tokens
        )
        self._busy = True
        pending: dict[Connection, tuple[_Actor, WorkItem, float]] = {}
        final_outputs = {}
        try:
            while not controller.done:
                for work in controller.next_work():
                    actor = self._actors[
                        0 if work.kind == "draft" else work.verifier_id + 1
                    ]
                    actor.connection.send(("work", (work, sampling_params)))
                    pending[actor.connection] = (actor, work, time.monotonic())
                if not pending:
                    raise RuntimeError("MineDraft scheduler made no progress")
                ready = wait(list(pending), timeout=min(self.timeout, 1))
                for connection, (actor, work, submitted) in list(pending.items()):
                    if connection not in ready:
                        if not actor.process.is_alive():
                            raise RuntimeError(
                                "MineDraft engine exited during generation"
                            )
                        if time.monotonic() - submitted > self.timeout:
                            raise TimeoutError("MineDraft work exceeded timeout")
                        continue
                    message = actor.receive(self.timeout)
                    if message[0] != "result":
                        raise RuntimeError("Invalid MineDraft work response")
                    result = message[1]
                    controller.complete(work, result["tokens"], result["finished"])
                    for output in result["outputs"]:
                        if output.finished:
                            final_outputs[output.request_id] = output
                    self.timeline.append(
                        {
                            "kind": work.kind,
                            "batch_id": work.batch_id,
                            "verifier_id": work.verifier_id,
                            "epoch": work.epoch,
                            "started": result["started"],
                            "ended": result["ended"],
                        }
                    )
                    pending.pop(connection)
            return [final_outputs[req.request_id] for req in requests]
        except BaseException:
            # A partially advanced native scheduler cannot be reused safely.
            self._close_actors(graceful=False)
            raise
        finally:
            self._busy = False

    def get_diagnostics(self) -> dict[str, Any]:
        """Return worker probes, native metrics and host operation intervals.

        Host intervals prove actor concurrency; use a device profiler to measure
        accelerator kernel overlap and speedup.
        """
        if self._closed or self._busy:
            raise RuntimeError("Diagnostics require an idle, open MineDraftLLM")
        results = []
        try:
            for actor in self._actors:
                actor.connection.send(("diagnostics", None))
            for actor in self._actors:
                message = actor.receive(self.timeout)
                if message[0] != "diagnostics":
                    raise RuntimeError("Invalid diagnostics response")
                results.append(message[1])
        except BaseException:
            self._close_actors(graceful=False)
            raise
        return {
            "draft": results[0],
            "verifiers": results[1:],
            "timeline": self.timeline,
        }

    def close(self) -> None:
        """Release engines, including all tensor-parallel worker processes."""
        self._close_actors(graceful=True)

    def _close_actors(self, *, graceful: bool) -> None:
        if self._closed:
            return
        self._closed = True
        errors = []
        for actor in self._actors:
            try:
                actor.close(graceful=graceful)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("Failed to close a MineDraft engine") from errors[0]

    def __enter__(self) -> MineDraftLLM:
        return self

    def __exit__(self, *args: Any) -> None:
        self._close_actors(graceful=args[0] is None)

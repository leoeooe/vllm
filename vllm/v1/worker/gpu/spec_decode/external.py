# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator


class ExternalSpeculator(BaseSpeculator):
    """Receive deterministic proposals without a draft model on the verifier."""

    supports_mm_inputs = False
    draft_logits = None

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.device = device
        self.num_speculative_steps = vllm_config.num_speculative_tokens
        self.drafts = torch.full(
            (
                vllm_config.scheduler_config.max_num_seqs,
                self.num_speculative_steps,
            ),
            -1,
            dtype=torch.int64,
            device=device,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass

    def capture(self) -> None:
        pass

    def propose(
        self, input_batch: InputBatch, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # Only profiling calls this method. Real steps obtain their candidates
        # before the target forward, outside the captured model graph.
        return self.drafts[: input_batch.num_reqs]

    def load_drafts(
        self,
        proposals: dict[str, list[int]],
        req_id_to_index: dict[str, int],
        draft_tokens: torch.Tensor,
    ) -> None:
        """Populate stable request-state storage before graph replay."""
        if not proposals:
            return
        indices = []
        rows = []
        for req_id, tokens in proposals.items():
            if len(tokens) > self.num_speculative_steps:
                raise ValueError("External proposal exceeds configured draft length.")
            indices.append(req_id_to_index[req_id])
            rows.append(tokens + [-1] * (self.num_speculative_steps - len(tokens)))
        # The compact host payload is staged once per batch. The graph's input
        # buffers and request-state allocation are never replaced.
        index = torch.tensor(indices, dtype=torch.int64, device=self.device)
        values = torch.tensor(rows, dtype=torch.int64, device=self.device)
        draft_tokens.index_copy_(0, index, values)

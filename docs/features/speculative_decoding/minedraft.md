# MineDraft batch-parallel engines (experimental)

This branch implements an offline, multi-engine adaptation of
[MineDraft](https://arxiv.org/html/2603.18016v1). The original implementation
targets vLLM 0.9.2; this implementation uses V1 and Model Runner V2.

Two or more independent request batches alternate between drafting and
verification. A request is verified only after its draft completes, while
another batch may draft concurrently. A single draft engine can serve multiple
verifier engines. Each verifier has its own TP group and retains its request
state and KV cache across rounds.

## Baseline and branch

The paired development baseline is:

| Repository | Commit |
| --- | --- |
| vLLM | `24420b7a341af37a9bc3186e6fc6062512e0aa58` |
| vLLM Ascend | `3b715ff78066d22b95be6f4a84e861a44930b25a` |

Both development branches are named `feat/minedraft-multibatch`. The Ascend
commit was fetched from upstream main on November 4, 2026; its
`.github/vllm-main-verified.commit` selects the vLLM commit above. Independent
vLLM main had already moved to a newer interface, so the verified pairing is
intentional. Preserve both repositories' commit IDs when recording experiments.

## Offline API

Use `MineDraftLLM`, which controls the native schedulers. Passing
`method="minedraft"` to an ordinary `LLM.generate()` or `vllm serve` does not
provide the coordinator and is unsupported.

```python
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.entrypoints.minedraft import MineDraftLLM


def main():
    target = "/models/Qwen3-32B"
    draft = "/models/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(target)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for text in [
            "Explain how a rainbow forms.",
            "Write a Python binary search.",
            "计算 17 乘以 23。",
            "用三句话解释机器学习。",
        ]
    ]
    with MineDraftLLM(
        draft_model=draft,
        target_model=target,
        draft_devices=[0],
        verifier_devices=[[1, 2], [3, 4]],
        num_batches=4,
        num_speculative_tokens=3,
        device="npu",
        enforce_eager=False,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=64,
        gpu_memory_utilization=0.8,
        worker_extension_cls=(
            "vllm_ascend.worker.v2.minedraft.MineDraftWorkerExtension"
        ),
        probe_method="minedraft_probe",
    ) as llm:
        outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=64))
        for output in outputs:
            print(output.outputs[0].text)
        diagnostics = llm.get_diagnostics()
        print(diagnostics)


if __name__ == "__main__":
    main()
```

Device IDs select the physical devices visible to the host process. The groups
must be disjoint. Each child inherits its device visibility before importing
vLLM and uses a separate process group. Do not launch this entry point with
`torchrun`. For a one-verifier dual-batch experiment, use
`verifier_devices=[[1, 2]]` and `num_batches=2`. Batch `i` belongs to verifier
`i % num_verifiers`; increasing batches does not load more model replicas.

The context manager releases engines and their TP workers. An actor exception
or work timeout closes the cluster; instantiate a new cluster after a failure.
Construct and operate the cluster from one coordinating thread. Startup briefly
sets device and engine environment variables while spawning each child, then
restores them; avoid concurrent environment-sensitive work in the parent during
construction. Calls to `generate()` must be serial. Returned outputs follow
prompt order.

NPU actors default `HCCL_NPU_SOCKET_PORT_RANGE` to `auto` to avoid fixed-port
collisions between independent TP groups. An explicit parent setting is preserved.

## Graph execution

The draft engine is an ordinary autoregressive vLLM engine, so its decode
steps use the ordinary model graph path. Verifier engines use native
speculative verification graphs. With `enforce_eager=False`, the wrapper
defaults to `FULL_DECODE_ONLY`; `compilation_config` can select another mode
supported by the backend. Prefills remain eager with this default.

The controller, IPC, rejection bookkeeping and external token staging are
outside captured model graphs. Candidate token IDs are copied into existing
request-state buffers before replay. Draft and verifier graphs have separate
buffer ownership; verification uses its own speculative query shapes.

Setting a graph configuration alone is not proof of replay. The Ascend probe
reports `captured_graphs`, `capture_token_sizes` and `graph_replays` for each TP
rank. Run enough decode rounds to exercise those shapes. With a draft length
of one, draft generation may consist only of prefill and provide no decode
graph replay.

## Sampling and lifecycle

The drafter uses deterministic greedy decoding. The verifier uses vLLM's
existing rejection sampler with `draft_logits=None`, representing a point-mass
proposal. Target sampling parameters are retained; random draft distributions
are not implemented. Tokenizer vocabulary mappings and EOS IDs must match.

Only verified tokens update the controller's committed prefix. Each immutable
work item has a batch epoch; stale and duplicate completions are rejected.
The native scheduler checks candidate prefix lengths and freezes unselected
requests. The worker stages actual candidate IDs, rather than only scheduling
candidate counts. Native verification handles EOS, length limits and rejected
KV positions. The pure controller also supports cancellation and draining
outstanding work; the public offline API currently has no per-request cancel
or streaming endpoint.

The draft service currently regenerates from the committed prefix using
ordinary `LLM.generate()` and prefix caching. It does not retain a speculative
draft request across rounds. Unaligned prefix tails can therefore be
recomputed, and host IPC introduces synchronization. Performance parity with
the paper, end-to-end speedup and device kernel overlap require measurement.

## Current scope and validation

Supported inputs are tokenized text prompts, one completion per request,
fixed draft length, synchronous scheduling within each engine, and independent
TP verifier replicas. PP, DP, context parallelism, LoRA, KV transfer,
watermarking, DBO, structured outputs, prompt logprobs and multimodal inputs
are outside this experimental path. The total resident request count on each
verifier must fit `max_num_seqs`; split larger datasets into successive
`generate()` calls. Use `stream_interval=1` so each verification exposes the
new committed frontier to the coordinator.

The source includes controller tests, native scheduler and candidate-buffer
tests, and an Ascend example/hardware test. CPU control tests do not establish
NPU graph correctness, numerical equivalence or throughput. Validate on the
intended hardware with:

1. One verifier and two batches, then two TP2 verifiers and four batches.
2. Eager execution and graph execution with actual replay counters.
3. Greedy output comparison against ordinary target decoding, plus EOS,
   short output limits, uneven batches, and cache pressure.
4. Random target sampling distribution tests before using that mode in a
   benchmark; a fixed seed alone does not imply identical samples across
   different batch layouts.
5. A device profiler and matched concurrency/token budgets for baseline,
   ordinary SD and MineDraft. Host `timeline` intervals establish overlapping
   actor operations, not overlapping accelerator kernels.

Greedy output can differ under different batches because of floating-point
rounding. Use a supported batch-invariant configuration for exact comparisons,
and investigate mismatches rather than relaxing checks automatically.

### Development checks without accelerator devices

From the vLLM repository, the control and transport tests can run without model
weights:

```bash
.venv/bin/python tests/v1/spec_decode/test_minedraft.py -v
.venv/bin/python tests/v1/spec_decode/test_minedraft_runtime.py -v
```

To exercise the real scheduler and candidate-buffer objects with the installed
CPU PyTorch environment, use:

```bash
VLLM_TARGET_DEVICE=cpu \
VLLM_CACHE_ROOT=/home/l00998189/.cache/vllm \
TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_PLUGINS='' \
.venv/bin/python -m pytest \
    tests/v1/core/test_scheduler.py \
    tests/v1/worker/test_draft_tokens_handler.py \
    -k 'minedraft or external_drafts' -q
```

The same environment can run the two control/transport files under ordinary
pytest. The transport suite replaces only the external engine service and
spawns real processes; it does not load real model weights. Its startup deadline
allows for importing installed accelerator packages, while a separate test
checks short RPC timeout handling. These checks supplement the hardware tests;
they do not replace NPU numerical, replay, or performance validation.

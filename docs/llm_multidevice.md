# Experimental layer-partitioned LLM serving

`tinygrad.llm` can opt into contiguous whole-layer placement on explicit devices.
This is a **CPU-validated experimental software path**, not certification of any
eGPU, enclosure, driver, GPU kernel, physical memory budget, or throughput.
Independent CPU logical devices exercise ownership and execution; they do not
simulate PCIe/Thunderbolt hardware.

This build is the `v0.14.0` release plus the placement code. The runtime, the
engine and `tensor.py` are byte-equal to `v0.14.0`. Without placement options,
existing text loading and serving are unchanged.

## Local-only configuration

The model must already exist locally. Explicit placement never resolves a model
alias or downloads weights. For a compatible model with **four transformer blocks**:

```sh
DEV=CPU python -m tinygrad.llm --model /models/local-four-block.gguf \
  --devices CPU,CPU:1 --layer-counts 1,3 --chunk-size 4 --placement-check
```

`--placement-check` validates the complete manifest and prints per-owner accounting,
then exits before model construction, tokenizer setup, device opening, or warmup.
It reports **fit unknown**, not a physical fit decision. Adjust counts to the actual
model; positive counts must cover every block. `CPU` and `CPU:00` are the same device.

Remove `--placement-check` for interactive inference or add `--serve 8000` for
serialized HTTP. The server is unauthenticated: use only a trusted network or a
secured gateway. Do not invoke the CLI without an explicit local model when testing
offline: legacy mode still offers downloadable model aliases.

Options:

- `--devices`: ordered, distinct, same-backend IDs (CPU, PYTHON, AMD, NV, CUDA).
  Parser acceptance is not GPU compatibility proof. `DEV` separately selects the
  compiler/interface. Mixed backends are rejected.
- `--layer-counts`: positive contiguous block counts, one per device; no auto-balancing.
- `--transfer host` (default): synchronous host-staged activation copies, with
  explicit source storage retained through destination synchronization. This is a
  reference path, not an overlapped/pinned transfer optimization.
- `--transfer native`: runtime `.to()` copying with synchronization; this does
  **not** guarantee peer access. Failure does not retry using another route.
- `--chunk-size`: fixed prefill chunk size, 1–32 and no larger than effective
  context; default 32. Single-token and full-chunk stages have separate JITs;
  other tails execute eagerly to bound capture count.

Reload to change placement or context. `REALIZE=1` is rejected; `HALF=0/1` retains
existing lazy weight-cast semantics. Custom quantized Linear kernels are disabled
per placed model, without modifying legacy models' settings.

## Supported model envelope

Only little-endian GGUF v2/v3, including validated multipart local files, with
`general.architecture=llama` and ordinary dense full-context attention. Batch one.
Full-head RoPE requires equal even Q/K/V/RoPE head dimensions,
`dim = heads * head_dim`, and valid GQA divisibility. Missing KV-head count defaults
to query heads. Effective context is capped by metadata as in legacy loading.

Supported weight types: F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K and Q6_K. An omitted
`general.quantization_version` is accepted for compatibility; an explicit value
must be integer 2. Unknown versions, formats, shapes and computation-changing
metadata fail before model construction or payload copying.

No MoE, MLA, recurrent layers, QKV biases, Q/K normalization, sliding windows,
attention gates/clamping/ALiBi, parallel residual, predictor layers, non-reference
layout, or nontrivial RoPE scaling. This excludes many otherwise valid GGUF models.
Legacy mode retains its architectures; placement never silently falls back to it.

Files are read-only and must remain trusted and unchanged during loading.
Bounds, alignment, split indices/counts, repeated metadata, names/shapes/types and
stat identities are checked. Stat checks detect ordinary mutation, not a malicious
writer preserving stat fields. Limits include 256 MiB encoded metadata, one million
tensors/keys, ten million array elements, 16 MiB value strings, depth eight and
rank one through four. These are implementation limits, not universal GGUF limits.

## Ownership and memory

Embedding stays first; output normalization/projection stay last. Each block group
retains local weights, FP16 KV and realized FP32 RoPE. Only FP32 activations cross
stage boundaries; the sampled ID returns to host. One token visits stages
sequentially. More devices may provide capacity, not lower single-request latency.

Each exact packed payload is copied to its final owner before lazy decoding;
the whole file is not first realized on GPU 0. Tied embedding/output weights share
same-owner backing or use immutable endpoint replicas counted once per owner.
There is no training or weight-update mechanism.

Raw payload bytes are exact; FP16 KV reserves
`4 * owned_blocks * context * kv_heads * head_dim` bytes. RoPE is shared per owner
and configuration. Boundary/logit provisions are estimates. Compilation, decoded
scratch, backend staging, allocator rounding and retained caches remain unknown.
Active loading can hold multiple copies of the largest tensor; total host RAM is
**not** bounded by it because allocator caches retain previous sizes. Neither
packed-file size nor summed nominal VRAM proves a model fits.

## Cache and failure lifecycle

Only one generation owns a placed model. Concurrent library callers receive a
busy error. Close retained iterators in `finally`, including early exit:

```python
from tinygrad.llm.model import Transformer
from tinygrad.llm.placement import LayerPlacement

model, metadata = Transformer.from_gguf(
  '/models/local-four-block.gguf', realize=False,
  placement=LayerPlacement(('CPU', 'CPU:1'), (1, 3), chunk_size=4))
gen = model.generate([1, 2, 3])  # valid IDs from this model's tokenizer
try:
  token = next(gen)
finally:
  gen.close()
```

An explicit `generate(..., chunk_size=...)` must equal the configured placed chunk.
Clean token-boundary close retains only committed prefix state, excluding the new
emitted token until processed. Direct `forward`/`__call__` invalidates reusable-prefix
metadata before writes and realizes its result in an exclusive transaction.
Warmup preserves KV buffer identity.

Inference/transfer/sampling failure after admission marks the model failed and
requires reload; reset does not revive it. Invalid direct input values leave the
prefix unchanged; backend errors while validating those inputs are wrapped but do
not poison model state before mutation.

## HTTP serving

A placed server answers before sending any header when it cannot serve:

- `400` for a `temperature` that is not a finite nonnegative number, or a
  `max_tokens`/`max_completion_tokens` that is not a positive integer;
- `503` once the model has failed (reload required), `409` while it is in use.

Placed SSE acquires generation and buffers its first delta before sending headers,
so first-step failures return an error status; later failures abort without normal
finish or `[DONE]`. Context exhaustion reports `length`. Failure logs carry only the
stage, device and operation, never backend text or prompts. A socket disconnect at a
completed token boundary closes cleanly; it cannot cancel an in-flight synchronous kernel.
Requests to an unplaced model are answered exactly as before.

Sampling retains Gumbel-max, including temperature zero handling. Placement samples
only after final prefill, not discarded intermediate chunks; identical stochastic
sequences across placements/warmups are not promised.

## Offline verification

Use Python 3.12+, a working CPU compiler, NumPy, pytest/xdist and test-only GGUF.
Tests construct nonzero synthetic models. Explicit `DEV=CPU` prevents GPU selection;
hosts requiring a compiler target can use the [DEV configuration](env_vars.md#dev-variable).

```sh
python -m ruff check tinygrad/llm
git diff --stat --exit-code v0.14.0 -- tinygrad/runtime tinygrad/engine tinygrad/tensor.py
DEV=CPU BEAM=0 python -m pytest -x -q -n auto \
  test/unit/test_gguf_placement.py \
  test/unit/test_llm_placement.py \
  test/unit/test_llm_placement_execution.py \
  test/unit/test_llm_placement_server.py
```

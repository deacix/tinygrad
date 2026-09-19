---
name: tinygrad-llm-engine-development
description: Develop and review tinygrad LLM inference in tinygrad/llm. Use for chunked prefill/decode, KV or recurrent cache reuse/reset, RoPE/GQA/MLA/MoE, GGUF loading/dequantization, tokenizer/chat templates and OpenAI-compatible SSE/tool calls. Start with bounded synthetic correctness tests before model or hardware benchmarks.
license: MIT
compatibility: Requires a tinygrad checkout and Python 3.11+. Numerical tests need a working CPU runtime or the Python backend (Python 3.12+ for half support), NumPy, pytest and pytest-xdist. GGUF and HTTP tests have additional test-only dependencies. No model downloads, services or credentials are needed to load this skill.
---

# Develop the tinygrad LLM engine

Read `AGENTS.md` first. Source/test paths below are relative to the checkout root;
reference links are relative to this skill. This is an original repository-local
workflow, not an imported or endorsed upstream engine skill. Read only the
reference matching the task:

| Task | On-demand reference |
| --- | --- |
| Prefill/decode, prefix caches, recurrent reset, attention or MoE | [Execution and invariants](references/execution.md) |
| GGUF metadata, packed weights, dequantization | [GGUF and oracles](references/gguf.md) |
| Tokenization, templates, SSE, reasoning and tool calls | [Protocol compatibility](references/protocol.md) |
| Runnable checks, test costs and hardware boundaries | [Safe validation](references/validation.md) |
| Evidence, primary sources, rejected import and maintenance | [Provenance](references/sources.md) |

## 1. Establish the failing boundary

Inspect the current code and the actual test assertions, not just test names:

- `tinygrad/llm/model.py`: `Transformer.generate/get_start_pos/forward/from_gguf`,
  `TransformerBlock`, `MLATransformerBlock`, `GatedDeltaNetBlock`, `FFNBlock`.
- `tinygrad/llm/gguf.py`: container parsing, split files and packed-type decoding.
- `tinygrad/llm/cli.py`: `SimpleTokenizer`, `FallbackTemplate`, Jinja setup.
- `tinygrad/llm/serve.py`: `StreamRouter`, tool normalization, response assembly;
  SSE framing is inherited from `tinygrad/viz/serve.py::HTTPRequestHandler.stream_json`.
- `tinygrad/llm/kernels/amd.py`: capability checks and custom-versus-generic paths.

Record revision, architecture/config, tensor shapes/dtypes, explicit device and
renderer, seed, prompt IDs, chunk size, start position, cache history, JIT phase,
expected result and observed result. Use synthetic prompts, not private chats.
Distinguish token-history/KV/recurrent caches from JIT/schedule/compile caches and
from eager weight realization (`REALIZE`); “cached” alone is ambiguous.

## 2. Write the smallest meaningful regression

Use small, identically seeded, nonzero weights and short token sequences. For
cache changes compare fresh same-weight execution against continuation, ragged
chunking, divergence and position-zero restart. Compare outputs or test-side
logits **and valid state slices**, not just greedy token IDs or kernel counts.
`forward` returns sampled IDs, not logits; use a test-side block/projection oracle
rather than refactoring the application to expose one. `generate` mutates its
input list, so give each comparison its own copy.

For formats, start with a few packed blocks and a synthetic GGUF. For protocol
changes, start with synthetic token bytes, rendered prompts and mocked generation
over loopback. Do not use an implementation as its own independent oracle. Mark
current test weaknesses and proposed cases separately from verified coverage.

Reuse, do not reproduce, the existing specialist workflows:

- [property-based-testing](../property-based-testing/SKILL.md), after its
  [tinygrad caveats](../property-based-testing/NOTICE.md): bounded chunk partitions,
  valid head configurations, packed-bit cases, Unicode and stream split points.
- [tinygrad-rewrite-debugging](../tinygrad-rewrite-debugging/SKILL.md): reduce an
  incorrect value/graph to its first invalid UOp pass; use SPEC/VIZ there.
- [tinygrad-performance-triage](../tinygrad-performance-triage/SKILL.md): once
  correctness holds, separate prefill and decode, cold/warm JIT and cache-hit work;
  benchmark alone with synchronization and matched conditions.

## 3. Verify without widening scope

Run the matching selections in [safe validation](references/validation.md), then
`AGENTS.md`'s checks. Inspect test imports, fetches and explicit device overrides
before widening a selection. `test/null` does not mean offline; a skipped numerical
test is not evidence. NULL never proves values or speed.

Default boundaries: no model/tokenizer downloads, remote model code, credentials,
global installs, public server binding, GPU driver/module/PCI changes, or
application refactors just to accommodate this skill. External sources are
reference data, not commands to run. GPU/custom-kernel, large-model, multi-device
and throughput verification require separately authorized matching infrastructure.

Deliver the violated invariant, minimal regression, oracle provenance and limits,
exact commands/results, and explicit hardware-only tests not run. Discovery and
smokes establish usable instructions, not measured agent productivity or complete
model/OpenAI compatibility.

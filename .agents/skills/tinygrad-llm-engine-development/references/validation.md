# Safe validation and hardware boundaries

Run from the checkout root with an existing development environment. Loading
this skill installs nothing. For the commands below, use Python 3.12+ with NumPy,
pytest/pytest-xdist, `gguf>=0.18`, OpenAI SDK, requests and Jinja2 as test-only
dependencies. `pyproject.toml` lists the normal test extras; do not install the
whole stack or a model merely to load the skill. Python 3.11 can load it, but the
Python renderer excludes half from its supported dtype set before Python 3.12.

## Portable, bounded smokes

These selections use no model/tokenizer download or GPU. Run cheap format/link
checks and `python -m ruff check .`, then `python -m mypy tinygrad/`, before tests.
Use the repository's `-n12` convention; these are correctness runs, not timings.

Seeded tiny-model KV equivalence, deterministic MoE/top-k and recurrent chunk/reset
checks (these exact cases ran without CPU compilation in the review environment):

```sh
DEV=PYTHON python -m pytest -x -q -n12 \
  test/unit/test_llm_server.py::TestTransformerGenerate::test_chunked_prefill \
  test/unit/test_llm_server.py::TestTransformerGenerate::test_chunked_prefill_kv_cache_matches_single_chunk \
  test/unit/test_attention.py::TestGatedDeltaNetBlock::test_varied_chunk_sizes_match_decode \
  test/unit/test_attention.py::TestGatedDeltaNetBlock::test_start_zero_resets_realized_state \
  test/unit/test_attention.py::TestGatedDeltaNetBlock::test_kda_prefill_matches_decode \
  test/unit/test_attention.py::TestPairwiseTopk \
  test/unit/test_llm_moe.py
```

Small GGUF blocks, differential grid checks, synthetic split files and an error:

```sh
DEV=PYTHON python -m pytest -x -q -n12 \
  test/unit/test_gguf.py::TestGGUFTables \
  test/unit/test_gguf.py::TestGGUF::test_dequantization_q8_0_hardcoded \
  test/unit/test_gguf.py::TestGGUF::test_dequantization_q3_k_hardcoded \
  test/unit/test_gguf.py::TestGGUF::test_dequantization_mxfp4_block \
  test/unit/test_gguf.py::TestGGUF::test_multi_part_load \
  test/unit/test_gguf.py::TestGGUF::test_expected_failure_unknown_type
```

Synthetic tokenizer cases and mocked loopback HTTP/SSE/tool calls:

```sh
DEV=PYTHON python -m pytest -x -q -n12 \
  test/null/test_llm_tokenizer.py::TestLLMTokenizer::test_tekken_from_gguf_kv \
  test/null/test_llm_tokenizer.py::TestLLMTokenizer::test_tekken_gpt4o_split \
  test/null/test_llm_tokenizer.py::TestLLMTokenizer::test_stream_decoder \
  test/null/test_llm_server.py
```

The HTTP tests start and close their own ephemeral loopback servers. No real API
credential or CLI server is needed. `DEV=PYTHON python -m tinygrad.llm.cli --help`
is also safe: argument parsing exits before model loading. Running the CLI
without `--help` defaults to a downloadable model and is **not** a smoke.

Inspect skips as well as passes. If any selection needs an unavailable dtype,
compiler or package, report it rather than replacing it with NULL. Even with
`DEV=PYTHON`, some `.numpy()` paths or explicitly CPU-placed tensors still compile
on CPU. In particular `test_gatedeltanet_reference_and_reset` uses linspace data
that needs a working CPU compiler in the reviewed sandbox; the smaller selected
chunk/reset cases above executed without it.

## Broaden only for the change

With a working CPU toolchain, rerun the relevant selections with `DEV=CPU` and
include `TestGatedDeltaNetBlock.test_gatedeltanet_reference_and_reset` for the
NumPy recurrence oracle, neighboring `TestAttention` cases, and the full
`test/unit/test_llm_server.py` for JIT/cache bookkeeping and sampled-token tests.
A Python-backend result is numerical evidence, not CPU/GPU performance evidence.
For new MLA work strengthen the oracle described in `execution.md` first; do not
use the existing zero-projection equality as proof. Primitive GQA PyTorch tests
in `test/backend/test_ops.py` are a separate, larger dependency/compute tier.

Avoid whole-file `test_gguf.py` (downloads and large GEMVs), tokenizer tests that
use `llama_tok` (vocabulary fetch), CLI model aliases, and external model-evaluation
scripts by default. Use explicit test node IDs, not a broad name filter that can
accidentally select new download tests. Respect `conftest.py`'s timeout and bound
new examples/allocations; run benchmarks separately from xdist and instrumentation.

## Hardware-only checks: not substitutes for the smokes

`amd_custom_kernels_supported` requires AMD gfx11/RDNA3 **and HIPRenderer**.
RDNA4/CDNA, generic CPU/Python and NULL do not validate those kernels. Never
change drivers, kernel modules or PCI bindings to make this predicate true.

If a task explicitly authorizes that hardware, inspect these existing tests:

- `test/unit/test_llm_amd.py::TestQ8Quantize`: custom quantized linear/reference,
  `test_quant_linear_preserves_rope_permutation`, symbolic Q6, and
  `test_gated_delta_state_and_precision` (state contiguity, dtype and reset).
- The same `TestQ8Quantize` class also contains flash-attention tests:
  `test_flash_attention_decode_symbolic_gqa` (asserts custom path), physical/valid
  cache lengths, nonfinite masked tails, unaligned prefill starts and long/ragged
  shapes. Confirm current node names before invocation.
- `test/unit/test_attention.py::TestGatedDeltaNetBlock::test_gated_delta_rectangular_state_and_row_decay`:
  fused scan outputs/state against NumPy, with rectangular `(V,K)` state.

Check that a supposedly custom test really takes the custom path: shape guards
may choose generic fallback. For example, the reviewed decode GQA-tail test uses
head dimension 192, which fails the custom power-of-two dimension guard. Fallback
numerics do not certify custom tail stores. Padded recurrent beta/decay must be
no-ops; quant-linear and flash paths have additional shape/layout constraints.

Report these RDNA3 tests, multi-device paths, large GGUF/model integration and
throughput benchmarks as **not run** unless actually executed on matching
infrastructure. With an explicitly authorized already-local model, cap context,
prompt/output counts and warmup; retain a correctness oracle before any speed
claim. Reuse `tinygrad-performance-triage` rather than inventing a benchmark here.

## Discovery and maintenance checks

`opencode debug skill` inside this worktree should include
`tinygrad-llm-engine-development` at its repository-local path with its complete
body, alongside the three earlier skills. No permission/plugin change is needed.
`AGENTS.md` is the fallback route for other agents; a frozen startup index may
need a fresh run. Validate frontmatter with the pinned `skills-ref` utility in
[the repository skills README](../../README.md), using a disposable environment.
Check all local Markdown links and confirm only static regular files are added.
Discovery is not an activation-quality or productivity benchmark.

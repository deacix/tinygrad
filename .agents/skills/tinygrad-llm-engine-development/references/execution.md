# Execution and architectural invariants

Observed in `tinygrad/llm/model.py` at the revision in `sources.md`. Recheck the
current implementation before applying these contracts to a new architecture.

## Prefill, decode and fresh execution

`Transformer` owns separate `prefill_jit` and `rollout_jit` instances. Dispatch
uses `resolve(tokens.shape[1] != 1)`, not the number of newly generated tokens.
A symbolic prefill chunk with a bound length of one can still use the prefill
path: `TestTransformerGenerate.test_chunked_prefill` expects 9 prompt tokens with
chunk size 4 to execute three prefill chunks (4+4+1), then rollout. `forward`
processes all input tokens but projects/samples only the final token.

`generate` defaults to chunks of 32, consumes the prompt before yielding, appends
to the caller's list, and records `_cached_tokens = tokens[:-1]`. The last yielded
prediction is **not yet in the model state**. It stops at context capacity;
EOS/EOT and output-token limits belong to CLI/server callers. Keep prompt length,
processed length and returned-token count distinct.

For recurrent models without the supported custom AMD path, `generate` forces
chunk size 1. CPU block-level multi-token recurrence tests do not prove that CPU
generation uses chunked recurrent prefill. `warmup` can exercise both JITs for
attention-only/default-chunk execution, but recurrent CPU/Python generation with
chunk size 1 exercises rollout only. Schedule cache reuse, JIT capture/replay,
prompt reuse and `from_gguf(realize=...)` are four different experiments. Changing chunk-size bounds after capture deserves a
separate regression, not an assumption that every JIT accepts the new shape.

## Prefix reuse and divergence

- Attention-only `get_start_pos` matches `tokens[:-1]` against `_cached_tokens`,
  then takes the minimum reusable length reported by the blocks. It leaves a
  token to recompute even for a fully matching prompt. Only valid cache positions
  may be read; a full allocation need not be cleared on restart.
- With **any** recurrent block, reuse is allowed only when the new prompt strictly
  extends the **entire** cached sequence. Divergence, a shorter prompt or an exact
  match to the cached sequence returns zero. A recurrent state is the result of
  a scan, not a position-indexed KV table that can be truncated at a branch point.
- `GatedDeltaNetBlock._attention` binds `start_pos` as a runtime variable. At zero,
  both `conv_state` and `recurrent_state` must reset, including on JIT replay.
  Clearing token bookkeeping alone is not proof of that reset.
- Symbolic recurrent chunks are padded. Padded beta=0 and log-alpha=0 (decay=1)
  must leave state unchanged; output is trimmed back to valid length. Preserve
  write/read ordering and state layout `(B, H, V, K)`.

Recommended regression matrix: single prompt versus partitions with a one-token
tail; prefill then several decode steps; continuation; divergence inside the
old prefix; exact/shorter cached prompt; zero restart after realized state; final
context slot. Use fresh same-weight objects, independent input lists and finite
nonzero weights. Check valid KV/latent slices, convolution and recurrent state,
and numerical output before sampling. Invalid/empty prompts and invalid chunk
sizes need an explicit intended error contract; the current generator does not
validate all of them. Do not fabricate passing coverage for these proposals.

## Attention and expert invariants

**RoPE:** `apply_rope` uses half-split feature pairs and requires even width.
Ordinary attention rotates the first `rope_dim` features; MLA uses a rotary
suffix after its non-positional features. Preserve absolute position offsets,
theta, even rotary width and untouched non-rotary coordinates. `from_gguf`
permutes interleaved Llama/MLA projection rows into half-split form; Kimi-linear
skips those permutations. Do not apply a second permutation or assume a paper's
matrix layout matches checkpoint storage. Use an analytic/NumPy rotation oracle,
not `apply_rope` on both sides.

**GQA and masking:** query heads must divide into groups sharing KV heads
(`n_heads % n_kv_heads == 0`); preserve KV repeat-interleave ordering. Ordinary
blocks require `v_head_dim == head_dim`. With cached prefix length P and chunk
length T, query row i may attend to keys through P+i. The mask is lower-right
causal, shape `(T, P+T)`, offset `P+1`; generic upper-left `is_causal=True` is not
interchangeable. Check nonzero starts, unequal query/KV head counts, partial
RoPE, Q/K normalization placement and optional attention output gating.

**MLA:** cache normalized compressed KV plus rotary key with shape
`(B, 1, max_context, kv_lora_rank + rope_dim)`. The non-positional key projection
is absorbed into Q; compressed values are expanded through `attn_v_b` after
attention. Scaling remains `1/sqrt(head_dim)`, **not** the compressed dot-product
width. Cover direct-Q and Q-LoRA paths and `v_head_dim != head_dim`. KDA-associated
MLA skips rotary application in this implementation. Compare the actual
`_attention` against independently expanded K/V with seeded nonzero projections.

**MoE:** preserve expert axis `(experts, out, in)`, top-k identities/ties and
mixture weights. `pairwise_topk` favors lower IDs for ties and returns selected
ranks ascending. Bias changes selection, not the gathered unbiased probabilities.
SOFTMAX, SIGMOID, SOFTMAX_WEIGHT and SQRT_SOFTPLUS are distinct; normalized softmax
can become selected-logit softmax only under the implemented no-bias condition.
Routed scaling applies to routed experts; shared experts are added separately,
optionally sigmoid-gated. Check batching, k bounds, normalization, selection bias,
nonunit scale and leading dense blocks; a shape-only check misses routing errors.

## Existing evidence versus gaps

- `test/unit/test_llm_server.py::TestTransformerGenerate`: mocked prefix/dispatch
  tests plus real warmup, schedule reuse and generation tests. The recurrent
  bookkeeping tests toggle a flag on a non-recurrent model; they do not run a
  recurrent block. `test_kv_cache_resume_matches_fresh` clears bookkeeping on the
  same object and compares token IDs, not independent fresh-state logits.
- `test_chunked_prefill_kv_cache_matches_single_chunk` uses seed 1234, dim 8,
  context 16, chunks 4/8 and valid KV comparisons. Its one block's K/V precede
  attention, so it is not by itself a causal-mask/output oracle.
- `test/unit/test_attention.py::TestGatedDeltaNetBlock` has deterministic NumPy
  recurrence, KDA/channel decay, chunk-partition and realized-state restart tests.
  `test_start_zero_resets_realized_state` tests automatic reset through direct
  `_attention` output comparison, not state-buffer assertions or TinyJit replay;
  some reference helpers explicitly zero state. Add captured/replayed reset and
  state assertions when changing that contract. Tiny dimensions use the generic scan.
- `TestAttention` tests partial RoPE keys and position sensitivity, not an
  independent full rotation oracle. `TestPairwiseTopk` covers ties and seeded
  NumPy comparisons. Primitive PyTorch GQA tests live in
  `test/backend/test_ops.py::TestOps`, not cached engine integration.
- `test/unit/test_llm_moe.py::TestMoEFeedForward` has constructed nonzero weights
  and numerical gating/normalization/shared-expert checks. Bias and scaling need
  additional dedicated cases when touched.
- `test/unit/test_llm_mla.py::TestMLA.test_mla_attention_matches_naive` reconstructs
  two expressions, does not call production `_attention`, and leaves K/V expansion
  weights zero. Treat its pass as weak evidence, not a validated MLA oracle.

For compiler failures route to the rewrite skill; for speed route to performance
triage only after this correctness boundary holds. Hardware path selection and
its tests are listed in `validation.md`.

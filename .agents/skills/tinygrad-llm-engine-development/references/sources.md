# Evidence, source selection and maintenance

Reviewed 2026-09-19 against
[tinygrad 644c5863112f1c9049aca097bc30c7415ae72580](https://github.com/deacix/tinygrad/tree/644c5863112f1c9049aca097bc30c7415ae72580).
This skill and its five references are original MIT repository guidance. **No
third-party skill text, scripts or assets were imported.** Primary sources are
cited for facts, not installed as instructions. No upstream tinygrad LLM-engine
skill was verified in this bounded research; that is not a claim none exists.

## Repository evidence

The reviewed tree above pins these paths; function/test names are more useful
maintenance anchors than shifting line numbers:

- `tinygrad/llm/model.py`: `Transformer` generation/loading, attention/MLA,
  GatedDeltaNet/KDA and expert routing.
- `tinygrad/llm/gguf.py`: type inventories and container/split parsing.
- `tinygrad/llm/cli.py`: tokenizer, fallback and GGUF Jinja template setup.
- `tinygrad/llm/serve.py` and `tinygrad/viz/serve.py`: protocol and inherited SSE.
- `tinygrad/llm/kernels/amd.py`: capability predicate, custom quant-linear,
  flash-attention and recurrent scan constraints.
- `test/unit/test_llm_server.py`, `test_attention.py`, `test_llm_mla.py`,
  `test_llm_moe.py`, `test_llm_amd.py`, `test_gguf.py`: inspect assertions and
  helpers; orchestration, numerical, structural and hardware tests differ.
- `test/null/test_llm_server.py`, `test_llm_tokenizer.py`: mock loopback HTTP and
  tokenizer tests, including some download-backed vocabulary cases.
- `test/backend/test_ops.py`: primitive PyTorch GQA comparison, not engine cache
  proof; `pyproject.toml` and `conftest.py`: dependencies and test timeout.

The references distinguish implementation, existing tests and proposed
regressions. A source comment or test name alone is not proof. MLA's zero-
projection comparison, one-block prefill K/V checks and mocked recurrent flags
do not establish complete engine equivalence.

## Primary model and format sources

1. [RoFormer v5, section 3](https://arxiv.org/html/2104.09864v5#S3): rotary
   construction and relative-position mathematics. Supports rotation oracles,
   not long-context quality claims or checkpoint row-layout assumptions.
2. [GQA v3, section 2.2](https://arxiv.org/html/2305.13245v3#S2.SS2): query-head
   groups sharing KV heads; MHA/MQA endpoints. Reported uptraining/speed results
   are not measurements on this checkout.
3. [DeepSeek-V2 v4, section 2 and Appendix C](https://arxiv.org/html/2405.04434v4#S2):
   low-rank KV, decoupled RoPE, absorption and shared/routed experts. Do not apply
   its router formula to all local `ExpertGating` variants.
4. [Qwen3-Next reference, Transformers v4.57.1](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/qwen3_next/modeling_qwen3_next.py):
   inspected dynamic cache, GatedDeltaNet forward and recurrent/chunk rule
   functions. Its state is `(B,H,K,V)`, unlike local `(B,H,V,K)`; its cached branch
   is single-token. This is an architectural reference, **not** proof of local
   `qwen3next` GGUF support: explicit local hybrid branches are `qwen35`,
   `qwen35moe` and `kimi-linear`.
5. [GGUF specification](https://github.com/ggml-org/ggml/blob/456172ec733a135778adcd32d00e576a58232e45/docs/gguf.md):
   structure, typed metadata, alignment, offsets and chat-template metadata.
   Container support does not guarantee support for every quant/model.
6. llama.cpp at `60b06ab9a9eeec26f8125c9316ccbf4ee4713d1f`:
   [ggml-common.h](https://github.com/ggml-org/llama.cpp/blob/60b06ab9a9eeec26f8125c9316ccbf4ee4713d1f/ggml/src/ggml-common.h)
   and [ggml-quants.c](https://github.com/ggml-org/llama.cpp/blob/60b06ab9a9eeec26f8125c9316ccbf4ee4713d1f/ggml/src/ggml-quants.c).
   Selected Q4/Q8/K-quant layouts and reference dequantizers were inspected,
   not an exhaustive audit. Local tests use external gguf-py as another oracle;
   record its exact version for golden fixtures (smoke: 0.18.0).
7. [Hugging Face chat templating](https://huggingface.co/docs/transformers/chat_templating):
   model-specific control tokens, generation prefixes and duplicate special-token
   avoidance. Moving documentation reviewed on the date above.
8. [OpenAI Chat Completions reference](https://developers.openai.com/api/reference/typescript/resources/chat/subresources/completions/methods/create)
   and [function-calling streaming](https://developers.openai.com/api/docs/guides/function-calling#streaming):
   `choices[].delta`, finish reasons, usage-only chunks and tool-call index/JSON
   argument accumulation. Moving documentation reviewed on the date above;
   distinguish Chat Completions from Responses named events. The archived
   [streaming cookbook](https://developers.openai.com/cookbook/examples/how_to_stream_completions)
   is illustrative, not the current contract. The legacy streaming-reference URL
   was unavailable; accessible official reference/guide pages supplied evidence.

## Compatible skill research: inspected, not installed

[Agent Skills specification](https://agentskills.io/specification) and
[OpenCode native skills documentation](https://opencode.ai/docs/skills/) establish
`SKILL.md` metadata, `.agents/skills` discovery and progressive disclosure. They
do not vet third-party instruction content or prove routing quality.

Inspected actual candidate:
[vllm-project/vllm-skills / vllm-prefix-cache-bench](https://github.com/vllm-project/vllm-skills/blob/c99623410c1531148ff9b39fe1dcb27efbf4bf23/plugins/vllm-skills/skills/vllm-prefix-cache-bench/SKILL.md),
revision `c99623410c1531148ff9b39fe1dcb27efbf4bf23`. Its directory contains only the
entry point. It is format-compatible and offers cached/uncached and repeated-
prefix workload ideas, but depends on a separate vLLM checkout and model/data
loading. It includes installation, serving and credential setup advice;
none was followed or copied.

Separately inspected its referenced
[benchmark_prefix_caching.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/benchmarks/benchmark_prefix_caching.py)
at `4cc15f2121b3421478f3186aac38de901b4a44ef` (the skill does not pin it). It loads
vLLM, enables remote tokenizer code, samples/repeats prompts and times one
`generate` call; it does not itself validate outputs or calculate cache-hit rate.
Do not carry its arbitrary shared-prefix assumptions into live recurrent state.
Online bench implementation, transitive helpers, other skills and license text
were not fully reviewed because unchanged adoption was rejected. No import or
endorsement is claimed, and no script was executed.

The existing `property-based-testing` import retains its own provenance and
CC BY-SA terms. This skill links to it and its caveats without copying that
content. Rewrite debugging and performance triage remain separate; no broad
bundle, plugin, MCP service or lifecycle skill is added.

## Reproduce, update or remove

Clone the committed skill and references; update the `AGENTS.md` route and skills
README together. Re-run native discovery and safe validation. The optional
[skills-ref validator](https://github.com/agentskills/agentskills/tree/69ef37e9424c0a7ea9dd2293b559e43ec8176379/skills-ref)
is demonstration tooling, not an instruction-safety validator; use a separate
reviewed disposable environment, not a repository dependency or global install.

When model/cache/format/protocol behavior changes, re-read the named source and
test helpers, refresh the evidence pin and smoke selections, and retain explicit
coverage limits. Pin and review any future imported subset with its license and
checksums before adoption; do not silently update from a marketplace head.
Removal deletes this directory and its routing/README entries only.
No runtime configuration, hooks or services need undoing.

Expected benefit is less repeated exploration and better-targeted oracles,
**not a measured agent-efficiency improvement**. Manual scenario review,
format/native discovery and smokes do not prove future agents will select this
skill correctly or deliver better patches.

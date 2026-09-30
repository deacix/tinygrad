# Conflict map for upstream tinygrad merges

Resolution rule per area when `git merge tinygrad/master` conflicts. Two
general rules frame the table: upstream's semantics win in shared core files
(any fork delta re-applies on top as its own edit), and fork-owned files never
take upstream content wholesale. Verified against the tree at the skill's
addition; re-verify paths after a large upstream restructure.

## Shared core (upstream-first, fork delta on top)

| Path | What lives there | Typical conflict | Resolution |
| --- | --- | --- | --- |
| `tinygrad/uop/ops.py`, `tinygrad/uop/symbolic.py` | Ops enum, pattern matcher, symbolic simplification | rule edits and enum additions on both sides | take upstream's rewrite set, re-apply any fork rule; run `test/null/test_graph_rewrite.py` and `test/null/test_uop_symbolic.py` |
| `tinygrad/codegen/` | lowering passes, opt search | new/renamed passes | upstream-first; re-map fork references (`tinygrad-rewrite-debugging` names the pass locations) |
| `tinygrad/schedule/` | fusion, kernel grouping | scheduler refactors | upstream-first; `test/null/test_schedule.py` + `test/helpers.py::check_schedule` decide |
| `tinygrad/helpers.py`, `tinygrad/dtype.py` | ContextVar, DType machinery | small textual conflicts, high blast radius | upstream-first, then run the dtype suites (`test/null/test_dtype.py`, `test/backend/test_dtype_alu.py`) |
| `tinygrad/engine/` (JIT, realize) | `engine/jit.py` | capture-phase behavior | upstream-first; `test/test_tiny.py::TestTiny::test_jit` is the smoke |
| `tinygrad/renderer/`, `tinygrad/runtime/ops_*.py` | backends | kernel text changes | upstream-first; process replay judges kernel diffs |
| `tinygrad/runtime/autogen/**`, `runtime/support/autogen.py` | generated bindings | upstream regenerated versions | prefer a local regeneration over a hand merge; the in-tree Autogen CI diff is authoritative |
| `tinygrad/llm/model.py`, `serve.py`, `cli.py`, `gguf.py`, `kernels/amd.py` | upstream's LLM engine, fork-extended | both sides change these files (UU in real syncs) | upstream-first + fork delta re-applied, exactly like any shared core file; only `multimodal.py`/`vision.py`/`placement.py` are fork-only |
| `pyproject.toml` | extras (`testing*`, `linting`, `docs`) | dependency pin drift | take upstream's pins unless the fork's tests need a fork pin; say so in the PR |
| `.github/workflows/test.yml` | CI matrix | both sides edited (the fork adds mockgpu/LLM jobs) | keep the fork's added jobs atop upstream's restructure |
| `test/` layout (`backend/`, `null/`, `unit/`) | test groups | files moving between groups; upstream renamed `test/unit/` → `test/runtime/` | follow upstream's new home (the rename relocates the fork's LLM tests under `test/unit/`); update skill references that name test paths |

## Fork-only (never replaced by upstream)

| Path | Why it is fork-only |
| --- | --- |
| `tinygrad/llm/multimodal.py`, `vision.py`, `placement.py` | fork-only modules (Qwen image inference, contiguous layer placement) — upstream has no equivalent |
| `.agents/skills/**`, the `# Repository skills` block of `AGENTS.md` | the committed agent skills; upstream merges must not delete their routes. (`AGENTS.md` itself is upstream's file — resolve shared, keep the block.) |
| fork-added LLM tests under `test/unit/` (e.g. `test_llm_placement*.py`, `test_gguf_placement.py`) | follow them into upstream's new `test/runtime/` home on rename conflicts; never drop them |

Files that look fork-only but are **not**: `opencode.json`,
`extra/gptoss_kernels`, the rest of `AGENTS.md`, and the shared LLM engine
files above all exist upstream — resolve them like shared core.

## After the resolution

1. Regenerate what the merge invalidated (autogen first).
2. Run the post-merge ladder in the sync skill (ruff, mypy, targeted
   `DEV=NULL` / `DEV=PYTHON` pytest with `-n12`).
3. Process-replay both sides when kernels can change; a routine sync does not
   need the `[pr]` assert — a merge that is also a refactor does.
4. `python sz.py` for core-size drift when `tinygrad/` files changed.
5. Re-check every path and command the agent skills name; update
   `.agents/skills/README.md`'s review pin.
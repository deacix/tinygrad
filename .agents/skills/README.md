# Repository-local agent skills

Reviewed on 2026-09-19 against tinygrad `289656f`. These are coding-agent
instructions, not tinygrad runtime features. Cloning this repository installs
the committed files; no plugin manager, service, credentials or global install
is needed. `AGENTS.md` is the small routing index. Load only the relevant skill.

## Selection

| Skill | Source | Why it belongs here |
| --- | --- | --- |
| [property-based-testing](property-based-testing/SKILL.md) | Trail of Bits, pinned below | Existing Hypothesis tests cover dtype and graph edge cases. Oracle selection, shrinking and test review complement example tests. Read [local caveats](property-based-testing/NOTICE.md) first. |
| [tinygrad-rewrite-debugging](tinygrad-rewrite-debugging/SKILL.md) | First-party, MIT | Connects UOps, named passes, SPEC, structural tests and numerical regressions. Includes an isolated VIZ capture and NULL-backend limitations. |
| [tinygrad-performance-triage](tinygrad-performance-triage/SKILL.md) | First-party, MIT | Separates compilation, dispatch, scheduler and kernel costs; bounds workloads and avoids false speedups from JIT warmup, async execution or instrumentation. |
| [tinygrad-llm-engine-development](tinygrad-llm-engine-development/SKILL.md) | First-party, MIT | Engine-specific cache/state, attention/MoE, GGUF and protocol invariants; five on-demand references and bounded synthetic checks. Reuses the three skills above. |

The first-party skills are original repository-specific workflows, not
renamed marketplace installs. No suitable off-the-shelf tinygrad-specific skill
was verified in this research. The LLM skill was reviewed separately against
`644c5863112f1c9049aca097bc30c7415ae72580`; its
[provenance and evidence](tinygrad-llm-engine-development/references/sources.md)
records primary model/format/protocol sources and the inspected, rejected vLLM
prefix-cache skill. No third-party material was imported for that skill. Its
[safe validation guide](tinygrad-llm-engine-development/references/validation.md)
separates small numerical/model-state tests, synthetic GGUF, loopback HTTP and
hardware-only checks. It adds no application behavior or dependencies.

The rewrite and performance skills' source of truth is this
[reviewed tinygrad tree](https://github.com/deacix/tinygrad/tree/289656f6e88e94ecce8dfb1dcf5378fccf577d0d),
especially [VIZ](https://github.com/deacix/tinygrad/blob/289656f6e88e94ecce8dfb1dcf5378fccf577d0d/tinygrad/viz/README.md)
and [speed categories](https://github.com/deacix/tinygrad/blob/289656f6e88e94ecce8dfb1dcf5378fccf577d0d/docs/developer/speed.md).
They use the project's existing mechanisms:

- `docs/developer/developer.md`, `docs/developer/layout.md`: pipeline and code map.
- `tinygrad/uop/ops.py`, `tinygrad/uop/symbolic.py`, `tinygrad/codegen/__init__.py`:
  pattern matching, symbolic rewrites, named lowering passes.
- `test/null/test_graph_rewrite.py`, `test/null/test_dtype_spec.py`: Hypothesis;
  `test/null/test_schedule.py` and `test/helpers.py`: structural/kernel-count tests.
- `test/backend/test_ops.py`: NumPy/PyTorch value and gradient comparisons.
- `tinygrad/runtime/ops_null.py`: no computation, synthetic timing, copyout restrictions.
- `docs/developer/speed.md`, `tinygrad/viz/README.md`, `extra/gemm/simple_matmul.py`:
  performance categories, traces, bounded numerical GEMM.
- `.github/workflows/test.yml`, `.github/workflows/benchmark.yml` and
  `test/external/process_replay/README.md`: backend matrix, hardware benchmarks,
  kernel-diff replay. Self-hosted device setup is not portable installation guidance.
- `pyproject.toml`, `.pre-commit-config.yaml`, `AGENTS.md`: existing Python checks.
  `opencode.json` disables formatter/LSP; this change leaves it untouched.

These should reduce repeated repository exploration and prevent weak test or
benchmark evidence. That benefit is reasoned from the workflows, **not a measured
agent-efficiency improvement**. Format validation and smoke execution do not
prove model routing quality or better future patches.

## Research and exclusions

Primary sources consulted (moving pages reviewed on the date above):

1. [Agent Skills specification](https://agentskills.io/specification): portable
   `SKILL.md`, frontmatter, local references and progressive disclosure.
2. [OpenCode skill documentation](https://opencode.ai/docs/skills/): discovers
   `.agents/skills/<name>/SKILL.md` and exposes `skill({ name: "..." })`.
3. [Trail of Bits source](https://github.com/trailofbits/skills/tree/123037ec8aed26f0d86327cc39137ee5043e5deb/plugins/property-based-testing/skills/property-based-testing):
   reviewed the skill and all five references before installation. The source's
   own eval README distinguishes activation from useful outcomes; its results
   are not measurements on this repo or on our agent.
4. [obra/superpowers systematic-debugging](https://github.com/obra/superpowers/blob/main/skills/systematic-debugging/SKILL.md):
   root-cause-first advice is useful, but the inspected example can print an
   identity/secret value and it requires sibling Superpowers workflows. Not
   installed; no sanitization of a blocked upstream package was attempted.
5. [wshobson Python performance skill](https://github.com/wshobson/agents/blob/main/plugins/python-development/skills/python-performance-optimization/SKILL.md):
   inspected its entry point. Generic Python profiling does not cover tinygrad's
   async devices, NULL timings, VIZ and JIT phases; optional profiler/native-
   extension/NumPy advice is broader than this core's needs. Not selected;
   bundled details were not fully audited because the entry point did not fit.
6. [Agent Skills reference validator](https://github.com/agentskills/agentskills/tree/69ef37e9424c0a7ea9dd2293b559e43ec8176379/skills-ref):
   validation and prompt-generation utility, explicitly demonstration software,
   not a skill or production/security validator.

No broad skill bundle, editor extension, Claude plugin, MCP server, GPU tooling
or generic git/PR/plan/review skill was installed. This Legwork run already
supplies the latter workflows as an ignored platform overlay; they are not
tracked repository assets and are not dependencies of the new skills.

## Provenance and maintenance

Trail of Bits revision: `123037ec8aed26f0d86327cc39137ee5043e5deb`.
Copied `SKILL.md`, all five `references/*.md` and root LICENSE as `LICENSE.txt`.
The only vendor-text change removes unsupported top-level `effort: low`.
`NOTICE.md` carries attribution, CC BY-SA 4.0 terms and tinygrad-specific caveats;
`SHA256SUMS` locks the installed vendor subset, including the patched entry point.
This third-party directory is **not covered by tinygrad's MIT license**.

For a fresh setup, use the committed files, not a moving marketplace installer.
To reproduce/update the import: clone the source into a separate review directory,
check out the exact revision, inspect and scan the skill and references, copy only
the subset above, apply the documented one-line portability change, and check the
hashes. On an intentional version update, review the diff, attribution and hashes
together. Do not overwrite NOTICE or silently refresh from `main`.

```sh
(cd .agents/skills/property-based-testing && sha256sum -c SHA256SUMS)
```

All retained instruction files were reviewed for purpose, secret access,
permissions, external execution and incompatible behavior. They contain no
executables, hooks, downloads or permission grants. The optional smart-contract
reference link is not required for tinygrad. No vendor rule was suppressed to
pass scanning. First-party examples can write temporary traces/caches and execute
local tests; they explicitly avoid GPU driver changes and untrusted pickles.

## Discovery and validation

OpenCode 1.18.31 was tested with the existing `opencode.json`, without adding
permissions or plugins:

```sh
opencode debug skill
```

Expect the four names above, with locations under this checkout and non-empty
instruction content. Run inside the worktree. Its `skill` tool loads these names
on demand; no model/provider is required for `debug skill`. If missing, check
capitalization, directory/name agreement and user/global permission overrides.
Do not broadly allow tools to fix discovery. `AGENTS.md` also links the files
for agents that read instructions but lack automatic skill discovery. Legwork's
startup index is frozen per run; review/restart is needed to see a changed index.

Optional independent format validation uses the pinned reference library above
in a disposable Python 3.11+ venv, **not tinygrad's dependencies**. After reviewing
that source, install its `skills-ref` directory into that venv and run:

```sh
for name in property-based-testing tinygrad-rewrite-debugging tinygrad-performance-triage tinygrad-llm-engine-development; do
  skills-ref validate ".agents/skills/$name" || exit 1
done
skills-ref to-prompt .agents/skills/property-based-testing \
  .agents/skills/tinygrad-rewrite-debugging .agents/skills/tinygrad-performance-triage \
  .agents/skills/tinygrad-llm-engine-development
```

The reference library needs Click and StrictYAML. This validates metadata and
emits names/descriptions/locations; it neither audits instruction safety nor
executes skill bodies. Use normal, trusted paths. No persistent custom validator
or installer is introduced.

The verification run used a disposable OpenCode npm install outside the checkout
with lifecycle scripts disabled, and executed its platform binary directly.
This was a discovery test only, not an additional skill or required repo setup.

## Smoke tests and limits

Use the existing development environment, Python 3.11+ and the test dependencies
from `pyproject.toml`. The following narrow tests need pytest, pytest-xdist,
NumPy and Hypothesis but not PyTorch:

```sh
DEV=NULL SPEC=2 python -m pytest test/null/test_dtype.py test/null/test_graph_rewrite.py -x -q -n12
DEV=PYTHON python -m pytest test/test_tiny.py::TestTiny::test_plus test/test_tiny.py::TestTiny::test_jit -x -q -n12
python -m ruff check .
python -m mypy tinygrad/
```

The rewrite skill contains the single-process VIZ capture smoke. The performance
skill contains a conditional CPU GEMM smoke. Respect test modules' imports:
`DEV=PYTHON` is not a blanket workaround for missing clang, as some paths still
compile on CPU. Do not replace numerical checks with NULL passes. Full backend,
GPU, SQTT, process-replay and throughput claims need their own infrastructure.

The LLM skill's validation guide adds explicit Python 3.12+ numerical selections
(half support), synthetic GGUF fixtures and mock loopback OpenAI-client tests.
Do not run the whole GGUF or tokenizer file as an offline smoke: both contain
fetch-capable tests. Hardware-only RDNA3 kernels and real-model checks remain a
separate opt-in tier; CPU/Python and skipped tests do not validate them.

Removal is simply deleting the selected skill directories and their AGENTS.md
links/README entries; there are no installed hooks, global config or services to undo.

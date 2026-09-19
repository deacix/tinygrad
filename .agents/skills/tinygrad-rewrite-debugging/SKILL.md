---
name: tinygrad-rewrite-debugging
description: Localize tinygrad UOp, PatternMatcher, symbolic simplification, scheduling or codegen regressions with minimal reproductions, SPEC checks and VIZ named-pass traces. Use for incorrect rewrites or kernel structure, not throughput tuning alone.
license: MIT
compatibility: Requires a tinygrad checkout and Python 3.11+. Tests need pytest and pytest-xdist; graph tests also need NumPy and Hypothesis. CPU execution requires a working compiler/runtime.
---

# Debug tinygrad rewrites

Read `AGENTS.md` and `tinygrad/viz/README.md` from the repository root first.
Commands below run from that root. This first-party workflow uses the existing
compiler and profiler; it installs no tooling, hooks or drivers.

## 1. Classify and reproduce

Record the revision, smallest failing input, shape, dtype, device/renderer,
expected result, actual result and exact command. Reduce the example before
collecting a whole model trace. Do not log credentials or dump the environment.

Follow the current pipeline rather than guessing from an older tinygrad version:

- Tensor semantics/gradients: `tinygrad/tensor.py`, `tinygrad/mixin/`,
  `test/backend/test_ops.py`, `test/unit/test_gradient.py`.
- UOp patterns and symbolic math: `tinygrad/uop/ops.py`,
  `tinygrad/uop/symbolic.py`, `test/null/test_graph_rewrite.py`,
  `test/null/test_pattern_matcher.py`, `test/null/test_uop_symbolic.py`.
- Scheduling/fusion: `tinygrad/schedule/`, `test/null/test_schedule.py`.
- Lowering/rendering: `tinygrad/codegen/__init__.py`, `tinygrad/renderer/`.

Use the smallest applicable test, then neighboring tests, with `-n12`:

```sh
DEV=NULL SPEC=2 python -m pytest test/null/test_dtype.py test/null/test_graph_rewrite.py -x -q -n12
```

`DEV=NULL` avoids GPU selection and checks structure, not numerical correctness.
Its programs do no computation and return synthetic timings. Some tests explicitly
choose other devices; inspect imports and device overrides before broadening a run.
For numerical changes use an executable backend and a value/gradient oracle.
`DEV=PYTHON` is useful for small cases but does not guarantee every test avoids
CPU compilation. When CPU is available, use `DEV=CPU` explicitly.

## 2. Find the first incorrect pass

Capture one reproduction in a single process, not an xdist run: parallel workers
can overwrite the shared trace files. Use a new private temporary directory for
each capture. VIZ serializes trusted local pickle files; never open a downloaded
or untrusted pickle. Redirecting stdout prevents interactive server startup.

A graph-only capture that needs no GPU or C compiler:

```sh
trace_dir=$(mktemp -d)
TMPDIR="$trace_dir" DEV=NULL VIZ=1 python -m unittest \
  test.null.test_graph_rewrite.TestSubstitute.test_simple > "$trace_dir/capture.log" 2>&1
TMPDIR="$trace_dir" NO_COLOR=1 python -m tinygrad.viz.cli --ls
TMPDIR="$trace_dir" NO_COLOR=1 DEBUG=7 python -m tinygrad.viz.cli --json > "$trace_dir/rewrites.jsonl"
```

For real failures substitute the reduced test/script. First list sources and
passes; use `-s TINY` and names from the actual capture to narrow the output.
Do not copy example schedule/kernel names from the README as stable identifiers.
DEBUG 3 shows ASTs, 4 generated source, 5 pass/kernel graphs, 6 UOp graphs,
7 every rewrite. Increase detail only around the suspected pass; JSON includes
raw `value` records, not just timing events. Save command, trace and relevant
before/after graph evidence with the diagnosis.

## 3. Test a hypothesis and protect semantics

Name the first invalid transformation, its rule location and violated invariant.
Change one hypothesis at a time. Add the smallest failing regression before a
fix: assert graph/spec invariants and, when values can change, execute the result
on a real backend. `test/helpers.py::check_schedule` also compiles and checks
kernel counts; a kernel count alone is not a value oracle.

For dtype/index edge cases, use the `property-based-testing` skill and its
`NOTICE.md` caveats. Check bounded integer arithmetic, overflow, empty shapes,
NaN/Inf, signed zero and precision losses as applicable; do not assume exact
real-number identities hold for floats. Keep expected kernel counts meaningful,
not merely updated to whatever the implementation now produces.

For compiler refactors/speedups, consult `test/external/process_replay/README.md`
(including its `[pr]` title convention). Replay requires a captured baseline and
may intentionally expose kernel diffs. Inspect helpers before use; do not blindly
run reset/checkout scripts, erase caches, or switch away from uncommitted work.

Finish with targeted regressions and the checks in `AGENTS.md`. Report numerical,
structural and hardware verification separately, naming anything not run.
Do not load GPU kernel modules or change PCI bindings to make a test pass.

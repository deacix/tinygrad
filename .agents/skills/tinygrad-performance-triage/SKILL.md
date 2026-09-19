---
name: tinygrad-performance-triage
description: Diagnose tinygrad compile, dispatch, scheduler and kernel performance using bounded workloads, VIZ, TinyJit and controlled BEAM comparisons. Use for slow compilation, excess kernels or measured runtime regressions; route incorrect values and rewrites to correctness debugging first.
license: MIT
compatibility: Requires Python 3.11+ and tinygrad. GEMM examples need NumPy and a working CPU compiler/runtime; GPU timing and AMD SQTT require matching authorized hardware. No services or credentials are required to load this skill.
---

# Triage tinygrad performance

Read `AGENTS.md`, `docs/developer/speed.md`, `docs/env_vars.md` and
`tinygrad/viz/README.md` from the checkout root. Their descriptions are context,
not measured claims for the current workload. Confirm current APIs in source.
This first-party skill adds no profiler dependency or runtime configuration.

## Establish the question before optimizing

Classify the bottleneck:

| Layer | Evidence to collect |
| --- | --- |
| Compile/Python | First-run wall time, rewrite counts, slow Python call paths |
| Dispatch/driver | Warm TinyJit replay, synchronization and launch overhead |
| Scheduler | Kernel count, materialized intermediates, memory traffic |
| Codegen/kernel | Device kernel duration, applied opts, source, utilization |

Record the revision, device/renderer, shape, dtype, input seed, compiler version,
BEAM setting, cache state, warmup/repetitions and timing boundary. Log only named
non-secret settings, never the entire environment. Use identical inputs and
conditions on baseline and candidate. Run each benchmark alone, not under
pytest-xdist or beside another benchmark.

A `realize()` is not a portable guarantee of completed asynchronous GPU work.
Synchronize the actual devices at measurement boundaries or use the backend's
kernel timing support. Separate cold compilation, TinyJit's initial execution
and capture calls, and steady-state replay. Read `tinygrad/engine/jit.py` and
`test/test_tiny.py::TestTiny.test_jit` before attributing a first-call cost to JIT.

## Start small and keep correctness attached

Run from the repository root with explicit `DEV`; auto-selection can touch GPUs.
For an existing working CPU backend and NumPy, this bounded smoke computes a GEMM
and compares with NumPy using the harness's float32 tolerances:

```sh
PYTHONPATH=. DEV=CPU N=32 M=32 K=32 CNT=2 BEAM=0 DEBUG=2 python extra/gemm/simple_matmul.py
```

Inspect `extra/gemm/simple_matmul.py` before changing dtype, dimensions or
repetitions. It defaults to 4096-sized matrices, supports extra dtype/tensor-core
overrides and uses random inputs; the small smoke is NOT a reproducible speedup
benchmark. For A/B measurements use a fixed-input harness and explicit timing,
not these two iterations. Preserve the oracle, don't relax tolerances to win.
If clang/runtime support is absent, report CPU timing unverified; a small
`DEV=PYTHON` execution can check the flow but is not a CPU or GPU timing result.
NULL's synthetic timings never establish performance or numerical correctness.

## Collect bounded evidence

Use `DEBUG=2` for runtime summaries. Use `VIZ=1` on one reduced process for
rewrites and timelines, redirecting stdout so no interactive server is launched.
Each capture writes local pickle files: isolate it with a fresh `TMPDIR`, use
the same directory for the CLI, and inspect only trusted captures. Do not run
captures concurrently in the same directory.

After that capture:

```sh
TMPDIR="$trace_dir" NO_COLOR=1 python -m tinygrad.viz.cli --ls
TMPDIR="$trace_dir" NO_COLOR=1 DEBUG=3 python -m tinygrad.viz.cli --json > "$trace_dir/events.jsonl"
TMPDIR="$trace_dir" NO_COLOR=1 python -m tinygrad.viz.cli -t 10
```

Here `trace_dir` is the directory used by the capture, not an arbitrary previous
run's output. The `tinygrad-rewrite-debugging` skill has a runnable graph-only
capture. Use `profile_marker`/`--interval` to isolate iterations in a larger
profile; list real markers before selecting them. DEBUG 4 reveals source; use
5-7 only to investigate graph passes. Unfiltered `-t` is an all-source diagnostic
summary, including nested TINY compiler ranges, not a device-kernel breakdown.
For device kernel counts, durations or percentages, use `-s` with the actual
device timeline listed by `--ls`. Even then, summed kernel times are not
overlapping end-to-end latency. For Python compile overhead, stdlib `cProfile`
can identify hot calls, but it cannot measure device throughput.

## Compare one controlled change

Start with BEAM=0. Only try a bounded BEAM search when kernel performance is the
hypothesis; it compiles alternatives and can greatly increase startup cost.
Keep search/compile cache conditions identical, use private caches if needed,
and document cold versus warm runs instead of deleting shared caches. Compare
applied opts and correctness as well as speed. For changed compiler kernels,
consult `test/external/process_replay/README.md` and its `[pr]` convention.

Gather multiple uninstrumented timing samples after diagnosis; report sample
count, median and spread with units, then baseline/candidate delta. Keep hardware,
inputs and warmup constant. A profiled run is diagnostic evidence, not the final
speed number. Reject apparent wins caused by skipped work or changed precision.

SQTT/PMC and VIZ=2 require AMD hardware and add overhead. GPU, tensor-core,
multi-device or PCI claims require matching hardware; emulation is not proof.
Do not copy self-hosted CI's device resets, module changes, process killing or
PCI unbinding into a development sandbox. No kernel modules are installed here.

Deliver: bottleneck hypothesis, exact commands/configuration, correctness check,
baseline/candidate measurements with units, relevant profile evidence and explicit
unverified hardware claims. Finish with applicable `AGENTS.md` quality checks.

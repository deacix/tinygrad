---
name: tinygrad-runtime-driver-development
description: Develop and review tinygrad runtime backends, generated bindings and user-space device drivers. Use for tinygrad/runtime/ops_*.py, runtime/support and runtime/autogen regeneration, HCQ/MMIO interfaces, mockgpu-simulated driver tests, hcqfuzz and process-replay kernel diffs. Climb the offline structural and mock tiers first; hardware-only checks require separately authorized matching hardware.
license: MIT
compatibility: Requires a tinygrad checkout and Python 3.11+. Offline tiers need pytest, pytest-xdist and NumPy; binding regeneration imports tinygrad.runtime.autogen modules (libclang bindings for the REGEN path). Hardware tiers need matching authorized devices. No services or credentials are required to load this skill.
---

# Develop tinygrad runtime backends and drivers

Read `AGENTS.md` and `docs/developer/runtime.md` from the repository root
first. Commands below run from that root. This first-party workflow uses the
existing drivers, mocks, fuzzers and replay tooling; it installs no tooling,
hooks or kernel modules. For incorrect values or kernel structure, start with
`tinygrad-rewrite-debugging`; for throughput, `tinygrad-performance-triage`.

## 1. Map the surface before changing it

| Surface | Where |
| --- | --- |
| Backends (device, compiler, allocator, program) | `tinygrad/runtime/ops_*.py` (14 today) |
| Shared runtime plumbing | `tinygrad/runtime/support/` (`hcq.py`, `hcq2.py`, `memory.py`, `elf.py`, `system.py`, `compiler_*`) |
| Generated C bindings | `tinygrad/runtime/autogen/` (41 modules) plus `support/c.py` ioctl macros |
| Graph capture backends | `tinygrad/runtime/graph/` |
| User-space device drivers | `extra/nv_gpu_driver`, `extra/qcom_gpu_driver`, `extra/hip_gpu_driver`, `extra/bnxt_driver`, `extra/usbgpu`, `extra/amdflash`, `extra/amdpci` |
| Simulated devices for offline tests | `test/mockgpu/` (NV, AMD, AM, USB mock drivers) |
| Driver/allocator fuzzers | `extra/hcqfuzz/`, `test/external/external_fuzz_*.py` |

Confirm the current split in source before relying on this table; the layout
doc `docs/developer/layout.md` tracks the compiler side.

## 2. Regenerate bindings, never hand-edit them

Read [references/autogen.md](references/autogen.md). Generated modules in
`tinygrad/runtime/autogen/` are produced by importing them; the in-tree Autogen
CI (`.github/workflows/autogen.yml`) deletes every regenerable module, re-imports
and fails on any `git diff`. Never patch a generated file by hand and never
commit a partial regeneration.

## 3. Climb the offline test ladder

Work as high on this ladder as the change allows; name the tier you actually
ran when reporting.

1. **Structural (no device):** `DEV=NULL` suites under `test/null/` —
   `test_hcq_iface.py` (MMIO/USBMMIO interface semantics against
   `test/mockgpu/usb.py` MockUSB), `test_elf.py`, `test_device.py`. Example:

   ```sh
   DEV=NULL python -m pytest test/null/test_hcq_iface.py test/null/test_elf.py -x -q -n12
   ```

2. **Simulated devices:** `test/mockgpu/` intercepts file descriptors and
   memoryviews with mock NV/AMD/AM/USB drivers selected through DEV interface
   names (`MOCK`, `MOCKKFD`, `MOCKPCI`, `MOCKUSB`; see `test/mockgpu/mockgpu.py`).
   CI runs these as `DEV=MOCKKFD+AMD` / `DEV=MOCKKFD+AMD:LLVM` against `test/amd/`
   (`.github/workflows/test.yml`). A green mock run proves driver-path logic,
   never hardware behavior.

3. **Hardware-only:** real-device suites (`test/device/`, `test/amd/` on real
   cards, SQTT/PMC, multi-device, throughput) need matching authorized hardware
   and are a separate opt-in tier. Report them as not run when the sandbox has
   no such device; NULL and mock runs never substitute.

## 4. Protect kernel structure with process replay

Compiler, autogen or driver changes that can alter generated kernels are
guarded by `test/external/process_replay/` (read its `README.md` first):
capture with `CAPTURE_PROCESS_REPLAY=1`, diff against master with
`process_replay.py`, and note the `[pr]` pull-request-title convention that
enables the assert for refactor/speedup PRs. Inspect `reset.py` and `local.sh`
before running them; never reset or overwrite a capture you still need.

## 5. Fuzz the interface you changed

`extra/hcqfuzz/` runs `TestSpec`-based cases from its `tests/` folder
(`PYTHONPATH=. RUN_FILES="hcq,allocator" python3 extra/hcqfuzz/fuzzer.py`;
`SKIP_FILES` excludes suites). `test/external/external_fuzz_*.py` cover HCQ
signals/multi-process, SDMA warm start, TLSF, AM PT, QCOM CPU cache and BEAM
timeout recovery. Fuzzers exercise long-tail states; they are evidence about
the paths they hit, not correctness oracles.

## 6. Hardware boundaries are hard rules

Read [references/hardware-boundaries.md](references/hardware-boundaries.md)
before any command that touches devices. In short: never insert kernel
modules, never rebind PCI devices, never copy self-hosted CI's device resets or
process killing into a development sandbox. `AGENTS.md` states the kernel-module
rule; violating it to make a test pass is never acceptable.

## 7. Finish scoped

Run the checks `AGENTS.md` names for the code you touched — targeted pytest
selections with `-n12`, `python -m mypy tinygrad/`, `python -m ruff check .` —
plus the process-replay diff when kernels can change. Report structural,
simulated and hardware verification separately, naming anything not run.
Do not widen a failing selection's device or skip a kernel-module check to get
green; report the boundary instead.
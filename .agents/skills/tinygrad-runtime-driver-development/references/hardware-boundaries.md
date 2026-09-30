# Hardware boundaries for runtime and driver work

These are hard rules for development sandboxes. They exist so a driver change
is proven with mock and structural evidence instead of by mutating the host's
device state.

## The prohibitions

1. **Never insert kernel modules.** tinygrad's AMD and NVIDIA paths are
   user-space PCI drivers (`AGENTS.md` states the rule directly). A test that
   "needs" a module loaded is asking for the hardware tier, not a sandbox hack.
2. **Never rebind PCI devices.** No unbinding vendor drivers, no rebinding to
   vfio/pci-stub, no changing PCI device ownership to make a path reachable.
3. **Never copy self-hosted CI's device resets into a sandbox.** Self-hosted
   benchmark and device jobs may reset GPUs, kill device processes or reload
   drivers as part of their runner image; that is runner-orchestration code,
   not portable development guidance. Do not run `extra/runbook_digitalocean_mi350x.sh`
   or similar runbooks against a development machine.
4. **Never swap a device claim for an emulation claim.** GPU, tensor-core,
   SQTT/PMC, multi-device and throughput results require matching authorized
   hardware. NULL's synthetic timings and mockgpu's simulated drivers prove
   neither values nor speed; `DEV=P`ython/CPU work is not a GPU result.

## What to do instead

- Prove driver-path logic offline: `DEV=NULL` structural tests and the
  `test/mockgpu/` simulated tiers (`DEV=MOCKKFD+AMD` and friends, as
  `.github/workflows/test.yml` does).
- Use process replay (`test/external/process_replay/`) to show generated
  kernels are unchanged when a change should not alter them.
- State hardware-only verification explicitly as **not run**, with the exact
  suites that would need authorized devices (for example `test/device/`, real
  `test/amd/`, SQTT captures).

## Reporting discipline

A report separates three tiers — structural, simulated, hardware — and names
what each one did and did not prove. "Tests passed" without the tier is
ambiguous and hides the boundary. When a fix genuinely needs hardware, stop and
say so; do not manufacture the environment inside the sandbox.
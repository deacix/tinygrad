# Generated bindings: regenerate, never hand-edit

`tinygrad/runtime/autogen/` holds the committed, generated C-binding modules
(`kfd.py`, `cuda.py`, `comgr.py`, `hsa.py`, `nv*.py`, `libc.py`, `io_uring.py`,
`libusb.py`, `mesa.py`, `bnxt.py`, `mlx5.py`, `llvm*.py`, `ggml_common.py`, the
`am/`, `nv_regs/` and `amd/` subpackages, and more — count them with the
in-tree Autogen job's own `find` expression, never from memory: 65 files
regenerate through it today). They share `# mypy: disable-error-code="empty-body"`
headers and build on the ioctl macros in `tinygrad/runtime/support/c.py`
(`_IO`, `_IOW`, `_IOR`, `_IOWR`).

## How regeneration works

Importing a `tinygrad.runtime.autogen` module regenerates its file when the file
is missing. The in-tree Autogen CI (`.github/workflows/autogen.yml`, jobs
`autogen` on Ubuntu and `autogen-mac` on macOS) proves this end to end:

1. Delete every regenerable `*.py` under `tinygrad/runtime/autogen/` — except
   `__init__.py`, `amd/`, `metal.py`, `iokit.py`, `corefoundation.py` and
   `libclang.py`.
2. Re-import each module group (`python3 -c "from tinygrad.runtime.autogen import cuda, ..."`,
   `from tinygrad.runtime.autogen.am import *`, and so on).
3. Regenerate the libclang bindings with `REGEN=1 python3 -c "from tinygrad.runtime.autogen import libclang"`
   (`tinygrad/runtime/support/autogen.py` is the libclang-based generator; its
   header notes the REGEN flag).
4. Fail on any `git diff`, uploading `autogen-ubuntu.patch` as the artifact.

`CACHE_VERSION` in `autogen.yml` is incremented when the downloads the
regeneration depends on change substantially, to keep CI hermetic.

## Working rules

- If a binding is wrong, change the generation input or generator behavior and
  regenerate; never edit the generated module text.
- Commit regenerated output together with the change that caused it; a
  regenerated-but-uncommitted tree fails the Autogen CI diff step.
- The generator imports host headers and toolchains (libclang, vendor SDK
  headers where installed). Regenerate on a machine matching CI where possible;
  a regeneration that silently depends on a locally-installed header version
  shows up as a CI diff, so treat CI's diff artifact as the authority.
- Platform exceptions (`metal.py`, `iokit.py`, `corefoundation.py`, `libclang.py`,
  `amd/`) have their own generation paths; inspect their headers before assuming
  the delete-and-import flow covers them.

## Verification for a bindings change

```sh
python -m pytest test/null/test_elf.py -x -q -n12   # structural sanity where relevant
python -m mypy tinygrad/                            # generated headers declare mypy disables; the rest must typecheck
python -m ruff check .
```

The authoritative check is the in-tree Autogen workflow on the pull request:
green means the committed files match what a clean regeneration produces.
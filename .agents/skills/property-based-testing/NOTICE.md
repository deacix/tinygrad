# Provenance and tinygrad integration

The `SKILL.md` and `references/*.md` in this directory are by **Trail of Bits**,
from [trailofbits/skills](https://github.com/trailofbits/skills/tree/123037ec8aed26f0d86327cc39137ee5043e5deb/plugins/property-based-testing/skills/property-based-testing),
revision `123037ec8aed26f0d86327cc39137ee5043e5deb`, reviewed 2026-09-19.
They remain licensed under **CC BY-SA 4.0**, including the adapted SKILL.md;
see [LICENSE.txt](LICENSE.txt). This notice's local integration guidance is
also CC BY-SA 4.0. No Trail of Bits endorsement is implied.

Only modification to the imported text: removed the top-level `effort: low`
frontmatter field, which the Agent Skills reference validator rejects and
OpenCode does not recognize. All five reference files are unchanged.
Upstream README/evals, client branding metadata and SVG assets are not installed;
none is referenced by the retained runtime guidance. No plugin or hooks are needed.

## Apply in tinygrad

Read this before applying the generic property catalog:

- Use existing Hypothesis infrastructure, not a new runtime dependency.
  See `test/null/test_graph_rewrite.py` and `test/null/test_dtype_spec.py`.
  The latter also imports PyTorch. Repo paths here are relative to the checkout root.
- State the input domain and independent oracle first. Integer identities need
  explicit bounds/overflow semantics; floating-point arithmetic is not generally
  associative or invertible. Account for dtype, NaNs, infinities, signed zero,
  subnormals and reduction order. A tolerance cannot rescue an invalid property.
- Bound shapes, allocations and examples. Follow the target module's Hypothesis
  settings; do not automatically disable deadlines or suppress health checks.
  Excessive `assume()` filtering can fail health checks or raise `Unsatisfiable`,
  not silently pass as some upstream examples imply. Preserve shrunk failures as
  small deterministic regressions where useful.
- Prefer test-side wrappers/oracles; do not add production abstractions just to
  satisfy the upstream refactoring suggestions. Honor the task's scope and
  repository conventions. Severity labels in the references rate test quality,
  not security vulnerabilities.
- Run focused tests with `-n12`. NULL tests verify graph structure, not numerical
  execution or speed. For value/gradient changes use a real backend and the
  NumPy/PyTorch comparisons in `test/backend/test_ops.py` with justified tolerances.

A small existing Hypothesis smoke (Python 3.11+, pytest, pytest-xdist,
Hypothesis and NumPy; no GPU or PyTorch needed):

```sh
DEV=NULL SPEC=2 python -m pytest test/null/test_graph_rewrite.py::TestRecurse -x -q -n12
```

This exercises existing generated matcher cases. It verifies the documented
entry point, not that an agent writes better properties after loading this skill.

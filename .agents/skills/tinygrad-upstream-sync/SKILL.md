---
name: tinygrad-upstream-sync
description: Merge upstream tinygrad into this fork without losing the fork's trees. Use when syncing with tinygrad/tinygrad master, resolving cross-subsystem merge conflicts, refreshing process-replay and sz.py expectations after a merge, or re-reviewing the committed agent skills against a new tree. Never rewrites published history and keeps the post-merge ladder bounded.
license: MIT
compatibility: Requires a tinygrad checkout with git fetch access to https://github.com/tinygrad/tinygrad and Python 3.11+ for the post-merge checks (ruff, mypy, pytest with NumPy). Process replay needs a captured baseline of both sides. No credentials beyond ordinary git fetch are required to load this skill.
---

# Sync upstream tinygrad into the fork

Read `AGENTS.md` from the repository root first. This fork (deacix/tinygrad)
tracks upstream `tinygrad/tinygrad` master while carrying fork-owned trees —
above all `tinygrad/llm/` and `.agents/skills/`. Every pull request's CI
remembers the relationship: `.github/workflows/szdiff.yml` fetches
`https://github.com/tinygrad/tinygrad`, prints "Behind N - Ahead M" against
`tinygrad/master`, and disables its core line-count bot while the branch is
behind. The goal of a sync is a merge where the fork's trees survive intact
and the upstream range is judged by the full ladder.

## 1. Set up the upstream remote the way CI names it

```sh
git remote add tinygrad https://github.com/tinygrad/tinygrad   # if absent
git fetch tinygrad master
git rev-list --left-right --count tinygrad/master...HEAD       # Behind / Ahead
```

Work on a branch, never on the protected integration branch. Merge upstream
into the fork (`git merge tinygrad/master`); never rebase published fork
history and never amend — `AGENTS.md` requires a new commit whenever a fix
would otherwise force-push.

## 2. Triage conflicts with the map

Open [references/conflict-map.md](references/conflict-map.md) and resolve by
its per-path rules: shared compiler/core files take upstream's change plus the
fork's delta re-applied on top; fork-owned files (`tinygrad/llm/**`,
`.agents/skills/**`) never take upstream content wholesale. When a conflict
mixes both sides in one hunk, split it: upstream's semantics first, then the
fork's behavior as its own clearly-scoped edit.

## 3. Run the post-merge ladder, cheap first

```sh
python -m ruff check .
python -m mypy tinygrad/
DEV=NULL SPEC=2 python -m pytest test/null/test_dtype.py test/null/test_graph_rewrite.py -x -q -n12
DEV=PYTHON python -m pytest test/test_tiny.py::TestTiny::test_plus test/test_tiny.py::TestTiny::test_jit -x -q -n12
```

Then widen only where the merge touched: `test/null/` for schedule/graph
changes, `test/backend/test_ops.py` selections for value-affecting changes.
Follow the property-based-testing caveats for dtype/symbolic changes. A merge
that changes generated kernels is exactly what
`test/external/process_replay/` exists for: capture both sides
(`CAPTURE_PROCESS_REPLAY=1`), run `process_replay.py`, and read its
`README.md` before deciding whether the kernel diffs are the merge's intent —
the `[pr]` title convention is for refactor/speedup PRs, not routine syncs.

## 4. Mind the size and provenance gates

- `sz.py` / `szdiff.yml`: the PR's line-count comment compares the fork's
  `tinygrad/` against upstream master's own `sz.py` (the base is always
  upstream master, by the workflow's security design). A sync should not
  inflate the core; run `python sz.py` locally when the merge edits core
  files and read the delta before pushing.
- Autogen: if the merge touches `tinygrad/runtime/autogen/**` or
  `runtime/support/autogen.py`, expect the in-tree Autogen CI to demand a
  clean regeneration (see `tinygrad-runtime-driver-development`'s autogen
  reference). Regenerate in the same change; never hand-merge generated files
  when a regeneration settles it.

## 5. Re-review the agent skills against the new tree

`.agents/skills/README.md` pins its review to a tree ("Reviewed on … against
tinygrad `<sha>`"). A merge that moved any path or command a skill names makes
that pin stale: grep each skill's named paths and commands against the merged
tree (`test -e` every path), update what drifted, and update the pin. The
skills' benefit depends on their claims matching the tree.

## 6. Ship the sync as a pull request

GitHub Issues are disabled on this repository; the pull request is the
tracker. Describe the merged range (`git log --oneline <old>..tinygrad/master`
count), the conflicts resolved and why, the ladder results, and anything the
fork deliberately kept. Push the branch and open the PR; never force-push, and
fix follow-ups with new commits.
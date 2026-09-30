# Notes

- Run tests with `-n12` for speed (e.g. `python -m pytest test/null/test_dtype.py -x -q -n12`)
- Run `python -m mypy tinygrad/` to typecheck
- Run `python -m ruff check .` to lint
- Read `./tinygrad/viz/README.md` for profiling and debugging rewrite rules
- Do not do amend commits. Always do a new commit if a force push to origin would be required.
- tinygrad has user space PCI drivers for AMD and NVIDIA GPUs. Do not insert the unneeded kernel modules.

# Repository skills

Load only the skill matching the task; these supplement the notes above.
- [property-based-testing](.agents/skills/property-based-testing/SKILL.md): write, review or shrink Hypothesis properties. Read its [tinygrad caveats](.agents/skills/property-based-testing/NOTICE.md) first, especially floating-point semantics and test bounds.
- [tinygrad-rewrite-debugging](.agents/skills/tinygrad-rewrite-debugging/SKILL.md): localize UOp, symbolic, scheduling and codegen regressions with SPEC and VIZ.
- [tinygrad-performance-triage](.agents/skills/tinygrad-performance-triage/SKILL.md): separate compile, dispatch, scheduler and kernel costs before optimizing.
- [tinygrad-llm-engine-development](.agents/skills/tinygrad-llm-engine-development/SKILL.md): develop LLM prefill/decode and cache semantics, attention/MoE, GGUF loading, tokenization and serving compatibility with bounded synthetic tests.
- [tinygrad-runtime-driver-development](.agents/skills/tinygrad-runtime-driver-development/SKILL.md): change runtime backends, generated bindings or user-space drivers, proven with mockgpu/process-replay evidence and hardware boundaries respected.
- [tinygrad-upstream-sync](.agents/skills/tinygrad-upstream-sync/SKILL.md): merge upstream tinygrad into the fork with the conflict map, the post-merge gate and the skill re-review pass.

See [.agents/skills/README.md](.agents/skills/README.md) for sources, installation, verification and exclusions.

# Multimodal input and multidevice placement

The modules added after this skill first landed: `tinygrad/llm/multimodal.py`
and `tinygrad/llm/vision.py` (native Qwen image inference), and
`tinygrad/llm/placement.py` (experimental contiguous layer placement for dense
models). Read the module before asserting any invariant below — these lines
name the checks the code itself enforces, and the tests that pin them.

## Image prompts: `multimodal.py`

- Token markers are fixed Qwen ids: `IMAGE_TOKEN_ID=248056`,
  `VISION_START_ID=248053`, `VISION_END_ID=248054`, with
  `IMAGE_MARKER="<|image_pad|>"` and `MEDIA_MARKERS` for template
  sanitizing. `ImageInputError(ValueError)` carries an HTTP `status`
  (default 400) — the serve path surfaces it as a client error, never a 500.
- `ImageLimits` + `image_size` implement the pinned Qwen2VL PIL smart-resize
  policy including ties-to-even rounding; image bytes rescale in float64 and
  cast float32 before normalization (HF reference behavior).
- `PreparedPrompt.__post_init__` is the invariant gate. A text prompt must
  carry no image metadata; an image prompt must have: grids of positive even
  `(1,h,w)` with `count == h*w//4` ordered non-overlapping spans, every span
  filled with `IMAGE_TOKEN_ID` placeholders, placeholder total matching the
  spans, `pixel_values` float32 of shape `(sum(h*w), 1536)` and `position_ids`
  int32 of shape `(3,1,len(tokens))` on the same device.
- `image_positions` keeps **logical** coordinates separate from physical
  sequence/cache offsets and returns the `rope_delta` between them; image
  patches get (t, h/2-grid, w/2-grid) coordinates and advance logical time by
  `max(h//2, w//2)`. Compare continuation and fresh-run state slices when
  touching this.
- `prepare_prompt` deep-copies `messages` and `tools`; the trusted template
  receives the image *kind*, never the URL or payload.

## Vision tower: `vision.py`

- `VisionConfig.from_dict`/`__post_init__` validate the tower config;
  `PatchEmbed`, `VisionMLP` (gelu `none` exact vs `tanh` fast path),
  `VisionAttention`, `VisionLayerNorm`, `VisionBlock`, `VisionMerger`
  (merges 2x2 patches: `x.shape[0]//4`) compose `QwenVision`.
- `VisionAttention` uses an explicit FP32 softmax to match reference eager
  attention in low precision — do not "optimize" it to the model dtype.

## Placement: `placement.py` (experimental)

- Pure and explicit: no device discovery. `LayerPlacement.validate` enforces
  positive integer layer counts covering every block exactly once, device
  grammar `(CPU|PYTHON|AMD|NV|CUDA)(:N)?` with distinct devices on **one**
  backend, `transfer` in `{'host','native'}`, `chunk_size` in `[1,32]`, and
  `chunk_size <= max_context`.
- `owner(parameter_name)`: `token_embd.weight` → first device,
  `output_norm.weight`/`output.weight` → last device, `blk.N.*.weight` → the
  block's device; anything else raises. `_placement_manifest` resolves the
  sole supported alias (tied `output.weight` ↔ `token_embd.weight`) and
  ignores `rope_freqs.weight`.
- `estimate_placement` is metadata accounting only — its docstring: fit is
  "unknown, not promised". Tied replicas are an explanatory subset of weight
  bytes, KV reserves batch-one FP16, one FP32 RoPE table per unique
  (owner, dimension, context, theta) is shared per owner, boundaries
  provision two full-chunk FP32 buffers, the final owner reserves
  `vocab_size*4` logits bytes; decoded scratch, JIT, allocator rounding and
  backend staging are unknown, and active loading may hold two largest
  payload copies. Never present these numbers as a hard memory guarantee.

## Regression shapes to prefer

Synthetic small weights, short token sequences, per-comparison copies of
mutable inputs (`generate` mutates its list). For placement: owner-local
payload loading, bounded stage JITs and serialized generation are the
invariants — compare fresh same-weight execution against continuation,
ragged chunking, divergence and position-zero restart, and compare outputs
or test-side logits **and** valid state slices (see the execution reference's
cache-comparison rules). The existing suites under `test/null/`
(`test_llm_multimodal.py`, `test_llm_server.py`, `test_llm_tokenizer.py`,
`test_qwen_evaluation.py`, `test_qwen_http.py`) show the current shapes, and
the placement suites under `test/unit/` (`test_llm_placement.py`,
`test_llm_placement_execution.py`, `test_llm_placement_server.py`,
`test_gguf_placement.py`) show the placement ones.
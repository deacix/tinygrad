"""Pinned, synthetic Qwen3.8 reference fixtures (no checkpoint weights).

Reference environment: transformers==5.8.0 torch==2.9.1 torchvision==0.24.1
pillow==12.3.0 numpy==2.4.6 jinja2==3.1.6. Source hashes below were checked
against Transformers commit 049d2bf1220747b6d39e2a978b9f5fe0defa1dca.

  python test/external/external_qwen_vl.py --generate
  python test/external/external_qwen_vl.py --verify
  python test/external/external_qwen_vl.py --check-reproducible

Generation is offline: small pinned tokenizer/processor metadata is bundled in
the archive. For bootstrap only, --metadata-dir accepts those two official JSON
files. --verify needs only NumPy; regeneration needs the reference environment.
  python test/external/external_qwen_vl.py --phase text --synthetic

Text phase uses tinygrad and the offline cast-tier fixtures (no reference install).
  python test/external/external_qwen_vl.py --phase vision --synthetic
Vision phase uses the pinned reference environment plus tinygrad for live full-tower
BF16 parity with existing synthetic weights rounded to BF16, unequal image lengths
and noninteger position interpolation. Reports max-absolute errors at rtol=atol=2e-2.
  python test/external/external_qwen_vl.py --phase fusion --synthetic
Fusion phase uses the pinned reference environment plus tinygrad, reusing existing
arrays for live complete synthetic vision/fusion/hybrid prefill and decode parity.
These phases do not establish full checkpoint, quantized or hardware acceptance.

--check-reproducible requires exact numerical-array hashes, archive bytes and manifest
identity, excluding only backend machine/Python/zlib execution provenance (reported
separately). It never claims cross-platform numerical or zlib byte identity in advance.

  python test/external/external_qwen_vl.py --phase processor-compare
  python test/external/external_qwen_vl.py --phase e2e --model-dir PATH --gguf PATH --image-suite controlled
  python test/external/external_qwen_vl.py --phase benchmark --model-dir PATH --gguf PATH --image-suite controlled

Processor comparison is offline and uses the pinned reference environment, bundled
RGB fixtures and a separately labeled synthetic vision tower. Real-checkpoint phases
NEVER fetch files: both paths must name the production trusted bundle/pairing. They
require sufficient hardware for Qwen3.8-27B, NOT the tiny synthetic model. New phases
emit JSON (diagnostics on stderr); --report also saves it. Benchmark runs the quality
suite once, then bounded warm repeats of text/single/max-area/two-image cases.
A first request is process-cold, not filesystem/compiler-cache cold. Device actual
peak memory is unavailable without an external profiler; counters are NOT that peak.
Passing quality alone does not establish component or AMD arithmetic acceptance.
"""
import argparse, contextlib, hashlib, importlib.metadata, io, json, platform, tempfile, zipfile, zlib
from pathlib import Path
from typing import Any
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "models/qwen_vl"
COMMIT = "049d2bf1220747b6d39e2a978b9f5fe0defa1dca"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SEED = 20260919
VERSIONS = {"transformers": "5.8.0", "torch": "2.9.1", "torchvision": "0.24.1", "pillow": "12.3.0", "numpy": "2.4.6", "jinja2": "3.1.6"}
SOURCE_HASHES = {
  "models/qwen3_5/modeling_qwen3_5.py": "641da34484f6b9769e28615c680cc4e3202afbf5bdbf1da7d56d5a3d678918c7",
  "models/qwen3_5/configuration_qwen3_5.py": "e2ca3e05c3f3e7ed2c4f452043bc33640c67a24799adb054ba24fab6badfbd8c",
  "models/qwen2_vl/image_processing_pil_qwen2_vl.py": "8a4f1b9ee48df6016c28f9e3aff65388ec513fd7344c713db8cb123c778a57ac",
  "image_processing_backends.py": "88ca316e2e804ccea2b0d6fb5971e2a4bc5765d89a114e902698b0b1909e34e1",
  "image_transforms.py": "dc08f6fba4097bba78a02b791f29dc7631b05ac7e11c98b36869fcef9bd8cdea",
  "cache_utils.py": "028a55fb163870d5f5d020473ee90fd3324e0a527ab33f1a626648f52690220a",
  "masking_utils.py": "54b23e9a269d44b6b85830b785b51fd8b62d34ebfd3c02665e40f86dc8416850",
  "modeling_rope_utils.py": "200d1fb4ed77132634761e279abba73d48d8bd8d8075de9c54ccc4f7f6671553",
  "utils/chat_template_utils.py": "b1f9ee2d4223677db1f6a6137ab413f5f69f91c418c99fd92165b5c55e77a326",
  "models/auto/image_processing_auto.py": "f1fecc9b070b80323a40367456d8055cdb5b446af1d3adbcf057f80a33afb4cb",
}
METADATA_HASHES = {"preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
                   "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"}
PROCESSOR = {"size": {"shortest_edge": 65536, "longest_edge": 262144}, "patch_size": 16, "temporal_patch_size": 2,
             "merge_size": 2, "image_mean": [0.5]*3, "image_std": [0.5]*3, "do_resize": True, "resample": 3,
             "do_rescale": True, "rescale_factor": 1/255, "do_normalize": True, "do_convert_rgb": True}
VISION:dict[str, Any] = {"depth": 2, "hidden_size": 16, "intermediate_size": 24, "num_heads": 2, "in_channels": 3,
          "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2, "out_hidden_size": 24,
          "num_position_embeddings": 25, "hidden_act": "gelu_pytorch_tanh", "initializer_range": 0.02}
TEXT:dict[str, Any] = {"vocab_size": 64, "hidden_size": 24, "intermediate_size": 40, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "linear_num_key_heads": 2, "linear_num_value_heads": 6,
        "linear_key_head_dim": 4, "linear_value_head_dim": 6, "linear_conv_kernel_dim": 4,
        "layer_types": ["linear_attention", "full_attention"], "max_position_embeddings": 2048, "rms_norm_eps": 1e-6,
        "hidden_act": "silu", "attention_bias": False, "attention_dropout": 0.0, "tie_word_embeddings": False,
        "initializer_range": 0.02, "use_cache": True,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.5,
                            "mrope_section": [2, 1, 1], "mrope_interleaved": True}}
TOLERANCE:dict[str, Any] = {"rtol": 1e-4, "atol": 1e-5}

def digest(data): return hashlib.sha256(data).hexdigest()

def deterministic_npz(arrays):
  data = io.BytesIO()
  with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for name, array in sorted(arrays.items()):
      buf = io.BytesIO()
      np.lib.format.write_array(buf, array, version=(1, 0), allow_pickle=False)
      info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
      info.create_system, info.external_attr = 3, 0o600 << 16
      archive.writestr(info, buf.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
  return data.getvalue()

def metadata(arrays, directory, archive_dir):
  out = {}
  for name, expected in METADATA_HASHES.items():
    if directory is None:
      with np.load(archive_dir / "reference.npz", allow_pickle=False) as src: raw = src["metadata." + name].tobytes()
    else: raw = (directory / name).read_bytes()
    if digest(raw) != expected: raise ValueError(f"metadata hash mismatch: {name}")
    arrays["metadata." + name] = np.frombuffer(raw, dtype=np.uint8).copy()
    out[name] = json.loads(raw)
  return out

def reference_imports():
  for name, version in VERSIONS.items():
    if importlib.metadata.version(name) != version: raise ValueError(f"reference requires {name}=={version}")
  import torch, transformers
  from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
  root = Path(transformers.__file__).parent
  for name, expected in SOURCE_HASHES.items():
    if digest((root / name).read_bytes()) != expected: raise ValueError(f"Transformers source mismatch: {name}")
  # These globals choose the official Torch fallbacks at module construction, even if optional packages are installed.
  for name in ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule", "FusedRMSNormGated"):
    setattr(hf, name, None)
  hf.is_fast_path_available = False
  torch.set_default_device("cpu")
  torch.set_default_dtype(torch.float32)
  torch.set_num_threads(1)
  torch.use_deterministic_algorithms(True)
  torch.manual_seed(SEED)
  return torch, hf

def rgb_pattern(height, width, checker=False):
  y, x = np.indices((height, width), dtype=np.int32)
  if checker: return np.stack((((x//7+y//5) % 2)*255, ((x//11) % 2)*255, ((y//3) % 2)*255), axis=-1).astype(np.uint8)
  return np.stack((x*255//max(width-1, 1), y*255//max(height-1, 1), (x+3*y) % 256), axis=-1).astype(np.uint8)

def processor_fixtures(arrays, meta):
  from PIL import Image
  from transformers import AutoImageProcessor
  # Local official metadata selects exactly the same PIL class as the pinned from_pretrained call, without Hub access.
  with tempfile.TemporaryDirectory() as tmp:
    Path(tmp, "preprocessor_config.json").write_text(json.dumps(meta["preprocessor_config.json"]))
    processor = AutoImageProcessor.from_pretrained(tmp, backend="pil", local_files_only=True, **PROCESSOR)
  if type(processor).__name__ != "Qwen2VLImageProcessorPil": raise ValueError("incorrect PIL processor selection")
  cases = []
  for name, h, w, checker in (("square", 256, 256, False), ("tiny", 17, 29, True),
                              ("rectangle", 95, 181, False), ("downscale", 517, 773, True)):
    prefix = "processor." + name + "."
    image = rgb_pattern(h, w, checker)
    arrays[prefix + "rgb"] = image
    resize, normalize = processor.resize, processor.normalize
    def capture_resize(*args, **kwargs):
      result = resize(*args, **kwargs)
      arrays[prefix + "resized_rgb"] = result.transpose(1, 2, 0).copy()
      return result
    def capture_normalize(*args, **kwargs):
      result = normalize(*args, **kwargs)
      arrays[prefix + "normalized_chw"] = result.copy()
      return result
    with patch.object(processor, "resize", capture_resize), patch.object(processor, "normalize", capture_normalize):
      output = processor(images=[Image.fromarray(image)], return_tensors="np")
    arrays[prefix + "pixel_values"], arrays[prefix + "grid_thw"] = output.pixel_values, output.image_grid_thw
    cases.append({"name": name, "input_hw": [h, w], "pattern": "checker" if checker else "gradient",
                  "grid_thw": output.image_grid_thw[0].tolist()})
  return processor, cases

def initialize_weights(torch, hf, model, seed):
  rng = np.random.default_rng(seed)
  modules = dict(model.named_modules())
  with torch.no_grad():
    for name, parameter in sorted(model.named_parameters()):
      module_name, kind = name.rsplit(".", 1)
      module = modules[module_name]
      values = rng.uniform(-1, 1, size=tuple(parameter.shape)).astype(np.float32)
      if kind == "A_log": values = values - 1
      elif kind == "dt_bias": pass
      elif kind == "weight" and isinstance(module, (torch.nn.LayerNorm, hf.Qwen3_5RMSNormGated)): values = 1 + values * 0.2
      elif isinstance(module, hf.Qwen3_5RMSNorm): values *= 0.2  # HF adds one at runtime, not here.
      elif kind == "bias": values *= 0.05
      elif isinstance(module, torch.nn.Embedding): values *= 0.5
      elif isinstance(module, torch.nn.Conv1d): values *= 0.3
      else: values *= 0.8 / np.sqrt(np.prod(parameter.shape[1:]))
      parameter.copy_(torch.from_numpy(values))

def put(arrays, name, value):
  arrays[name] = value.detach().cpu().numpy().copy()

def vision_fixtures(torch, hf, arrays, processor):
  from PIL import Image
  model = hf.Qwen3_5VisionModel(hf.Qwen3_5VisionConfig(**VISION, attn_implementation="eager")).eval()
  initialize_weights(torch, hf, model, SEED+1)
  for name, value in model.state_dict().items(): put(arrays, "vision.weights." + name, value)
  inputs = processor(images=[Image.fromarray(rgb_pattern(64, 96)), Image.fromarray(rgb_pattern(96, 64, True))],
                     do_resize=False, return_tensors="pt")
  pixels, grid = inputs.pixel_values, inputs.image_grid_thw
  put(arrays, "vision.pixel_values", pixels)
  put(arrays, "vision.grid_thw", grid)
  positions, rotary = model.fast_pos_embed_interpolate(grid), model.rot_pos_emb(grid)
  put(arrays, "vision.positions", positions)
  put(arrays, "vision.rotary_freqs", rotary)
  emb = torch.cat((rotary, rotary), dim=-1)
  put(arrays, "vision.rotary_cos", emb.cos())
  put(arrays, "vision.rotary_sin", emb.sin())
  handles = []
  def hook(name):
    def capture(_module, _inputs, output): put(arrays, name, output)
    return capture
  for name, module in [("patch_embed", model.patch_embed), ("merger", model.merger), ("merger_norm", model.merger.norm)]:
    handles.append(module.register_forward_hook(hook("vision." + name)))
  for i, block in enumerate(model.blocks):
    handles.append(block.register_forward_hook(hook(f"vision.block.{i}")))
    handles.append(block.attn.qkv.register_forward_hook(hook(f"vision.block.{i}.qkv")))
  output = model(pixels, grid)
  for handle in handles: handle.remove()
  for i in range(VISION["depth"]):
    q, k, _ = torch.from_numpy(arrays[f"vision.block.{i}.qkv"]).reshape(len(pixels), 3, VISION["num_heads"], -1).unbind(1)
    q_rot, k_rot = hf.apply_rotary_pos_emb_vision(q, k, emb.cos(), emb.sin())
    put(arrays, f"vision.block.{i}.rotary_q", q_rot)
    put(arrays, f"vision.block.{i}.rotary_k", k_rot)
  separate = torch.cat([model(chunk, g[None]).pooler_output for chunk, g in zip(pixels.split([24, 24]), grid)])
  put(arrays, "vision.separate_merger", separate)
  torch.testing.assert_close(output.pooler_output, separate, **TOLERANCE)
  return model

def text_fixtures(torch, hf, arrays):
  model = hf.Qwen3_5ForCausalLM(hf.Qwen3_5TextConfig(**TEXT, attn_implementation="eager")).eval()
  initialize_weights(torch, hf, model, SEED+2)
  for name, value in model.state_dict().items(): put(arrays, "text.weights." + name, value)
  ids = torch.tensor([[3, 17, 5, 31, 9, 23, 11, 42, 8, 19]])
  put(arrays, "text.input_ids", ids)
  put(arrays, "text.inputs_embeds", model.model.embed_tokens(ids))
  gdn = model.model.layers[0].linear_attn
  assert gdn.chunk_gated_delta_rule is hf.torch_chunk_gated_delta_rule
  assert gdn.recurrent_gated_delta_rule is hf.torch_recurrent_gated_delta_rule
  assert gdn.causal_conv1d_update is hf.torch_causal_conv1d_update
  assert gdn.causal_conv1d_fn is None
  assert isinstance(gdn.norm, hf.Qwen3_5RMSNormGated)
  handles = []
  def capture(name):
    def hook(_module, _args, output): put(arrays, "text.full." + name, output[0] if isinstance(output, tuple) else output)
    return hook
  for i, layer in enumerate(model.model.layers): handles.append(layer.register_forward_hook(capture(f"block.{i}")))
  handles.append(model.model.norm.register_forward_hook(capture("hidden")))
  handles.append(gdn.in_proj_qkv.register_forward_hook(capture("gdn_qkv")))
  handles.append(model.model.layers[1].self_attn.q_proj.register_forward_hook(capture("attn_q_gate")))
  chunk = gdn.chunk_gated_delta_rule
  def capture_chunk(query, key, value, **kwargs):
    for name, tensor in {"query": query, "key": key, "value": value, "g": kwargs["g"], "beta": kwargs["beta"]}.items():
      put(arrays, "text.full.gdn_" + name, tensor)
    output = chunk(query, key, value, **kwargs)
    put(arrays, "text.full.gdn_core", output[0])
    return output
  def save_cache(prefix, cache):
    for name, value in (("conv_state", cache.layers[0].conv_states), ("recurrent_state", cache.layers[0].recurrent_states),
                        ("keys", cache.layers[1].keys), ("values", cache.layers[1].values)):
      put(arrays, prefix + name, value)
  with patch.object(gdn, "chunk_gated_delta_rule", capture_chunk): full = model(ids, use_cache=True)
  for handle in handles: handle.remove()
  put(arrays, "text.full.logits", full.logits)
  save_cache("text.full.", full.past_key_values)
  put(arrays, "text.embeddings.logits", model(inputs_embeds=model.model.embed_tokens(ids), use_cache=False).logits)
  for mode, lengths in (("tokenwise", [1]*10), ("chunked", [3, 2, 5]), ("prefill_decode", [6, 1, 1, 1, 1])):
    cache, start, logits, conv, recurrent = None, 0, [], [], []
    blocks:list[list[Any]] = [[] for _ in model.model.layers]
    def collect(index):
      def hook(_module, _args, output): blocks[index].append(output.detach().clone())
      return hook
    handles = [layer.register_forward_hook(collect(i)) for i, layer in enumerate(model.model.layers)]
    for step, length in enumerate(lengths):
      out = model(ids[:, start:start+length], past_key_values=cache, use_cache=True)
      cache = out.past_key_values
      logits.append(out.logits)
      conv.append(cache.layers[0].conv_states.clone())
      recurrent.append(cache.layers[0].recurrent_states.clone())
      save_cache(f"text.{mode}.step.{step}.", cache)
      start += length
    for handle in handles: handle.remove()
    for i, outputs in enumerate(blocks): put(arrays, f"text.{mode}.block.{i}", torch.cat(outputs, dim=1))
    put(arrays, f"text.{mode}.logits", torch.cat(logits, dim=1))
    put(arrays, f"text.{mode}.conv_history", torch.stack(conv))
    put(arrays, f"text.{mode}.recurrent_history", torch.stack(recurrent))
    arrays[f"text.{mode}.chunk_lengths"] = np.array(lengths, dtype=np.int64)
    save_cache(f"text.{mode}.", cache)
    torch.testing.assert_close(torch.cat(logits, dim=1), full.logits, **TOLERANCE)
  low_norm = torch.tensor([[0, 0, 0, 0], [1e-8, -2e-8, 3e-8, -4e-8], [1e-4, 2e-4, -3e-4, 4e-4], [1, -2, 3, -4]])
  put(arrays, "gdn.low_norm.input", low_norm)
  put(arrays, "gdn.low_norm.output", hf.l2norm(low_norm, dim=-1, eps=1e-6))
  return model

def native_cast_text_fixtures(torch, hf, arrays, *, round_input=True, prefix="text.native_cast"):
  """Official HF algebra with ONLY native cast boundaries, separately named from the immutable FP32 oracle.

  Synthetic weights remain FP32. GDN rounds its normalized input and the gated RMSNorm output to FP16;
  all projections promote those rounded operands to FP32 with FP32 weights. Conv/recurrent state stays FP32.
  Full attention stores post-RoPE K and V in FP16, promoting reads for FP32 query/attention arithmetic.
  No head/layout/algebra patches: official grouped heads, epsilon, SiLU/sigmoid and cache updates are unchanged.
  round_input=False captures the existing codegen/late/coalesce.py pm_simplify_add_image rewrite removing a
  fused float->half->float roundtrip with FP32 projection weights. It does NOT remove materialized FP16 outputs.
  Keep BOTH variants; do not silently pass this backend/compiler distinction off as HF FP32 or nominal FP16.
  """
  from transformers.cache_utils import Cache
  model = hf.Qwen3_5ForCausalLM(hf.Qwen3_5TextConfig(**TEXT, attn_implementation="eager")).eval()
  model.load_state_dict({name.removeprefix("text.weights."): torch.from_numpy(value)
                         for name, value in arrays.items() if name.startswith("text.weights.")}, strict=True)
  gdn = model.model.layers[0].linear_attn
  assert gdn.chunk_gated_delta_rule is hf.torch_chunk_gated_delta_rule
  assert gdn.recurrent_gated_delta_rule is hf.torch_recurrent_gated_delta_rule
  assert gdn.causal_conv1d_fn is None and gdn.causal_conv1d_update is hf.torch_causal_conv1d_update
  assert isinstance(gdn.norm, hf.Qwen3_5RMSNormGated) and model.config._attn_implementation == "eager"
  def round_hidden(_module, inputs, kwargs):
    return inputs, kwargs | {"hidden_states": kwargs["hidden_states"].half().float()}
  def round_output(_module, inputs): return (inputs[0].half().float(), *inputs[1:])
  handles = [gdn.out_proj.register_forward_pre_hook(round_output)]
  if round_input: handles.append(gdn.register_forward_pre_hook(round_hidden, with_kwargs=True))
  original_update = Cache.update
  def update(cache, key, value, *args, **kwargs):
    k, v = original_update(cache, key.half(), value.half(), *args, **kwargs)
    return k.float(), v.float()
  ids = torch.from_numpy(arrays["text.input_ids"])
  def save_cache(prefix, cache):
    for name, value in (("conv_state", cache.layers[0].conv_states), ("recurrent_state", cache.layers[0].recurrent_states),
                        ("keys", cache.layers[1].keys), ("values", cache.layers[1].values)):
      put(arrays, prefix + name, value)
  comparisons = {}
  with patch.object(Cache, "update", update):
    for mode, lengths in (("full", [10]), ("tokenwise", [1]*10), ("chunked", [3, 2, 5]), ("prefill_decode", [6, 1, 1, 1, 1])):
      blocks:list[list[Any]] = [[] for _ in model.model.layers]
      def collect(index):
        def hook(_module, _args, output): blocks[index].append(output.detach().clone())
        return hook
      hooks = [layer.register_forward_hook(collect(i)) for i, layer in enumerate(model.model.layers)]
      cache, start, logits = None, 0, []
      for step, length in enumerate(lengths):
        out = model(ids[:, start:start+length], past_key_values=cache, use_cache=True)
        cache, start = out.past_key_values, start+length
        logits.append(out.logits)
        if mode == "tokenwise": save_cache(f"{prefix}.tokenwise.step.{step}.", cache)
      for hook in hooks: hook.remove()
      components = {"logits": torch.cat(logits, dim=1), **{f"block.{i}": torch.cat(b, dim=1) for i, b in enumerate(blocks)}}
      if mode == "full":
        for name, value in components.items(): put(arrays, prefix + ".full." + name, value)
        save_cache(prefix + ".full.", cache)
      else:
        for name, value in components.items():
          actual, expected = value.numpy(), arrays[prefix + ".full." + name]
          np.testing.assert_allclose(actual, expected, **TOLERANCE)
          comparisons[f"{mode}_{name}_vs_full"] = error_metrics(actual, expected)
    embeddings = model(inputs_embeds=torch.from_numpy(arrays["text.inputs_embeds"]), use_cache=True).logits.numpy()
    np.testing.assert_allclose(embeddings, arrays[prefix + ".full.logits"], **TOLERANCE)
    comparisons["embeddings_logits_vs_full"] = error_metrics(embeddings, arrays[prefix + ".full.logits"])
  for handle in handles: handle.remove()
  return comparisons


def coordinate_fixtures(torch, hf, arrays):
  config = hf.Qwen3_5Config(text_config=TEXT, vision_config=VISION)
  # Official class on meta: no extra random weights/allocations are needed for the coordinate methods.
  with torch.device("meta"): model = hf.Qwen3_5Model(config)
  cases = {"text": [], "rectangle": [[1, 4, 6]], "two_images": [[1, 4, 6], [1, 6, 4]],
           "square256": [[1, 16, 16]], "square1024": [[1, 64, 64]]}
  for name, grid_list in cases.items():
    ids, types, spans = [101, 102], [0, 0], []
    for t, h, w in grid_list:
      ids.append(248053)
      types.append(0)
      spans.append([len(ids), t*h*w//4])
      ids += [248056] * (t*h*w//4) + [248054, 103]
      types += [1] * (t*h*w//4) + [0, 0]
    ids += [104, 105]
    types += [0, 0]
    tokens, modality = torch.tensor([ids]), torch.tensor([types])
    grid = torch.tensor(grid_list, dtype=torch.int64).reshape(-1, 3)
    positions, delta = model.get_rope_index(tokens, modality, image_grid_thw=grid)
    for key, value in {"input_ids": tokens, "mm_token_type_ids": modality, "grid_thw": grid,
                       "position_ids": positions, "rope_delta": delta}.items(): put(arrays, f"coordinates.{name}.{key}", value)
    arrays[f"coordinates.{name}.image_spans"] = np.array(spans, dtype=np.int64).reshape(-1, 2)
  # Target mRoPE dimensions, independent of the small text model's reduced rotary width.
  cfg = hf.Qwen3_5TextConfig(**(TEXT | {"head_dim": 256, "rope_parameters": TEXT["rope_parameters"] |
                                     {"partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10]}}))
  rotary = hf.Qwen3_5TextRotaryEmbedding(cfg)
  pos = torch.from_numpy(arrays["coordinates.two_images.position_ids"])
  q = torch.linspace(-1, 1, 2*pos.shape[-1]*256).reshape(1, 2, -1, 256)
  k = q.flip(-1).clone()
  cos, sin = rotary(q, pos)
  q_rot, k_rot = hf.apply_rotary_pos_emb(q, k, cos, sin)
  for key, value in {"query": q, "key": k, "cos": cos, "sin": sin, "rotary_q": q_rot, "rotary_k": k_rot}.items():
    put(arrays, "coordinates.mrope." + key, value)
  return list(cases)

def template_fixtures(arrays, meta):
  from transformers.utils.chat_template_utils import render_jinja_template
  config = meta["tokenizer_config.json"]
  markers = {"<|vision_start|>": 248053, "<|vision_end|>": 248054, "<|image_pad|>": 248056, "<|video_pad|>": 248057}
  for token, token_id in markers.items():
    entry = config["added_tokens_decoder"][str(token_id)]
    assert entry["content"] == token and entry["special"]
  assert config["extra_special_tokens"]["image_token"] == "<|image_pad|>"
  messages = [{"role": "user", "content": [{"type": "text", "text": "What color?"}, {"type": "image"}]}]
  for name, options in (("thinking", {}), ("no_thinking", {"enable_thinking": False})):
    rendered, _ = render_jinja_template([messages], chat_template=config["chat_template"], add_generation_prompt=True, **options)
    assert rendered[0].count("<|vision_start|><|image_pad|><|vision_end|>") == 1
    assert rendered[0].endswith("<think>\n" if name == "thinking" else "<think>\n\n</think>\n\n")
    arrays["template." + name] = np.frombuffer(rendered[0].encode(), dtype=np.uint8).copy()
  return {"special_tokens": markers, "chat_template_sha256": digest(config["chat_template"].encode()),
          "scope": "Pinned metadata IDs and official Jinja renderer; full BPE vocabulary/GGUF tokenizer audit deferred to A2/E."}

def error_metrics(actual, expected):
  error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
  relative = error / np.maximum(np.abs(expected), 1e-12)
  return {"max_abs": float(error.max()), "mean_abs": float(error.mean()), "max_relative": float(relative.max()),
          "mean_relative": float(relative.mean()), "relative_denominator_floor": 1e-12}

def generate(output, metadata_dir=None, archive_dir=ROOT):
  torch, hf = reference_imports()
  arrays:dict[str, np.ndarray] = {}
  meta = metadata(arrays, metadata_dir, archive_dir)
  processor, cases = processor_fixtures(arrays, meta)
  template = template_fixtures(arrays, meta)
  with torch.no_grad():
    vision_fixtures(torch, hf, arrays, processor)
    text_fixtures(torch, hf, arrays)
    coordinate_cases = coordinate_fixtures(torch, hf, arrays)
    native_comparisons = native_cast_text_fixtures(torch, hf, arrays)
    lowered_comparisons = native_cast_text_fixtures(torch, hf, arrays, round_input=False, prefix="text.native_lowered")
  # No pickle/object arrays; normalize byte order and C layout before hashing or serialization.
  arrays = {name: np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<")) for name, a in arrays.items()}
  comparisons = {"vision_packed_vs_separate": error_metrics(arrays["vision.merger"], arrays["vision.separate_merger"])}
  for mode in ("tokenwise", "chunked", "prefill_decode", "embeddings"):
    components = ["logits"] if mode == "embeddings" else ["logits", "block.0", "block.1", "conv_state", "recurrent_state", "keys", "values"]
    for component in components:
      actual, expected = arrays[f"text.{mode}.{component}"], arrays[f"text.full.{component}"]
      np.testing.assert_allclose(actual, expected, **TOLERANCE)
      comparisons[f"text_{mode}_{component}_vs_full"] = error_metrics(actual, expected)
  for name, a in arrays.items():
    if a.dtype.hasobject or not np.isfinite(a).all(): raise ValueError(f"invalid reference array: {name}")
  raw = deterministic_npz(arrays)
  manifest = {
    "schema_version": 1,
    "source": {"model_id": "Qwen/Qwen3.8-27B", "model_revision": REVISION, "transformers_commit": COMMIT,
               "transformers_tag": "v5.8.0", "source_sha256": SOURCE_HASHES, "metadata_sha256": METADATA_HASHES,
               "source_license": "Apache-2.0", "weights": "Synthetic random weights only; no checkpoint tensors downloaded."},
    "versions": VERSIONS,
    "backend": {"device": "cpu", "attention": "eager", "optional_kernels": False, "dtype": "float32",
                "gdn_prefill": "torch_chunk_gated_delta_rule", "gdn_decode": "torch_recurrent_gated_delta_rule",
                "conv_decode": "torch_causal_conv1d_update", "torch_threads": 1, "deterministic_algorithms": True,
                "machine": platform.machine(), "python": platform.python_version(), "zlib": zlib.ZLIB_RUNTIME_VERSION},
    "seed": SEED, "weight_seeds": {"vision": SEED+1, "text": SEED+2},
    "rng": "NumPy PCG64 default_rng, sorted named_parameters; see initialize_weights; Torch seed fixes constructors only",
    "generator": {"path": "test/external/external_qwen_vl.py", "sha256": digest(Path(__file__).read_bytes()),
                  "regenerate": "python test/external/external_qwen_vl.py --generate",
                  "reproducibility": "--check-reproducible requires exact array hashes, archive bytes and manifest identity; "
                                     "backend machine/python/zlib execution provenance is compared and reported separately"},
    "configs": {"vision": VISION, "text": TEXT}, "processor": PROCESSOR, "processor_cases": cases,
    "processor_contract": {"class": "Qwen2VLImageProcessorPil", "backend": "pil", "input": "decoded RGB uint8 HWC",
                           "exif": "not exercised: synthetic RGB has no EXIF; policy belongs to B1",
                           "resample": "Pillow BICUBIC", "limits": "Oracle and later native use min65536/max262144 areas",
                           "patch_order": "image, t, merged_row, merged_col, intra_row, intra_col",
                           "patch_elements": "channel, temporal_duplicate, patch_y, patch_x",
                           "small_vision": "do_resize=False only for isolated 64x96/96x64 tower, not production preprocessing"},
    "coordinate_cases": coordinate_cases, "template": template,
    "layouts": {"vision.weights.*": "Official Qwen3_5VisionModel.state_dict keys; no model.visual prefix",
                "vision.block.i.qkv": "N,3*hidden_size; reshape N,3,heads,head_dim",
                "vision.block.i.rotary_q/k": "N,heads,head_dim", "vision.merger": "N/4,out_hidden_size",
                "text.weights.*": "Official Qwen3_5ForCausalLM.state_dict keys (model.* and lm_head.weight); no GGUF conversion",
                "text.*.recurrent_state": "B,V_heads,K_dim,V_dim, HF grouped order; tinygrad transposes last two and permutes heads",
                "text.*.conv_state": "B,2*K_heads*K_dim+V_heads*V_dim,kernel; oldest to newest including current token",
                "text.*.keys/values": "B,KV_heads,physical_sequence,head_dim",
                "text.*.step.i.*": "Snapshot after chunk i; chunk_lengths gives physical consumption",
                "text.*.recurrent_history": "step,B,V_heads,K_dim,V_dim",
                "text.full.gdn_query/key": "B,L,V_heads,K_dim; repeat_interleave ratio3, heads 0/1/2 and 3/4/5",
                "text.full.attn_q_gate": "B,L,heads*(2*head_dim); Q then gate WITHIN each head, sigmoid output gate",
                "coordinates.*.position_ids": "3,1,L (T,H,W); physical spans exclude delimiters, official not small-vocab IDs",
                "coordinates.mrope.*": "target rotary64/head256 sections[11,11,10], from two_images; query/key layout B,H,L,D",
                "metadata.* and template.*": "raw UTF-8 bytes as uint8, never object/pickle arrays"},
    "parity_tiers": {
      "hf_fp32_algebra": {"status": "captured", "tolerance": TOLERANCE, "processor_tolerance": {"rtol": 0, "atol": 1e-7},
                          "integer_tolerance": "exact", "self_variation": comparisons,
                          "scope": "Official HF eager+Torch fallback FP32; tolerances predeclared, not fitted to native errors"},
      "native_cast": {"status": "captured", "tolerance": TOLERANCE, "self_variation": native_comparisons,
                      "scope": "Official HF eager/F32 weights with explicit FP16-roundtrip GDN input and gated norm output; "
                               "FP16 post-RoPE K/V cache, FP32 reads/arithmetic/conv/recurrent state. No algebra/layout patches. "
                               "text.native_cast.* only; the original text.* HF algebra arrays remain unchanged.",
                      "algebra_drift": {name: error_metrics(arrays[f"text.native_cast.full.{name}"], arrays[f"text.full.{name}"])
                                        for name in ("logits", "block.0", "block.1", "conv_state", "recurrent_state", "keys", "values")},
                      "lowered_fp32_weights": {
                        "prefix": "text.native_lowered", "tolerance": TOLERANCE, "self_variation": lowered_comparisons,
                        "scope": "Existing codegen/late/coalesce.py pm_simplify_add_image removes fused float->half->float "
                                 "roundtrips: FP32-weight GDN input stays FP32. Materialized GDN output and KV remain FP16. "
                                 "Separate HF cast-only oracle; compiler behavior is explicitly probed in native tests. "
                                 "Not quantized, BF16, FP16-weight or hardware acceptance.",
                        "algebra_drift": {name: error_metrics(arrays[f"text.native_lowered.full.{name}"], arrays[f"text.full.{name}"])
                                          for name in ("logits", "block.0", "block.1", "conv_state", "recurrent_state", "keys", "values")}}},
      "amd_quantized": {"status": "deferred_E", "tolerance": None,
                        "scope": "Hardware/quantization-specific acceptance must be declared before AMD evaluation"}},
    "archive": {"file": "reference.npz", "sha256": digest(raw), "bytes": len(raw),
                "format": "sorted little-endian C-contiguous NPY v1.0, ZIP deflate9, epoch1980, Unix0600, no extra/comment"},
    "arrays": {name: {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": digest(a.tobytes(order="C"))}
               for name, a in sorted(arrays.items())},
  }
  manifest_raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
  if len(raw) + len(manifest_raw) > 2*1024*1024: raise ValueError(f"fixture budget exceeded: {len(raw) + len(manifest_raw)} bytes")
  output.mkdir(parents=True, exist_ok=True)
  (output / "reference.npz").write_bytes(raw)
  (output / "manifest.json").write_bytes(manifest_raw)
  print(f"Generated {len(arrays)} arrays, {len(raw) + len(manifest_raw)} bytes total; npz sha256={digest(raw)}")

def verify(directory):
  m = json.loads((directory / "manifest.json").read_text())
  raw = (directory / "reference.npz").read_bytes()
  if m["schema_version"] != 1: raise ValueError("unsupported fixture schema")
  if digest(raw) != m["archive"]["sha256"] or len(raw) != m["archive"]["bytes"]: raise ValueError("archive digest/size mismatch")
  if len(raw) + (directory / "manifest.json").stat().st_size > 2*1024*1024: raise ValueError("fixture budget exceeded")
  if digest(Path(__file__).read_bytes()) != m["generator"]["sha256"]: raise ValueError("generator changed: regenerate the fixtures")
  with np.load(io.BytesIO(raw), allow_pickle=False) as src:
    if set(src.files) != set(m["arrays"]): raise ValueError("array names mismatch")
    arrays = {name: src[name] for name in src.files}
  for name, a in arrays.items():
    spec = m["arrays"][name]
    if a.dtype.str != spec["dtype"] or list(a.shape) != spec["shape"]: raise ValueError(f"array schema mismatch: {name}")
    if not np.isfinite(a).all() or digest(a.tobytes(order="C")) != spec["sha256"]: raise ValueError(f"array digest mismatch: {name}")
  if deterministic_npz(arrays) != raw: raise ValueError("noncanonical NPZ archive")
  print(f"Verified {len(arrays)} arrays; npz sha256={digest(raw)}")

def compare_reproduction(expected_dir, actual_dir):
  """Identity is exact, not an allclose test; execution provenance is retained but is not identity."""
  expected = json.loads((expected_dir / "manifest.json").read_text())
  actual = json.loads((actual_dir / "manifest.json").read_text())
  provenance = {}
  for name, manifest in (("recorded", expected), ("regenerated", actual)):
    provenance[name] = {key:manifest["backend"].pop(key) for key in ("machine", "python", "zlib")}
  if expected["arrays"] != actual["arrays"]: raise ValueError("regeneration differs: numerical array hashes/schema (not execution provenance)")
  if (expected_dir / "reference.npz").read_bytes() != (actual_dir / "reference.npz").read_bytes():
    raise ValueError("regeneration differs: reference.npz bytes (array hashes match; check archive serialization/zlib)")
  if expected != actual: raise ValueError("regeneration differs: manifest identity (excluding machine/python/zlib execution provenance)")
  provenance["matching"] = provenance["recorded"] == provenance["regenerated"]
  return {"numerical_array_hashes":"exact", "archive":"byte-for-byte", "manifest_identity":"exact excluding execution provenance",
          "execution_provenance":provenance}


def vision_phase(manifest, arrays):
  """Live official BF16 full tower, no checkpoint and no patches to reference arithmetic."""
  from tinygrad import Tensor, nn, dtypes, Device
  from tinygrad.llm.vision import QwenVision, VisionConfig
  torch, hf = reference_imports()
  config = manifest["configs"]["vision"]
  tolerance = {"rtol":2e-2, "atol":2e-2}
  oracle = hf.Qwen3_5VisionModel(hf.Qwen3_5VisionConfig(**config, attn_implementation="eager")).eval().bfloat16()
  state = {k.removeprefix("vision.weights."):torch.from_numpy(v).bfloat16() for k,v in arrays.items() if k.startswith("vision.weights.")}
  oracle.load_state_dict(state, strict=True)
  native = QwenVision(VisionConfig.from_dict(config))
  # Transfer the same rounded values; materialize the BF16 storage before execution.
  weights = {k:Tensor(v.float().numpy()).cast(dtypes.bfloat16).realize() for k,v in state.items()}
  nn.state.load_state_dict(native, weights, verbose=False)
  assert all(v.dtype == dtypes.bfloat16 for v in nn.state.get_parameters(native))
  # A 5x5 learned position table interpolates at thirds/fifths, not integer-only locations.
  # Reuse 24 patches of image 1 and 16 of image 2: packed boundaries are deliberately unequal.
  grids = ((1,4,6), (1,4,4))
  lengths = [h*w for _,h,w in grids]
  pixels = np.concatenate((arrays["vision.pixel_values"][:24], arrays["vision.pixel_values"][24:40]))
  grid = torch.tensor(grids)
  expected, metrics = {}, {}
  def capture(name):
    def hook(_module, _inputs, output): expected[name] = output.float().numpy().copy()
    return hook
  def compare(name, actual, reference):
    if not np.isfinite(actual).all() or not np.isfinite(reference).all(): raise ValueError(f"nonfinite vision output: {name}")
    metrics[name] = error_metrics(actual, reference)
    np.testing.assert_allclose(actual, reference, **tolerance, err_msg=f"{name}: {metrics[name]}")
  with torch.no_grad():
    expected["positions"] = oracle.fast_pos_embed_interpolate(grid).float().numpy()
    expected["rotary"] = oracle.rot_pos_emb(grid).float().numpy()
    modules = [("patch_embed",oracle.patch_embed), ("merger_norm",oracle.merger.norm)]
    modules += [(f"block.{i}",block) for i,block in enumerate(oracle.blocks)]
    handles = [module.register_forward_hook(capture(name)) for name,module in modules]
    try: expected["merger"] = oracle(torch.from_numpy(pixels), grid).pooler_output.float().numpy()
    finally:
      for handle in handles: handle.remove()
    separate = torch.cat([oracle(p, g[None]).pooler_output for p,g in zip(torch.from_numpy(pixels).split(lengths), grid)]).float().numpy()
  compare("hf_packed_vs_separate", expected["merger"], separate)
  pos, rotary = native.positions(grids)
  compare("positions", pos.float().numpy(), expected["positions"])
  compare("rotary", rotary.float().numpy(), expected["rotary"])
  patches = native.patch_embed(Tensor(pixels))
  compare("patch_embed", patches.float().numpy(), expected["patch_embed"])
  x = patches + pos
  for i,block in enumerate(native.blocks):
    x = block(x, grids, rotary).realize()
    compare(f"block.{i}", x.float().numpy(), expected[f"block.{i}"])
  compare("merger_norm", native.merger.norm(x).float().numpy(), expected["merger_norm"])
  compare("merger", native.merger(x).float().numpy(), expected["merger"])
  # This is the unmodified production entry point, not only manually driven components.
  full = native(Tensor(pixels), grids).float().numpy()
  compare("full_tower", full, expected["merger"])
  separate = np.concatenate([native(Tensor(p), (g,)).float().numpy() for p,g in zip(np.split(pixels, np.cumsum(lengths)[:-1]), grids)])
  compare("native_packed_vs_separate", full, separate)
  return {"phase":"vision", "synthetic":True, "device":Device.DEFAULT, "versions":VERSIONS, "transformers_commit":COMMIT,
          "dtype":"bfloat16", "input_dtype":"float32 (official patch embed casts to BF16)", "grid_thw":grids,
          "patch_lengths":lengths, "tolerance":tolerance, "metrics":metrics, "max_abs":max(v["max_abs"] for v in metrics.values()),
          "scope":"Official HF eager BF16 full tower, existing synthetic weights rounded to BF16; component and packed/separate checks. "
                  "No checkpoint, quantized, decoder or hardware acceptance."}


def fusion_phase(manifest, arrays):
  """Live pinned complete synthetic vision->fusion->hybrid logits, not merely rotary component parity.

  Reuses immutable A1/A2 weights/pixels; writes no old arrays. Requires the reference environment AND tinygrad.
  FP32 HF, explicit native casts and existing lowered FP32-weight casts are separate executions/metrics.
  """
  from dataclasses import replace
  from tinygrad import Tensor, nn, dtypes, Device
  from tinygrad.llm.multimodal import PreparedPrompt, embed_prompt, image_positions
  from tinygrad.llm.vision import QwenVision, VisionConfig
  from test.unit.test_llm_multimodal import native_text_model
  from transformers.cache_utils import Cache
  torch, hf = reference_imports()
  cfg = hf.Qwen3_5Config(text_config=TEXT, vision_config=VISION, image_token_id=248056 % 64,
                         vision_start_token_id=248053 % 64, vision_end_token_id=248054 % 64, video_token_id=248057 % 64)
  cfg._attn_implementation = "eager"
  oracle = hf.Qwen3_5ForConditionalGeneration(cfg).eval()
  state = {}
  for key, value in arrays.items():
    if key.startswith("vision.weights."): state["model.visual."+key.removeprefix("vision.weights.")] = torch.from_numpy(value)
    if key.startswith("text.weights."):
      name = key.removeprefix("text.weights.")
      state[name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name] = torch.from_numpy(value)
  oracle.load_state_dict(state, strict=True)
  native, _, _ = native_text_model(manifest, arrays)
  for block in native.blk: block.config = replace(block.config, rope_sections=(2, 1, 1))
  vision = QwenVision(VisionConfig.from_dict(VISION))
  nn.state.load_state_dict(vision, {k.removeprefix("vision.weights."):Tensor(v) for k,v in arrays.items() if k.startswith("vision.weights.")},
                          verbose=False)
  # Only the tiny test vocabulary maps official marker IDs. Production uses the audited GGUF tokenizer/table unchanged.
  embedding = native.token_embd
  class SmallVocabulary:
    weight = embedding.weight
    def __call__(self, ids): return embedding(ids % TEXT["vocab_size"])
  native.token_embd = SmallVocabulary()
  metrics, final_logits = {}, []
  def compare(name, actual, expected, assert_close=True):
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), name
    metrics[name] = error_metrics(actual, expected)
    if assert_close: np.testing.assert_allclose(actual, expected, **TOLERANCE, err_msg=name)
  gdn = oracle.model.language_model.layers[0].linear_attn
  original_update = Cache.update
  def update(cache, key, value, *args, **kwargs):
    k, v = original_update(cache, key.half(), value.half(), *args, **kwargs)
    return k.float(), v.float()
  def round_input(_module, inputs, kwargs): return inputs, kwargs | {"hidden_states":kwargs["hidden_states"].half().float()}
  def round_output(_module, inputs): return (inputs[0].half().float(), *inputs[1:])
  with torch.no_grad():
    for case, variant in (("rectangle", 0), ("two_images", 0), ("two_images", 1)):
      label, prefix = f"{case}.pixels{variant}", f"coordinates.{case}."
      tokens = arrays[prefix+"input_ids"][0].tolist()
      grids = tuple(tuple(int(v) for v in g) for g in arrays[prefix+"grid_thw"])
      spans = tuple(tuple(int(v) for v in s) for s in arrays[prefix+"image_spans"])
      pos, delta = image_positions(tuple(tokens), grids, spans)
      np.testing.assert_array_equal(pos, arrays[prefix+"position_ids"])
      assert delta == int(arrays[prefix+"rope_delta"].item())
      pixels = arrays["vision.pixel_values"][:sum(h*w for _,h,w in grids)].copy() * (1 if variant == 0 else -1)
      prompt = PreparedPrompt(tuple(tokens), Tensor(pixels), grids, spans, Tensor(pos), delta)
      with patch.object(QwenVision, "__call__", autospec=True, side_effect=QwenVision.__call__) as encode:
        fused = embed_prompt(native, vision, prompt)
        encode.assert_called_once()
      references = {}
      for tier in ("hf_fp32", "native_cast", "native_lowered"):
        capture, handles = {}, []
        def capture_inputs(_module, args, kwargs):
          capture["embeddings"], capture["positions"] = kwargs["inputs_embeds"].clone(), kwargs["position_ids"].clone()
        handles.append(oracle.model.language_model.register_forward_pre_hook(capture_inputs, with_kwargs=True))
        if tier != "hf_fp32": handles.append(gdn.out_proj.register_forward_pre_hook(round_output))
        if tier == "native_cast": handles.append(gdn.register_forward_pre_hook(round_input, with_kwargs=True))
        oracle.model.rope_deltas = None
        with patch.object(Cache, "update", update) if tier != "hf_fp32" else contextlib.nullcontext():
          output = oracle(input_ids=torch.tensor([tokens]) % 64, pixel_values=torch.from_numpy(pixels),
                          image_grid_thw=torch.tensor(grids), mm_token_type_ids=torch.from_numpy(arrays[prefix+"mm_token_type_ids"]), use_cache=True)
          for handle in handles[:1]: handle.remove()
          result = {"embeddings":capture["embeddings"].numpy(), "positions":capture["positions"].numpy(),
                    "logits":[output.logits[:, -1].numpy()], "tokens":[]}
          for step in range(6):
            token = output.logits[:, -1].argmax(-1, keepdim=True)
            result["tokens"].append(int(token.item()))
            if step < 5:
              output = oracle(input_ids=token, past_key_values=output.past_key_values, use_cache=True)
              result["logits"].append(output.logits[:, -1].numpy())
          references[tier] = result
        for handle in handles[1:]: handle.remove()
      ref = references["native_lowered"]
      compare(label+".fused_embeddings", fused.numpy(), ref["embeddings"])
      np.testing.assert_array_equal(pos, ref["positions"])
      logits = native.forward_embeddings(fused, 0, Tensor(pos)).numpy()
      compare(label+".prefill_logits", logits, ref["logits"][0])
      final_logits.append(logits)
      for step in range(5):
        physical = len(tokens)+step
        logits = native.forward_embeddings(embedding(Tensor([[ref["tokens"][step]]], dtype=dtypes.int32)).float(), physical,
                    Tensor.full((3,1,1), physical+delta, dtype=dtypes.int32)).numpy()
        compare(label+f".decode_logits.{step}", logits, ref["logits"][step+1])
      gen = native.generate([t % 64 for t in tokens], inputs_embeds=fused, position_ids=Tensor(pos), rope_delta=delta)
      try: assert [next(gen) for _ in range(6)] == ref["tokens"]
      finally: gen.close()
      for tier in ("native_cast", "hf_fp32"):
        compare(label+f".lowered_vs_{tier}", np.concatenate(ref["logits"]), np.concatenate(references[tier]["logits"]), False)
  assert np.abs(final_logits[-1]-final_logits[-2]).max() > 1e-3, "pixel ablation must change fused decoder logits"
  return {"phase":"fusion", "synthetic":True, "device":Device.DEFAULT, "versions":VERSIONS, "transformers_commit":COMMIT,
          "tolerance":TOLERANCE, "scope":"Complete tiny weighted vision + hybrid decoder, prefill and five decode steps; "
          "native comparisons use the lowered FP32-weight cast oracle. No checkpoint/quantized/hardware acceptance.", "metrics":metrics}


def array_stats(value):
  value = np.asarray(value)
  if not np.isfinite(value).all(): raise ValueError("nonfinite evaluation output")
  return {"shape":list(value.shape), "dtype":str(value.dtype), "sha256":digest(value.tobytes()),
          "min":float(value.min()), "max":float(value.max()), "mean":float(value.mean()), "std":float(value.std())}

def image_url(image):
  import base64
  buf = io.BytesIO()
  image.save(buf, format="PNG", optimize=False, compress_level=9)
  return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def processor_compare(directory):
  from PIL import Image
  from tinygrad.llm.multimodal import ImageLimits, preprocess_image
  torch, hf = reference_imports()
  import transformers
  from transformers import AutoImageProcessor
  tv_source = "models/qwen2_vl/image_processing_qwen2_vl.py"
  tv_hash = "f19497685281402691f0049a0ab5be8085f92fb5c37adfcc680e4ccd17863a77"
  if digest((Path(transformers.__file__).parent/tv_source).read_bytes()) != tv_hash: raise ValueError("torchvision processor source mismatch")
  with np.load(directory/"reference.npz", allow_pickle=False) as src: arrays = {k:src[k] for k in src.files}
  meta = metadata({}, None, directory)
  # Feature drift is an explicitly synthetic sensitivity probe, never checkpoint vision parity.
  tower = hf.Qwen3_5VisionModel(hf.Qwen3_5VisionConfig(**VISION, attn_implementation="eager")).eval()
  tower.load_state_dict({k.removeprefix("vision.weights."):torch.from_numpy(v) for k,v in arrays.items() if k.startswith("vision.weights.")})
  records, outputs, settings = [], {}, {}
  with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
    Path(tmp, "preprocessor_config.json").write_text(json.dumps(meta["preprocessor_config.json"]))
    for policy, overrides in (("production", PROCESSOR), ("checkpoint_default", {})):
      processors = {b:AutoImageProcessor.from_pretrained(tmp, backend=b, local_files_only=True, **overrides) for b in ("pil", "torchvision")}
      settings[policy] = {b:{"class":type(p).__name__, **{k:dict(p.size) if k == "size" else getattr(p, k) for k in PROCESSOR}}
                          for b,p in processors.items()}
      for name in ("square", "tiny", "rectangle", "downscale"):
        rgb = arrays[f"processor.{name}.rgb"]
        row = {"fixture":name, "policy":policy, "input":array_stats(rgb), "backends":{}}
        features = {}
        for backend, processor in processors.items():
          image = Image.fromarray(rgb)
          result = processor(images=[image], return_tensors="pt")
          repeated = processor(images=[image], return_tensors="pt")
          pixels, grid = result.pixel_values.numpy(), result.image_grid_thw.numpy()
          np.testing.assert_array_equal(pixels, repeated.pixel_values.numpy())
          np.testing.assert_array_equal(grid, repeated.image_grid_thw.numpy())
          features[backend] = tower(result.pixel_values, result.image_grid_thw).pooler_output.numpy()
          row["backends"][backend] = {"grid_thw":grid.tolist(), "pixels":array_stats(pixels),
                                       "synthetic_features":array_stats(features[backend]), "repeat_exact":True}
          outputs[policy, name, backend] = pixels, grid
        pil, grid = outputs[policy, name, "pil"]
        tv, tv_grid = outputs[policy, name, "torchvision"]
        np.testing.assert_array_equal(grid, tv_grid)
        row["pil_vs_torchvision"] = {"pixels":error_metrics(pil, tv), "synthetic_features":error_metrics(features["pil"], features["torchvision"])}
        if policy == "production":
          native, native_grid = preprocess_image(image_url(Image.fromarray(rgb)), ImageLimits())
          np.testing.assert_array_equal([native_grid], grid)
          np.testing.assert_allclose(native, pil, rtol=0, atol=1e-7)
          np.testing.assert_array_equal(pil, arrays[f"processor.{name}.pixel_values"])
          row["native_vs_pil"] = error_metrics(native, pil)
        else:
          row["vs_production"] = {}
          for backend in processors:
            previous, previous_grid = outputs["production", name, backend]
            current, current_grid = outputs[policy, name, backend]
            same = np.array_equal(previous_grid, current_grid)
            row["vs_production"][backend] = {"same_grid":same, "pixels":error_metrics(previous, current) if same else None,
                                             "note":"different grids are not elementwise comparable" if not same else "same grid"}
        records.append(row)
  return {"phase":"processor-compare", "versions":VERSIONS, "transformers_commit":COMMIT, "checkpoint_revision":REVISION,
          "metadata_sha256":METADATA_HASHES, "extra_source_sha256":{tv_source:tv_hash}, "hardware":platform.platform(),
          "backend":"CPU HF PIL and torchvision; native host preprocessing", "dtype":"float32", "context":None,
          "checkpoint_weights_loaded":False, "feature_scope":"Existing seeded tiny FP32 vision fixture only, NOT real checkpoint features",
          "tolerance":{"native_vs_pil":{"rtol":0, "atol":1e-7}, "pil_vs_torchvision":"drift report, not bitwise parity"},
          "settings":settings, "fixtures":records}

def normalize_answer(text):
  import re, unicodedata
  # Exact normalized equality, NOT a substring/keyword test: "red and blue", "not red" and "blue, red" all fail "red".
  return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip().strip(". !?\"'")

def answer_matches(answer, expected): return normalize_answer(answer) in {normalize_answer(x) for x in expected}

def controlled_suite():
  """20 predeclared visual questions plus unscored-in-the-20 controls; no system fonts/network/RNG needed."""
  from PIL import Image, ImageDraw
  colors = {"red":(255,0,0), "blue":(0,0,255), "green":(0,180,0), "yellow":(255,255,0), "gray":(128,128,128)}
  images = {k:Image.new("RGB", (512,512) if k == "yellow" else (256,256), v) for k,v in colors.items()}
  for shape in ("circle", "square", "triangle", "rectangle"):
    image = Image.new("RGB", (256,256), "white")
    draw = ImageDraw.Draw(image)
    if shape == "circle": draw.ellipse((48,48,208,208), fill="black")
    elif shape == "triangle": draw.polygon([(128,32),(32,224),(224,224)], fill="black")
    else: draw.rectangle((48,48,208,208) if shape == "square" else (24,80,232,176), fill="black")
    images[shape] = image
  # Explicit 5x7 glyphs make OCR pixels reproducible across Pillow/system font versions.
  glyphs = {"S":["01111","10000","10000","01110","00001","00001","11110"],
            "T":["11111","00100","00100","00100","00100","00100","00100"],
            "O":["01110","10001","10001","10001","10001","10001","01110"],
            "P":["11110","10001","10001","11110","10000","10000","10000"],
            "C":["01111","10000","10000","10000","10000","10000","01111"],
            "A":["01110","10001","10001","11111","10001","10001","10001"],
            "1":["00100","01100","00100","00100","00100","00100","01110"],
            "2":["01110","10001","00001","00010","00100","01000","11111"],
            "3":["11110","00001","00001","01110","00001","00001","11110"],
            "4":["00010","00110","01010","10010","11111","00010","00010"]}
  for word in ("STOP", "CAT", "123", "42"):
    image = Image.new("RGB", (256,256), "white")
    draw, scale = ImageDraw.Draw(image), 10
    left, top = (256-(len(word)*6-1)*scale)//2, 93
    for i, char in enumerate(word):
      for y, line in enumerate(glyphs[char]):
        for x, bit in enumerate(line):
          if bit == "1": draw.rectangle((left+(i*6+x)*scale, top+y*scale, left+(i*6+x+1)*scale-1, top+(y+1)*scale-1), fill="black")
    images[word] = image
  for name, heights in (("chart_red", (150,65)), ("chart_blue", (65,150)), ("chart_equal", (120,120)), ("chart_three", (70,150,110))):
    image = Image.new("RGB", (256,256), "white")
    draw = ImageDraw.Draw(image)
    draw.line((20,20,20,220,240,220), fill="black", width=3)
    for i, height in enumerate(heights): draw.rectangle((42+i*68,220-height,86+i*68,218), fill=colors[("red","blue","green")[i]])
    images[name] = image
  cases = []
  def add(name, category, media, question, expected, scored=True):
    cases.append({"id":name, "category":category, "images":list(media), "question":question, "expected":list(expected), "scored":scored})
  color_question = "What color fills the image? Reply with just one color word. If there is no image, reply unknown."
  for color in ("red", "blue", "green", "yellow"): add("color_"+color, "color", [color], color_question, [color])
  for shape in ("circle", "square", "triangle", "rectangle"):
    add("shape_"+shape, "shape", [shape], "Name the single black shape. Reply with just the shape name.", [shape])
  for word in ("STOP", "CAT", "123", "42"):
    add("ocr_"+word.lower(), "ocr", [word], "Read the text in the image. Reply with only the text.", [word])
  for name, question, answer in (("chart_red","Which bar is taller? Reply red or blue.","red"),
                                  ("chart_blue","Which bar is taller? Reply red or blue.","blue"),
                                  ("chart_equal","Are the two bars the same height? Reply yes or no.","yes"),
                                  ("chart_three","How many colored bars are there? Reply with a digit.","3")):
    add(name, "chart", [name], question, [answer])
  for name, media, answer in (("order_red_blue", ["red","blue"], "red"), ("order_blue_red", ["blue","red"], "blue"),
                              ("order_circle_triangle", ["circle","triangle"], "circle"),
                              ("order_triangle_circle", ["triangle","circle"], "triangle")):
    noun = "color" if media[0] in colors else "shape"
    add(name, "ordering", media, f"What {noun} is in the FIRST image? Reply with just the {noun} name.", [answer])
  add("text_math", "text_control", [], "What is 2 + 2? Reply with just a digit.", ["4"], False)
  add("text_capital", "text_control", [], "What is the capital of France? Reply with just the city name.", ["Paris"], False)
  add("ablation_blank", "ablation", ["gray"], color_question, ["gray", "grey"], False)
  add("ablation_removed", "ablation", [], color_question, ["unknown"], False)
  return images, cases

ABLATION_PAIRS = (("color_red", "color_blue"), ("color_red", "ablation_blank"), ("color_red", "ablation_removed"))

def suite_manifest(images, cases):
  return {"name":"controlled-v1", "cases":cases,
          "images":{name:array_stats(np.asarray(image)) for name,image in images.items()},
          "scoring":"NFKC/casefold/whitespace and terminal punctuation normalization, then exact equality to predeclared answers",
          "acceptance":{"visual_correct_min":18, "visual_total":20, "all_ordering":True, "all_text_controls":True,
                        "all_ablations":True, "all_ablation_pair_endpoints":True},
          "ablation_pairs":[list(pair) for pair in ABLATION_PAIRS]}

def quality_summary(cases, results):
  by_id = {r["id"]:r for r in results}
  passed = {c["id"]:answer_matches(by_id[c["id"]]["answer"], c["expected"]) and by_id[c["id"]]["stop_reason"] == "eos" for c in cases}
  visual = sum(passed[c["id"]] for c in cases if c["scored"])
  groups = {group:all(passed[c["id"]] for c in cases if c["category"] == group) for group in ("ordering", "text_control", "ablation")}
  pairs = [{"endpoints":list(pair), "passed":all(passed[name] for name in pair)} for pair in ABLATION_PAIRS]
  groups["ablation_pairs"] = all(pair["passed"] for pair in pairs)
  return {"visual_correct":visual, "visual_total":20, "minimum":18, "groups_passed":groups, "case_passed":passed,
          "ablation_pairs":pairs, "passed":visual >= 18 and all(groups.values())}

def rss_record(raw, system):
  # getrusage is bytes on Darwin, KiB on Linux. Do not guess units on other OSes.
  unit = "bytes" if system == "Darwin" else "KiB" if system == "Linux" else "platform-dependent"
  return {"raw":raw, "raw_unit":unit, "bytes":raw*(1024 if system == "Linux" else 1) if unit != "platform-dependent" else None,
          "scope":"process lifetime high-water RSS, not per-request incremental peak; includes libraries and mapped weights"}

def memory_snapshot():
  from tinygrad.helpers import GlobalCounters
  try:
    import resource
    host = rss_record(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, platform.system())
  except ImportError: host = {"bytes":None, "scope":"resource.getrusage unavailable"}
  return {"host_peak_rss":host, "tinygrad_live_buffer_bytes":GlobalCounters.mem_used,
          "tinygrad_live_buffer_bytes_by_device":dict(GlobalCounters.mem_used_per_device),
          "actual_device_peak_bytes":None,
          "device_scope":"GlobalCounters tracks live Buffer allocations, NOT allocator-reserved/driver memory or actual peak; "
                         "external profiler required"}

def installed_versions():
  versions = {"python":platform.python_version()}
  for name in ("tinygrad", *VERSIONS):
    try: versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: versions[name] = None
  return versions

def trusted_checkpoint_record(model_dir, gguf):
  from tinygrad.llm.vision import BUNDLE_FILES, GGUF_DIGEST, MODEL_REVISION, GGUF_REVISION
  def record(path, spec):
    size, checksum = spec
    return {"path":str(path.resolve()), "bytes":size, "digest":checksum,
            "algorithm":"git-blob-sha1" if len(checksum) == 40 else "sha256", "validated":True}
  return {"model_revision":MODEL_REVISION, "gguf_revision":GGUF_REVISION, "gguf":record(gguf, GGUF_DIGEST),
          "bundle":{name:record(model_dir/name, spec) for name,spec in BUNDLE_FILES.items()},
          "tokenizer":"SimpleTokenizer from validated GGUF token table; no alternate tokenizer"}

def timed_request(model, vision, tokenizer, template, limits, case, urls, args):
  import time
  from tinygrad import Device
  from tinygrad.llm.multimodal import prepare_prompt, embed_prompt
  from tinygrad.llm.vision import QwenVision
  device = model.token_embd.weight.device
  def sync(): Device[device].synchronize()
  model.reset_generation_state()  # Also isolate text repeats: no prefix-cache shortcut in warm measurements.
  sync()
  samples, times = {"before":memory_snapshot()}, {}
  start = time.perf_counter()
  content = [{"type":"image_url", "image_url":{"url":urls[name]}} for name in case["images"]]
  content.append({"type":"text", "text":case["question"]})
  prompt = prepare_prompt([{"role":"user", "content":content}], tokenizer, template, limits=limits, device=device,
                          max_context=model.max_context, template_kwargs={"enable_thinking":False})
  if prompt.starts_reasoning: raise ValueError("quality protocol requires non-thinking answers")
  if len(prompt.tokens)+args.max_output_tokens > model.max_context: raise ValueError("prompt plus output budget exceeds context")
  sync()
  times["preprocess_s"] = time.perf_counter()-start
  samples["prepared"] = memory_snapshot()
  times["vision_s"] = 0.0
  embeddings = None
  if prompt.pixel_values is not None:
    encode = QwenVision.__call__
    def timed_vision(self, *a, **kw):
      sync()
      before = time.perf_counter()
      result = encode(self, *a, **kw)
      result.realize()
      sync()
      times["vision_s"] += time.perf_counter()-before
      samples["vision"] = memory_snapshot()
      return result
    before = time.perf_counter()
    with patch.object(QwenVision, "__call__", timed_vision): embeddings = embed_prompt(model, vision, prompt)
    sync()
    times["embed_fusion_s"] = time.perf_counter()-before-times["vision_s"]
  else: times["embed_fusion_s"] = 0.0
  samples["embedded"] = memory_snapshot()
  gen = model.generate(list(prompt.tokens), chunk_size=args.chunk_size, temperature=0.0,
                       inputs_embeds=embeddings, position_ids=prompt.position_ids, rope_delta=prompt.rope_delta)
  ids, latencies, stop_reason = [], [], "output_limit"
  try:
    for step in range(args.max_output_tokens):
      before = time.perf_counter()
      try: token = next(gen)
      except StopIteration:
        stop_reason = "context_limit"
        break
      sync()
      elapsed = time.perf_counter()-before
      latencies.append(elapsed)
      if step == 0:
        times["prefill_s"], times["first_token_s"] = elapsed, time.perf_counter()-start
        samples["first_token"] = memory_snapshot()
      ids.append(token)
      if tokenizer.is_end(token):
        stop_reason = "eos"
        break
    sync()
    times["decode_s"] = sum(latencies[1:])
    times["total_s"] = time.perf_counter()-start
    samples["completed"] = memory_snapshot()
  finally:
    gen.close()
    model.reset_generation_state()
  answer = tokenizer.decode([token for token in ids if not tokenizer.is_end(token)])
  decoded = max(0, len(ids)-1)
  return {"id":case["id"], "category":case["category"], "answer":answer, "expected":case["expected"],
          "answer_match":answer_matches(answer, case["expected"]), "stop_reason":stop_reason, "generated_token_ids":ids,
          "prompt_tokens":len(prompt.tokens), "generated_tokens_including_eos":len(ids), "decode_tokens_including_eos":decoded,
          "grid_thw":prompt.grid_thw, "visual_tokens":sum(count for _,count in prompt.image_spans), "rope_delta":prompt.rope_delta,
          "timing":times, "decode_token_seconds":latencies[1:],
          "decode_tokens_per_second":decoded/times["decode_s"] if times["decode_s"] else None,
          "memory":samples, "sampled_max_live_buffer_bytes":max(x["tinygrad_live_buffer_bytes"] for x in samples.values())}

def checkpoint_phase(args):
  import os, time
  import jinja2
  from tinygrad import Device, Tensor, nn
  from tinygrad.helpers import DEV, getenv
  from tinygrad.llm.kernels.amd import amd_custom_kernels_supported
  from tinygrad.llm.cli import SimpleTokenizer
  from tinygrad.llm.model import Transformer
  from tinygrad.llm.multimodal import ImageLimits
  from tinygrad.llm.vision import validate_vision_bundle, validate_vision_metadata, load_vision
  if not args.model_dir.is_dir() or not args.gguf.is_file():
    raise ValueError("real phases require existing local --model-dir and --gguf; no downloads")
  # No model construction, template compilation or warmup before production trust validation.
  start = time.perf_counter()
  verified = validate_vision_bundle(args.model_dir, args.gguf)
  validation_s = time.perf_counter()-start
  checkpoint = trusted_checkpoint_record(args.model_dir, args.gguf)
  start = time.perf_counter()
  model, kv = Transformer.from_gguf(args.gguf, args.max_context, realize=True)
  validate_vision_metadata(kv)
  device = model.token_embd.weight.device
  if not isinstance(device, str) or device.split(":")[0] in ("NULL", "DISK", "NPY", "PYTHON"):
    raise ValueError("real checkpoint evaluation requires a single executing decoder device")
  vision = load_vision(args.model_dir, device=device)
  Device[device].synchronize()
  load_s = time.perf_counter()-start
  loaded_memory = memory_snapshot()
  # Snapshot before JIT/cache tensors become part of the model's object graph.
  weight_dtypes = {name:sorted({str(t.dtype) for t in nn.state.get_parameters(module)}) for name,module in (("decoder",model),("vision",vision))}
  start = time.perf_counter()
  tokenizer = SimpleTokenizer.from_gguf_kv(kv)
  env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
  env.filters["tojson"] = lambda obj, **kwargs: json.dumps(obj, **kwargs)
  env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
  env.globals["strftime_now"] = lambda fmt: time.strftime(fmt)
  env.globals["bos_token"] = tokenizer.decode([tokenizer.bos_id]) if tokenizer.bos_id is not None else ""
  env.globals["eos_token"] = tokenizer.decode([tokenizer.eos_id])
  template = env.from_string(verified["chat_template.jinja"].decode())
  tokenizer_template_s = time.perf_counter()-start
  limits = ImageLimits()
  images, cases = controlled_suite()
  suite = suite_manifest(images, cases)
  suite_hash = digest(json.dumps(suite, sort_keys=True).encode())
  urls = {name:image_url(image) for name,image in images.items()}
  # The input manifest and all answers are frozen before generation. No adaptive prompts or retries.
  Tensor.manual_seed(SEED)
  results = []
  for index, case in enumerate(cases):
    row = timed_request(model, vision, tokenizer, template, limits, case, urls, args)
    row["run_state"] = "process_first_request" if index == 0 else "suite_first_pass_shared_compiler_and_jit_state"
    results.append(row)
  quality = quality_summary(cases, results)
  repeats = []
  if args.phase == "benchmark":
    # Four bounded workloads: text, default-area image, maximum-area image, two images/order.
    for name in ("text_math", "color_red", "color_yellow", "order_red_blue"):
      case = next(c for c in cases if c["id"] == name)
      for repeat in range(args.warm_runs):
        row = timed_request(model, vision, tokenizer, template, limits, case, urls, args)
        row.update({"run_state":"warm_repeat_no_prefix_reuse", "repeat":repeat+1})
        repeats.append(row)
  warm_passed = all(r["answer_match"] and r["stop_reason"] == "eos" for r in repeats)
  renderer = Device[device].renderer
  return {"phase":args.phase, "synthetic":False, "checkpoint_weights_loaded":True, "versions":installed_versions(),
          "checkpoint":checkpoint, "harness_sha256":digest(Path(__file__).read_bytes()),
          "production_source_sha256":{name:digest((Path(__file__).resolve().parents[2]/"tinygrad/llm"/name).read_bytes())
                                      for name in ("cli.py", "model.py", "vision.py", "multimodal.py", "gguf.py", "kernels/amd.py")},
          "hardware":{"platform":platform.platform(), "machine":platform.machine(), "processor":platform.processor(),
                      "logical_cpus":os.cpu_count(), "device":device, "arch":str(Device[device].arch),
                      "renderer":type(renderer).__name__, "renderer_target":str(renderer.target), "DEV":str(DEV)},
          "backend":{"runtime":type(Device[device]).__name__, "temperature":0.0, "seed":SEED,
                     "amd_custom_kernels_supported":amd_custom_kernels_supported(device), "HALF":getenv("HALF", 1),
                     "realize_weights_before_requests":True, "prefix_cache_reuse":False,
                     "note":"production dispatch unchanged; quantized/backend arithmetic is not an FP32 parity claim"},
          "dtype":{"decoder_parameter_types":weight_dtypes["decoder"], "vision_parameter_types":weight_dtypes["vision"],
                   "pixels_and_fused_embeddings":"float32", "kv_cache":"float16", "note":"mixed native arithmetic; not a single-dtype model"},
          "context":{"max_context":model.max_context, "max_output_tokens":args.max_output_tokens, "requested_chunk_size":args.chunk_size,
                     "multimodal_chunk_size":min(args.chunk_size,32), "text_chunk_size":"production CPU GDN may force 1"},
          "image_limits":vars(limits), "suite":suite, "suite_sha256":suite_hash, "quality":quality,
          "timing":{"bundle_validation_s":validation_s, "model_load_and_realize_s":load_s, "tokenizer_template_s":tokenizer_template_s},
          "timing_scope":{"preprocess_s":"decode/resize/normalize/patch, official template, tokenize and device transfer",
                          "vision_s":"synchronized actual QwenVision call within production embed_prompt",
                          "embed_fusion_s":"embedding and splice excluding timed vision call",
                          "prefill_s":"first generate.next: prefill INCLUDING validation, sample and token copyout (not isolated forward)",
                          "first_token_s":"request wall time from preparation start to first generated token, not first displayed text",
                          "decode_s":"sum of synchronized later next calls, including EOS; excludes first token",
                          "total_s":"preparation through final generated token including Python orchestration; excludes model loading",
                          "cold":"first request in this process; disk/compiler caches NOT purged; model already realized",
                          "warm":"bounded repeats after suite; retains compiled kernels/JIT but resets request state/prefix cache"},
          "results":results, "warm_results":repeats, "warm_runs_per_workload":args.warm_runs if args.phase == "benchmark" else 0,
          "memory_after_load":loaded_memory, "memory":memory_snapshot(), "quality_gate_passed":quality["passed"] and warm_passed,
          "release_acceptance":False, "remaining_gates":["component/reference regression evidence on this backend",
            "actual device peak memory via external profiler and provisioned-hardware capacity review",
            "matching-weight generic/AMD arithmetic drift and hybrid tail counts 1/15/16/17/31/32/33 if advertising AMD",
            "CLI/API full-checkpoint hardware acceptance; this harness tests the shared production pipeline only"]}

def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  mode = parser.add_mutually_exclusive_group(required=True)
  mode.add_argument("--generate", action="store_true")
  mode.add_argument("--verify", action="store_true")
  mode.add_argument("--check-reproducible", action="store_true")
  mode.add_argument("--phase", choices=("text", "vision", "fusion", "processor-compare", "e2e", "benchmark"))
  parser.add_argument("--synthetic", action="store_true", help="Required for synthetic text/vision/fusion phases; no checkpoint execution")
  parser.add_argument("--output-dir", type=Path, default=ROOT)
  parser.add_argument("--metadata-dir", type=Path, help="Bootstrap only: directory containing the two pinned metadata JSON files")
  parser.add_argument("--model-dir", type=Path, help="Existing local trusted production vision bundle (never downloaded)")
  parser.add_argument("--gguf", type=Path, help="Existing local trusted Qwen3.8 GGUF (never downloaded)")
  parser.add_argument("--image-suite", choices=("controlled",), default="controlled")
  parser.add_argument("--max-context", type=int, default=4096)
  parser.add_argument("--max-output-tokens", type=int, default=32, help="Includes EOS; hard maximum 256")
  parser.add_argument("--chunk-size", type=int, default=32, help="Requested production prefill size, in [1,32]")
  parser.add_argument("--warm-runs", type=int, default=2, help="Benchmark repeats per workload, in [1,3]")
  parser.add_argument("--report", type=Path, help="Write JSON for processor-compare/e2e/benchmark, also emitted on stdout")
  args = parser.parse_args(argv)
  if args.metadata_dir is not None and not args.generate: parser.error("--metadata-dir requires --generate")
  if (args.phase in ("text", "vision", "fusion")) != args.synthetic:
    parser.error("only text/vision/fusion phases require --synthetic (and vice versa)")
  real = args.phase in ("e2e", "benchmark")
  if real:
    if args.model_dir is None or args.gguf is None: parser.error("e2e/benchmark require --model-dir and --gguf local paths; no downloads")
    if not args.model_dir.is_dir() or not args.gguf.is_file(): parser.error("--model-dir and --gguf must exist locally; no downloads")
  elif args.model_dir is not None or args.gguf is not None: parser.error("checkpoint paths require e2e/benchmark")
  if not 1 <= args.warm_runs <= 3: parser.error("--warm-runs must be in [1,3]")
  if not 1 <= args.chunk_size <= 32: parser.error("--chunk-size must be in [1,32]")
  if not 1 <= args.max_output_tokens <= 256: parser.error("--max-output-tokens must be in [1,256]")
  if not args.max_output_tokens < args.max_context <= 4096: parser.error("output budget must be less than context, capped at 4096")
  if args.report is not None and args.phase not in ("processor-compare", "e2e", "benchmark"):
    parser.error("--report requires processor-compare/e2e/benchmark")
  return args

def main():
  import sys
  args = parse_args()
  if args.phase in ("processor-compare", "e2e", "benchmark"):
    # JSON-only stdout; production diagnostics and verification go to stderr.
    with contextlib.redirect_stdout(sys.stderr):
      if args.phase == "processor-compare":
        verify(args.output_dir)
        result = processor_compare(args.output_dir)
      else: result = checkpoint_phase(args)
    raw = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n"
    if args.report is not None:
      args.report.parent.mkdir(parents=True, exist_ok=True)
      args.report.write_text(raw)
    print(raw, end="")
    if result.get("quality_gate_passed") is False: raise SystemExit(1)
    return
  if args.phase:
    verify(args.output_dir)
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    with np.load(args.output_dir / "reference.npz", allow_pickle=False) as src: arrays = {k: src[k] for k in src.files}
    if args.phase in ("vision", "fusion"):
      print(json.dumps((vision_phase if args.phase == "vision" else fusion_phase)(manifest, arrays), indent=2))
      return
    from tinygrad import Device
    from test.unit.test_llm_multimodal import compare_native_text
    metrics = compare_native_text(manifest, arrays)
    tier = manifest["parity_tiers"]["native_cast"]["lowered_fp32_weights"]
    print(json.dumps({"phase": "text", "synthetic": True, "device": Device.DEFAULT, "tier": "native_cast.lowered_fp32_weights",
                      "scope": tier["scope"], "tolerance": tier["tolerance"], "relative_denominator_floor": 1e-12,
                      "metrics": metrics}, indent=2))
  elif args.generate: generate(args.output_dir, args.metadata_dir)
  else:
    verify(args.output_dir)
    if args.check_reproducible:
      with tempfile.TemporaryDirectory() as tmp:
        generate(Path(tmp), archive_dir=args.output_dir)
        result = compare_reproduction(args.output_dir, Path(tmp))
      print("Regeneration matches numerical array hashes, byte-for-byte archive and manifest identity excluding execution provenance.")
      print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__": main()

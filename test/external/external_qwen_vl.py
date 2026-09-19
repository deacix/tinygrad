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
  python test/external/external_qwen_vl.py --phase fusion --synthetic
Fusion phase uses the pinned reference environment plus tinygrad, reusing existing
arrays for live complete synthetic vision/fusion/hybrid prefill and decode parity.
Neither phase establishes full checkpoint, quantized or hardware acceptance.
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
                  "reproducibility": "--check-reproducible regenerates and compares BOTH files byte-for-byte"},
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


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  mode = parser.add_mutually_exclusive_group(required=True)
  mode.add_argument("--generate", action="store_true")
  mode.add_argument("--verify", action="store_true")
  mode.add_argument("--check-reproducible", action="store_true")
  mode.add_argument("--phase", choices=("text", "fusion"))
  parser.add_argument("--synthetic", action="store_true", help="Required for the synthetic text/fusion phases; no checkpoint execution")
  parser.add_argument("--output-dir", type=Path, default=ROOT)
  parser.add_argument("--metadata-dir", type=Path, help="Bootstrap only: directory containing the two pinned metadata JSON files")
  args = parser.parse_args()
  if args.metadata_dir is not None and not args.generate: parser.error("--metadata-dir requires --generate")
  if bool(args.phase) != args.synthetic: parser.error("--phase requires --synthetic (and vice versa)")
  if args.phase:
    verify(args.output_dir)
    from tinygrad import Device
    from test.unit.test_llm_multimodal import compare_native_text
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    with np.load(args.output_dir / "reference.npz", allow_pickle=False) as src: arrays = {k: src[k] for k in src.files}
    if args.phase == "fusion":
      print(json.dumps(fusion_phase(manifest, arrays), indent=2))
      return
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
        for name in ("reference.npz", "manifest.json"):
          if (Path(tmp) / name).read_bytes() != (args.output_dir / name).read_bytes(): raise ValueError(f"regeneration differs: {name}")
      print("Regeneration is byte-for-byte reproducible (archive and manifest).")

if __name__ == "__main__": main()

import unittest
import numpy as np
from tinygrad import Tensor, nn
from test.null.test_llm_multimodal import load_fixture

class TestQwenVision(unittest.TestCase):
  def make_model(self):
    from tinygrad.llm.vision import VisionConfig, QwenVision
    manifest, arrays = load_fixture()
    model = QwenVision(VisionConfig.from_dict(manifest["configs"]["vision"]))
    weights = {k.removeprefix("vision.weights."):Tensor(v) for k,v in arrays.items() if k.startswith("vision.weights.")}
    self.assertEqual(set(weights), set(nn.state.get_state_dict(model)))
    nn.state.load_state_dict(model, weights, verbose=False)
    return model, manifest, arrays

  def test_vision_reference(self):
    model, manifest, a = self.make_model()
    tolerance = manifest["parity_tiers"]["hf_fp32_algebra"]["tolerance"]
    grids = tuple(tuple(int(x) for x in g) for g in a["vision.grid_thw"])
    pixels = Tensor(a["vision.pixel_values"])
    patches = model.patch_embed(pixels)
    np.testing.assert_allclose(patches.numpy(), a["vision.patch_embed"], **tolerance)
    pos, rotary = model.positions(grids)
    np.testing.assert_allclose(pos.numpy(), a["vision.positions"], **tolerance)
    np.testing.assert_allclose(rotary.numpy(), a["vision.rotary_freqs"], **tolerance)
    x = patches + pos
    for i, block in enumerate(model.blocks):
      x = block(x, grids, rotary).realize()
      np.testing.assert_allclose(x.numpy(), a[f"vision.block.{i}"], **tolerance)
    np.testing.assert_allclose(model.merger(x).numpy(), a["vision.merger"], **tolerance)
    np.testing.assert_allclose(model(pixels, grids).numpy(), a["vision.merger"], **tolerance)

  def test_vision_images_are_independent(self):
    model, manifest, a = self.make_model()
    grids = tuple(tuple(int(x) for x in g) for g in a["vision.grid_thw"])
    pixels = Tensor(a["vision.pixel_values"])
    together = model(pixels, grids).numpy()
    separate = np.concatenate([model(pixels[:24], grids[:1]).numpy(), model(pixels[24:], grids[1:]).numpy()])
    np.testing.assert_allclose(together, separate, **manifest["parity_tiers"]["hf_fp32_algebra"]["tolerance"])

  def test_vision_bfloat16_layernorm(self):
    import torch
    from tinygrad import dtypes
    from tinygrad.llm.vision import VisionLayerNorm
    x = np.array([[256.,258.,256.,258.]], dtype=np.float32)
    norm = VisionLayerNorm(4, eps=1e-6)
    norm.weight = Tensor.ones(4).cast(dtypes.bfloat16).realize()
    norm.bias = Tensor.zeros(4).cast(dtypes.bfloat16).realize()
    actual = norm(Tensor(x).cast(dtypes.bfloat16).realize()).float().numpy()
    expected = torch.nn.functional.layer_norm(torch.tensor(x).bfloat16(), (4,), eps=1e-6).float().numpy()
    np.testing.assert_array_equal(actual, expected)

  def test_vision_bfloat16_full_tower_live_reference(self):
    import importlib.metadata
    from test.external.external_qwen_vl import VERSIONS, vision_phase
    # Reference dependencies are optional for ordinary unit runs, and never imported during collection.
    for name, version in VERSIONS.items():
      try: installed = importlib.metadata.version(name)
      except importlib.metadata.PackageNotFoundError: self.skipTest(f'live vision parity requires {name}=={version}')
      if installed != version: self.skipTest(f'live vision parity requires {name}=={version}, found {installed}')
    manifest, arrays = load_fixture()
    result = vision_phase(manifest, arrays)
    self.assertEqual(result['phase'], 'vision')
    self.assertEqual(result['dtype'], 'bfloat16')
    self.assertEqual(result['patch_lengths'], [24,16])
    self.assertEqual(result['tolerance'], {'rtol':2e-2, 'atol':2e-2})
    self.assertEqual(set(result['metrics']), {'positions', 'rotary', 'patch_embed', 'block.0', 'block.1', 'merger_norm',
                                             'merger', 'full_tower', 'hf_packed_vs_separate', 'native_packed_vs_separate'})
    self.assertEqual(result['max_abs'], max(v['max_abs'] for v in result['metrics'].values()))
    print(f"BF16 vision full-tower max_abs={result['metrics']['full_tower']['max_abs']}; all components max_abs={result['max_abs']}")

  def test_vision_rejects_bad_grids(self):
    model, _, a = self.make_model()
    for grids in ((), ((2, 4, 6),), ((1, 3, 16),), ((1, 0, 48),), ((1, 4, 6),)):
      with self.subTest(grids=grids), self.assertRaises(ValueError): model(Tensor(a["vision.pixel_values"]), grids)

class TestQwenVisionLoader(unittest.TestCase):
  def test_loader_digests_and_paths(self):
    import hashlib, tempfile
    from pathlib import Path
    from tinygrad.llm.vision import _verify_file, _bundle_paths, BUNDLE_FILES
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      data = b"verified fixture bytes"
      p = root/"input"
      p.write_bytes(data)
      _verify_file(p, (len(data), hashlib.sha256(data).hexdigest()))
      _verify_file(p, (len(data), hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()))
      for spec in ((len(data)+1, "0"*64), (len(data), "0"*64)):
        with self.assertRaises(ValueError): _verify_file(p, spec)
      bundle = root/"bundle"
      bundle.mkdir()
      for name in BUNDLE_FILES: (bundle/name).write_bytes(b"test")
      (bundle/"config.json").unlink()
      (bundle/"config.json").symlink_to(p)
      with self.assertRaisesRegex(ValueError, "symlink"): _bundle_paths(bundle)

  def test_loader_index_strict(self):
    from tinygrad.llm.vision import _visual_keys, _validate_visual_index, VISION_SHARD
    keys = _visual_keys()
    self.assertEqual(len(keys), 333)
    index = {"weight_map":{f"model.visual.{k}":VISION_SHARD for k in keys}}
    _validate_visual_index(index)
    key = next(iter(index["weight_map"]))
    for change in ("missing", "extra", "path"):
      mapping = index["weight_map"].copy()
      if change == "missing": del mapping[key]
      elif change == "extra": mapping["model.visual.unexpected"] = VISION_SHARD
      else: mapping[key] = "../escape.safetensors"
      with self.subTest(change=change), self.assertRaises(ValueError): _validate_visual_index({"weight_map":mapping})

  def test_loader_validates_before_transfer(self):
    from tinygrad import dtypes
    from tinygrad.llm.vision import QwenVision, VisionConfig, _load_visual_tensors
    from unittest.mock import patch
    cfg = VisionConfig(depth=1, hidden_size=8, intermediate_size=12, num_heads=1, out_hidden_size=8, num_position_embeddings=4)
    model = QwenVision(cfg)
    weights = {k:Tensor.zeros(*v.shape, dtype=dtypes.bfloat16) for k,v in nn.state.get_state_dict(model).items()}
    key = next(iter(weights))
    for bad in ({k:v for k,v in weights.items() if k != key}, weights | {"unused":Tensor([0])}, weights | {key:Tensor([0])}):
      with patch.object(Tensor, "to", side_effect=AssertionError("transferred invalid weights")), self.assertRaises(ValueError):
        _load_visual_tensors(model, bad, "CPU")
    _load_visual_tensors(model, weights, "CPU:1")
    self.assertTrue(all(v.device == "CPU:1" and v.dtype == dtypes.bfloat16 for v in nn.state.get_parameters(model)))

  def test_loader_checks_all_files_before_parsing(self):
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    from tinygrad.llm.vision import BUNDLE_FILES, validate_vision_bundle
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      for name in (*BUNDLE_FILES, "text.gguf"): (root/name).write_bytes(b"untrusted")
      with patch("tinygrad.llm.vision.json.loads", side_effect=AssertionError("parsed before verification")):
        with self.assertRaises(ValueError): validate_vision_bundle(root, root/"text.gguf")
      with patch("tinygrad.llm.vision._verify_file") as verify, patch("tinygrad.llm.vision.json.loads", side_effect=ValueError("invalid JSON")):
        with self.assertRaisesRegex(ValueError, "invalid JSON"): validate_vision_bundle(root, root/"text.gguf")
        self.assertEqual(verify.call_count, len(BUNDLE_FILES)+1)

  def test_loader_config_and_pairing(self):
    from tinygrad.llm.vision import VisionConfig, validate_vision_metadata
    for cfg in ({"patch_size":14}, {"hidden_size":13}, {"num_position_embeddings":3}, {"deepstack_visual_indexes":[1]}, {"depth":True}):
      with self.subTest(cfg=cfg), self.assertRaises(ValueError): VisionConfig.from_dict(cfg)
    with self.assertRaises(ValueError): validate_vision_metadata({"general.architecture":"llama"})

if __name__ == "__main__": unittest.main()

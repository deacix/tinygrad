"""CPU fixture sanity only; native vision/text parity is added in later phases."""
import unittest
import numpy as np
from test.null.test_llm_multimodal import load_fixture

class TestQwenFixtureSanity(unittest.TestCase):
  def test_fixture_vision_patch_projection(self):
    m, a = load_fixture()
    weight, bias = (a["vision.weights.patch_embed.proj." + k] for k in ("weight", "bias"))
    expected = a["vision.pixel_values"] @ weight.reshape(weight.shape[0], -1).T + bias
    np.testing.assert_allclose(expected, a["vision.patch_embed"], **m["parity_tiers"]["hf_fp32_algebra"]["tolerance"])
    self.assertEqual(a["vision.grid_thw"].tolist(), [[1, 4, 6], [1, 6, 4]])
    self.assertEqual(a["vision.merger"].shape, (12, m["configs"]["vision"]["out_hidden_size"]))
    self.assertGreater(np.abs(a["vision.block.1"] - a["vision.block.0"]).max(), 1e-3)
    self.assertGreater(np.abs(a["vision.merger"][:6] - a["vision.merger"][6:]).max(), 1e-3)
    np.testing.assert_allclose(a["vision.separate_merger"], a["vision.merger"],
                               **m["parity_tiers"]["hf_fp32_algebra"]["tolerance"])

  def test_fixture_hybrid_text_and_cache(self):
    m, a = load_fixture()
    cfg = m["configs"]["text"]
    self.assertEqual(cfg["layer_types"], ["linear_attention", "full_attention"])
    self.assertEqual(cfg["linear_num_value_heads"], cfg["linear_num_key_heads"] * 3)
    self.assertNotEqual(cfg["linear_key_head_dim"], cfg["linear_value_head_dim"])
    ids = a["text.input_ids"]
    np.testing.assert_array_equal(a["text.inputs_embeds"], a["text.weights.model.embed_tokens.weight"][ids])
    logits = a["text.full.logits"]
    self.assertEqual(logits.shape, (1, ids.shape[1], cfg["vocab_size"]))
    self.assertGreater(np.ptp(logits), 0.1)
    np.testing.assert_allclose(a["text.full.hidden"] @ a["text.weights.lm_head.weight"].T, logits,
                               **m["parity_tiers"]["hf_fp32_algebra"]["tolerance"])
    for mode in ("tokenwise", "chunked", "prefill_decode", "embeddings"):
      np.testing.assert_allclose(a[f"text.{mode}.logits"], logits, **m["parity_tiers"]["hf_fp32_algebra"]["tolerance"])
    for mode in ("full", "tokenwise", "chunked", "prefill_decode"):
      state = a[f"text.{mode}.recurrent_state"]
      self.assertEqual(state.shape, (1, cfg["linear_num_value_heads"], cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]))
      self.assertGreater(np.abs(state[:, 0] - state[:, 3]).max(), 1e-6)
      for name in ("block.0", "block.1", "recurrent_state", "conv_state", "keys", "values"):
        np.testing.assert_allclose(a[f"text.{mode}.{name}"], a[f"text.full.{name}"],
                                   **m["parity_tiers"]["hf_fp32_algebra"]["tolerance"])
    # Final HF conv cache includes the newest token; tinygrad's K-1 history drops the oldest entry.
    projected = a["text.full.gdn_qkv"]
    np.testing.assert_allclose(a["text.full.conv_state"], projected[:, -cfg["linear_conv_kernel_dim"]:].transpose(0, 2, 1),
                               **m["parity_tiers"]["hf_fp32_algebra"]["tolerance"])

  def test_fixture_state_snapshots(self):
    m, a = load_fixture()
    for mode in ("tokenwise", "chunked", "prefill_decode"):
      lengths = a[f"text.{mode}.chunk_lengths"]
      self.assertEqual(lengths.sum(), 10)
      for step, consumed in enumerate(np.cumsum(lengths)):
        prefix = f"text.{mode}.step.{step}."
        self.assertEqual(a[prefix + "keys"].shape[2], consumed)
        np.testing.assert_array_equal(a[prefix + "conv_state"], a[f"text.{mode}.conv_history"][step])
        np.testing.assert_array_equal(a[prefix + "recurrent_state"], a[f"text.{mode}.recurrent_history"][step])
        for name in ("conv_state", "recurrent_state", "keys", "values"):
          np.testing.assert_allclose(a[prefix + name], a[f"text.tokenwise.step.{consumed-1}." + name],
                                     **m["parity_tiers"]["hf_fp32_algebra"]["tolerance"])

  def test_fixture_vision_positions_and_rotary(self):
    m, a = load_fixture()
    # Endpoint-aligned learned positions, in block-major order; exercise a noninteger interior coordinate too.
    table = a["vision.weights.pos_embed.weight"].reshape(5, 5, 16)
    np.testing.assert_array_equal(a["vision.positions"][0], table[0, 0])
    np.testing.assert_array_equal(a["vision.positions"][23], table[4, 4])
    np.testing.assert_allclose(a["vision.positions"][1], table[0, 0]*0.2 + table[0, 1]*0.8, atol=1e-7)
    for i in range(2):
      qkv = a[f"vision.block.{i}.qkv"].reshape(48, 3, 2, 8)
      cos, sin = a["vision.rotary_cos"][:, None], a["vision.rotary_sin"][:, None]
      for j, name in ((0, "q"), (1, "k")):
        v = qkv[:, j]
        rotated = np.concatenate((-v[..., 4:], v[..., :4]), axis=-1)
        np.testing.assert_allclose(v*cos + rotated*sin, a[f"vision.block.{i}.rotary_{name}"],
                                   **m["parity_tiers"]["hf_fp32_algebra"]["tolerance"])

  def test_fixture_target_mrope_partial_rotation(self):
    _, a = load_fixture()
    for name in ("q", "k"):
      v = a["coordinates.mrope." + ("query" if name == "q" else "key")]
      out = a["coordinates.mrope.rotary_" + name]
      np.testing.assert_array_equal(out[..., 64:], v[..., 64:])
      self.assertGreater(np.abs(out[..., :64] - v[..., :64]).max(), 0.1)
    # First three frequencies use T,H,W respectively, not contiguous frequency chunks.
    pos = a["coordinates.two_images.position_ids"]
    cos = a["coordinates.mrope.cos"]
    for frequency, axis in ((0, 0), (1, 1), (2, 2), (3, 0), (31, 1)):
      inv = np.float32(1 / (10000000 ** (frequency / 32)))
      np.testing.assert_allclose(cos[..., frequency], np.cos(pos[axis].astype(np.float32) * inv), atol=1e-6, rtol=1e-6)

  def test_fixture_low_norm_and_distinct_heads(self):
    _, a = load_fixture()
    x = a["gdn.low_norm.input"]
    np.testing.assert_allclose(a["gdn.low_norm.output"], x / np.sqrt(np.sum(x*x, axis=-1, keepdims=True) + 1e-6), rtol=1e-6)
    clamp_after_sqrt = x / np.maximum(np.sqrt(np.sum(x*x, axis=-1, keepdims=True)), 1e-6)
    self.assertGreater(np.abs(a["gdn.low_norm.output"] - clamp_after_sqrt).max(), 0.1)
    q = a["text.full.gdn_query"]
    self.assertEqual(q.shape[2], 6)
    np.testing.assert_array_equal(q[:, :, 0], q[:, :, 2])
    np.testing.assert_array_equal(q[:, :, 3], q[:, :, 5])
    self.assertGreater(np.abs(q[:, :, 0] - q[:, :, 3]).max(), 1e-4)

if __name__ == "__main__": unittest.main()

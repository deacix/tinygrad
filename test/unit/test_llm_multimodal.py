"""Offline Qwen fixtures and native hybrid text parity. No Transformers/Torch dependency."""
import unittest
from typing import Any
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig
from test.null.test_llm_multimodal import load_fixture


def native_text_model(manifest, arrays):
  """Explicit official synthetic -> GGUF-layout mapping, not a general HF/checkpoint importer."""
  c = manifest["configs"]["text"]
  nk, nv, dk, dv = (c[k] for k in ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
  # HF repeats each Q/K head; native tiles the Q/K heads. All V-associated weights must change order together.
  heads = np.arange(nv).reshape(nk, nv//nk).T.reshape(-1)
  values = (heads[:, None]*dv + np.arange(dv)).reshape(-1)
  channels = np.concatenate((np.arange(2*nk*dk), 2*nk*dk + values))
  rope = c["rope_parameters"]
  model = Transformer(TransformerConfig(num_blocks=c["num_hidden_layers"], dim=c["hidden_size"], hidden_dim=c["intermediate_size"],
    n_heads=c["num_attention_heads"], n_kv_heads=c["num_key_value_heads"], norm_eps=c["rms_norm_eps"], vocab_size=c["vocab_size"],
    head_dim=c["head_dim"], v_head_dim=c["head_dim"], rope_theta=rope["rope_theta"],
    rope_dim=int(c["head_dim"]*rope["partial_rotary_factor"]), max_context=32, attn_output_gate=True,
    ssm_layers=tuple(t == "linear_attention" for t in c["layer_types"]),
    ssm=SSMConfig(c["linear_conv_kernel_dim"], dk, nk, nv, nv*dv)))
  weights, consumed = {}, set()
  def add(dst, src, transform=lambda x: x):
    key = "text.weights." + src
    consumed.add(key)
    weights[dst] = Tensor(np.ascontiguousarray(transform(arrays[key])))
  add("token_embd.weight", "model.embed_tokens.weight")
  add("output.weight", "lm_head.weight")
  add("output_norm.weight", "model.norm.weight", lambda x: x+1)
  for i, kind in enumerate(c["layer_types"]):
    dst, src = f"blk.{i}.", f"model.layers.{i}."
    for n, h in (("attn_norm", "input_layernorm"), ("ffn_norm", "post_attention_layernorm")):
      add(dst+n+".weight", src+h+".weight", lambda x: x+1)
    for n in ("gate", "up", "down"): add(dst+f"ffn_{n}.weight", src+f"mlp.{n}_proj.weight")
    if kind == "linear_attention":
      src += "linear_attn."
      add(dst+"attn_qkv.weight", src+"in_proj_qkv.weight", lambda x: x[channels])
      add(dst+"attn_gate.weight", src+"in_proj_z.weight", lambda x: x[values])
      add(dst+"ssm_alpha.weight", src+"in_proj_a.weight", lambda x: x[heads])
      add(dst+"ssm_beta.weight", src+"in_proj_b.weight", lambda x: x[heads])
      add(dst+"ssm_dt.bias", src+"dt_bias", lambda x: x[heads])
      add(dst+"ssm_a", src+"A_log", lambda x: -np.exp(x[heads]))
      add(dst+"ssm_conv1d.weight", src+"conv1d.weight", lambda x: x[channels, 0])
      add(dst+"ssm_out.weight", src+"out_proj.weight", lambda x: x[:, values])
      add(dst+"ssm_norm.weight", src+"norm.weight")  # Gated RMSNorm is already multiplicative: no +1.
    else:
      src += "self_attn."
      for n in ("q", "k", "v"): add(dst+f"attn_{n}.weight", src+f"{n}_proj.weight")
      # Q and sigmoid gate are interleaved WITHIN each full-attention head in both layouts.
      add(dst+"attn_output.weight", src+"o_proj.weight")
      for n in ("q", "k"): add(dst+f"attn_{n}_norm.weight", src+f"{n}_norm.weight", lambda x: x+1)
  assert consumed == {k for k in arrays if k.startswith("text.weights.")}, "unused official text weights"
  assert weights.keys() == nn.state.get_state_dict(model).keys(), "incomplete native text mapping"
  nn.state.load_state_dict(model, weights, verbose=False)
  return model, heads, channels


def native_text_run(model, arrays, lengths):
  """Eager numeric path; exercises real blocks and caches, not generate/JIT lifecycle."""
  start = 0
  result:dict[str, Any] = {f"block.{i}": [] for i in range(len(model.blk))}
  result["logits"], result["snapshots"] = [], []
  for length in lengths:
    end = start + length
    x = model.token_embd(Tensor(arrays["text.input_ids"][:, start:end])).float()
    blocks = []
    for block in model.blk:
      x = block(x, start)
      blocks.append(x)
    logits = model.output(model.output_norm(x))
    # One copyout for the whole graph: reading intermediate function results separately would replay cache writes.
    outputs = np.split(logits.cat(*blocks, dim=-1).numpy(), np.cumsum([logits.shape[-1]] + [b.shape[-1] for b in blocks[:-1]]), axis=-1)
    for i, output in enumerate(outputs[1:]): result[f"block.{i}"].append(output)
    result["logits"].append(outputs[0])
    result["snapshots"].append({"conv_state": model.blk[0].conv_state.numpy(), "recurrent_state": model.blk[0].recurrent_state.numpy(),
                               "keys": model.blk[1].cache_kv[0, :, :, :end].float().numpy(),
                               "values": model.blk[1].cache_kv[1, :, :, :end].float().numpy()})
    start = end
  return {k: v if k == "snapshots" else np.concatenate(v, axis=1) for k, v in result.items()}


def compare_native_text(manifest, arrays):
  """Shared assertions/metrics for the offline unit gate and --phase text --synthetic."""
  tier = manifest["parity_tiers"]["native_cast"]["lowered_fp32_weights"]
  prefix, tolerance = tier["prefix"] + ".full", tier["tolerance"]
  # Declare, don't infer/flex tolerances around, the compiler semantics this oracle covers.
  probe = np.array([1.0004, -0.9872], dtype=np.float32)
  np.testing.assert_array_equal(Tensor(probe).half().float().numpy(), probe,
                                err_msg="FP32->FP16->FP32 lowering changed: re-audit the native cast tier")
  np.testing.assert_array_equal(Tensor(probe).half().contiguous().realize().float().numpy(), probe.astype(np.float16).astype(np.float32))
  model, heads, channels = native_text_model(manifest, arrays)
  metrics = {}
  def compare(name, actual, expected):
    assert np.isfinite(actual).all(), name
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    relative = error / np.maximum(np.abs(expected), 1e-12)
    metrics[name] = {"max_abs": float(error.max()), "mean_abs": float(error.mean()),
                     "max_relative": float(relative.max()), "mean_relative": float(relative.mean())}
    np.testing.assert_allclose(actual, expected, **tolerance, err_msg=name)
  for mode, lengths in (("full", [10]), ("tokenwise", [1]*10), ("chunked", [3, 2, 5]), ("prefill_decode", [6, 1, 1, 1, 1])):
    result = native_text_run(model, arrays, lengths)
    for component in ("logits", "block.0", "block.1"):
      compare(f"{mode}.{component}", result[component], arrays[f"{prefix}.{component}"])
    # Independently exercise the public final-position seam on the real weighted hybrid and compare each chunk endpoint.
    start = 0
    for length in lengths:
      end = start + length
      logits = model.forward_embeddings(Tensor(arrays["text.inputs_embeds"][:, start:end]), start).numpy()
      compare(f"{mode}.embeddings.{end}", logits, arrays[f"{prefix}.logits"][:, end-1])
      np.testing.assert_array_equal(logits, result["logits"][:, end-1])
      start = end
    for step, consumed in enumerate(np.cumsum(lengths)):
      ref = prefix + "." if mode == "full" else prefix.rsplit(".", 1)[0] + f".tokenwise.step.{consumed-1}."
      for name, actual in result["snapshots"][step].items():
        expected = arrays[ref + name]
        if name == "recurrent_state": expected = expected[:, heads].swapaxes(-1, -2)
        if name == "conv_state": expected = expected[:, channels, 1:].transpose(0, 2, 1)
        if name in ("keys", "values"):
          # FP32 reduction orders can straddle one FP16 midpoint; use the chunk-replay single-storage-ULP contract.
          error = np.abs(actual.astype(np.float32)-expected.astype(np.float32))
          ulp = np.maximum(np.abs(np.spacing(actual.astype(np.float16))), np.abs(np.spacing(expected.astype(np.float16))))
          np.testing.assert_array_equal(error <= ulp, True, err_msg=f"{mode}.step.{step}.{name}: exceeds one FP16 storage ULP")
          metrics[f"{mode}.step.{step}.{name}"] = {"max_abs":float(error.max()), "max_storage_ulps":float((error/ulp).max())}
        else: compare(f"{mode}.step.{step}.{name}", actual, expected)
  return metrics


class TestQwenNativeText(unittest.TestCase):
  def test_native_text_parity(self):
    m, a = load_fixture()
    self.assertEqual(m["parity_tiers"]["native_cast"]["status"], "captured")
    compare_native_text(m, a)

  def test_native_text_mapping_and_sampling(self):
    m, a = load_fixture()
    model, heads, channels = native_text_model(m, a)
    np.testing.assert_array_equal(heads, [0, 3, 1, 4, 2, 5])
    self.assertEqual(len(np.unique(channels)), 52)
    gdn, attention = model.blk
    # Direct projection checks independently constrain the tiled V and per-head Q/gate layouts.
    x = Tensor(a["text.inputs_embeds"])
    np.testing.assert_allclose(gdn.attn_qkv(gdn.attn_norm(x)).numpy(), a["text.full.gdn_qkv"][..., channels], rtol=1e-4, atol=1e-5)
    x = Tensor(a["text.full.block.0"])
    np.testing.assert_allclose(attention.attn_q(attention.attn_norm(x)).numpy(), a["text.full.attn_q_gate"], rtol=1e-4, atol=1e-5)
    tokens = Tensor(a["text.input_ids"])
    logits = model.forward_embeddings(model.token_embd(tokens).float(), 0).numpy()
    sampled = model.forward(tokens, 0, Tensor([0.0])).numpy()
    np.testing.assert_array_equal(sampled, logits.argmax(-1, keepdims=True))
    np.testing.assert_array_equal(sampled[:, 0], a["text.full.logits"][:, -1].argmax(-1))

  def test_fixture_native_cast_boundaries(self):
    m, a = load_fixture()
    tier = m["parity_tiers"]["native_cast"]
    self.assertEqual(tier["tolerance"], {"rtol": 1e-4, "atol": 1e-5})
    # Removing a fused input cast is observably different from the nominal explicit half-rounding tier.
    self.assertGreater(np.abs(a["text.native_cast.full.logits"] - a["text.native_lowered.full.logits"]).max(), 1e-4)
    for prefix in ("text.native_cast", "text.native_lowered"):
      self.assertEqual(a[prefix+".full.keys"].dtype, np.float16)
      self.assertEqual(a[prefix+".full.values"].dtype, np.float16)
      self.assertEqual(a[prefix+".full.recurrent_state"].dtype, np.float32)
      for name in ("conv_state", "recurrent_state", "keys", "values"):
        np.testing.assert_allclose(a[prefix+".full."+name], a[prefix+".tokenwise.step.9."+name], **tier["tolerance"])


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

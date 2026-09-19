"""Offline model-side mRoPE, weighted hybrid generation, and cache/JIT lifecycle tests."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, nn, dtypes
from tinygrad.llm.model import apply_rope
from tinygrad.llm.multimodal import PreparedPrompt, embed_prompt, image_positions
from test.null.test_llm_multimodal import load_fixture
from test.unit.test_llm_multimodal import native_text_model


def make_model(context=64):
  manifest, arrays = load_fixture()
  model, _, _ = native_text_model(manifest, arrays)
  model.max_context = context
  for block in model.blk: block.config = replace(block.config, max_context=context, rope_sections=(2, 1, 1))
  return model


def image_inputs(model, case="two_images", variant=0, prefix=0):
  _, a = load_fixture()
  base = "coordinates." + case + "."
  # Retain official physical spans/positions, map marker IDs into the synthetic vocabulary.
  tokens = [1]*prefix + (a[base+"input_ids"][0] % 64).tolist()
  pos = a[base+"position_ids"].astype(np.int32) + prefix
  if prefix: pos = np.concatenate((np.broadcast_to(np.arange(prefix, dtype=np.int32), (3, 1, prefix)), pos), axis=-1)
  embeds = model.token_embd(Tensor([tokens], dtype=dtypes.int32, device=model.token_embd.weight.device)).float().numpy()
  for i, (start, count) in enumerate(a[base+"image_spans"]):
    embeds[:, prefix+start:prefix+start+count] = a["vision.merger"][i*6:i*6+count] * (1 if variant == 0 else -3)
  return tokens, Tensor(embeds), Tensor(pos), int(a[base+"rope_delta"].item())


def take(model, inputs, count=6, chunk_size=32):
  tokens, embeddings, positions, delta = inputs
  gen = model.generate(list(tokens), chunk_size=chunk_size, inputs_embeds=embeddings, position_ids=positions, rope_delta=delta)
  try: return [next(gen) for _ in range(count)]
  finally: gen.close()


class TestQwenMRope(unittest.TestCase):
  def test_target_rotary_fixture(self):
    from tinygrad.llm.model import multimodal_freqs_cis
    m, a = load_fixture()
    pos = Tensor(a["coordinates.two_images.position_ids"].astype(np.int32))
    freqs = multimodal_freqs_cis(pos, 64, 1e7, (11, 11, 10))
    cos, sin = freqs.numpy().reshape(1, pos.shape[-1], 2, 32).transpose(2, 0, 1, 3)
    for name, actual in (("cos", cos), ("sin", sin)):
      np.testing.assert_allclose(actual, a["coordinates.mrope."+name][..., :32], **m["parity_tiers"]["hf_fp32_algebra"]["tolerance"])
    for name, src in (("q", "query"), ("k", "key")):
      x = Tensor(a["coordinates.mrope."+src])
      out = apply_rope(x[..., :64], freqs).cat(x[..., 64:], dim=-1).numpy()
      np.testing.assert_allclose(out, a["coordinates.mrope.rotary_"+name], rtol=1e-4, atol=1e-5)
      np.testing.assert_array_equal(out[..., 64:], a["coordinates.mrope."+src][..., 64:])

  def test_sequential_positions_equal_text(self):
    m, a = load_fixture()
    model = make_model()
    x = Tensor(a["text.inputs_embeds"])
    plain = model.forward_embeddings(x, 0).numpy()
    positions = Tensor(np.broadcast_to(np.arange(x.shape[1], dtype=np.int32), (3, 1, x.shape[1])).copy())
    coords = model.forward_embeddings(x, 0, positions).numpy()
    np.testing.assert_allclose(coords, plain, rtol=1e-4, atol=1e-5)

  def test_forward_rejects_bad_coordinates_and_physical_offsets(self):
    model = make_model()
    _, x, positions, _ = image_inputs(model)
    for start, embeds, pos in ((-1, x, positions), (model.max_context, x, positions), (0, x.half(), positions),
                                (0, x, positions.cast(dtypes.int64)), (0, x, positions[:, 0]),
                                (0, x, positions.to("CPU:1")), (0, x[:, :0], positions[:, :, :0])):
      with self.assertRaises(ValueError): model.forward_embeddings(embeds, start, pos)
    for sections in ((1, 1), (1, 1, 1), (0, 2, 2)):
      model.blk[1].config = replace(model.blk[1].config, rope_sections=sections)
      with self.assertRaises(ValueError): model.forward_embeddings(x, 0, positions)

  def test_physical_cache_and_chunked_logits(self):
    model = make_model()
    tokens, embeds, positions, _ = image_inputs(model)
    baseline = model.forward_embeddings(embeds, 0, positions).numpy()
    kv = model.blk[1].cache_kv.numpy().copy()
    # The second chunk begins inside the image, where all three coordinates differ from the physical offset.
    for start, end in ((0, 5), (5, 10), (10, len(tokens))):
      actual = model.forward_embeddings(embeds[:, start:end].contiguous(), start, positions[:, :, start:end].contiguous()).numpy()
    np.testing.assert_allclose(actual, baseline, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(model.blk[1].cache_kv.numpy()[:, :, :, :len(tokens)], kv[:, :, :, :len(tokens)], rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(kv[:, :, :, len(tokens):], 0)


class TestQwenGenerate(unittest.TestCase):
  def test_temperature_is_runtime_decode_input(self):
    from tinygrad import UOp
    model = make_model()
    def logits(x, start_pos, position_ids=None):
      # Preserve all JIT tensor inputs while making stochastic sampling independently predictable.
      return Tensor([[0., 1.]], device=x.device) + x.sum()*0 + position_ids.sum()*0 + Tensor(start_pos, device=x.device)*0
    uniform = Tensor([[0.9, 0.1]], device=model.token_embd.weight.device).realize()
    tokens = Tensor([[1]], dtype=dtypes.int32, device=uniform.device).realize()
    coords = Tensor([[[0]]]*3, dtype=dtypes.int32, device=uniform.device).realize()
    temp = Tensor([0.1], device=uniform.device).realize()
    physical = UOp.variable("start_pos",0,model.max_context-1).bind(0)
    with patch.object(model,"forward_embeddings",logits), patch.object(Tensor,"rand_like",return_value=uniform):
      for value, expected in ((0.1,1),(0.1,1),(10.,0),(0.1,1),(10.,0)):
        temp.assign(Tensor([value],device=uniform.device)).realize()
        self.assertEqual(model.multimodal_rollout_jit(tokens,physical,coords,temp).item(),expected)
    self.assertGreaterEqual(model.multimodal_rollout_jit.cnt,5)

  def test_chunk_partitions_and_decode_replay(self):
    model = make_model(context=96)
    inputs = image_inputs(model, prefix=28)
    original, calls = model.forward_embeddings, []
    def forward(x, start_pos, position_ids=None):
      physical = start_pos if isinstance(start_pos, int) else start_pos.unbind()[1]
      calls.append((physical, x.shape[1]))
      return original(x, start_pos, position_ids)
    with patch.object(model, "forward_embeddings", forward): result = take(model, inputs, chunk_size=32)
    self.assertEqual(calls[:2], [(0, 32), (32, len(inputs[0])-32)])
    self.assertGreaterEqual(model.multimodal_rollout_jit.cnt, 3)
    buffers = (model.blk[0].conv_state, model.blk[0].recurrent_state, model.blk[1].cache_kv)
    snapshot = [b.numpy().copy() for b in buffers]
    calls.clear()
    with patch.object(model, "forward_embeddings", forward): tokenwise = take(model, inputs, chunk_size=1)
    self.assertEqual(calls[:len(inputs[0])], [(i, 1) for i in range(len(inputs[0]))])
    self.assertEqual(tokenwise, result)
    for b, reference in zip(buffers, snapshot):
      actual = b.numpy()
      if b.dtype == dtypes.float16:
        # Different FP32 reduction orders can straddle an FP16 midpoint. Bound cache drift to ONE stored ULP,
        # not the FP32 algebra tolerance (nor a blanket relaxed tolerance for the recurrent state).
        ulp = np.maximum(np.abs(np.spacing(actual)), np.abs(np.spacing(reference)))
        self.assertTrue((np.abs(actual.astype(np.float32)-reference.astype(np.float32)) <= ulp).all())
      else: np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-5)
    self.assertEqual(model._cached_tokens, [])

  def test_changed_images_and_lengths(self):
    model = make_model()
    for case, variant in (("two_images", 0), ("rectangle", 1), ("two_images", 1), ("rectangle", 0)):
      inputs = image_inputs(model, case, variant)
      self.assertEqual(take(model, inputs), take(make_model(), inputs))
    a, b = image_inputs(model), image_inputs(model, variant=1)
    self.assertEqual(a[0], b[0])
    logits_a = model.forward_embeddings(a[1], 0, a[2]).numpy()
    logits_b = model.forward_embeddings(b[1], 0, b[2]).numpy()
    self.assertGreater(np.abs(logits_a-logits_b).max(), 1e-3)

  def test_multimodal_nondefault_device(self):
    model = make_model()
    for weight in nn.state.get_parameters(model): weight.to_("CPU:1")
    ids, x, pos, delta = image_inputs(model)
    inputs = ids, x.to("CPU:1"), pos.to("CPU:1"), delta
    self.assertEqual(take(model, inputs), take(make_model(), (ids, x, pos, delta)))

  def test_context_exhaustion_and_output_cap(self):
    model = make_model(context=26)
    inputs = image_inputs(model)
    tokens, embeddings, positions, delta = inputs
    gen = model.generate(tokens.copy(), inputs_embeds=embeddings, position_ids=positions, rope_delta=delta)
    with patch.object(model, "multimodal_rollout_jit", wraps=model.multimodal_rollout_jit) as decode:
      self.assertEqual(len(list(gen)), model.max_context-len(tokens))
      self.assertEqual(decode.call_count, model.max_context-len(tokens)-1)
    self.assertEqual(model._cached_tokens, [])
    with patch.object(model, "multimodal_rollout_jit", wraps=model.multimodal_rollout_jit) as decode:
      self.assertEqual(len(take(model, inputs, count=1)), 1)
      decode.assert_not_called()
    self.assertEqual(model._cached_tokens, [])

  def test_text_warmup_image_text_and_old_iterators(self):
    model = make_model()
    model.warmup()
    image = image_inputs(model)
    self.assertEqual(take(model, image), take(make_model(), image))
    text = model.generate([3, 17, 5])
    expected = make_model().generate([3, 17, 5])
    self.assertEqual([next(text) for _ in range(6)], [next(expected) for _ in range(6)])
    old = model.generate(image[0].copy(), inputs_embeds=image[1], position_ids=image[2], rope_delta=image[3])
    next(old)
    new = model.generate([3, 17, 5])
    next(new)
    cached, epoch = model._cached_tokens.copy(), model._generation_epoch
    old.close()
    self.assertEqual(model._generation_epoch, epoch)
    self.assertEqual(model._cached_tokens, cached)
    self.assertIsInstance(next(new), int)
    # Retained text and image iterators must stop rather than writing into a newer request's caches.
    self.assertEqual(list(text), [])
    a = model.generate(image[0].copy(), inputs_embeds=image[1], position_ids=image[2], rope_delta=image[3])
    next(a)
    b = model.generate(image[0].copy(), inputs_embeds=image[1], position_ids=image[2], rope_delta=image[3])
    next(b)
    self.assertEqual(list(a), [])
    self.assertIsInstance(next(b), int)
    b.close()
    new.close()
    expected.close()

  def test_reset_and_partial_prefill_failure(self):
    model = make_model()
    image = image_inputs(model)
    take(model, image)
    buffers = (model.blk[0].conv_state, model.blk[0].recurrent_state, model.blk[1].cache_kv)
    model._cached_tokens = [1, 2, 3]
    original, starts = model.forward_embeddings, []
    def fail(x, start, position_ids=None):
      self.assertEqual(model._cached_tokens, [])
      starts.append(start.unbind()[1])
      if len(starts) == 2: raise RuntimeError("prefill interrupted")
      return original(x, start, position_ids)
    with patch.object(model, "forward_embeddings", fail), self.assertRaisesRegex(RuntimeError, "prefill interrupted"):
      take(model, image, chunk_size=4)
    self.assertEqual(starts, [0, 4])
    self.assertEqual(model._cached_tokens, [])
    model.reset_generation_state()
    self.assertEqual(buffers, (model.blk[0].conv_state, model.blk[0].recurrent_state, model.blk[1].cache_kv))
    self.assertEqual(take(model, image), take(make_model(), image))

  def test_decode_failure_and_explicit_reset_cancel(self):
    model = make_model()
    ids, x, pos, delta = image_inputs(model)
    def generate(): return model.generate(ids.copy(), inputs_embeds=x, position_ids=pos, rope_delta=delta)
    gen = generate()
    next(gen)
    with patch.object(model, "multimodal_rollout_jit", side_effect=RuntimeError("decode interrupted")):
      with self.assertRaisesRegex(RuntimeError, "decode interrupted"): next(gen)
    self.assertEqual(model._cached_tokens, [])
    self.assertFalse(model._multimodal_active)
    gen = generate()
    next(gen)
    model.reset_generation_state()
    with patch.object(model, "multimodal_rollout_jit") as decode:
      self.assertEqual(list(gen), [])
      decode.assert_not_called()
    self.assertEqual(take(model, (ids, x, pos, delta)), take(make_model(), (ids, x, pos, delta)))

  def test_decode_coordinates_and_cache_match_eager(self):
    model = make_model()
    inputs = image_inputs(model)
    ids, x, pos, delta = inputs
    eager = make_model()
    logits = eager.forward_embeddings(x, 0, pos).numpy()
    gen = model.generate(ids.copy(), inputs_embeds=x, position_ids=pos, rope_delta=delta)
    original, signatures, physicals, coordinates = model.multimodal_rollout_jit, [], [], []
    def decode(tokens, start, positions, temperature):
      signatures.append((tokens.shape, tokens.dtype, tokens.device, positions.shape, positions.dtype, positions.device))
      physicals.append(start.unbind()[1])
      coordinates.append(positions.numpy().copy())
      return original(tokens, start, positions, temperature)
    with patch.object(model, "multimodal_rollout_jit", decode):
      for step in range(6):
        token = next(gen)
        self.assertEqual(token, int(logits.argmax()))
        consumed = len(ids) + step
        np.testing.assert_allclose(model.blk[1].cache_kv.numpy()[:, :, :, :consumed],
                                   eager.blk[1].cache_kv.numpy()[:, :, :, :consumed], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(model.blk[0].recurrent_state.numpy(), eager.blk[0].recurrent_state.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(model.blk[0].conv_state.numpy(), eager.blk[0].conv_state.numpy(), rtol=1e-4, atol=1e-5)
        if step < 5:
          embeds = eager.token_embd(Tensor([[token]], dtype=dtypes.int32)).float()
          logits = eager.forward_embeddings(embeds, consumed, Tensor.full((3, 1, 1), consumed+delta, dtype=dtypes.int32)).numpy()
    gen.close()
    self.assertEqual(len(signatures), 5)  # two captures plus THREE actual replays
    self.assertEqual(len(set(signatures)), 1)
    self.assertEqual(physicals, list(range(len(ids), len(ids)+5)))
    for p, c in zip(physicals, coordinates): np.testing.assert_array_equal(c, p+delta)

  def test_invalid_inputs_clear_reuse_before_work(self):
    model = make_model()
    ids, x, pos, delta = image_inputs(model)
    good = {"inputs_embeds":x, "position_ids":pos, "rope_delta":delta}
    cases = [{"inputs_embeds":None}, {"position_ids":None}, {"inputs_embeds":x.half()}, {"inputs_embeds":x[:, :-1]},
             {"position_ids":pos.cast(dtypes.int64)}, {"position_ids":pos[:, 0]}, {"position_ids":pos.to("CPU:1")},
             {"inputs_embeds":x.to("CPU:1")}, {"inputs_embeds":x*float("nan")}, {"position_ids":pos-1},
             {"rope_delta":-100}, {"rope_delta":2**31}, {"rope_delta":1.5}, {"rope_delta":delta+1}, {"chunk_size":0},
             {"temperature":float("nan")}, {"temperature":float("inf")}, {"temperature":-1},
             {"temperature":10**400}, {"temperature":True}]
    for bad in cases:
      with self.subTest(bad=list(bad)), patch.object(model, "forward_embeddings") as forward:
        model._cached_tokens = [1, 2]
        gen = model.generate(ids.copy(), **(good | bad))
        with self.assertRaises(ValueError): next(gen)
        forward.assert_not_called()
        self.assertEqual(model._cached_tokens, [])
    with self.assertRaisesRegex(ValueError, "rope_delta"):
      next(model.generate(ids.copy(), inputs_embeds=x, position_ids=pos))
    for tokens in ([], ids+[1]*model.max_context, [-1]*len(ids), [1000]*len(ids)):
      with self.assertRaises(ValueError): next(model.generate(tokens, **good))

  def test_four_image_encode_once_outside_generate(self):
    from tinygrad.llm.vision import QwenVision, VisionConfig
    m, a = load_fixture()
    model = make_model()
    vision = QwenVision(VisionConfig.from_dict(m["configs"]["vision"]))
    nn.state.load_state_dict(vision, {k.removeprefix("vision.weights."):Tensor(v) for k,v in a.items() if k.startswith("vision.weights.")},
                            verbose=False)
    embedding = model.token_embd
    class SmallVocabulary:
      weight = embedding.weight
      def __call__(self, ids): return embedding(ids % 64)
    model.token_embd = SmallVocabulary()
    tokens, spans = [1, 2], []
    grids = ((1, 4, 6), (1, 6, 4))*2
    for _ in grids:
      tokens.append(248053)
      spans.append((len(tokens), 6))
      tokens += [248056]*6 + [248054, 3]
    pos, delta = image_positions(tuple(tokens), grids, tuple(spans))
    pixels = Tensor(np.tile(a["vision.pixel_values"], (2, 1)))
    prompt = PreparedPrompt(tuple(tokens), pixels, grids, tuple(spans), Tensor(pos), delta)
    with patch.object(QwenVision, "__call__", autospec=True, side_effect=QwenVision.__call__) as encode:
      fused = embed_prompt(model, vision, prompt)
      inputs = ([t % 64 for t in tokens], fused, prompt.position_ids, delta)
      result = take(model, inputs, chunk_size=32)
      encode.assert_called_once()  # prefill/decode never own the encoder
    actual = fused.numpy()
    for i, (start, count) in enumerate(spans):
      np.testing.assert_allclose(actual[0, start:start+count], a["vision.merger"][(i%2)*6:(i%2+1)*6], rtol=1e-4, atol=1e-5)
    self.assertEqual(result, take(make_model(), inputs, chunk_size=1))

  def test_different_grids_same_physical_length(self):
    model = make_model()
    ids, embeds, _, _ = image_inputs(model, "rectangle")
    variants = []
    for grid in ((1, 4, 6), (1, 6, 4), (1, 2, 12)):
      # All have six image tokens; the last grid also changes the rotary delta.
      pos, delta = image_positions(tuple([101, 102, 248053]+[248056]*6+[248054, 103, 104, 105]), (grid,), ((3, 6),))
      inputs = ids, embeds, Tensor(pos), delta
      self.assertEqual(take(model, inputs), take(make_model(), inputs))
      variants.append(model.forward_embeddings(embeds, 0, Tensor(pos)).numpy())
    self.assertGreater(np.abs(variants[0]-variants[1]).max(), 1e-4)
    self.assertGreater(np.abs(variants[0]-variants[2]).max(), 1e-4)

  def test_full_context_does_no_work(self):
    model = make_model(context=22)
    ids, x, pos, delta = image_inputs(model)
    with patch.object(model, "forward_embeddings") as forward:
      self.assertEqual(list(model.generate(ids, inputs_embeds=x, position_ids=pos, rope_delta=delta)), [])
      forward.assert_not_called()
    self.assertEqual(model._cached_tokens, [])

if __name__ == "__main__": unittest.main()

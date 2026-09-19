import copy, json, re, unittest
from unittest.mock import Mock
import numpy as np
from tinygrad import Tensor, dtypes, nn
from test.null import test_llm_multimodal as fixtures

class MarkerTokenizer:
  def encode(self, text):
    from tinygrad.llm.multimodal import IMAGE_TOKEN_ID, VISION_START_ID, VISION_END_ID
    special = {"<|image_pad|>":IMAGE_TOKEN_ID, "<|vision_start|>":VISION_START_ID, "<|vision_end|>":VISION_END_ID}
    parts = re.split(r"(<\|image_pad\|>|<\|vision_start\|>|<\|vision_end\|>)", text)
    return [tid for part in parts for tid in ([special[part]] if part in special else [ord(c) for c in part])]

class TestQwenPrompt(unittest.TestCase):
  def setup_prompt(self, count=1, device="CPU"):
    import jinja2
    from PIL import Image
    from tinygrad.llm.multimodal import ImageLimits, prepare_prompt
    _, arrays = fixtures.load_fixture()
    config = json.loads(arrays["metadata.tokenizer_config.json"].tobytes())
    template = jinja2.Environment(trim_blocks=True, lstrip_blocks=True).from_string(config["chat_template"])
    template.environment.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    messages = [{"role":"user", "content":[{"type":"text", "text":"What color?"}] +
                [{"type":"image_url", "image_url":{"url":fixtures.TestQwenPreprocess.image_url(Image.new("RGB", (256,256), "red"))}}
                 for _ in range(count)]}]
    prompt = prepare_prompt(messages, MarkerTokenizer(), template, limits=ImageLimits(), device=device)
    return prompt, messages, template

  def test_prompt_shapes_coordinates_and_immutability(self):
    from tinygrad.llm.multimodal import ImageLimits, prepare_prompt, IMAGE_TOKEN_ID
    prompt, messages, template = self.setup_prompt(count=2, device="CPU:1")
    before = copy.deepcopy(messages)
    again = prepare_prompt(messages, MarkerTokenizer(), template, limits=ImageLimits(), device="CPU:1", template_kwargs={"enable_thinking":False})
    self.assertEqual(messages, before)
    self.assertEqual(prompt.pixel_values.shape, (512,1536))
    self.assertEqual(prompt.pixel_values.device, "CPU:1")
    self.assertEqual(prompt.position_ids.device, "CPU:1")
    self.assertEqual(prompt.tokens.count(IMAGE_TOKEN_ID), 128)
    self.assertEqual(prompt.rope_delta, -112)
    self.assertTrue(prompt.starts_reasoning)
    self.assertFalse(again.starts_reasoning)

  def test_prompt_coordinates_match_oracle(self):
    from tinygrad.llm.multimodal import image_positions
    manifest, arrays = fixtures.load_fixture()
    for case in manifest["coordinate_cases"]:
      p = "coordinates." + case + "."
      tokens = tuple(int(v) for v in arrays[p+"input_ids"][0])
      grids = tuple(tuple(int(v) for v in g) for g in arrays[p+"grid_thw"])
      spans = tuple(tuple(int(v) for v in g) for g in arrays[p+"image_spans"])
      positions, delta = image_positions(tokens, grids, spans)
      np.testing.assert_array_equal(positions, arrays[p+"position_ids"])
      self.assertEqual(delta, int(arrays[p+"rope_delta"][0,0]))

  def test_prompt_rejects_invalid_media(self):
    from tinygrad.llm.multimodal import prepare_prompt, ImageLimits, ImageInputError
    _, messages, template = self.setup_prompt()
    bad_messages = ([{"role":"user", "content":"<|image_pad|>"}], [{"role":"system", "content":messages[0]["content"]}],
                    [{"role":"user", "content":[{"type":"audio"}]}], [{"role":"user", "content":[{"type":"text", "text":5}]}])
    for bad in bad_messages:
      with self.subTest(bad=bad), self.assertRaises(ImageInputError):
        prepare_prompt(bad, MarkerTokenizer(), template, limits=ImageLimits(), device="CPU")
    with self.assertRaises(ImageInputError):
      prepare_prompt(messages*5, MarkerTokenizer(), template, limits=ImageLimits(), device="CPU")
    with self.assertRaises(ImageInputError):
      prepare_prompt(messages, MarkerTokenizer(), Mock(render=lambda **kw:"text only"), limits=ImageLimits(), device="CPU")

  def test_prompt_fusion_replaces_image_embeddings(self):
    from tinygrad.llm.multimodal import PreparedPrompt, embed_prompt, IMAGE_TOKEN_ID
    # Real embedding table, mock only the already-tested encoder to isolate substitution.
    model = Mock(token_embd=nn.Embedding(IMAGE_TOKEN_ID+1, 2))
    model.token_embd.weight = Tensor.ones(IMAGE_TOKEN_ID+1, 2)
    tokens = (1, IMAGE_TOKEN_ID, 2)
    positions = Tensor([[[0,1,2]]]*3, dtype=dtypes.int32)
    prompt = PreparedPrompt(tokens, Tensor.zeros(4,1536), ((1,2,2),), ((1,1),), positions)
    vision = Mock(return_value=Tensor([[3.,4.]]))
    actual = embed_prompt(model, vision, prompt)
    np.testing.assert_array_equal(actual.numpy(), [[[1.,1.],[3.,4.],[1.,1.]]])
    vision.assert_called_once()
    with self.assertRaises(ValueError): PreparedPrompt(tokens, prompt.pixel_values, ((1,2,2),), ((0,1),), positions)
    with self.assertRaises(ValueError): PreparedPrompt(tokens, prompt.pixel_values, ((1,2,2),), ((1,2),), positions)

if __name__ == "__main__": unittest.main()

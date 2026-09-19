"""Host-only contracts: these tests also run with DEV=NULL, without an oracle install."""
import hashlib, io, json, unittest, zipfile
from pathlib import Path
import numpy as np

FIXTURES = Path(__file__).resolve().parents[1] / "models/qwen_vl"

def load_fixture():
  manifest = json.loads((FIXTURES / "manifest.json").read_text())
  with np.load(FIXTURES / "reference.npz", allow_pickle=False) as archive:
    arrays = {name: archive[name] for name in archive.files}
  return manifest, arrays

class TestQwenFixtureContracts(unittest.TestCase):
  def test_fixture_schema(self):
    m, arrays = load_fixture()
    self.assertEqual(m["schema_version"], 1)
    self.assertEqual(m["source"]["transformers_commit"], "049d2bf1220747b6d39e2a978b9f5fe0defa1dca")
    self.assertEqual(m["source"]["model_revision"], "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    self.assertEqual(m["versions"]["transformers"], "5.8.0")
    self.assertEqual(m["backend"]["attention"], "eager")
    self.assertFalse(m["backend"]["optional_kernels"])
    self.assertEqual(set(m["parity_tiers"]), {"hf_fp32_algebra", "native_cast", "amd_quantized"})
    self.assertEqual(m["parity_tiers"]["native_cast"]["status"], "captured")
    self.assertEqual(m["parity_tiers"]["amd_quantized"]["status"], "deferred_E")
    self.assertEqual(m["seed"], 20260919)
    self.assertEqual(m["parity_tiers"]["hf_fp32_algebra"]["tolerance"], {"rtol": 1e-4, "atol": 1e-5})
    self.assertEqual(set(arrays), set(m["arrays"]))
    generator = FIXTURES.parents[1] / "external/external_qwen_vl.py"
    self.assertEqual(hashlib.sha256(generator.read_bytes()).hexdigest(), m["generator"]["sha256"])
    for name, a in arrays.items():
      with self.subTest(array=name):
        self.assertEqual(list(a.shape), m["arrays"][name]["shape"])
        self.assertEqual(a.dtype.str, m["arrays"][name]["dtype"])
        self.assertFalse(a.dtype.hasobject)
        self.assertTrue(np.isfinite(a).all())

  def test_fixture_digests_and_deterministic_archive(self):
    m, arrays = load_fixture()
    raw = (FIXTURES / "reference.npz").read_bytes()
    self.assertLessEqual(len(raw) + (FIXTURES / "manifest.json").stat().st_size, 2 * 1024 * 1024)
    self.assertEqual(hashlib.sha256(raw).hexdigest(), m["archive"]["sha256"])
    self.assertEqual(len(raw), m["archive"]["bytes"])
    rebuilt = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as src, zipfile.ZipFile(rebuilt, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as dst:
      self.assertEqual(src.namelist(), sorted(name + ".npy" for name in arrays))
      for info in src.infolist():
        self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0))
        self.assertEqual(info.extra, b"")
        self.assertEqual(info.comment, b"")
        name = info.filename[:-4]
        a = arrays[name]
        self.assertEqual(hashlib.sha256(a.tobytes(order="C")).hexdigest(), m["arrays"][name]["sha256"])
        payload = io.BytesIO()
        np.lib.format.write_array(payload, a, version=(1, 0), allow_pickle=False)
        self.assertEqual(payload.getvalue(), src.read(info.filename))
        dst.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    self.assertEqual(rebuilt.getvalue(), raw)

  def test_fixture_pinned_metadata_and_template(self):
    m, arrays = load_fixture()
    for name, expected in m["source"]["metadata_sha256"].items():
      self.assertEqual(hashlib.sha256(arrays["metadata." + name].tobytes()).hexdigest(), expected)
    config = json.loads(arrays["metadata.tokenizer_config.json"].tobytes())
    self.assertEqual(hashlib.sha256(config["chat_template"].encode()).hexdigest(), m["template"]["chat_template_sha256"])
    for token, token_id in m["template"]["special_tokens"].items():
      self.assertEqual(config["added_tokens_decoder"][str(token_id)]["content"], token)
    for name, suffix in (("thinking", "<think>\n"), ("no_thinking", "<think>\n\n</think>\n\n")):
      rendered = arrays["template." + name].tobytes().decode()
      self.assertEqual(rendered.count("<|vision_start|><|image_pad|><|vision_end|>"), 1)
      self.assertIn("What color?", rendered)
      self.assertTrue(rendered.endswith(suffix))

  def test_fixture_processor_layout(self):
    m, arrays = load_fixture()
    self.assertEqual(m["processor"]["size"], {"shortest_edge": 65536, "longest_edge": 262144})
    for case in m["processor_cases"]:
      p = "processor." + case["name"] + "."
      grid, patches = arrays[p + "grid_thw"][0], arrays[p + "pixel_values"]
      t, h, w = grid
      self.assertEqual(t, 1)
      self.assertEqual((h % 2, w % 2), (0, 0))
      self.assertEqual(patches.shape, (h * w, 1536))
      self.assertEqual(list(grid), case["grid_thw"])
      self.assertGreaterEqual(h * w * 256, 65536)
      self.assertLessEqual(h * w * 256, 262144)
      unpacked = patches.reshape(h//2, w//2, 2, 2, 3, 2, 16, 16)
      np.testing.assert_array_equal(unpacked[..., 0, :, :], unpacked[..., 1, :, :])
      image = unpacked[..., 0, :, :].transpose(0, 2, 5, 1, 3, 6, 4).reshape(h*16, w*16, 3)
      np.testing.assert_array_equal(image, arrays[p + "normalized_chw"].transpose(1, 2, 0))
      np.testing.assert_allclose(image, arrays[p + "resized_rgb"].astype(np.float32)/255*2-1, atol=1e-7, rtol=0)

  def test_fixture_coordinates(self):
    m, arrays = load_fixture()
    for case in m["coordinate_cases"]:
      p = "coordinates." + case + "."
      ids, types = arrays[p + "input_ids"], arrays[p + "mm_token_type_ids"]
      pos, delta, spans, grids = (arrays[p + k] for k in ("position_ids", "rope_delta", "image_spans", "grid_thw"))
      self.assertEqual(pos.shape, (3, 1, ids.shape[1]))
      self.assertEqual(int(delta[0, 0]), int(pos.max()) + 1 - ids.shape[1])
      self.assertEqual(len(spans), len(grids))
      for (start, count), (_, h, w) in zip(spans, grids):
        self.assertEqual(count, h*w//4)
        self.assertTrue((types[0, start:start+count] == 1).all())
        self.assertTrue((ids[0, start:start+count] == 248056).all())
        logical_start = pos[0, 0, start]
        np.testing.assert_array_equal(pos[0, 0, start:start+count], np.full(count, logical_start))
        np.testing.assert_array_equal(pos[1, 0, start:start+count], logical_start + np.repeat(np.arange(h//2), w//2))
        np.testing.assert_array_equal(pos[2, 0, start:start+count], logical_start + np.tile(np.arange(w//2), h//2))
    self.assertEqual(int(arrays["coordinates.square256.rope_delta"][0, 0]), -56)
    self.assertEqual(int(arrays["coordinates.square1024.rope_delta"][0, 0]), -992)

class TestQwenPreprocess(unittest.TestCase):
  @staticmethod
  def image_url(image, image_format="PNG"):
    import base64
    buf = io.BytesIO()
    image.save(buf, format=image_format)
    return f"data:image/{'jpeg' if image_format == 'JPEG' else 'png'};base64," + base64.b64encode(buf.getvalue()).decode()

  def test_preprocess_reference(self):
    from PIL import Image
    from tinygrad.llm.multimodal import ImageLimits, preprocess_image
    m, arrays = load_fixture()
    for case in m["processor_cases"]:
      p = "processor." + case["name"] + "."
      pixels, grid = preprocess_image(self.image_url(Image.fromarray(arrays[p + "rgb"])), ImageLimits())
      self.assertEqual(grid, tuple(arrays[p + "grid_thw"][0]))
      np.testing.assert_array_equal(pixels, arrays[p + "pixel_values"])

  def test_preprocess_color_modes(self):
    from PIL import Image
    from tinygrad.llm.multimodal import ImageLimits, preprocess_image
    for mode in ("L", "RGBA", "RGB"):
      image = Image.new(mode, (256, 256))
      actual, grid = preprocess_image(self.image_url(image), ImageLimits())
      expected, _ = preprocess_image(self.image_url(image.convert("RGB")), ImageLimits())
      np.testing.assert_array_equal(actual, expected)
      self.assertEqual(grid, (1, 16, 16))

  def test_preprocess_jpeg_and_exif(self):
    from PIL import Image, ImageOps
    from tinygrad.llm.multimodal import ImageLimits, preprocess_image
    image = Image.fromarray(np.arange(128*256*3, dtype=np.uint8).reshape(128, 256, 3))
    exif = Image.Exif()
    exif[274] = 6
    buf = io.BytesIO()
    image.save(buf, format="JPEG", exif=exif)
    import base64
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    actual, grid = preprocess_image(url, ImageLimits())
    canonical = ImageOps.exif_transpose(Image.open(io.BytesIO(buf.getvalue()))).convert("RGB")
    expected, expected_grid = preprocess_image(self.image_url(canonical), ImageLimits())
    self.assertEqual(grid, expected_grid)
    np.testing.assert_array_equal(actual, expected)

  def test_limits_invalid_input(self):
    from PIL import Image
    from tinygrad.llm.multimodal import ImageLimits, ImageInputError, preprocess_image
    for url in ("https://example.com/image.png", "/tmp/image.png", "data:image/gif;base64,AAAA", "data:image/png;base64,!!!",
                "data:image/png;base64,aA==", "data:image/png;base64,", 4, None):
      with self.subTest(url=url), self.assertRaises(ImageInputError): preprocess_image(url, ImageLimits())
    for kwargs in ({"max_pixels":True}, {"max_pixels":0}, {"max_pixels":16777217}, {"max_images":5}, {"max_visual_tokens":0}):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): ImageLimits(**kwargs)
    url = self.image_url(Image.new("RGB", (256, 256)))
    with self.assertRaises(ImageInputError): preprocess_image(url.replace("image/png", "image/jpeg"), ImageLimits())
    with self.assertRaises(ImageInputError): preprocess_image(url, ImageLimits(max_image_bytes=32))
    with self.assertRaises(ImageInputError): preprocess_image(url, ImageLimits(max_original_pixels=100))
    with self.assertRaises(ImageInputError): preprocess_image(self.image_url(Image.new("RGB", (1, 201))), ImageLimits())

  def test_limits_resize_and_dependency(self):
    from PIL import Image
    from unittest.mock import patch
    from tinygrad.llm.multimodal import ImageLimits, ImageInputError, image_size, preprocess_image
    self.assertEqual(image_size(256, 272, ImageLimits()), (256, 256))
    with self.assertRaises(ImageInputError): image_size(1, 200, ImageLimits(max_pixels=65536))
    with self.assertRaises(ImageInputError): image_size(512, 512, ImageLimits(max_visual_tokens=64))
    url = self.image_url(Image.new("RGB", (256, 256)))
    with patch.dict("sys.modules", {"PIL":None}):
      with self.assertRaisesRegex(ImageInputError, r"tinygrad\[vision\]"): preprocess_image(url, ImageLimits())

  def test_limits_animation(self):
    from PIL import Image
    from tinygrad.llm.multimodal import ImageLimits, ImageInputError, preprocess_image
    import base64
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buf, format="PNG", save_all=True, append_images=[Image.new("RGB", (32, 32), "blue")])
    with self.assertRaises(ImageInputError):
      preprocess_image("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), ImageLimits())

class TestQwenCLI(unittest.TestCase):
  def test_cli_flags(self):
    from tinygrad.llm.cli import parse_args
    args = parse_args(["--vision-dir", "/model", "--image", "a.png", "--image", "b.jpg", "--prompt", "Compare"])
    self.assertEqual(args.image, ["a.png", "b.jpg"])
    self.assertEqual(args.max_output_tokens, 256)
    self.assertEqual(args.image_max_pixels, 262144)
    for argv in (["--image", "x"], ["--prompt", "x"], ["--max-output-tokens", "0"],
                 ["--vision-dir", "x", "--no_chat_template"], ["--vision-dir", "x", "--max-images", "5"],
                 ["--vision-dir", "x", "--image", "x", "--prompt", "x", "--serve"],
                 ["--vision-dir", "x", "--image-max-pixels", "3"]):
      with self.subTest(argv=argv), self.assertRaises(SystemExit): parse_args(argv)

  def test_cli_local_image_canonical(self):
    from PIL import Image
    import tempfile
    from tinygrad.llm.multimodal import ImageLimits, ImageInputError, local_image_url, preprocess_image
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp)/"image.bin"
      image = Image.new("RGB", (256,256), "red")
      image.save(path, format="PNG")
      url = local_image_url(path, ImageLimits())
      a, grid = preprocess_image(url, ImageLimits())
      b, expected_grid = preprocess_image(TestQwenPreprocess.image_url(image), ImageLimits())
      np.testing.assert_array_equal(a,b)
      self.assertEqual(grid, expected_grid)
      with self.assertRaises(ImageInputError): local_image_url(path, ImageLimits(max_image_bytes=1))
      path.write_bytes(b"invalid")
      with self.assertRaises(ImageInputError): local_image_url(path, ImageLimits())

if __name__ == "__main__": unittest.main()

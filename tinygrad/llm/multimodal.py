from __future__ import annotations
import base64, binascii, io, math
from dataclasses import dataclass

IMAGE_TOKEN_ID, VISION_START_ID, VISION_END_ID = 248056, 248053, 248054
IMAGE_MARKER = "<|image_pad|>"
MEDIA_MARKERS = ("<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", IMAGE_MARKER, "<|video_pad|>")

class ImageInputError(ValueError):
  def __init__(self, message:str, status:int=400):
    super().__init__(message)
    self.status = status

@dataclass(frozen=True)
class ImageLimits:
  min_pixels:int = 65536
  max_pixels:int = 262144
  max_images:int = 4
  max_visual_tokens:int = 1024
  max_image_bytes:int = 8 * 1024 * 1024
  max_original_pixels:int = 16777216
  max_request_bytes:int = 16 * 1024 * 1024

  def __post_init__(self):
    ceilings = {"min_pixels":65536, "max_pixels":16777216, "max_images":4, "max_visual_tokens":1024,
                "max_image_bytes":8*1024*1024, "max_original_pixels":16777216, "max_request_bytes":16*1024*1024}
    for name, ceiling in ceilings.items():
      value = getattr(self, name)
      if type(value) is not int or not 0 < value <= ceiling: raise ValueError(f"invalid {name}: expected integer in [1, {ceiling}]")
    if self.min_pixels != 65536 or self.max_pixels < self.min_pixels: raise ValueError("image area minimum must be 65536 pixels")

# Matches the pinned Qwen2VL PIL smart-resize policy, including ties-to-even rounding.
def image_size(height:int, width:int, limits:ImageLimits) -> tuple[int, int]:
  if min(height, width) <= 0 or max(height, width) / min(height, width) > 200: raise ImageInputError("unsupported image aspect ratio")
  h, w = round(height/32)*32, round(width/32)*32
  if h*w > limits.max_pixels:
    scale = math.sqrt(height*width / limits.max_pixels)
    h, w = max(32, math.floor(height/scale/32)*32), max(32, math.floor(width/scale/32)*32)
  elif h*w < limits.min_pixels:
    scale = math.sqrt(limits.min_pixels / (height*width))
    h, w = math.ceil(height*scale/32)*32, math.ceil(width*scale/32)*32
  if not limits.min_pixels <= h*w <= limits.max_pixels: raise ImageInputError("image cannot fit the resized pixel budget", 413)
  if h*w//1024 > limits.max_visual_tokens: raise ImageInputError("image exceeds visual token budget", 413)
  return h, w

def preprocess_image(url:str, limits:ImageLimits):
  """Decode bounded PNG/JPEG data into host float32 patches in Qwen's block-major order."""
  if not isinstance(url, str): raise ImageInputError("image_url must be a PNG or JPEG data URL")
  header, separator, encoded = url.partition(",")
  formats = {"data:image/png;base64":"PNG", "data:image/jpeg;base64":"JPEG"}
  if not separator or header not in formats: raise ImageInputError("only PNG and JPEG base64 data URLs are supported")
  if len(encoded) > 4*((limits.max_image_bytes+2)//3): raise ImageInputError("image byte budget exceeded", 413)
  try: raw = base64.b64decode(encoded, validate=True)
  except (ValueError, binascii.Error): raise ImageInputError("invalid image base64") from None
  if not raw: raise ImageInputError("empty image")
  if len(raw) > limits.max_image_bytes: raise ImageInputError("image byte budget exceeded", 413)
  try:
    import numpy as np
    from PIL import Image, ImageOps, UnidentifiedImageError
  except ImportError: raise ImageInputError("image input requires tinygrad[vision]") from None
  try:
    with Image.open(io.BytesIO(raw)) as source:
      if source.format != formats[header]: raise ImageInputError("image MIME type does not match its bytes")
      if source.width*source.height > limits.max_original_pixels: raise ImageInputError("original image pixel budget exceeded", 413)
      if getattr(source, "n_frames", 1) != 1: raise ImageInputError("animated images are not supported")
      image = ImageOps.exif_transpose(source).convert("RGB")
      h, w = image_size(image.height, image.width, limits)
      rgb = np.asarray(image.resize((w, h), Image.Resampling.BICUBIC))
  except ImageInputError: raise
  except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError): raise ImageInputError("invalid image data") from None
  # HF PIL rescales in float64 then casts float32 before normalization.
  normalized = ((rgb.astype(np.float64)/255).astype(np.float32)-np.float32(0.5))/np.float32(0.5)
  gh, gw = h//16, w//16
  patches = normalized.reshape(gh//2, 2, 16, gw//2, 2, 16, 3).transpose(0, 3, 1, 4, 6, 2, 5)
  patches = np.repeat(patches[..., None, :, :], 2, axis=-3).reshape(gh*gw, 1536)
  return np.ascontiguousarray(patches), (1, gh, gw)

from __future__ import annotations
import base64, binascii, copy, io, json, math
from dataclasses import dataclass
from typing import TYPE_CHECKING
from pathlib import Path
from tinygrad import Tensor, dtypes
if TYPE_CHECKING:
  from tinygrad.llm.cli import SimpleTokenizer
  from tinygrad.llm.model import Transformer
  from tinygrad.llm.vision import QwenVision

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

def local_image_url(path:Path, limits:ImageLimits) -> str:
  """CLI-only ingress; HTTP never resolves a path."""
  if not path.is_file() or path.stat().st_size > limits.max_image_bytes: raise ImageInputError("invalid local image size", 413)
  with path.open("rb") as stream: raw = stream.read(limits.max_image_bytes+1)
  if len(raw) > limits.max_image_bytes: raise ImageInputError("image byte budget exceeded", 413)
  if raw.startswith(b"\x89PNG\r\n\x1a\n"): mime = "png"
  elif raw.startswith(b"\xff\xd8\xff"): mime = "jpeg"
  else: raise ImageInputError("local image must be PNG or JPEG")
  return f"data:image/{mime};base64," + base64.b64encode(raw).decode("ascii")

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

@dataclass(frozen=True)
class PreparedPrompt:
  tokens:tuple[int,...]
  pixel_values:Tensor|None = None
  grid_thw:tuple[tuple[int,int,int],...] = ()
  image_spans:tuple[tuple[int,int],...] = ()
  position_ids:Tensor|None = None
  rope_delta:int = 0
  starts_reasoning:bool = False

  def __post_init__(self):
    if not self.tokens or any(type(t) is not int or t < 0 for t in self.tokens): raise ImageInputError("invalid prompt tokens")
    if not self.grid_thw:
      if self.pixel_values is not None or self.image_spans or self.position_ids is not None or self.rope_delta != 0:
        raise ImageInputError("text prompt has unexpected image metadata")
      return
    if len(self.grid_thw) != len(self.image_spans) or any(len(g) != 3 or any(type(n) is not int for n in g) or
        g[0] != 1 or min(g[1:]) <= 0 or g[1]%2 or g[2]%2 for g in self.grid_thw): raise ImageInputError("invalid image grids")
    if (self.pixel_values is None or self.pixel_values.shape != (sum(h*w for _,h,w in self.grid_thw),1536) or
        self.pixel_values.dtype != dtypes.float32): raise ImageInputError("patch count does not match grids")
    if (self.position_ids is None or self.position_ids.shape != (3,1,len(self.tokens)) or self.position_ids.dtype != dtypes.int32 or
        self.position_ids.device != self.pixel_values.device): raise ImageInputError("invalid image position tensor")
    end = 0
    for (start, count), (_,h,w) in zip(self.image_spans, self.grid_thw):
      if type(start) is not int or type(count) is not int or start < end or count != h*w//4 or start+count > len(self.tokens):
        raise ImageInputError("invalid image embedding span")
      if self.tokens[start:start+count] != (IMAGE_TOKEN_ID,)*count: raise ImageInputError("image span is not an image placeholder")
      end = start+count
    if self.tokens.count(IMAGE_TOKEN_ID) != sum(c for _,c in self.image_spans): raise ImageInputError("unmatched image placeholder")

def image_positions(tokens:tuple[int,...], grids:tuple[tuple[int,int,int],...], spans:tuple[tuple[int,int],...]):
  """Host integer coordinates, kept separate from physical sequence/cache offsets."""
  import numpy as np
  positions = np.empty((3,1,len(tokens)), dtype=np.int32)
  physical = logical = 0
  for (start, count), (_,h,w) in zip(spans, grids):
    if start < physical or start+count > len(tokens) or count != h*w//4: raise ImageInputError("invalid image position span")
    text = start-physical
    positions[:,0,physical:start] = np.arange(logical, logical+text)
    logical += text
    positions[0,0,start:start+count] = logical
    positions[1,0,start:start+count] = logical + np.repeat(np.arange(h//2), w//2)
    positions[2,0,start:start+count] = logical + np.tile(np.arange(w//2), h//2)
    physical, logical = start+count, logical+max(h//2,w//2)
  positions[:,0,physical:] = np.arange(logical, logical+len(tokens)-physical)
  return positions, int(positions.max())+1-len(tokens)

def prepare_prompt(messages:list[dict], tokenizer:SimpleTokenizer, template, *, limits:ImageLimits, device:str,
                   tools:list[dict]|None=None, template_kwargs:dict|None=None, max_context:int|None=None) -> PreparedPrompt:
  import numpy as np
  messages, tools = copy.deepcopy(messages), copy.deepcopy(tools)
  if not isinstance(messages, list) or not messages: raise ImageInputError("messages must be a nonempty array")
  options = dict(template_kwargs or {})
  if set(options) - {"enable_thinking", "preserve_thinking", "reasoning_effort"}: raise ImageInputError("unsupported chat template option")
  for key in ("enable_thinking", "preserve_thinking"):
    if key in options and type(options[key]) is not bool: raise ImageInputError(f"{key} must be boolean")
  if "reasoning_effort" in options and options["reasoning_effort"] not in ("xhigh", "medium", "low"):
    raise ImageInputError("unsupported reasoning effort")
  options.setdefault("preserve_thinking", True)
  patches:list[np.ndarray] = []
  grids:list[tuple[int,int,int]] = []
  for msg in messages:
    if not isinstance(msg, dict) or msg.get("role") not in ("system", "user", "assistant", "tool"):
      raise ImageInputError("invalid message role")
    content = msg.get("content")
    parts:list[dict]
    if isinstance(content, str): parts = [{"type":"text", "text":content}]
    elif isinstance(content, list): parts = content
    elif content is None: parts = []
    else: raise ImageInputError("invalid message content")
    for part in parts:
      if not isinstance(part, dict): raise ImageInputError("invalid content part")
      if part.get("type") == "text":
        if not isinstance(part.get("text"), str): raise ImageInputError("text part must contain a string")
        if any(marker in part["text"] for marker in MEDIA_MARKERS): raise ImageInputError("literal media markers are not allowed")
        text = part["text"]
        part.clear()
        part.update({"type":"text", "text":text})
      elif part.get("type") == "image_url":
        if msg["role"] != "user": raise ImageInputError("images are supported only in user messages")
        if len(grids) >= limits.max_images: raise ImageInputError("image count limit exceeded", 413)
        image = part.get("image_url")
        if not isinstance(image, dict) or not isinstance(image.get("url"), str): raise ImageInputError("invalid image_url part")
        if image.get("detail", "auto") != "auto": raise ImageInputError("only automatic image detail is supported")
        pixels, grid = preprocess_image(image["url"], limits)
        patches.append(pixels)
        grids.append(grid)
        if sum(h*w//4 for _,h,w in grids) > limits.max_visual_tokens: raise ImageInputError("total visual token budget exceeded", 413)
        # The trusted template needs only image kind, never the URL or payload.
        part.clear()
        part.update({"type":"image"})
      else: raise ImageInputError("unsupported content part")
    for tc in msg.get("tool_calls") or []:
      if not isinstance(tc, dict) or not isinstance(tc.get("function"), dict): raise ImageInputError("invalid tool call")
      args = tc["function"].get("arguments")
      if isinstance(args, str):
        try: tc["function"]["arguments"] = json.loads(args)
        except json.JSONDecodeError: pass
  try: rendered = template.render(messages=messages, tools=tools, add_generation_prompt=True, **options)
  except Exception as exc: raise ImageInputError(f"chat template rejected the messages ({type(exc).__name__})") from None
  marker_run = "<|vision_start|>"+IMAGE_MARKER+"<|vision_end|>"
  without_images = rendered.replace(marker_run, "")
  if rendered.count(marker_run) != len(grids) or any(marker in without_images for marker in MEDIA_MARKERS):
    raise ImageInputError("template media markers do not match payloads")
  pieces = rendered.split(IMAGE_MARKER)
  expanded = pieces[0]
  for i, (_,h,w) in enumerate(grids): expanded += IMAGE_MARKER*(h*w//4) + pieces[i+1]
  tokens = tuple(tokenizer.encode(expanded))
  if max_context is not None and len(tokens) >= max_context: raise ImageInputError("expanded prompt exceeds model context")
  if not grids: return PreparedPrompt(tokens, starts_reasoning=rendered.rstrip().endswith("<think>"))
  spans, cursor = [], 0
  for _,h,w in grids:
    try: start = tokens.index(IMAGE_TOKEN_ID, cursor)
    except ValueError: raise ImageInputError("tokenizer does not support image markers") from None
    count = h*w//4
    if start == 0 or start+count >= len(tokens) or tokens[start-1] != VISION_START_ID or tokens[start+count] != VISION_END_ID:
      raise ImageInputError("invalid vision delimiters")
    spans.append((start,count))
    cursor = start+count
  grid_thw, image_spans = tuple(grids), tuple(spans)
  positions, delta = image_positions(tokens, grid_thw, image_spans)
  return PreparedPrompt(tokens, Tensor(np.concatenate(patches), device=device).contiguous().realize(), grid_thw, image_spans,
                        Tensor(positions, device=device).contiguous().realize(), delta, rendered.rstrip().endswith("<think>"))

def embed_prompt(model:Transformer, vision:QwenVision, prompt:PreparedPrompt) -> Tensor:
  if prompt.pixel_values is None: raise ImageInputError("prompt has no image pixels")
  device = model.token_embd.weight.device
  if prompt.pixel_values.device != device: raise ImageInputError("prompt pixels must be on the decoder device")
  features = vision(prompt.pixel_values, prompt.grid_thw)
  if features.shape != (sum(count for _,count in prompt.image_spans),model.token_embd.weight.shape[-1]):
    raise ImageInputError("vision features do not match decoder embedding spans")
  x = model.token_embd(Tensor([prompt.tokens], device=device)).float()
  parts, cursor, feature_start = [], 0, 0
  for start,count in prompt.image_spans:
    if start > cursor: parts.append(x[:,cursor:start])
    parts.append(features[feature_start:feature_start+count].cast(x.dtype).unsqueeze(0))
    cursor, feature_start = start+count, feature_start+count
  if cursor < len(prompt.tokens): parts.append(x[:,cursor:])
  return parts[0].cat(*parts[1:], dim=1).contiguous().realize()

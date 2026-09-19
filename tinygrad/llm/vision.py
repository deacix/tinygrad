from __future__ import annotations
import hashlib, json, math
from pathlib import Path
from dataclasses import dataclass, fields
from tinygrad import Tensor, nn, dtypes

@dataclass(frozen=True)
class VisionConfig:
  depth:int = 27
  hidden_size:int = 1152
  intermediate_size:int = 4304
  num_heads:int = 16
  in_channels:int = 3
  patch_size:int = 16
  temporal_patch_size:int = 2
  spatial_merge_size:int = 2
  out_hidden_size:int = 5120
  num_position_embeddings:int = 2304

  @staticmethod
  def from_dict(config:dict) -> VisionConfig:
    if config.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh" or config.get("deepstack_visual_indexes", []):
      raise ValueError("unsupported Qwen vision architecture")
    return VisionConfig(**{f.name:config[f.name] for f in fields(VisionConfig) if f.name in config})

  def __post_init__(self):
    if any(type(v:=getattr(self, f.name)) is not int or v <= 0 for f in fields(self)): raise ValueError("invalid vision dimensions")
    if (self.in_channels, self.patch_size, self.temporal_patch_size, self.spatial_merge_size) != (3, 16, 2, 2):
      raise ValueError("unsupported Qwen patch layout")
    if self.hidden_size % (4*self.num_heads) or math.isqrt(self.num_position_embeddings)**2 != self.num_position_embeddings:
      raise ValueError("invalid vision head or position dimensions")

class PatchEmbed:
  def __init__(self, config:VisionConfig):
    self.proj = nn.Conv2d(config.in_channels, config.hidden_size, (2, 16, 16), stride=(2, 16, 16), bias=True)
  def __call__(self, x:Tensor) -> Tensor:
    return x.cast(self.proj.weight.dtype).linear(self.proj.weight.reshape(self.proj.weight.shape[0], -1).T, self.proj.bias)

class VisionMLP:
  def __init__(self, dim:int, hidden:int, out:int, exact:bool=False):
    self.linear_fc1, self.linear_fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, out)
    self.exact = exact
  def __call__(self, x:Tensor) -> Tensor: return self.linear_fc2(self.linear_fc1(x).gelu("none" if self.exact else "tanh"))

class VisionAttention:
  def __init__(self, config:VisionConfig):
    self.num_heads, self.head_dim = config.num_heads, config.hidden_size//config.num_heads
    self.qkv, self.proj = nn.Linear(config.hidden_size, 3*config.hidden_size), nn.Linear(config.hidden_size, config.hidden_size)
  def __call__(self, x:Tensor, grids:tuple[tuple[int,int,int],...], rotary:Tensor) -> Tensor:
    qkv = self.qkv(x).reshape(x.shape[0], 3, self.num_heads, self.head_dim)
    q, k, v = (qkv[:, i] for i in range(3))
    freqs = rotary.cat(rotary, dim=-1).unsqueeze(1)
    def rotate(z:Tensor):
      first, last = z.float().chunk(2, dim=-1)
      return (z.float()*freqs.cos() + (-last).cat(first, dim=-1)*freqs.sin()).cast(z.dtype)
    q, k = rotate(q), rotate(k)
    out, start = [], 0
    for _, h, w in grids:
      end = start+h*w
      qi, ki, vi = (t[start:end].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
      # Explicit FP32 softmax matches reference eager attention for low precision too.
      scores = (qi @ ki.transpose(-1, -2)) * self.head_dim**-0.5
      out.append((scores.float().softmax(-1).cast(q.dtype) @ vi).squeeze(0).transpose(0, 1).reshape(h*w, -1))
      start = end
    return self.proj(out[0].cat(*out[1:], dim=0))

class VisionLayerNorm(nn.LayerNorm):
  def __call__(self, x:Tensor) -> Tensor:
    assert self.weight is not None and self.bias is not None
    return (x.float().layernorm(axis=self.axis, eps=self.eps)*self.weight.float()+self.bias.float()).cast(x.dtype)

class VisionBlock:
  def __init__(self, config:VisionConfig):
    self.norm1, self.norm2 = VisionLayerNorm(config.hidden_size, eps=1e-6), VisionLayerNorm(config.hidden_size, eps=1e-6)
    self.attn = VisionAttention(config)
    self.mlp = VisionMLP(config.hidden_size, config.intermediate_size, config.hidden_size)
  def __call__(self, x:Tensor, grids:tuple[tuple[int,int,int],...], rotary:Tensor) -> Tensor:
    x = x + self.attn(self.norm1(x), grids, rotary)
    return x + self.mlp(self.norm2(x))

class VisionMerger(VisionMLP):
  def __init__(self, config:VisionConfig):
    super().__init__(config.hidden_size*4, config.hidden_size*4, config.out_hidden_size, exact=True)
    self.norm = VisionLayerNorm(config.hidden_size, eps=1e-6)
  def __call__(self, x:Tensor) -> Tensor: return super().__call__(self.norm(x).reshape(x.shape[0]//4, -1))

class QwenVision:
  def __init__(self, config:VisionConfig):
    self.config = config
    self.patch_embed, self.pos_embed = PatchEmbed(config), nn.Embedding(config.num_position_embeddings, config.hidden_size)
    self.blocks = [VisionBlock(config) for _ in range(config.depth)]
    self.merger = VisionMerger(config)

  def positions(self, grids:tuple[tuple[int,int,int],...]) -> tuple[Tensor, Tensor]:
    device = self.pos_embed.weight.device
    side, dim = math.isqrt(self.config.num_position_embeddings), self.config.hidden_size//self.config.num_heads
    positions, frequencies = [], []
    for _, h, w in grids:
      row = Tensor.arange(h).to(device).reshape(h, 1).expand(h, w)
      col = Tensor.arange(w).to(device).reshape(1, w).expand(h, w)
      def block_order(z:Tensor): return z.reshape(h//2, 2, w//2, 2).permute(0, 2, 1, 3).reshape(-1)
      row, col = block_order(row), block_order(col)
      ry, rx = row.float()*(side-1)/(h-1), col.float()*(side-1)/(w-1)
      iy, ix = ry.cast(dtypes.int32), rx.cast(dtypes.int32)
      dy, dx = ry-iy, rx-ix
      jy, jx = (iy+1).minimum(side-1), (ix+1).minimum(side-1)
      weighted = [self.pos_embed(a*side+b)*weight.cast(self.pos_embed.weight.dtype).unsqueeze(-1)
                  for a,b,weight in ((iy,ix,(1-dy)*(1-dx)), (iy,jx,(1-dy)*dx), (jy,ix,dy*(1-dx)), (jy,jx,dy*dx))]
      positions.append(weighted[0]+weighted[1]+weighted[2]+weighted[3])
      inv = 1 / (10000 ** (Tensor.arange(dim//4).to(device).float()*4/dim))
      frequencies.append((row.unsqueeze(-1)*inv).cat(col.unsqueeze(-1)*inv, dim=-1))
    return positions[0].cat(*positions[1:]), frequencies[0].cat(*frequencies[1:])

  def __call__(self, pixel_values:Tensor, grid_thw:tuple[tuple[int,int,int],...]) -> Tensor:
    if not grid_thw or any(len(g) != 3 or any(type(n) is not int for n in g) or g[0] != 1 or min(g[1:]) <= 0 or
                           g[1]%2 or g[2]%2 for g in grid_thw): raise ValueError("invalid still-image grids")
    if pixel_values.shape != (sum(h*w for _,h,w in grid_thw), 1536): raise ValueError("patch count does not match image grids")
    if pixel_values.device != self.pos_embed.weight.device: raise ValueError("vision pixels and weights must share a device")
    positions, rotary = self.positions(grid_thw)
    x = self.patch_embed(pixel_values) + positions
    for block in self.blocks: x = block(x, grid_thw, rotary).realize()
    return self.merger(x).realize()

# Explicit allowed pairing. Normal files use Git's framed blob digest; LFS files use payload SHA256.
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
GGUF_REVISION = "b62a80264f8b0c1bb849ee1c9c487415ebeca194"
VISION_SHARD = "model-00001-of-00018.safetensors"
BUNDLE_FILES = {
  "config.json": (4312, "706cebd746c4b6f2b1d1f892630867acfdfd3df8"),
  "preprocessor_config.json": (390, "2ea84a437d448ff71b08df68fdd949d5cc4ebb64"),
  "chat_template.jinja": (8952, "c0c686f9c38d70d179fb7b5f5aa7530bc913dda3"),
  "model.safetensors.index.json": (112216, "da35e3c564457dface7d138f0b6cac284ff8958c"),
  VISION_SHARD: (3966730552, "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"),
}
GGUF_DIGEST = (15705861088, "9fd40d7036f5e0918e20aaeebf11468fafd06bb53d4d980eef6bb7e4e4ace666")

def _verify_file(path:Path, spec:tuple[int,str]) -> None:
  size, digest = spec
  if not path.is_file() or path.stat().st_size != size: raise ValueError(f"missing or incorrect bundle file: {path.name}")
  hasher = hashlib.sha1(f"blob {size}\0".encode()) if len(digest) == 40 else hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024*1024): hasher.update(chunk)
  if hasher.hexdigest() != digest: raise ValueError(f"bundle digest mismatch: {path.name}")

def _bundle_paths(model_dir:Path) -> dict[str,Path]:
  root = model_dir.resolve(strict=True)
  paths = {}
  for name in BUNDLE_FILES:
    path = (root/name).resolve(strict=True)
    if not path.is_relative_to(root): raise ValueError("bundle symlink escapes model directory")
    paths[name] = path
  return paths

def _visual_keys() -> set[str]:
  keys = {f"blocks.{i}.{module}.{param}" for i in range(27)
          for module in ("attn.proj", "attn.qkv", "mlp.linear_fc1", "mlp.linear_fc2", "norm1", "norm2") for param in ("weight", "bias")}
  return keys | {f"{module}.{param}" for module in ("merger.linear_fc1", "merger.linear_fc2", "merger.norm", "patch_embed.proj")
                 for param in ("weight", "bias")} | {"pos_embed.weight"}

def _validate_visual_index(index:dict) -> None:
  actual = {k.removeprefix("model.visual."):v for k,v in index["weight_map"].items() if k.startswith("model.visual.")}
  if set(actual) != _visual_keys() or set(actual.values()) != {VISION_SHARD}: raise ValueError("unsupported visual tensor index")

def _validate_vision_files(model_dir:Path) -> dict[str,Path]:
  paths = _bundle_paths(model_dir)
  for name, spec in BUNDLE_FILES.items(): _verify_file(paths[name], spec)
  cfg = json.loads(paths["config.json"].read_bytes())
  if (cfg.get("architectures") != ["Qwen3_5ForConditionalGeneration"] or cfg.get("model_type") != "qwen3_5" or
      cfg.get("language_model_only") is not False or VisionConfig.from_dict(cfg["vision_config"]) != VisionConfig() or
      cfg["text_config"]["hidden_size"] != 5120): raise ValueError("unsupported Qwen vision bundle configuration")
  _validate_visual_index(json.loads(paths["model.safetensors.index.json"].read_bytes()))
  return paths

def validate_vision_bundle(model_dir:Path, gguf_path:Path) -> None:
  _verify_file(gguf_path.resolve(strict=True), GGUF_DIGEST)
  _validate_vision_files(model_dir)

def validate_vision_metadata(kv:dict) -> None:
  expected = {"general.architecture":"qwen35", "qwen35.block_count":64, "qwen35.embedding_length":5120,
              "qwen35.attention.head_count":24, "qwen35.attention.head_count_kv":4, "qwen35.attention.key_length":256,
              "qwen35.rope.dimension_count":64}
  if any(kv.get(k) != v for k,v in expected.items()): raise ValueError("GGUF is not the supported Qwen3.8 decoder")
  tokens = kv.get("tokenizer.ggml.tokens", [])
  for token, index in (("<|vision_start|>",248053), ("<|vision_end|>",248054), ("<|image_pad|>",248056),
                       ("<|video_pad|>",248057), ("<|im_end|>",248046), ("<|endoftext|>",248044)):
    if len(tokens) <= index or tokens[index] != token: raise ValueError("GGUF vision token mismatch")

def load_vision(model_dir:Path, *, device:str) -> QwenVision:
  paths = _validate_vision_files(model_dir)
  tensors = nn.state.safe_load(paths[VISION_SHARD])
  selected = {k.removeprefix("model.visual."):v for k,v in tensors.items() if k.startswith("model.visual.")}
  if set(selected) != _visual_keys(): raise ValueError("unexpected visual checkpoint keys")
  model = QwenVision(VisionConfig())
  _load_visual_tensors(model, selected, device)
  return model

def _load_visual_tensors(model:QwenVision, selected:dict[str,Tensor], device:str) -> None:
  expected = nn.state.get_state_dict(model)
  if set(selected) != set(expected): raise ValueError("unexpected visual checkpoint keys")
  for name, tensor in selected.items():
    if tensor.shape != expected[name].shape or tensor.dtype != dtypes.bfloat16: raise ValueError(f"invalid visual tensor: {name}")
  # Validate every tensor before transferring any; language/MTP weights never reach the device.
  for name, target in expected.items(): target.replace(selected[name].to(device)).realize()

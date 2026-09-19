from __future__ import annotations
import enum, functools, itertools, math, pathlib, threading
from contextlib import contextmanager
from collections.abc import Callable
from dataclasses import dataclass, replace
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes
from tinygrad.llm.kernels.amd import Linear, gated_delta_prefill, flash_attention, amd_custom_kernels_supported
from tinygrad.llm.gguf import gguf_load, GGUFIndex, index_gguf, load_gguf_tensor
from tinygrad.llm.placement import LayerPlacement, _placement_manifest
from tinygrad.uop.ops import resolve
from tinygrad.device import Device


def _transfer_activation(x:Tensor, destination:str, mode:str) -> Tensor:
  """Serialized concrete FP32/int32 transport, outside capture. Native is not a P2P guarantee."""
  if x.dtype not in (dtypes.float32, dtypes.int32) or any(type(n) is not int for n in x.shape) or not isinstance(x.device, str):
    raise ValueError('placed transport requires concrete FP32 or int32 storage on one device')
  if mode not in ('host', 'native'): raise ValueError('unknown placed transfer mode')
  destination, source_device = Device.canonicalize(destination), x.device
  x = x.contiguous().realize()
  if source_device == destination: return x
  if mode == 'native':
    out = x.to(destination).realize()
    Device[source_device].synchronize()
    Device[destination].synchronize()
    return out
  Device[source_device].synchronize()
  # bytes owns the copyout; source owns _frompy's separate buffer until the destination is finished.
  source = Tensor(bytes(x.data()), dtype=dtypes.uint8, device='PYTHON').realize()
  out = source.to(destination).realize()
  Device[destination].synchronize()
  return out.bitcast(x.dtype).reshape(x.shape)

class ExpertGating(enum.IntEnum):
  SOFTMAX = 1
  SIGMOID = 2
  SOFTMAX_WEIGHT = 3  # softmax over the top-k selected logits
  SQRT_SOFTPLUS = 4

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device)

class ExpertWeights:
  """Like Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def multimodal_freqs_cis(position_ids:Tensor, dim:int, theta:float, sections:tuple[int, ...]) -> Tensor:
  # Interleaved T/H/W frequencies: H occupies 1,4,... and W 2,5,...; remaining frequencies use T.
  if len(sections) != 3 or any(s <= 0 for s in sections) or sum(sections)*2 != dim:
    raise ValueError("mRoPE sections must contain three positive counts summing to half the rotary dimension")
  idx = Tensor.arange(dim//2).to(position_ids.device)
  positions = position_ids.float().squeeze(1).unsqueeze(-1)
  freqs = ((idx % 3 == 1) & (idx < 3*sections[1])).where(positions[1],
    ((idx % 3 == 2) & (idx < 3*sections[2])).where(positions[2], positions[0]))
  angles = freqs * (1.0 / (theta ** (idx.float() * 2 / dim)))
  return angles.cos().cat(angles.sin(), dim=-1)

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int
  kda: bool = False

@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  norm_eps: float
  vocab_size: int
  head_dim: int
  rope_theta: float
  rope_dim: int
  v_head_dim: int
  max_context: int = 0
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  expert_gating_func: ExpertGating = ExpertGating.SOFTMAX
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  ssm_layers: tuple[bool, ...] = ()
  attn_output_gate: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False
  rope_sections: tuple[int, ...] = (11, 11, 10)

def _llama_parameter_shapes(config:TransformerConfig) -> dict[str, tuple[int, ...]]:
  d, h, k = config.dim, config.hidden_dim, config.n_kv_heads*config.head_dim
  shapes = {'token_embd.weight':(config.vocab_size,d), 'output_norm.weight':(d,), 'output.weight':(config.vocab_size,d)}
  block = {'attn_norm':(d,), 'ffn_norm':(d,), 'attn_q':(d,d), 'attn_k':(k,d), 'attn_v':(k,d), 'attn_output':(d,d),
           'ffn_gate':(h,d), 'ffn_up':(h,d), 'ffn_down':(d,h)}
  shapes.update((f'blk.{i}.{name}.weight', shape) for i in range(config.num_blocks) for name,shape in block.items())
  return shapes


def preflight_placement(index:GGUFIndex, placement:LayerPlacement, *, max_context:int|None=None,
                        realize:bool=False) -> tuple[TransformerConfig, LayerPlacement]:
  """Validate the entire dense-Llama configuration/manifest without Tensors, payload reads or device opening.

  Consumes the checked local-path index. Returns config with effective context and canonical placement, suitable
  for metadata-only accounting. Unknown Llama computation fields fail closed; descriptive/tokenizer metadata is retained.
  """
  if realize: raise ValueError('explicit placement requires REALIZE=0')
  kv = index.kv
  if kv.get('general.architecture') != 'llama': raise ValueError('explicit placement supports only dense llama')
  # Omitted by some legacy writers. Explicit versions must describe the layouts decoded here (GGML_QUANT_VERSION=2).
  if 'general.quantization_version' in kv and (type(kv['general.quantization_version']) is not int or kv['general.quantization_version'] != 2):
    raise ValueError('unsupported GGUF quantization version for placement')
  def positive_int(key:str, default=None) -> int:
    value = kv.get(key, default)
    if type(value) is not int or value <= 0: raise ValueError(f'{key} must be a positive integer')
    return value
  def positive_float(key:str) -> float:
    value = kv.get(key, 0)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
      raise ValueError(f'{key} must be finite and positive')
    return float(value)
  blocks = positive_int('llama.block_count')
  placement = placement.validate(blocks)
  dim, hidden = positive_int('llama.embedding_length'), positive_int('llama.feed_forward_length')
  heads = positive_int('llama.attention.head_count')
  kv_heads = positive_int('llama.attention.head_count_kv', heads)
  head_dim = positive_int('llama.attention.key_length', dim//heads)
  rope_dim = positive_int('llama.rope.dimension_count', head_dim)
  value_dim = positive_int('llama.attention.value_length', head_dim)
  if dim != heads*head_dim or head_dim != rope_dim or head_dim != value_dim or rope_dim % 2 or heads % kv_heads:
    raise ValueError('placement requires full even-head RoPE, dim=heads*head_dim, equal Q/K/V head dimensions and divisible GQA')
  context = positive_int('llama.context_length')
  if max_context is not None:
    if type(max_context) is not int or max_context <= 0: raise ValueError('max_context must be a positive integer')
    context = min(context, max_context)
  if placement.chunk_size > context: raise ValueError('chunk size must not exceed effective context')
  tokens = kv.get('tokenizer.ggml.tokens')
  if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) for t in tokens):
    raise ValueError('tokenizer.ggml.tokens must be a nonempty string array')
  if positive_int('llama.vocab_size', len(tokens)) != len(tokens): raise ValueError('vocabulary size disagrees with tokenizer')

  # Reference dense computation only. Neutral settings are allowed explicitly, never through truthiness/coercion.
  neutral = {'rope.scaling.type':'none', 'rope.scaling.factor':1.0, 'rope.scale_linear':1.0, 'rope.scaling.finetuned':False,
             'tensor_data_layout':'reference', 'expert_count':0, 'expert_used_count':0, 'nextn_predict_layers':0,
             'use_parallel_residual':False, 'attention.causal':True}
  supported = {'block_count', 'embedding_length', 'feed_forward_length', 'context_length', 'vocab_size',
               'attention.head_count', 'attention.head_count_kv', 'attention.key_length', 'attention.value_length',
               'attention.layer_norm_rms_epsilon', 'rope.dimension_count', 'rope.freq_base'}
  for key,value in kv.items():
    if key in ('general.tensor_data_layout', 'llama.tensor_data_layout') and value != 'reference':
      raise ValueError(f'unsupported placement metadata: {key}')
    if not key.startswith('llama.'): continue
    field = key[len('llama.'):]
    if field in supported: continue
    if field not in neutral: raise ValueError(f'unsupported placement computation metadata: {key}')
    expected = neutral[field]
    if (type(value) is not type(expected) and not (type(expected) is float and type(value) is int)) or value != expected:
      raise ValueError(f'unsupported placement metadata value: {key}')
  config = TransformerConfig(blocks, dim, hidden, heads, kv_heads, positive_float('llama.attention.layer_norm_rms_epsilon'), len(tokens),
                             head_dim, positive_float('llama.rope.freq_base'), rope_dim, value_dim, max_context=context)
  # Bound manifest expansion by the already bounded descriptor inventory, not a potentially enormous metadata count.
  if not 2 + 9*blocks <= len(index.tensors) <= 4 + 9*blocks: raise ValueError('placed tensor count disagrees with block count')
  shapes = _llama_parameter_shapes(config)
  names = set()
  for info in index.tensors:
    if info.name in names: raise ValueError(f'duplicate placed tensor: {info.name}')
    names.add(info.name)
    if info.ggml_type not in (0,1,30,2,8,12,13,14): raise ValueError(f'unsupported placed GGML type for {info.name}: {info.ggml_type}')
    shape = (rope_dim//2,) if info.name == 'rope_freqs.weight' else shapes.get(info.name)
    if shape is None: raise ValueError(f'unsupported placed tensor: {info.name}')
    if info.shape != shape: raise ValueError(f'placed tensor shape mismatch: {info.name}: {info.shape} != {shape}')
  missing = shapes.keys() - names - {'output.weight'}
  if missing: raise ValueError(f'missing placed tensors: {sorted(missing)}')
  return config, placement


class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=False)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      logits = self.ffn_gate_inp(x)
      bias = self.exp_probs_b["bias"] if hasattr(self, 'exp_probs_b') else None
      gating, normalize_topk = self.config.expert_gating_func, self.config.norm_topk_prob
      # fast path: without selection bias, normalized SOFTMAX is equivalent to SOFTMAX_WEIGHT
      if gating == ExpertGating.SOFTMAX and bias is None and normalize_topk:
        gating, normalize_topk = ExpertGating.SOFTMAX_WEIGHT, False
      if   gating == ExpertGating.SOFTMAX_WEIGHT: scores = logits
      elif gating == ExpertGating.SOFTMAX:        scores = logits.softmax(-1)
      elif gating == ExpertGating.SIGMOID:        scores = logits.sigmoid()
      elif gating == ExpertGating.SQRT_SOFTPLUS:  scores = logits.softplus().sqrt()

      _, sel = pairwise_topk(scores if bias is None else scores + bias, self.config.num_experts_per_tok)
      probs = scores.gather(-1, sel)
      # SOFTMAX_WEIGHT applies softmax after top-k selection
      if gating == ExpertGating.SOFTMAX_WEIGHT: probs = probs.softmax(-1)
      if normalize_topk: probs = probs / probs.sum(axis=-1, keepdim=True)
      probs = probs * self.config.routed_scaling_factor
      x_down = self.ffn_down_exps(sel, (self.ffn_gate_exps(sel, h).silu() * self.ffn_up_exps(sel, h)).contiguous())  # (B, T, k, D)
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(self.ffn_gate_shexp(x).silu().contiguous() * self.ffn_up_shexp(x))
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    # TODO: remove the need for this contiguous
    return self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp, position_ids:Tensor|None=None):
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp, position_ids:Tensor|None=None):
      attn = self._attention(self.attn_norm(x), start_pos, position_ids) if isinstance(self, TransformerBlock) and position_ids is not None \
        else self._attention(self.attn_norm(x), start_pos)
      h = x + attn
      return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
    return _run(x, start_pos) if position_ids is None else _run(x, start_pos, position_ids)

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = Linear(config.head_dim * config.n_heads, config.dim, bias=False)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)

  def _attention(self, x:Tensor, start_pos:int|UOp, position_ids:Tensor|None=None) -> Tensor:
    q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    if self.config.attn_output_gate:
      qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, self.config.n_heads * self.config.head_dim)
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    freqs = self.freqs_cis[start_pos:start_pos+T] if position_ids is None else \
      multimodal_freqs_cis(position_ids, self.config.rope_dim, self.config.rope_theta, self.config.rope_sections)
    q = apply_rope(q[..., :self.config.rope_dim], freqs).cat(q[..., self.config.rope_dim:], dim=-1)
    k = apply_rope(k[..., :self.config.rope_dim], freqs).cat(k[..., self.config.rope_dim:], dim=-1)

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    store = self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).cast(self.cache_kv.dtype).uop)
    assigned_kv = Tensor(self.cache_kv.uop.after(store))
    # on RDNA3, hybrid models use custom flash attention kernels on the KV cache
    if amd_custom_kernels_supported(x.device) and self.config.ssm is not None:
      attn = flash_attention(q, assigned_kv, start_pos+T)
      attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
      return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))
    k = assigned_kv[0, :, :, 0:start_pos+T, :]
    v = assigned_kv[1, :, :, 0:start_pos+T, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # zeroed so the flash kernels can safely read whole tiles past the valid region (masked lanes multiply by 0)
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim,
                                   dtype=dtypes.half, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    if not self.config.ssm or not self.config.ssm.kda: q_rope = apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T])
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(q_rope, dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2)
    if not self.config.ssm or not self.config.ssm.kda: k_rope = apply_rope(k_rope, self.freqs_cis[start_pos:start_pos+T])

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.uop)))[:, :, 0:start_pos+T, :]
    v = k[..., :self.config.kv_lora_rank]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv = Linear(config.dim, self.conv_channels, bias=False)
    if ssm.kda:
      self.ssm_g_a, self.ssm_g_b = Linear(config.dim, self.head_v_dim, bias=False), Linear(self.head_v_dim, ssm.inner_size, bias=False)
      self.ssm_f_a, self.ssm_f_b = Linear(config.dim, self.head_k_dim, bias=False), Linear(self.head_k_dim, ssm.inner_size, bias=False)
    else:
      self.attn_gate = Linear(config.dim, ssm.inner_size, bias=False)
      self.ssm_alpha = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(ssm.inner_size if ssm.kda else self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads, 1) if ssm.kda else Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    # bind ints to a variable so the reset flag stays a runtime value (it toggles when generation restarts at position 0)
    start_pos = start_pos if isinstance(start_pos, UOp) else UOp.variable("start_pos", 0, self.config.max_context-1).bind(start_pos)
    initial = Tensor(start_pos, device=x.device).eq(0)
    is_kda = hasattr(self, "ssm_g_a")
    symbolic = isinstance(T, UOp)
    T_pad = x.max_shape[1]  # symbolic chunks are padded to their max size: one graph serves every size

    # input processing
    x = x.half()
    out_gate = self.ssm_g_b(self.ssm_g_a(x)) if is_kda else self.attn_gate(x)
    out_gate = out_gate.reshape(B, T, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, T, self.num_v_heads)
    alpha = self.ssm_f_b(self.ssm_f_a(x)) if is_kda else self.ssm_alpha(x)
    log_alpha = ((alpha.float() + self.ssm_dt["bias"]).softplus().reshape(B, T, self.num_v_heads, -1) *
                 self.ssm_a.reshape(self.num_v_heads, -1))

    # qkv conv, conv_state is reset when starting from position 0
    conv_state = initial.where(0, self.conv_state)
    # assemble the conv window in a static-size buffer: [conv_state | qkv rows | zero-pad].
    # padded steps are exact no-ops: beta=0 (delta rule off), log_alpha=0 (decay 1 after exp)
    win = Tensor.zeros(B, self.ssm_conv_kernel-1 + T_pad, self.conv_channels, device=x.device).uop
    win = win.after(win[:, :self.ssm_conv_kernel-1].store(conv_state.cast(win.dtype).uop))
    win = win.after(win[:, self.ssm_conv_kernel-1:self.ssm_conv_kernel-1+T].store(self.attn_qkv(x).cast(win.dtype).uop))
    conv_window = Tensor(win)
    # the last conv_kernel-1 columns of the window become the next conv state
    conv_state_store = self.conv_state.uop.store(conv_window[:, T:T+self.ssm_conv_kernel-1].cast(self.conv_state.dtype).uop)

    conv_out = functools.reduce(lambda a,b: a+b,
      (conv_window[:, i:i+T_pad] * self.ssm_conv1d["weight"][:, i] for i in range(self.ssm_conv_kernel))).silu()
    if symbolic:
      out_gate = out_gate.pad_to((B, T_pad, self.num_v_heads, self.head_v_dim))
      beta, log_alpha = beta.pad_to((B, T_pad, self.num_v_heads)), log_alpha.pad_to((B, T_pad, *log_alpha.shape[2:]))
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    q, k = (z.reshape(B, T_pad, self.num_k_heads, self.head_k_dim) for z in (q, k))
    # GDN uses FLA/HF l2norm: epsilon inside rsqrt, not normalize's clamp after sqrt. Keep KDA's original normalization.
    q, k = ((z.normalize(dim=-1, eps=1e-12) if is_kda else z * (z.square().sum(-1, keepdim=True) + 1e-6).rsqrt())
            .repeat(1, 1, self.num_v_heads//self.num_k_heads, 1) for z in (q, k))
    v = v.reshape(B, T_pad, self.num_v_heads, self.head_v_dim)
    # layout the per-step operands to broadcast against the (B, H, V, K) state
    q, k, v, beta = (z.transpose(1, 2).float() for z in (q, k, v, beta))
    q = q * self.head_k_dim**-0.5
    alpha = log_alpha.transpose(1, 2).exp()  # per-channel decay for kda, per-head otherwise (B, H, T, V|1)

    # recurrent: scan over the (padded) tokens, updating the recurrent state. collect the per-step outputs
    state = Tensor(self.recurrent_state.uop.after(conv_state_store))  # carry the conv write into this graph
    if self.head_k_dim % 32 == 0 and self.head_v_dim % 4 == 0 and amd_custom_kernels_supported(x.device):
      # one fused kernel for the whole scan; it resets and updates the recurrent state in place (RDNA3)
      core = gated_delta_prefill(q, k, v, beta, alpha, state, Tensor(start_pos, device=x.device)).transpose(1, 2)
    else:
      q, k, v, beta = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
      alpha = alpha.unsqueeze(-1)
      state = initial.where(0, state.float())
      outs = []
      for t in range(T_pad):
        s1 = state * alpha[:, :, t]  # decay the state
        delta = (v[:, :, t] - (s1*k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]  # the delta rule update
        state = s1 + delta * k[:, :, t]
        outs.append((state * q[:, :, t]).sum(-1))

      # store the updated recurrent state in place, then read the stacked outputs after the write
      state_store = self.recurrent_state.uop.store(state.cast(self.recurrent_state.dtype).uop)
      core = Tensor(outs[0].stack(*outs[1:], dim=1).contiguous().uop.after(state_store))

    # output; undo the padding before the output projection
    z = (self.ssm_norm(core) * (out_gate.sigmoid() if is_kda else out_gate.silu())).cast(x.dtype).contiguous()
    if symbolic: z = z[:, :T]
    return self.ssm_out(z.reshape(B, T, -1))

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim, device=x.device).clone()

def _shape_only_llama(config:TransformerConfig) -> tuple[list[FFNBlock], nn.Embedding, nn.RMSNorm, Linear]:
  # nn.Linear/Embedding call Tensor.rand, which allocates RNG seed/counter storage even without weight realization.
  # Bypass those initializers locally, not via global monkeypatches. All placeholders are unrealized host buffers.
  def norm(dim:int) -> nn.RMSNorm:
    layer = nn.RMSNorm(dim, config.norm_eps, elementwise_affine=False)
    layer.weight = Tensor.empty(dim, device='PYTHON')
    return layer
  def linear(out_features:int, in_features:int) -> Linear:
    layer = Linear.__new__(Linear)
    layer.in_features, layer.out_features, layer.bias = in_features, out_features, None
    layer.use_custom_quant = False
    layer.weight = Tensor.empty(out_features, in_features, device='PYTHON')
    return layer
  blocks:list[FFNBlock] = []
  for _ in range(config.num_blocks):
    block = TransformerBlock.__new__(TransformerBlock)
    block.config = config
    blocks.append(block)
  for name,shape in _llama_parameter_shapes(config).items():
    if name.startswith('blk.'):
      _, index, component, _ = name.split('.')
      setattr(blocks[int(index)], component, norm(shape[0]) if len(shape) == 1 else linear(*shape))
  embedding = nn.Embedding.__new__(nn.Embedding)
  embedding.weight = Tensor.empty(config.vocab_size, config.dim, device='PYTHON')
  return blocks, embedding, norm(config.dim), linear(config.vocab_size, config.dim)


class PlacedInferenceError(RuntimeError):
  """Backend failure. Metadata only: never include backend messages, prompts or payloads."""
  def __init__(self, stage:int|None, device:str, operation:str):
    self.stage, self.device, self.operation = stage, device, operation
    super().__init__(f'placed inference failed: stage={stage}, device={device}, operation={operation}; reload required')


class PlacedModelUnavailableError(RuntimeError): pass
class PlacedModelBusyError(RuntimeError): pass


class _LayerStage:
  __slots__ = ('device', '_forward', 'single_jit', 'chunk_jit', 'chunk_size')

  def __init__(self, device:str, blocks:list[FFNBlock], embedding:nn.Embedding|None,
               norm:nn.RMSNorm|None, output:Linear|None, chunk_size:int):
    self.device, self.chunk_size = device, chunk_size
    # No parent pointer or second registered weight tree: closures live behind the slotted executor.
    def forward(value:Tensor, start_pos:UOp) -> Tensor:
      x = embedding(value).float() if embedding is not None else value
      for block in blocks: x = block(x, start_pos)
      if norm is not None and output is not None: x = output(norm(x[:, -1:]))[:, -1, :]
      return x.float().contiguous().realize()
    self._forward = forward
    self.single_jit = TinyJit(forward)
    self.chunk_jit = TinyJit(forward) if chunk_size > 1 else None

  def run(self, value:Tensor, start_pos:UOp, *, jit:bool=True) -> Tensor:
    runner = self.single_jit if value.shape[1] == 1 else self.chunk_jit if value.shape[1] == self.chunk_size else None
    return runner(value, start_pos) if jit and runner is not None else self._forward(value, start_pos)


class _PlacedExecutor:
  # Generic get_state_dict must not traverse stage closures or CapturedJit.ret.
  __slots__ = ('stages', 'placement', 'config', 'state', 'lock', 'owner')

  def __init__(self, model:Transformer, config:TransformerConfig, placement:LayerPlacement):
    self.placement, self.config = placement, config
    self.state, self.lock = 'ready', threading.Lock()
    self.owner:object|None = None
    stages, start = [], 0
    for i,(device,count) in enumerate(zip(placement.devices, placement.layer_counts)):
      Device[device]  # device opening is forbidden inside function bodies
      for block in model.blk[start:start+count]:
        assert isinstance(block, TransformerBlock)
        block.cache_kv = Tensor.zeros(2, 1, config.n_kv_heads, config.max_context, config.head_dim,
                                      dtype=dtypes.half, device=device).contiguous().realize()
        block.freqs_cis = precompute_freqs_cis(config.rope_dim, config.max_context, config.rope_theta, device).float().contiguous().realize()
      Device[device].synchronize()
      last = i == len(placement.devices)-1
      stages.append(_LayerStage(device, model.blk[start:start+count], model.token_embd if i == 0 else None,
                                model.output_norm if last else None, model.output if last else None, placement.chunk_size))
      start += count
    self.stages = tuple(stages)


class Transformer:
  placement: LayerPlacement  # present only for explicit placed loading
  _placed: _PlacedExecutor

  def __setattr__(self, name, value):
    if name in ('placement', 'max_context') and hasattr(self, '_placed'):
      raise ValueError('placed configuration is immutable; reload required')
    super().__setattr__(name, value)

  def __init__(self, config:TransformerConfig, *, _shape_only:bool=False):
    self.blk:list[FFNBlock]
    if _shape_only:
      self.blk, self.token_embd, self.output_norm, self.output = _shape_only_llama(config)
    else:
      dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0,
                             hidden_dim=config.dense_hidden_dim or config.hidden_dim)
      if config.ssm: config = replace(config, qk_norm=config.head_dim)
      block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
      self.blk = [GatedDeltaNetBlock(dense_config if i < config.leading_dense_blocks else config, config.ssm)
                                 if config.ssm and config.ssm_layers[i] else
                                 block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
      self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
      self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
      self.output = Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)
    self.multimodal_rollout_jit = TinyJit(self._multimodal_decode)
    self._generation_epoch = 0
    self._multimodal_active = False

  def forward_embeddings(self, x:Tensor, start_pos:int|UOp, position_ids:Tensor|None=None) -> Tensor:
    if hasattr(self, 'placement'): raise ValueError('placement does not support public embeddings')
    if position_ids is not None:
      if x.ndim != 3 or x.shape[0] != 1 or x.shape[2] != self.token_embd.weight.shape[1] or x.dtype != dtypes.float32:
        raise ValueError("multimodal embeddings must be float32 [1, L, decoder_dim]")
      physical = start_pos if isinstance(start_pos, int) else start_pos.unbind()[1]
      if not 0 <= physical < physical+x.shape[1] <= self.max_context: raise ValueError("physical positions must fit the context")
      if position_ids.shape != (3, 1, x.shape[1]) or position_ids.dtype != dtypes.int32:
        raise ValueError("position_ids must be int32 [3, 1, L]")
      if position_ids.device != x.device or x.device != self.token_embd.weight.device:
        raise ValueError("embeddings and positions must be on the decoder device")
      if any(isinstance(b, MLATransformerBlock) for b in self.blk): raise ValueError("mRoPE is not supported for MLA")
    for block in self.blk:
      x = block(x, start_pos, position_ids) if position_ids is not None and isinstance(block, TransformerBlock) else block(x, start_pos)
    # only run the output projection on the last token
    return self.output(self.output_norm(x[:, -1:]))[:, -1, :]

  @staticmethod
  def _sample(logits:Tensor, temperature:Tensor) -> Tensor:
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def check_available(self):
    """Synchronous availability check, including before a caller sends streaming headers."""
    if hasattr(self, '_placed'):
      if self._placed.state == 'failed': raise PlacedModelUnavailableError('placed model failed; reload required')
      if self._placed.state != 'ready' or self._placed.lock.locked(): raise PlacedModelBusyError('placed model is in use')

  @contextmanager
  def _placed_transaction(self, operation:str, validate:Callable[[], None]|None=None):
    executor = self._placed
    if not executor.lock.acquire(blocking=False): raise PlacedModelBusyError('placed model is in use')
    try:
      if executor.state == 'failed': raise PlacedModelUnavailableError('placed model failed; reload required')
      if executor.state != 'ready': raise PlacedModelBusyError('placed model is in use')
      if self.max_context != executor.config.max_context or any(b.config is not executor.config for b in self.blk):
        raise ValueError('placed configuration changed; reload required')
      # Input readback must share admission's lock, but invalid values must not poison committed caches.
      if validate is not None: validate()
      owner = object()
      executor.owner, executor.state = owner, 'generating'
      try: yield owner
      except BaseException as exc:
        executor.state, self._cached_tokens = 'failed', []
        if isinstance(exc, Exception) and not isinstance(exc, PlacedInferenceError):
          raise PlacedInferenceError(None, executor.placement.devices[-1], operation) from None
        raise
      finally:
        executor.owner = None
        if executor.state != 'failed': executor.state = 'ready'
    finally: executor.lock.release()

  @contextmanager
  def _placed_operation(self, stage:int, operation:str):
    try: yield
    except Exception: raise PlacedInferenceError(stage, self._placed.placement.devices[stage], operation) from None

  def _placed_sample(self, logits:Tensor, temperature:Tensor) -> Tensor:
    with self._placed_operation(len(self._placed.stages)-1, 'sample'):
      out = self._sample(logits, temperature).realize()
      Device[self._placed.placement.devices[-1]].synchronize()
      return out

  def _placed_direct(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, *, jit:bool) -> Tensor:
    self.check_available()
    self._placed_position(tokens, start_pos)
    if temperature.device != self._placed.placement.devices[-1] or temperature.shape != (1,) or temperature.dtype != dtypes.float32:
      raise ValueError('placed temperature must be FP32 [1] on the final owner')
    def validate():
      with self._placed_operation(len(self._placed.stages)-1, 'input'): value = temperature.item()
      if not math.isfinite(value) or value < 0: raise ValueError('temperature must be finite and nonnegative')
      with self._placed_operation(0, 'input'): ids = list(tokens.flatten().data())
      if any(not 0 <= t < self._placed.config.vocab_size for t in ids): raise ValueError('invalid placed token ID')
    with self._placed_transaction('direct', validate) as owner:
      self._cached_tokens = []  # direct callers have no complete history contract, even on success
      return self._placed_sample(self._placed_step(tokens, start_pos, jit=jit, owner=owner), temperature)

  def _placed_position(self, tokens:Tensor, start_pos:int|UOp) -> int:
    if tokens.dtype != dtypes.int32 or tokens.ndim != 2 or any(type(n) is not int for n in tokens.shape) or tokens.shape[0] != 1:
      raise ValueError('placed tokens must be concrete int32 [1, C]')
    if tokens.device != self.placement.devices[0]: raise ValueError('placed tokens must be on the first owner')
    try: pos = start_pos.unbind()[1] if isinstance(start_pos, UOp) else start_pos
    except (AssertionError, ValueError): raise ValueError('placed start_pos must be an integer or bound variable') from None
    if type(pos) is not int or not 0 <= pos < pos+tokens.shape[1] <= self.max_context:
      raise ValueError('placed token positions must fit the context')
    if tokens.shape[1] > self.placement.chunk_size: raise ValueError('placed token count exceeds fixed chunk size')
    return pos

  def _placed_step(self, tokens:Tensor, start_pos:int|UOp, *, jit:bool=True, owner:object|None=None) -> Tensor:
    if owner is None or owner is not self._placed.owner: raise PlacedModelBusyError('placed step requires its transaction owner')
    pos = self._placed_position(tokens, start_pos)
    sp = UOp.variable('start_pos', 0, self.max_context-1).bind(pos)
    with self._placed_operation(0, 'input'): value = tokens.contiguous().realize()
    for i,stage in enumerate(self._placed.stages):
      if i:
        with self._placed_operation(i, 'transfer'): value = _transfer_activation(value, stage.device, self._placed.placement.transfer)
      with self._placed_operation(i, 'stage'): value = stage.run(value, sp, jit=jit)
    return value

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    if hasattr(self, 'placement'): return self._placed_direct(tokens, start_pos, temperature, jit=False)
    return self._sample(self.forward_embeddings(self.token_embd(tokens).float(), start_pos), temperature)

  def _multimodal_decode(self, tokens:Tensor, start_pos:UOp, position_ids:Tensor, temperature:Tensor) -> Tensor:
    if hasattr(self, 'placement'): raise ValueError('placement does not support multimodal decode')
    return self._sample(self.forward_embeddings(self.token_embd(tokens).float(), start_pos, position_ids), temperature)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    if hasattr(self, 'placement'): return self._placed_direct(tokens, start_pos, temperature, jit=True)
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0)), *, placement:LayerPlacement|None=None) -> tuple[Transformer, dict]:
    if placement is not None:
      if not isinstance(gguf, (str, pathlib.Path)): raise TypeError('explicit placement requires a local GGUF path')
      if isinstance(gguf, str) and '://' in gguf: raise ValueError('explicit placement requires a local GGUF path, not a URL')
      if realize: raise ValueError('explicit placement requires REALIZE=0')
      index = index_gguf(gguf)
      config, placement = preflight_placement(index, placement, max_context=max_context, realize=realize)
      manifest = _placement_manifest(index, placement)
      model = Transformer(config, _shape_only=True)
      parameters = nn.state.get_state_dict(model)
      # Check construction against the complete validated manifest before the very first payload transfer.
      if parameters.keys() != {name for name,_,_ in manifest} or any(parameters[n].shape != info.shape for n,info,_ in manifest):
        raise ValueError('placed model construction disagrees with GGUF manifest')
      model.placement = placement
      loaded:dict[tuple[int, int, int, str], Tensor] = {}
      for name,info,owner in manifest:
        key = (info.part, info.offset, info.nbytes, owner)
        if key not in loaded:
          weight = load_gguf_tensor(index, info, device=owner)
          if getenv('HALF', 1): weight = weight.half()
          # Full-head Llama RoPE: interleaved rows to half-split, matching the legacy Q/K transforms below.
          if name.endswith(('attn_q.weight', 'attn_k.weight')):
            heads = config.n_heads if name.endswith('attn_q.weight') else config.n_kv_heads
            weight = weight.reshape(heads, config.head_dim, config.dim).rearrange('n (h two) d -> n (two h) d', two=2).reshape(info.shape)
          loaded[key] = weight
        if loaded[key].device != owner: raise ValueError(f'placed tensor is not on its owner: {name}')
        parameters[name].replace(loaded[key])
      model._placed = _PlacedExecutor(model, config, placement)
      return model, index.kv

    # TODO: remove the need for copy to default device
    kv, state_dict = gguf_load(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    ssm = None
    ssm_layers: tuple[bool, ...] = ()
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
      ssm_layers = tuple((i+1) % kv[f'{arch}.full_attention_interval'] != 0 for i in range(kv[f'{arch}.block_count']))
    elif arch == 'kimi-linear':
      ssm_layers = tuple(x == 0 for x in n_kv_heads)
      n_kv_heads = max(n_kv_heads)
      ssm = SSMConfig(kv[f'{arch}.ssm.conv_kernel'], kv[f'{arch}.kda.head_dim'], n_heads, n_heads, n_heads*kv[f'{arch}.kda.head_dim'], kda=True)
      for i, is_ssm in enumerate(ssm_layers):
        if not is_ssm: continue
        state_dict[f"blk.{i}.attn_qkv.weight"] = state_dict.pop(f"blk.{i}.attn_q.weight").cat(
          state_dict.pop(f"blk.{i}.attn_k.weight"), state_dict.pop(f"blk.{i}.attn_v.weight"), dim=0).contiguous()
        state_dict[f"blk.{i}.ssm_conv1d.weight"] = state_dict.pop(f"blk.{i}.ssm_conv1d_q.weight").cat(
          state_dict.pop(f"blk.{i}.ssm_conv1d_k.weight"), state_dict.pop(f"blk.{i}.ssm_conv1d_v.weight"), dim=0).squeeze(1).contiguous()
        state_dict[f"blk.{i}.ssm_out.weight"] = state_dict.pop(f"blk.{i}.attn_output.weight")
    if arch in ('qwen35', 'qwen35moe', 'glm4moe'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if arch == 'kimi-linear': continue
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
        w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
        prefix = head_dim-rope_dim
        state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
      elif arch == 'llama' and 'attn_k.weight' in name:
        w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
        state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)
    config = TransformerConfig(
      num_blocks=kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0), dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv[f'{arch}.rope.freq_base'],
      rope_dim=rope_dim,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe', 'kimi-linear')),
      expert_gating_func=ExpertGating(kv.get(f'{arch}.expert_gating_func', ExpertGating.SOFTMAX)),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict)
    model = Transformer(config)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def warmup(self):
    if hasattr(self, '_placed'):
      with self._placed_transaction('warmup') as owner:
        self._cached_tokens = []
        try:
          for count in dict.fromkeys((1, self._placed.placement.chunk_size)):
            for i in range(3):  # first eager, then capture, then replay
              tokens = Tensor([[i % self._placed.config.vocab_size]*count], dtype=dtypes.int32, device=self._placed.placement.devices[0])
              self._placed_step(tokens, 0, owner=owner)
          for stage in self._placed.stages: Device[stage.device].synchronize()
        finally: self._cached_tokens = []
      return
    for _ in range(2):
      gen = self.generate([0])
      try: list(zip(range(2), gen))
      finally: gen.close()

  def get_start_pos(self, tokens:list[int]) -> int:
    # recurrent state can't be partially reused after divergence: reuse it only when tokens extend the cached prefix
    if self.has_recurrent_block:
      return len(self._cached_tokens) if self._cached_tokens and len(self._cached_tokens) < len(tokens) \
        and tokens[:len(self._cached_tokens)] == self._cached_tokens else 0
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def reset_generation_state(self):
    if hasattr(self, '_placed'):
      with self._placed_transaction('reset'):
        self._cached_tokens = []
        self._generation_epoch += 1
      return
    # Position zero resets GDN and overwrites the valid KV prefix. Never replace buffers captured by a JIT.
    self._cached_tokens = []
    self._generation_epoch += 1
    self._multimodal_active = False

  def _generate_multimodal(self, tokens:list[int], chunk_size:int, temperature:float,
                           inputs_embeds:Tensor|None, position_ids:Tensor|None, rope_delta:int):
    self.reset_generation_state()
    epoch = self._generation_epoch
    self._multimodal_active = True
    try:
      length, device = len(tokens), self.token_embd.weight.device
      if not isinstance(inputs_embeds, Tensor) or not isinstance(position_ids, Tensor):
        raise ValueError("multimodal generation requires embeddings and positions tensors together")
      if not 0 < length <= self.max_context: raise ValueError("multimodal prompt must fit the context")
      if inputs_embeds.shape != (1, length, self.token_embd.weight.shape[1]) or inputs_embeds.dtype != dtypes.float32:
        raise ValueError("inputs_embeds must be float32 [1, prompt_length, decoder_dim]")
      if position_ids.shape != (3, 1, length) or position_ids.dtype != dtypes.int32:
        raise ValueError("position_ids must be int32 [3, 1, prompt_length]")
      if inputs_embeds.device != device or position_ids.device != device:
        raise ValueError("embeddings and positions must be on the decoder device")
      if type(chunk_size) is not int or chunk_size <= 0: raise ValueError("chunk_size must be positive")
      try: valid_temperature = type(temperature) in (float, int) and math.isfinite(temperature) and temperature >= 0
      except OverflowError: valid_temperature = False
      if not valid_temperature: raise ValueError("temperature must be finite and nonnegative")
      if type(rope_delta) is not int or not 0 <= length+rope_delta <= self.max_context+rope_delta < 2**31:
        raise ValueError("invalid rope_delta")
      if any(type(t) is not int or not 0 <= t < self.token_embd.weight.shape[0] for t in tokens): raise ValueError("invalid token ID")
      # Value checks are request-boundary work, never copyouts inside the decode JIT.
      if not inputs_embeds.isfinite().all().item() or not (position_ids >= 0).all().item():
        raise ValueError("embeddings must be finite and coordinates nonnegative")
      if int(position_ids.max().item())+1-length != rope_delta: raise ValueError("rope_delta does not match prompt coordinates")
      if length == self.max_context: return
      temp = Tensor([temperature], dtype=dtypes.float32, device=device).realize()
      # Eager bounded chunks also on CPU GDN; unlike text prefill, do not force token-serial processing.
      chunk_size = min(chunk_size, 32)
      physical = UOp.variable("start_pos", 0, self.max_context-1)
      for start in range(0, length, chunk_size):
        end = min(start+chunk_size, length)
        logits = self.forward_embeddings(inputs_embeds[:, start:end].contiguous(), physical.bind(start),
                                         position_ids[:, :, start:end].contiguous()).realize()
      out = self._sample(logits, temp).realize()
      # Owned, realized buffers keep identical dtype/view/device signatures across requests and JIT replay.
      token = Tensor.empty(1, 1, dtype=dtypes.int32, device=device).realize()
      coords = Tensor.empty(3, 1, 1, dtype=dtypes.int32, device=device).realize()
      while length < self.max_context:
        tokens.append(int(out.item()))
        yield tokens[-1]
        if epoch != self._generation_epoch: return
        if length+1 == self.max_context: return
        token.assign(Tensor([[tokens[-1]]], dtype=dtypes.int32, device=device)).realize()
        coords.assign(Tensor.full((3, 1, 1), length+rope_delta, dtype=dtypes.int32, device=device)).realize()
        out = self.multimodal_rollout_jit(token, physical.bind(length), coords, temp).realize()
        length += 1
    finally:
      # A retained old iterator must not reset a newer request's live state when it is eventually closed.
      if epoch == self._generation_epoch: self.reset_generation_state()

  def _generate_placed(self, tokens:list[int], chunk_size:int, temperature:float):
    if type(chunk_size) is not int or chunk_size != self.placement.chunk_size:
      raise ValueError('placed chunk_size is fixed at load time')
    if not isinstance(tokens, list) or not 0 < len(tokens) <= self.max_context or any(
        type(t) is not int or not 0 <= t < self._placed.config.vocab_size for t in tokens): raise ValueError('invalid placed prompt tokens')
    try: valid_temperature = type(temperature) in (int, float) and math.isfinite(temperature) and temperature >= 0
    except OverflowError: valid_temperature = False
    if not valid_temperature: raise ValueError('temperature must be finite and nonnegative')
    with self._placed_transaction('generation') as owner:
      temp = Tensor([temperature], dtype=dtypes.float32, device=self._placed.placement.devices[-1]).realize()
      pos = self.get_start_pos(tokens)
      # Own the working history, so a caller mutating its list between yields cannot corrupt prefix metadata.
      history = tokens.copy()
      while len(history) < self.max_context:
        end = min(pos+chunk_size, len(history))
        value = Tensor([history[pos:end]], dtype=dtypes.int32, device=self._placed.placement.devices[0]).realize()
        logits = self._placed_step(value, pos, owner=owner)
        pos = end
        if pos < len(history): continue
        with self._placed_operation(len(self._placed.stages)-1, 'sample'):
          token = int(self._placed_sample(logits, temp).item())
        history.append(token)
        tokens.append(token)  # retain the legacy list-append surface
        self._cached_tokens = history[:-1]
        try: yield token
        except GeneratorExit: return  # only this clean, fully committed token boundary can release as ready

  def generate(self, tokens:list[int], chunk_size:int|None=None, temperature:float=0.0, *,
               inputs_embeds:Tensor|None=None, position_ids:Tensor|None=None, rope_delta:int=0):
    if hasattr(self, 'placement'):
      if inputs_embeds is not None or position_ids is not None or rope_delta != 0: raise ValueError('placement does not support multimodal inputs')
      yield from self._generate_placed(tokens, self.placement.chunk_size if chunk_size is None else chunk_size, temperature)
      return
    if chunk_size is None: chunk_size = 32
    if inputs_embeds is not None or position_ids is not None or rope_delta != 0:
      yield from self._generate_multimodal(tokens, chunk_size, temperature, inputs_embeds, position_ids, rope_delta)
      return
    if self._multimodal_active: self.reset_generation_state()
    self._generation_epoch += 1
    epoch = self._generation_epoch
    if self.has_recurrent_block and not amd_custom_kernels_supported(self.token_embd.weight.device): chunk_size = 1
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor([temperature])
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      n_toks = min(chunk_size, len(tokens) - start_pos)
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
      out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp).realize()
      start_pos += n_toks
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
      if epoch != self._generation_epoch: return

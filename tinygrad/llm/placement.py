"""Pure, explicit placement of whole dense transformer layers (no device discovery)."""
from __future__ import annotations
import re
from dataclasses import dataclass
from itertools import accumulate
from typing import Literal, TYPE_CHECKING
if TYPE_CHECKING:
  from tinygrad.llm.gguf import GGUFIndex, GGUFTensorInfo
  from tinygrad.llm.model import TransformerConfig

_BLOCK_WEIGHTS = ('attn_norm', 'attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_norm', 'ffn_gate', 'ffn_up', 'ffn_down')

@dataclass(frozen=True)
class LayerPlacement:
  devices: tuple[str, ...]
  layer_counts: tuple[int, ...]
  transfer: Literal['host', 'native'] = 'host'
  chunk_size: int = 32

  def validate(self, num_blocks:int) -> LayerPlacement:
    if type(num_blocks) is not int or num_blocks <= 0: raise ValueError('block count must be a positive integer')
    if not self.devices or len(self.devices) != len(self.layer_counts): raise ValueError('devices and layer counts must be nonempty and equal length')
    if any(type(n) is not int or n <= 0 for n in self.layer_counts) or sum(self.layer_counts) != num_blocks:
      raise ValueError('positive integer layer counts must cover every block exactly once')
    devices = []
    for device in self.devices:
      if not isinstance(device, str) or (match := re.fullmatch(r'(CPU|PYTHON|AMD|NV|CUDA)(?::([0-9]+))?', device.upper())) is None:
        raise ValueError(f'unsupported placement device: {device!r}')
      devices.append(match[1] + (f':{index}' if (index := int(match[2] or '0')) else ''))
    if len(set(devices)) != len(devices): raise ValueError('placement devices must be distinct')
    if len({d.split(':')[0] for d in devices}) != 1: raise ValueError('placement requires one backend')
    if self.transfer not in ('host', 'native'): raise ValueError('transfer must be host or native')
    if type(self.chunk_size) is not int or not 1 <= self.chunk_size <= 32: raise ValueError('chunk size must be an integer in [1, 32]')
    return LayerPlacement(tuple(devices), tuple(self.layer_counts), self.transfer, self.chunk_size)

  def block_device(self, index:int) -> str:
    if type(index) is not int or not 0 <= index < sum(self.layer_counts): raise ValueError(f'invalid block index: {index!r}')
    for device, end in zip(self.devices, accumulate(self.layer_counts)):
      if index < end: return device
    raise ValueError('invalid placement coverage')

  def owner(self, parameter_name:str) -> str:
    if parameter_name == 'token_embd.weight': return self.devices[0]
    if parameter_name in ('output_norm.weight', 'output.weight'): return self.devices[-1]
    if (m := re.fullmatch(r'blk\.(0|[1-9][0-9]*)\.([a-z_]+)\.weight', parameter_name)) and m[2] in _BLOCK_WEIGHTS:
      return self.block_device(int(m[1]))
    raise ValueError(f'unsupported placed parameter: {parameter_name}')


def _placement_manifest(index:GGUFIndex, placement:LayerPlacement) -> tuple[tuple[str, GGUFTensorInfo, str], ...]:
  """Resolve the sole supported alias before loading/accounting. The caller preflights the complete index."""
  infos = {t.name:t for t in index.tensors if t.name != 'rope_freqs.weight'}
  if 'output.weight' not in infos: infos['output.weight'] = infos['token_embd.weight']
  return tuple((name, info, placement.owner(name)) for name, info in infos.items())


@dataclass(frozen=True)
class PlacementMemory:
  device: str
  weight_bytes: int
  tied_replica_bytes: int
  kv_bytes: int
  rope_bytes: int
  boundary_bytes: int
  largest_payload_bytes: int


def estimate_placement(index:GGUFIndex, config:TransformerConfig, placement:LayerPlacement) -> tuple[PlacementMemory, ...]:
  """Pure metadata accounting for a preflighted dense model; fit is unknown, not promised.

  Weight bytes are exact raw payloads, including ties replicated on different owners. tied_replica_bytes is an
  explanatory subset, not an additional charge. KV reserves batch-one FP16 K/V; one FP32 RoPE table per unique
  (owner, dimension, context, theta) is shared by that owner's blocks. Boundaries provision two full-chunk FP32 buffers.
  In addition, reserve vocab_size*4 bytes for final-owner logits; decoded scratch, JIT, allocator rounding/caches and
  backend staging are unknown. Active loading may hold two largest payload copies plus staging, not bounded process RAM.
  """
  placement = placement.validate(config.num_blocks)
  if config.max_context <= 0 or placement.chunk_size > config.max_context: raise ValueError('chunk size must fit a positive context')
  manifest = _placement_manifest(index, placement)
  tied = not any(t.name == 'output.weight' for t in index.tensors)
  result = []
  for device, blocks in zip(placement.devices, placement.layer_counts):
    payloads = {(info.part,info.offset,info.nbytes):info.nbytes for _,info,owner in manifest if owner == device}
    replica = next((info.nbytes for name,info,owner in manifest if name == 'output.weight' and owner == device), 0) \
      if tied and device != placement.devices[0] else 0
    result.append(PlacementMemory(device, sum(payloads.values()), replica,
      4*blocks*config.max_context*config.n_kv_heads*config.head_dim, config.max_context*config.rope_dim*4,
      2*placement.chunk_size*config.dim*4, max(payloads.values(), default=0)))
  return tuple(result)

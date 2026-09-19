"""Pure, explicit placement of whole dense transformer layers (no device discovery)."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Literal

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
      index = int(match[2] or '0')
      devices.append(match[1] + (f':{index}' if index else ''))
    if len(set(devices)) != len(devices): raise ValueError('placement devices must be distinct')
    if len({d.split(':')[0] for d in devices}) != 1: raise ValueError('placement requires one backend')
    if self.transfer not in ('host', 'native'): raise ValueError('transfer must be host or native')
    if type(self.chunk_size) is not int or not 1 <= self.chunk_size <= 32: raise ValueError('chunk size must be an integer in [1, 32]')
    return LayerPlacement(tuple(devices), tuple(self.layer_counts), self.transfer, self.chunk_size)

  def block_device(self, index:int) -> str:
    if type(index) is not int or not 0 <= index < sum(self.layer_counts): raise ValueError(f'invalid block index: {index!r}')
    end = 0
    for device, count in zip(self.devices, self.layer_counts):
      end += count
      if index < end: return device
    raise ValueError('invalid placement coverage')

  def owner(self, parameter_name:str) -> str:
    if parameter_name == 'token_embd.weight': return self.devices[0]
    if parameter_name in ('output_norm.weight', 'output.weight'): return self.devices[-1]
    if (m := re.fullmatch(r'blk\.(0|[1-9][0-9]*)\.([a-z_]+)\.weight', parameter_name)) and m[2] in _BLOCK_WEIGHTS:
      return self.block_device(int(m[1]))
    raise ValueError(f'unsupported placed parameter: {parameter_name}')

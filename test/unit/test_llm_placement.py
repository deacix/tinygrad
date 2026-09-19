import itertools, unittest
from unittest.mock import patch
from tinygrad.llm.placement import LayerPlacement

class TestLayerPlacement(unittest.TestCase):
  def test_canonical_devices_without_opening(self):
    with patch('tinygrad.device._Device.__getitem__', side_effect=AssertionError('device opened')):
      p = LayerPlacement(('cpu:00', 'CPU:01'), (1, 3)).validate(4)
      self.assertEqual(p.devices, ('CPU', 'CPU:1'))
      self.assertEqual([p.block_device(i) for i in range(4)], ['CPU', 'CPU:1', 'CPU:1', 'CPU:1'])
      self.assertEqual(p.owner('token_embd.weight'), 'CPU')
      self.assertEqual(p.owner('output_norm.weight'), 'CPU:1')
      self.assertEqual(p.owner('output.weight'), 'CPU:1')
      self.assertEqual(p.owner('blk.2.attn_q.weight'), 'CPU:1')

  def test_positive_compositions(self):
    for layers in range(1, 9):
      for stages in range(1, min(4, layers)+1):
        for cuts in itertools.combinations(range(1, layers), stages-1):
          ends = (0, *cuts, layers)
          counts = tuple(b-a for a,b in zip(ends, ends[1:]))
          devices = tuple('CPU' if i == 0 else f'CPU:{i}' for i in range(stages))
          p = LayerPlacement(devices, counts).validate(layers)
          self.assertEqual(tuple(p.block_device(i) for i in range(layers)),
                           tuple(d for d,n in zip(devices, counts) for _ in range(n)))

  def test_invalid_placement(self):
    invalid = [
      LayerPlacement((), ()), LayerPlacement(('CPU',), (0,)), LayerPlacement(('CPU',), (-1,)),
      LayerPlacement(('CPU',), (True,)), LayerPlacement(('CPU',), (1.0,)),
      LayerPlacement(('CPU',), (1, 1)), LayerPlacement(('CPU', 'CPU:00'), (1, 1)),
      LayerPlacement(('CUDA:1', 'CUDA:01'), (1, 1)), LayerPlacement(('CPU', 'PYTHON'), (1, 1)),
      LayerPlacement(('CPU:-1',), (2,)), LayerPlacement(('CPU:+1',), (2,)),
      LayerPlacement(('CPU:1.0',), (2,)), LayerPlacement(('CPU:CLANG',), (2,)),
      LayerPlacement(('CPU:١',), (2,)), LayerPlacement(('METAL',), (2,)),
      LayerPlacement(('CPU',), (2,), transfer='auto'), LayerPlacement(('CPU',), (2,), chunk_size=0),
      LayerPlacement(('CPU',), (2,), chunk_size=33), LayerPlacement(('CPU',), (2,), chunk_size=True),
    ]
    for p in invalid:
      with self.subTest(p=p), self.assertRaises(ValueError): p.validate(2)
    for blocks in (0, -1, True, 2.0, 3):
      with self.subTest(blocks=blocks), self.assertRaises(ValueError): LayerPlacement(('CPU',), (2,)).validate(blocks)

  def test_invalid_parameter_names_and_indices(self):
    p = LayerPlacement(('CPU', 'CPU:1'), (2, 2)).validate(4)
    for name in ('other.weight', 'token_embd.bias', 'blk.0.attn_q.bias', 'blk.0.attn_q_norm.weight',
                 'blk.-1.attn_q.weight', 'blk.00.attn_q.weight', 'blk.4.attn_q.weight', 'blk.1.unknown.weight'):
      with self.subTest(name=name), self.assertRaises(ValueError): p.owner(name)
    for index in (-1, 4, True, 1.0):
      with self.subTest(index=index), self.assertRaises(ValueError): p.block_device(index)

  def test_one_stage_and_backends(self):
    for device in ('CPU', 'PYTHON', 'AMD', 'NV', 'CUDA'):
      p = LayerPlacement((device,), (1,), transfer='native', chunk_size=1).validate(1)
      self.assertEqual(p.owner('output.weight'), p.owner('token_embd.weight'))
      self.assertEqual(p.owner('blk.0.ffn_down.weight'), device)

if __name__ == '__main__': unittest.main()

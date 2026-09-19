import dataclasses, gc, itertools, struct, unittest, weakref
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import pytest
from tinygrad import Tensor, dtypes, nn, getenv
from tinygrad.llm import model as model_module
from tinygrad.llm.gguf import index_gguf
from tinygrad.llm.model import Transformer
from tinygrad.llm.placement import LayerPlacement
from tinygrad.uop.ops import Ops
from test.unit.test_gguf_placement import build_gguf, quant_block


def llama_fixture(*, dim=32, blocks=4, tied=True, typ=0):
  """Independent manifest/values: no production config, model constructors or decoder used."""
  rng = np.random.default_rng(1234)
  metadata = {'general.architecture':(8,'llama'), 'general.name':(8,'offline fixture'),
    'llama.block_count':(4,blocks), 'llama.embedding_length':(4,dim), 'llama.feed_forward_length':(4,dim*2),
    'llama.attention.head_count':(4,2), 'llama.attention.head_count_kv':(4,1),
    'llama.attention.layer_norm_rms_epsilon':(6,1e-5), 'llama.rope.freq_base':(6,10000.0),
    'llama.context_length':(4,32), 'tokenizer.ggml.tokens':(9,(8,[str(i) for i in range(32)]))}
  shapes = {'token_embd.weight':(32,dim), 'output_norm.weight':(dim,)}
  if not tied: shapes['output.weight'] = (32,dim)
  for i in range(blocks):
    shapes.update({f'blk.{i}.{name}.weight':shape for name,shape in [
      ('attn_norm',(dim,)), ('attn_q',(dim,dim)), ('attn_k',(dim//2,dim)), ('attn_v',(dim//2,dim)),
      ('attn_output',(dim,dim)), ('ffn_norm',(dim,)), ('ffn_gate',(2*dim,dim)), ('ffn_up',(2*dim,dim)), ('ffn_down',(dim,2*dim))]})
  tensors, values = [], {}
  for name,shape in shapes.items():
    t = 0 if len(shape) == 1 else typ
    v = (1 + rng.uniform(-0.05,0.05,shape) if len(shape) == 1 else rng.uniform(-0.1,0.1,shape)).astype(np.float32)
    if t in (0,1):
      v = v.astype(np.float32 if t == 0 else np.float16)
      payload = v.tobytes()
    elif t == 30:
      bits = v.view(np.uint32) >> 16
      payload, v = bits.astype('<u2').tobytes(), (bits << 16).view(np.float32)
    else:
      # Rotate/scalewise vary whole independent blocks so a row permutation cannot accidentally be a no-op.
      _, oracle = quant_block(t)
      size = len(oracle)
      packed, rows = [], []
      for n in range(np.prod(shape)//size):
        block, expected = quant_block(t,n % 3)
        if t in (2,8):
          scale = 1 + n % 3
          block = struct.pack('<e',struct.unpack('<e',block[:2])[0]*scale) + block[2:]
          expected = [v*scale for v in expected]
        packed.append(block)
        rows.extend(expected)
      payload, v = b''.join(packed), np.array(rows,dtype=np.float32).reshape(shape)
    tensors.append((name,shape,t,payload))
    values[name] = v.astype(np.float32)
  return tensors, metadata, values


def save_llama(tmp_path, *, tensors=None, metadata=None, **kwargs):
  ts, kv, values = llama_fixture(**kwargs)
  path = tmp_path / 'model.gguf'
  path.write_bytes(build_gguf(ts if tensors is None else tensors, kv if metadata is None else metadata, alignment=64))
  return path, values


def buffer_roots(t): return {u for u in t.uop.toposort() if u.op is Ops.BUFFER}


def no_allocations():
  stack = ExitStack()
  for target in ('tinygrad.device._Device.__getitem__', 'tinygrad.device.Buffer.__init__',
                 'tinygrad.llm.model.Transformer.__init__', 'tinygrad.llm.model.load_gguf_tensor'):
    stack.enter_context(patch(target, side_effect=AssertionError('preflight allocated'), create=True))
  return stack

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

class TestPlacedLoading:
  @pytest.mark.parametrize('typ,value', [(4,999),(4,1),(4,0),(8,'2'),(7,True),(6,2.0)])
  def test_unknown_quantization_version_fails_before_copies(self, tmp_path, typ, value):
    ts, kv, _ = llama_fixture(typ=2)
    kv['general.quantization_version'] = (typ,value)
    path, _ = save_llama(tmp_path,tensors=ts,metadata=kv)
    with no_allocations(), pytest.raises(ValueError,match='quantization version'):
      Transformer.from_gguf(path,realize=False,placement=LayerPlacement(('CPU','CPU:1'),(1,3)))

  def test_explicit_quantization_version_two_is_accepted(self, tmp_path):
    ts, kv, _ = llama_fixture(typ=2)
    kv['general.quantization_version'] = (4,2)
    path, _ = save_llama(tmp_path,tensors=ts,metadata=kv)
    with no_allocations():
      config, _ = model_module.preflight_placement(index_gguf(path),LayerPlacement(('CPU',),(4,)))
    assert config.num_blocks == 4

  @pytest.mark.parametrize('key,typ,value', [
    ('general.architecture',8,'qwen2'), ('general.tensor_data_layout',8,'other'),
    ('llama.rope.scaling.type',8,'linear'), ('llama.rope.scaling.factor',6,2.0), ('llama.rope.scale_linear',6,2.0),
    ('llama.rope.scaling.original_context_length',4,16), ('llama.rope.scaling.finetuned',7,True),
    ('llama.attention.sliding_window',4,16), ('llama.attention.sliding_window',4,0),
    ('llama.attention.max_alibi_bias',6,8.0), ('llama.attention.clamp_kqv',6,1.0),
    ('llama.attention.output_gate',7,True), ('llama.attention.qkv_bias',7,True), ('llama.attention.qk_norm',7,True),
    ('llama.attention.scale',6,0.5), ('llama.attention.logit_softcap',6,20.0),
    ('llama.use_parallel_residual',7,True), ('llama.nextn_predict_layers',4,1),
    ('llama.expert_count',4,8), ('llama.expert_used_count',4,2), ('llama.expert_feed_forward_length',4,64),
    ('llama.attention.kv_lora_rank',4,16), ('llama.attention.key_length_mla',4,16),
    ('llama.ssm.state_size',4,16), ('llama.rope.dimension_sections',9,(4,[2,3,3])),
    ('llama.tensor_data_layout',8,'other'), ('llama.attention.causal',7,False),
  ])
  def test_matching_shape_unsupported_features_fail_before_copies(self, tmp_path, key, typ, value):
    ts, kv, _ = llama_fixture()
    kv[key] = (typ,value)
    path, _ = save_llama(tmp_path,tensors=ts,metadata=kv)
    with no_allocations(), pytest.raises(ValueError):
      Transformer.from_gguf(path, realize=False, placement=LayerPlacement(('CPU','CPU:1'),(1,3)))

  @pytest.mark.parametrize('key,typ,value', [
    ('llama.block_count',4,0), ('llama.embedding_length',4,0), ('llama.feed_forward_length',5,-1),
    ('llama.attention.head_count',4,0), ('llama.attention.head_count',4,3), ('llama.attention.head_count_kv',4,0),
    ('llama.attention.head_count_kv',4,3), ('llama.context_length',4,0), ('llama.context_length',7,True),
    ('llama.embedding_length',6,32.0), ('llama.rope.dimension_count',4,15), ('llama.rope.dimension_count',4,8),
    ('llama.attention.key_length',4,8), ('llama.attention.value_length',4,8),
    ('llama.attention.layer_norm_rms_epsilon',6,0.0), ('llama.attention.layer_norm_rms_epsilon',6,float('nan')),
    ('llama.rope.freq_base',6,float('inf')), ('llama.rope.freq_base',6,-1.0),
    ('tokenizer.ggml.tokens',9,(8,[])), ('tokenizer.ggml.tokens',8,'not an array'), ('llama.vocab_size',4,33),
  ])
  def test_invalid_config_fails_before_copies(self, tmp_path, key, typ, value):
    ts, kv, _ = llama_fixture()
    kv[key] = (typ,value)
    path, _ = save_llama(tmp_path,tensors=ts,metadata=kv)
    with no_allocations(), pytest.raises(ValueError):
      Transformer.from_gguf(path, realize=False, placement=LayerPlacement(('CPU',),(4,)))

  @pytest.mark.parametrize('kind', ['missing','shape','extra','bias','qnorm','expert','recurrent','predictor','type','ignored_shape','ignored_type'])
  def test_entire_manifest_fails_before_copies(self, tmp_path, kind):
    ts, kv, _ = llama_fixture()
    # Invalid items go last, after otherwise valid weights, to detect incremental validation/copying.
    if kind == 'missing': ts.pop()
    elif kind == 'shape': ts[-1] = (*ts[-1][:1],(64,32),*ts[-1][2:])
    elif kind == 'type': ts[-1] = (*ts[-1][:2],26,ts[-1][3])  # native int32: indexed but not accepted for placed weights
    else:
      name = {'extra':'other.weight', 'bias':'blk.0.attn_q.bias', 'qnorm':'blk.0.attn_q_norm.weight',
              'expert':'blk.0.ffn_gate_inp.weight','recurrent':'blk.0.ssm_a','predictor':'blk.4.attn_q.weight',
              'ignored_shape':'rope_freqs.weight','ignored_type':'rope_freqs.weight'}[kind]
      ts.append((name,(9,) if kind == 'ignored_shape' else (8,),26 if kind == 'ignored_type' else 0,bytes(36)))
    path, _ = save_llama(tmp_path,tensors=ts,metadata=kv)
    with no_allocations(), pytest.raises(ValueError):
      Transformer.from_gguf(path, realize=False, placement=LayerPlacement(('CPU','CPU:1'),(2,2)))

  @pytest.mark.parametrize('context', [0,-1,True,1.5,3])
  def test_bad_effective_context(self, tmp_path, context):
    path, _ = save_llama(tmp_path)
    with no_allocations(), pytest.raises(ValueError):
      Transformer.from_gguf(path, context, False, placement=LayerPlacement(('CPU',),(4,),chunk_size=4))

  def test_effective_realize_default_and_pure_check(self, tmp_path):
    path, _ = save_llama(tmp_path)
    placement = LayerPlacement(('CPU',),(4,))
    with no_allocations():
      # The legacy default is captured at import. Simulate importing with REALIZE=1 without reloading the module.
      with patch.object(Transformer.from_gguf,'__defaults__',(None,True)), pytest.raises(ValueError,match='REALIZE=0'):
        Transformer.from_gguf(path,placement=placement)
      with pytest.raises(ValueError,match='REALIZE=0'):
        model_module.preflight_placement(index_gguf(path),placement,realize=True)

  def test_path_realize_and_placement_validation(self, tmp_path):
    path, _ = save_llama(tmp_path)
    tensor = Tensor.empty(1,device='PYTHON')
    with no_allocations():
      for source in (tensor, b'GGUF', 'https://example.invalid/model.gguf'):
        with pytest.raises((TypeError,ValueError)): Transformer.from_gguf(source,placement=LayerPlacement(('CPU',),(4,)))
      with pytest.raises(ValueError): Transformer.from_gguf(path,realize=True,placement=LayerPlacement(('CPU',),(4,)))
      with pytest.raises(ValueError): Transformer.from_gguf(path,realize=False,placement=LayerPlacement(('CPU',),(3,)))
      with pytest.raises(FileNotFoundError): Transformer.from_gguf(tmp_path/'missing',placement=LayerPlacement(('CPU',),(4,)))

  @pytest.mark.parametrize('name', list(llama_fixture()[2]) + ['output.weight','rope_freqs.weight'])
  def test_every_name_has_a_shape_check(self, tmp_path, name):
    ts, kv, _ = llama_fixture(tied=False)
    ts += [('rope_freqs.weight',(8,),0,bytes(32))]
    ts = [(n,(shape[0]+1,*shape[1:]),t,raw+bytes(np.prod(shape[1:],dtype=int)*4)) if n == name else (n,shape,t,raw)
          for n,shape,t,raw in ts]
    path, _ = save_llama(tmp_path,tensors=ts,metadata=kv)
    with no_allocations(), pytest.raises(ValueError,match='shape mismatch'):
      Transformer.from_gguf(path,realize=False,placement=LayerPlacement(('CPU',),(4,)))

  @pytest.mark.parametrize('typ,size', [(3,20),(6,22),(7,24),(10,84),(11,110),(26,128),(28,256)])
  def test_known_but_unaccepted_formats(self, tmp_path, typ, size):
    ts, kv, _ = llama_fixture(dim=256,blocks=2)
    name, shape, _, _ = ts[-1]
    block = 256 if typ in (10,11) else 32
    ts[-1] = (name,shape,typ,bytes(np.prod(shape)//block*size))
    path, _ = save_llama(tmp_path,tensors=ts,metadata=kv)
    with no_allocations(), pytest.raises(ValueError,match='unsupported placed GGML type'):
      Transformer.from_gguf(path,realize=False,placement=LayerPlacement(('CPU',),(2,)))

  def test_shape_only_construction_opens_nothing(self, tmp_path):
    path, expected = save_llama(tmp_path,tied=False)
    config, _ = model_module.preflight_placement(index_gguf(path),LayerPlacement(('CPU',),(4,)))
    with patch('tinygrad.device._Device.__getitem__',side_effect=AssertionError('device opened')), \
         patch('tinygrad.device.Buffer.allocate',side_effect=AssertionError('placeholder storage')), \
         patch.object(Tensor,'_next_counter',side_effect=AssertionError('RNG state')):
      model = Transformer(config,_shape_only=True)
      state = nn.state.get_state_dict(model)
      assert {name:t.shape for name,t in state.items()} == {name:v.shape for name,v in expected.items()}
      assert all(t.device == 'PYTHON' and all(u.op in (Ops.BUFFER,Ops.RESHAPE,Ops.STACK,Ops.CONST) for u in t.uop.toposort())
                 for t in state.values())
      assert all(not u.buffer.is_allocated() for t in state.values() for u in buffer_roots(t))

  def test_huge_block_count_does_not_expand_manifest(self, tmp_path):
    ts, kv, _ = llama_fixture()
    kv['llama.block_count'] = (10,2**63)
    path, _ = save_llama(tmp_path,tensors=ts,metadata=kv)
    with no_allocations(), patch.object(model_module,'_llama_parameter_shapes',side_effect=AssertionError('unbounded manifest')), \
         pytest.raises(ValueError,match='tensor count'):
      Transformer.from_gguf(path,placement=LayerPlacement(('CPU',),(2**63,)))

  def test_pure_preflight_defaults_and_neutral_metadata(self, tmp_path):
    ts, kv, _ = llama_fixture()
    # Full multi-head attention when the optional KV-head count is omitted.
    del kv['llama.attention.head_count_kv']
    ts = [(n,(32,32),t,raw*2) if n.endswith(('attn_k.weight','attn_v.weight')) else (n,s,t,raw) for n,s,t,raw in ts]
    kv.update({'llama.rope.scaling.type':(8,'none'), 'llama.rope.scaling.factor':(6,1.0),
               'llama.rope.scale_linear':(6,1.0), 'llama.nextn_predict_layers':(4,0), 'llama.expert_count':(4,0),
               'general.tensor_data_layout':(8,'reference'), 'general.description':(8,'kept verbatim')})
    path, _ = save_llama(tmp_path,tensors=ts,metadata=kv)
    with no_allocations(), patch.object(Tensor,'__init__',side_effect=AssertionError('Tensor in preflight')):
      idx = index_gguf(path)
      before = dict(idx.kv)
      config, placed = model_module.preflight_placement(idx,LayerPlacement(('cpu:00','cpu:01'),(1,3),chunk_size=4),max_context=12)
    assert config.n_kv_heads == config.n_heads == 2 and config.max_context == 12
    assert config.head_dim == config.v_head_dim == config.rope_dim == 16
    assert placed.devices == ('CPU','CPU:1') and idx.kv == before

  @pytest.mark.parametrize('counts', [(4,),(1,3),(2,2)])
  @pytest.mark.parametrize('tied', [False,True])
  def test_exact_owners_roots_ties_and_no_random_initialization(self, tmp_path, counts, tied):
    path, values = save_llama(tmp_path,typ=2,tied=tied)
    path.chmod(0o444)
    before = path.read_bytes(), path.stat().st_mtime_ns
    devices = tuple('CPU' if i == 0 else f'CPU:{i}' for i in range(len(counts)))
    placement = LayerPlacement(devices,counts,chunk_size=4)
    idx = index_gguf(path)
    infos = {t.name:t for t in idx.tensors}
    if tied: infos['output.weight'] = infos['token_embd.weight']
    calls, copies, refs = [], [], []
    original_loader, original_index, original_to = model_module.load_gguf_tensor, model_module.index_gguf, Tensor.to
    def load(index, info, *, device):
      calls.append((info.part,info.offset,info.nbytes,device))
      return original_loader(index,info,device=device)
    def index(source):
      result = original_index(source)
      refs.append(weakref.ref(result))
      return result
    def to(t, device):
      assert t.dtype == dtypes.uint8 and t.device == 'PYTHON'
      copies.append((t.numel(),device))
      return original_to(t,device)
    with patch.object(model_module,'getenv',side_effect=lambda key,default=0: 0 if key == 'HALF' else getenv(key,default)), \
         patch.object(model_module,'load_gguf_tensor',load), \
         patch.object(model_module,'index_gguf',index), patch.object(Tensor,'to',to), \
         patch.object(Tensor,'rand',side_effect=AssertionError('random placeholder')), \
         patch.object(Tensor,'_next_counter',side_effect=AssertionError('RNG state')), \
         patch('tinygrad.nn.state.load_state_dict',side_effect=AssertionError('moving state loader')):
      model, kv = Transformer.from_gguf(path,realize=False,placement=placement)
    state = nn.state.get_state_dict(model)
    params = {name:t for name,t in state.items() if name.endswith('.weight')}
    assert set(params) == set(infos) and kv['general.name'] == 'offline fixture'
    assert set(state) - set(params) == {f'blk.{i}.{name}' for i in range(4) for name in ('cache_kv','freqs_cis')}
    expected = {(info.part,info.offset,info.nbytes,placement.owner(name)) for name,info in infos.items()}
    assert len(calls) == len(expected) and set(calls) == expected
    assert copies == [(size,device) for _,_,size,device in calls]
    all_roots = set()
    for name,t in params.items():
      roots = buffer_roots(t)
      assert t.device == placement.owner(name) and t.shape == infos[name].shape
      assert len(roots) == 1
      root, = roots
      assert root.buffer.device == t.device and root.buffer.nbytes == infos[name].nbytes
      all_roots.update(roots)
    assert len(all_roots) == len(expected)
    if tied and len(devices) == 1: assert model.output.weight.uop is model.token_embd.weight.uop
    else: assert buffer_roots(model.output.weight).isdisjoint(buffer_roots(model.token_embd.weight))
    assert model.placement == placement
    assert all(b.cache_kv.device == b.freqs_cis.device == placement.block_device(i) for i,b in enumerate(model.blk))
    assert all(b.cache_kv.uop.is_realized and b.freqs_cis.uop.is_realized for b in model.blk)
    assert model.output.__dict__['use_custom_quant'] is False
    for b in model.blk:
      for value in vars(b).values():
        if isinstance(value,model_module.Linear): assert value.__dict__['use_custom_quant'] is False
    assert model_module.Linear.use_custom_quant is True
    gc.collect()
    assert all(ref() is None for ref in refs)  # no runtime index, reader, or load-map inventory
    assert (path.read_bytes(),path.stat().st_mtime_ns) == before
    path.unlink()
    np.testing.assert_array_equal(model.output.weight.float().numpy(),values['token_embd.weight' if tied else 'output.weight'])

  @pytest.mark.parametrize('typ', [0,1,30,2,8,12,13,14])
  @pytest.mark.parametrize('half', [0,1])
  def test_all_formats_half_and_llama_permutations(self, tmp_path, typ, half):
    dim, blocks = (256,2) if typ in (12,13,14) else (32,4)
    path, expected = save_llama(tmp_path,dim=dim,blocks=blocks,typ=typ)
    with patch.object(model_module,'getenv',side_effect=lambda key,default=0: half if key == 'HALF' else getenv(key,default)):
      model, _ = Transformer.from_gguf(path,realize=False,placement=LayerPlacement(('CPU','CPU:1'),(1,blocks-1),chunk_size=4))
    state = nn.state.get_state_dict(model)
    names = ('token_embd.weight','output.weight','blk.0.attn_q.weight','blk.1.attn_k.weight','blk.1.ffn_down.weight','output_norm.weight')
    for name in names:
      value = expected['token_embd.weight' if name == 'output.weight' else name]
      if half: value = value.astype(np.float16).astype(np.float32)
      if name.endswith(('attn_q.weight','attn_k.weight')):
        heads = 2 if 'attn_q' in name else 1
        value = value.reshape(heads,dim//4,2,dim).transpose(0,2,1,3).reshape(-1,dim)
      target_dtype = dtypes.half if half else (dtypes.float32 if name == 'output_norm.weight' or typ not in (1,30) else
                                             dtypes.half if typ == 1 else dtypes.bfloat16)
      assert state[name].dtype == target_dtype
      # Read the declared dtype directly: a test-side .float() can fold an unrealized F32 -> F16 -> F32 cast pair.
      np.testing.assert_array_equal(state[name].numpy().astype(np.float32),value)

  def test_one_stage_packed_half_tie_has_shared_lazy_backing(self, tmp_path):
    path, _ = save_llama(tmp_path,typ=2)
    with patch.object(model_module,'getenv',return_value=1):
      model, _ = Transformer.from_gguf(path,realize=False,placement=LayerPlacement(('CPU',),(4,)))
    assert model.output.weight.dtype == dtypes.half and model.output.weight.uop is model.token_embd.weight.uop
    root, = buffer_roots(model.output.weight)
    assert root.buffer.dtype == dtypes.uint8 and root.buffer.nbytes == 32*32//32*18

  def test_mixed_formats_and_ignored_rope(self, tmp_path):
    ts, kv, _ = llama_fixture()
    qts, _, _ = llama_fixture(typ=8)
    ts[0] = qts[0]
    ts += [('rope_freqs.weight',(8,),0,np.ones(8,dtype=np.float32).tobytes())]
    path, _ = save_llama(tmp_path,tensors=ts,metadata=kv)
    with patch.object(model_module,'load_gguf_tensor',wraps=model_module.load_gguf_tensor) as load:
      model, _ = Transformer.from_gguf(path,placement=LayerPlacement(('CPU',),(4,)))
    assert 'rope_freqs.weight' not in [call.args[1].name for call in load.call_args_list]
    assert len(buffer_roots(model.token_embd.weight)) == 1

  @pytest.mark.parametrize('invalid', [False,True])
  def test_multipart_preflight_and_owners(self, tmp_path, invalid):
    ts, kv, values = llama_fixture()
    paths = [tmp_path/f'model-{i:05d}-of-00002.gguf' for i in (1,2)]
    split = len(ts)//2
    for i,(path,part) in enumerate(zip(paths,(ts[:split],ts[split:]))):
      if invalid and i == 1: part[-1] = (part[-1][0],(64,32),part[-1][2],part[-1][3])
      path.write_bytes(build_gguf(part,{**(kv if i == 0 else {}),'split.no':(2,i),'split.count':(2,2),
                                      'split.tensors.count':(10,len(ts))}))
      path.chmod(0o444)
    placement = LayerPlacement(('CPU','CPU:1'),(1,3))
    if invalid:
      with no_allocations(), pytest.raises(ValueError,match='shape mismatch'): Transformer.from_gguf(paths[0],placement=placement)
    else:
      with patch.object(model_module,'getenv',return_value=0): model, _ = Transformer.from_gguf(paths[0],placement=placement)
      assert model.output.weight.device == 'CPU:1' and model.token_embd.weight.device == 'CPU'
      np.testing.assert_array_equal(model.blk[3].ffn_down.weight.numpy(),values['blk.3.ffn_down.weight'])

  @pytest.mark.parametrize('key,typ,value', [('general.quantization_version',4,999),('general.tensor_data_layout',8,'other')])
  def test_later_shard_layout_metadata_rejected_before_allocations(self, tmp_path, key, typ, value):
    ts, kv, _ = llama_fixture()
    paths = [tmp_path/f'model-{i:05d}-of-00002.gguf' for i in (1,2)]
    split = len(ts)//2
    for i,(path,part) in enumerate(zip(paths,(ts[:split],ts[split:]))):
      path.write_bytes(build_gguf(part,{**(kv if i == 0 else {key:(typ,value)}), 'split.no':(2,i),'split.count':(2,2)}))
    with no_allocations(), pytest.raises(ValueError,match='Later-only GGUF'):
      Transformer.from_gguf(paths[0],placement=LayerPlacement(('CPU','CPU:1'),(1,3)))

  def test_midload_failure_discards_partial_model(self, tmp_path):
    import pathlib
    path, _ = save_llama(tmp_path)
    original_init, original_load, original_open = Transformer.__init__, model_module.load_gguf_tensor, pathlib.Path.open
    models, files, calls = [], [], []
    def init(model, *args, **kwargs):
      original_init(model,*args,**kwargs)
      models.append(weakref.ref(model))
    def opened(pp, *args, **kwargs):
      f = original_open(pp,*args,**kwargs)
      files.append(f)
      return f
    def load(index, info, *, device):
      calls.append(info.name)
      if len(calls) == 2: path.write_bytes(b'changed')
      return original_load(index,info,device=device)
    with patch.object(Transformer,'__init__',init), patch.object(model_module,'load_gguf_tensor',load), \
         patch.object(pathlib.Path,'open',opened), pytest.raises(ValueError,match='changed since indexing'):
      Transformer.from_gguf(path,placement=LayerPlacement(('CPU','CPU:1'),(1,3)))
    gc.collect()
    assert len(calls) == 2 and all(f.closed for f in files)
    assert models and all(ref() is None for ref in models)

  def test_legacy_dispatch_unchanged(self, tmp_path):
    path, expected = save_llama(tmp_path,tied=False)
    with patch.object(model_module,'index_gguf',side_effect=AssertionError('placed path in legacy')), \
         patch.object(model_module,'getenv',side_effect=lambda key,default=0: 0 if key == 'HALF' else getenv(key,default)):
      model, _ = Transformer.from_gguf(str(path),16,False)
    assert model.max_context == 16 and not hasattr(model,'placement')
    assert 'use_custom_quant' not in model.output.__dict__
    np.testing.assert_array_equal(model.token_embd.weight.numpy(),expected['token_embd.weight'])


class TestPlacementMemory:
  @pytest.mark.parametrize('counts', [(4,),(1,3),(2,2),(1,1,2)])
  @pytest.mark.parametrize('tied', [False,True])
  def test_exact_arithmetic_without_devices(self, tmp_path, counts, tied):
    from tinygrad.llm.placement import estimate_placement, PlacementMemory
    ts, kv, _ = llama_fixture(typ=2,tied=tied)
    ts += [('rope_freqs.weight',(8,),0,bytes(32))]
    path, _ = save_llama(tmp_path,tensors=ts,metadata=kv)
    devices = tuple('CPU' if i == 0 else f'CPU:{i}' for i in range(len(counts)))
    with no_allocations(), patch.object(Tensor,'__init__',side_effect=AssertionError('Tensor in estimate')):
      idx = index_gguf(path)
      config, placement = model_module.preflight_placement(idx,LayerPlacement(devices,counts,chunk_size=4),max_context=16)
      memory = estimate_placement(idx,config,placement)
    # Q4_0: 18 bytes/32 elements. One block: two F32 norms, Q/K/V/O plus gate/up/down.
    block_bytes = 2*32*4 + (32*32*2 + 16*32*2 + 64*32*3)//32*18
    emb_bytes, norm_bytes = 32*32//32*18, 32*4
    expected = []
    for i,(device,count) in enumerate(zip(devices,counts)):
      weights = count*block_bytes + (emb_bytes if i == 0 else 0)
      weights += (norm_bytes + (0 if tied and len(counts) == 1 else emb_bytes)) if i == len(counts)-1 else 0
      replica = emb_bytes if tied and len(counts)>1 and i == len(counts)-1 else 0
      expected.append(PlacementMemory(device,weights,replica,4*count*16*1*16,16*16*4,2*4*32*4,64*32//32*18))
    assert memory == tuple(expected)
    with pytest.raises(dataclasses.FrozenInstanceError): memory[0].weight_bytes = 0
    # Conditional synthetic budget only, never a fit promise. Logits are additional final-owner scratch.
    known = sum(m.weight_bytes + m.kv_bytes + m.rope_bytes + m.boundary_bytes for m in memory) + config.vocab_size*4
    supplied_workspace = 12345
    assert known + supplied_workspace <= known + 12345
    assert not known + supplied_workspace <= known + 12344

  @pytest.mark.parametrize('vocab', [8,2048])
  def test_largest_payload_is_owner_specific(self, tmp_path, vocab):
    from tinygrad.llm.placement import estimate_placement
    ts, kv, _ = llama_fixture(tied=False)
    kv['tokenizer.ggml.tokens'] = (9,(8,[str(i) for i in range(vocab)]))
    # Small/large F32 embedding and F16 explicit output exercise both endpoint and block maxima.
    ts = [(n,(vocab,32),t,bytes(vocab*32*4)) if n == 'token_embd.weight' else
          (n,(vocab,32),1,bytes(vocab*32*2)) if n == 'output.weight' else (n,s,t,raw) for n,s,t,raw in ts]
    path, _ = save_llama(tmp_path,tensors=ts,metadata=kv)
    with no_allocations():
      idx = index_gguf(path)
      config, placement = model_module.preflight_placement(idx,LayerPlacement(('CPU','CPU:1','CPU:2'),(1,2,1),chunk_size=1))
      first, middle, last = estimate_placement(idx,config,placement)
    assert first.largest_payload_bytes == max(vocab*32*4,64*32*4)
    assert middle.largest_payload_bytes == 64*32*4
    assert last.largest_payload_bytes == max(vocab*32*2,64*32*4)
    assert all(m.tied_replica_bytes == 0 for m in (first,middle,last))

  def test_raw_not_decoded_and_repeated_descriptors(self, tmp_path):
    from tinygrad.llm.placement import estimate_placement
    path, _ = save_llama(tmp_path,typ=14,dim=256,blocks=2)
    with no_allocations():
      idx = index_gguf(path)
      config, placement = model_module.preflight_placement(idx,LayerPlacement(('cpu:00',),(2,),chunk_size=1))
      memory, = estimate_placement(idx,config,placement)
      # Accounting remains descriptor+owner deduplicated even with repeated inventory entries.
      repeated = dataclasses.replace(idx,tensors=idx.tensors + idx.tensors)
      assert estimate_placement(repeated,config,placement) == (memory,)
    assert memory.weight_bytes == sum(len(raw) for _,_,_,raw in llama_fixture(dim=256,blocks=2,typ=14)[0])
    assert memory.largest_payload_bytes == 256*512//256*210
    assert memory.tied_replica_bytes == 0
    assert memory.rope_bytes == 32*128*4  # one unique table, not two blocks

if __name__ == '__main__': unittest.main()

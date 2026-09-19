"""Mandatory CPU logical-device execution tests. No downloads or GPU availability skips."""
import gc, weakref
from unittest.mock import patch
import numpy as np
import pytest
from tinygrad import Tensor, UOp, dtypes, nn
from tinygrad.device import Device
from tinygrad.llm import model as mm
from tinygrad.llm.model import Transformer, TransformerConfig
from tinygrad.llm.placement import LayerPlacement
from test.unit.test_llm_placement import save_llama, buffer_roots


def reference(values, *, dim=32, blocks=4, half=0, context=32):
  # Independent single-device model, with NumPy's GGUF row transform, not the placed loader/coordinator.
  cfg = TransformerConfig(blocks,dim,dim*2,2,1,1e-5,32,dim//2,10000.0,dim//2,dim//2,max_context=context)
  ref = Transformer(cfg, _shape_only=True)
  for name,t in nn.state.get_state_dict(ref).items():
    v = values['token_embd.weight' if name == 'output.weight' and name not in values else name].copy()
    if half: v = v.astype(np.float16)
    if name.endswith(('attn_q.weight','attn_k.weight')):
      heads = 2 if 'attn_q' in name else 1
      v = v.reshape(heads,dim//heads//2 if heads == 2 else dim//4,2,dim).transpose(0,2,1,3).reshape(t.shape)
    t.replace(Tensor(v,device='CPU').realize())
  return ref


def load_pair(tmp_path, counts=(1,3), mode='host', half=0, chunk=4, context=32, **kwargs):
  path, values = save_llama(tmp_path, **kwargs)
  devices = tuple('CPU' if i == 0 else f'CPU:{i}' for i in range(len(counts)))
  with patch.object(mm,'getenv',return_value=half):
    model, _ = Transformer.from_gguf(path,context,False,placement=LayerPlacement(devices,counts,mode,chunk))
  return model, reference(values,dim=kwargs.get('dim',32),blocks=sum(counts),half=half,context=context)


def ref_step(ref, ids, pos):
  x = ref.token_embd(Tensor([ids],dtype=dtypes.int32,device='CPU')).float()
  sp = UOp.variable('start_pos',0,ref.max_context-1).bind(pos)
  for block in ref.blk: x = block(x,sp)
  return ref.output(ref.output_norm(x[:, -1:]))[:, -1, :].numpy().copy()


def placed_logits(model, ids, pos, *, jit=False):
  tokens = Tensor([ids],dtype=dtypes.int32,device=model.placement.devices[0]).realize()
  return model._placed_step(tokens,pos,jit=jit).numpy().copy()


def assert_parity(model, ref, actual, expected, valid, *, half=False):
  atol, rtol = (3e-3,3e-3) if half else (1e-5,1e-4)
  np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol,
                             err_msg=f'logits max error {np.max(np.abs(actual-expected))}')
  assert np.max(np.abs(expected)) > 1e-3 and np.isfinite(expected).all()
  for i,(a,b) in enumerate(zip(model.blk,ref.blk)):
    av = a.cache_kv.numpy()[:, :, :, :valid, :].copy()
    bv = b.cache_kv.numpy()[:, :, :, :valid, :].copy()
    np.testing.assert_allclose(av,bv,rtol=rtol,atol=atol,err_msg=f'block {i} KV max error {np.max(np.abs(av-bv))}')
    assert np.max(np.abs(bv)) > 1e-3


class TestPlacedExecution:
  @pytest.mark.parametrize('counts', [(4,),(2,2),(1,3),(1,2,1)])
  @pytest.mark.parametrize('mode', ['host','native'])
  @pytest.mark.parametrize('length', [1,4,5,6,7,8,9])
  def test_eager_logits_and_every_valid_kv(self,tmp_path,counts,mode,length):
    model, ref = load_pair(tmp_path,counts,mode)
    ids = [(i*7+3)%32 for i in range(length)]
    for pos in range(0,length,4):
      chunk = ids[pos:pos+4]
      actual = placed_logits(model,chunk,pos)
      expected = ref_step(ref,chunk,pos)
      assert_parity(model,ref,actual,expected,pos+len(chunk))

  @pytest.mark.parametrize('mode', ['host','native'])
  def test_owner_state_and_boundary_only_copies(self,tmp_path,mode):
    model, ref = load_pair(tmp_path,(1,2,1),mode)
    weights = {name:t for name,t in nn.state.get_state_dict(model).items() if name.endswith('.weight')}
    persistent_roots = {u for t in weights.values() for u in buffer_roots(t)}
    for i,b in enumerate(model.blk):
      owner = model.placement.block_device(i)
      assert b.cache_kv.device == b.freqs_cis.device == owner
      assert b.cache_kv.dtype == dtypes.half and b.freqs_cis.dtype == dtypes.float32
      for t in (b.cache_kv,b.freqs_cis):
        roots = buffer_roots(t)
        assert roots and all(u.buffer.device == owner and u.buffer.is_allocated() for u in roots)
        persistent_roots.update(roots)
      np.testing.assert_array_equal(b.cache_kv.numpy(),0)
    original = mm._transfer_activation
    calls = []
    def transfer(x,destination,mode):
      assert x.shape == (1,3,32) and x.dtype == dtypes.float32
      assert buffer_roots(x).isdisjoint(persistent_roots)
      calls.append((x.device,destination,mode))
      return original(x,destination,mode)
    with patch.object(mm,'_transfer_activation',transfer): actual = placed_logits(model,[2,9,6],0)
    assert calls == [('CPU','CPU:1',mode),('CPU:1','CPU:2',mode)]
    assert_parity(model,ref,actual,ref_step(ref,[2,9,6],0),3)

  @pytest.mark.parametrize('mode', ['host','native'])
  @pytest.mark.parametrize('dtype', [dtypes.float32,dtypes.int32])
  def test_transfer_values_lifetime_and_same_owner(self,mode,dtype):
    x = Tensor([[1,7,3],[9,2,8]],dtype=dtype,device='CPU').transpose(0,1)
    expected = x.numpy().copy()
    refs, synced = [], []
    orig_to, orig_sync = Tensor.to, Device['CPU:1'].synchronize
    def to(t,device):
      if mode == 'host':
        assert t.device == 'PYTHON'
        refs.append(weakref.ref(t))
      return orig_to(t,device)
    def sync():
      gc.collect()
      if mode == 'host': assert refs and all(r() is not None for r in refs)
      synced.append(True)
      return orig_sync()
    with patch.object(Tensor,'to',to), patch.object(Device['CPU:1'],'synchronize',sync):
      out = mm._transfer_activation(x,'CPU:1',mode)
    assert synced and out.dtype == dtype and out.shape == (3,2)
    np.testing.assert_array_equal(out.numpy(),expected)
    assert all(u.buffer.device == 'CPU:1' for u in buffer_roots(out))
    with patch.object(Tensor,'to',side_effect=AssertionError('same-owner copy')):
      same = mm._transfer_activation(out,'CPU:0' if out.device == 'CPU' else 'CPU:1',mode)
    assert buffer_roots(same) == buffer_roots(out)
    gc.collect()
    assert all(r() is None for r in refs)

  @pytest.mark.parametrize('length', [1,4,5,6,7,8,9,31,32])
  def test_generation_concrete_chunks_sample_only_final(self,tmp_path,length):
    model, ref = load_pair(tmp_path)
    ids = [(i*7+3)%32 for i in range(length)]
    calls, samples = [], []
    step, sample = model._placed_step, model._sample
    def record(tokens,pos,**kwargs):
      assert all(type(n) is int for n in tokens.shape)
      assert tokens.device == 'CPU' and tokens.dtype == dtypes.int32
      calls.append((tokens.shape,pos))
      return step(tokens,pos,**kwargs)
    def sampling(logits,temp):
      assert logits.device == temp.device == 'CPU:1'
      samples.append(logits.numpy().copy())
      return sample(logits,temp)
    gen = model.generate(ids.copy())
    with patch.object(model,'_placed_step',record), patch.object(model,'_sample',sampling):
      try:
        if length == 32:
          assert list(gen) == []
          assert calls == samples == []
          return
        token = next(gen)
      finally: gen.close()
    assert calls == [((1,min(4,length-pos)),pos) for pos in range(0,length,4)]
    assert len(samples) == 1
    for pos in range(0,length,4): expected = ref_step(ref,ids[pos:pos+4],pos)
    assert_parity(model,ref,samples[0],expected,length)
    assert token == int(expected.argmax(-1)[0]) and model._cached_tokens == ids

  @pytest.mark.parametrize('temperature', [0,1e-12,0.7])
  def test_temperature_formula_controlled_uniform(self,temperature):
    logits = np.array([[0.3,0.5,0.49,-0.4]],dtype=np.float32)
    uniform = np.array([[0.98,0.1,0.7,0.5]],dtype=np.float32)
    x = Tensor(logits,device='CPU:1')
    t = Tensor([temperature],dtype=dtypes.float32,device='CPU:1')
    def rand_like(value):
      assert value.device == 'CPU:1' and value.shape == (1,4)
      return Tensor(uniform,device=value.device)
    with patch.object(Tensor,'rand_like',side_effect=rand_like) as rand:
      out = Transformer._sample(x,t)
      assert out.device == 'CPU:1'
      actual = out.item()
    expected = (logits / max(temperature,1e-12) - np.log(-np.log(np.maximum(uniform,1e-12)))).argmax(-1)[0]
    assert actual == expected and rand.call_count == 1

  def test_transfer_rejects_symbolic_and_unsupported_dtype(self):
    for x in (Tensor.ones(2,dtype=dtypes.half,device='CPU'),
              Tensor.ones(4,dtype=dtypes.float32,device='CPU')[:UOp.variable('count',1,4).bind(2)]):
      with pytest.raises(ValueError): mm._transfer_activation(x,'CPU:1','host')
    with pytest.raises(ValueError): mm._transfer_activation(Tensor([1],device='CPU'),'CPU:1','auto')

"""Mandatory CPU logical-device execution tests. No downloads or GPU availability skips."""
import dataclasses, gc, struct, weakref, threading
from unittest.mock import patch
import numpy as np
import pytest
from tinygrad import Tensor, UOp, dtypes, nn
from tinygrad.device import Device
from tinygrad.llm import model as mm
from tinygrad.llm.model import Transformer, TransformerConfig
from tinygrad.llm.placement import LayerPlacement
from test.unit.test_llm_placement import llama_fixture, save_llama, buffer_roots
from test.unit.test_gguf_placement import build_gguf


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
  with model._placed_transaction('test') as owner:
    return model._placed_step(tokens,pos,jit=jit,owner=owner).numpy().copy()


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
  @pytest.mark.parametrize('jit', [False,True])
  def test_logits_and_every_valid_kv(self,tmp_path,counts,mode,length,jit):
    model, ref = load_pair(tmp_path,counts,mode)
    ids = [(i*7+3)%32 for i in range(length)]
    for pos in range(0,length,4):
      chunk = ids[pos:pos+4]
      actual = placed_logits(model,chunk,pos,jit=jit)
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

  def test_nondefault_first_owner_tokens_and_final_owner_temperature(self,tmp_path):
    path, values = save_llama(tmp_path)
    with patch.object(mm,'getenv',return_value=0):
      model, _ = Transformer.from_gguf(path,placement=LayerPlacement(('CPU:2','CPU:1'),(2,2),chunk_size=4))
    ref = reference(values)
    samples = []
    sample = model._sample
    def sampling(logits,temp):
      assert logits.device == temp.device == 'CPU:1'
      samples.append(logits.numpy().copy())
      return sample(logits,temp)
    original = mm._LayerStage.run
    calls = []
    def stage(s,value,*args,**kwargs):
      assert value.device == s.device
      if s.device == 'CPU:2':
        assert value.dtype == dtypes.int32
        calls.append(value.shape)
      return original(s,value,*args,**kwargs)
    with patch.object(mm._LayerStage,'run',stage), patch.object(model,'_sample',sampling):
      gen = model.generate([3,1,9,7,2])
      try: next(gen)
      finally: gen.close()
    assert calls == [(1,4),(1,1)] and len(samples) == 1
    ref_step(ref,[3,1,9,7],0)
    assert_parity(model,ref,samples[0],ref_step(ref,[2],4),5)

  @pytest.mark.parametrize('mode', ['host','native'])
  @pytest.mark.parametrize('dtype', [dtypes.float32,dtypes.int32])
  def test_transfer_values_lifetime_and_same_owner(self,mode,dtype):
    x = Tensor([[1,7,3],[9,2,8]],dtype=dtype,device='CPU').transpose(0,1)
    expected = x.numpy().copy()
    refs, buffer_refs, synced = [], [], []
    orig_to, orig_sync = Tensor.to, Device['CPU:1'].synchronize
    def to(t,device):
      if mode == 'host':
        assert t.device == 'PYTHON'
        refs.append(weakref.ref(t))
        buffer_refs.append(weakref.ref(t.uop.buffer))
      return orig_to(t,device)
    def sync():
      gc.collect()
      if mode == 'host': assert refs and all(r() is not None for r in refs+buffer_refs)
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
    assert all(r() is None for r in refs+buffer_refs)

  @pytest.mark.parametrize('counts', [(4,),(1,3)])
  @pytest.mark.parametrize('warmup', [False,True])
  def test_coordinator_uploads_retain_sources_until_sync(self,tmp_path,counts,warmup):
    from contextlib import ExitStack
    model, _ = load_pair(tmp_path,counts=counts,chunk=1)
    last = model.placement.devices[-1]
    logits = Tensor.zeros(1,32,device=last).contiguous().realize()
    sampled = Tensor([[3]],dtype=dtypes.int32,device=last).realize()
    pending, completed = [], []
    copy, frompy, init = UOp.copy_to_device, UOp._frompy, Tensor.__init__
    def construct(tensor,data,*args,**kwargs):
      if isinstance(data,(list,tuple,bytes)):
        assert kwargs.get('device') == 'PYTHON', 'coordinator must retain an explicit host source before uploading'
      init(tensor,data,*args,**kwargs)
    def capture(src,device,*args,**kwargs):
      if src.device == 'PYTHON' and device in model.placement.devices:
        roots = buffer_roots(Tensor(src))
        assert len(roots) == 1
        buf = next(iter(roots)).buffer
        pending.append((device,weakref.ref(buf),bytes(buf.host)))
      return copy(src,device,*args,**kwargs)
    def allocate(*args,**kwargs):
      assert not pending, 'new host allocation before preceding coordinator upload completed'
      return frompy(*args,**kwargs)
    def sync(device,original):
      gc.collect()
      for dst,ref,value in pending:
        if dst == device:
          assert ref() is not None, 'upload source was released before destination synchronization'
          assert bytes(ref().host) == value
      original()
      completed.extend(value for dst,_,value in pending if dst == device)
      pending[:] = [item for item in pending if item[0] != device]
    def step(tokens,*args,**kwargs):
      assert not pending, 'coordinator input was not synchronized before stage execution'
      assert tokens.tolist() == [[3]] if not warmup else tokens.shape == (1,1)
      return logits
    def sample(_logits,temp):
      assert not pending and temp.item() == 0.75
      return sampled
    with ExitStack() as stack:
      for device in model.placement.devices:
        original = Device[device].synchronize
        stack.enter_context(patch.object(Device[device],'synchronize',lambda d=device,f=original: sync(d,f)))
      stack.enter_context(patch.object(Tensor,'__init__',construct))
      stack.enter_context(patch.object(UOp,'copy_to_device',capture))
      stack.enter_context(patch.object(UOp,'_frompy',allocate))
      stack.enter_context(patch.object(model,'_placed_step',step))
      stack.enter_context(patch.object(model,'_placed_sample',sample))
      if warmup: model.warmup()
      else:
        gen = model.generate([3],temperature=0.75)
        try: assert next(gen) == 3
        finally: gen.close()
    assert not pending
    assert completed == ([struct.pack('<i',i) for i in range(3)] if warmup else [struct.pack('<f',0.75),struct.pack('<i',3)])

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


class TestPlacedJit:
  @pytest.mark.parametrize('counts', [(4,),(2,2),(1,3),(1,2,1)])
  @pytest.mark.parametrize('mode', ['host','native'])
  @pytest.mark.parametrize('count', [1,4])
  def test_changing_inputs_eager_capture_two_replays(self,tmp_path,counts,mode,count):
    model, ref = load_pair(tmp_path,counts,mode)
    state = nn.state.get_state_dict(model)
    kv_ids = [(id(b.cache_kv),buffer_roots(b.cache_kv)) for b in model.blk]
    snapshots = []
    with patch.object(model,'prefill_jit',side_effect=AssertionError('nested JIT')), \
         patch.object(model,'rollout_jit',side_effect=AssertionError('nested JIT')):
      for step in range(4):
        ids = [(3+step*5+i*7)%32 for i in range(count)]
        actual = placed_logits(model,ids,step*count,jit=True)
        expected = ref_step(ref,ids,step*count)
        assert_parity(model,ref,actual,expected,(step+1)*count)
        snapshots.append(actual)
    assert not np.array_equal(snapshots[0],snapshots[-1])
    assert kv_ids == [(id(b.cache_kv),buffer_roots(b.cache_kv)) for b in model.blk]
    after = nn.state.get_state_dict(model)
    assert set(after) == set(state) and all(after[name] is t for name,t in state.items())
    assert not hasattr(model._placed,'__dict__') and nn.state.get_state_dict(model._placed) == {}
    for stage in model._placed.stages:
      used, unused = (stage.single_jit,stage.chunk_jit) if count == 1 else (stage.chunk_jit,stage.single_jit)
      assert used.cnt == 4 and used.captured is not None and unused.cnt == 0
      assert used.captured.ret.device == stage.device
    for name,t in after.items():
      if name.endswith('.weight'):
        assert all(u.buffer.device == model.placement.owner(name) for u in buffer_roots(t))

  @pytest.mark.parametrize('half', [0,1])
  @pytest.mark.parametrize('typ', [0,1,30])
  def test_native_formats_half_policy_full_chunk_replay(self,tmp_path,half,typ):
    model, ref = load_pair(tmp_path,half=half,typ=typ)
    for step in range(4):
      ids = [(step*5+i*7+3)%32 for i in range(4)]
      actual = placed_logits(model,ids,step*4,jit=True)
      assert_parity(model,ref,actual,ref_step(ref,ids,step*4),(step+1)*4,half=half or typ == 1)

  def test_same_input_buffers_two_shape_captures_are_isolated(self,tmp_path):
    model, ref = load_pair(tmp_path)
    inputs = {c:Tensor.empty(1,c,dtype=dtypes.int32,device='CPU').realize() for c in (1,4)}
    roots = {c:buffer_roots(t) for c,t in inputs.items()}
    pos = 0
    for iteration in range(4):
      for count in (1,4):
        ids = [(iteration*5+i*7+3)%32 for i in range(count)]
        tokens = inputs[count]
        tokens.assign(Tensor([ids],dtype=dtypes.int32,device='CPU')).realize()
        with model._placed_transaction('test') as owner:
          actual = model._placed_step(tokens,pos,owner=owner).numpy().copy()
        assert_parity(model,ref,actual,ref_step(ref,ids,pos),pos+count)
        assert buffer_roots(tokens) == roots[count]
        pos += count
    assert all(s.single_jit.cnt == s.chunk_jit.cnt == 4 for s in model._placed.stages)
    assert all(s.single_jit.captured is not s.chunk_jit.captured for s in model._placed.stages)

  @pytest.mark.parametrize('tail', [2,3])
  def test_eager_tail_does_not_create_capture(self,tmp_path,tail):
    model, ref = load_pair(tmp_path)
    for step in range(4):
      ids = [(3+step*5+i*7)%32 for i in range(tail)]
      actual = placed_logits(model,ids,step*tail,jit=True)
      assert_parity(model,ref,actual,ref_step(ref,ids,step*tail),(step+1)*tail)
    assert all(s.single_jit.cnt == s.chunk_jit.cnt == 0 for s in model._placed.stages)

  @pytest.mark.parametrize('typ', [2,8,12,13,14])
  @pytest.mark.parametrize('counts', [(2,),(1,1)])
  @pytest.mark.parametrize('half,count', [(1,1),(1,4),(0,4)])
  def test_packed_ties_replay_and_detached_roots(self,tmp_path,typ,counts,half,count):
    dim = 256 if typ in (12,13,14) else 32
    ts, kv, values = llama_fixture(dim=dim,blocks=2,typ=typ)
    scaled = []
    # Rescale only packed scale fields by powers of two for nonsaturated execution.
    # The independent scalar loader oracle supplies the expected weights, not the tinygrad decoder.
    factor = 1/4096 if typ in (12,13,14) else 1/1024
    blocksize = {2:18,8:34,12:144,13:176,14:210}[typ]
    for name,shape,t,raw in ts:
      if t == typ:
        out = bytearray(raw)
        for start in range(0,len(raw),blocksize):
          offsets = (208,) if typ == 14 else (0,2) if typ in (12,13) else (0,)
          for offset in offsets: struct.pack_into('<e',out,start+offset,struct.unpack_from('<e',raw,start+offset)[0]*factor)
        raw = bytes(out)
        values[name] *= factor
      scaled.append((name,shape,t,raw))
    path = tmp_path/'packed.gguf'
    path.write_bytes(build_gguf(scaled,kv))
    devices = ('CPU',) if len(counts) == 1 else ('CPU','CPU:1')
    with patch.object(mm,'getenv',return_value=half):
      model, _ = Transformer.from_gguf(path,placement=LayerPlacement(devices,counts,chunk_size=4))
    ref = reference(values,dim=dim,blocks=2,half=half)
    weights = {n:t for n,t in nn.state.get_state_dict(model).items() if n.endswith('.weight')}
    roots = {n:buffer_roots(t) for n,t in weights.items()}
    path.unlink()
    for step in range(4):
      ids = [(step*3+i*7+7)%32 for i in range(count)]
      actual = placed_logits(model,ids,step*count,jit=True)
      assert_parity(model,ref,actual,ref_step(ref,ids,step*count),(step+1)*count,half=True)
    for stage in model._placed.stages:
      assert (stage.single_jit if count == 1 else stage.chunk_jit).captured is not None
    if count == 4:
      actual = placed_logits(model,[3,11],16,jit=True)
      assert_parity(model,ref,actual,ref_step(ref,[3,11],16),18,half=True)
      assert all(s.chunk_jit.cnt == 4 and s.single_jit.cnt == 0 for s in model._placed.stages)
    assert {n:buffer_roots(t) for n,t in weights.items()} == roots
    assert roots['output.weight'] == roots['token_embd.weight'] if len(counts) == 1 else \
      roots['output.weight'].isdisjoint(roots['token_embd.weight'])
    for name,t in weights.items():
      root, = buffer_roots(t)
      assert root.buffer.device == model.placement.owner(name)
      expected_name = 'token_embd.weight' if name == 'output.weight' else name
      assert root.buffer.nbytes == len(next(raw for n,_,_,raw in scaled if n == expected_name))

class TestLegacyPublicPlacementIsolation:
  def test_legacy_gguf_public_generation_chunk_defaults_and_prefix(self,tmp_path):
    path, _ = save_llama(tmp_path,tied=False)
    def load():
      with patch.object(mm,'getenv',return_value=0): return Transformer.from_gguf(path,realize=False)[0]
    def take(model,prompt,**kwargs):
      gen = model.generate(prompt,**kwargs)
      try: return [next(gen) for _ in range(4)]
      finally: gen.close()
    left, right = load(), load()
    with patch.object(Transformer,'_placed_step',side_effect=AssertionError('placed dispatch in legacy')):
      history = [3,1,9,7,2]
      assert take(left,history) == take(right,[3,1,9,7,2],chunk_size=32)
      assert left.prefill_jit.cnt > 0 and left.rollout_jit.cnt >= 3
      assert left.get_start_pos(history+[4]) == len(history)-1
      assert take(left,history+[4]) == take(load(),history+[4],chunk_size=32)
      assert not hasattr(left,'placement')

# Lifecycle tests use public entrypoints, never a private step to bypass ownership.
class TestPlacedLifecycle:
  def test_direct_admission_race_has_no_unowned_input_reads(self,tmp_path):
    model, _ = load_pair(tmp_path)
    tokens = Tensor([[3]],dtype=dtypes.int32,device='CPU').realize()
    temp = Tensor([0.],dtype=dtypes.float32,device='CPU:1').realize()
    checked, proceed = threading.Event(), threading.Event()
    errors, unowned_reads = [], []
    check, item = model.check_available, Tensor.item
    def availability():
      check()
      if threading.current_thread() is thread:
        checked.set()
        assert proceed.wait(5)
    def guarded_item(t):
      if threading.current_thread() is thread and model._placed.owner is not None:
        unowned_reads.append(t)
      return item(t)
    def direct():
      try: model(tokens,0,temp)
      except Exception as exc: errors.append(exc)
    thread = threading.Thread(target=direct)
    with patch.object(model,'check_available',availability), patch.object(Tensor,'item',guarded_item):
      thread.start()
      try:
        assert checked.wait(5)
        # A paused before admission must not read backend data while this owner is active.
        if model._placed.lock.acquire(blocking=False):
          try:
            model._placed.owner, model._placed.state = object(), 'generating'
            proceed.set()
            thread.join(5)
          finally:
            model._placed.owner, model._placed.state = None, 'ready'
            model._placed.lock.release()
          assert len(errors) == 1 and isinstance(errors[0],mm.PlacedModelBusyError)
          assert not unowned_reads
        else:
          # Admission was already atomic: our competitor fails before any mutation/read.
          with pytest.raises(mm.PlacedModelBusyError): next(model.generate([4]))
          proceed.set()
          thread.join(5)
          assert not errors
      finally:
        proceed.set()
        thread.join(5)
    assert not thread.is_alive() and model._placed.state == 'ready'

  def test_clean_close_competitors_and_reset(self,tmp_path):
    model, _ = load_pair(tmp_path)
    gen = model.generate([3,1,9,7,2])
    first = next(gen)
    assert model._placed.state == 'generating' and model._cached_tokens == [3,1,9,7,2]
    tokens = Tensor([[2]],dtype=dtypes.int32,device='CPU')
    temp = Tensor([0.],dtype=dtypes.float32,device='CPU:1')
    for competitor in (lambda: next(model.generate([4])),lambda: model(tokens,0,temp),lambda: model.forward(tokens,0,temp),
                       model.reset_generation_state,model.warmup):
      with pytest.raises(mm.PlacedModelBusyError): competitor()
      assert model._placed.state == 'generating' and model._cached_tokens == [3,1,9,7,2]
    assert isinstance(next(gen),int) and model._cached_tokens == [3,1,9,7,2,first]
    gen.close()
    assert model._placed.state == 'ready'
    model.reset_generation_state()
    assert model._cached_tokens == [] and model._placed.state == 'ready'

  @pytest.mark.parametrize('entry', ['forward','call'])
  def test_direct_overwrite_between_generations_invalidates_prefix(self,tmp_path,entry):
    model, ref = load_pair(tmp_path)
    prompt = [3,1,9,7,2]
    gen = model.generate(prompt.copy())
    next(gen)
    gen.close()
    tokens = Tensor([[11,12,13,14]],dtype=dtypes.int32,device='CPU')
    temp = Tensor([0.],dtype=dtypes.float32,device='CPU:1')
    out = (model.forward if entry == 'forward' else model)(tokens,0,temp)
    assert out.uop.is_realized and out.shape == (1,1) and model._cached_tokens == [] and model._placed.state == 'ready'
    with patch.object(model,'_sample',wraps=model._sample) as sample:
      gen = model.generate(prompt.copy())
      try: next(gen)
      finally: gen.close()
    actual = sample.call_args.args[0].numpy().copy()
    for pos in range(0,len(prompt),4): expected = ref_step(ref,prompt[pos:pos+4],pos)
    assert_parity(model,ref,actual,expected,len(prompt))

  @pytest.mark.parametrize('kind', ['extend','diverge','same'])
  def test_committed_prefix_resume_matches_fresh(self,tmp_path,kind):
    model, ref = load_pair(tmp_path)
    history = [3,1,9,7,2]
    gen = model.generate(history)
    next(gen)
    next(gen)
    gen.close()
    prompt = history + [4,8] if kind == 'extend' else [3,1,5,8,7] if kind == 'diverge' else history[:-1]
    start = model.get_start_pos(prompt)
    expected_start = len(history)-1 if kind == 'extend' else 2 if kind == 'diverge' else len(prompt)-1
    assert start == expected_start
    calls = []
    sample = model._sample
    def record(logits,temp):
      calls.append(logits.numpy().copy())
      return sample(logits,temp)
    with patch.object(model,'_sample',record):
      gen = model.generate(prompt.copy())
      try: next(gen)
      finally: gen.close()
    for pos,end in ((0,4),(4,5),(5,6)):
      if pos >= start: break
      ref_step(ref,prompt[pos:min(end,start)],pos)
    for pos in range(start,len(prompt),4): expected = ref_step(ref,prompt[pos:pos+4],pos)
    assert_parity(model,ref,calls[0],expected,len(prompt))

  @pytest.mark.parametrize('kind', ['stage','copy','sample','brokenpipe','resetpipe','unexpected'])
  @pytest.mark.parametrize('mode', ['host','native'])
  def test_owned_failure_is_sanitized_and_requires_reload(self,tmp_path,kind,mode):
    model, _ = load_pair(tmp_path,mode=mode)
    failure = BrokenPipeError('private backend detail') if kind == 'brokenpipe' else \
      ConnectionResetError('private backend detail') if kind == 'resetpipe' else RuntimeError('private backend detail')
    original = mm._LayerStage.run
    def stage(s,*args,**kwargs):
      if s.device == 'CPU:1': raise failure
      return original(s,*args,**kwargs)
    target, attr, replacement = (mm._LayerStage,'run',stage) if kind in ('stage','brokenpipe','resetpipe') else \
      (mm,'_transfer_activation',None) if kind == 'copy' else (model,'_sample',None) if kind == 'sample' else (model,'get_start_pos',None)
    with patch.object(target,attr,**({'new':replacement} if replacement else {'side_effect':failure})):
      gen = model.generate([3,1,9,7,2])
      with pytest.raises(mm.PlacedInferenceError) as error: next(gen)
      gen.close()
    assert 'private backend detail' not in str(error.value)
    assert not isinstance(error.value,(BrokenPipeError,ConnectionResetError))
    assert error.value.operation in ('stage','transfer','sample','generation')
    assert model._placed.state == 'failed' and model._cached_tokens == []
    if kind != 'unexpected': assert np.any(model.blk[0].cache_kv.numpy() != 0)
    for attempt in (lambda: next(model.generate([2])),model.reset_generation_state,model.warmup):
      with pytest.raises(mm.PlacedModelUnavailableError): attempt()
    fresh, _ = load_pair(tmp_path)
    gen = fresh.generate([2])
    try: assert isinstance(next(gen),int)
    finally: gen.close()

  @pytest.mark.parametrize('chunk', [1,4])
  def test_warmup_capture_replay_no_state_replacement(self,tmp_path,chunk):
    model, ref = load_pair(tmp_path,chunk=chunk)
    kv_ids = [(id(b.cache_kv),buffer_roots(b.cache_kv)) for b in model.blk]
    model.warmup()
    assert model._cached_tokens == [] and model._placed.state == 'ready'
    assert kv_ids == [(id(b.cache_kv),buffer_roots(b.cache_kv)) for b in model.blk]
    for stage in model._placed.stages:
      assert stage.single_jit.cnt >= 3
      assert stage.chunk_jit is None if chunk == 1 else stage.chunk_jit.cnt >= 3
    calls = []
    sample = model._sample
    def record(logits,temp):
      calls.append(logits.numpy().copy())
      return sample(logits,temp)
    with patch.object(model,'_sample',record):
      gen = model.generate([3,1,9,7,2])
      try: next(gen)
      finally: gen.close()
    for pos in range(0,5,chunk): expected = ref_step(ref,[3,1,9,7,2][pos:pos+chunk],pos)
    assert_parity(model,ref,calls[0],expected,5)

  @pytest.mark.parametrize('entry', ['forward','call'])
  @pytest.mark.parametrize('kind', ['shape','dtype','owner','symbolic','empty','count','position','unbound','token',
                                   'temp_owner','temp_shape','temp_value'])
  def test_invalid_direct_inputs_do_not_mutate(self,tmp_path,entry,kind):
    model, _ = load_pair(tmp_path)
    model._cached_tokens = [3,1]
    tokens = Tensor([[3]],dtype=dtypes.int32,device='CPU')
    temp = Tensor([0.],dtype=dtypes.float32,device='CPU:1')
    pos = 0
    if kind == 'shape': tokens = tokens.repeat(2,1)
    if kind == 'dtype': tokens = tokens.float()
    if kind == 'owner': tokens = tokens.to('CPU:1')
    if kind == 'symbolic': tokens = tokens.repeat(1,4)[:, :UOp.variable('count',1,4).bind(1)]
    if kind == 'empty': tokens = tokens[:, :0]
    if kind == 'count': tokens = tokens.repeat(1,5)
    if kind == 'position': pos = 32
    if kind == 'unbound': pos = UOp.variable('pos',0,31)
    if kind == 'token': tokens = Tensor([[32]],dtype=dtypes.int32,device='CPU')
    if kind == 'temp_owner': temp = temp.to('CPU')
    if kind == 'temp_shape': temp = temp.repeat(2)
    if kind == 'temp_value': temp = Tensor([float('nan')],dtype=dtypes.float32,device='CPU:1')
    with patch.object(mm._LayerStage,'run',side_effect=AssertionError('mutating invalid input')), pytest.raises(ValueError):
      (model.forward if entry == 'forward' else model)(tokens,pos,temp)
    assert model._cached_tokens == [3,1] and model._placed.state == 'ready'
    assert not model._placed.lock.locked()

  @pytest.mark.parametrize('kwargs', [{'inputs_embeds':1},{'position_ids':1},{'rope_delta':1},{'chunk_size':3},
                                     {'temperature':-1},{'temperature':float('inf')}])
  def test_bad_generation_options_keep_active_owner(self,tmp_path,kwargs):
    model, _ = load_pair(tmp_path)
    gen = model.generate([3,1])
    next(gen)
    try:
      with pytest.raises(ValueError): next(model.generate([2],**kwargs))
      assert model._placed.state == 'generating' and model._cached_tokens == [3,1]
      assert isinstance(next(gen),int)
    finally: gen.close()
    assert model._placed.state == 'ready'

  def test_public_multimodal_seams_rejected_and_configuration_frozen(self,tmp_path):
    model, _ = load_pair(tmp_path)
    tokens = Tensor([[3]],dtype=dtypes.int32,device='CPU')
    with pytest.raises(ValueError): model.forward_embeddings(tokens,0)
    with pytest.raises(ValueError): model._multimodal_decode(tokens,0,tokens,tokens)
    with pytest.raises(ValueError): model.max_context = 64
    with pytest.raises(ValueError): model.placement = dataclasses.replace(model.placement,chunk_size=1)
    with pytest.raises(dataclasses.FrozenInstanceError): model.placement.chunk_size = 1
    with pytest.raises(dataclasses.FrozenInstanceError): model.blk[0].config.max_context = 64
    assert model._placed.state == 'ready' and model._cached_tokens == []
    with pytest.raises(mm.PlacedModelBusyError): model._placed_step(tokens,0)
    gen = model.generate([3,1])
    next(gen)
    try:
      with pytest.raises(mm.PlacedModelBusyError): model._placed_step(tokens,0,owner=object())
      assert model._placed.state == 'generating'
    finally: gen.close()

  @pytest.mark.parametrize('chunk', [1,4])
  @pytest.mark.parametrize('explicit', [False,True])
  def test_chunk_resolution_context_edge_and_sample_counts(self,tmp_path,chunk,explicit):
    model, ref = load_pair(tmp_path,chunk=chunk,context=9)
    ids = [3,1,9,7,2]
    with patch.object(model,'_sample',wraps=model._sample) as sample:
      gen = model.generate(ids,**({'chunk_size':chunk} if explicit else {}))
      result = list(gen)
    assert len(result) == sample.call_count == 4
    assert model._cached_tokens == ids[:-1] and len(model._cached_tokens) == 8
    assert model._placed.state == 'ready' and not model._placed.lock.locked()
    for pos in range(0,5,chunk): ref_step(ref,ids[pos:min(pos+chunk,5)],pos)
    for pos in range(5,8): expected = ref_step(ref,[ids[pos]],pos)
    assert_parity(model,ref,sample.call_args.args[0].numpy().copy(),expected,8)
    # The direct API may fill the final context slot, but not go one past it.
    value = Tensor([[ids[-1]]],dtype=dtypes.int32,device='CPU')
    temp = Tensor([0.],dtype=dtypes.float32,device='CPU:1')
    assert model(value,8,temp).shape == (1,1)
    with pytest.raises(ValueError): model(value,9,temp)

  @pytest.mark.parametrize('entry', ['forward','call'])
  @pytest.mark.parametrize('kind', ['lazy_sample','sync','stage','interrupt'])
  def test_direct_mutation_failures_fail_closed(self,tmp_path,entry,kind):
    model, _ = load_pair(tmp_path)
    model._cached_tokens = [3,1]
    tokens = Tensor([[3]],dtype=dtypes.int32,device='CPU').realize()
    temp = Tensor([0.],dtype=dtypes.float32,device='CPU:1').realize()
    failure = KeyboardInterrupt() if kind == 'interrupt' else BrokenPipeError('backend private text')
    output = Tensor([[7]],dtype=dtypes.int32,device='CPU:1')
    sample = model._sample
    sync = Device['CPU:1'].synchronize
    def synchronized():
      if model._placed.state == 'generating': raise failure
      return sync()
    def sampled(logits,temperature):
      if kind == 'interrupt': raise failure
      if kind == 'lazy_sample':
        output.realize = lambda: (_ for _ in ()).throw(failure)
        return output
      return sample(logits,temperature)
    from contextlib import ExitStack
    with ExitStack() as stack:
      stack.enter_context(patch.object(model,'_sample',sampled))
      if kind == 'sync': stack.enter_context(patch.object(Device['CPU:1'],'synchronize',synchronized))
      if kind == 'stage': stack.enter_context(patch.object(mm._LayerStage,'run',side_effect=failure))
      with pytest.raises(KeyboardInterrupt if kind == 'interrupt' else mm.PlacedInferenceError):
        (model.forward if entry == 'forward' else model)(tokens,0,temp)
    assert model._placed.state == 'failed' and model._cached_tokens == [] and not model._placed.lock.locked()
    with pytest.raises(mm.PlacedModelUnavailableError): model(tokens,0,temp)

  def test_failure_after_committed_token_clears_prefix(self,tmp_path):
    model, _ = load_pair(tmp_path)
    gen = model.generate([3,1,9,7,2])
    next(gen)
    assert model._cached_tokens == [3,1,9,7,2]
    with patch.object(model,'_sample',side_effect=ConnectionResetError('backend private text')):
      with pytest.raises(mm.PlacedInferenceError): next(gen)
    gen.close()
    assert model._placed.state == 'failed' and model._cached_tokens == []

  @pytest.mark.parametrize('tokens', [[],[32],[-1],[True],[1.0],list(range(32))+[0]])
  def test_invalid_prompt_keeps_prefix(self,tmp_path,tokens):
    model, _ = load_pair(tmp_path)
    model._cached_tokens = [3,1]
    with pytest.raises(ValueError): next(model.generate(tokens))
    assert model._cached_tokens == [3,1] and model._placed.state == 'ready'

  def test_partial_prefill_does_not_publish_prefix(self,tmp_path):
    model, _ = load_pair(tmp_path)
    model._cached_tokens = [20,21]
    original = model._placed_step
    calls = []
    def step(*args,**kwargs):
      assert model._cached_tokens == [20,21]
      calls.append(True)
      if len(calls) == 2: raise BrokenPipeError('private text')
      return original(*args,**kwargs)
    with patch.object(model,'_placed_step',step), patch.object(model,'_sample',wraps=model._sample) as sample:
      gen = model.generate([3,1,9,7,2])
      with pytest.raises(mm.PlacedInferenceError): next(gen)
      gen.close()
    assert len(calls) == 2 and sample.call_count == 0
    assert model._cached_tokens == [] and model._placed.state == 'failed'

  def test_direct_transaction_clears_before_write_and_finishes_output_before_release(self,tmp_path):
    model, _ = load_pair(tmp_path)
    model._cached_tokens = [20,21]
    tokens = Tensor([[3]],dtype=dtypes.int32,device='CPU').realize()
    temp = Tensor([0.],dtype=dtypes.float32,device='CPU:1').realize()
    original, events = model._placed_step, []
    sync = Device['CPU:1'].synchronize
    def step(*args,**kwargs):
      assert model._cached_tokens == [] and model._placed.state == 'generating' and model._placed.lock.locked()
      events.append('step')
      return original(*args,**kwargs)
    def synchronize():
      if events: assert model._placed.state == 'generating' and model._placed.lock.locked()
      return sync()
    with patch.object(model,'_placed_step',step), patch.object(Device['CPU:1'],'synchronize',synchronize):
      out = model(tokens,0,temp)
    assert events == ['step'] and out.uop.is_realized
    assert model._cached_tokens == [] and model._placed.state == 'ready' and not model._placed.lock.locked()

  def test_check_available_and_warmup_failure(self,tmp_path):
    model, _ = load_pair(tmp_path)
    assert model.check_available() is None
    with patch.object(mm._LayerStage,'run',side_effect=BrokenPipeError('private text')):
      with pytest.raises(mm.PlacedInferenceError): model.warmup()
    assert model._cached_tokens == [] and model._placed.state == 'failed' and not model._placed.lock.locked()
    with pytest.raises(mm.PlacedModelUnavailableError): model.check_available()

  def test_unowned_backend_validation_failure_is_wrapped_but_not_poisoned(self,tmp_path):
    model, _ = load_pair(tmp_path)
    model._cached_tokens = [3,1]
    tokens = Tensor([[3]],dtype=dtypes.int32,device='CPU').realize()
    temp = Tensor([0.],dtype=dtypes.float32,device='CPU:1').realize()
    with patch.object(Device['CPU:1'],'synchronize',side_effect=BrokenPipeError('private text')):
      with pytest.raises(mm.PlacedInferenceError) as error: model(tokens,0,temp)
    assert error.value.operation == 'input' and error.value.device == 'CPU:1'
    assert model._cached_tokens == [3,1] and model._placed.state == 'ready' and not model._placed.lock.locked()

  def test_caller_history_mutation_does_not_publish_invalid_prefix(self,tmp_path):
    model, _ = load_pair(tmp_path)
    tokens = [3,1,9]
    gen = model.generate(tokens)
    first = next(gen)
    tokens[0] = 12
    second = next(gen)
    gen.close()
    assert model._cached_tokens == [3,1,9,first] and tokens == [12,1,9,first,second]

  def test_owned_unexpected_generator_exit_fails_not_clean_close(self,tmp_path):
    model, _ = load_pair(tmp_path)
    with patch.object(model,'_sample',side_effect=GeneratorExit):
      gen = model.generate([3])
      with pytest.raises(GeneratorExit): next(gen)
    assert model._placed.state == 'failed' and model._cached_tokens == []

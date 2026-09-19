"""Offline GGUF fixtures: shapes use tinygrad order; no external GGUF writer/oracle required."""
import dataclasses, gc, hashlib, os, pathlib, struct, weakref
from unittest.mock import patch
from typing import Any
from tinygrad.device import Buffer
import pytest
from tinygrad import Device, Tensor
from tinygrad.llm import gguf

# Explicit GGUF metadata type tags. Arrays are (element_type, values), including nested arrays.
_FORMATS = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}

def _string(value: str | bytes) -> bytes:
  raw = value.encode('utf-8') if isinstance(value, str) else value
  return struct.pack('<Q', len(raw)) + raw

def _value(typ: int, value) -> bytes:
  if typ == 8: return _string(value)
  if typ == 9:
    subtype, values = value
    return struct.pack('<IQ', subtype, len(values)) + b''.join(_value(subtype, v) for v in values)
  return struct.pack('<' + _FORMATS[typ], value)

def build_gguf(tensors=(), metadata=(), *, alignment: int = 32, version: int = 3, offsets=None) -> bytes:
  """Build aligned multi-tensor GGUF bytes. tensors: (name, shape, ggml_type, payload).

  metadata: iterable of (key, GGUF type tag, value), or dict of key -> (type tag, value).
  Explicit offsets permit malformed fixtures; alignment metadata is added when nondefault.
  """
  entries = [(k, *v) for k, v in metadata.items()] if isinstance(metadata, dict) else list(metadata)
  if alignment != 32 and not any(k == 'general.alignment' for k, _, _ in entries): entries.append(('general.alignment', 4, alignment))
  header = b'GGUF' + struct.pack('<IQQ', version, len(tensors), len(entries))
  header += b''.join(_string(k) + struct.pack('<I', typ) + _value(typ, v) for k, typ, v in entries)
  payload = bytearray()
  for i, (name, shape, typ, raw) in enumerate(tensors):
    off = (len(payload)+alignment-1)//alignment*alignment if offsets is None else offsets[i]
    header += _string(name) + struct.pack('<I', len(shape)) + b''.join(struct.pack('<Q', d) for d in reversed(shape))
    header += struct.pack('<IQ', typ, off)
    if offsets is None or off < 16_000_000:
      payload.extend(bytes(max(0, off + len(raw) - len(payload))))
      payload[off:off+len(raw)] = raw
  return header + bytes((-len(header)) % alignment) + payload

@pytest.fixture
def path(tmp_path): return tmp_path / 'model.gguf'

def _save(path, data):
  path.write_bytes(data)
  return path

def _index(path, data): return gguf.index_gguf(_save(path, data))

class TestGGUFIndex:
  @pytest.fixture(autouse=True)
  def forbid_allocations(self):
    with patch.object(gguf,'Tensor',side_effect=AssertionError('Tensor during index')), \
         patch.object(type(Device),'__getitem__',side_effect=AssertionError('device during index')), \
         patch('tinygrad.device.Buffer.__init__',side_effect=AssertionError('buffer during index')):
      yield

  def test_all_scalar_types_nested_arrays_and_boundaries(self, path):
    entries: list[tuple[str,int,Any]] = [(f't{typ}',typ,value) for typ,value in [(0,255),(1,-128),(2,65535),(3,-32768),(4,2**32-1),
      (5,-2**31),(6,0.125),(7,False),(10,2**64-1),(11,-2**63),(12,-1.25)]]
    nested: tuple[int, list[Any]] = (4,[42])
    expected: list[Any] = [42]
    for _ in range(7): nested, expected = (9,[nested]), [expected]
    entries += [('depth',9,nested), ('empty',9,(8,[])), ('k'*65535,8,'')]
    idx = _index(path,build_gguf([('n'*64,(1,1,1,1),0,bytes(4))],entries))
    assert idx.kv['depth'] == expected and idx.kv['empty'] == []
    for key,typ,value in entries[:11]: assert idx.kv[key] == value and type(idx.kv[key]) is type(value)

  @pytest.mark.parametrize('version', [2, 3])
  @pytest.mark.parametrize('alignment', [8, 24, 32, 1048576])
  def test_metadata_shapes_readonly_no_allocations(self, path, version, alignment):
    data = build_gguf([('é', (2,3), 0, bytes(24)), ('second', (4,), 1, bytes(8))],
                      [('u0',0,0), ('u255',0,255), ('text',8,'héllo'), ('bool',7,True), ('arr',9,(8,['a','β']))],
                      alignment=alignment, version=version)
    _save(path, data).chmod(0o444)
    before = path.stat()
    with patch.object(gguf, 'Tensor', side_effect=AssertionError('Tensor during index')), \
         patch.object(type(Device), '__getitem__', side_effect=AssertionError('device during index')), \
         patch('tinygrad.device.Buffer.__init__', side_effect=AssertionError('buffer during index')):
      idx = gguf.index_gguf(path)
    assert idx.kv == {'u0':0, 'u255':255, 'text':'héllo', 'bool':True, 'arr':['a','β'],
                      **({'general.alignment':alignment} if alignment != 32 else {})}
    assert [(t.name,t.shape,t.part,t.ggml_type,t.nbytes) for t in idx.tensors] == [('é',(2,3),0,0,24), ('second',(4,),0,1,8)]
    assert all(t.offset % alignment == 0 for t in idx.tensors)
    assert idx.parts == (gguf.GGUFPart(path.resolve(),len(data),(before.st_dev,before.st_ino,len(data),before.st_mtime_ns)),)
    assert path.read_bytes() == data and path.stat().st_mtime_ns == before.st_mtime_ns
    for obj in (idx, idx.parts[0], idx.tensors[0]):
      with pytest.raises(dataclasses.FrozenInstanceError): setattr(obj,'extra',1)

  @pytest.mark.parametrize('data', [b'', b'GGUF', b'FU GG'+bytes(20), b'GGUF'+struct.pack('>IQQ',3,0,0),
    build_gguf(version=1), build_gguf(version=4), b'GGUF'+struct.pack('<IQQ',3,1000001,0),
    b'GGUF'+struct.pack('<IQQ',3,0,1000001), b'GGUF'+struct.pack('<IQQ',3,1,0),
    b'GGUF'+struct.pack('<IQQ',3,0,1)+struct.pack('<Q',65536)])
  def test_bad_header_and_counts(self, path, data):
    with pytest.raises(ValueError): _index(path, data)

  @pytest.mark.parametrize('metadata', [[('a',8,b'\xff')], [(b'\xff',4,1)], [('x',4,1),('x',4,1)],
    [('general.alignment',4,0)], [('general.alignment',4,7)], [('general.alignment',4,1048584)],
    [('general.alignment',8,'32')], [('general.alignment',7,True)], [('x'*65536,4,1)]])
  def test_bad_metadata(self, path, metadata):
    with pytest.raises(ValueError): _index(path, build_gguf(metadata=metadata))

  @pytest.mark.parametrize('tail', [struct.pack('<I',99), struct.pack('<I',7)+b'\x02',
    struct.pack('<IQ',8,16777217), struct.pack('<IIQ',9,4,10000001), struct.pack('<IIQ',9,99,0),
    struct.pack('<IIQ',9,4,100), struct.pack('<I',9)+struct.pack('<IQ',9,1)*9+struct.pack('<IQ',4,0)])
  def test_bad_values(self, path, tail):
    with pytest.raises(ValueError): _index(path, b'GGUF'+struct.pack('<IQQ',3,0,1)+_string('x')+tail)

  @pytest.mark.parametrize('name,shape,typ,raw', [('a',(),0,b''), ('a',(1,1,1,1,1),0,bytes(4)), ('a',(0,),0,b''),
    ('a',(2**63,),0,b''), ('a',(2**32,2**32),0,b''), ('a',(1,),999,b''), ('a'*65,(1,),0,bytes(4)),
    (b'\xff',(1,),0,bytes(4)), ('a',(33,),2,bytes(36)), ('a',(32,1),2,bytes(18)), ('a',(2,128),12,bytes(144))])
  def test_bad_descriptors(self, path, name, shape, typ, raw):
    with pytest.raises(ValueError): _index(path, build_gguf([(name,shape,typ,raw)]))

  @pytest.mark.parametrize('offsets', [[0,0], [0,1], [0,2**64-32]])
  def test_bad_ranges(self, path, offsets):
    with pytest.raises(ValueError): _index(path, build_gguf([('a',(16,),0,bytes(64)),('b',(1,),0,bytes(4))],offsets=offsets))

  def test_duplicates_and_truncation(self, path):
    with pytest.raises(ValueError): _index(path, build_gguf([('a',(1,),0,bytes(4))]*2))
    data = build_gguf([('a',(3,),0,bytes(12))])
    for cut in range(len(data)):
      with pytest.raises(ValueError): _index(path, data[:cut])

  def test_aggregate_limits(self, path):
    data = build_gguf([('a',(1,),0,bytes(4))], [('x',9,(4,[1,2])),('y',9,(4,[3,4]))])
    for name, cap in [('_GGUF_MAX_METADATA',64), ('_GGUF_MAX_ARRAY',3), ('_GGUF_MAX_KEYS',1), ('_GGUF_MAX_TENSORS',0)]:
      with patch.object(gguf,name,cap), pytest.raises(ValueError): _index(path,data)

  def _parts(self, tmp_path, later=(), first=(), *, second_name='b'):
    paths = [tmp_path / f'model-{i:05}-of-00002.gguf' for i in (1,2)]
    for no, (pp, name, extra) in enumerate(zip(paths, ['a',second_name], [first,later])):
      _save(pp, build_gguf([(name,(1,),0,bytes(4))], [('split.no',2,no), ('split.count',2,2),
        ('split.tensors.count',10,2), ('general.architecture',8,'llama'), ('llama.block_count',4,2), *extra]))
    return paths

  def test_split_merge_omitted_metadata(self, tmp_path):
    a,b = self._parts(tmp_path, first=[('tokenizer.ggml.tokens',9,(8,['a','b']))])
    idx = gguf.index_gguf(a)
    assert len(idx.parts) == 2 and [(t.name,t.part) for t in idx.tensors] == [('a',0),('b',1)]
    assert idx.kv['tokenizer.ggml.tokens'] == ['a','b']
    with pytest.raises(ValueError): gguf.index_gguf(b)
    b.unlink()
    with pytest.raises((ValueError,FileNotFoundError)): gguf.index_gguf(a)

  @pytest.mark.parametrize('key,typ,value', [('split.no',2,0), ('split.count',2,3), ('split.tensors.count',10,3),
    ('general.architecture',8,'qwen2'), ('llama.block_count',4,3), ('tokenizer.ggml.model',8,'other'),
    ('llama.context_length',4,4096), ('general.quantization_version',4,999), ('general.tensor_data_layout',8,'other')])
  def test_split_conflicts(self, tmp_path, key, typ, value):
    a,b = self._parts(tmp_path)
    metadata = {'split.no':(2,1), 'split.count':(2,2), 'split.tensors.count':(10,2),
                'general.architecture':(8,'llama'), 'llama.block_count':(4,2), key:(typ,value)}
    _save(b,build_gguf([('b',(1,),0,bytes(4))], metadata))
    with pytest.raises(ValueError): gguf.index_gguf(a)

  def test_split_duplicates_identity_and_filename(self, tmp_path):
    a,b = self._parts(tmp_path,second_name='a')
    with pytest.raises(ValueError): gguf.index_gguf(a)
    a,b = self._parts(tmp_path)
    b.unlink()
    os.link(a,b)
    with pytest.raises(ValueError): gguf.index_gguf(a)
    wrong = a.rename(tmp_path / 'model-00001-of-00003.gguf')
    with pytest.raises(ValueError): gguf.index_gguf(wrong)
    with pytest.raises(ValueError): gguf.index_gguf(wrong.rename(tmp_path / 'model.gguf'))

  def test_later_only_architecture(self, tmp_path):
    a,b = self._parts(tmp_path)
    _save(a,build_gguf([('a',(1,),0,bytes(4))], [('split.no',2,0),('split.count',2,2)]))
    _save(b,build_gguf([('b',(1,),0,bytes(4))], [('split.no',2,1),('split.count',2,2),('general.architecture',8,'llama')]))
    with pytest.raises(ValueError): gguf.index_gguf(a)

  @pytest.mark.parametrize('metadata', [[('split.count',2,0)], [('split.count',7,True)], [('split.no',2,1)],
    [('split.count',2,2)], [('split.tensors.count',10,2)]])
  def test_invalid_single_split(self, path, metadata):
    with pytest.raises(ValueError): _index(path,build_gguf([('a',(1,),0,bytes(4))],metadata))

  def test_split_global_caps(self, tmp_path):
    a,_ = self._parts(tmp_path)
    for name, cap in [('_GGUF_MAX_TENSORS',1), ('_GGUF_MAX_KEYS',9)]:
      with patch.object(gguf,name,cap), pytest.raises(ValueError): gguf.index_gguf(a)

  def test_index_fstat_detects_mutation(self, path):
    _save(path,build_gguf([('a',(1,),0,bytes(4))]))
    original = os.fstat
    calls = []
    def changed(fd):
      calls.append(fd)
      if len(calls) == 2: os.utime(path,ns=(path.stat().st_atime_ns,path.stat().st_mtime_ns+1))
      return original(fd)
    with patch.object(gguf.os,'fstat',side_effect=changed), pytest.raises(ValueError): gguf.index_gguf(path)
    assert len(calls) == 2

def quant_block(typ: int, variant: int = 0) -> tuple[bytes, list[float]]:
  """Independent scalar GGML layout oracle; construct bit planes from logical codes, not the tinygrad decoder.

  Layout: ggml-common.h block_q{4,5,6}_K and ggml-quants.c dequantize_row_q*_K.
  Vary every subblock's 6-bit scales/minima, signed Q6 scales and all quant bit planes.
  """
  if typ == 2:
    packed = bytes.fromhex('10 32 54 76 98 ba dc fe ef cd ab 89 67 45 23 01')
    expected: list[float] = [-8,-6,-4,-2,0,2,4,6,7,5,3,1,-1,-3,-5,-7,-7,-5,-3,-1,1,3,5,7,6,4,2,0,-2,-4,-6,-8]
    return struct.pack('<e',0.5) + packed, [v*0.5 for v in expected]
  if typ == 8:
    codes = [-128,-127,-64,-33,-32,-17,-16,-9,-8,-7,-3,-2,-1,0,1,2,3,7,8,9,15,16,17,31,32,33,63,64,65,100,126,127]
    return struct.pack('<e',0.25)+struct.pack('<32b',*codes), [v*0.25 for v in codes]
  if typ in (12,13):
    scales = [1,17,33,63,49,34,18,3]
    minima = [62,35,19,2,4,20,36,61]
    s = bytearray(12)
    for i in range(4):
      s[i] = scales[i] | ((scales[i+4] >> 4) << 6)
      s[i+4] = minima[i] | ((minima[i+4] >> 4) << 6)
      s[i+8] = (scales[i+4] & 15) | ((minima[i+4] & 15) << 4)
    codes = [(i*13+i//7+variant*3) % (32 if typ == 13 else 16) for i in range(256)]
    low, high = bytearray(128), bytearray(32)
    for i,q in enumerate(codes):
      group, lane = divmod(i,32)
      low[(group//2)*32+lane] |= (q & 15) << (4*(group%2))
      high[lane] |= (q >> 4) << group
    expected = [0.125*scales[i//32]*q - 0.25*minima[i//32] for i,q in enumerate(codes)]
    return struct.pack('<ee',0.125,0.25)+s+(high if typ == 13 else b'')+low, expected
  if typ == 14:
    scales = [-128,-63,-31,-7,-1,1,3,7,15,31,63,127,-13,19,-23,29]
    codes = [(i*17+i//5+variant*7) % 64 for i in range(256)]
    low, high = bytearray(128), bytearray(64)
    for i,q in enumerate(codes):
      half, within = divmod(i,128)
      group, lane = divmod(within,32)
      low[half*64+(group%2)*32+lane] |= (q & 15) << (4*(group//2))
      high[half*32+lane] |= (q >> 4) << (2*group)
    return bytes(low+high)+struct.pack('<16be',*scales,0.0625), [0.0625*scales[i//16]*(q-32) for i,q in enumerate(codes)]
  raise ValueError(typ)

class TestGGUFLoader:
  @pytest.mark.parametrize('typ', [0,1,30,2,8,12,13,14])
  def test_values_exact_payload_root_and_lifetime(self, path, typ):
    from tinygrad import dtypes
    from tinygrad.uop.ops import Ops
    if typ in (0,1,30):
      expected = [-3.5,0.0,0.125,1.0,16.0,-0.5]
      payload = b''.join(struct.pack('<f',v)[2:] if typ == 30 else struct.pack('<f' if typ == 0 else '<e',v) for v in expected)
      shape = (2,3)
    else:
      blocks = [quant_block(typ,i) for i in range(4)]
      payload, expected = b''.join(b for b,_ in blocks), sum((v for _,v in blocks),[])
      shape = (2,len(expected)//2)
    data = build_gguf([('before',(16,),0,bytes(64)), ('weight',shape,typ,payload), ('after',(32,),0,bytes(128))],alignment=64)
    idx = _index(path,data)
    path.chmod(0o444)
    before = (hashlib.sha256(data).digest(),path.stat().st_mtime_ns)
    info = idx.tensors[1]
    events: list[Any] = []
    source_refs, buffer_refs, reads, files = [], [], [], []
    original_to, original_realize, original_open = Tensor.to, Tensor.realize, pathlib.Path.open
    owner = Device['CPU:1']
    original_sync = owner.synchronize
    def transfer(t, device):
      assert t.device == 'PYTHON' and device == 'CPU:1' and t.dtype == dtypes.uint8
      assert t.shape == (len(payload),) and t.uop.buffer.nbytes == len(payload)
      source_refs.append(weakref.ref(t))
      buffer_refs.append(weakref.ref(t.uop.buffer))
      events.append('copy')
      return original_to(t,device)
    def realize(t, *args, **kwargs):
      events.append(('realize', t.dtype, t.shape))
      return original_realize(t,*args,**kwargs)
    def sync():
      gc.collect()
      assert all(ref() is not None for ref in source_refs+buffer_refs)
      assert len(source_refs) == 1
      events.append('sync')
      original_sync()
    class Reader:
      def __init__(self,f): self.f = f
      def __enter__(self): return self
      def __exit__(self,*args): self.f.close()
      def fileno(self): return self.f.fileno()
      def seek(self,*args): return self.f.seek(*args)
      def read(self,n):
        reads.append((self.f.tell(),n))
        return self.f.read(n)
    def opened(pp,mode):
      assert pp == path and mode == 'rb'
      f = original_open(pp,mode)
      files.append(f)
      return Reader(f)
    with patch.object(Tensor,'to',transfer), patch.object(Tensor,'realize',realize), patch.object(owner,'synchronize',sync), \
         patch.object(pathlib.Path,'open',opened):
      result = gguf.load_gguf_tensor(idx,info,device='CPU:1')
    assert reads == [(info.offset,len(payload))] and all(f.closed for f in files)
    assert events[:3] == [('realize',dtypes.uint8,(len(payload),)), 'copy', ('realize',dtypes.uint8,(len(payload),))]
    assert events[3:] and all(e == 'sync' for e in events[3:])  # CPU copy may also synchronize internally
    roots = [u.buffer for u in result.uop.toposort() if u.op is Ops.BUFFER]
    assert len(roots) == 1 and isinstance(roots[0],Buffer) and roots[0].device == 'CPU:1' and roots[0].nbytes == len(payload)
    gc.collect()
    assert all(ref() is None for ref in source_refs+buffer_refs)
    assert (hashlib.sha256(path.read_bytes()).digest(),path.stat().st_mtime_ns) == before
    path.unlink()  # result must not depend on an index/reader/file or the PYTHON source
    del idx, info
    assert result.device == 'CPU:1' and result.shape == shape
    assert result.cast(dtypes.float32).flatten().tolist() == expected

  @pytest.mark.parametrize('owner', ['PYTHON','PYTHON:1','CPU','CPU:2'])
  def test_owner_and_independent_replicas(self, path, owner):
    from tinygrad.uop.ops import Ops
    idx = _index(path,build_gguf([('w',(2,),0,struct.pack('<ff',1.25,-2.5))]))
    result = gguf.load_gguf_tensor(idx,idx.tensors[0],device=owner)
    assert result.device == owner and result.tolist() == [1.25,-2.5]
    other = gguf.load_gguf_tensor(idx,idx.tensors[0],device='CPU:3')
    def roots(t): return {u for u in t.uop.toposort() if u.op is Ops.BUFFER}
    assert roots(result).isdisjoint(roots(other)) and other.tolist() == result.tolist()

  @pytest.mark.parametrize('mutation', ['truncate','mtime','replace','same_size'])
  def test_changed_file_fails_before_source(self, path, mutation):
    data = build_gguf([('w',(2,),0,struct.pack('<ff',1,2))])
    idx = _index(path,data)
    if mutation == 'truncate': path.write_bytes(data[:-1])
    elif mutation == 'mtime': os.utime(path,ns=(path.stat().st_atime_ns,path.stat().st_mtime_ns+1))
    elif mutation == 'replace': _save(path.with_suffix('.new'),data).replace(path)
    else:
      path.write_bytes(data[:-1]+b'\xff')
      os.utime(path,ns=(path.stat().st_atime_ns,idx.parts[0].identity[-1]+1))
    with patch.object(gguf,'Tensor',side_effect=AssertionError('allocation before validation')), pytest.raises(ValueError):
      gguf.load_gguf_tensor(idx,idx.tensors[0],device='CPU:1')

  @pytest.mark.parametrize('mode', ['short','mutate','wrong_inode'])
  def test_descriptor_fstat_postread_and_close(self, path, mode):
    data = build_gguf([('w',(2,),0,struct.pack('<ff',1,2))])
    idx = _index(path,data)
    other = _save(path.with_suffix('.other'),data)
    original_open = pathlib.Path.open
    files = []
    class Reader:
      def __init__(self,f): self.f = f
      def __enter__(self): return self
      def __exit__(self,*args): self.f.close()
      def fileno(self): return self.f.fileno()
      def seek(self,*args): return self.f.seek(*args)
      def read(self,n):
        if mode == 'mutate': os.utime(path,ns=(path.stat().st_atime_ns,path.stat().st_mtime_ns+1))
        return self.f.read(n-1 if mode == 'short' else n)
    def opened(pp,access):
      assert access == 'rb'
      f = original_open(other if mode == 'wrong_inode' else pp,access)
      files.append(f)
      return Reader(f)
    with patch.object(pathlib.Path,'open',opened), patch.object(gguf,'Tensor',side_effect=AssertionError('allocation')), \
         pytest.raises(ValueError): gguf.load_gguf_tensor(idx,idx.tensors[0],device='CPU:1')
    assert files and all(f.closed for f in files)

  def test_forged_descriptor_rejected(self, path):
    idx = _index(path,build_gguf([('w',(2,),0,bytes(8))]))
    updates: list[dict[str,Any]] = [{'nbytes':4}, {'offset':0}, {'part':-1}, {'part':1}, {'shape':(3,)}, {'ggml_type':999}]
    for update in updates:
      with patch.object(gguf,'Tensor',side_effect=AssertionError('allocation')), pytest.raises(ValueError):
        gguf.load_gguf_tensor(idx,dataclasses.replace(idx.tensors[0],**update),device='CPU:1')

  @pytest.mark.parametrize('owner', [None, '', ('CPU','CPU:1'), 'DISK:forbidden', 'CPU:CLANG', 'BOGUS'])
  def test_invalid_owner_rejected_before_io(self, path, owner):
    idx = _index(path,build_gguf([('w',(2,),0,bytes(8))]))
    with patch.object(pathlib.Path,'open',side_effect=AssertionError('file I/O')), pytest.raises(ValueError):
      gguf.load_gguf_tensor(idx,idx.tensors[0],device=owner)

  @pytest.mark.parametrize('typ,shape,size', [(24,(2,),2),(3,(32,),20),(39,(32,),17),(41,(128,),18)])
  def test_known_but_unreviewed_types_fail_before_copy(self, path, typ, shape, size):
    idx = _index(path,build_gguf([('w',shape,typ,bytes(size))]))  # structural index knows legacy formats
    with patch.object(gguf,'Tensor',side_effect=AssertionError('allocation')), \
         patch.object(pathlib.Path,'open',side_effect=AssertionError('payload I/O')), pytest.raises(ValueError):
      gguf.load_gguf_tensor(idx,idx.tensors[0],device='CPU:1')

  @pytest.mark.parametrize('fail_sync', [False,True])
  def test_explicit_sync_keeps_actual_source_alive(self, path, fail_sync):
    from tinygrad import dtypes
    idx = _index(path,build_gguf([('w',(2,),0,struct.pack('<ff',1,2))]))
    # Return preallocated raw storage from the copy seam, so no CPU-internal copy sync can satisfy this test.
    raw = Tensor(struct.pack('<ff',1,2),device='CPU:1',dtype=dtypes.uint8).realize()
    refs, events = [], []
    def copy(source,device):
      assert source.device == 'PYTHON' and device == 'CPU:1'
      refs.extend([weakref.ref(source),weakref.ref(source.uop.buffer)])
      return raw
    def sync():
      gc.collect()
      assert all(ref() is not None for ref in refs)
      events.append('sync')
      if fail_sync: raise RuntimeError('injected synchronization failure')
    with patch.object(Tensor,'to',copy), patch.object(Device['CPU:1'],'synchronize',sync):
      if fail_sync:
        with patch.object(gguf,'ggml_data_to_tensor',side_effect=AssertionError('decode after failed sync')):
          with pytest.raises(RuntimeError,match='injected'): gguf.load_gguf_tensor(idx,idx.tensors[0],device='CPU:1')
      else: gguf.load_gguf_tensor(idx,idx.tensors[0],device='CPU:1')
    assert events == ['sync']
    gc.collect()
    assert all(ref() is None for ref in refs)

  def test_decode_follows_synchronized_raw(self, path):
    from tinygrad import dtypes
    from tinygrad.uop.ops import Ops
    idx = _index(path,build_gguf([('w',(32,),2,quant_block(2)[0])]))
    owner = Device['CPU:1']
    original_sync, original_decode = owner.synchronize, gguf.ggml_data_to_tensor
    events = []
    def sync():
      events.append('sync')
      original_sync()
    def decode(t,n,typ):
      assert events[-1] == 'sync' and t.device == 'CPU:1' and t.dtype == dtypes.uint8
      assert t.uop.op is Ops.BUFFER and t.uop.buffer.is_allocated()
      events.append('decode')
      return original_decode(t,n,typ)
    with patch.object(owner,'synchronize',sync), patch.object(gguf,'ggml_data_to_tensor',decode):
      gguf.load_gguf_tensor(idx,idx.tensors[0],device='CPU:1')
    assert events[-1] == 'decode'

  def test_later_mutation_does_not_invalidate_prior_result(self, tmp_path):
    a,b = TestGGUFIndex()._parts(tmp_path)
    idx = gguf.index_gguf(a)
    first = gguf.load_gguf_tensor(idx,idx.tensors[0],device='CPU:1')
    b.write_bytes(b'')
    with pytest.raises(ValueError): gguf.load_gguf_tensor(idx,idx.tensors[1],device='CPU:1')
    assert first.tolist() == [0.0]  # no claim of zero allocations after earlier successful payloads

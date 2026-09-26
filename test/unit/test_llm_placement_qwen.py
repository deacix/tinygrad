"""Placed qwen2 (QKV bias) and qwen3 (Q/K RMSNorm) dense models against the legacy single-device loader. CPU only."""
from unittest.mock import patch
import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
from tinygrad.llm import model as mm
from tinygrad.llm.gguf import index_gguf
from tinygrad.llm.model import Transformer
from tinygrad.llm.placement import LayerPlacement, estimate_placement
from test.unit.test_gguf_placement import build_gguf
from test.unit.test_llm_placement import no_allocations
from test.unit.test_llm_placement_execution import ref_step

# qwen3's 2*24 query width differs from dim=32, as in Qwen3-0.6B and Qwen3-4B.
HEAD_DIMS = {'qwen2':16, 'qwen3':24}
# Two full chunks, then single tokens: each stage JIT runs eagerly, captures and replays.
STEPS = ((0,(3,5,7,9)),(4,(11,13,15,17)),(8,(19,21,23,25)),(12,(2,)),(13,(4,)),(14,(6,)))


def qwen_fixture(arch, *, dim=32, blocks=4, tied=True):
  """Independent manifest/values in the qwen GGUF naming: NEOX RoPE rows, so no llama Q/K permutation."""
  rng = np.random.default_rng(4321)
  head_dim, heads, kv_heads = HEAD_DIMS[arch], 2, 1
  q, k = heads*head_dim, kv_heads*head_dim
  metadata = {'general.architecture':(8,arch), 'general.name':(8,f'offline {arch} fixture'),
    f'{arch}.block_count':(4,blocks), f'{arch}.embedding_length':(4,dim), f'{arch}.feed_forward_length':(4,dim*2),
    f'{arch}.attention.head_count':(4,heads), f'{arch}.attention.head_count_kv':(4,kv_heads),
    f'{arch}.attention.layer_norm_rms_epsilon':(6,1e-6), f'{arch}.rope.freq_base':(6,1000000.0),
    f'{arch}.context_length':(4,32), 'tokenizer.ggml.tokens':(9,(8,[str(i) for i in range(32)]))}
  if arch == 'qwen3': metadata |= {f'{arch}.attention.key_length':(4,head_dim), f'{arch}.attention.value_length':(4,head_dim)}
  shapes = {'token_embd.weight':(32,dim), 'output_norm.weight':(dim,)}
  if not tied: shapes['output.weight'] = (32,dim)
  for i in range(blocks):
    block = {'attn_norm.weight':(dim,), 'attn_q.weight':(q,dim), 'attn_k.weight':(k,dim), 'attn_v.weight':(k,dim),
             'attn_output.weight':(dim,q), 'ffn_norm.weight':(dim,), 'ffn_gate.weight':(2*dim,dim), 'ffn_up.weight':(2*dim,dim),
             'ffn_down.weight':(dim,2*dim)}
    if arch == 'qwen2': block |= {'attn_q.bias':(q,), 'attn_k.bias':(k,), 'attn_v.bias':(k,)}
    if arch == 'qwen3': block |= {'attn_q_norm.weight':(head_dim,), 'attn_k_norm.weight':(head_dim,)}
    shapes.update({f'blk.{i}.{name}':shape for name,shape in block.items()})
  tensors = []
  for name,shape in shapes.items():
    # Norm gains near one; matrices and biases small but large enough that dropping any of them moves the logits.
    v = 1 + rng.uniform(-0.05,0.05,shape) if name.endswith('norm.weight') else rng.uniform(-0.1,0.1,shape)
    tensors.append((name,shape,0,v.astype(np.float32).tobytes()))
  return tensors, metadata


def save_qwen(tmp_path, arch, *, tensors=None, metadata=None, **kwargs):
  ts, kv = qwen_fixture(arch, **kwargs)
  path = tmp_path / f'{arch}.gguf'
  path.write_bytes(build_gguf(ts if tensors is None else tensors, kv if metadata is None else metadata, alignment=64))
  return path


def load(path, half, placement=None):
  with patch.object(mm,'getenv',return_value=half):
    if placement is not None: return Transformer.from_gguf(path,32,False,placement=placement)[0]
    # The legacy loader uses the default device; the reference must be on CPU whatever DEV the suite runs under.
    with Context(DEV='CPU'): return Transformer.from_gguf(path,32,False)[0]


def refused(path, match):
  with no_allocations(), pytest.raises(ValueError,match=match):
    Transformer.from_gguf(path,realize=False,placement=LayerPlacement(('CPU','CPU:1'),(1,3)))


@pytest.mark.parametrize('arch', ['qwen2','qwen3'])
@pytest.mark.parametrize('half', [0,1])
@pytest.mark.parametrize('mode', ['host','native'])
@pytest.mark.parametrize('jit', [False,True])
def test_teacher_forced_logits_and_kv_match_legacy(tmp_path, arch, half, mode, jit):
  path = save_qwen(tmp_path, arch)
  legacy, placed = load(path, half), load(path, half, LayerPlacement(('CPU','CPU:1'),(1,3),mode,4))
  assert [b.config for b in placed.blk] == [placed._placed.config]*4
  atol, rtol = (3e-3,3e-3) if half else (1e-5,1e-4)
  for pos,ids in STEPS:
    expected = ref_step(legacy,list(ids),pos)
    tokens = Tensor([list(ids)],dtype=dtypes.int32,device='CPU').realize()
    with placed._placed_transaction('test') as owner: actual = placed._placed_step(tokens,pos,jit=jit,owner=owner).numpy()
    np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol,err_msg=f'step {pos} logits')
    assert np.max(np.abs(expected)) > 1e-3
    valid = pos+len(ids)
    for i,(a,b) in enumerate(zip(placed.blk,legacy.blk,strict=True)):
      np.testing.assert_allclose(a.cache_kv.numpy()[..., :valid, :],b.cache_kv.numpy()[..., :valid, :],rtol=rtol,atol=atol,
                                 err_msg=f'step {pos} block {i} KV')


@pytest.mark.parametrize('arch', ['qwen2','qwen3'])
def test_family_tensors_live_on_their_block_owner(tmp_path, arch):
  placed = load(save_qwen(tmp_path, arch), 0, LayerPlacement(('CPU','CPU:1'),(1,3),chunk_size=4))
  for i,block in enumerate(placed.blk):
    owner = 'CPU' if i == 0 else 'CPU:1'
    family = (block.attn_q.bias, block.attn_k.bias, block.attn_v.bias) if arch == 'qwen2' else \
      (block.attn_q_norm.weight, block.attn_k_norm.weight)
    assert all(t.device == owner for t in family)
    if arch == 'qwen2': assert not hasattr(block,'attn_q_norm')
    else: assert block.attn_q.bias is None and block.attn_output.bias is None


def test_generation_matches_legacy_greedy_ids(tmp_path):
  path = save_qwen(tmp_path, 'qwen3')
  legacy, placed = load(path, 0), load(path, 0, LayerPlacement(('CPU','CPU:1'),(2,2),chunk_size=4))
  prompt = [3,1,9,7,2]
  expected = [int(ref_step(legacy,prompt,0).argmax())]
  for n in range(3): expected.append(int(ref_step(legacy,[expected[-1]],len(prompt)+n).argmax()))
  gen = placed.generate(list(prompt))
  try: assert [next(gen) for _ in range(4)] == expected
  finally: gen.close()
  placed.check_available()


def test_accounting_counts_family_tensors(tmp_path):
  index = index_gguf(save_qwen(tmp_path, 'qwen3'))
  config, placement = mm.preflight_placement(index, LayerPlacement(('CPU','CPU:1'),(1,3),chunk_size=4))
  assert (config.head_dim, config.qk_norm, config.qkv_bias) == (24, 24, False)
  memory = estimate_placement(index, config, placement)
  # Independent F32 arithmetic: 48-wide Q/output, 24-wide K/V and two 24-wide Q/K norms per block.
  per_block = 4*(32 + 48*32 + 24*32 + 24*32 + 32*48 + 32 + 3*64*32 + 2*24)
  assert memory[0].weight_bytes == 4*32*32 + per_block
  assert memory[1].weight_bytes == 3*per_block + 4*32 + 4*32*32 and memory[1].tied_replica_bytes == 4*32*32
  assert (memory[0].kv_bytes, memory[1].kv_bytes) == (4*1*32*1*24, 4*3*32*1*24)


def test_qwen2_config_carries_qkv_bias(tmp_path):
  config, _ = mm.preflight_placement(index_gguf(save_qwen(tmp_path, 'qwen2')), LayerPlacement(('CPU',),(4,)))
  assert (config.head_dim, config.qk_norm, config.qkv_bias) == (16, 0, True)


@pytest.mark.parametrize('arch', ['qwen35','qwen35moe','qwen3moe','qwen2moe','qwen2vl'])
def test_other_qwen_architectures_stay_refused(tmp_path, arch):
  ts, kv = qwen_fixture('qwen3')
  kv = {k.replace('qwen3.', f'{arch}.'):v for k,v in kv.items()} | {'general.architecture':(8,arch)}
  refused(save_qwen(tmp_path,'qwen3',tensors=ts,metadata=kv), 'supports only dense')


@pytest.mark.parametrize('arch', ['qwen2','qwen3'])
@pytest.mark.parametrize('key,typ,value', [('expert_count',4,8), ('expert_used_count',4,2), ('rope.scaling.type',8,'yarn'),
  ('rope.scaling.factor',6,4.0), ('attention.sliding_window',4,16), ('nextn_predict_layers',4,1), ('tensor_data_layout',8,'other')])
def test_computation_metadata_outside_the_envelope_is_refused(tmp_path, arch, key, typ, value):
  ts, kv = qwen_fixture(arch)
  kv[f'{arch}.{key}'] = (typ,value)
  refused(save_qwen(tmp_path,arch,tensors=ts,metadata=kv), 'unsupported placement')


@pytest.mark.parametrize('arch,kind,match', [
  ('qwen2','no_bias','missing placed tensors'), ('qwen3','no_qk_norm','missing placed tensors'),
  ('qwen2','no_biases','tensor count'), ('qwen3','no_qk_norms','tensor count'),
  ('qwen2','qk_norm','unsupported placed tensor'), ('qwen3','bias','unsupported placed tensor'),
  ('qwen2','output_bias','unsupported placed tensor'), ('qwen3','norm_shape','shape mismatch'),
])
def test_each_family_requires_exactly_its_tensors(tmp_path, arch, kind, match):
  # Untied keeps one dropped tensor inside the count bound, so the manifest check itself names it.
  ts, kv = qwen_fixture(arch, tied=kind not in ('no_bias','no_qk_norm'))
  hd = HEAD_DIMS[arch]
  if kind == 'no_bias': ts = [t for t in ts if t[0] != 'blk.3.attn_v.bias']
  elif kind == 'no_qk_norm': ts = [t for t in ts if t[0] != 'blk.3.attn_k_norm.weight']
  elif kind == 'no_biases': ts = [t for t in ts if not t[0].endswith('.bias')]
  elif kind == 'no_qk_norms': ts = [t for t in ts if '_q_norm' not in t[0] and '_k_norm' not in t[0]]
  elif kind == 'qk_norm': ts.append(('blk.0.attn_q_norm.weight',(hd,),0,bytes(4*hd)))
  elif kind == 'bias': ts.append(('blk.0.attn_q.bias',(2*hd,),0,bytes(8*hd)))
  elif kind == 'output_bias': ts.append(('blk.0.attn_output.bias',(32,),0,bytes(128)))
  else: ts = [('blk.0.attn_q_norm.weight',(hd+2,),0,bytes(4*(hd+2))) if t[0] == 'blk.0.attn_q_norm.weight' else t for t in ts]
  refused(save_qwen(tmp_path,arch,tensors=ts,metadata=kv), match)


def test_qwen3_width_rule_does_not_relax_llama_or_qwen2(tmp_path):
  ts, kv = qwen_fixture('qwen2')
  kv['qwen2.attention.key_length'] = kv['qwen2.attention.value_length'] = (4,24)
  refused(save_qwen(tmp_path,'qwen2',tensors=ts,metadata=kv), 'dim=heads\\*head_dim')

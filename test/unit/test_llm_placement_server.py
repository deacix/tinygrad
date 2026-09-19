"""Offline placement CLI/HTTP acceptance: mandatory CPU, local synthetic GGUF only."""
import subprocess, sys, textwrap
from unittest.mock import Mock, patch
import pytest
from tinygrad.llm import cli
from tinygrad.llm.placement import LayerPlacement
from test.unit.test_llm_placement import save_llama


def placement_args(path):
  return ['--model',str(path),'--devices','CPU,CPU:1','--layer-counts','1,3','--chunk-size','4']


class TrackedIterator:
  # Retain after CLI returns: refcount/GC cleanup cannot satisfy the assertions.
  def __init__(self, values=(2,3,999), error=None):
    self.values, self.error, self.closed, self.advances = iter(values), error, False, 0
  def __iter__(self): return self
  def __next__(self):
    self.advances += 1
    if self.error: raise self.error
    return next(self.values)
  def close(self): self.closed = True


def tokenizer():
  tok = Mock(bos_id=1, eos_id=999, preset='llama3')
  tok.encode.side_effect = lambda text: [1,2]
  tok.decode.side_effect = lambda ids: ''.join(f'{i} ' for i in ids)
  tok.stream_decoder.side_effect = lambda: lambda tid=None: f'{tid} ' if tid is not None else ''
  tok.is_end.side_effect = lambda tid: tid == 999
  return tok


def run_cli(argv, model, *, tok=None):
  from tinygrad import Tensor
  with patch.object(sys,'argv',['tinygrad.llm',*argv]), \
       patch.object(cli.Transformer,'from_gguf',return_value=(model,{})) as load, \
       patch.object(cli.SimpleTokenizer,'from_gguf_kv',return_value=tok or tokenizer()), \
       patch.object(cli,'fetch',return_value='legacy.gguf'), \
       patch.object(cli.nn.state,'get_parameters',return_value=[Tensor.empty(1,device='CPU')]):
    cli.main()
  return load


class TestPlacementCLI:
  def test_defaults_and_legacy_options(self):
    args = cli.parse_args([])
    assert args.model == next(iter(cli.models))
    assert args.placement is None
    assert all(getattr(args,k) is None for k in ('devices','layer_counts','transfer','chunk_size','placement_check'))
    args = cli.parse_args(['--serve','--max-output-tokens','7','--warmup','--no_chat_template'])
    assert args.serve == 8000 and args.max_output_tokens == 7 and args.warmup and args.no_chat_template
    assert cli.parse_args(['--benchmark']).benchmark == 20

  @pytest.mark.parametrize('flags', [
    ['--devices','CPU'], ['--layer-counts','4'], ['--transfer','host'], ['--transfer','native'],
    ['--chunk-size','32'], ['--placement-check'], ['--devices','','--layer-counts',''],
  ])
  def test_placement_only_flags_require_both_lists(self,flags):
    with pytest.raises(SystemExit) as exc: cli.parse_args(flags)
    assert exc.value.code == 2

  @pytest.mark.parametrize('model', [None,'llama3.2:1b','https://example.com/no-fetch.gguf','missing.gguf'])
  def test_requires_existing_explicit_local_model(self,model):
    flags = ['--devices','CPU','--layer-counts','4'] + ([] if model is None else ['--model',model])
    with patch.object(cli,'fetch',side_effect=AssertionError('network')), pytest.raises(SystemExit) as exc: cli.parse_args(flags)
    assert exc.value.code == 2

  @pytest.mark.parametrize('extra', [
    ['--chunk-size','0'], ['--chunk-size','33'], ['--chunk-size','5','--max_context','4'],
    ['--layer-counts','0,4'], ['--layer-counts','-1,5'], ['--layer-counts','x,3'], ['--layer-counts','1,,3'],
    ['--layer-counts','4'], ['--devices','CPU,CPU:00'], ['--devices','CPU,AMD:1'], ['--devices','CPU,DISK:1'],
    ['--devices','CPU,CPU:-1'], ['--transfer','auto'],
    ['--vision-dir','missing'], ['--image','missing.png'], ['--prompt','hello'],
    ['--vision-dir','missing','--image','missing.png','--prompt','hello'],
  ])
  def test_invalid_combinations_before_file_or_device_access(self,extra):
    from pathlib import Path
    from tinygrad.device import Device
    with patch.object(Path,'is_file',side_effect=AssertionError('file access')), \
         patch.object(type(Device),'__getitem__',side_effect=AssertionError('device access')), pytest.raises(SystemExit) as exc:
      cli.parse_args(placement_args('missing.gguf')+extra)
    assert exc.value.code == 2

  def test_canonical_placement_and_defaults(self,tmp_path):
    path, _ = save_llama(tmp_path)
    args = cli.parse_args(['--model',str(path),'--devices','cpu:00,CPU:01','--layer-counts','1,3'])
    assert args.placement == LayerPlacement(('CPU','CPU:1'),(1,3))
    assert cli.parse_args(placement_args(path)+['--transfer','native']).placement.transfer == 'native'

  @pytest.mark.parametrize('tied', [False,True])
  def test_check_fresh_process_no_optional_dependencies_model_network_or_devices(self,tmp_path,tied):
    path, _ = save_llama(tmp_path,tied=tied)
    # Import from scratch with optional deps forbidden, then forbid construction, network and devices.
    script = textwrap.dedent('''
      import builtins, sys
      original_import = builtins.__import__
      def checked_import(name, *args, **kwargs):
        if name.split('.')[0] in ('numpy','jinja2','PIL'): raise AssertionError('optional dependency: '+name)
        return original_import(name, *args, **kwargs)
      builtins.__import__ = checked_import
      from unittest.mock import patch
      from tinygrad import Tensor
      from tinygrad.device import Device
      from tinygrad.llm import cli
      def forbidden(*args, **kwargs): raise AssertionError('forbidden side effect')
      with patch.object(type(Device),'__getitem__',forbidden), patch.object(Tensor,'__init__',forbidden), \\
           patch.object(cli,'fetch',forbidden), patch('socket.socket',forbidden), \\
           patch.object(cli.Transformer,'from_gguf',forbidden), patch.object(cli.Transformer,'warmup',forbidden), \\
           patch.object(cli.SimpleTokenizer,'from_gguf_kv',forbidden):
        cli.main()
    ''')
    result = subprocess.run([sys.executable,'-c',script,*placement_args(path),'--placement-check','--warmup'],
                            capture_output=True,text=True,timeout=30)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # Independent arithmetic: each block has 9216 float matrix values and 64 float norm values.
    assert 'CPU blocks=[0,1) raw_weight_bytes=41216 tied_replica_bytes=0 kv_bytes=2048 rope_bytes=2048' in out
    replica = 4096 if tied else 0
    assert f'CPU:1 blocks=[1,4) raw_weight_bytes=115584 tied_replica_bytes={replica} kv_bytes=6144 rope_bytes=2048' in out
    assert out.count('boundary_bytes=1024 largest_payload_bytes=8192') == 2
    assert 'logits_bytes=128 owner=CPU:1' in out
    for text in ('fit unknown','experimental','subset','scratch','allocator caches','staging','private'):
      assert text in out

  @pytest.mark.parametrize('extra', [['--layer-counts','2,3'],['--max_context','2']])
  def test_metadata_invalid_before_load(self,tmp_path,extra):
    path, _ = save_llama(tmp_path)
    with patch.object(sys,'argv',['llm',*placement_args(path),*extra,'--placement-check']), \
         patch.object(cli.Transformer,'from_gguf',side_effect=AssertionError('load')), pytest.raises((ValueError,SystemExit)):
      cli.main()

  def test_actual_cpu_load_reports_scalars_not_registered_state(self,tmp_path,capsys):
    path, _ = save_llama(tmp_path)
    original = cli.Transformer.from_gguf
    loaded = []
    def load(*args,**kwargs):
      model, kv = original(*args,**kwargs)
      loaded.append(model)
      return model, kv
    with patch.object(sys,'argv',['llm',*placement_args(path)]), patch.object(cli,'fetch',side_effect=AssertionError('fetch')), \
         patch.object(cli.Transformer,'from_gguf',side_effect=load), \
         patch.object(cli.SimpleTokenizer,'from_gguf_kv',return_value=tokenizer()), patch('builtins.input',side_effect=EOFError), \
         patch.object(cli.nn.state,'get_parameters',side_effect=AssertionError('generic state discovery')):
      cli.main()
    assert len(loaded) == 1 and loaded[0].placement == LayerPlacement(('CPU','CPU:1'),(1,3),chunk_size=4)
    assert 'raw_weight_bytes=41216' in capsys.readouterr().out
    loaded[0].check_available()

  @pytest.mark.parametrize('error', [None,KeyboardInterrupt(),RuntimeError('decode')])
  def test_benchmark_closes_on_limit_interrupt_and_error(self,tmp_path,error):
    path, _ = save_llama(tmp_path)
    gen = TrackedIterator(error=error)
    model = Mock(generate=Mock(return_value=gen))
    if error:
      with pytest.raises(type(error)): run_cli(placement_args(path)+['--benchmark','1'],model)
    else: run_cli(placement_args(path)+['--benchmark','1'],model)
    assert gen.closed and gen.advances == 1

  def test_benchmark_closes_on_eos(self,tmp_path):
    path, _ = save_llama(tmp_path)
    gen = TrackedIterator((999,))
    run_cli(placement_args(path)+['--benchmark','5'],Mock(generate=Mock(return_value=gen)))
    assert gen.closed and gen.advances == 1

  def test_two_interactive_eos_turns_close_before_next(self,tmp_path):
    path, _ = save_llama(tmp_path)
    gens = [TrackedIterator((2,999)),TrackedIterator((3,999))]
    calls = []
    def generate(ids):
      if calls: assert gens[0].closed
      calls.append(list(ids))
      return gens[len(calls)-1]
    with patch('builtins.input',side_effect=['one','two',EOFError]):
      run_cli(placement_args(path),Mock(generate=generate))
    assert len(calls) == 2 and all(g.closed for g in gens)

  @pytest.mark.parametrize('placed', [False,True])
  def test_text_server_cap_forwarding_and_legacy_load_signature(self,tmp_path,placed):
    path, _ = save_llama(tmp_path)
    model = Mock()
    argv = placement_args(path) if placed else ['--model','legacy.gguf']
    with patch.object(cli,'LLMServer') as server:
      load = run_cli(argv+['--serve','12345','--max-output-tokens','7','--benchmark','0'],model)
    model.warmup.assert_called_once()
    server.assert_called_once()
    assert server.call_args.kwargs['max_output_tokens'] == 7
    assert load.call_args.kwargs == ({'placement':LayerPlacement(('CPU','CPU:1'),(1,3),chunk_size=4)} if placed else {})

  @pytest.mark.parametrize('error', [KeyboardInterrupt(),RuntimeError('decode')])
  def test_interactive_closes_on_interrupt_and_error(self,tmp_path,error):
    path, _ = save_llama(tmp_path)
    gen = TrackedIterator(error=error)
    with patch('builtins.input',return_value='hello'), pytest.raises(type(error)):
      run_cli(placement_args(path),Mock(generate=Mock(return_value=gen)))
    assert gen.closed

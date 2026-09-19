"""Offline placement CLI/HTTP acceptance: mandatory CPU, local synthetic GGUF only."""
import http.client, json, socket, subprocess, sys, textwrap, threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
from tinygrad.llm import cli
from tinygrad.llm.placement import LayerPlacement
from test.unit.test_llm_placement import save_llama
from test.unit.test_llm_placement_execution import load_pair, ref_step
from tinygrad.llm import model as mm
from tinygrad.llm.model import PlacedInferenceError, PlacedModelUnavailableError, PlacedModelBusyError
from tinygrad.llm.serve import LLMServer, Handler


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


@contextmanager
def running_server(model, *, tok=None, **kwargs):
  server = LLMServer(('127.0.0.1',0),model,'offline',tok or tokenizer(),Mock(render=Mock(return_value='private prompt')),**kwargs)
  thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval':0.01})
  thread.start()
  try: yield server
  finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    assert not thread.is_alive()


def request(server, **options):
  conn = http.client.HTTPConnection(*server.server_address,timeout=30)
  try:
    body = {'model':'offline','messages':[{'role':'user','content':'private prompt'}]} | options
    conn.request('POST','/v1/chat/completions',body=json.dumps(body),headers={'Content-Type':'application/json'})
    response = conn.getresponse()
    return response.status, response.getheader('Content-Type'), response.read()
  finally: conn.close()


def events(payload):
  return [json.loads(line[6:]) for line in payload.splitlines() if line.startswith(b'data: ') and line != b'data: [DONE]']


class PlacedStub:
  # An explicit LayerPlacement is essential: Mock's dynamic attributes must not enable the placed path.
  placement = LayerPlacement(('CPU','CPU:1'),(1,3),chunk_size=4)
  max_context = 32
  def __init__(self, tokens=(2,3,999), fail_after=None):
    self.tokens, self.fail_after = tokens, fail_after
    self.state, self.closed, self.calls, self.checks = 'ready', [], [], 0
  def check_available(self):
    self.checks += 1
    if self.state == 'failed': raise PlacedModelUnavailableError('backend private data')
    if self.state != 'ready': raise PlacedModelBusyError('backend private data')
  def get_start_pos(self,ids): return 0
  def generate(self,ids,**kwargs):
    self.check_available()
    self.state = 'generating'
    self.calls.append((list(ids),kwargs))
    try:
      for i,tid in enumerate(self.tokens):
        if i == self.fail_after:
          self.state = 'failed'
          raise PlacedInferenceError(1,'CPU:1','stage') from RuntimeError('backend private data')
        yield tid
    finally:
      self.closed.append(True)
      if self.state != 'failed': self.state = 'ready'


class TestPlacementHTTP:
  def test_rejects_placed_vision_before_socket_open_and_preserves_dynamic_mocks(self):
    with patch.object(socket.socket,'bind',side_effect=AssertionError('socket opened')), pytest.raises(ValueError,match='placement.*vision'):
      LLMServer(('127.0.0.1',0),PlacedStub(),'offline',tokenizer(),Mock(),vision=object())
    with running_server(Mock(max_context=32),vision=object()): pass

  @pytest.mark.parametrize('stream', [False,True])
  @pytest.mark.parametrize('state,status', [('failed',503),('generating',409)])
  def test_availability_checked_before_headers(self,stream,state,status):
    model = PlacedStub()
    model.state = state
    with running_server(model) as server: code, content_type, body = request(server,stream=stream)
    assert code == status and content_type == 'application/json'
    assert json.loads(body)['error']['type'] == 'server_error'
    assert b'backend private data' not in body and b'[DONE]' not in body
    assert model.checks == 1 and model.calls == [] and model.closed == []

  @pytest.mark.parametrize('stream', [False,True])
  @pytest.mark.parametrize('fail_after', [0,1])
  def test_execution_failure_is_sanitized_nonstream_500_or_stream_abort(self,stream,fail_after):
    model = PlacedStub(fail_after=fail_after)
    with patch('tinygrad.llm.serve.stderr_log') as log, running_server(model) as server:
      status, content_type, body = request(server,stream=stream)
      retry, _, retry_body = request(server,stream=True)
    assert retry == 503 and b'[DONE]' not in retry_body
    assert model.closed == [True] and model.state == 'failed'
    if stream:
      assert status == 200 and content_type == 'text/event-stream'
      assert b'[DONE]' not in body and b'"server_error"' not in body
      assert all(c['choices'][0]['finish_reason'] is None for c in events(body))
      assert len(events(body)) == 1+fail_after
    else:
      assert status == 500 and json.loads(body)['error']['type'] == 'server_error'
    logs = ''.join(c.args[0] for c in log.call_args_list)
    assert 'stage=1' in logs and 'device=CPU:1' in logs and 'operation=stage' in logs and 'PlacedInferenceError' in logs
    assert 'backend private data' not in logs and 'private prompt' not in logs
    assert b'backend private data' not in body and b'private prompt' not in body

  @pytest.mark.parametrize('stream', [False,True])
  @pytest.mark.parametrize('limit,finish,count', [(None,'stop',2),(1,'length',1)])
  def test_eos_limits_usage_temperature_and_close(self,stream,limit,finish,count):
    model = PlacedStub()
    with running_server(model) as server:
      status, _, body = request(server,stream=stream,max_completion_tokens=None,max_tokens=limit,
                                temperature=0.7,stream_options={'include_usage':True})
    assert status == 200 and model.closed == [True] and model.state == 'ready'
    assert model.checks == 2 and model.calls == [([1,2],{'temperature':0.7})]
    if stream:
      chunks = events(body)
      assert b'[DONE]' in body and chunks[0]['choices'][0]['delta']['role'] == 'assistant'
      usage, choice = chunks[-1]['usage'], chunks[-2]['choices'][0]
    else:
      obj = json.loads(body)
      usage, choice = obj['usage'], obj['choices'][0]
    assert choice['finish_reason'] == finish
    assert usage == {'prompt_tokens':2,'completion_tokens':count,'total_tokens':2+count}

  @pytest.mark.parametrize('stream', [False,True])
  def test_iterators_without_close_and_operator_cap(self,stream):
    model = PlacedStub()
    model.generate = Mock(side_effect=lambda *a,**kw: iter((2,3,999)))
    with running_server(model,max_output_tokens=1) as server:
      status, _, body = request(server,stream=stream,max_tokens=9)
    assert status == 200
    if stream:
      assert b'[DONE]' in body
      assert events(body)[-1]['choices'][0]['finish_reason'] == 'length'
    else: assert json.loads(body)['usage']['completion_tokens'] == 1

  def test_run_model_closes_before_final_chunk_and_preserves_later_owner(self):
    model = PlacedStub()
    handler = SimpleNamespace(server=SimpleNamespace(model=model,tok=tokenizer()))
    first = Handler.run_model(handler,[1,2],'offline',max_tokens=1)
    next(first)  # role
    next(first)  # content
    assert model.state == 'generating'
    assert next(first)['choices'][0]['finish_reason'] == 'length'
    assert model.closed == [True] and model.state == 'ready'
    second = Handler.run_model(handler,[1,3],'offline',max_tokens=1)
    try:
      next(second)
      next(second)
      first.close()
      assert model.state == 'generating'
    finally:
      first.close()
      second.close()
    assert model.state == 'ready' and model.closed == [True,True]

  def test_failure_log_rejects_untrusted_metadata(self):
    from tinygrad.llm.serve import log_placed_failure
    with patch('tinygrad.llm.serve.stderr_log') as log:
      log_placed_failure(PlacedInferenceError('private prompt','CPU:1\nbackend private data','stage\nprivate prompt'))
    assert log.call_args.args[0] == ('placed inference failure: stage=unknown device=unknown operation=unknown '
                                    'exception=PlacedInferenceError; reload required\n')

  def test_two_requests_are_serialized_until_first_generator_closes(self):
    model = PlacedStub()
    entered, release, second_sent = threading.Event(), threading.Event(), threading.Event()
    order, results, errors = [], [], []
    def generate(ids, **kwargs):
      number = len(model.calls)+1
      model.check_available()
      model.state = 'generating'
      model.calls.append(number)
      order.append(('start',number))
      try:
        if number == 1:
          entered.set()
          assert release.wait(5)
        yield 2
      finally:
        model.state = 'ready'
        order.append(('close',number))
    model.generate = generate
    with running_server(model) as server:
      def client(number):
        conn = http.client.HTTPConnection(*server.server_address,timeout=10)
        try:
          body = {'model':'offline','messages':[{'role':'user','content':str(number)}],'max_tokens':1}
          conn.request('POST','/v1/chat/completions',body=json.dumps(body))
          if number == 2: second_sent.set()
          response = conn.getresponse()
          results.append((number,response.status,response.read()))
        except Exception as exc: errors.append(exc)
        finally: conn.close()
      first, second = (threading.Thread(target=client,args=(i,)) for i in (1,2))
      first.start()
      try:
        assert entered.wait(5)
        second.start()
        assert second_sent.wait(5)
        assert order == [('start',1)]
      finally:
        release.set()
        first.join(timeout=10)
        if second.ident is not None: second.join(timeout=10)
        assert not first.is_alive() and not second.is_alive()
    assert not errors and len(results) == 2 and all(status == 200 for _,status,_ in results)
    assert order == [('start',1),('close',1),('start',2),('close',2)]

  def test_loopback_disconnect_closes_generation_and_allows_next_request(self):
    import struct
    model = PlacedStub()
    resumed, closed = threading.Event(), threading.Event()
    original = model.generate
    def generate(ids, **kwargs):
      try:
        gen = original(ids,**kwargs)
        try:
          yield next(gen)
          assert resumed.wait(5)
          yield from gen
        finally: gen.close()
      finally: closed.set()
    model.generate = generate
    with running_server(model) as server:
      conn = socket.create_connection(server.server_address,timeout=5)
      try:
        body = json.dumps({'model':'offline','messages':[{'role':'user','content':'hello'}],'stream':True}).encode()
        conn.sendall(f'POST /v1/chat/completions HTTP/1.0\r\nContent-Length: {len(body)}\r\n\r\n'.encode()+body)
        received = b''
        while b'"content": "2 "' not in received:
          part = conn.recv(65536)
          assert part
          received += part
        conn.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0))
      finally:
        conn.close()
        resumed.set()
      assert closed.wait(5)
      assert model.state == 'ready' and model.closed == [True]
      status, _, _ = request(server,max_tokens=1)
      assert status == 200 and model.closed == [True,True]

  @pytest.mark.parametrize('stream', [False,True])
  def test_unrelated_errors_are_not_converted_to_placed_execution_errors(self,stream):
    model = PlacedStub()
    model.generate = Mock(side_effect=ValueError('programmer bug'))
    handler = object.__new__(Handler)
    handler.server = SimpleNamespace(model=model,vision=None,template=Mock(render=Mock(return_value='hello')),
                                     tok=tokenizer(),max_output_tokens=None)
    handler.path = '/v1/chat/completions'
    handler.read_chat_request = lambda: {'model':'offline','messages':[],'stream':stream}
    handler.send_data = Mock()
    handler.send_response, handler.send_header, handler.end_headers, handler.wfile = Mock(), Mock(), Mock(), Mock()
    with pytest.raises(ValueError,match='programmer bug'): handler.do_POST()
    handler.send_data.assert_not_called()
    assert not any(b'[DONE]' in c.args[0] for c in handler.wfile.write.call_args_list)


class TestPlacementHTTPRealCPU:
  def test_actual_placement_vision_rejected_before_binding_socket(self,tmp_path):
    model, _ = load_pair(tmp_path)
    with patch.object(socket.socket,'bind',side_effect=AssertionError('socket opened')), pytest.raises(ValueError,match='placement.*vision'):
      LLMServer(('127.0.0.1',0),model,'offline',tokenizer(),Mock(),vision=object())
    model.check_available()

  def test_busy_real_generator_returns_409_without_releasing_owner(self,tmp_path):
    model, _ = load_pair(tmp_path)
    gen = model.generate([1,2])
    try:
      next(gen)
      with running_server(model) as server:
        status, content_type, body = request(server,stream=True)
      assert status == 409 and content_type == 'application/json' and b'[DONE]' not in body
      assert model._placed.state == 'generating' and model._placed.lock.locked()
      next(gen)
    finally: gen.close()
    model.check_available()

  @pytest.mark.parametrize('mode', ['host','native'])
  def test_stream_nonstream_and_independent_cpu_reference_parity(self,tmp_path,mode):
    model, ref = load_pair(tmp_path,mode=mode)
    # Two greedy IDs computed independently, including the teacher-forced second step.
    first = int(ref_step(ref,[1,2],0).argmax())
    second = int(ref_step(ref,[first],2).argmax())
    with running_server(model) as server:
      status, _, body = request(server,max_tokens=2)
      assert status == 200
      result = json.loads(body)
      model.check_available()
      status, _, body = request(server,stream=True,max_tokens=2,stream_options={'include_usage':True})
      assert status == 200 and body.endswith(b'data: [DONE]\n\n')
      chunks = events(body)
      model.check_available()
    assert result['choices'][0]['message']['content'] == f'{first} {second} '
    assert ''.join(c['choices'][0]['delta'].get('content','') for c in chunks if c['choices']) == f'{first} {second} '
    assert result['usage'] == chunks[-1]['usage'] == {'prompt_tokens':2,'completion_tokens':2,'total_tokens':4}
    assert result['choices'][0]['finish_reason'] == chunks[-2]['choices'][0]['finish_reason'] == 'length'
    assert model._cached_tokens == [1,2,first]

  @pytest.mark.parametrize('stream,fail_at', [(False,1),(True,2)])
  @pytest.mark.parametrize('backend_error', [RuntimeError,BrokenPipeError,ConnectionResetError])
  def test_fault_after_real_kv_mutation_fails_closed(self,tmp_path,stream,fail_at,backend_error):
    model, _ = load_pair(tmp_path)
    transfer = mm._transfer_activation
    calls = []
    def fail(x,destination,mode):
      calls.append(True)
      if len(calls) == fail_at:
        # Stage zero has already synchronously written actual owner-local KV.
        valid = 2 if fail_at == 1 else 3
        assert (model.blk[0].cache_kv.numpy()[:,:,:,:valid,:] != 0).any()
        raise backend_error('backend private data with private prompt')
      return transfer(x,destination,mode)
    with patch.object(mm,'_transfer_activation',fail), patch('tinygrad.llm.serve.stderr_log') as log, running_server(model) as server:
      status, _, body = request(server,stream=stream,max_tokens=3)
      retry, content_type, retry_body = request(server,stream=True)
    assert retry == 503 and content_type == 'application/json' and b'[DONE]' not in retry_body
    assert model._cached_tokens == [] and model._placed.state == 'failed' and not model._placed.lock.locked()
    if stream:
      assert status == 200 and len(events(body)) == 2
      assert b'[DONE]' not in body and b'"server_error"' not in body
      assert all(c['choices'][0]['finish_reason'] is None for c in events(body))
    else: assert status == 500 and json.loads(body)['error']['type'] == 'server_error'
    logs = ''.join(c.args[0] for c in log.call_args_list)
    assert 'stage=1 device=CPU:1 operation=transfer exception=PlacedInferenceError' in logs
    assert 'backend private data' not in logs and 'private prompt' not in logs and 'Traceback' not in logs
    assert b'backend private data' not in body and b'private prompt' not in body

  @pytest.mark.parametrize('error', [BrokenPipeError,ConnectionResetError])
  def test_socket_write_disconnect_does_not_poison_real_model(self,tmp_path,error):
    model, _ = load_pair(tmp_path)
    handler = Mock(server=SimpleNamespace(model=model,tok=tokenizer()))
    chunks = Handler.run_model(handler,[1,2],'offline',max_tokens=3)
    writes = []
    def write(data):
      writes.append(data)
      if len(writes) == 3: raise error('client disconnected')
    handler.wfile.write.side_effect = write
    Handler.stream_json(handler,chunks)
    model.check_available()
    assert not model._placed.lock.locked() and len(model._cached_tokens) == 3
    assert not any(b'[DONE]' in data for data in writes)
    # A clean token-boundary close preserves a reusable prefix for the next real request.
    with running_server(model) as server:
      status, _, _ = request(server,max_tokens=1)
      assert status == 200
    model.check_available()

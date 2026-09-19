import json, socket, threading, unittest
from unittest.mock import Mock, patch
from tinygrad import Tensor, dtypes
from tinygrad.llm.serve import LLMServer, Handler
from tinygrad.llm.multimodal import ImageLimits, PreparedPrompt, IMAGE_TOKEN_ID

class TestQwenHTTP(unittest.TestCase):
  def setUp(self):
    self.model = Mock(max_context=32)
    self.model.get_start_pos.return_value = 0
    self.closed = []
    def generate(ids, **kwargs):
      try: yield from (300,301,999)
      finally: self.closed.append(True)
    self.model.generate.side_effect = generate
    self.tok = Mock()
    self.tok.encode.return_value = [1,2]
    self.tok.is_end.side_effect = lambda t: t == 999
    self.tok.stream_decoder.return_value = lambda tid=None: "x" if tid is not None else ""
    self.template = Mock()
    self.template.render.return_value = "prompt"
    self.server = LLMServer(("127.0.0.1",0), self.model, "test", self.tok, self.template)
    self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
    self.thread.start()

  def tearDown(self):
    self.server.shutdown()
    self.server.server_close()
    self.thread.join()

  def request(self, body, headers=None):
    raw = json.dumps(body).encode()
    headers = headers if headers is not None else [f"Content-Length: {len(raw)}"]
    request = ("POST /v1/chat/completions HTTP/1.0\r\nHost: localhost\r\n"+"\r\n".join(headers)+"\r\n\r\n").encode()+raw
    with socket.create_connection(self.server.server_address, timeout=3) as conn:
      conn.sendall(request)
      conn.shutdown(socket.SHUT_WR)
      chunks = []
      while chunk := conn.recv(65536): chunks.append(chunk)
    head, payload = b"".join(chunks).split(b"\r\n\r\n",1)
    return int(head.split()[1]), payload

  def test_wire_invalid_framing_and_options(self):
    body = {"model":"test", "messages":[{"role":"user", "content":"hello"}]}
    for headers in ([], ["Content-Length: -1"], ["Content-Length: 2", "Content-Length: 2"],
                    ["Content-Length: 100000000"], ["Transfer-Encoding: chunked"], ["Content-Length: 2000"]):
      with self.subTest(headers=headers): self.assertIn(self.request(body, headers)[0], (400,413))
    for options in ({"temperature":True}, {"temperature":float("nan")}, {"max_tokens":0}, {"stream":"yes"},
                    {"messages":None}, {"chat_template_kwargs":[]}, {"messages":[{"role":"user", "content":[5]}]}):
      with self.subTest(options=options): self.assertEqual(self.request(body|options)[0],400)
    self.model.generate.assert_not_called()

  def test_wire_image_without_bundle_is_rejected(self):
    status, body = self.request({"model":"test", "messages":[{"role":"user", "content":[
      {"type":"image_url", "image_url":{"url":"https://example.com/no-fetch.png"}}]}]})
    self.assertEqual(status,400)
    self.assertIn("no vision", json.loads(body)["error"]["message"])
    self.model.generate.assert_not_called()

  def test_wire_completion_alias_and_closure(self):
    status, body = self.request({"model":"test", "messages":[{"role":"user", "content":"hi"}],
                                 "max_tokens":2, "max_completion_tokens":1})
    self.assertEqual(status,200)
    self.assertEqual(json.loads(body)["usage"]["completion_tokens"],1)
    self.assertEqual(json.loads(body)["choices"][0]["finish_reason"],"length")
    self.assertEqual(self.closed,[True])

  def test_inner_generator_closed_before_next_request(self):
    chunks = Handler.run_model(Mock(server=self.server), [1,2], "test", max_tokens=1)
    next(chunks)
    next(chunks)
    chunks.close()
    self.assertEqual(self.closed,[True])
    list(Handler.run_model(Mock(server=self.server), [1,2], "test"))
    self.assertEqual(self.closed,[True,True])

  def test_body_absolute_deadline(self):
    from email.message import Message
    from tinygrad.llm.multimodal import ImageInputError
    handler = Mock(server=self.server)
    handler.headers = Message()
    handler.headers["Content-Length"] = "3"
    handler.connection.gettimeout.return_value = None
    handler.rfile.read1.side_effect = [b"a", b"b", b"c"]
    with patch("tinygrad.llm.serve.time.monotonic", side_effect=[0,1,9,11]):
      with self.assertRaisesRegex(ImageInputError, "deadline"): Handler.read_chat_request(handler)
    self.assertEqual(handler.rfile.read1.call_count, 2)

  def test_wire_generation_error_envelope(self):
    def broken(ids, **kwargs):
      try:
        yield 300
        raise RuntimeError("backend failure")
      finally: self.closed.append(True)
    self.model.generate.side_effect = broken
    request = {"model":"test", "messages":[{"role":"user", "content":"hi"}]}
    status, body = self.request(request)
    self.assertEqual(status,500)
    self.assertEqual(json.loads(body)["error"]["type"],"server_error")
    status, body = self.request(request|{"stream":True})
    self.assertEqual(status,200)
    self.assertIn(b'"server_error"',body)
    self.assertIn(b"[DONE]",body)
    self.assertEqual(self.closed,[True,True])

  def test_wire_prepared_image_usage_and_options(self):
    self.server.vision = Mock()
    self.server.image_limits = ImageLimits()
    self.server.max_output_tokens = 1
    self.model.token_embd.weight = Tensor.empty(2,2)
    prompt = PreparedPrompt((1,IMAGE_TOKEN_ID,2), Tensor.zeros(4,1536), ((1,2,2),), ((1,1),),
                            Tensor([[[0,1,2]]]*3, dtype=dtypes.int32), starts_reasoning=True)
    with patch("tinygrad.llm.multimodal.prepare_prompt", return_value=prompt) as prepare, \
         patch("tinygrad.llm.multimodal.embed_prompt", return_value=Tensor.zeros(1,3,2)) as embed:
      status, body = self.request({"model":"test", "messages":[{"role":"user", "content":"hi"}],
                                   "tools":[], "chat_template_kwargs":{"enable_thinking":True}, "max_tokens":100})
      self.assertEqual(status,200)
      response = json.loads(body)
      self.assertEqual(response["usage"], {"prompt_tokens":3, "completion_tokens":1, "total_tokens":4})
      self.assertEqual(response["choices"][0]["message"]["reasoning_content"], "x")
      self.assertEqual(prepare.call_args.kwargs["template_kwargs"], {"enable_thinking":True})
      embed.assert_called_once()
      self.assertIn("position_ids", self.model.generate.call_args.kwargs)
      self.assertEqual(self.closed,[True])

if __name__ == "__main__": unittest.main()

from __future__ import annotations
import copy, json, math, pathlib, re, socket, time, typing, uuid
from typing import TYPE_CHECKING
from tinygrad.helpers import DEBUG, colored, stderr_log
from tinygrad.viz.serve import TCPServerWithReuse, Handler as VizHandler
if TYPE_CHECKING:
  from tinygrad.llm.cli import SimpleTokenizer
  from tinygrad.llm.model import Transformer

def parse_tool_call(s:str) -> tuple[str, typing.Any]|None:
  s = s.strip()
  if s.startswith("{"):  # hermes JSON format: {"name": ..., "arguments": {...}}
    try:
      call = json.loads(s)
      return call["name"], call.get("arguments", call.get("parameters", {}))
    except (json.JSONDecodeError, KeyError): return None
  # XML format: <function=name>\n<parameter=key>\nvalue\n</parameter>...</function>
  if (fm := re.match(r"<function=([^>]+)>\s*(.*?)\s*(?:</function>)?$", s, re.DOTALL)):
    args = {}
    for pm in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", fm.group(2), re.DOTALL):
      value = re.sub(r"^\r?\n|\r?\n\Z", "", pm.group(2))
      try: args[pm.group(1)] = json.loads(value)
      except json.JSONDecodeError: args[pm.group(1)] = value
    return fm.group(1), args
  return None

def normalize_messages(messages:list[dict]) -> None:
  # chat templates expect tool_call arguments as dicts (OpenAI clients send JSON strings)
  for m in messages:
    for tc in m.get("tool_calls") or []:
      if "function" in tc and isinstance(args := tc["function"].get("arguments"), str):
        try: tc["function"]["arguments"] = json.loads(args)
        except json.JSONDecodeError: pass

class StreamRouter:
  # routes streamed output text to (field, text) deltas, keeping tool_call regions in .buf for the final parse
  def __init__(self, reasoning:bool=False):
    self.buf = ""
    self.mode = "reasoning" if reasoning else "undecided"  # output inside a think block is sent as reasoning_content
  def split(self, tag:str, final:bool) -> tuple[str, bool]:
    # split buf on the first full tag, holding back a partial tag at the end unless final
    if tag in self.buf:
      before, self.buf = self.buf.split(tag, 1)
      return before, True
    hold = max((i for i in range(1, min(len(self.buf), len(tag))+1) if tag.startswith(self.buf[-i:])), default=0) if not final else 0
    emit, self.buf = self.buf[:len(self.buf)-hold], self.buf[len(self.buf)-hold:]
    return emit, False
  def route(self, piece:str, final:bool=False) -> typing.Iterator[tuple[str, str]]:
    self.buf += piece
    if self.mode == "undecided":  # decide whether the output starts with a think block
      if not final and len(self.buf) < len("<think>") and "<think>".startswith(self.buf): return
      self.mode, self.buf = ("reasoning", self.buf[len("<think>"):]) if self.buf.startswith("<think>") else ("content", self.buf)
    if self.mode == "reasoning":
      emit, done = self.split("</think>", final)
      if emit: yield "reasoning_content", emit
      if not done: return
      self.mode = "content"
    if self.mode == "tool": return
    emit, found = self.split("<tool_call>", final)
    if emit: yield "content", emit
    if found: self.mode, self.buf = "tool", "<tool_call>" + self.buf

class Handler(VizHandler):
  server: LLMServer
  def log_request(self, code='-', size='-'): pass
  def do_GET(self):
    if self.path == "/v1/models": self.send_data(json.dumps({"object":"list","data":[{"id":self.server.model_name,"object":"model"}]}).encode())
    elif self.path.startswith("/assets/"): super().do_GET()
    else: self.send_data((pathlib.Path(__file__).parent / "chat.html").read_bytes(), content_type="text/html")
  def run_model(self, ids, model_name:str, include_usage=False, max_tokens:int|None=None, temperature:float=0.0,
                reasoning:bool=False, embeddings=None):
    model, tok = self.server.model, self.server.tok
    generation_kwargs:dict[str,typing.Any] = {}
    if not isinstance(ids, list):
      from tinygrad.llm.multimodal import PreparedPrompt, embed_prompt
      if not isinstance(ids, PreparedPrompt): raise ValueError("invalid prepared prompt")
      prompt = ids
      reasoning = prompt.starts_reasoning
      if prompt.pixel_values is not None:
        if embeddings is None: embeddings = embed_prompt(model, self.server.vision, prompt)
        generation_kwargs = {"inputs_embeds":embeddings, "position_ids":prompt.position_ids, "rope_delta":prompt.rope_delta}
      ids = list(prompt.tokens)
    prompt_tokens = len(ids)
    remaining = model.max_context-prompt_tokens
    if generation_kwargs: max_tokens = min(max_tokens or self.server.max_output_tokens or 256, remaining)
    cache_start_pos = 0 if generation_kwargs else model.get_start_pos(ids)
    stderr_log(f"in:{colored(f'{cache_start_pos:5d}', 'green')} +{len(ids)-cache_start_pos:5d}  {colored('--', 'BLACK')}  ")
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    def chunk(d:dict): return {"choices": [{"index":0, "delta":d, "finish_reason":None}], **tmpl}
    out: list[int] = []
    finish_reason = "stop"
    st = pt = time.perf_counter()
    dec = tok.stream_decoder()
    router = StreamRouter(reasoning)
    def log_stats(interrupted:bool=False):
      et = time.perf_counter()
      total = f"total:{et-st:6.2f}s"
      stderr_log(f"gen:{len(out)/(et-pt) if len(out) > 1 else 0:4.0f} tok/s  {colored('--', 'BLACK')}  "
                 f"out:{len(out):5d}  {colored('--', 'BLACK')}  {colored(total, 'red') if interrupted else total}\n")
    completed = False
    gen = model.generate(ids, temperature=temperature, **generation_kwargs)
    try:
      yield chunk({"role":"assistant", "content":""})
      for next_id in gen:
        if len(out) == 0:
          stderr_log(f"prefill:{(prompt_tokens-cache_start_pos)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
        if tok.is_end(next_id): break
        out.append(next_id)
        for field, delta in router.route(dec(next_id)): yield chunk({field:delta})
        if max_tokens is not None and len(out) >= max_tokens:
          finish_reason = "length"
          break
      # Close before yielding the final chunk so a retained iterator cannot reset a later request.
      if close := getattr(gen, "close", None): close()
      if generation_kwargs and len(out) >= remaining: finish_reason = "length"
      for field, delta in router.route(dec(), final=True): yield chunk({field:delta})
      tool_calls: list[dict] = []
      for m in re.finditer(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", router.buf, re.DOTALL):
        if (parsed := parse_tool_call(m.group(1))) is None:
          stderr_log(f"failed to parse tool call: {m.group(1)[:200]}")
          yield chunk({"content":m.group(0)})  # don't silently drop output the client can't use
        else:
          name, args = parsed
          tool_calls.append({"index":len(tool_calls), "id":f"call_{uuid.uuid4().hex[:24]}", "type":"function",
                             "function":{"name":name, "arguments":args if isinstance(args, str) else json.dumps(args)}})
      if tool_calls:
        yield chunk({"tool_calls":tool_calls})
        if finish_reason == "stop": finish_reason = "tool_calls"
      completed = True
      yield {"choices": [{"index":0, "delta":{},"finish_reason":finish_reason}], **tmpl}
      if include_usage:
        yield {"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(out),
                                        "total_tokens": prompt_tokens + len(out)}, **tmpl}
      log_stats()
    except GeneratorExit:
      if not completed: log_stats(interrupted=True)
      raise
    finally:
      if close := getattr(gen, "close", None): close()

  def read_chat_request(self) -> dict:
    from tinygrad.llm.multimodal import ImageInputError
    limits = self.server.image_limits
    cap = limits.max_request_bytes if limits is not None else 16*1024*1024
    lengths = self.headers.get_all("Content-Length", [])
    if self.headers.get("Transfer-Encoding") or len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0]):
      raise ImageInputError("a single Content-Length without Transfer-Encoding is required")
    length = int(lengths[0])
    if length > cap: raise ImageInputError("request byte budget exceeded", 413)
    previous = self.connection.gettimeout()
    try:
      deadline, remaining, chunks = time.monotonic()+10, length, []
      while remaining:
        timeout = deadline-time.monotonic()
        if timeout <= 0: raise ImageInputError("request body deadline exceeded")
        self.connection.settimeout(timeout)
        chunk = self.rfile.read1(min(65536, remaining))
        if not chunk: raise ImageInputError("truncated request body")
        chunks.append(chunk)
        remaining -= len(chunk)
      body = json.loads(b"".join(chunks))
    except ImageInputError: raise
    except (ValueError, UnicodeError, RecursionError, socket.timeout): raise ImageInputError("invalid or incomplete JSON request") from None
    finally: self.connection.settimeout(previous)
    stack = [(body, 0)]
    while stack:
      value, depth = stack.pop()
      if depth > 32: raise ImageInputError("request JSON nesting exceeds 32 levels")
      if isinstance(value, dict): stack.extend((item,depth+1) for item in value.values())
      elif isinstance(value, list): stack.extend((item,depth+1) for item in value)
    if not isinstance(body, dict) or not isinstance(body.get("model"), str): raise ImageInputError("request model must be a string")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages: raise ImageInputError("messages must be a nonempty array")
    for message in messages:
      if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant", "tool"):
        raise ImageInputError("invalid message role")
      content = message.get("content")
      if not isinstance(content, (str,list,type(None))): raise ImageInputError("invalid message content")
      if isinstance(content, list):
        for part in content:
          if not isinstance(part, dict) or part.get("type") not in ("text", "image_url"): raise ImageInputError("unsupported content part")
          if part["type"] == "text" and not isinstance(part.get("text"), str): raise ImageInputError("invalid text content")
          if part["type"] == "image_url" and self.server.vision is None: raise ImageInputError("this server has no vision bundle")
      calls = message.get("tool_calls")
      if calls is not None and (not isinstance(calls, list) or any(not isinstance(tc, dict) or not isinstance(tc.get("function"), dict)
                                                                for tc in calls)): raise ImageInputError("invalid tool calls")
    for key in ("max_completion_tokens", "max_tokens"):
      if key in body and body[key] is not None and (type(body[key]) is not int or body[key] <= 0): raise ImageInputError(f"{key} must be positive")
    temperature = body.get("temperature", 0.0)
    try: valid_temperature = type(temperature) in (int,float) and math.isfinite(temperature) and temperature >= 0
    except OverflowError: valid_temperature = False
    if not valid_temperature: raise ImageInputError("invalid temperature")
    if body.get("stream") is None: body["stream"] = False
    if type(body["stream"]) is not bool: raise ImageInputError("stream must be boolean")
    if body.get("stream_options") is None: body["stream_options"] = {}
    if not isinstance(body["stream_options"], dict): raise ImageInputError("invalid stream options")
    if not isinstance(body.get("chat_template_kwargs", {}), dict): raise ImageInputError("invalid chat template options")
    if body.get("tools") is not None and not isinstance(body["tools"], list): raise ImageInputError("tools must be an array")
    return body

  def do_POST(self):
    request_st = time.perf_counter()
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  ")
    from tinygrad.llm.multimodal import ImageInputError, prepare_prompt
    if self.path != "/v1/chat/completions":
      return self.send_data(json.dumps({"error":{"message":"unknown endpoint", "type":"invalid_request_error"}}).encode(), status_code=404)
    try:
      body = self.read_chat_request()
      prepared = None
      if self.server.vision is not None:
        options = dict(body.get("chat_template_kwargs", {}))
        if "reasoning_effort" in body: options["reasoning_effort"] = body["reasoning_effort"]
        device = self.server.model.token_embd.weight.device
        if not isinstance(device, str): raise ImageInputError("vision requires a single decoder device")
        prepared = prepare_prompt(body["messages"], self.server.tok, self.server.template, limits=self.server.image_limits,
                                  device=device, tools=body.get("tools"), template_kwargs=options, max_context=self.server.model.max_context)
        ids = list(prepared.tokens)
        starts_reasoning = prepared.starts_reasoning
      else:
        messages = copy.deepcopy(body["messages"])
        normalize_messages(messages)
        rendered = self.server.template.render(messages=messages, tools=body.get("tools"), add_generation_prompt=True, preserve_thinking=True)
        ids = self.server.tok.encode(rendered)
        starts_reasoning = rendered.rstrip().endswith("<think>")
    except (ImageInputError, ValueError, TypeError, KeyError, RecursionError) as exc:
      status = exc.status if isinstance(exc, ImageInputError) else 400
      return self.send_data(json.dumps({"error":{"message":str(exc), "type":"invalid_request_error", "param":"messages"}}).encode(),
                            status_code=status)
    if DEBUG >= 1: print(f"prepared {len(ids)} tokens")
    if self.path == "/v1/chat/completions":
      stderr_log(f"prep:{(time.perf_counter()-request_st)*1e3:5.0f} ms  {colored('--', 'BLACK')}  ")
      if len(ids) >= self.server.model.max_context:
        stderr_log(f"{colored('context length exceeded', 'red')}  in:{len(ids):5d}  max:{self.server.model.max_context:5d}\n")
        return self.send_data(json.dumps({"error":{"message":f"prompt has {len(ids)} tokens, but the model context is "
          f"{self.server.model.max_context}", "type":"invalid_request_error", "param":"messages", "code":"context_length_exceeded"}}).encode(),
          status_code=400)

      embeddings = None
      if prepared is not None and prepared.pixel_values is not None:
        from tinygrad.llm.multimodal import embed_prompt
        try: embeddings = embed_prompt(self.server.model, self.server.vision, prepared)
        except Exception:
          return self.send_data(json.dumps({"error":{"message":"vision inference failed", "type":"server_error"}}).encode(), status_code=500)
      # reply
      max_tokens = body.get("max_completion_tokens")
      if max_tokens is None: max_tokens = body.get("max_tokens")
      if self.server.max_output_tokens is not None:
        max_tokens = min(max_tokens or self.server.max_output_tokens, self.server.max_output_tokens)
      chunks = self.run_model(prepared if prepared is not None else ids, body["model"],
                              not body.get("stream") or body.get("stream_options",{}).get("include_usage", False),
                              max_tokens=max_tokens, temperature=float(body.get("temperature", 0.0)),
                              reasoning=starts_reasoning, embeddings=embeddings)
      if body.get("stream"):
        def guarded_stream():
          try: yield from chunks
          except Exception: yield {"error":{"message":"generation failed", "type":"server_error"}}
          finally: chunks.close()
        self.stream_json(guarded_stream())
      else:
        try: results = list(chunks)
        except Exception:
          return self.send_data(json.dumps({"error":{"message":"generation failed", "type":"server_error"}}).encode(), status_code=500)
        finally: chunks.close()
        out, reasoning, tool_calls, finish_reason = [], [], [], "stop"
        for c in results:
          if not c["choices"]: continue
          choice = c["choices"][0]
          if (delta := choice.get("delta", {})):
            if delta.get("content"): out.append(delta["content"])
            if delta.get("reasoning_content"): reasoning.append(delta["reasoning_content"])
            tool_calls += [{k:v for k, v in tc.items() if k != "index"} for tc in delta.get("tool_calls", [])]
          if choice.get("finish_reason"): finish_reason = choice["finish_reason"]
        message: dict[str, typing.Any] = {"role":"assistant", "content":"".join(out) or None}
        if reasoning: message["reasoning_content"] = "".join(reasoning)
        if tool_calls: message["tool_calls"] = tool_calls
        self.send_data(json.dumps({**c, "object":"chat.completion",
          "choices":[{"index":0, "message":message, "finish_reason":finish_reason}]}).encode())
    else:
      raise RuntimeError(f"unhandled path {self.path}")

class LLMServer(TCPServerWithReuse):
  def __init__(self, server_address:tuple, model:Transformer, model_name:str, tok:SimpleTokenizer, template:typing.Any,
               *, vision=None, image_limits=None, max_output_tokens:int|None=None):
    if max_output_tokens is not None and (type(max_output_tokens) is not int or max_output_tokens <= 0):
      raise ValueError("max_output_tokens must be positive")
    self.model, self.model_name, self.tok, self.template = model, model_name, tok, template
    self.vision, self.image_limits = vision, image_limits
    self.max_output_tokens = 256 if vision is not None and max_output_tokens is None else max_output_tokens
    super().__init__(server_address, Handler)

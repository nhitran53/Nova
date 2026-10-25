"""Shared fixtures: a fake OpenAI-compatible chat server on 127.0.0.1 (ephemeral port, http.server in a thread). No real model, no other network."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def chat_completion(content, **message_extra):
    return {"id": "x", "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "stop",
                                                                "message": {"role": "assistant", "content": content, **message_extra}}]}


def reply_json(text, refusal=None):
    """The raw model output the generation layer expects: {"refusal": null|code, "text": "..."}."""
    return json.dumps({"refusal": refusal, "text": text})


class FakeLLMServer:
    """`reply` is what POST /v1/chat/completions answers: a dict/str (200), a callable(body) -> same, or a (status, payload) tuple.
    `delay_s` sleeps before answering. `health` / `models` are (status, payload) for GET /health and GET /v1/models (None -> 404).
    `requests` records the parsed JSON bodies of every chat call."""

    def __init__(self):
        self.reply = chat_completion(reply_json("Hello, happy to help."))
        self.delay_s = 0.0
        self.health = (200, {"status": "ok"})
        self.models = (200, {"object": "list", "data": [{"id": "fake-qwen"}]})
        self.requests = []
        self.headers = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, payload):
                data = payload if isinstance(payload, bytes) else (payload if isinstance(payload, str) else json.dumps(payload)).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except OSError:
                    pass

            def do_GET(self):
                entry = {"/health": outer.health, "/v1/models": outer.models}.get(self.path)
                self._send(*(entry if entry else (404, {"error": "not found"})))

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                outer.headers.append(dict(self.headers))
                if outer.delay_s:
                    time.sleep(outer.delay_s)
                if self.path != "/v1/chat/completions":
                    return self._send(404, {"error": "not found"})
                reply = outer.reply(body) if callable(outer.reply) else outer.reply
                if isinstance(reply, tuple):
                    return self._send(*reply)
                self._send(200, reply)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def llm_server():
    server = FakeLLMServer().start()
    yield server
    server.stop()


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch):
    for name in ("NOVA_LLM_RUNTIME", "NOVA_LLM_BASE_URL", "NOVA_LLM_MODEL", "NOVA_LLM_CONSTRAIN", "NOVA_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)

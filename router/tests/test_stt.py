"""stt.py with a fake Scribe client and fake router: no network, no mic, no model load."""
import pytest

from router import stt

KEY = "test-key-not-real"


class FakeConnection:
    def __init__(self, text, fail=False):
        self.handlers, self.sent, self.text, self.fail, self.closed = {}, [], text, fail, False

    def on(self, event, handler):
        self.handlers[event.value if hasattr(event, "value") else event] = handler

    async def send(self, msg):
        self.sent.append(msg)
        if len(self.sent) == 2:
            if self.fail:
                self.handlers["error"]({"message": f"bad {KEY}"})
            elif self.text:
                self.handlers["committed_transcript"]({"text": self.text})

    async def commit(self):
        pass

    async def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, conn):
        self.conn = conn
        self.speech_to_text = self
        self.realtime = self
        self.options = None

    async def connect(self, options):
        self.options = options
        return self.conn


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setattr(stt, "load_dotenv", lambda path: None)
    monkeypatch.setenv(stt.KEY_VAR, KEY)


def frames(n=50):
    return iter([b"\x00\x01" * 1600] * n)


def test_returns_first_committed_transcript_and_closes():
    conn = FakeConnection("  hey nova check oxygen ")
    client = FakeClient(conn)
    assert stt.transcribe(frames(), client_factory=lambda k: client) == "hey nova check oxygen"
    assert conn.closed and client.options["commit_strategy"].value == "vad"


def test_silence_times_out_to_empty():
    conn = FakeConnection("")
    assert stt.transcribe(frames(5), client_factory=lambda k: FakeClient(conn), timeout_s=0.2) == ""


def test_server_error_is_redacted():
    with pytest.raises(stt.SttError) as caught:
        stt.transcribe(frames(), client_factory=lambda k: FakeClient(FakeConnection("x", fail=True)))
    assert KEY not in str(caught.value) and "[redacted]" in str(caught.value)


def test_missing_key(monkeypatch):
    monkeypatch.delenv(stt.KEY_VAR)
    with pytest.raises(stt.SttError, match="not set"):
        stt.transcribe(frames())


def test_client_factory_failure_is_sanitized():
    def boom(key):
        raise ValueError(key)
    with pytest.raises(stt.SttError) as caught:
        stt.transcribe(frames(), client_factory=boom)
    assert KEY not in str(caught.value)


def test_listen_and_route_loads_router_in_background_and_handles_transcript():
    seen = {}

    class Handler:
        def handle(self, text):
            seen["text"] = text
            return {"route": "tool", "spoken": "ok"}

    out = stt.listen_and_route(frames=frames(), router_loader=lambda: "ROUTER",
                               handler_factory=lambda router: seen.setdefault("router", router) and Handler(),
                               client_factory=lambda k: FakeClient(FakeConnection("check oxygen")))
    assert out == {"transcript": "check oxygen", "turn": {"route": "tool", "spoken": "ok"}}
    assert seen["router"] == "ROUTER" and seen["text"] == "check oxygen"


def test_nothing_heard_skips_the_turn():
    out = stt.listen_and_route(frames=frames(3), router_loader=lambda: "R", timeout_s=0.2,
                               client_factory=lambda k: FakeClient(FakeConnection("")))
    assert out == {"transcript": "", "turn": None}


def test_router_load_failure_surfaces():
    def bad():
        raise FileNotFoundError("no checkpoint")
    with pytest.raises(FileNotFoundError):
        stt.listen_and_route(frames=frames(), router_loader=bad, handler_factory=lambda r: None,
                             client_factory=lambda k: FakeClient(FakeConnection("hi")))


def test_converse_loops_until_silence_and_reuses_one_handler():
    built, texts = [], []

    class Handler:
        def handle(self, text):
            texts.append(text)
            return {"route": "tool", "spoken": "re: " + text}

    scripts = iter(["one", "two", ""])

    def client_factory(key):
        return FakeClient(FakeConnection(next(scripts)))

    events = []
    turns = stt.converse(router_loader=lambda: "R", handler_factory=lambda r: built.append(r) or Handler(),
                         client_factory=client_factory, frames=__import__("itertools").cycle([b"\x00\x01" * 1600]), timeout_s=0.2, on_event=events.append)
    assert [t["transcript"] for t in turns] == ["one", "two"]
    assert built == ["R"] and texts == ["one", "two"]
    assert any("silence" in e for e in events)

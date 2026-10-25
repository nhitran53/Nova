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


# ---- live LLM + speech queue wiring (fake Scribe, fake generator, fake tts; no network) ----
import itertools
import json
import threading
import time

from router import llm_generator
from router.generation import FillerGenerator, GenerationResult
from router.tests.test_turn import FakeRouter

AUDIO = itertools.cycle([b"\x00\x01" * 1600])


class FakeTts:
    """Stands in for tts.speak: logs start/end of each line, optionally slow, optionally failing for chosen lines."""

    def __init__(self, delay=0.0, fail_on=()):
        self.events, self.delay, self.fail_on = [], delay, fail_on

    def __call__(self, text, route=None, block=True):
        assert block is True
        self.events.append(("start", text))
        time.sleep(self.delay)
        if text in self.fail_on:
            raise RuntimeError(f"tts down for {text!r}")
        self.events.append(("end", text))


class GeneratorThatLogs:
    def __init__(self, tts):
        self.tts = tts
        self.interim_playing_during_generation = None

    def generate(self, req):
        deadline = time.monotonic() + 1.0
        while ("start", "Let me think.") not in self.tts.events and time.monotonic() < deadline:
            time.sleep(0.005)  # the worker thread picks the enqueued interim line up asynchronously
        self.interim_playing_during_generation = ("start", "Let me think.") in self.tts.events and ("end", "Let me think.") not in self.tts.events
        return GenerationResult(json.dumps({"refusal": None, "text": "Hello there."}))


def healthy_server(monkeypatch):
    monkeypatch.setattr(llm_generator, "check_server", lambda url, timeout, **kw: (True, "server reachable (fake)"))


def scripted_client(*texts):
    scripts = iter(texts)
    return lambda key: FakeClient(FakeConnection(next(scripts, "")))


def test_speech_queue_plays_lines_in_order_one_at_a_time():
    tts = FakeTts(delay=0.05)
    speech = stt.SpeechQueue(tts)
    speech.enqueue("first", "llm_interim")
    speech.enqueue("second", "llm")
    assert speech.drain() == []
    assert tts.events == [("start", "first"), ("end", "first"), ("start", "second"), ("end", "second")]
    speech.close()


def test_speech_queue_error_is_kept_surfaced_once_and_does_not_stop_later_lines():
    tts = FakeTts(fail_on=("bad",))
    speech = stt.SpeechQueue(tts)
    speech.enqueue("bad")
    speech.enqueue("good")
    errors = speech.drain()
    assert len(errors) == 1 and "RuntimeError" in errors[0] and "bad" in errors[0]
    assert ("end", "good") in tts.events and speech.drain() == []
    speech.close()


def test_default_handler_gets_the_http_generator_when_the_server_is_healthy(monkeypatch, capsys):
    healthy_server(monkeypatch)
    handler = stt._default_handler(FakeRouter("NO_TOOL", 0.97), speak=False)
    primary = handler.generator.generators[0] if isinstance(handler.generator, llm_generator.FailoverGenerator) else handler.generator
    assert isinstance(primary, llm_generator.HTTPChatGenerator) and not isinstance(handler.generator, FillerGenerator)
    assert "using" in capsys.readouterr().err and handler.interim_fn is None and handler.tts_fn is None


def test_default_handler_falls_back_to_filler_with_one_visible_line(monkeypatch, capsys):
    monkeypatch.setattr(llm_generator, "check_server", lambda url, timeout, **kw: (False, "no LLM server reachable (fake)"))
    handler = stt._default_handler(FakeRouter("NO_TOOL", 0.97), speak=False)
    err = capsys.readouterr().err.strip().splitlines()
    assert isinstance(handler.generator, FillerGenerator) and len(err) == 1 and "falling back" in err[0]


def test_default_handler_options_no_llm_and_deadline(monkeypatch):
    healthy_server(monkeypatch)
    handler = stt._default_handler(FakeRouter(), speak=False, llm_options={"no_llm": True, "deadline_ms": 9000})
    assert isinstance(handler.generator, FillerGenerator) and handler.policy["generation"]["deadline_ms"] == 9000


def test_converse_plays_interim_then_final_while_generation_overlaps_and_ends_on_thank_you(monkeypatch):
    healthy_server(monkeypatch)
    tts = FakeTts(delay=0.1)
    speech = stt.SpeechQueue(tts)
    generator = GeneratorThatLogs(tts)

    def factory(router):
        handler = stt._default_handler(router, speak=True, speech=speech)
        handler.generator = generator
        return handler

    events = []
    turns = stt.converse(router_loader=lambda: FakeRouter("NO_TOOL", 0.97), handler_factory=factory, speech_queue=speech,
                         client_factory=scripted_client("tell me something", "thank you", "never reached"), frames=AUDIO,
                         timeout_s=2.0, on_event=events.append)
    assert [t["transcript"] for t in turns] == ["tell me something", "thank you"]
    assert turns[0]["turn"]["route"] == "llm" and turns[1]["turn"]["end_session"] is True
    assert generator.interim_playing_during_generation is True  # generation started while the interim line was still playing
    assert tts.events == [("start", "Let me think."), ("end", "Let me think."), ("start", "Hello there."), ("end", "Hello there."),
                          ("start", "Happy to help!"), ("end", "Happy to help!")]  # everything played before converse returned
    assert any("goodbye" in e for e in events) and sum(e == "listening..." for e in events) == 2


def test_converse_surfaces_a_tts_error_from_the_queue(monkeypatch):
    healthy_server(monkeypatch)
    tts = FakeTts(fail_on=("Hello there.",))
    speech = stt.SpeechQueue(tts)

    def factory(router):
        handler = stt._default_handler(router, speak=True, speech=speech)
        handler.generator = GeneratorThatLogs(tts)
        return handler

    events = []
    turns = stt.converse(router_loader=lambda: FakeRouter("NO_TOOL", 0.97), handler_factory=factory, speech_queue=speech,
                         client_factory=scripted_client("tell me something"), frames=AUDIO, timeout_s=2.0, max_turns=1,
                         on_event=events.append)
    assert "Hello there." in turns[0]["turn"]["tts_error"] and "RuntimeError" in turns[0]["turn"]["tts_error"]
    assert any(e.startswith("(speech failed:") and "tts down" in e for e in events)


def test_main_passes_llm_flags_to_converse(monkeypatch):
    seen = {}
    monkeypatch.setattr(stt, "converse", lambda *a, **k: seen.update(k) or [])
    assert stt.main(["--no-llm", "--runtime", "ollama", "--base-url", "http://x/v1", "--llm-model", "m", "--deadline-ms", "5000",
                     "--no-speak"]) == 0
    assert seen["llm_options"] == {"no_llm": True, "runtime": "ollama", "base_url": "http://x/v1", "llm_model": "m",
                                   "deadline_ms": 5000} and seen["speak"] is False


# ---- stop_event / on_turn (voice_server hooks); default behaviour is covered by the tests above ----
def test_transcribe_stop_event_ends_the_wait_early_with_empty_text():
    import threading
    import time
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    started = time.monotonic()
    assert stt.transcribe(frames(1), client_factory=lambda k: FakeClient(FakeConnection("")), timeout_s=30, stop_event=stop) == ""
    assert time.monotonic() - started < 5


def test_transcribe_with_an_unset_stop_event_still_returns_the_transcript():
    import threading
    assert stt.transcribe(frames(), client_factory=lambda k: FakeClient(FakeConnection("hi nova")), stop_event=threading.Event()) == "hi nova"


def test_converse_stops_before_listening_when_the_stop_event_is_already_set():
    import threading
    stop, events = threading.Event(), []
    stop.set()
    turns = stt.converse(router_loader=lambda: "R", handler_factory=lambda r: None, stop_event=stop, on_event=events.append)
    assert turns == [] and not any(e == "listening..." for e in events) and any("stopped" in e for e in events)


def test_converse_on_turn_gets_each_turn_and_stop_between_turns_ends_it():
    import itertools
    import threading
    stop, seen, events = threading.Event(), [], []

    class Handler:
        def handle(self, text):
            return {"route": "tool", "spoken": "re: " + text}

    def on_turn(out):
        seen.append(out["transcript"])
        stop.set()  # the caller asks to stop after the first reply

    turns = stt.converse(router_loader=lambda: "R", handler_factory=lambda r: Handler(), stop_event=stop, on_turn=on_turn,
                         client_factory=lambda k: FakeClient(FakeConnection("one")), frames=itertools.cycle([b"\x00\x01" * 1600]),
                         timeout_s=0.5, on_event=events.append)
    assert seen == ["one"] and [t["transcript"] for t in turns] == ["one"]
    assert sum(e == "listening..." for e in events) == 1 and any("stopped" in e for e in events)


@pytest.mark.parametrize('clear', [False, True])
def test_speech_queue_priority_order_and_clear_pending_preserve_current_and_alerts(clear):
    started, release = threading.Event(), threading.Event()
    played = []

    def speak(text, route, block):
        played.append(text)
        if text == 'current':
            started.set()
            assert release.wait(5)

    speech = stt.SpeechQueue(speak)
    speech.enqueue('current')
    assert started.wait(5)
    speech.enqueue('normal')
    speech.enqueue('alert first', priority=True)
    speech.enqueue('alert second', priority=True)
    if clear:
        speech.clear_pending()
    speech.enqueue('answer')
    release.set()
    assert speech.drain() == []
    assert played == ['current', 'alert first', 'alert second'] + ([] if clear else ['normal']) + ['answer']
    speech.close()


def test_default_handler_enqueues_alert_with_priority_before_read_only_answer(monkeypatch):
    healthy_server(monkeypatch)
    events = []

    class QueueSpy:
        def clear_pending(self):
            events.append('clear')

        def enqueue(self, text, route=None, priority=False):
            events.append((text, route, priority))

    handler = stt._default_handler(FakeRouter('call_o2', .99), speak=True, speech=QueueSpy())
    handler.alert_source = lambda: {'o2_pct': 14.2}
    out = handler.handle('check oxygen')
    assert events == ['clear', (out['lines'][0], 'alert', True), (out['lines'][1], 'tool', False)]

def test_converse_on_transcript_fires_before_the_handler_routes_the_text():
    import itertools
    import threading
    stop, order = threading.Event(), []

    class Handler:
        def handle(self, text):
            order.append(("handle", text))
            stop.set()
            return {"route": "tool", "spoken": "re: " + text}

    stt.converse(router_loader=lambda: "R", handler_factory=lambda r: Handler(), stop_event=stop,
                 on_transcript=lambda text: order.append(("transcript", text)),
                 client_factory=lambda k: FakeClient(FakeConnection("one")), frames=itertools.cycle([b"\x00\x01" * 1600]),
                 timeout_s=0.5, on_event=lambda e: None)
    assert order == [("transcript", "one"), ("handle", "one")]


@pytest.mark.parametrize("demo", [False, True])
def test_default_handler_datasource_preserves_live_readings_without_demo_faults(monkeypatch, demo):
    monkeypatch.delenv("NOVA_DOCS_DIR", raising=False)
    monkeypatch.delenv("NOVA_DEMO_SCENE", raising=False)

    class Datasource:
        def snapshot(self):
            return {"o2_pct": 50.0, "minutes_remaining": 100, "bpm": 80}

    datasource = Datasource()
    handler = stt._default_handler(FakeRouter("call_o2", .99), speak=False, datasource=datasource,
                                   llm_options={"no_llm": True, "demo_scene": demo})
    assert handler.alert_source == datasource.snapshot
    assert "rover_faults" not in handler.alert_source()
    assert handler.policy["demo_scene"]["enabled"] is demo
    if demo:
        out = handler.handle("what is this")
        assert out["route"] == "scene"
        assert out["spoken"] == handler.policy["demo_scene"]["no_fault_line"]
        assert handler.pending_confirmation is None and out["tool_calls"] == 0
    else:
        assert handler.retriever is None
        out = handler.handle("check oxygen")
        assert out["tool_calls"] == 1 and out["tool_result"]["ok"]
        assert out["tool_result"]["data"]["o2_pct"] == 50.0

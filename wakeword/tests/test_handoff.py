import importlib.util
import json
import sys
import threading
import types
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from conftest import silence, speech
from detector import Detection
from handoff import VoiceServerClient, VoiceServerError, WakeService


class FakeTime:
    """Injected clock + sleep: sleeping just advances the clock and is recorded."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class ScriptedDetector:
    """Fires on a loud frame, like a wake word would."""

    def __init__(self, log=None):
        self.resets = 0
        self.log = log if log is not None else []

    def process(self, frame):
        return Detection("hey_nova", 0.9) if np.abs(frame).max() > 1000 else None

    def reset(self):
        self.resets += 1


class FakeMic:
    """open_mic() factory: each open yields the next frame list; open/close are logged to `events`."""

    def __init__(self, events, sessions, open_errors=0):
        self.events = events
        self._sessions = list(sessions)
        self.open_errors = open_errors
        self.opens = 0

    def __call__(self):
        return self

    def __enter__(self):
        if self.open_errors:
            self.open_errors -= 1
            self.events.append("mic-open-failed")
            raise OSError("no such device")
        self.opens += 1
        self.events.append("mic-open")
        frames = self._sessions.pop(0) if self._sessions else []
        return iter(frames)

    def __exit__(self, *exc):
        self.events.append("mic-close")


class FakeClient:
    """VoiceServerClient stand-in. `statuses` is consumed by status(); the last one repeats."""

    base_url = "http://fake"

    def __init__(self, events, statuses=(), start_failures=0, start_reply=None, start_replies=None):
        self.events = events
        self.statuses = list(statuses)
        self.start_failures = start_failures
        self.start_reply = start_reply or {"state": "loading", "session_id": 1, "started": True}
        self.start_replies = list(start_replies or [])  # consumed first; then `start_reply` repeats
        self.stops = 0

    def start(self):
        self.events.append("POST /voice/start")
        if self.start_failures:
            self.start_failures -= 1
            raise VoiceServerError("POST /voice/start failed: ConnectionRefusedError: refused")
        return self.start_replies.pop(0) if self.start_replies else self.start_reply

    def stop(self):
        self.events.append("POST /voice/stop")
        self.stops += 1
        self.statuses = [st("idle", self.statuses[-1].get("session_id", 0))]  # like voice_server: idle at once
        return self.statuses[0]

    def status(self):
        self.events.append("GET /voice/status")
        status = self.statuses[0] if len(self.statuses) == 1 else self.statuses.pop(0)
        if isinstance(status, Exception):
            raise status
        return status


def st(state, session_id=1, **extra):
    return {"state": state, "session_id": session_id, **extra}


def make_service(events, sessions, client, fake_time=None, detector=None, **kwargs):
    fake_time = fake_time or FakeTime()
    mic = FakeMic(events, sessions, open_errors=kwargs.pop("open_errors", 0))
    wakes = []
    service = WakeService(mic, detector or ScriptedDetector(), client, on_wake=wakes.append, clock=fake_time.clock,
                          sleep=fake_time.sleep, **kwargs)
    return service, mic, wakes, fake_time


def test_wake_starts_conversation_releases_mic_then_reopens_after_cooldown():
    events = []
    client = FakeClient(events, [st("idle", 0), st("loading"), st("listening"), st("speaking"), st("ended", last_transcript="hi", last_reply="yo")])
    service, mic, wakes, fake_time = make_service(
        events, [silence(2) + speech(1) + silence(1), silence(3)], client, cooldown_s=3.0, status_check_interval_s=1e9)

    assert service.run() == 1
    assert [w.name for w in wakes] == ["hey_nova"]
    # status is checked before every mic open; the mic is closed BEFORE the conversation is started and only reopened after it ended
    assert events[:4] == ["GET /voice/status", "mic-open", "mic-close", "POST /voice/start"]
    assert events.index("mic-close") < events.index("POST /voice/start")
    assert events.count("GET /voice/status") == 6
    assert events[-3:] == ["GET /voice/status", "mic-open", "mic-close"]
    assert mic.opens == 2
    # polling at 0.5 s (3 waits before 'ended'), then the 3 s cooldown before the mic reopens
    assert fake_time.sleeps == [0.5, 0.5, 0.5, 3.0]


def test_detector_is_reset_whenever_the_mic_reopens():
    events = []
    detector = ScriptedDetector()
    client = FakeClient(events, [st("ended")])
    service, _, _, _ = make_service(events, [speech(1), silence(1)], client, detector=detector, status_check_interval_s=1e9)
    service.run()
    assert detector.resets == 2


def test_stale_ended_status_from_previous_session_is_not_taken_for_the_end():
    events = []
    client = FakeClient(events, [st("ended", session_id=1), st("ended", session_id=1), st("speaking", session_id=2), st("ended", session_id=2)],
                        start_reply={"state": "loading", "session_id": 2, "started": True})
    service, _, _, fake_time = make_service(events, [speech(1), silence(1)], client, status_check_interval_s=1e9)
    service.run()
    assert events.count("GET /voice/status") == 5  # mic gate, stale ended, speaking, ended, mic gate
    assert fake_time.sleeps[:2] == [0.5, 0.5]


def test_conversation_already_running_is_joined():
    events = []
    client = FakeClient(events, [st("idle", session_id=0), st("listening", session_id=4), st("ended", session_id=4)],
                        start_reply={"state": "listening", "session_id": 4, "started": False})
    service, _, _, _ = make_service(events, [speech(1), silence(1)], client, status_check_interval_s=1e9)
    assert service.run() == 1
    assert events.count("POST /voice/stop") == 0 and events.count("GET /voice/status") == 4


def test_server_down_retries_with_backoff_then_gives_up_and_keeps_listening(caplog):
    events = []
    client = FakeClient(events, [st("idle")], start_failures=99)
    service, mic, _, fake_time = make_service(
        events, [speech(1), silence(1)], client, start_attempts=3, start_retry_s=1.0, cooldown_s=3.0, status_check_interval_s=1e9)
    with caplog.at_level("WARNING"):
        service.run()
    assert events.count("POST /voice/start") == 3
    assert fake_time.sleeps == [1.0, 2.0, 3.0]  # exponential backoff between attempts, then the normal cooldown
    assert "is router/voice_server running at http://fake?" in caplog.text
    assert "giving up on this wake word" in caplog.text
    assert mic.opens == 2  # went back to listening


def test_server_recovers_on_a_retry():
    events = []
    client = FakeClient(events, [st("ended")], start_failures=1)
    service, _, _, fake_time = make_service(events, [speech(1), silence(1)], client, status_check_interval_s=1e9, cooldown_s=3.0)
    service.run()
    assert events.count("POST /voice/start") == 2
    assert fake_time.sleeps == [1.0, 3.0]


def test_conversation_timeout_asks_server_to_stop():
    events = []
    client = FakeClient(events, [st("idle", 0), st("listening")])  # never ends (until stopped)
    service, mic, _, fake_time = make_service(
        events, [speech(1), silence(1)], client, conversation_timeout_s=5.0, poll_interval_s=0.5, cooldown_s=3.0, status_check_interval_s=1e9)
    service.run()
    assert events.count("POST /voice/stop") == 1
    assert 5.0 <= sum(fake_time.sleeps[:-1]) < 6.0
    assert fake_time.sleeps[-1] == 3.0 and mic.opens == 2


def test_server_vanishing_mid_conversation_ends_the_wait_after_the_grace_period():
    events = []
    client = FakeClient(events, [st("idle", 0), st("listening"), VoiceServerError("GET /voice/status failed: refused")])
    service, mic, _, fake_time = make_service(
        events, [speech(1), silence(1)], client, status_failure_grace_s=2.0, conversation_timeout_s=300.0, status_check_interval_s=1e9)
    service.run()
    assert sum(fake_time.sleeps) < 10 and mic.opens == 2


def test_error_state_ends_the_wait(caplog):
    events = []
    client = FakeClient(events, [st("idle", 0), st("loading"), st("error", error="mic busy")])
    service, mic, _, _ = make_service(events, [speech(1), silence(1)], client, status_check_interval_s=1e9)
    with caplog.at_level("WARNING"):
        service.run()
    assert "mic busy" in caplog.text and mic.opens == 2


def test_ctrl_c_during_conversation_stops_it_and_propagates():
    events = []
    client = FakeClient(events, [st("idle", 0), st("listening")])
    fake_time = FakeTime()

    def interrupting_sleep(seconds):
        raise KeyboardInterrupt

    service, _, _, _ = make_service(events, [speech(1)], client, fake_time=fake_time, status_check_interval_s=1e9)
    service._sleep = interrupting_sleep
    with pytest.raises(KeyboardInterrupt):
        service.run()
    assert "POST /voice/stop" in events


def test_conversation_started_elsewhere_releases_the_mic_too():
    events = []
    client = FakeClient(events, [st("idle", session_id=0), st("listening", session_id=7), st("listening", session_id=7), st("ended", session_id=7)])
    service, mic, wakes, _ = make_service(
        events, [silence(1) + silence(1) + silence(1), silence(1)], client, status_check_interval_s=0.0)
    assert service.run() == 0 and wakes == []
    assert "POST /voice/start" not in events
    assert events.index("mic-close") < events.index("GET /voice/status", events.index("mic-close")) and mic.opens == 2


def test_mic_errors_back_off_exponentially_and_recover():
    events = []
    client = FakeClient(events, [st("ended")])
    service, mic, wakes, fake_time = make_service(
        events, [speech(1), silence(1)], client, open_errors=3, cooldown_s=0.0, status_check_interval_s=1e9)
    service.run()
    assert fake_time.sleeps[:3] == [1.0, 2.0, 4.0]
    assert len(wakes) == 1


def test_mic_stream_dying_mid_listen_is_retried():
    events = []

    class DyingFrames:
        def __iter__(self):
            return self

        def __next__(self):
            raise RuntimeError("no audio from the microphone for 5 s")

    class Mic(FakeMic):
        def __enter__(self):
            super().__enter__()
            return DyingFrames() if self.opens == 1 else iter(speech(1))

    fake_time = FakeTime()
    mic = Mic(events, [])
    client = FakeClient(events, [st("ended")])
    service = WakeService(mic, ScriptedDetector(), client, cooldown_s=0.0, status_check_interval_s=1e9,
                          clock=fake_time.clock, sleep=fake_time.sleep)
    assert service.run(max_wakes=1) == 1
    assert fake_time.sleeps[0] == 1.0 and mic.opens == 2


def test_mic_is_not_opened_while_the_server_is_busy_and_polling_sleeps():
    events = []
    client = FakeClient(events, [st("loading"), st("listening"), st("speaking"), st("idle")])
    service, mic, _, fake_time = make_service(events, [silence(1)], client, status_check_interval_s=1e9)
    service.run()
    assert events[:5] == ["GET /voice/status"] * 4 + ["mic-open"]
    assert fake_time.sleeps == [1.0, 1.0, 1.0]  # no hot loop


def test_mic_gate_runs_before_every_open_including_after_errors_and_conversations():
    events = []
    client = FakeClient(events, [st("idle", 0), st("ended")])
    service, mic, _, _ = make_service(events, [speech(1), silence(1)], client, open_errors=1, status_check_interval_s=1e9)
    service.run()
    opens = [i for i, e in enumerate(events) if e in ("mic-open", "mic-open-failed")]
    assert len(opens) == 3 and all(events[i - 1] == "GET /voice/status" for i in opens)


def test_unreachable_server_does_not_hold_the_mic_closed():
    events = []
    client = FakeClient(events, [VoiceServerError("refused")])
    service, mic, _, fake_time = make_service(events, [silence(1)], client, status_check_interval_s=1e9)
    service.run()
    assert mic.opens == 1 and fake_time.sleeps == []


def test_start_refused_while_idle_is_retried_with_backoff_before_the_mic_reopens():
    events = []
    refused = {**st("idle", 1), "started": False}
    accepted = {**st("loading", 2), "started": True}
    client = FakeClient(events, [st("idle", 1), st("ended", 2)], start_replies=[refused, refused], start_reply=accepted)
    service, mic, _, fake_time = make_service(events, [speech(1), silence(1)], client, cooldown_s=3.0, status_check_interval_s=1e9)
    service.run()
    assert events.count("POST /voice/start") == 3
    assert fake_time.sleeps[:2] == [0.5, 1.0]
    last_start = len(events) - 1 - events[::-1].index("POST /voice/start")
    assert "mic-open" not in events[events.index("mic-close"):last_start]  # mic stayed closed until a start was accepted


def test_start_refused_forever_is_bounded_then_the_mic_reopens(caplog):
    events = []
    refused = {**st("idle", 1), "started": False}
    client = FakeClient(events, [st("idle", 1)], start_reply=refused)
    service, mic, _, fake_time = make_service(
        events, [speech(1), silence(1)], client, cooldown_s=3.0, start_refused_attempts=5, status_check_interval_s=1e9)
    with caplog.at_level("WARNING"):
        service.run()
    assert events.count("POST /voice/start") == 5
    assert fake_time.sleeps == [0.5, 1.0, 2.0, 4.0, 3.0]  # backoff capped at 4 s, then the normal cooldown
    assert "giving up on this wake word" in caplog.text and mic.opens == 2


# --- the REAL router/voice_server.VoiceSession (loaded from its file with torch-free stubs for its imports) ---

def load_real_voice_session():
    path = Path(__file__).resolve().parents[2] / "router" / "voice_server.py"
    package = "_nova_router_under_test"
    stt = types.ModuleType(package + ".stt")
    stt.KEY_VAR = "ELEVENLABS_API_KEY"
    stt.converse = None
    stt.BackgroundRouter = lambda loader: types.SimpleNamespace(get=loader)
    llm = types.ModuleType(package + ".llm_generator")
    llm.API_KEY_ENV = "NOVA_LLM_API_KEY"
    llm.redact_secrets = lambda text, secret: text
    root = types.ModuleType(package)
    root.__path__ = []
    stubs = {package: root, package + ".stt": stt, package + ".llm_generator": llm}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(package + ".voice_server", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        for name in [*stubs, package + ".voice_server"]:
            sys.modules.pop(name, None)
    return module.VoiceSession


class SessionClient:
    """What POST /voice/start|stop and GET /voice/status return, straight from a real VoiceSession (same merge as the HTTP handler)."""
    base_url = "http://real-session"

    def __init__(self, session, events):
        self.session = session
        self.events = events

    def start(self):
        self.events.append("POST /voice/start")
        started, status = self.session.start()
        return {**status, "started": started}

    def stop(self):
        self.events.append("POST /voice/stop")
        return self.session.stop()

    def status(self):
        self.events.append("GET /voice/status")
        return self.session.status()


def test_real_voice_session_with_delayed_worker_shutdown_is_retried_until_the_start_is_accepted():
    """stop() makes the real session idle at once, but start() is refused until the old worker thread is gone."""
    VoiceSession = load_real_voice_session()
    events = []
    release_worker = threading.Event()
    calls = []

    def converse(on_event, on_turn, stop_event, router_loader, **options):
        calls.append(1)
        on_event("listening...")
        if len(calls) == 1:  # conversation that never ends by itself, and is slow to shut down once stopped
            stop_event.wait()
            release_worker.wait()

    session = VoiceSession(lambda: object(), converse_fn=converse)
    client = SessionClient(session, events)
    fake_time = FakeTime()
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        fake_time.sleep(seconds)
        if events.count("POST /voice/start") >= 3:  # the old worker finishes shutting down during the retry backoff
            release_worker.set()
            session.join(5)

    mic = FakeMic(events, [speech(1), speech(1), silence(1)])
    service = WakeService(mic, ScriptedDetector(), client, cooldown_s=0.0, conversation_timeout_s=5.0, poll_interval_s=0.5,
                          status_check_interval_s=1e9, clock=fake_time.clock, sleep=sleep)
    service.run()
    session.join(5)

    assert events.count("POST /voice/stop") == 1  # our own timeout-stop of conversation 1 (state went idle at once)
    starts = [i for i, e in enumerate(events) if e == "POST /voice/start"]
    assert len(starts) >= 3  # conversation 1, then the second wake: refused (worker alive) and retried
    # the mic stayed closed from the second wake until a start was accepted
    second_wake_close = [i for i, e in enumerate(events) if e == "mic-close"][1]
    assert "mic-open" not in events[second_wake_close:starts[-1]]
    assert session.status()["session_id"] == 2 and calls == [1, 1]


def test_dry_run_reports_detections_without_touching_the_server_or_the_mic():
    events = []
    service, mic, wakes, fake_time = make_service(events, [silence(1) + speech(1) + silence(1) + speech(1) + silence(1)], None)
    assert service.run() == 2
    assert len(wakes) == 2
    assert events == ["mic-open", "mic-close"] and fake_time.sleeps == []


def test_dry_run_max_wakes_stops():
    events = []
    service, _, wakes, _ = make_service(events, [speech(1) + speech(1) + speech(1)], None)
    assert service.run(max_wakes=2) == 2 and len(wakes) == 2


# --- the real HTTP client against a fake voice_server (same routes/shape as router/voice_server.py) ---

class FakeVoiceServer:
    def __init__(self):
        self.requests = []  # (method, path, Origin header)
        self.status = {"state": "idle", "session_id": 0, "seq": 0, "router_ready": True, "last_transcript": None, "last_reply": None,
                       "route": None, "error": None}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _handle(self, method):
                outer.requests.append((method, self.path, self.headers.get("Origin")))
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                if (method, self.path) == ("GET", "/voice/status"):
                    self._send(200, outer.status)
                elif (method, self.path) == ("POST", "/voice/start"):
                    started = outer.status["state"] not in ("loading", "listening", "speaking")
                    if started:
                        outer.status = {**outer.status, "state": "loading", "session_id": outer.status["session_id"] + 1}
                    self._send(200, {**outer.status, "started": started})
                elif (method, self.path) == ("POST", "/voice/stop"):
                    outer.status = {**outer.status, "state": "idle"}
                    self._send(200, outer.status)
                else:
                    self._send(404, {"error": "not found"})

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def close(self):
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def voice_server():
    server = FakeVoiceServer()
    yield server
    server.close()


def test_client_talks_to_voice_server_without_an_origin_header(voice_server):
    client = VoiceServerClient(voice_server.url)
    started = client.start()
    assert started["started"] is True and started["session_id"] == 1 and started["state"] == "loading"
    assert client.start()["started"] is False  # already running
    assert client.status()["state"] == "loading"
    assert client.stop()["state"] == "idle"
    assert voice_server.requests == [("POST", "/voice/start", None), ("POST", "/voice/start", None),
                                     ("GET", "/voice/status", None), ("POST", "/voice/stop", None)]


def test_client_unreachable_server_raises_voice_server_error():
    client = VoiceServerClient("http://127.0.0.1:1", timeout_s=0.5)
    with pytest.raises(VoiceServerError):
        client.status()


def test_client_http_error_raises_voice_server_error(voice_server):
    client = VoiceServerClient(voice_server.url)
    with pytest.raises(VoiceServerError, match="HTTP 404"):
        client._request("GET", "/nope")


def test_full_handoff_loop_over_http_with_fake_mic(voice_server):
    """Real client + real HTTP server; conversation 'runs' for a few polls then ends. Mic must be closed in between."""
    events = []
    polls = {"n": 0}

    class ConversationScript(VoiceServerClient):
        def status(self):
            result = super().status()
            if voice_server.requests[-1][:2] == ("GET", "/voice/status") and "POST /voice/start" in events:
                polls["n"] += 1
                if polls["n"] == 3:
                    voice_server.status = {**voice_server.status, "state": "ended", "last_transcript": "weather", "last_reply": "sunny"}
            return result

        def start(self):
            events.append("POST /voice/start")
            return super().start()

    fake_time = FakeTime()
    mic = FakeMic(events, [silence(1) + speech(1) + silence(1), silence(2)])
    service = WakeService(mic, ScriptedDetector(), ConversationScript(voice_server.url), cooldown_s=1.0, status_check_interval_s=1e9,
                          clock=fake_time.clock, sleep=fake_time.sleep)
    assert service.run() == 1
    assert events[:3] == ["mic-open", "mic-close", "POST /voice/start"]
    assert events[-2:] == ["mic-open", "mic-close"]
    assert voice_server.status["state"] == "ended"
    assert all(origin is None for _, _, origin in voice_server.requests)

import json
import threading
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

    def __init__(self, events, statuses=(), start_failures=0, start_reply=None):
        self.events = events
        self.statuses = list(statuses)
        self.start_failures = start_failures
        self.start_reply = start_reply or {"state": "loading", "session_id": 1, "started": True}
        self.stops = 0

    def start(self):
        self.events.append("POST /voice/start")
        if self.start_failures:
            self.start_failures -= 1
            raise VoiceServerError("POST /voice/start failed: ConnectionRefusedError: refused")
        return self.start_reply

    def stop(self):
        self.events.append("POST /voice/stop")
        self.stops += 1
        return {}

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
    client = FakeClient(events, [st("loading"), st("listening"), st("speaking"), st("ended", last_transcript="hi", last_reply="yo")])
    service, mic, wakes, fake_time = make_service(
        events, [silence(2) + speech(1) + silence(1), silence(3)], client, cooldown_s=3.0, status_check_interval_s=1e9)

    assert service.run() == 1
    assert [w.name for w in wakes] == ["hey_nova"]
    # the mic is closed BEFORE the conversation is started and only reopened after it ended
    assert events[:3] == ["mic-open", "mic-close", "POST /voice/start"]
    assert events.index("mic-close") < events.index("POST /voice/start")
    assert events.count("GET /voice/status") == 4
    assert events[-2:] == ["mic-open", "mic-close"]
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
    client = FakeClient(events, [st("ended", session_id=1), st("speaking", session_id=2), st("ended", session_id=2)],
                        start_reply={"state": "loading", "session_id": 2, "started": True})
    service, _, _, fake_time = make_service(events, [speech(1), silence(1)], client, status_check_interval_s=1e9)
    service.run()
    assert events.count("GET /voice/status") == 3
    assert fake_time.sleeps[:2] == [0.5, 0.5]


def test_conversation_already_running_is_joined():
    events = []
    client = FakeClient(events, [st("listening", session_id=4), st("ended", session_id=4)],
                        start_reply={"state": "listening", "session_id": 4, "started": False})
    service, _, _, _ = make_service(events, [speech(1), silence(1)], client, status_check_interval_s=1e9)
    assert service.run() == 1
    assert events.count("POST /voice/stop") == 0 and events.count("GET /voice/status") == 2


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
    client = FakeClient(events, [st("listening")])  # never ends
    service, mic, _, fake_time = make_service(
        events, [speech(1), silence(1)], client, conversation_timeout_s=5.0, poll_interval_s=0.5, cooldown_s=3.0, status_check_interval_s=1e9)
    service.run()
    assert events.count("POST /voice/stop") == 1
    assert 5.0 <= sum(fake_time.sleeps[:-1]) < 6.0
    assert fake_time.sleeps[-1] == 3.0 and mic.opens == 2


def test_server_vanishing_mid_conversation_ends_the_wait_after_the_grace_period():
    events = []
    client = FakeClient(events, [st("listening"), VoiceServerError("GET /voice/status failed: refused")])
    service, mic, _, fake_time = make_service(
        events, [speech(1), silence(1)], client, status_failure_grace_s=2.0, conversation_timeout_s=300.0, status_check_interval_s=1e9)
    service.run()
    assert sum(fake_time.sleeps) < 10 and mic.opens == 2


def test_error_state_ends_the_wait(caplog):
    events = []
    client = FakeClient(events, [st("loading"), st("error", error="mic busy")])
    service, mic, _, _ = make_service(events, [speech(1), silence(1)], client, status_check_interval_s=1e9)
    with caplog.at_level("WARNING"):
        service.run()
    assert "mic busy" in caplog.text and mic.opens == 2


def test_ctrl_c_during_conversation_stops_it_and_propagates():
    events = []
    client = FakeClient(events, [st("listening")])
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

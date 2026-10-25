"""voice_server with a fake converse and a fake router loader: localhost ephemeral port only, no mic, no TTS, no model."""
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from router import stt, voice_server


def fake_converse_factory(script, calls):
    """A converse stand-in: `script(on_event, on_turn, stop_event)` plays the conversation."""
    def converse(**kwargs):
        calls.append(kwargs)
        return script(kwargs["on_event"], kwargs["on_turn"], kwargs["stop_event"])
    return converse


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def serve():
    started = []

    def make(script, router_loader=lambda: "ROUTER", options=None):
        calls = []
        session = voice_server.VoiceSession(router_loader, converse_fn=fake_converse_factory(script, calls),
                                            converse_options=options)
        server = voice_server.make_server(session, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started.append((server, session))
        return f"http://127.0.0.1:{server.server_address[1]}", session, calls

    yield make
    for server, session in started:
        session.stop()
        server.shutdown()
        server.server_close()


def call(base, path, method="GET", headers=None):
    request = urllib.request.Request(base + path, method=method, data=b"" if method == "POST" else None, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def blocking_until_stopped(on_event, on_turn, stop_event):
    on_event("listening...")
    stop_event.wait(5)
    return []


def test_status_is_idle_before_any_session_and_has_the_documented_keys(serve):
    base, _, _ = serve(blocking_until_stopped)
    code, status = call(base, "/voice/status")
    assert code == 200
    assert set(status) >= {"state", "last_transcript", "last_reply", "route", "error", "session_id", "seq"}
    assert status["state"] == "idle" and status["session_id"] == 0 and status["error"] is None


def test_start_runs_converse_and_reports_listening_then_turn_then_ended(serve):
    release = threading.Event()

    def script(on_event, on_turn, stop_event):
        on_event("listening...")
        on_turn({"transcript": "check oxygen", "turn": {"spoken": "Oxygen is 84 percent.", "route": "tool"}})
        release.wait(5)
        return [object()]

    base, session, calls = serve(script, options={"speak": False, "timeout_s": 3.0})
    code, started = call(base, "/voice/start", "POST")
    assert code == 200 and started["started"] is True and started["session_id"] == 1
    assert wait_for(lambda: call(base, "/voice/status")[1]["state"] == "speaking")
    status = call(base, "/voice/status")[1]
    assert (status["last_transcript"], status["last_reply"], status["route"]) == ("check oxygen", "Oxygen is 84 percent.", "tool")
    release.set()
    assert wait_for(lambda: call(base, "/voice/status")[1]["state"] == "ended")
    assert calls[0]["speak"] is False and calls[0]["timeout_s"] == 3.0 and callable(calls[0]["router_loader"])
    assert calls[0]["router_loader"]() == "ROUTER"


def test_seq_increases_on_every_change(serve):
    base, _, _ = serve(blocking_until_stopped)
    seqs = [call(base, "/voice/status")[1]["seq"]]
    call(base, "/voice/start", "POST")
    assert wait_for(lambda: call(base, "/voice/status")[1]["state"] == "listening")
    seqs.append(call(base, "/voice/status")[1]["seq"])
    call(base, "/voice/stop", "POST")
    seqs.append(call(base, "/voice/status")[1]["seq"])
    assert seqs == sorted(set(seqs)) and len(seqs) == 3


def test_second_start_while_running_is_ignored(serve):
    base, _, calls = serve(blocking_until_stopped)
    assert call(base, "/voice/start", "POST")[1]["started"] is True
    assert wait_for(lambda: call(base, "/voice/status")[1]["state"] == "listening")
    code, second = call(base, "/voice/start", "POST")
    assert code == 200 and second["started"] is False and second["session_id"] == 1
    assert len(calls) == 1


def test_stop_sets_the_flag_goes_idle_and_allows_a_new_session(serve):
    base, session, calls = serve(blocking_until_stopped)
    call(base, "/voice/start", "POST")
    assert wait_for(lambda: call(base, "/voice/status")[1]["state"] == "listening")
    code, status = call(base, "/voice/stop", "POST")
    assert code == 200 and status["state"] == "idle"
    assert calls[0]["stop_event"].is_set()
    session.join(3)
    assert call(base, "/voice/status")[1]["state"] == "idle"  # the late "ended" of the stopped run does not overwrite idle
    again = call(base, "/voice/start", "POST")[1]
    assert again["started"] is True and again["session_id"] == 2


def test_stop_without_a_session_is_harmless(serve):
    base, _, _ = serve(blocking_until_stopped)
    code, status = call(base, "/voice/stop", "POST")
    assert code == 200 and status["state"] == "idle"


def test_start_while_the_stopped_run_is_still_shutting_down_is_refused(serve):
    release = threading.Event()

    def stubborn(on_event, on_turn, stop_event):
        on_event("listening...")
        release.wait(5)  # ignores the stop flag until released (a reply still being spoken)
        return []

    base, session, _ = serve(stubborn)
    call(base, "/voice/start", "POST")
    assert wait_for(lambda: call(base, "/voice/status")[1]["state"] == "listening")
    call(base, "/voice/stop", "POST")
    refused = call(base, "/voice/start", "POST")[1]
    assert refused["started"] is False and refused["state"] == "idle"
    release.set()
    session.join(3)
    assert call(base, "/voice/start", "POST")[1]["started"] is True


def test_converse_exception_becomes_error_state_without_the_secret(serve, monkeypatch):
    monkeypatch.setenv(stt.KEY_VAR, "sk-supersecret-123")

    def script(on_event, on_turn, stop_event):
        raise stt.SttError("server error: bad key sk-supersecret-123")

    base, session, _ = serve(script)
    call(base, "/voice/start", "POST")
    assert wait_for(lambda: call(base, "/voice/status")[1]["state"] == "error")
    status = call(base, "/voice/status")[1]
    assert "SttError" in status["error"] and "sk-supersecret-123" not in json.dumps(status)
    session.join(3)
    assert call(base, "/voice/start", "POST")[1]["started"] is True  # an error session can be restarted


def test_playback_failure_is_reported_without_ending_the_session(serve):
    def script(on_event, on_turn, stop_event):
        on_event("listening...")
        on_event("(speech failed: RuntimeError: no audio device)")
        stop_event.wait(5)
        return []

    base, _, _ = serve(script)
    call(base, "/voice/start", "POST")
    assert wait_for(lambda: call(base, "/voice/status")[1]["error"] is not None)
    status = call(base, "/voice/status")[1]
    assert status["state"] == "listening" and "no audio device" in status["error"]


def test_router_is_prewarmed_at_construction_and_ready_flag_is_reported(serve):
    gate = threading.Event()
    loads = []

    def slow_loader():
        loads.append(1)
        gate.wait(5)
        return "ROUTER"

    base, _, _ = serve(blocking_until_stopped, router_loader=slow_loader)
    assert wait_for(lambda: loads)  # loading began before any /voice/start
    assert call(base, "/voice/status")[1]["router_ready"] is False
    gate.set()
    assert wait_for(lambda: call(base, "/voice/status")[1]["router_ready"] is True)
    call(base, "/voice/start", "POST")
    call(base, "/voice/stop", "POST")
    assert loads == [1]  # starting a session does not reload


def test_router_load_failure_surfaces_as_error_through_the_converse_loader(serve):
    def bad_loader():
        raise FileNotFoundError("no checkpoint")

    def script(on_event, on_turn, stop_event):
        calls[0]["router_loader"]()  # what stt.converse's BackgroundRouter.get() does with the loader
        return []

    base, session, calls = serve(script, router_loader=bad_loader)
    call(base, "/voice/start", "POST")
    assert wait_for(lambda: call(base, "/voice/status")[1]["state"] == "error")
    assert "FileNotFoundError" in call(base, "/voice/status")[1]["error"]


def test_browser_requests_with_an_origin_header_are_refused(serve):
    base, _, calls = serve(blocking_until_stopped)
    code, _ = call(base, "/voice/start", "POST", headers={"Origin": "http://evil.example"})
    assert code == 403 and calls == []
    assert call(base, "/voice/status", headers={"Origin": "http://evil.example"})[0] == 403


def test_unknown_paths_and_methods_are_404(serve):
    base, _, _ = serve(blocking_until_stopped)
    assert call(base, "/nope")[0] == 404
    assert call(base, "/nope", "POST")[0] == 404
    assert call(base, "/voice/start")[0] == 404  # start is POST only


def test_real_stt_converse_honours_the_stop_flag_through_the_session(serve, monkeypatch):
    """The real stt.converse with a fake listen_and_route: stop between turns ends it with on_turn called per turn."""
    session_events = []
    stop_holder = {}

    def fake_listen(timeout_s=None, background=None, handler=None, stop_event=None, **kwargs):
        stop_holder["event"] = stop_event
        if len(session_events) == 0:
            session_events.append("turn")
            return {"transcript": "hello", "turn": {"spoken": "Hi.", "route": "NO_TOOL"}}
        stop_event.wait(5)
        return {"transcript": "", "turn": None}

    monkeypatch.setattr(stt, "listen_and_route", fake_listen)
    session = voice_server.VoiceSession(lambda: "R", converse_options={"speak": False, "handler_factory": lambda r: None})
    server = voice_server.make_server(session, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        call(base, "/voice/start", "POST")
        assert wait_for(lambda: call(base, "/voice/status")[1]["last_reply"] == "Hi.")
        assert wait_for(lambda: stop_holder.get("event") is not None and len(session_events) == 1)
        call(base, "/voice/stop", "POST")
        session.join(3)
        assert not session.is_busy() and call(base, "/voice/status")[1]["state"] == "idle"
    finally:
        server.shutdown()
        server.server_close()

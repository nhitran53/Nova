"""voice_server with a fake converse and a fake router loader: localhost ephemeral port only, no mic, no TTS, no model."""
import json
import sys
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

    def make(script, router_loader=lambda: "ROUTER", options=None, datasource=None):
        calls = []
        session = voice_server.VoiceSession(router_loader, converse_fn=fake_converse_factory(script, calls),
                                            converse_options=options)
        server = voice_server.make_server(session, "127.0.0.1", 0, datasource=datasource)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started.append((server, session))
        return f"http://127.0.0.1:{server.server_address[1]}", session, calls

    yield make
    for server, session in started:
        session.stop()
        server.shutdown()
        server.server_close()


def call(base, path, method="GET", headers=None, body=None):
    data = json.dumps(body).encode() if body is not None else b"" if method == "POST" else None
    request = urllib.request.Request(base + path, method=method, data=data, headers=headers or {})
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
    assert set(status) >= {"state", "last_transcript", "last_reply", "route", "error", "session_id", "seq", "display", "display_seq", "alert",
                          "navigation_action", "navigation_destination", "navigation_seq"}
    assert status["state"] == "idle" and status["session_id"] == 0 and status["error"] is None
    assert status["display"] is None and status["display_seq"] == 0 and status["alert"] is None


DISPLAY = {"doc_id": "assembledrivemodules", "section_id": "assembledrivemodules:drive-module-assembly:0",
           "title": "Drive Module Assembly", "asset": "docs/AssembleDriveModules.md", "source": "docs/AssembleDriveModules.md"}


def test_display_is_exposed_kept_after_a_turn_without_one_and_display_seq_counts_new_displays(serve):
    step, done = threading.Event(), threading.Event()

    def script(on_event, on_turn, stop_event):
        on_turn({"transcript": "what is this", "turn": {"spoken": "scene", "route": "scene"}})
        step.wait(5)
        on_turn({"transcript": "yes please", "turn": {"spoken": "queued", "route": "print_confirmed", "display": DISPLAY}})
        done.wait(5)
        on_turn({"transcript": "thanks", "turn": {"spoken": "Happy to help!", "route": "session_end"}})
        done.wait(5)
        return []

    base, session, _ = serve(script)
    call(base, "/voice/start", "POST")
    assert wait_for(lambda: call(base, "/voice/status")[1]["last_reply"] == "scene")
    status = call(base, "/voice/status")[1]
    assert status["display"] is None and status["display_seq"] == 0 and status["alert"] is None
    step.set()
    assert wait_for(lambda: call(base, "/voice/status")[1]["last_reply"] == "queued")
    status = call(base, "/voice/status")[1]
    assert status["display"] == DISPLAY and status["display_seq"] == 1
    done.set()
    assert wait_for(lambda: call(base, "/voice/status")[1]["last_reply"] == "Happy to help!")
    status = call(base, "/voice/status")[1]
    assert status["display"] == DISPLAY and status["display_seq"] == 1 and status["route"] == "session_end"


def test_telemetry_can_be_read_without_starting_voice(serve):
    class Source:
        def telemetry(self):
            return {"schema_version": 1, "available": True, "source": "postgres", "o2_percent": 87.5}

    base, session, calls = serve(blocking_until_stopped, datasource=Source())
    code, payload = call(base, "/telemetry")
    assert code == 200 and payload["o2_percent"] == 87.5
    assert payload["available"] and session.status()["state"] == "idle" and calls == []
    assert call(base, "/telemetry", headers={"Origin": "http://example.com"})[0] == 403


def test_telemetry_requires_the_postgres_source(serve):
    base, _, _ = serve(blocking_until_stopped)
    code, payload = call(base, "/telemetry")
    assert code == 503 and not payload["available"]


def test_telemetry_errors_do_not_expose_database_details(serve):
    from router.datasource import DataUnavailableError

    class Source:
        def telemetry(self):
            raise DataUnavailableError("password=secret host=private")

    base, _, _ = serve(blocking_until_stopped, datasource=Source())
    code, payload = call(base, "/telemetry")
    assert code == 503 and not payload["available"]
    assert "secret" not in json.dumps(payload) and "private" not in json.dumps(payload)


def test_wrist_oxygen_controls_use_the_shared_source(serve):
    changes = []

    class Source:
        def set_demo_oxygen(self, value, relative):
            changes.append((value, relative))
            return 80

    base, session, calls = serve(blocking_until_stopped, datasource=Source())
    for command in ({"delta": -10}, {"delta": 10}, {"value": 100}):
        code, result = call(base, "/demo/oxygen", "POST", body=command)
        assert code == 200 and result["o2_percent"] == 80
    assert changes == [(-10, True), (10, True), (100, False)]
    assert calls == [] and session.status()["state"] == "idle"
    assert call(base, "/demo/oxygen", "POST", headers={"Origin": "http://example.com"}, body={"value": 100})[0] == 403


@pytest.mark.parametrize("command", [{}, {"value": 101}, {"value": -1}, {"delta": 101},
                                      {"delta": float("nan")}, {"value": True}, {"value": "80"},
                                      {"value": 80, "delta": 10}, [80]])
def test_wrist_oxygen_controls_reject_invalid_commands(serve, command):
    base, _, _ = serve(blocking_until_stopped)
    assert call(base, "/demo/oxygen", "POST", body=command)[0] == 400


def test_wrist_oxygen_controls_need_postgres(serve):
    base, _, _ = serve(blocking_until_stopped)
    code, result = call(base, "/demo/oxygen", "POST", body={"value": 80})
    assert code == 503 and "o2_percent" not in result


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


def test_committed_utterance_reports_processing_with_the_transcript_until_the_reply(serve):
    heard, release = threading.Event(), threading.Event()

    def script(on_event, on_turn, stop_event, on_transcript):
        on_event("listening...")
        on_transcript("check oxygen")
        heard.set()
        release.wait(5)
        on_turn({"transcript": "check oxygen", "turn": {"spoken": "Oxygen is 84 percent.", "route": "tool"}})
        stop_event.wait(5)
        return []

    calls = []
    session = voice_server.VoiceSession(lambda: "ROUTER", converse_fn=lambda **kw: calls.append(kw) or script(
        kw["on_event"], kw["on_turn"], kw["stop_event"], kw["on_transcript"]))
    server = voice_server.make_server(session, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        call(base, "/voice/start", "POST")
        assert heard.wait(3)
        status = call(base, "/voice/status")[1]
        assert (status["state"], status["last_transcript"], status["last_reply"]) == ("processing", "check oxygen", None)
        release.set()
        assert wait_for(lambda: call(base, "/voice/status")[1]["state"] == "speaking")
        assert call(base, "/voice/status")[1]["last_reply"] == "Oxygen is 84 percent."
    finally:
        session.stop()
        server.shutdown()
        server.server_close()


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


def test_converse_exception_becomes_error_state_without_the_secret(serve, monkeypatch, capsys):
    monkeypatch.setenv(stt.KEY_VAR, "sk-supersecret-123")

    def script(on_event, on_turn, stop_event):
        raise stt.SttError("server error: bad key sk-supersecret-123")

    base, session, _ = serve(script)
    call(base, "/voice/start", "POST")
    assert wait_for(lambda: call(base, "/voice/status")[1]["state"] == "error")
    status = call(base, "/voice/status")[1]
    assert "SttError" in status["error"] and "sk-supersecret-123" not in json.dumps(status)
    session.join(3)
    console = capsys.readouterr().err
    assert "voice session failed: SttError:" in console
    assert "sk-supersecret-123" not in console
    assert call(base, "/voice/start", "POST")[1]["started"] is True  # an error session can be restarted


def test_playback_failure_is_reported_without_ending_the_session(serve, monkeypatch, capsys):
    monkeypatch.setenv(stt.KEY_VAR, "secret-playback-key")
    emitted = threading.Event()

    def script(on_event, on_turn, stop_event):
        on_event("listening...")
        on_event("(speech failed: RuntimeError: no audio device secret-playback-key)")
        on_event("(silence - ending conversation)")
        emitted.set()
        stop_event.wait(5)
        return []

    base, _, _ = serve(script)
    call(base, "/voice/start", "POST")
    assert wait_for(lambda: call(base, "/voice/status")[1]["error"] is not None)
    status = call(base, "/voice/status")[1]
    assert status["state"] == "listening" and "no audio device" in status["error"]
    assert emitted.wait(3)
    console = capsys.readouterr().err
    assert "(speech failed: RuntimeError: no audio device" in console
    assert "(silence - ending conversation)" in console
    assert "listening..." not in console
    assert "secret-playback-key" not in console + status["error"]


def test_router_is_prewarmed_at_construction_and_ready_flag_is_reported(serve):
    gate = threading.Event()
    loads = []

    def slow_loader():
        loads.append(1)
        gate.wait(5)
        return "ROUTER"

    base, _, _ = serve(blocking_until_stopped, router_loader=slow_loader)
    assert wait_for(lambda: loads)  # loading began before any /voice/start
    status = call(base, "/voice/status")[1]
    assert status["router_ready"] is False and status["router_error"] is None
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


def test_status_alert_tracks_turn_and_clears_on_next_non_alert_turn(serve):
    advance, release = threading.Event(), threading.Event()

    def script(on_event, on_turn, stop_event):
        on_turn({'transcript': 'directions', 'turn': {'spoken': 'Alert. Directions.', 'alert_line': 'Alert.', 'route': 'tool'}})
        assert advance.wait(5)
        on_turn({'transcript': 'hello', 'turn': {'spoken': 'Hello.', 'route': 'fast_reply'}})
        release.wait(5)

    base, _, _ = serve(script)
    call(base, '/voice/start', 'POST')
    assert wait_for(lambda: call(base, '/voice/status')[1]['alert'] == 'Alert.')
    assert call(base, '/voice/status')[1]['last_reply'] == 'Alert. Directions.'
    advance.set()
    assert wait_for(lambda: call(base, '/voice/status')[1]['last_reply'] == 'Hello.')
    assert call(base, '/voice/status')[1]['alert'] is None
    release.set()

def test_datasource_sim_is_the_default_and_builds_nothing():
    assert voice_server._make_datasource("sim") is None


def test_datasource_postgres_reports_a_db_that_is_not_ready_without_failing(monkeypatch):
    def refused(url):
        raise ConnectionError(f"could not connect to {url}")

    said = []
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    ds = voice_server._make_datasource("postgres", "host=localhost dbname=nova", connect=refused, say=said.append)
    assert ds is not None and ds.database_url == "host=localhost dbname=nova"
    assert len(said) == 1 and "not ready" in said[0] and "simulator.py" in said[0]


def test_default_handler_reads_a_given_datasource_instead_of_simstate(monkeypatch):
    from router import tools

    class FakeDatasource:
        def snapshot(self):
            return {"o2_pct": 50.0, "minutes_remaining": 100, "bpm": 80, "radiation_usv_h": 1.0, "temp_c": 20.0,
                    "suit_pressure_kpa": 29.6}

        def place(self, name):
            return None

        def queue_message(self, *args):
            return False

    seen = {}
    real_registry = tools.default_registry
    monkeypatch.setattr(tools, "default_registry", lambda sim=None, datasource=None: seen.update(sim=sim, ds=datasource)
                        or real_registry(sim=sim, datasource=datasource))
    ds = FakeDatasource()
    handler = stt._default_handler(router=object(), speak=False, llm_options={"no_llm": True}, datasource=ds)
    assert seen["ds"] is ds and seen["sim"] is None
    assert handler.alert_source == ds.snapshot


def _main_options(tmp_path, monkeypatch, *flags):
    """Capture the server conversation options without starting audio or HTTP."""
    seen = {}
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)

    class Session:
        def __init__(self, loader, converse_options):
            seen.update(converse_options)

        def stop(self):
            pass

    monkeypatch.setattr(voice_server, "VoiceSession", Session)

    class Server:
        server_address = ("127.0.0.1", 8766)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    monkeypatch.setattr(voice_server, "make_server", lambda *a, **kw: Server())
    assert voice_server.main(["--model", str(tmp_path), *flags]) == 0
    return seen


def test_server_uses_the_live_llm_by_default_and_offline_with_no_llm(tmp_path, monkeypatch):
    assert _main_options(tmp_path, monkeypatch)["llm_options"] == {"no_llm": False}
    assert _main_options(tmp_path, monkeypatch, "--no-llm")["llm_options"] == {"no_llm": True}
    for flag in ("--runtime", "--base-url", "--llm-model", "--deadline-ms"):
        with pytest.raises(SystemExit):
            voice_server.main([flag, "x"])


def _fake_gemini(*replies):
    """Use fake HTTP replies to test Gemini failover."""
    import io
    import urllib.error
    from router import llm_generator

    class Opener:
        def __init__(self, replies):
            self.replies = list(replies)

        def open(self, request, timeout=None):
            reply = self.replies.pop(0) if self.replies else 503
            if isinstance(reply, int):
                raise urllib.error.HTTPError("https://x", reply, "err", {}, io.BytesIO(b"body"))
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": reply}}]}).encode())

    config = llm_generator.load_live_llm_config(env={})
    opener = Opener(replies)
    return llm_generator.FailoverGenerator([
        llm_generator.HTTPChatGenerator(config, api_key="fake-key", opener=opener),
        llm_generator.HTTPChatGenerator({**config, "model": config["fallback_model"]}, api_key="fake-key", opener=opener),
    ])


def _server_handler(monkeypatch, generator, label="NO_TOOL", confidence=0.99):
    """Build a server conversation with a fake generator."""
    from router import llm_generator
    from test_turn import FakeRouter

    monkeypatch.setattr(llm_generator, "build_generator", lambda say, no_llm=False, **kw: (generator, "fake gemini"))
    return stt._default_handler(router=FakeRouter(label, confidence), speak=False, llm_options=voice_server.llm_options())


def test_server_conversation_wires_gemini_and_its_second_routing_layer(monkeypatch):
    handler = _server_handler(monkeypatch, _fake_gemini('{"tool": "call_o2"}'))
    assert handler.generator_label == "fake gemini" and handler.pipeline.tool_classifier_fn is not None
    turn = handler.handle("how much air do I have")
    assert turn["route"] == "tool" and turn["pipeline"]["gate"] == "llm_second_layer"


def test_server_conversation_falls_back_when_gemini_fails(monkeypatch):
    handler = _server_handler(monkeypatch, _fake_gemini(503, 503, 503, 503, 503, 503))
    turn = handler.handle("tell me something fun")
    assert turn["route"] == "llm" and turn["generation"]["status"] == "error"
    assert turn["spoken"] == "I could not complete that request."


def test_server_conversation_is_offline_when_gemini_is_unreachable(monkeypatch):
    from router import llm_generator
    from router.generation import FillerGenerator

    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(llm_generator, "check_server", lambda *a, **k: (False, "server down"))
    generator, label = llm_generator.build_generator(lambda line: None, **voice_server.llm_options())
    assert isinstance(generator, FillerGenerator) and "offline" in label


def test_no_llm_server_conversation_has_no_gemini_and_no_second_layer(monkeypatch):
    from router import llm_generator
    from router.generation import FillerGenerator
    from test_turn import FakeRouter

    monkeypatch.setattr(llm_generator, "check_server", lambda *a, **k: pytest.fail("offline mode must not contact Gemini"))
    handler = stt._default_handler(router=FakeRouter(), speak=False, llm_options=voice_server.llm_options(no_llm=True))
    assert isinstance(handler.generator, FillerGenerator) and handler.pipeline.tool_classifier_fn is None


def test_router_load_failure_is_visible_before_start_and_redacted(serve, monkeypatch, capsys):
    monkeypatch.setenv(stt.KEY_VAR, "secret-router-key")

    def bad_loader():
        raise RuntimeError("bad checkpoint secret-router-key")

    base, session, calls = serve(blocking_until_stopped, router_loader=bad_loader)
    assert wait_for(lambda: call(base, "/voice/status")[1]["router_error"] is not None)
    status = call(base, "/voice/status")[1]
    assert status["router_ready"] is False
    assert "RuntimeError: bad checkpoint" in status["router_error"]
    assert calls == []
    with pytest.raises(RuntimeError):
        session._router.get(timeout=3)
    console = capsys.readouterr().err
    assert "router load failed: RuntimeError: bad checkpoint" in console
    assert "secret-router-key" not in console + json.dumps(status)


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_main_rejects_model_that_is_not_a_directory(tmp_path, monkeypatch, capsys, kind):
    model = tmp_path / kind
    if kind == "file":
        model.write_text("not a checkpoint directory")
    monkeypatch.setattr(voice_server, "VoiceSession", lambda *a, **kw: pytest.fail("must not start loading"))
    assert voice_server.main(["--model", str(model)]) == 2
    assert f"model path is not an existing directory: {model}" in capsys.readouterr().err


@pytest.mark.parametrize("key", ["", "secret-startup-key"])
@pytest.mark.parametrize("sounddevice_available", [True, False])
def test_main_prints_startup_diagnostics(tmp_path, monkeypatch, capsys, key, sounddevice_available):
    env = tmp_path / ".env"
    env.write_text(f"ELEVENLABS_API_KEY={key}\n")
    monkeypatch.setattr(stt, "ENV_PATH", env)
    monkeypatch.delenv(stt.KEY_VAR, raising=False)
    monkeypatch.setattr(stt, "_load_router", lambda *a: "ROUTER")
    monkeypatch.setitem(sys.modules, "sounddevice", object() if sounddevice_available else None)

    class Server:
        server_address = ("127.0.0.1", 8766)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    monkeypatch.setattr(voice_server, "make_server", lambda *a, **kw: Server())
    assert voice_server.main(["--model", str(tmp_path)]) == 0
    console = capsys.readouterr().err
    assert f"router model: {tmp_path.resolve()}" in console
    assert f".env path: {env.resolve()}" in console
    assert f"ELEVENLABS_API_KEY: {'found' if key else 'missing'}" in console
    assert f"sounddevice import: {'OK' if sounddevice_available else 'failed'}" in console
    assert "voice server on http://127.0.0.1:8766" in console
    assert "secret-startup-key" not in console


def test_main_flag_help_clarifies_device_and_timeout(capsys):
    with pytest.raises(SystemExit) as exc:
        voice_server.main(["--help"])
    assert exc.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "torch device for the router (not the microphone)" in help_text
    assert "seconds of silence that end a conversation" in help_text


def test_status_contains_router_error_alert_and_display_together(serve):
    base, _, _ = serve(blocking_until_stopped)
    code, status = call(base, "/voice/status")
    assert code == 200
    assert {"router_error", "alert", "display", "display_seq"} <= status.keys()
    assert status["router_error"] is status["alert"] is status["display"] is None
    assert status["display_seq"] == 0


def _shuttle_turn(text, label, **sim):
    """Run an offline turn with a fixed router prediction."""
    from router.tools import SimState, default_registry
    from router.turn import TurnHandler
    from test_turn import FakeRouter

    sim_state = SimState(**sim)
    return TurnHandler(FakeRouter(label, 0.99), sim_state.snapshot, default_registry(sim_state)).handle(text)


@pytest.mark.parametrize("text,label,expected", [
    ("take me to the shuttle", "call_navigation", ("start", "shuttle_001")),
    ("navigate to the viking lander", "call_navigation", ("start", "shuttle_001")),
    ("show me the shuttle on the map", "call_map", ("find", "shuttle_001")),
    ("how far is the shuttle", "call_navigation", None),
    ("take me to the habitat", "call_navigation", None),
])
def test_shuttle_tool_calls_become_unity_navigation_commands(text, label, expected):
    turn = _shuttle_turn(text, label)
    assert turn["tool_call"]["name"] == {"call_navigation": "navigation", "call_map": "map"}[label]
    assert voice_server.unity_navigation(turn) == expected


def test_no_shuttle_route_when_nova_says_there_is_not_enough_oxygen():
    turn = _shuttle_turn("take me to the shuttle", "call_navigation", o2_pct=17.0, minutes_remaining=2)
    assert turn["tool_result"]["data"]["feasibility"]["outcome"] in voice_server.NO_GO_OUTCOMES
    assert voice_server.unity_navigation(turn) is None


def test_status_exposes_the_navigation_command_and_counts_new_ones(serve):
    step = threading.Event()
    start = {"spoken": "Shuttle is 3 minutes away.", "route": "tool", "blocked_text": None,
             "tool_call": {"name": "navigation", "args": {"place": "shuttle"}},
             "tool_result": {"tool": "navigation", "ok": True, "data": {"place": "shuttle", "response_mode": "guidance",
                                                                       "feasibility": {"outcome": "comfortable"}}}}

    def script(on_event, on_turn, stop_event):
        on_turn({"transcript": "take me to the shuttle", "turn": start})
        step.wait(5)
        on_turn({"transcript": "thanks", "turn": {"spoken": "Happy to help!", "route": "session_end"}})
        stop_event.wait(5)
        return []

    base, _, _ = serve(script)
    status = call(base, "/voice/status")[1]
    assert status["navigation_action"] is None and status["navigation_destination"] is None and status["navigation_seq"] == 0
    call(base, "/voice/start", "POST")
    assert wait_for(lambda: call(base, "/voice/status")[1]["navigation_seq"] == 1)
    status = call(base, "/voice/status")[1]
    assert (status["navigation_action"], status["navigation_destination"]) == ("start", "shuttle_001")
    step.set()
    assert wait_for(lambda: call(base, "/voice/status")[1]["last_reply"] == "Happy to help!")
    assert call(base, "/voice/status")[1]["navigation_seq"] == 1

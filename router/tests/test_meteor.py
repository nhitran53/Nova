"""Meteor datasource, turn, tools, SpeechQueue and voice HTTP contract with offline fakes."""
import io
import json
import threading

import pytest

from router import stt
from router.datasource import DataUnavailableError, FixtureDatasource, PostgresDatasource
from router.hazard import nearest_safe_point
from router.stt import SpeechQueue
from router.tools import OneCallPerTurnError, default_registry
from router.turn import TurnHandler, evaluate_alerts, load_policy
from router.voice_server import VoiceSession, make_handler


class Reference:
    def radiation_baseline_usv_h(self):
        return {"tissue_usv_h_median": 0.3}


class Router:
    def __init__(self, label="call_o2"):
        self.calls = []
        self.label = label

    def predict(self, text):
        self.calls.append(text)
        return {"label": self.label, "confidence": 0.99, "topk": []}


@pytest.fixture
def ds():
    source = FixtureDatasource(reference=Reference())
    source._hazard_clock = lambda: 100.0
    return source


def position(ds, dx=0, dz=0):
    ds.set_position(ds.hazard_policy["center_x_m"] + dx, ds.hazard_policy["center_z_m"] + dz)


@pytest.mark.parametrize("dx,dz,inside,edge", [(0, 0, True, 300), (180, 240, True, 0),
                                                   (301, 0, False, 1), (0, -400, False, 100)])
def test_datasource_meteor_inside_and_edge_distance(ds, dx, dz, inside, edge):
    position(ds, dx, dz)
    assert ds.set_demo_meteor(True)
    hazard = ds.hazard_status()
    assert hazard["inside"] is inside
    assert hazard["edge_distance_m"] == edge
    assert ds.snapshot()["meteor_inside"] is inside
    alerts = evaluate_alerts(ds.snapshot(), load_policy())
    assert any(a["field"] == "meteor_inside" and a["severity"] == "critical" for a in alerts) is inside


def test_datasource_meteor_eta_retrigger_and_clear(ds):
    position(ds)
    ds.set_demo_meteor(True)
    assert ds.hazard_status()["eta_s"] == 30 * 3600
    ds._hazard_clock = lambda: 111.0
    assert not ds.set_demo_meteor(True)
    assert ds.hazard_status()["eta_s"] == 30 * 3600 - 11
    ds.set_demo_meteor(False)
    assert not ds.hazard_status()["active"]
    assert "meteor_inside" not in ds.snapshot()
    assert ds.hazard_tool_data() == {}
    assert ds.set_demo_meteor(True)
    assert ds.hazard_status()["eta_s"] == 30 * 3600


def test_turn_meteor_defers_llm_but_preserves_tools_then_resumes_outside_and_after_clear(ds):
    position(ds)
    ds.set_demo_meteor(True)
    router = Router("NO_TOOL")
    handler = TurnHandler(router, ds.snapshot, default_registry(datasource=ds))
    out = handler.handle("tell me something")
    assert out["alert_line"] == "You are currently in the blast radius. Move away now."
    assert out["tool_calls"] == 0 and not out["llm_called"]
    assert out["route"] == "alert_active_deferred"
    router.label = "call_o2"
    assert handler.handle("oxygen level")["tool_calls"] == 1
    router.label = "call_map"
    mapped = handler.handle("show homebase on the map")
    assert mapped["tool_calls"] == 1
    assert mapped["tool_result"]["data"]["inside"] is True
    assert mapped["alert_line"]
    router.label = "call_o2"
    position(ds, 400)
    out = handler.handle("oxygen level")
    assert out["tool_calls"] == 1 and not out.get("alert_line")
    position(ds)
    ds.set_demo_meteor(False)
    assert handler.handle("oxygen level")["tool_calls"] == 1


@pytest.mark.parametrize("tool", ["navigation", "map"])
def test_tools_meteor_numbers_and_deterministic_safe_point(ds, tool):
    position(ds)
    ds.set_demo_meteor(True)
    registry = default_registry(datasource=ds)
    registry.begin_turn()
    data = registry.call(tool, {"place": "homebase"})["data"]
    assert {k: data[k] for k in ("radius_m", "edge_distance_m", "inside", "eta_hours")} == {
        "radius_m": 300, "edge_distance_m": 300, "inside": True, "eta_hours": 30}
    assert data["safe_distance_m"] == 325
    assert data["safe_direction"] == {"x": 1, "z": 0}
    assert data["safe_place"] == {"x_m": ds.hazard_policy["center_x_m"] + 325,
                                  "z_m": ds.hazard_policy["center_z_m"]}
    with pytest.raises(OneCallPerTurnError):
        registry.call(tool, {"place": "homebase"})
    position(ds, 400)
    assert nearest_safe_point(ds.hazard_position(), ds.hazard_policy)["safe_distance_m"] == 0
    registry.begin_turn()
    assert registry.call("safety", {})["data"]["status_word"] == "go"


def test_postgres_meteor_missing_position_reports_active_unknown_and_fails_closed():
    ds = PostgresDatasource(connect=lambda _: pytest.fail("no database access expected"))
    ds._query = lambda *args: None
    ds.set_demo_meteor(True)
    hazard = ds.hazard_status()
    assert hazard["active"] and hazard["inside"] is None
    with pytest.raises(DataUnavailableError):
        ds.hazard_snapshot()


def post(session, ds, body):
    """Invoke the actual HTTP handler without binding a socket."""
    session._hazard_datasource = ds
    handler_class = make_handler(session, ds)
    handler = handler_class.__new__(handler_class)
    handler.path = "/demo/meteor"
    raw = json.dumps(body).encode()
    handler.headers = {"Content-Length": str(len(raw))}
    handler.rfile = io.BytesIO(raw)
    responses = []
    handler._send = lambda code, payload: responses.append((code, payload))
    handler.do_POST()
    return responses[0]


class Queue:
    def __init__(self):
        self.lines = []
        self.clears = 0
        self.closed = False

    def clear_pending(self):
        self.clears += 1

    def enqueue(self, *args, **kwargs):
        assert not self.closed
        self.lines.append((args, kwargs))

    def close(self):
        self.closed = True


def test_voice_server_meteor_trigger_status_seq_clear_and_validation(ds):
    queue = Queue()
    session = VoiceSession(lambda: Router(), converse_options={"speech_queue": queue})
    assert session.status()["hazard_seq"] == 0
    position(ds)
    code, status = post(session, ds, {"active": True})
    assert code == 200 and status["state"] == "idle"
    assert set(status["hazard"]) == {"active", "kind", "center_x_m", "center_z_m", "radius_m", "eta_s", "inside", "edge_distance_m"}
    assert status["hazard_seq"] == 1
    assert queue.lines == [((ds.hazard_policy["broadcast_line"], "alert"), {"priority": True})]
    assert post(session, ds, {"active": True})[1]["hazard_seq"] == 1
    assert len(queue.lines) == 1
    ds._hazard_clock = lambda: 101.0
    assert session.status()["hazard_seq"] == 2
    assert session.status()["hazard_seq"] == 2
    position(ds, 400)
    assert session.status()["hazard_seq"] == 3
    assert session.status()["hazard"]["inside"] is False
    assert post(session, ds, {"active": False})[1]["hazard_seq"] == 4
    assert session.status()["hazard"]["active"] is False
    assert post(session, ds, {"active": False})[1]["hazard_seq"] == 4
    for body in ({"active": 1}, {}, {"active": True, "radius_m": 4}, []):
        assert post(session, ds, body)[0] == 400
    assert post(session, None, {"active": True})[0] == 503


def test_stt_speech_queue_meteor_broadcast_and_critical_preempt_pending_normal(ds):
    entered, release = threading.Event(), threading.Event()
    played = []

    def speak(text, route, block=True):
        played.append((text, route))
        if text == "in progress":
            entered.set()
            assert release.wait(3)

    speech = SpeechQueue(speak)
    try:
        speech.enqueue("in progress")
        assert entered.wait(3)
        speech.enqueue("discard this normal reply")
        session = VoiceSession(lambda: Router(), converse_options={"speech_queue": speech})
        position(ds)
        assert post(session, ds, {"active": True})[0] == 200

        def deliver(text, route):
            if route == "alert":
                speech.clear_pending()
            speech.enqueue(text, route, priority=route == "alert")

        handler = TurnHandler(Router("NO_TOOL"), ds.snapshot, default_registry(datasource=ds), tts_fn=deliver)
        out = handler.handle("tell me something")
        release.set()
        speech.drain()
        assert [route for _, route in played] == [None, "alert", "alert", "alert_active_deferred"]
        assert played[1][0] == ds.hazard_policy["broadcast_line"]
        assert played[2][0] == out["alert_line"]
    finally:
        release.set()
        speech.close()


def test_voice_server_lazy_queue_survives_conversation_close_for_idle_broadcast(ds, monkeypatch):
    queue = Queue()
    created = []

    def make_queue():
        created.append(queue)
        return queue

    def converse(**kwargs):
        kwargs["speech_queue"].close()  # stt.converse does this after every conversation

    monkeypatch.setattr(stt, "default_speech_queue", make_queue)
    monkeypatch.setattr(stt, "converse", converse)
    session = VoiceSession(lambda: Router(), converse_fn=converse)
    assert post(session, ds, {"active": True})[0] == 200
    assert len(created) == 1
    assert session.start()[0]
    session.join(3)
    assert not queue.closed
    post(session, ds, {"active": False})
    assert post(session, ds, {"active": True})[0] == 200
    assert len(created) == 1 and len(queue.lines) == 2

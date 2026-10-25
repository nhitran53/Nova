import threading
import time

import pytest

from router.location import describe_location, make_location_tool, match_location, match_return_to_start, validate_snapshot
from router.tools import SimState, default_registry
from router.turn import TurnHandler
from router.voice_server import VoiceSession
from test_voice_server import serve, call, blocking_until_stopped
from router.voice_server import unity_navigation
from test_turn import FakeRouter


def snapshot(seq=1):
    vector = {"x": 0, "y": 0, "z": 0}
    return {"schema_version": 1, "seq": seq, "available": True, "coordinates_verified": True,
            "environment": "Mars", "region": "Jezero Crater", "latitude": 18.444, "longitude": 77.44,
            "world_position": vector, "head_position": vector, "head_forward": vector, "world_north": vector,
            "nearby_landmarks": [{"id": "shuttle_001", "name": "Shuttle", "position": vector,
                                  "distance_m": 140.2, "bearing_deg": 45, "direction": "northeast",
                                  "relative_direction": "to your right"}]}


@pytest.mark.parametrize("question", ["NOVA, where am I?", "What's my current location on Mars?", "What are my coordinates?",
    "How far am I from the shuttle?", "Where is the rover relative to me?", "What region of Mars am I exploring?"])
def test_location_questions_use_live_tool_without_generation(question):
    class Bridge:
        calls = 0
        def request_spatial(self):
            self.calls += 1
            return validate_snapshot(snapshot())
    bridge, sim = Bridge(), SimState()
    registry = default_registry(sim, location=make_location_tool(bridge))
    handler = TurnHandler(FakeRouter("NO_TOOL"), sim.snapshot, registry)
    result = handler.handle(question)
    assert result["tool_call"]["name"] == "get_current_location"
    assert bridge.calls == 1 and result["blocked_text"] is None
    assert "166" not in result["spoken"]  # simulator shuttle distance


@pytest.mark.parametrize("question", ["take me to the shuttle", "show me the shuttle on the map",
                                      "why is Jezero Crater scientifically important", "where am i and take me home"])
def test_other_intents_are_preserved(question):
    assert not match_location(question)


def test_missing_georeference_never_returns_zero_coordinates():
    raw = snapshot()
    raw.update(coordinates_verified=False, latitude=0, longitude=0, region="unverified", environment="Mars")
    data = validate_snapshot(raw)
    assert data["latitude"] is None and data["longitude"] is None and data["region"] is None
    assert "unavailable" in describe_location(data, "my coordinates")
    assert "140" in describe_location(data, "how far is the shuttle")


@pytest.mark.parametrize("field,value", [("latitude", float("nan")), ("longitude", 360), ("head_position", {}),
                                        ("nearby_landmarks", [{}]), ("seq", True)])
def test_malformed_snapshots_are_rejected(field, value):
    raw = snapshot()
    raw[field] = value
    with pytest.raises(ValueError):
        validate_snapshot(raw)


def test_handshake_accepts_only_requested_snapshot_and_refreshes():
    session = VoiceSession(lambda: object())
    try:
        assert session.submit_spatial(snapshot())[0] == 409
        results = []
        for distance in (140.2, 23.4):
            worker = threading.Thread(target=lambda: results.append(session.request_spatial(1)))
            worker.start()
            deadline = time.monotonic() + 1
            while not session.status()["spatial_pending"] and time.monotonic() < deadline:
                time.sleep(.001)
            seq = session.status()["spatial_seq"]
            assert session.submit_spatial(snapshot(seq - 1))[0] == 409
            raw = snapshot(seq)
            raw["nearby_landmarks"][0]["distance_m"] = distance
            assert session.submit_spatial(raw)[0] == 200
            assert session.submit_spatial(raw)[0] == 409
            worker.join(2)
            assert not worker.is_alive()
        assert [r["nearby_landmarks"][0]["distance_m"] for r in results] == [140.2, 23.4]
        with pytest.raises(TimeoutError):
            session.request_spatial(.001)
        assert not session.status()["spatial_pending"]
        assert session.submit_spatial(snapshot(session.status()["spatial_seq"]))[0] == 409
    finally:
        session.stop()


def test_unavailable_bridge_does_not_use_simulator():
    class Bridge:
        def request_spatial(self):
            raise TimeoutError()
    sim = SimState()
    result = TurnHandler(FakeRouter(), sim.snapshot, default_registry(sim, location=make_location_tool(Bridge()))).handle("where am I")
    assert result["tool_calls"] == 1 and "can't read" in result["spoken"]


def test_location_and_vision_can_share_registry():
    registry = default_registry(vision=lambda **kw: None, location=lambda **kw: None)
    assert len(registry.names()) == 9


@pytest.mark.parametrize("question", ["NOVA, return to start", "take me back to the starting position",
                                      "Hey Nova bring me back to the damaged wheel please"])
def test_return_to_start_requests_existing_rover_reset(question):
    sim = SimState()
    registry = default_registry(sim, return_to_start=True)
    result = TurnHandler(FakeRouter("NO_TOOL"), sim.snapshot, registry).handle(question)
    assert result["tool_call"] == {"name": "return_to_start", "args": {}}
    assert result["tool_calls"] == 1 and not result["llm_called"] and result["blocked_text"] is None
    assert unity_navigation(result) == ("return_to_start", "rover_inspection")


@pytest.mark.parametrize("question", ["don't return to start", "where is the starting position",
                                      "take me to the shuttle", "return to start and then take me to the shuttle"])
def test_return_to_start_does_not_match_other_requests(question):
    assert not match_return_to_start(question)


def test_return_to_start_is_available_during_critical_alerts():
    sim = SimState(o2_pct=10)
    registry = default_registry(sim, return_to_start=True)
    result = TurnHandler(FakeRouter(), sim.snapshot, registry).handle("return to start")
    assert result["alert_line"] and unity_navigation(result) == ("return_to_start", "rover_inspection")


def test_return_without_unity_does_not_send_movement_command():
    sim = SimState()
    result = TurnHandler(FakeRouter(), sim.snapshot, default_registry(sim)).handle("return to start")
    assert result["tool_calls"] == 0 and unity_navigation(result) is None
    assert "Unity voice connection" in result["spoken"]


def test_all_unity_tools_fit_in_the_registry():
    registry = default_registry(vision=lambda **kw: None, location=lambda **kw: None, return_to_start=True)
    assert len(registry.names()) == 10


def test_spatial_http_bridge_and_cancellation(serve):
    base, session, _ = serve(blocking_until_stopped)
    replies = []
    worker = threading.Thread(target=lambda: replies.append(session.request_spatial(2)))
    worker.start()
    deadline = time.monotonic() + 1
    while not call(base, "/voice/status")[1]["spatial_pending"] and time.monotonic() < deadline:
        time.sleep(.001)
    assert call(base, "/spatial/context", "POST", body=snapshot())[0] == 200
    worker.join(2)
    assert replies[0]["nearby_landmarks"][0]["distance_m"] == 140.2
    assert call(base, "/spatial/context", "POST", body=snapshot())[0] == 409
    assert call(base, "/spatial/context", "POST", body={})[0] == 400
    assert call(base, "/spatial/context", "POST", headers={"Origin": "https://example.com"}, body=snapshot())[0] == 403
    cancelled = []
    def request():
        try:
            session.request_spatial(2)
        except ValueError:
            cancelled.append(True)
    worker = threading.Thread(target=request)
    worker.start()
    deadline = time.monotonic() + 1
    while not session.status()["spatial_pending"] and time.monotonic() < deadline:
        time.sleep(.001)
    session.stop()
    worker.join(1)
    assert cancelled and not worker.is_alive()


def _two_landmarks():
    raw = snapshot()
    raw["nearby_landmarks"].append({**raw["nearby_landmarks"][0], "id": "rover", "name": "Perseverance rover", "distance_m": 2.6,
                                    "direction": "east"})
    return validate_snapshot(raw)


@pytest.mark.parametrize("question,expected", [("Where are we?", "We're in Jezero Crater on Mars."),
                                               ("NOVA, where am I?", "You're in Jezero Crater on Mars."),
                                               ("What region of Mars am I exploring?", "You're in Jezero Crater on Mars.")])
def test_general_location_is_only_the_region(question, expected):
    assert match_location(question) and describe_location(_two_landmarks(), question) == expected


def test_nearby_lists_landmarks_closest_first():
    assert match_location("What's nearby?") and match_location("What is around us?")
    assert describe_location(_two_landmarks(), "What's nearby?") == (
        "Perseverance rover is about 3 meters east. Shuttle is about 140 meters northeast.")


def test_unverified_region_is_never_guessed():
    data = validate_snapshot({**snapshot(), "coordinates_verified": False})
    assert describe_location(data, "where are we") == "We're on Mars, but the exact region could not be verified."

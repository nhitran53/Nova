"""Vision: phrase routing, the Gemini image request, the Unity frame handshake. Fakes only; no network, no Unity."""
import base64
import json
import threading
import urllib.error
import urllib.request

import pytest

from router import stt, vision, voice_server
from router.tools import SimState, default_registry
from router.turn import TurnHandler, load_policy
from test_turn import FakeRouter

POLICY = load_policy()
CFG = POLICY["vision"]
JPEG = bytes((0xFF, 0xD8)) + b"fake jpeg body"
ROVER = {"object_id": "perseverance", "display_name": "Perseverance rover", "kind": "rover", "distance_m": 4.23,
         "damaged": True, "screen_x": 0.5, "screen_y": 0.4, "source": "rover", "hand": "right"}


class Frames:
    """vision frame source stand-in: returns `frame` (or raises it) and records HUD reports."""
    def __init__(self, frame=None):
        self.frame, self.modes, self.reports = frame, [], []

    def request(self, mode):
        self.modes.append(mode)
        if isinstance(self.frame, Exception):
            raise self.frame
        return self.frame

    def report(self, state, target):
        self.reports.append((state, target))


class Gemini:
    """generator stand-in: classify() returns the scripted replies in order and keeps the messages it got."""
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def classify(self, messages, max_tokens, deadline_ms, timeout_cap_s=None):
        self.calls.append({"messages": messages, "timeout_cap_s": timeout_cap_s})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def frame(target=None, hand="right"):
    return {"jpeg": JPEG, "width": 1024, "height": 576, "target": target, "pointing": {"hand": hand} if hand else {}}


def answer(text, target=None, uncertain=False):
    return json.dumps({"answer": text, "target": target, "uncertain": uncertain})


@pytest.mark.parametrize("text,mode", [
    ("NOVA, what am I looking at?", "looking"), ("what's in front of me", "looking"), ("Describe what I'm seeing.", "looking"),
    ("hey nova what am i pointing at right now", "pointing"), ("Can you tell me what I'm pointing at?", "pointing"),
    ("What is this I'm pointing at?", "pointing"), ("Tell me what I'm pointing to.", "pointing"),
    ("Can you identify this object?", "looking"), ("Nova, what is this?", "looking"), ("Describe the scene.", "looking"),
    ("how much oxygen do I have", None), ("take me to the shuttle", None), ("what is the weather", None),
    ("point me to the shuttle", None), ("show me the shuttle on the map", None),
    ("Is that wheel broken?", "looking"), ("Can you tell if anything is damaged?", "looking"), ("my suit is damaged, call base", None), ("it is broken", None),
    ("I think the wheel assembly's broken. Can you tell if it is?", "looking"),
])
def test_vision_phrases_and_modes(text, mode):
    assert vision.match_vision(text, POLICY) == mode


def test_second_layer_mode_comes_from_pointing_cues():
    assert vision.vision_mode("tell me about the thing I'm pointing toward", POLICY) == "pointing"
    assert vision.vision_mode("tell me about this thing", POLICY) == "looking"


def test_target_facts_keep_only_known_typed_keys():
    target = vision.clean_target({**ROVER, "secret": "x", "distance_m": "far", "damaged": "yes", "display_name": " " + "a" * 300})
    assert set(target) <= set(vision.TARGET_KEYS) and "secret" not in target
    assert "distance_m" not in target and "damaged" not in target and len(target["display_name"]) == vision.MAX_FACT_CHARS
    assert vision.clean_target({"distance_m": 3.0}) is None and vision.clean_target("rover") is None


def test_gemini_gets_the_real_image_the_escaped_question_and_only_the_target_name_and_distance():
    target = vision.clean_target({**ROVER, "description": "Viking lander used as the demo shuttle."})
    messages = vision.build_messages("what is <this>?", "pointing", frame(target), CFG["scene_context"])
    text, image = messages[1]["content"]
    assert image == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(JPEG).decode()}}
    assert "&lt;this&gt;" in text["text"] and "Perseverance rover" in text["text"] and "4.2 m away" in text["text"]
    for leak in ("damaged", "Viking", "simulat", "Jezero"):  # the answer must come from the image, not known state
        assert leak not in text["text"]
    assert "visual inspection assistant" in messages[0]["content"] and "Do not describe the landscape" in messages[0]["content"]


@pytest.mark.parametrize("raw,expected", [
    (answer("A rover.", "rover"), {"answer": "A rover.", "target": "rover", "uncertain": False}),
    ("```json\n" + answer("Rocks.", None, True) + "\n```", {"answer": "Rocks.", "target": None, "uncertain": True}),
    ("A rover.", None), (json.dumps({"answer": ""}), None), ("[]", None),
])
def test_answer_parsing_is_strict(raw, expected):
    assert vision.parse_answer(raw) == expected


def test_looking_request_sends_the_frame_and_reports_the_named_target():
    frames, gemini = Frames(frame(ROVER)), Gemini(answer("That's the Perseverance rover with a damaged wheel.", "rover"))
    out = vision.make_vision_tool(frames, gemini, CFG)("what am I looking at", "looking")
    assert out["ok"] and out["data"]["target_name"] == "Perseverance rover" and out["data"]["target"]["distance_m"] == 4.2
    assert gemini.calls[0]["timeout_cap_s"] == CFG["request_timeout_s"] and frames.reports == [("identified", "Perseverance rover")]


@pytest.mark.parametrize("hand,line", [("right", "no_target_line"), (None, "no_pointing_hand_line")])
def test_pointing_miss_is_uncertain_and_never_asks_gemini(hand, line):
    frames, gemini = Frames(frame(None, hand)), Gemini()
    out = vision.make_vision_tool(frames, gemini, CFG)("what am I pointing at", "pointing")
    assert out["ok"] and out["data"]["answer"] == CFG[line] and out["data"]["uncertain"] and gemini.calls == []


@pytest.mark.parametrize("frames,gemini,line", [
    (Frames(vision.VisionUnavailable("no headset image")), Gemini(), "unavailable_line"),
    (Frames(frame()), Gemini(RuntimeError("HTTP 503")), "error_line"),
    (Frames(frame()), Gemini("not json"), "error_line"),
    (Frames(frame()), object(), "unavailable_line"),  # offline FillerGenerator: no classify
])
def test_failures_are_fixed_lines(frames, gemini, line):
    out = vision.make_vision_tool(frames, gemini, CFG)("what am I looking at", "looking")
    assert not out["ok"] and out["data"]["line"] == CFG[line] and frames.reports[-1] == ("error", None)


def handler(tool=None, snapshot=None, label="NO_TOOL", **kw):
    sim = SimState()
    spoken = []
    h = TurnHandler(FakeRouter(label, 0.99), snapshot or sim.snapshot, default_registry(sim, **({"vision": tool} if tool else {})),
                    interim_fn=lambda line, route: spoken.append(line), **kw)
    return h, spoken


def test_vision_phrase_turn_speaks_the_wait_line_then_the_answer():
    frames, gemini = Frames(frame(ROVER)), Gemini(answer("That's the Perseverance rover, about 4 meters away.", "rover"))
    h, interim = handler(vision.make_vision_tool(frames, gemini, CFG))
    turn = h.handle("Nova, what am I pointing at?")
    assert interim == [CFG["interim_line"]] and frames.modes == ["pointing"]
    assert turn["route"] == "vision" and turn["tool_call"]["name"] == "vision" and turn["blocked_text"] is None
    assert turn["spoken"] == "That's the Perseverance rover, about 4 meters away." and turn["vision"]["target_name"] == "Perseverance rover"


def test_a_number_that_is_not_in_the_unity_facts_is_never_spoken():
    gemini = Gemini(answer("The rover's oxygen tank reads 40 percent.", "rover"))
    turn = handler(vision.make_vision_tool(Frames(frame(ROVER)), gemini, CFG))[0].handle("what am I looking at")
    assert turn["blocked_text"] is not None and turn["spoken"] == POLICY["templates"]["verifier_blocked"]


def test_without_a_vision_tool_the_request_is_honestly_unavailable():
    turn = handler()[0].handle("what am I looking at")
    assert turn["spoken"] == CFG["unavailable_line"] and turn["tool_calls"] == 0


def test_vision_waits_while_a_critical_alert_is_active():
    snapshot = {**SimState().snapshot(), "o2_pct": 10.0}
    frames = Frames(frame(ROVER))
    turn = handler(vision.make_vision_tool(frames, Gemini(), CFG), snapshot=lambda: snapshot)[0].handle("what am I looking at")
    assert turn["route"] == "alert_active_deferred" and frames.modes == []


def test_the_second_layer_can_pick_vision_when_the_phrase_list_misses(monkeypatch):
    from router import llm_generator as L
    gemini = Gemini('{"tool": "call_vision"}', answer("It's a lander with a dish antenna.", "lander"))
    gemini.generate = lambda req: pytest.fail("no reply generation for a vision turn")
    monkeypatch.setattr(L, "build_generator", lambda say, **kw: (gemini, "fake gemini"))
    frames = Frames(frame(None))
    h = stt._default_handler(router=FakeRouter("NO_TOOL", 0.99), speak=False, llm_options={}, vision_frames=frames)
    assert "vision" in h.registry.names() and "call_vision" in h.pipeline.config["tool_labels"]
    turn = h.handle("tell me about that machine over there")
    assert turn["pipeline"]["gate"] == "llm_second_layer" and frames.modes == ["looking"]
    assert turn["spoken"] == "It's a lander with a dish antenna." and turn["vision"]["target_name"] == "lander"


def test_without_frames_the_handler_has_no_vision_tool_and_bert_labels_are_unchanged(monkeypatch):
    from router import llm_generator as L
    monkeypatch.setattr(L, "build_generator", lambda say, **kw: (Gemini(), "fake gemini"))
    h = stt._default_handler(router=FakeRouter(), speak=False, llm_options={})
    assert "vision" not in h.registry.names() and "call_vision" not in h.pipeline.config["tool_labels"]


# --- voice_server frame handshake ---------------------------------------------------------------------------------

@pytest.fixture
def server():
    session = voice_server.VoiceSession(lambda: "ROUTER", converse_fn=lambda **kw: [])
    srv = voice_server.make_server(session, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", session
    srv.shutdown()
    srv.server_close()


def post(base, payload, raw=None):
    data = raw if raw is not None else json.dumps(payload).encode()
    request = urllib.request.Request(base + "/vision/frame", data=data, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def status(base):
    with urllib.request.urlopen(base + "/voice/status", timeout=5) as response:
        return json.loads(response.read())


def ask(session):
    """Run session.request("pointing") on a thread; -> (thread, box) with box["frame"] or box["error"] once it returns."""
    box = {}

    def run():
        try:
            box["frame"] = session.request("pointing")
        except vision.VisionUnavailable as exc:
            box["error"] = str(exc)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


def wait_capturing(base, seq):
    import time
    for _ in range(300):
        s = status(base)
        if s["vision_seq"] == seq and s["vision_state"] == "capturing":
            return s
        time.sleep(0.01)
    raise AssertionError("no capture request")


def test_unity_frame_reaches_the_waiting_request_decoded_with_its_target(server):
    base, session = server
    assert status(base)["vision_seq"] == 0 and status(base)["vision_state"] is None
    thread, box = ask(session)
    s = wait_capturing(base, 1)
    assert s["vision_mode"] == "pointing"
    code, reply = post(base, {"seq": 1, "image_jpeg_base64": base64.b64encode(JPEG).decode(), "width": 1024, "height": 576,
                              "has_target": True, "target": ROVER, "pointing": {"hand": "right"}})
    thread.join(5)
    assert code == 200 and reply == {"accepted": True, "bytes": len(JPEG)}
    assert box["frame"]["jpeg"] == JPEG and box["frame"]["target"] == ROVER and box["frame"]["pointing"] == {"hand": "right"}
    assert status(base)["vision_state"] == "analyzing"
    session.report("identified", "Perseverance rover")
    assert status(base)["vision_target"] == "Perseverance rover"


def test_bad_frames_are_refused_and_a_capture_error_ends_the_wait(server):
    base, session = server
    good = {"image_jpeg_base64": base64.b64encode(JPEG).decode(), "width": 1024, "height": 576}
    assert post(base, {"seq": 1, **good})[0] == 409  # nothing pending
    thread, box = ask(session)
    wait_capturing(base, 1)
    assert post(base, {"seq": 2, **good})[0] == 409  # wrong seq
    assert post(base, {"seq": 1, **good, "image_jpeg_base64": "%%%"})[0] == 400
    assert post(base, {"seq": 1, **good, "image_jpeg_base64": base64.b64encode(b"PNG...").decode()})[0] == 400
    assert post(base, {"seq": 1, **good, "width": 0})[0] == 400
    assert post(base, None, raw=b"{not json")[0] == 400
    assert post(base, {"seq": 1, "error": "GPU readback failed"})[0] == 200
    thread.join(5)
    assert box["error"] == "Unity capture failed: GPU readback failed"


def test_frames_over_the_body_limit_are_refused(server, monkeypatch):
    base, _ = server
    monkeypatch.setattr(voice_server, "MAX_FRAME_BODY_BYTES", 100)  # small, so the refused upload finishes sending
    assert post(base, None, raw=b"x" * 101)[0] == 400
    assert voice_server.MAX_BODY_BYTES < 100_000  # every other route keeps the small limit


def test_no_unity_answer_times_out(server, monkeypatch):
    from router import turn
    base, session = server
    monkeypatch.setattr(turn, "load_policy", lambda: {**POLICY, "vision": {**CFG, "frame_timeout_s": 0.2}})
    thread, box = ask(session)
    thread.join(5)
    assert box["error"] == "no headset image from Unity within 0.2 s"


def test_session_gives_converse_itself_as_the_frame_source():
    seen = {}
    session = voice_server.VoiceSession(lambda: "ROUTER", converse_fn=lambda **kw: seen.update(kw) or [])
    session.start()
    session.join(5)
    assert seen["vision_frames"] is session


def test_the_zoomed_focus_image_goes_to_gemini_as_image_two():
    focus = {"jpeg": JPEG + b"focus", "size": 768, "fov": 22.0}
    content = vision.build_messages("what am I looking at", "looking", {**frame(vision.clean_target(ROVER)), "focus": focus},
                                    CFG["scene_context"])[1]["content"]
    images = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
    assert len(images) == 2 and images[1].endswith(base64.b64encode(JPEG + b"focus").decode())
    assert "centre of image 2" in content[0]["text"] and "22 degree" in content[2]["text"]


def test_diagnostics_are_only_sent_when_switched_on_and_then_labelled():
    target = vision.clean_target(ROVER)
    off = vision.build_messages("q", "looking", frame(target), CFG["scene_context"])[1]["content"][0]["text"]
    on = vision.build_messages("q", "looking", frame(target), CFG["scene_context"], diagnostics=True)[1]["content"][0]["text"]
    assert "DIAGNOSTICS" not in off and CFG["share_diagnostics"] is False
    assert "VERIFIED DIAGNOSTICS (reported by the vehicle, not seen)" in on


def test_debug_capture_saves_the_exact_images_and_selection_only_when_enabled(tmp_path, monkeypatch):
    frames, gemini = Frames({**frame(ROVER), "focus": {"jpeg": JPEG + b"z", "size": 768, "fov": 20.0}}), Gemini(answer("A rover wheel."))
    vision.make_vision_tool(frames, gemini, CFG)("what am I looking at", "looking")
    assert list(tmp_path.iterdir()) == []  # off by default
    monkeypatch.setenv("NOVA_VISION_DEBUG_DIR", str(tmp_path))
    vision.make_vision_tool(frames, Gemini(answer("A rover wheel.")), CFG)("what am I looking at", "looking")
    files = sorted(f.name.split("_", 3)[-1] for f in tmp_path.iterdir())
    assert files == ["focus.jpg", "full.jpg", "meta.json"]
    meta = json.loads(next(tmp_path.glob("*_meta.json")).read_text())
    assert meta["target_id"] == "perseverance" and meta["images_sent"] == ["full", "focus"] and meta["focus_size"] == 768
    assert next(tmp_path.glob("*_full.jpg")).read_bytes() == JPEG


def test_server_accepts_a_focus_image_and_refuses_a_bad_one(server):
    base, session = server
    good = {"image_jpeg_base64": base64.b64encode(JPEG).decode(), "width": 1024, "height": 576}
    thread, box = ask(session)
    wait_capturing(base, 1)
    assert post(base, {"seq": 1, **good, "focus_jpeg_base64": base64.b64encode(b"PNG").decode(), "focus_size": 768, "focus_fov": 20})[0] == 400
    assert post(base, {"seq": 1, **good, "focus_jpeg_base64": base64.b64encode(JPEG).decode(), "focus_size": 768, "focus_fov": 20})[0] == 200
    thread.join(5)
    assert box["frame"]["focus"] == {"jpeg": JPEG, "size": 768, "fov": 20.0}


WHEEL = {**ROVER, "part": "front_right_wheel"}


def test_looking_at_the_damaged_wheel_says_it_appears_broken_and_points_to_the_manual():
    from router.demo_scene import DEFAULT_DOCS_DIR
    from router.retrieval import make_retriever
    frames, gemini = Frames(frame(WHEEL)), Gemini()
    h, _ = handler(vision.make_vision_tool(frames, gemini, CFG, make_retriever(str(DEFAULT_DOCS_DIR))))
    turn = h.handle("What am I looking at?")
    assert turn["spoken"] == ("That's the rover's front right wheel. It appears to be broken. "
                              "The rover manual covers this in Drive Module Assembly. "
                              "The Perseverance assembly instructions for the wheel are on your HUD.")
    assert turn["display"] == {"doc_id": "perseverance_assembly", "section_id": "wheel_assembly", "title": "Wheel Assembly",
                               "asset": "wheel_assembly", "source": "Mars 2020 Perseverance Rover assembly instructions"}
    assert turn["blocked_text"] is None
    assert gemini.calls == [] and frames.reports[-1] == ("identified", "Perseverance rover")


@pytest.mark.parametrize("target", [{**WHEEL, "damaged": False}, ROVER, {**WHEEL, "object_id": "shuttle_001"}])
def test_the_fixed_line_needs_the_damaged_rover_and_the_ray_on_that_wheel(target):
    gemini = Gemini(answer("You're looking at the rover's wheel."))
    turn = handler(vision.make_vision_tool(Frames(frame(target)), gemini, CFG))[0].handle("What am I looking at?")
    assert turn["spoken"] == "You're looking at the rover's wheel." and len(gemini.calls) == 1


def test_without_the_manual_the_fixed_line_still_works():
    turn = handler(vision.make_vision_tool(Frames(frame(WHEEL)), Gemini(), CFG))[0].handle("Is that wheel broken?")
    assert turn["spoken"] == ("That's the rover's front right wheel. It appears to be broken. "
                              "The Perseverance assembly instructions for the wheel are on your HUD.")
    assert turn["display"]["asset"] == "wheel_assembly"

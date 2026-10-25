"""Demo "what is this" scene + simulated print confirmation (fake router, fake registry spy, real Sawppy corpus, no models)."""
import copy

import pytest

from router import llm_generator, stt
from router.demo_scene import DEFAULT_DOCS_DIR, match_demo_scene, validate_demo_scene
from router.retrieval import make_retriever
from router.tests.test_turn import CRITICAL, FakeRouter, SpyRegistry
from router.tools import DEMO_ROVER_FAULTS, SimState, default_registry
from router.turn import TurnHandler, load_policy
from router.verifier import verify_numbers

SCENE = "That's your rover. Its diagnostics flag damage on the left front tire. Want me to send a replacement wheel to the 3D printer?"
QUEUED, DECLINED, NO_FAULT = "Replacement wheel queued.", "Okay, I won't send it.", "That's your rover. No faults reported."
RETRIEVER = make_retriever(str(DEFAULT_DOCS_DIR))


def policy(enabled=True):
    p = load_policy()
    p["demo_scene"]["enabled"] = enabled
    return p


class Rig:
    def __init__(self, enabled=True, faults=DEMO_ROVER_FAULTS, label="call_llm", snapshot=None, retriever=RETRIEVER, verifier=verify_numbers):
        self.sim = SimState(rover_faults=faults)
        self.registry = SpyRegistry(self.sim)
        self.router = FakeRouter(label, 0.99)
        self.snapshot, self.spoken, self.interim, self.llm = snapshot, [], [], []
        self.handler = TurnHandler(self.router, lambda: self.snapshot or self.sim.snapshot(), self.registry, policy=policy(enabled),
                                   retriever=retriever, verifier=verifier, llm_fn=None, generator=self,
                                   interim_fn=lambda line, route: self.interim.append(line),
                                   tts_fn=lambda line, route: self.spoken.append((line, route)))

    def generate(self, request):  # any call here means the LLM path ran
        self.llm.append(request)
        raise AssertionError("LLM generator must not run")

    def say(self, text):
        return self.handler.handle(text)


def test_scene_phrase_speaks_the_fixed_line_without_router_tool_llm_or_interim_and_arms_the_offer():
    rig = Rig()
    out = rig.say("hey nova, what is this")
    assert out["route"] == "scene" and out["spoken"] == SCENE and out["demo_scene"] is True and out["verifier"]["ok"]
    assert out["tool_calls"] == 0 and not out["llm_called"] and out["pipeline"] is None and "display" not in out
    assert rig.router.seen == [] and rig.llm == [] and rig.interim == [] and rig.registry.log == []
    assert rig.handler.pending_confirmation["kind"] == "print_offer" and rig.handler.pending_confirmation["part"] == "wheel"
    assert out["pending_confirmation"]["query"] == "left front wheel" and rig.spoken == [(SCENE, "scene")]


@pytest.mark.parametrize("text", ["what is this", "What is that?", "what am I looking at", "whats this", "What's this?", "nova what's this"])
def test_every_scene_phrase_is_matched(text):
    assert match_demo_scene(text, policy())
    assert Rig().say(text)["route"] == "scene"


@pytest.mark.parametrize("text", ["what is this rover for", "hey nova what is the moon", "what is", "tell me what is this"])
def test_only_a_whole_utterance_matches(text):
    assert not match_demo_scene(text, policy())


def test_flag_off_keeps_the_old_routing():
    rig = Rig(enabled=False, label="NO_TOOL")
    out = rig.say("what is this")
    assert out["route"] != "scene" and "demo_scene" not in out  # with the flag off "what is this" is a vision question
    assert rig.handler.pending_confirmation is None and load_policy()["demo_scene"]["enabled"] is False


def test_yes_please_makes_exactly_one_queue_print_call_and_points_the_display_at_the_wheel_page():
    rig = Rig()
    rig.say("what is this")
    out = rig.say("yes please")
    assert out["route"] == "print_confirmed" and out["spoken"] == QUEUED and out["verifier"]["ok"] and out["demo_scene"] is True
    assert rig.registry.log == [("queue_print", {"part": "wheel"})] and out["tool_calls"] == 1 and rig.registry.calls_this_turn == 1
    data = out["tool_result"]["data"]
    assert data["status"] == "queued" and data["part"] == "wheel" and data["simulated"] is True and data["demo"] is True and data["job_id"]
    display = out["display"]
    assert set(display) == {"doc_id", "section_id", "title", "asset", "source"}
    assert "Wheel" in display["source"] or "DriveModules" in display["source"]
    assert display["asset"] and display["doc_id"] and display["section_id"].startswith(display["doc_id"])
    assert rig.router.seen == [] and rig.llm == [] and rig.interim == [] and rig.handler.pending_confirmation is None
    assert data["job_id"] not in out["spoken"]
    assert rig.say("yes please")["route"] != "print_confirmed" and len(rig.registry.log) == 1  # the offer is consumed


def test_yes_without_a_retriever_still_queues_but_shows_nothing():
    rig = Rig(retriever=None)
    rig.say("what is this")
    out = rig.say("yes")
    assert out["route"] == "print_confirmed" and "display" not in out and len(rig.registry.log) == 1


def test_no_acknowledges_and_queues_nothing():
    rig = Rig()
    rig.say("what is this")
    out = rig.say("no thanks")
    assert out["route"] == "print_declined" and out["spoken"] == DECLINED and out["tool_calls"] == 0 and rig.registry.log == []
    assert "display" not in out and rig.handler.pending_confirmation is None


def test_expired_offer_goes_back_to_normal_routing():
    rig = Rig(label="NO_TOOL")
    rig.say("what is this")
    for _ in range(policy()["confirmation"]["expiry_turns"]):
        rig.say("open the pod bay doors please now")
    out = rig.say("yes please")
    assert out["route"] != "print_confirmed" and rig.registry.log == [] and "yes please" in rig.router.seen


def test_critical_alert_during_the_offer_wins_and_clears_it():
    rig = Rig()
    rig.say("what is this")
    rig.snapshot = CRITICAL
    out = rig.say("yes please")
    assert out["alert_line"] and rig.registry.log == [] and "display" not in out and rig.handler.pending_confirmation is None
    rig.snapshot = None
    assert rig.say("yes please")["route"] != "print_confirmed" and rig.registry.log == []


def test_critical_alert_preempts_the_scene_phrase():
    rig = Rig(snapshot=CRITICAL)
    out = rig.say("what is this")
    assert out["route"] == "scene" and out["alert_line"] and rig.router.seen == [] and rig.handler.pending_confirmation is None
    assert "Want me" not in out["spoken"] and "3D" not in out["spoken"]


@pytest.mark.parametrize("faults", [None, [], [{"part": "x", "label": "x", "status": "ok", "print_part": "wheel"}]]) 
def test_no_fault_fixture_speaks_the_no_fault_line_and_offers_nothing(faults):
    rig = Rig(faults=faults)
    out = rig.say("what is this")
    assert out["route"] == "scene" and out["spoken"] == NO_FAULT and rig.handler.pending_confirmation is None and rig.registry.log == []
    assert rig.say("yes please")["route"] != "print_confirmed"


def test_malformed_fixture_entries_are_ignored():
    from router.demo_scene import printable_fault
    assert printable_fault({"rover_faults": ["junk", {"status": "damaged"}, {"part": "a", "label": "b", "status": "damaged", "print_part": 3}]}) is None
    assert printable_fault({"rover_faults": "junk"}) is None and printable_fault({}) is None


def test_refill_confirmation_is_unchanged_while_the_scene_is_enabled():
    sim = SimState(minutes_remaining=45)
    handler = TurnHandler(FakeRouter("call_navigation", .99), sim.snapshot, default_registry(sim), policy=policy())
    offer = handler.handle("take me to the outpost")
    assert handler.pending_confirmation["kind"] == "refill_call", offer["spoken"]
    out = handler.handle("yes please")
    assert out["route"] == "confirmed_yes" and out["tool_call"]["name"] == "send_message" and out["tool_calls"] == 1 and "display" not in out


def test_a_scene_during_a_refill_offer_replaces_it():
    sim = SimState(minutes_remaining=45, rover_faults=DEMO_ROVER_FAULTS)
    handler = TurnHandler(FakeRouter("call_navigation", .99), sim.snapshot, default_registry(sim), policy=policy())
    handler.handle("take me to the outpost")
    assert handler.handle("what is this")["route"] == "scene" and handler.pending_confirmation["kind"] == "print_offer"


def test_a_blocked_scene_line_arms_no_offer():
    rig = Rig(verifier=lambda text, source: {"ok": False, "spoken": [], "unsupported": []})
    out = rig.say("what is this")
    assert out["route"] == "scene" and out["spoken"] == load_policy()["templates"]["verifier_blocked"]
    assert rig.handler.pending_confirmation is None


def test_verifier_passes_every_demo_line_and_the_words_avoid_the_repair_terms():
    section = load_policy()["demo_scene"]
    for line in (SCENE, QUEUED, DECLINED, NO_FAULT):
        assert verify_numbers(line, {"printer": "3D printer"})["ok"], line
    assert not verify_numbers(SCENE, {})["ok"]  # the "3D" needs its source
    assert all(not any(c.isdigit() for c in section[k]) for k in ("no_fault_line", "queued_line", "declined_line"))
    scope = load_policy()["llm_scope"]["speaking_scope"]
    repair = [scope["repair_terms"], scope["consequential_terms"]]
    words = " ".join(section[k] for k in ("scene_line", "no_fault_line", "queued_line", "declined_line")).lower()
    assert repair and not any(term in words for terms in repair for term in terms)
    assert not any(w in words for w in ("safe", "unsafe", "drive", "camera", "see "))


def test_policy_section_is_a_placeholder_off_by_default_and_the_validator_rejects_bad_sections():
    section = load_policy()["demo_scene"]
    assert section["status"].startswith("PLACEHOLDER") and section["enabled"] is False
    for mutate in (lambda s: s.update(enabled="yes"), lambda s: s.update(phrases=[]), lambda s: s.update(queued_line="Queued 5."),
                   lambda s: s.update(scene_line="Damage on the thing."), lambda s: s.pop("declined_line")):
        bad = copy.deepcopy(section)
        mutate(bad)
        with pytest.raises(ValueError):
            validate_demo_scene(bad)
    validate_demo_scene(section)


def test_queue_print_is_the_seventh_tool_with_a_fixed_schema_and_unique_job_ids():
    registry = default_registry()
    assert registry.names()[-1] == "queue_print" and len(registry.names()) == 7
    assert registry.schema("queue_print") == {"required": ["part"], "properties": {"part": str}}
    registry.begin_turn()
    first = registry.call("queue_print", {"part": "wheel"})
    registry.begin_turn()
    assert registry.call("queue_print", {"part": "wheel"})["data"]["job_id"] != first["data"]["job_id"]
    registry.begin_turn()
    assert registry.call("queue_print", {"part": "laser"})["ok"] is False


def test_snapshot_only_carries_rover_faults_when_the_fixture_is_set():
    assert "rover_faults" not in SimState().snapshot() and SimState(rover_faults=DEMO_ROVER_FAULTS).snapshot()["rover_faults"] == DEMO_ROVER_FAULTS


# ---- stt wiring ----
@pytest.fixture
def stt_env(monkeypatch):
    monkeypatch.setattr(llm_generator, "check_server", lambda url, timeout, **kw: (True, "fake"))
    monkeypatch.delenv("NOVA_DEMO_SCENE", raising=False)
    monkeypatch.delenv("NOVA_DOCS_DIR", raising=False)


def test_default_handler_leaves_the_demo_off_when_unset(stt_env):
    handler = stt._default_handler(FakeRouter("NO_TOOL", 0.97), speak=False)
    assert handler.policy["demo_scene"]["enabled"] is False and handler.retriever is None
    assert "rover_faults" not in handler.alert_source()


@pytest.mark.parametrize("how", ["env", "option"])
def test_default_handler_enables_the_demo_from_env_or_option(stt_env, monkeypatch, how):
    if how == "env":
        monkeypatch.setenv("NOVA_DEMO_SCENE", "1")
    handler = stt._default_handler(FakeRouter("NO_TOOL", 0.97), speak=False, llm_options={"demo_scene": True} if how == "option" else None)
    assert handler.policy["demo_scene"]["enabled"] is True and handler.retriever is not None
    assert handler.alert_source()["rover_faults"] == DEMO_ROVER_FAULTS
    assert handler.handle("what is this")["spoken"] == SCENE
    out = handler.handle("yes please")
    assert out["route"] == "print_confirmed" and out["display"]["source"]


def test_default_handler_docs_dir_option_overrides_the_default_corpus(stt_env, tmp_path):
    (tmp_path / "a.md").write_text("# Other\n\nSomething unrelated to rovers at all, just filler words for the chunk.\n")
    handler = stt._default_handler(FakeRouter("NO_TOOL", 0.97), speak=False, llm_options={"demo_scene": True, "docs_dir": str(tmp_path)})
    handler.handle("what is this")
    assert "display" not in handler.handle("yes")


@pytest.mark.parametrize('phrase', ['no just navigate', 'just go'])
def test_accept_as_order_during_print_offer_only_declines(phrase):
    rig = Rig()
    rig.say('what is this')
    out = rig.say(phrase)
    assert out['route'] == 'print_declined' and out['spoken'] == DECLINED
    assert out['tool_calls'] == 0 and rig.registry.log == []
    assert 'accepted_as_order' not in out and rig.handler.pending_confirmation is None

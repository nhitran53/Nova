"""Whole-utterance session end and the interim "Let me think." line (fake router, fake generator, no model)."""
import copy
import json

import pytest

from router.generation import GenerationResult
from router.tools import SimState, default_registry
from router.turn import TurnHandler, load_policy
from router.tests.test_turn import CRITICAL, FakeRouter

MOON = {"text": "The Moon is Earth's only natural satellite, about 384400 kilometers away.", "score": 0.9}


class RecordingGenerator:
    """Fake generator that logs 'generate' into a shared event list, optionally running a hook mid-generation."""

    def __init__(self, events, text="Hello there.", during=None):
        self.events, self.text, self.during = events, text, during

    def generate(self, req):
        self.events.append(("generate", req.kind))
        if self.during:
            self.during()
        return GenerationResult(json.dumps({"refusal": None, "text": self.text}))


class Rig:
    def __init__(self, label="NO_TOOL", confidence=0.97, snapshot=None, policy=None, interim_fn="record", during=None,
                 retriever=None, tts=True):
        self.events, self.sim, self.snapshot, self.spoken = [], SimState(), snapshot, []
        self.polls = 0
        if interim_fn == "record":
            interim_fn = lambda line, route: self.events.append(("interim", line, route))
        self.router = FakeRouter(label, confidence)
        self.handler = TurnHandler(
            self.router, self._source, default_registry(self.sim), policy=policy,
            generator=RecordingGenerator(self.events, during=(lambda: during(self)) if during else None),
            interim_fn=interim_fn, retriever=retriever, tts_fn=(lambda line, route: self.spoken.append((line, route))) if tts else None)

    def _source(self):
        self.polls += 1
        return self.snapshot or self.sim.snapshot()

    def say(self, text):
        return self.handler.handle(text)


# ---- session end ----
@pytest.mark.parametrize("text", ["thank you", "Thanks!", "thank you Nova", "Thanks, Nova", "that's all", "Goodbye.", "bye"])
def test_whole_utterance_session_end_speaks_the_line_and_flags_end_session(text):
    rig = Rig()
    out = rig.say(text)
    assert out["route"] == "session_end" and out["spoken"] == "Happy to help!" and out["end_session"] is True
    assert out["verifier"]["ok"] and out["tool_calls"] == 0 and not out["llm_called"] and out["pipeline"] is None
    assert rig.events == [] and rig.router.seen == [] and rig.spoken == [("Happy to help!", "session_end")]


def test_session_end_policy_is_a_placeholder_section_with_the_agreed_phrases():
    section = load_policy()["session_end"]
    assert section["status"].startswith("PLACEHOLDER") and section["line"] == "Happy to help!"
    assert section["phrases"] == ["thank you", "thanks", "thank you nova", "thanks nova", "that's all", "goodbye", "bye"]


def test_ordinary_turns_do_not_set_end_session():
    assert "end_session" not in Rig("call_o2", .99).say("how is my oxygen")


def test_critical_alert_wins_over_session_end():
    rig = Rig(snapshot=CRITICAL)
    out = rig.say("thank you")
    assert out["alert_line"] and out["route"] == "session_end" and out["end_session"]


def test_session_end_clears_pending_clarification_and_confirmation():
    rig = Rig("call_navigation", .9)
    rig.handler.pending_clarification = {"kind": "stale"}
    rig.handler.pending_confirmation = {"kind": "refill_call", "turn": 0}
    out = rig.say("thanks")
    assert out["end_session"] is True and rig.handler.pending_clarification is None and rig.handler.pending_confirmation is None


def test_yes_thank_you_still_confirms_a_pending_offer():
    sim = SimState(minutes_remaining=45)
    handler = TurnHandler(FakeRouter("call_navigation", .99), sim.snapshot, default_registry(sim))
    offer = handler.handle("take me to the outpost")
    assert handler.pending_confirmation is not None, offer["spoken"]
    out = handler.handle("yes thank you")
    assert out["route"] == "confirmed_yes" and "end_session" not in out


def test_thanks_take_me_home_routes_normally():
    rig = Rig("call_navigation", .99)
    out = rig.say("thanks, take me home")
    assert out["route"] == "tool" and out["tool_call"]["args"] == {"place": "homebase"} and "end_session" not in out


# ---- interim ----
def test_interim_policy_is_enabled_placeholder_line():
    gen = load_policy()["generation"]
    assert gen["interim"] == {"enabled": True, "line": "Let me think."} and gen["status"].startswith("PLACEHOLDER")


@pytest.mark.parametrize("label,text,kind,retriever", [
    ("NO_TOOL", "hello there", "chitchat", None),
    ("call_llm", "why is the sky blue", "general_explanation", None),
    ("call_llm", "how far away is the moon", "grounded_answer", lambda t: [MOON]),
])
def test_interim_is_spoken_before_generation_for_llm_kinds_and_tts_stays_once(label, text, kind, retriever):
    rig = Rig(label, retriever=retriever)
    out = rig.say(text)
    assert out["route"] == "llm" and out["interim"] == "Let me think." and "interim_error" not in out
    assert rig.events == [("interim", "Let me think.", "llm_interim"), ("generate", kind)]
    assert len(rig.spoken) == 1 and rig.spoken[0][1] == "llm" and out["spoken"] == "Hello there."


def test_no_interim_on_tool_template_greeting_or_alert_routes():
    for label, conf, text, snapshot, route in [("call_o2", .99, "how is my oxygen", None, "tool"),
                                              ("call_o2", .1, "blorp", None, "low_confidence"),
                                              ("NO_TOOL", .97, "hi", None, "fast_reply"),
                                              ("NO_TOOL", .97, "hello there", CRITICAL, "alert_active_deferred")]:
        rig = Rig(label, conf, snapshot=snapshot)
        out = rig.say(text)
        assert out["route"] == route and rig.events == [] and "interim" not in out, (route, rig.events)
        assert len(rig.spoken) == (2 if snapshot else 1)


def test_interim_can_be_disabled_or_absent_or_have_no_callback():
    policy = copy.deepcopy(load_policy())
    policy["generation"]["interim"]["enabled"] = False
    rig = Rig(policy=policy)
    assert "interim" not in rig.say("hello there") and rig.events == [("generate", "chitchat")]
    del policy["generation"]["interim"]
    rig = Rig(policy=policy)
    assert "interim" not in rig.say("hello there") and rig.events == [("generate", "chitchat")]
    rig = Rig(interim_fn=None)
    out = rig.say("hello there")
    assert "interim" not in out and out["spoken"] == "Hello there." and rig.events == [("generate", "chitchat")]


def test_interim_failure_is_recorded_and_the_turn_still_answers():
    def broken(line, route):
        raise RuntimeError("audio device gone")

    out = Rig(interim_fn=broken).say("hello there")
    assert out["route"] == "llm" and out["spoken"] == "Hello there." and "interim" not in out
    assert out["interim_error"] == "RuntimeError: audio device gone"


def test_interim_line_with_a_number_is_not_spoken():
    policy = copy.deepcopy(load_policy())
    policy["generation"]["interim"]["line"] = "Give me 5 seconds."
    rig = Rig(policy=policy)
    out = rig.say("hello there")
    assert out["interim_error"] and "interim" not in out and not [e for e in rig.events if e[0] == "interim"]


def test_loader_refuses_an_interim_line_with_a_number():
    from router.generation import validate_generation_policy
    policy = copy.deepcopy(load_policy())
    policy["generation"]["interim"]["line"] = "Give me 5 seconds."
    with pytest.raises(ValueError, match="interim"):
        validate_generation_policy(policy["generation"])


def test_critical_alert_appearing_during_generation_preempts_the_answer():
    def raise_alert(rig):
        rig.snapshot = CRITICAL

    rig = Rig(during=raise_alert)
    out = rig.say("hello there")
    assert out["route"] == "alert" and out["spoken"].startswith("Critical alert. Oxygen at 14.2 percent")
    assert out["verifier"]["ok"] and out["alerts"] and out["interim"] == "Let me think."
    assert rig.events == [("interim", "Let me think.", "llm_interim"), ("generate", "chitchat")]
    assert len(rig.spoken) == 1 and rig.spoken[0][1] == "alert" and rig.handler.history == []
    assert rig.polls == 2


def test_warning_appearing_during_generation_does_not_preempt():
    def raise_warning(rig):
        rig.snapshot = {**rig.sim.snapshot(), "temp_c": 33.0}

    out = Rig(during=raise_warning).say("hello there")
    assert out["route"] == "llm" and out["spoken"] == "Hello there."

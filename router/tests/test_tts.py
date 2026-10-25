"""TurnHandler tts_fn hook: called once per turn with the final (verified / substituted) spoken line. No network, no audio."""
import pytest

from router.tools import SimState, default_registry
from router.turn import TurnHandler


class FakeRouter:
    def __init__(self):
        self.label, self.confidence = "call_o2", 0.99

    def predict(self, text):
        return {"label": self.label, "confidence": self.confidence,
                "topk": [{"label": self.label, "p": self.confidence}, {"label": "NO_TOOL", "p": 1 - self.confidence}]}


class Recorder:
    def __init__(self, error=None):
        self.calls, self.error = [], error

    def __call__(self, text, route):
        self.calls.append((text, route))
        if self.error:
            raise self.error


def rig(tts_fn, llm=None, **sim_values):
    sim, router = SimState(**sim_values), FakeRouter()
    handler = TurnHandler(router, lambda: sim.snapshot(), default_registry(sim), tts_fn=tts_fn,
                          llm_fn=llm or (lambda text, policy, passages=None: "Hello there!"),
                          retriever=lambda text: [{"text": "The Moon is a satellite.", "score": 0.9}])
    return handler, router


def say(handler, router, text, label, confidence=0.99):
    router.label, router.confidence = label, confidence
    return handler.handle(text)


SCENARIOS = [
    ("tool", "how much oxygen", "call_o2", 0.99, {}, "tool"),
    ("slot_question", "take me there", "call_navigation", 0.97, {}, "slot_question"),
    ("ood", "capital of france", "OOD", 0.95, {}, "ood"),
    ("low_confidence", "mumble", "call_o2", 0.2, {}, "low_confidence"),
    ("llm", "thanks", "NO_TOOL", 0.97, {}, "llm"),
    ("alert", "take me home", "call_navigation", 0.99, {"o2_pct": 14.2}, "alert"),
]


@pytest.mark.parametrize("name, text, label, confidence, sim, route", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_tts_fn_is_called_exactly_once_with_the_spoken_line_and_route(name, text, label, confidence, sim, route):
    tts = Recorder()
    handler, router = rig(tts, **sim)
    out = say(handler, router, text, label, confidence)
    assert out["route"] == route and tts.calls == [(out["spoken"], route)]
    assert out["spoken"] and "tts_error" not in out


def test_blocked_turn_speaks_the_safe_fallback_not_the_unverified_text():
    tts = Recorder()
    handler, router = rig(tts, llm=lambda text, policy, passages=None: "The moon is 384400 kilometers away.")
    out = say(handler, router, "how far is the moon", "NO_TOOL", 0.97)
    assert out["blocked_text"] == "The moon is 384400 kilometers away."
    assert tts.calls == [(handler.policy["templates"]["verifier_blocked"], "llm")]
    assert "384400" not in tts.calls[0][0]


def test_confirmation_turns_are_spoken_once_each():
    tts = Recorder()
    handler, router = rig(tts, minutes_remaining=55)
    offer = say(handler, router, "take me to the outpost", "call_navigation")
    assert offer["pending_confirmation"] and tts.calls[-1] == (offer["spoken"], "tool")
    yes = say(handler, router, "yes please", "call_o2")
    assert yes["route"] == "confirmed_yes" and tts.calls[-1] == (yes["spoken"], "confirmed_yes") and len(tts.calls) == 2
    say(handler, router, "take me to the outpost", "call_navigation")
    no = say(handler, router, "no thanks", "call_o2")
    assert no["route"] == "confirmed_no" and tts.calls[-1] == (no["spoken"], "confirmed_no") and len(tts.calls) == 4


def test_tts_exception_does_not_break_the_turn_and_is_recorded():
    handler, router = rig(Recorder(error=RuntimeError("quota exceeded")))
    out = say(handler, router, "how much oxygen", "call_o2")
    assert out["tts_error"] == "RuntimeError: quota exceeded"
    assert out["route"] == "tool" and out["spoken"].startswith("Oxygen at") and out["tool_calls"] == 1


def test_tts_failure_leaves_result_identical_apart_from_tts_error():
    ok_handler, ok_router = rig(Recorder())
    bad_handler, bad_router = rig(Recorder(error=ValueError("boom")))
    good = say(ok_handler, ok_router, "how much oxygen", "call_o2")
    bad = say(bad_handler, bad_router, "how much oxygen", "call_o2")
    assert {k: v for k, v in bad.items() if k != "tts_error"} == good


def test_default_tts_fn_none_makes_no_call_and_adds_no_key():
    handler, router = rig(None)
    out = say(handler, router, "how much oxygen", "call_o2")
    assert handler.tts_fn is None and "tts_error" not in out and out["spoken"]

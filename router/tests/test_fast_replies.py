"""Policy-driven fast replies: whole-utterance greeting/identity answered before the router (fake router, fake generator)."""
import copy
import json

import pytest

from router.fast_replies import match_fast_reply
from router.tests.test_session_end_interim import Rig
from router.tests.test_turn import CRITICAL
from router.tools import SimState, default_registry
from router.turn import TurnHandler, load_policy
from router.tests.test_turn import FakeRouter

GREETING, IDENTITY = "Hello. How can I help?", "I'm Nova, your navigation and observation voice assistant."

CASES = (
    [(t, "greeting", GREETING) for t in ["hi", "Hello!", "hey", "Hi Nova.", "hey nova", "Good morning", "good morning, Nova",
                                           "Hello Nova", "okay nova hello", "nova hi", "hey, hello"]]
    + [(t, "identity", IDENTITY) for t in ["who are you", "Who are you?", "what's your name", "What’s your name?".replace("’", "'"),
                                           "what is your name", "whats your name", "hey nova, what's your name", "nova who are you",
                                           "hi who are you"]]
)


@pytest.mark.parametrize("text,intent,line", CASES)
def test_fast_reply_speaks_the_fixed_line_without_router_llm_or_interim(text, intent, line):
    rig = Rig("call_llm", 0.99)  # a router that would otherwise send everything to the LLM
    out = rig.say(text)
    assert out["route"] == "fast_reply" and out["fast_intent"] == intent and out["spoken"] == line
    assert out["verifier"]["ok"] and out["tool_calls"] == 0 and not out["llm_called"] and out["pipeline"] is None
    assert rig.router.seen == [] and rig.events == [] and rig.spoken == [(line, "fast_reply")]


@pytest.mark.parametrize("text", ["hello how much oxygen do I have", "hi nova take me home", "who are you and where am I",
                                  "what is your name for the rover", "good evening", "nova", "hey nova what is the moon"])
def test_only_a_whole_utterance_match_takes_the_fast_path(text):
    policy = load_policy()
    assert match_fast_reply(text, policy) is None
    rig = Rig("call_o2", 0.99)
    out = rig.say(text)
    assert out["route"] != "fast_reply" and "fast_intent" not in out and rig.router.seen == [text]


def test_a_greeting_phrase_is_matched_before_a_wake_prefix_is_stripped():
    policy = load_policy()
    assert match_fast_reply("hey", policy)[0] == "greeting" and match_fast_reply("hi nova", policy)[0] == "greeting"
    assert match_fast_reply("hey nova who are you", policy) == ("identity", IDENTITY)


def test_critical_alert_wins_over_a_fast_reply():
    rig = Rig(snapshot=CRITICAL)
    out = rig.say("hello")
    assert out["alert_line"] and out["route"] == "fast_reply" and out["fast_intent"] == "greeting" and rig.router.seen == []


def test_pending_confirmation_answer_is_not_swallowed_by_a_fast_reply():
    sim = SimState(minutes_remaining=45)
    handler = TurnHandler(FakeRouter("call_navigation", .99), sim.snapshot, default_registry(sim))
    offer = handler.handle("take me to the outpost")
    assert handler.pending_confirmation is not None, offer["spoken"]
    assert handler.handle("yes")["route"] == "confirmed_yes"
    handler.handle("take me to the outpost")
    out = handler.handle("hello")  # not a yes/no: the fast reply answers it and the offer is dropped
    assert out["route"] == "fast_reply" and handler.pending_confirmation is None


def test_pending_clarification_answer_is_handled_before_the_fast_reply():
    rig = Rig("call_navigation", .9)
    rig.handler.pending_clarification = {"kind": "stale"}
    called = []
    rig.handler._continue_clarification = lambda result, snapshot: called.append(1) or rig.handler._speak(result, "clarify", "ok", {})
    out = rig.say("hello")
    assert called == [1] and out["route"] == "clarify"


def test_disabled_flag_restores_the_router_path():
    policy = copy.deepcopy(load_policy())
    policy["fast_replies"]["enabled"] = False
    rig = Rig("NO_TOOL", 0.97, policy=policy)
    out = rig.say("hello")
    assert out["route"] != "fast_reply" and rig.router.seen == ["hello"] and rig.events == [("generate", "general_explanation")]
    assert Rig("NO_TOOL", 0.97, policy=policy).say("hi")["route"] == "greeting"  # speaking_scope.greetings still works


def test_missing_section_is_the_same_as_disabled():
    policy = copy.deepcopy(load_policy())
    del policy["fast_replies"]
    assert match_fast_reply("hello", policy) is None
    assert Rig("NO_TOOL", 0.97, policy=policy).say("hi")["route"] == "greeting"


def test_verifier_still_runs_on_the_fast_line():
    seen = []

    def spy(text, source):
        seen.append(text)
        return {"ok": False, "spoken": [], "unsupported": []}

    sim = SimState()
    out = TurnHandler(FakeRouter(), sim.snapshot, default_registry(sim), verifier=spy).handle("hello")
    assert seen == [GREETING] and out["route"] == "fast_reply" and out["blocked_text"] == GREETING
    assert out["spoken"] == load_policy()["templates"]["verifier_blocked"]


def test_fast_replies_section_is_an_enabled_placeholder_without_digits():
    section = load_policy()["fast_replies"]
    assert section["status"].startswith("PLACEHOLDER") and section["enabled"] is True
    assert not any(c.isdigit() for intent in section["intents"].values() for c in intent["line"])


def _write(tmp_path, mutate):
    policy = copy.deepcopy(load_policy())
    mutate(policy["fast_replies"])
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    return path


def test_loader_accepts_a_well_formed_section(tmp_path):
    assert load_policy(_write(tmp_path, lambda s: s["intents"]["greeting"].update(line="Hi there.")))["fast_replies"]


@pytest.mark.parametrize("mutate,message", [
    (lambda s: s.update(enabled="yes"), "enabled must be a boolean"),
    (lambda s: s.update(wake_prefixes="nova"), "wake_prefixes"),
    (lambda s: s.update(wake_prefixes=["nova", ""]), "wake_prefixes"),
    (lambda s: s.update(intents={}), "intents must be a non-empty object"),
    (lambda s: s["intents"]["greeting"].update(phrases=[]), "greeting.phrases"),
    (lambda s: s["intents"]["greeting"].update(phrases=["hi", "!!"]), "greeting.phrases"),
    (lambda s: s["intents"]["greeting"].update(line=""), "greeting.line"),
    (lambda s: s["intents"]["greeting"].update(line="You have 3 messages."), "without digits"),
    (lambda s: s["intents"]["identity"]["phrases"].append("hello"), "two intents"),
    (lambda s: s.update(status="approved"), "needs a 'status' starting with PLACEHOLDER"),
])
def test_loader_rejects_a_malformed_section(tmp_path, mutate, message):
    with pytest.raises(ValueError, match=message):
        load_policy(_write(tmp_path, mutate))

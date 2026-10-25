"""turn.py with a fake router, fake arbiter, fake alert source and fake tools (no model, no LLM)."""
import copy
import json

import pytest

from router import load_labels
from router.tools import (MAX_TOOLS, OneCallPerTurnError, SchemaError, SimState, ToolLimitError, ToolRegistry,
                          default_registry)
from router.turn import TurnHandler, assess_feasibility, evaluate_alerts, load_policy, match_confirmation
from router.verifier import verify_numbers

PASSAGE = {"text": "The Moon is Earth's only natural satellite, about 384400 kilometers away.", "score": 0.9}
LABELS = load_labels(__import__("pathlib").Path(__file__).parent.parent / "labels.json")
TOOL_LABELS = ["call_safety", "call_navigation", "call_map", "call_data", "call_o2"]
CRITICAL = {"o2_pct": 14.2, "bpm": 72, "radiation_usv_h": 0.34, "temp_c": 21.5, "suit_pressure_kpa": 29.6}


class FakeRouter:
    def __init__(self, label="call_o2", confidence=0.99):
        self.label, self.confidence, self.seen = label, confidence, []

    def predict(self, text):
        self.seen.append(text)
        rest = (1 - self.confidence) / 2
        return {"label": self.label, "confidence": self.confidence,
                "topk": [{"label": self.label, "p": self.confidence}, {"label": "call_data", "p": rest},
                         {"label": "NO_TOOL", "p": rest}]}


class FakeArbiter:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def __call__(self, text, topk):
        self.calls.append(text)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class FakeLLM:
    def __init__(self, reply="Hello there!"):
        self.reply, self.calls, self.passages_seen = reply, [], []

    def __call__(self, text, prompt_policy, passages=None):
        self.calls.append((text, prompt_policy))
        self.passages_seen.append(passages)
        return self.reply


class SpyRegistry(ToolRegistry):
    """default fake tools, recording every call that reaches a tool."""

    def __init__(self, sim=None):
        super().__init__()
        self.log = []
        inner = default_registry(sim)
        for name in inner.names():
            self.register(name, inner.schema(name), self._wrap(name, inner._tools[name][1]))

    def _wrap(self, name, fn):
        def spy(**args):
            self.log.append((name, args))
            return fn(**args)
        return spy


class Rig:
    def __init__(self, label="call_o2", confidence=0.99, snapshot=None, arbiter=None, llm=None, policy=None, verifier=None, retriever=None, **sim_values):
        self.sim = SimState(**sim_values)
        self.snapshot = snapshot
        self.router, self.arbiter, self.llm = FakeRouter(label, confidence), arbiter, llm or FakeLLM()
        self.registry = SpyRegistry(self.sim)
        kwargs = {} if verifier is None else {"verifier": verifier}
        self.handler = TurnHandler(self.router, lambda: self.snapshot or self.sim.snapshot(), self.registry,
                                   llm_fn=self.llm, arbiter_fn=arbiter, policy=policy, retriever=retriever, **kwargs)

    def say(self, text):
        return self.handler.handle(text)


# ---- 1. critical alert preempts everything ----
@pytest.mark.parametrize("confidence", [0.99, 0.7, 0.2], ids=["high", "mid-band", "low"])
@pytest.mark.parametrize("label", LABELS)
def test_critical_alert_preempts_every_class_and_calls_neither_router_nor_llm(label, confidence):
    arbiter = FakeArbiter(label)
    rig = Rig(label, confidence, snapshot=CRITICAL, arbiter=arbiter)
    out = rig.say("take me to the airlock")
    assert out["route"] == "alert" and out["spoken"].startswith("Critical alert. Oxygen at 14.2 percent")
    assert rig.router.seen == [] and rig.llm.calls == [] and arbiter.calls == []
    assert rig.registry.log == [] and out["tool_calls"] == 0 and out["pipeline"] is None and not out["llm_called"]
    assert out["verifier"]["ok"]


def test_several_critical_alerts_are_all_spoken_in_policy_priority_order():
    rig = Rig(snapshot={**CRITICAL, "radiation_usv_h": 60.0})
    spoken = rig.say("hi")["spoken"]
    assert spoken.index("Oxygen") < spoken.index("Radiation")


def test_warning_tier_does_not_preempt_and_is_reported():
    rig = Rig(snapshot={**CRITICAL, "o2_pct": 87.5, "temp_c": 33.0})
    out = rig.say("how much oxygen")
    assert out["route"] == "tool" and [a["severity"] for a in out["alerts"]] == ["warning"]


def test_alert_source_failure_is_not_swallowed():
    def broken():
        raise RuntimeError("sim offline")

    handler = TurnHandler(FakeRouter(), broken, default_registry())
    with pytest.raises(RuntimeError, match="sim offline"):
        handler.handle("hi")


def test_evaluate_alerts_thresholds_are_exclusive():
    policy = load_policy()
    assert evaluate_alerts({"o2_pct": 16.0}, policy) == [{"field": "o2_pct", "severity": "warning", "value": 16.0}]
    assert evaluate_alerts({"o2_pct": 15.9}, policy)[0]["severity"] == "critical"
    assert evaluate_alerts({"unknown_field": 1}, policy) == []


# ---- 3. slots ----
@pytest.mark.parametrize("label, question", [("call_navigation", "Where would you like to go?"),
                                             ("call_map", "Which place should I show on the map?")])
def test_navigation_or_map_without_a_place_asks_a_slot_question_and_calls_no_tool(label, question):
    rig = Rig(label)
    out = rig.say("take me there")
    assert out["route"] == "slot_question" and out["spoken"] == question
    assert rig.registry.log == [] and out["tool_calls"] == 0 and out["tool_call"] is None


@pytest.mark.parametrize("text", ["do not go to the crater", "never go to the crater", "please don't go to the crater"])
def test_negated_navigation_verb_does_not_navigate_to_the_forbidden_place(text):
    rig = Rig("call_navigation")
    out = rig.say(text)
    assert out["route"] == "slot_question" and rig.registry.log == [] and out["tool_calls"] == 0


def test_no_then_a_new_command_still_navigates_to_the_new_place():
    rig = Rig("call_navigation")
    out = rig.say("no, take me to the airlock")
    assert out["tool_call"] == {"name": "navigation", "args": {"place": "airlock"}}


def test_ambiguous_place_asks_which_place_and_calls_no_tool():
    rig = Rig("call_navigation")
    out = rig.say("go to the rover or the airlock")
    assert out["route"] == "slot_question" and out["spoken"] == "Which place do you mean: rover or airlock?"
    assert rig.registry.log == []


def test_turn_rechecks_required_slots_against_the_registry_schema():
    """Defence in depth: even if the pipeline let a tool route through without args, the turn does not call."""
    rig = Rig("call_navigation")
    rig.handler.pipeline.handle = lambda text: {"route": "tool", "tool": "navigation", "args": {}, "label": "call_navigation",
                                                "confidence": 0.99, "gate": "tau_hi"}
    out = rig.say("anything")
    assert out["route"] == "slot_question" and rig.registry.log == []


def test_navigation_with_a_place_runs_one_tool_and_speaks_the_tool_result():
    rig = Rig("call_navigation")
    out = rig.say("go from the rover to the airlock")
    assert rig.registry.log == [("navigation", {"place": "airlock"})] and out["tool_calls"] == 1
    assert out["spoken"].startswith("Navigation information for airlock: estimated straight-line distance 80 meters, walking time 1 minutes. ") and out["verifier"]["ok"]


# ---- 5. low confidence / OOD / llm ----
def test_low_confidence_gets_a_template_reply_not_the_llm():
    rig = Rig("call_navigation", 0.3)
    out = rig.say("hmm take me")
    assert out["route"] == "low_confidence" and out["spoken"] == load_policy()["templates"]["low_confidence"]
    assert rig.llm.calls == [] and rig.registry.log == []


def test_ood_gets_the_template_reply_not_the_llm():
    rig = Rig("OOD", 0.97)
    out = rig.say("what is the capital of france")
    assert out["route"] == "ood" and out["spoken"] == load_policy()["templates"]["ood"] and rig.llm.calls == []


def test_chitchat_uses_the_injected_llm_stub_ungrounded():
    rig = Rig("NO_TOOL", 0.97, llm=FakeLLM("Hello there!"))
    out = rig.say("hello there")
    assert out["route"] == "llm" and out["spoken"] == "Hello there!" and out["llm_called"]
    assert len(rig.llm.calls) == 1 and rig.llm.calls[0][0] == "hello there" and rig.registry.log == []
    assert rig.llm.passages_seen == [None]


def test_llm_output_with_an_unsupported_number_is_blocked():
    rig = Rig("call_llm", 0.97, llm=FakeLLM("Your oxygen is 99 percent."), retriever=lambda text: [PASSAGE])
    out = rig.say("how is my oxygen")
    assert out["spoken"] == load_policy()["templates"]["verifier_blocked"]
    assert out["blocked_text"] == "Your oxygen is 99 percent." and out["verifier"]["unsupported"] == [99.0]


def test_llm_output_with_a_spelled_out_number_is_blocked_too():
    rig = Rig("NO_TOOL", 0.97, llm=FakeLLM("You have about twenty one minutes left."))
    assert rig.say("hello")["blocked_text"] is not None


# ---- 4. one call per turn, tool limit, schemas ----
def test_one_tool_call_per_turn_and_counter_resets_between_turns():
    rig = Rig("call_o2")
    assert rig.say("oxygen?")["tool_calls"] == 1 and rig.say("oxygen again?")["tool_calls"] == 1
    assert len(rig.registry.log) == 2
    rig.registry.begin_turn()
    rig.registry.call("o2", {})
    with pytest.raises(OneCallPerTurnError):
        rig.registry.call("data", {})
    assert len(rig.registry.log) == 3  # the refused second call never reached a tool


def test_more_than_eight_tools_raises():
    registry = ToolRegistry()
    for i in range(MAX_TOOLS):
        registry.register(f"t{i}", {"required": [], "properties": {}}, lambda: {})
    with pytest.raises(ToolLimitError):
        registry.register("t8", {"required": [], "properties": {}}, lambda: {})
    assert MAX_TOOLS == 8 and len(registry.names()) == 8


def test_default_registry_has_six_fixed_schema_tools_and_rejects_bad_calls():
    registry = default_registry()
    assert registry.names() == ["safety", "navigation", "map", "data", "o2", "send_message"]
    assert len(registry.names()) <= MAX_TOOLS
    registry.begin_turn()
    with pytest.raises(SchemaError):
        registry.call("navigation", {})
    with pytest.raises(SchemaError):
        registry.call("navigation", {"place": 3})
    with pytest.raises(SchemaError):
        registry.call("o2", {"extra": 1})
    with pytest.raises(SchemaError):
        registry.call("coolant", {})
    with pytest.raises(SchemaError):
        registry.call("send_message", {"recipient": "outpost", "urgency": "urgent"})
    with pytest.raises(SchemaError):
        registry.call("send_message", {"recipient": "outpost", "urgency": 1, "message": "x"})
    assert registry.calls_this_turn == 0  # rejected calls do not use up the turn's one call


@pytest.mark.parametrize("label", TOOL_LABELS)
def test_each_tool_class_runs_exactly_one_fake_tool(label):
    rig = Rig(label)
    out = rig.say("check my health" if label == "call_safety" else "do it at the airlock")
    assert out["route"] == "tool" and out["tool_calls"] == 1 and len(rig.registry.log) == 1 and out["verifier"]["ok"]


def test_data_and_o2_speak_the_sim_values():
    rig = Rig("call_data")
    assert rig.say("vitals")["spoken"] == ("Heart rate 72 beats per minute. Radiation 0.34 microsieverts per hour. "
                                           "Temperature 21.5 degrees Celsius.")
    rig = Rig("call_o2")
    assert rig.say("oxygen")["spoken"] == "Oxygen at 87.5 percent, about 212 minutes remaining."


# ---- 6. number verifier ----
def test_template_number_that_differs_from_the_tool_result_blocks_the_speech():
    policy = copy.deepcopy(load_policy())
    policy["templates"]["o2"] = "Oxygen at 99 percent."  # tool_result says 87.5
    rig = Rig("call_o2", policy=policy)
    out = rig.say("oxygen?")
    assert out["tool_result"]["data"]["o2_pct"] == 87.5
    assert out["spoken"] == policy["templates"]["verifier_blocked"] and out["blocked_text"] == "Oxygen at 99 percent."
    assert out["verifier"] == {"ok": False, "spoken": [99.0], "unsupported": [99.0]}


def test_tool_result_with_a_different_number_than_the_template_expects_blocks_the_speech():
    policy = copy.deepcopy(load_policy())
    policy["templates"]["o2"] = "Oxygen at 87.5 percent."
    rig = Rig("call_o2", policy=policy)
    rig.sim.values["o2_pct"] = 42.0  # tool_result now disagrees with the hardcoded template number
    out = rig.say("oxygen?")
    assert out["blocked_text"] == "Oxygen at 87.5 percent." and out["spoken"] == policy["templates"]["verifier_blocked"]


def test_template_that_does_not_fit_the_tool_result_is_never_spoken():
    policy = copy.deepcopy(load_policy())
    policy["templates"]["o2"] = "Oxygen at {no_such_field} percent."
    out = Rig("call_o2", policy=policy).say("oxygen?")
    assert out["spoken"] == policy["templates"]["verifier_blocked"]


def test_verifier_hook_is_injectable_and_applies_to_every_route():
    seen = []

    def spy(text, source):
        seen.append((text, source))
        return {"ok": True, "spoken": [], "unsupported": []}

    for label, conf in (("call_o2", 0.99), ("call_llm", 0.97), ("OOD", 0.97), ("call_o2", 0.3), ("call_o2", 0.7)):
        before = len(seen)
        Rig(label, conf, verifier=spy).say("take me to the airlock")
        assert len(seen) == before + 1
    Rig(snapshot=CRITICAL, verifier=spy).say("x")
    assert seen[-1][1] == CRITICAL  # alerts are verified against the sim snapshot


@pytest.mark.parametrize("text, source, ok", [
    ("Oxygen at 87.5 percent.", {"o2_pct": 87.5}, True),
    ("Oxygen at 87.5 percent.", {"o2_pct": 87.4}, False),
    ("About 21 percent.", {"o2_pct": 20.94}, True),       # rounded to the spoken precision
    ("About 22 percent.", {"o2_pct": 20.94}, False),
    ("It is 1,200 meters.", {"a": {"d": [1200]}}, True),
    ("twenty one", {"x": 21}, True),
    ("two hundred and five", {"x": 205}, True),
    ("nine", {"x": 90}, False),
    ("Hello there.", {}, True),
    ("Count: 1", {"flag": True}, False),                   # bool is not a number
    ("Heading to base 2", {"place": "base 2"}, True),     # digits inside source strings count
    ("Heading to base 7", {"place": "base 2"}, False),
    ("-5", {"v": 5}, False),                              # the sign is part of the number
    ("5", {"v": -5}, False),
    ("-20.0", {"t": -20.0}, True),
    ("Suit temperature -20.0 degrees.", {"temp_c": -20}, True),
    ("Suit temperature \u221220.0 degrees.", {"temp_c": -20}, True),  # Unicode minus
    ("minus twenty", {"t": -20}, True),
    ("minus twenty", {"t": 20}, False),
    ("minus 20", {"t": 20}, False),
    ("minus 20", {"t": -20}, True),
    ("between 5-10 minutes", {"a": 5, "b": 10}, True),    # a range is two positive numbers, not 5 and -10
    ("between 5-10 minutes", {"a": 5, "b": -10}, False),
    ("Heading to ISS-12", {"x": "ISS-12"}, True),         # a hyphen after a word character is not a sign
    ("Logged 2026-10-24.", {"d": "2026-10-24"}, True),
    ("Temperature -5.", {"note": "limit is -5"}, True),   # the sign is read the same way inside source strings
    ("Temperature -5.", {"note": "limit is 5"}, False),
    ("About half of it.", {}, False),
    ("A quarter tank, a third, a dozen, a couple.", {}, False),
    ("Quarters and halves.", {}, False),
    ("About half of it.", {"note": "half"}, True),        # backed by a source string
    ("Takes a few minutes, roughly.", {}, True),          # "few" and "point" are ordinary speech
])
def test_verify_numbers(text, source, ok):
    assert verify_numbers(text, source)["ok"] is ok


def test_vague_quantity_words_are_reported_as_unsupported():
    assert verify_numbers("half a dozen", {})["unsupported"] == ["half", "dozen"]


def test_negative_temperature_critical_alert_is_spoken_unblocked():
    rig = Rig(snapshot={**CRITICAL, "o2_pct": 87.5, "temp_c": -20})
    out = rig.say("hi")
    assert out["route"] == "alert" and out["blocked_text"] is None and out["verifier"]["ok"]
    assert out["spoken"] == "Critical alert. Suit temperature -20.0 degrees Celsius. Return to the airlock now."
    assert rig.router.seen == [] and rig.llm.calls == []


def test_two_critical_alerts_with_a_negative_value_are_both_spoken():
    rig = Rig(snapshot={**CRITICAL, "temp_c": -20})  # CRITICAL already has o2 at 14.2
    out = rig.say("hi")
    assert out["blocked_text"] is None and out["verifier"]["ok"]
    assert "Oxygen at 14.2 percent" in out["spoken"] and "Suit temperature -20.0 degrees Celsius" in out["spoken"]


def test_hard_tool_cap_applies_to_the_constructor():
    for bad in (0, 9, -1):
        with pytest.raises(ValueError, match="max_tools"):
            ToolRegistry(max_tools=bad)
    assert ToolRegistry(max_tools=MAX_TOOLS).max_tools == 8 and ToolRegistry(max_tools=1).max_tools == 1


def test_turn_policy_ships_as_package_data():
    from importlib.resources import files

    assert json.loads(files("router").joinpath("turn_policy.json").read_text())["severity_tiers"]
    pyproject = (__import__("pathlib").Path(__file__).parent.parent / "pyproject.toml").read_text()
    assert '"turn_policy.json"' in pyproject


@pytest.mark.parametrize("tier", ["critical", "warning"])
def test_loader_refuses_a_tier_that_preempts_the_router_and_uses_the_llm(tier, tmp_path):
    policy = copy.deepcopy(load_policy())
    policy["severity_tiers"][tier].update(preempts_router=True, uses_llm=True)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="cannot use the LLM"):
        load_policy(path)
    policy["severity_tiers"][tier].update(preempts_router=False)  # an LLM tier that does not preempt is not refused
    path.write_text(json.dumps(policy))
    assert load_policy(path)


# ---- 5b. arbiter ----
def test_arbiter_agree_executes_the_tool():
    arbiter = FakeArbiter("call_o2")
    rig = Rig("call_o2", 0.7, arbiter=arbiter)
    out = rig.say("oxygen")
    assert out["route"] == "tool" and out["pipeline"]["gate"] == "arbiter_agree" and len(arbiter.calls) == 1
    assert rig.registry.log == [("o2", {})]


@pytest.mark.parametrize("answer", ["call_data", "none", RuntimeError("down")], ids=["disagree", "none", "error"])
def test_arbiter_disagree_none_or_error_gives_a_clarification_and_no_tool(answer):
    rig = Rig("call_o2", 0.7, arbiter=FakeArbiter(answer))
    out = rig.say("oxygen")
    assert out["route"] == "clarify" and out["spoken"].startswith("Please choose:") and rig.registry.log == []
    assert rig.llm.calls == []


def test_no_arbiter_gives_a_clarification_and_no_tool():
    rig = Rig("call_o2", 0.7)
    out = rig.say("oxygen")
    assert out["route"] == "clarify" and out["pipeline"]["reason"] == "no_arbiter" and rig.registry.log == []


# ---- 5. policy ----
def test_every_policy_section_is_labelled_placeholder():
    policy = load_policy()
    sections = [k for k in policy if not k.startswith("_")]
    assert sections and all(policy[k]["status"].startswith("PLACEHOLDER") for k in sections)
    assert "PLACEHOLDER" in policy["_comment"]


def test_load_policy_refuses_an_unlabelled_section(tmp_path):
    policy = load_policy()
    policy["templates"]["status"] = "approved"
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="PLACEHOLDER"):
        load_policy(path)


def test_every_critical_field_has_an_alert_line_and_every_tool_a_template():
    policy = load_policy()
    fields = [k for k in policy["sim_fields"] if k != "status"]
    assert set(fields) <= set(policy["alert_lines"])
    assert {"safety", "navigation", "map", "data", "o2"} <= set(policy["templates"])


# ---- A. return-to-home feasibility (outpost: 1500 m -> 25 min out at 60 m/min, 25 back, 10 reserve, 60 required) ----
def nav(minutes, text="take me to the outpost", **kwargs):
    rig = Rig("call_navigation", minutes_remaining=minutes, **kwargs)
    return rig, rig.say(text)


def feasibility(out):
    return out["tool_result"]["data"]["feasibility"]


@pytest.mark.parametrize("minutes, outcome", [(212, "comfortable"), (60, "comfortable"), (59, "tight"), (50, "tight"),
                                              (49, "cannot_return"), (25, "cannot_return"), (24, "cannot_reach"), (0, "cannot_reach")])
def test_feasibility_outcome_boundaries_for_a_round_trip(minutes, outcome):
    rig, out = nav(minutes)
    assert feasibility(out) == {"outcome": outcome, "is_home": False, "outbound_min": 25, "return_min": 25, "trip_min": 50,
                                "reserve_min": 10, "required_min": 60, "o2_minutes_remaining": minutes, "margin_min": minutes - 60}
    assert out["tool_calls"] == 1 and out["tool_call"] == {"name": "navigation", "args": {"place": "outpost"}}


@pytest.mark.parametrize("minutes, tail", [
    (212, "Oxygen check: the round trip needs about 60 minutes including a 10 minute reserve, and you have 212."),
    (55, "Warning, oxygen margin is tight: 25 minutes out, 25 minutes back and a 10 minute reserve need 60 minutes, and you have 55. "
         "Do you want me to call outpost and check if they have oxygen refills?"),
    (45, "Warning, on your current oxygen you cannot get back home without a refill: 25 minutes out plus 25 minutes back is 50, "
         "and you have 45. Do you want me to call outpost and check if they have oxygen refills?"),
    (20, "Warning, you cannot reach it on your current oxygen: the trip out takes 25 minutes and you have 20. "
         "Do you want me to call outpost and check if they have oxygen refills?"),
])
def test_feasibility_line_is_spoken_after_the_route_line_with_numbers_from_the_tool_result(minutes, tail):
    rig, out = nav(minutes)
    expected = load_policy()['clarification']['guidance'].format(**out['tool_result']['data'])
    budget = load_policy()['clarification']['budget'].format(**feasibility(out))
    if ' Do you want' in tail:
        warning, offer = tail.split(' Do you want', 1)
        tail = warning + budget + ' Do you want' + offer
    else:
        tail += budget
    assert out['spoken'] == expected + ' ' + tail
    assert out["verifier"]["ok"] and out["blocked_text"] is None
    spoken = set(out["verifier"]["spoken"])
    assert spoken <= {1500.0, 25.0, 50.0, 10.0, 60.0, float(minutes), float(minutes - 60)}


def test_the_route_is_never_refused_even_when_the_destination_cannot_be_reached():
    rig, out = nav(5)
    assert feasibility(out)["outcome"] == "cannot_reach" and out["route"] == "tool"
    assert rig.registry.log == [("navigation", {"place": "outpost"})] and out["spoken"].startswith("Navigation information for outpost:")


@pytest.mark.parametrize("minutes, outcome, tail", [
    (212, "comfortable", "Oxygen check: the way home needs about 12 minutes including a 10 minute reserve, and you have 212."),
    (12, "comfortable", "Oxygen check: the way home needs about 12 minutes including a 10 minute reserve, and you have 12."),
    (11, "tight", "Warning, oxygen margin is tight: the way home takes 2 minutes plus a 10 minute reserve, 12 in all, and you have 11."),
    (1, "cannot_reach", "Warning, you cannot reach home on your current oxygen: the way home takes 2 minutes and you have 1."),
])
def test_home_as_destination_skips_the_return_leg_and_never_says_cannot_return(minutes, outcome, tail):
    rig, out = nav(minutes, text="take me home")
    assert feasibility(out) == {"outcome": outcome, "is_home": True, "outbound_min": 2, "return_min": 0, "trip_min": 2,
                                "reserve_min": 10, "required_min": 12, "o2_minutes_remaining": minutes, "margin_min": minutes - 12}
    assert tail in out["spoken"] and out["verifier"]["ok"]
    assert "refills" not in out["spoken"] and out["pending_confirmation"] is None and rig.handler.pending_confirmation is None


def test_o2_minutes_fall_back_to_the_percent_rate_and_no_o2_data_means_no_feasibility_claim():
    policy = load_policy()
    data = {"place": "outpost", "distance_m": 1500}
    fallback = assess_feasibility(data, {"o2_pct": 20.0}, policy)  # 20 * 2.4 placeholder minutes per percent
    assert fallback["o2_minutes_remaining"] == 48 and fallback["outcome"] == "cannot_return"
    assert assess_feasibility(data, {"bpm": 72}, policy) is None
    assert assess_feasibility(data, {"o2_pct": 20.0, "minutes_remaining": 300}, policy)["o2_minutes_remaining"] == 300


def test_without_o2_data_navigation_explicitly_reports_unavailable_and_no_offer_is_pending():
    rig = Rig("call_navigation", snapshot={"bpm": 72})
    out = rig.say("take me to the outpost")
    assert out["spoken"].endswith(load_policy()["clarification"]["unavailable"])
    assert out["tool_result"]["data"]["feasibility"]["outcome"] == "unavailable" and rig.handler.pending_confirmation is None


def test_legs_are_rounded_up_and_remaining_down_so_the_check_is_conservative():
    policy = load_policy()
    result = assess_feasibility({"place": "airlock", "distance_m": 80}, {"minutes_remaining": 13.9}, policy)
    assert (result["outbound_min"], result["return_min"], result["o2_minutes_remaining"], result["outcome"]) == (2, 2, 13, "tight")


def test_a_feasibility_line_with_a_number_not_in_the_tool_result_is_blocked_and_no_offer_is_stored():
    policy = copy.deepcopy(load_policy())
    policy["feasibility"]["lines"]["tight"] = "Warning, tight: {required_min} needed, 99 available."
    rig, out = nav(55, policy=policy)
    assert out["spoken"] == policy["templates"]["verifier_blocked"] and out["verifier"]["unsupported"] == [99.0]
    assert rig.handler.pending_confirmation is None and out["pending_confirmation"] is None


def test_comfortable_navigation_and_non_navigation_tools_store_no_offer():
    rig, out = nav(212)
    assert rig.handler.pending_confirmation is None and out["pending_confirmation"] is None
    rig = Rig("call_o2", minutes_remaining=5)
    assert "feasibility" not in rig.say("oxygen")["tool_result"]["data"] and rig.handler.pending_confirmation is None


# ---- B. refill-call offer: stateful confirmation ----
@pytest.mark.parametrize("minutes, urgency", [(55, "normal"), (45, "urgent"), (20, "urgent")])
def test_yes_makes_exactly_one_send_message_with_urgency_from_the_feasibility_outcome(minutes, urgency):
    rig, offer = nav(minutes)
    assert offer["pending_confirmation"]["kind"] == "refill_call" and rig.handler.pending_confirmation is not None
    seen_before = list(rig.router.seen)
    out = rig.say("yes please")
    message = (f"Nova relay: an astronaut heading to outpost has {minutes} minutes of oxygen and the trip needs 60. "
               "Please confirm whether oxygen refills are available.")
    assert rig.registry.log[1:] == [("send_message", {"recipient": "outpost", "urgency": urgency, "message": message})]
    assert out["route"] == "confirmed_yes" and out["tool_calls"] == 1 and out["tool_call"]["args"]["urgency"] == urgency
    assert out["spoken"] == f"Okay, I sent the refill request to outpost, marked {urgency}." and out["verifier"]["ok"]
    assert rig.router.seen == seen_before and rig.llm.calls == [] and out["pipeline"] is None and not out["llm_called"]
    assert rig.handler.pending_confirmation is None


def test_confirming_recomputes_feasibility_from_the_current_snapshot():
    rig, offer = nav(55)  # outpost: tight, "normal" urgency at offer time
    assert offer["pending_confirmation"]["outcome"] == "tight" and offer["pending_confirmation"]["distance_m"] == 1500
    rig.sim.values["minutes_remaining"] = 20  # oxygen dropped while the offer was pending: cannot_reach now
    out = rig.say("yes")
    message = ("Nova relay: an astronaut heading to outpost has 20 minutes of oxygen and the trip needs 60. "
               "Please confirm whether oxygen refills are available.")
    assert rig.registry.log[1:] == [("send_message", {"recipient": "outpost", "urgency": "urgent", "message": message})]
    assert out["route"] == "confirmed_yes" and out["spoken"] == "Okay, I sent the refill request to outpost, marked urgent."
    assert "55" not in out["tool_call"]["args"]["message"]


@pytest.mark.parametrize("override", [{"minutes_remaining": 212}, {"minutes_remaining": 60}], ids=["recovered", "boundary"])
def test_confirming_when_no_refill_is_needed_any_more_sends_nothing_and_speaks_the_fixed_line(override):
    rig, _ = nav(55)
    rig.sim.values.update(override)
    out = rig.say("yes")
    assert out["route"] == "refill_not_needed" and out["spoken"] == load_policy()["confirmation"]["no_longer_needed"]
    assert out["tool_calls"] == 0 and out["tool_call"] is None and len(rig.registry.log) == 1
    assert out["verifier"]["ok"] and rig.handler.pending_confirmation is None and rig.llm.calls == []


def test_confirming_without_o2_data_any_more_sends_nothing():
    rig, _ = nav(55)
    rig.snapshot = {"bpm": 72}
    out = rig.say("yes")
    assert out["route"] == "refill_not_needed" and len(rig.registry.log) == 1


def test_send_message_urgency_comes_from_the_policy_mapping_not_from_the_llm_or_the_utterance():
    policy = copy.deepcopy(load_policy())
    policy["send_message"]["urgency_by_outcome"]["tight"] = "urgent"
    rig, _ = nav(55, policy=policy)
    out = rig.say("yes, and mark it low priority")  # extra words are not in the vocabulary -> NOT a yes
    assert out["route"] != "confirmed_yes"
    rig, _ = nav(55, policy=policy)
    out = rig.say("yes")
    assert out["tool_call"]["args"]["urgency"] == "urgent" and rig.llm.calls == []


def test_no_acknowledges_clears_and_calls_no_tool():
    rig, _ = nav(45)
    out = rig.say("no thanks")
    assert out["route"] == "confirmed_no" and out["spoken"] == "Okay, I won't call. Say so if you change your mind."
    assert out["tool_calls"] == 0 and len(rig.registry.log) == 1 and rig.handler.pending_confirmation is None
    assert rig.say("yes")["route"] != "confirmed_yes"  # nothing pending any more


@pytest.mark.parametrize("text", ["take me to the airlock", "no oxygen left", "yes no", "what time is it", "hmm"])
def test_anything_else_clears_the_pending_offer_and_routes_normally(text):
    rig, _ = nav(45)
    out = rig.say(text)
    assert out["route"] not in ("confirmed_yes", "confirmed_no") and rig.router.seen[-1] == text
    assert [name for name, _ in rig.registry.log].count("send_message") == 0
    assert out["pipeline"] is not None and rig.handler.pending_confirmation is None
    assert rig.say("yes")["route"] not in ("confirmed_yes", "confirmed_no")


def test_a_new_offer_replaces_the_pending_one():
    rig, _ = nav(30)
    rig.say("take me to the quarry")
    out = rig.say("yes")
    assert out["tool_call"]["args"]["recipient"] == "quarry"


@pytest.mark.parametrize("elapsed, answered", [(0, True), (1, True), (2, False)])
def test_pending_offer_expires_after_the_policy_turn_count(elapsed, answered):
    rig, _ = nav(45)
    rig.handler.turn_index += elapsed  # extra turns that passed unseen; the offer answers on its next 2 turns (this one counts)
    assert (rig.say("yes")["route"] == "confirmed_yes") is answered


def test_zero_expiry_policy_drops_the_offer_immediately():
    policy = copy.deepcopy(load_policy())
    policy["confirmation"]["expiry_turns"] = 0
    rig, _ = nav(45, policy=policy)
    assert rig.say("yes")["route"] != "confirmed_yes"


def test_critical_alert_preempts_and_clears_a_pending_offer():
    rig, _ = nav(45)
    rig.snapshot = CRITICAL
    out = rig.say("yes")
    assert out["route"] == "alert" and out["tool_calls"] == 0 and rig.handler.pending_confirmation is None
    rig.snapshot = None
    assert rig.say("yes")["route"] != "confirmed_yes" and len(rig.registry.log) == 1


def test_warning_alert_does_not_clear_the_offer():
    rig, _ = nav(45)
    rig.snapshot = {**rig.sim.snapshot(), "temp_c": 33.0}
    assert rig.say("yes")["route"] == "confirmed_yes"


def test_a_message_with_a_number_not_from_the_feasibility_data_is_not_sent():
    policy = copy.deepcopy(load_policy())
    policy["send_message"]["message"] = "Need oxygen, 99 minutes left at {place_name}."
    rig, _ = nav(45, policy=policy)
    out = rig.say("yes")
    assert out["spoken"] == policy["templates"]["verifier_blocked"] and len(rig.registry.log) == 1 and out["tool_calls"] == 0


# ---- C. tool / LLM / retrieval failures are spoken, never raised (an alert_source failure still propagates) ----
class Boom(Exception):
    pass


def break_tool(rig, name):
    schema, _ = rig.registry._tools[name]

    def boom(**args):
        raise Boom("secret internals")
    rig.registry._tools[name] = (schema, boom)


def test_a_raising_tool_speaks_the_tool_error_line_and_is_not_retried():
    rig = Rig("call_o2")
    break_tool(rig, "o2")
    out = rig.say("how much oxygen")
    assert out["route"] == "tool" and out["spoken"] == load_policy()["templates"]["tool_error"] and out["verifier"]["ok"]
    assert out["tool_error"].startswith("Boom") and out["tool_calls"] == 1 and out["tool_result"] is None
    assert rig.say("how much oxygen")["tool_calls"] == 1  # the next turn is a fresh single call


def test_a_raising_send_message_on_confirmation_speaks_the_tool_error_line():
    rig, _ = nav(45)
    break_tool(rig, "send_message")
    out = rig.say("yes")
    assert out["route"] == "confirmed_yes" and out["spoken"] == load_policy()["templates"]["tool_error"]
    assert out["tool_calls"] == 1 and "tool_error" in out and rig.handler.pending_confirmation is None


class RaisingLLM(FakeLLM):
    def __call__(self, text, prompt_policy, passages=None):
        super().__call__(text, prompt_policy, passages)
        raise Boom("model offline")


def test_a_raising_llm_speaks_the_tool_error_line_for_chitchat_and_call_llm():
    expected = load_policy()["templates"]["tool_error"]
    chat = Rig("NO_TOOL", 0.99, llm=RaisingLLM())
    out = chat.say("hello there")
    assert out["route"] == "llm" and out["spoken"] == expected and out["llm_error"].startswith("Boom") and len(chat.llm.calls) == 1
    grounded = Rig("call_llm", 0.97, llm=RaisingLLM(), retriever=lambda t: [PASSAGE])
    out = grounded.say("how far away is the moon")
    assert out["route"] == "llm" and out["spoken"] == expected and out["llm_called"] and len(grounded.llm.calls) == 1


@pytest.mark.parametrize("retriever", [
    lambda t: None,
    lambda t: iter([]),
    lambda t: [{"text": "x", "score": "high"}],
    lambda t: [{"text": "x", "score": None}],
    lambda t: [{"text": "x", "score": float("nan")}],
    lambda t: [{"text": "x", "score": True}],
    lambda t: [{"text": "x"}],
    lambda t: [{"text": 5, "score": 0.9}],
    lambda t: [PASSAGE, {"text": "x", "score": "high"}],
    lambda t: 5,
], ids=["none", "empty-iterator", "string-score", "none-score", "nan-score", "bool-score", "no-score", "non-string-text", "one-bad-passage", "not-iterable"])
def test_malformed_retrieval_output_means_no_document(retriever):
    rig = Rig("call_llm", 0.97, retriever=retriever)
    out = rig.say("how far away is the moon")
    assert out["route"] == "no_document" and rig.llm.calls == []


def test_retrieval_accepts_a_generator_and_materializes_it_once():
    rig = Rig("call_llm", 0.97, llm=FakeLLM("Moon facts."), retriever=lambda t: (p for p in [PASSAGE]))
    out = rig.say("tell me about the moon")
    assert out["route"] == "llm" and rig.llm.passages_seen == [[PASSAGE["text"]]]


def test_alert_source_failure_still_propagates_with_a_working_tool_layer():
    handler = TurnHandler(FakeRouter(), lambda: 1 / 0, default_registry())
    with pytest.raises(ZeroDivisionError):
        handler.handle("hi")


@pytest.mark.parametrize("text, answer", [
    ("yes", "yes"), ("Yes!", "yes"), ("yeah sure", "yes"), ("okay, go ahead", "yes"), ("yes please call them", "yes"),
    ("no", "no"), ("No thanks.", "no"), ("nope", "no"), ("don't", "no"), ("never mind", "no"), ("not now", "no"),
    ("yes no", None), ("no problem yes", None), ("yes take me to the airlock", None), ("", None), ("maybe", None),
    ("no oxygen", None), ("go", None),
])
def test_match_confirmation_word_lists(text, answer):
    assert match_confirmation(text, load_policy()) == answer


# ---- C. LLM scope: router class + retrieval score + risk terms ----
@pytest.mark.parametrize("text", ["how is my oxygen", "tell me a joke about my suit", "is this Radiation bad", "what dose is safe",
                                  "my battery is dying lol", "any emergency", "o2 chat", "suit pressure"])
def test_chitchat_with_a_risk_term_gets_the_template_and_the_llm_is_not_called(text):
    rig = Rig("NO_TOOL", 0.97)
    out = rig.say(text)
    assert out["route"] in ("risk_diverted", "repair_offer") and out["spoken"] == load_policy()["llm_scope"]["speaking_scope"]["repair_offer"]
    assert rig.llm.calls == [] and not out["llm_called"] and rig.registry.log == [] and out["verifier"]["ok"]


@pytest.mark.parametrize("text", ["hello there", "tell me a joke", "thanks, nice work"])
def test_chitchat_without_a_risk_term_still_reaches_the_llm_stub(text):
    rig = Rig("NO_TOOL", 0.97, llm=FakeLLM("Glad to help!"))
    out = rig.say(text)
    assert out["route"] == "llm" and out["spoken"] == "Glad to help!" and len(rig.llm.calls) == 1


def test_chitchat_never_consults_the_retriever():
    calls = []
    rig = Rig("NO_TOOL", 0.97, retriever=lambda text: calls.append(text) or [PASSAGE])
    rig.say("hello")
    assert calls == []


def test_call_llm_without_a_retriever_says_no_document_and_does_not_call_the_llm():
    rig = Rig("call_llm", 0.97)
    out = rig.say("how far away is the moon")
    assert out["route"] == "no_document" and out["spoken"] == "I don't have a document for that yet."
    assert rig.llm.calls == [] and not out["llm_called"] and out["verifier"]["ok"]


@pytest.mark.parametrize("retriever", [lambda t: [], lambda t: [{"text": "weak", "score": 0.49}], lambda t: 1 / 0], ids=["empty", "below-threshold", "raises"])
def test_call_llm_with_no_usable_passage_says_no_document(retriever):
    rig = Rig("call_llm", 0.97, retriever=retriever)
    out = rig.say("how far away is the moon")
    assert out["route"] == "no_document" and rig.llm.calls == []


def test_call_llm_with_passages_calls_the_llm_with_them_best_first_and_top_k_only():
    passages = [{"text": f"p{i}", "score": s} for i, s in enumerate([0.6, 0.9, 0.2, 0.7, 0.8])]
    rig = Rig("call_llm", 0.97, llm=FakeLLM("Moon facts."), retriever=lambda text: passages)
    out = rig.say("tell me about the moon")
    assert out["route"] == "llm" and out["llm_called"] and out["spoken"] == "Moon facts."
    assert rig.llm.passages_seen == [["p1", "p4", "p3"]] and [p["text"] for p in out["passages"]] == ["p1", "p4", "p3"]


def test_call_llm_output_may_use_numbers_that_are_in_the_retrieved_passages():
    rig = Rig("call_llm", 0.97, llm=FakeLLM("The Moon is about 384400 kilometers away."), retriever=lambda t: [PASSAGE])
    out = rig.say("how far away is the moon")
    assert out["spoken"] == "According to the document, the Moon is about 384400 kilometers away." and out["verifier"]["ok"]


def test_call_llm_output_with_a_number_missing_from_the_passages_is_blocked():
    rig = Rig("call_llm", 0.97, llm=FakeLLM("The Moon is 400000 kilometers away."), retriever=lambda t: [PASSAGE])
    out = rig.say("how far away is the moon")
    assert out["spoken"] == load_policy()["templates"]["verifier_blocked"] and out["verifier"]["unsupported"] == [400000.0]


def test_a_number_only_in_a_below_threshold_passage_is_not_a_source():
    weak = {"text": "Mars is 227900000 kilometers from the Sun.", "score": 0.1}
    rig = Rig("call_llm", 0.97, llm=FakeLLM("Mars is 227900000 kilometers away."), retriever=lambda t: [PASSAGE, weak])
    assert rig.say("how far is mars")["blocked_text"] is not None


def test_risk_terms_do_not_divert_call_llm_which_is_scoped_by_retrieval_instead():
    rig = Rig("call_llm", 0.97, llm=FakeLLM("Per the document."), retriever=lambda t: [PASSAGE])
    out = rig.say("what is the oxygen pressure rule")
    assert out["route"] == "llm" and out["spoken"] == "Per the document."


def test_ood_and_low_confidence_never_reach_the_retriever_or_llm():
    calls = []
    for label, conf in (("OOD", 0.97), ("call_llm", 0.3)):
        rig = Rig(label, conf, retriever=lambda text: calls.append(text) or [PASSAGE])
        rig.say("how far away is the moon")
        assert rig.llm.calls == []
    assert calls == []


def test_new_policy_sections_exist():  # that every section is labelled PLACEHOLDER is pinned by the loader test above
    assert {"feasibility", "confirmation", "send_message", "llm_scope"} <= set(load_policy())

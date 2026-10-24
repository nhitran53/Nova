"""Pipeline tests with a fake router (no model needed)."""
import pytest

from router import load_labels
from router.pipeline import Pipeline, load_config

LABELS = load_labels(__import__("pathlib").Path(__file__).parent.parent / "labels.json")


class FakeRouter:
    def __init__(self, label, confidence=0.99, others=("call_data", "NO_TOOL")):
        self.label, self.confidence = label, confidence
        self.topk = [{"label": label, "p": confidence}] + [{"label": o, "p": (1 - confidence) / 2} for o in others]
        self.seen = []

    def predict(self, text):
        self.seen.append(text)
        return {"label": self.label, "confidence": self.confidence, "topk": self.topk}


class FakeArbiter:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def __call__(self, text, topk):
        self.calls.append((text, topk))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def run(label, text, confidence=0.99, **kwargs):
    return Pipeline(FakeRouter(label, confidence), **kwargs).handle(text)


def test_config_routes_every_label_exactly_once():
    config = load_config(labels=LABELS)
    assert set(config["tool_labels"]) | set(config["llm_labels"]) == set(LABELS)


def test_navigation_resolves_place():
    out = run("call_navigation", "take me to home")
    assert out == {"route": "tool", "tool": "navigation", "args": {"place": "homebase"}, "confidence": 0.99,
                   "label": "call_navigation", "gate": "tau_hi"}


def test_map_resolves_place():
    assert run("call_map", "zoom in on the airlock")["args"] == {"place": "airlock"}


@pytest.mark.parametrize("label, tool", [("call_navigation", "navigation"), ("call_map", "map")])
@pytest.mark.parametrize("text", ["take me there", "zoom out", "show me the way"])
def test_required_place_missing_is_a_clarify_route_never_a_tool_route(label, tool, text):
    out = run(label, text)
    assert out["route"] == "clarify" and out["reason"] == "missing_slot"
    assert out["tool"] == tool and out["missing"] == ["place"] and out["question"] is None


def test_missing_slot_also_blocks_the_arbiter_agree_path():
    out = run("call_navigation", "take me there", confidence=0.7, arbiter_fn=FakeArbiter("call_navigation"))
    assert out["route"] == "clarify" and out["reason"] == "missing_slot"


@pytest.mark.parametrize("label, tool", [("call_safety", "safety"), ("call_data", "data"), ("call_o2", "o2")])
def test_other_tools_have_empty_args_even_with_a_place_in_text(label, tool):
    out = run(label, "check things at home base")
    assert out["route"] == "tool" and out["tool"] == tool and out["args"] == {}


@pytest.mark.parametrize("label", ["NO_TOOL", "call_llm", "OOD"])
def test_llm_labels_route_to_llm_with_distinct_policies(label):
    out = run(label, "hi")
    assert out["route"] == "llm" and out["reason"] == label and "tool" not in out
    assert out["response"].startswith("[LLM placeholder]")


def test_prompt_policies_are_distinct_per_class_and_ood_is_out_of_scope():
    policies = {label: run(label, "x")["prompt_policy"] for label in ("NO_TOOL", "call_llm", "OOD")}
    policies["low"] = run("call_o2", "x", confidence=0.1)["prompt_policy"]
    assert len(set(policies.values())) == 4
    assert "out of scope" in policies["OOD"]


def test_low_confidence_goes_to_llm_even_for_tool_label():
    out = run("call_navigation", "take me home", confidence=0.3, tau_lo=0.5)
    assert out["route"] == "llm" and out["reason"] == "low_confidence" and out["label"] == "call_navigation"


# ---- two-threshold confidence gate (review finding 1): tau_hi=0.9, tau_lo=0.5 from pipeline_config.json ----
def test_shipped_thresholds_are_tau_hi_090_tau_lo_050():
    config = load_config()
    assert (config["tau_hi"], config["tau_lo"]) == (0.9, 0.5)


def gate(confidence, arbiter=None, label="call_o2", **kwargs):
    return Pipeline(FakeRouter(label, confidence), arbiter_fn=arbiter, **kwargs).handle("check my oxygen at the airlock")


@pytest.mark.parametrize("confidence", [0.9, 0.95, 1.0])
def test_at_or_above_tau_hi_executes_without_asking_the_arbiter(confidence):
    arbiter = FakeArbiter("none")
    out = gate(confidence, arbiter)
    assert out["route"] == "tool" and out["gate"] == "tau_hi" and arbiter.calls == []


@pytest.mark.parametrize("confidence", [0.8999999, 0.7, 0.5])
def test_middle_band_includes_exactly_tau_lo_and_stops_just_below_tau_hi(confidence):
    arbiter = FakeArbiter("call_o2")
    out = gate(confidence, arbiter)
    assert out["route"] == "tool" and out["gate"] == "arbiter_agree" and len(arbiter.calls) == 1


def test_just_below_tau_lo_goes_to_llm_clarify_policy_and_skips_arbiter():
    arbiter = FakeArbiter("call_o2")
    out = gate(0.4999999, arbiter)
    assert out["route"] == "llm" and out["reason"] == "low_confidence" and arbiter.calls == []


def test_arbiter_receives_text_and_the_routers_topk():
    arbiter = FakeArbiter("call_o2")
    Pipeline(FakeRouter("call_o2", 0.7), arbiter_fn=arbiter).handle("check my oxygen")
    text, topk = arbiter.calls[0]
    assert text == "check my oxygen"
    assert [t["label"] for t in topk] == ["call_o2", "call_data", "NO_TOOL"] and len(topk) == 3


@pytest.mark.parametrize("answer", ["call_data", "none", "NO_TOOL", "something_else", None, ""])
def test_arbiter_disagreement_clarifies_and_never_executes_any_tool(answer):
    out = gate(0.7, FakeArbiter(answer))
    assert out["route"] == "clarify" and out["reason"] == "arbiter_disagree" and "tool" not in out
    assert out["options"] == ["call_o2", "call_data"]  # only tool classes from the router's top-k
    assert out["question"] == "I'm not sure I understood. Did you want your oxygen status or a sensor readout?"


def test_arbiter_naming_another_tool_class_does_not_run_that_tool():
    out = gate(0.7, FakeArbiter("call_data"))
    assert out["route"] == "clarify" and "tool" not in out and out["label"] == "call_o2"


def test_no_arbiter_supplied_is_safe_default_clarify_never_execute():
    out = gate(0.7)
    assert out["route"] == "clarify" and out["reason"] == "no_arbiter" and "tool" not in out


def test_arbiter_exception_clarifies_instead_of_executing():
    out = gate(0.7, FakeArbiter(RuntimeError("timeout")))
    assert out["route"] == "clarify" and out["reason"] == "arbiter_error"


@pytest.mark.parametrize("label", ["call_safety", "call_navigation", "call_map", "call_data", "call_o2"])
def test_every_tool_class_is_gated(label):
    assert gate(0.7, FakeArbiter("none"), label=label)["route"] == "clarify"
    assert gate(0.95, label=label)["route"] == "tool"


@pytest.mark.parametrize("label", ["NO_TOOL", "call_llm", "OOD"])
def test_non_tool_top_classes_keep_llm_routing_and_never_call_arbiter(label):
    arbiter = FakeArbiter("none")
    for confidence in (0.99, 0.7, 0.5):
        out = gate(confidence, arbiter, label=label)
        assert out["route"] == "llm" and out["reason"] == label
    assert gate(0.3, arbiter, label=label)["reason"] == "low_confidence"
    assert arbiter.calls == []


def test_ood_never_reaches_a_tool_even_at_full_confidence():
    out = gate(1.0, FakeArbiter("call_o2"), label="OOD")
    assert out["route"] == "llm" and out["prompt_policy"].startswith("out_of_scope")


def test_thresholds_are_configurable_and_validated():
    assert gate(0.7, label="call_o2", tau_hi=0.6)["route"] == "tool"
    assert gate(0.4, FakeArbiter("call_o2"), tau_lo=0.3)["gate"] == "arbiter_agree"
    with pytest.raises(ValueError, match="tau_lo <= tau_hi"):
        Pipeline(FakeRouter("call_o2"), tau_hi=0.4, tau_lo=0.6)


def test_router_without_topk_still_clarifies_with_the_top_class():
    class Bare:
        def predict(self, text):
            return {"label": "call_o2", "confidence": 0.7}

    out = Pipeline(Bare()).handle("x")
    assert out["route"] == "clarify" and out["options"] == ["call_o2"]


# ---- destination-aware place slots ----
def test_origin_is_not_the_destination_in_the_pipeline():
    out = run("call_navigation", "go from the rover to the airlock")
    assert out["route"] == "tool" and out["args"] == {"place": "airlock"}


def test_negated_place_is_not_the_destination_in_the_pipeline():
    assert run("call_navigation", "take me to the rover not home")["args"] == {"place": "rover"}


def test_ambiguous_places_emit_slot_question_and_no_tool_call():
    out = run("call_navigation", "go to the rover or the airlock")
    assert out["route"] == "clarify" and out["reason"] == "ambiguous_place" and "tool" not in out
    assert out["question"] == "Which place do you mean: rover or airlock?"
    assert out["options"] == ["rover", "airlock"]
    out = run("call_map", "zoom in on the ridge and the crater and the airlock")
    assert out["question"] == "Which place do you mean: ridge, crater, or airlock?"


def test_ambiguity_also_applies_after_arbiter_agreement():
    out = Pipeline(FakeRouter("call_navigation", 0.7), arbiter_fn=FakeArbiter("call_navigation")).handle(
        "go to the rover or the airlock")
    assert out["route"] == "clarify" and out["reason"] == "ambiguous_place"


def test_place_resolver_not_run_for_non_place_tools():
    out = run("call_data", "go to the rover or the airlock")
    assert out["route"] == "tool" and out["args"] == {}


def test_llm_fn_is_injected_and_gets_text_and_policy():
    calls = []

    def llm_fn(text, policy):
        calls.append((text, policy))
        return "canned"

    out = run("OOD", "what's for dinner", llm_fn=llm_fn)
    assert out["response"] == "canned" and calls == [("what's for dinner", out["prompt_policy"])]


def test_tool_label_set_comes_from_config_not_code():
    config = load_config()
    config = {**config, "tool_labels": {}, "llm_labels": {**config["llm_labels"], **{
        k: "p" for k in config["tool_labels"]}}}
    assert run("call_o2", "x", config=config)["route"] == "llm"


def test_arbiter_only_receives_top_three_even_if_router_returns_more():
    router = FakeRouter("call_o2", 0.7, others=("call_data", "call_navigation", "call_map"))
    arbiter = FakeArbiter("none")
    Pipeline(router, arbiter_fn=arbiter).handle("check oxygen")
    assert arbiter.calls[0][1] == router.topk[:3]

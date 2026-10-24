"""Pipeline tests with a fake router (no model needed)."""
import pytest

from router import load_labels
from router.pipeline import Pipeline, load_config

LABELS = load_labels(__import__("pathlib").Path(__file__).parent.parent / "labels.json")


class FakeRouter:
    def __init__(self, label, confidence=0.99):
        self.label, self.confidence = label, confidence
        self.seen = []

    def predict(self, text):
        self.seen.append(text)
        return {"label": self.label, "confidence": self.confidence}


def run(label, text, confidence=0.99, **kwargs):
    return Pipeline(FakeRouter(label, confidence), **kwargs).handle(text)


def test_config_routes_every_label_exactly_once():
    config = load_config(labels=LABELS)
    assert set(config["tool_labels"]) | set(config["llm_labels"]) == set(LABELS)


def test_navigation_resolves_place():
    out = run("call_navigation", "take me to home")
    assert out == {"route": "tool", "tool": "navigation", "args": {"place": "homebase"}, "confidence": 0.99,
                   "label": "call_navigation"}


def test_map_resolves_place_and_omits_it_when_absent():
    assert run("call_map", "zoom in on the airlock")["args"] == {"place": "airlock"}
    assert run("call_map", "zoom out")["args"] == {}


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


def test_tau_lo_is_configurable_and_boundary_is_inclusive():
    assert run("call_o2", "x", confidence=0.4, tau_lo=0.3)["route"] == "tool"
    assert run("call_o2", "x", confidence=0.5, tau_lo=0.5)["route"] == "tool"
    assert run("call_o2", "x", confidence=0.49, tau_lo=0.5)["route"] == "llm"


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

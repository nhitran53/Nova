"""Owner-requested speaking scope, without a model/server or router substitution in production."""
import json

import pytest

from router.generation import FillerGenerator, build_request, postprocess, render_prompt
from router.tools import SimState, default_registry
from router.turn import TurnHandler, load_policy
from router.tests.test_turn import FakeRouter


def handler(label="call_llm", confidence=0.99, reply="Cloud droplets scatter light.", retriever=None):
    sim = SimState()
    generator = FillerGenerator(script=[json.dumps({"refusal": None, "text": reply})])
    h = TurnHandler(FakeRouter(label, confidence), sim.snapshot, default_registry(sim),
                    generator=generator, retriever=retriever)
    return h, generator


@pytest.mark.parametrize("text", ["Hi!", "who are you", "what is your name"])
def test_hi_is_exact_user_wording_without_generator_and_still_verified(text):
    h, generator = handler("NO_TOOL")
    result = h.handle(text)
    assert result["spoken"] == "Hey, I'm Nova, your navigation and observation voice assistant."
    assert result["route"] == "greeting" and result["verifier"]["ok"]
    assert not generator.calls and result["tool_calls"] == 0
    assert h.handle("thanks")["llm_called"]  # not every conversational turn is a greeting


@pytest.mark.parametrize("text", ["why is the sky grey", "why is the sky grey on Mars", "why is the sky grey on mar"])
def test_general_explanation_can_answer_without_document_but_has_no_numeric_provenance(text):
    h, generator = handler()
    result = h.handle(text)
    assert result["spoken"] == "Cloud droplets scatter light."
    req = generator.calls[0]
    assert req.kind == "general_explanation" and req.provenance_mode == "general_knowledge"
    assert req.passage_numbers == "off" and req.style.no_digits_in_prose
    assert not req.passages and not req.allowed_numbers and not req.facts and not req.slots
    assert result["generation"]["request"]["provenance_mode"] == "general_knowledge"
    assert load_policy()["llm_scope"]["speaking_scope"]["general_prompt"] in render_prompt(req)[0]["content"]
    assert result["tool_calls"] == 0 and result["verifier"]["ok"]


@pytest.mark.parametrize("text,violation", [
    ("Clouds contain 20 droplets.", "digit_in_prose"),
    ("Clouds contain five droplets.", "number_word_in_prose"),
    ("Your oxygen is normal.", "live_reading_claim"),
    ("I called base.", "action_claim"),
    ("You should seal it.", "directive_verb"),
])
def test_general_explanation_reuses_number_action_live_and_directive_gates(text, violation):
    policy = load_policy()
    req = build_request(turn_id=1, kind="general_explanation", route="llm", user_text="why is the sky grey", policy=policy)
    result = postprocess(req, json.dumps({"refusal": None, "text": text}), policy["generation"])
    assert not result.ok and violation in result.violations


@pytest.mark.parametrize("label", ["NO_TOOL", "call_llm"])
@pytest.mark.parametrize("text", ["how can i fix this", "fix the leak", "repair my battery", "it is broken"])
def test_repair_never_synthesizes_instructions_or_arms_contact(label, text):
    h, generator = handler(label, reply="Invented repair instructions.", retriever=lambda _: [{"text": "Unrelated facts", "score": 0.99}])
    result = h.handle(text)
    assert result["route"] == "repair_offer"
    assert result["spoken"] == load_policy()["llm_scope"]["speaking_scope"]["repair_offer"]
    assert result["verifier"]["ok"] and result["tool_calls"] == 0 and not generator.calls
    assert h.pending_confirmation is None
    assert h.handle("yes")["tool_calls"] == 0


@pytest.mark.parametrize("text", ["hi", "why is the sky grey", "how can i fix this"])
@pytest.mark.parametrize("label,confidence,route", [("OOD", 0.99, "ood"), ("call_llm", 0.2, "low_confidence")])
def test_scope_does_not_bypass_router_gates(text, label, confidence, route):
    h, generator = handler(label, confidence)
    result = h.handle(text)
    assert result["route"] == route and not generator.calls and result["tool_calls"] == 0


def test_general_prefers_available_document_and_risky_questions_stay_retrieval_only():
    h, generator = handler(retriever=lambda _: [{"text": "Clouds scatter light.", "score": 0.9}])
    h.handle("why is the sky grey")
    assert generator.calls[0].provenance_mode == "retrieved_passages"
    h, generator = handler()
    assert h.handle("why is my suit pressure low")["route"] == "risk_diverted"
    assert not generator.calls


def test_scope_never_overrides_selected_tool_or_critical_alert():
    h, generator = handler("call_o2")
    result = h.handle("how can i fix this")
    assert result["tool_call"]["name"] == "o2" and result["tool_calls"] == 1 and not generator.calls
    sim = SimState(o2_pct=10)
    h.alert_source = sim.snapshot
    result = h.handle("hi")
    assert result["route"] == "alert" and not generator.calls and result["tool_calls"] == 0


@pytest.mark.parametrize("text", ["can i make it to base 2", "can i make it to base 3", "can I get there", "is it safe at the tower"])
def test_destination_safety_restricts_generic_go_without_calling_any_tool(text):
    h, generator = handler("call_safety")
    result = h.handle(text)
    assert result["route"] == "destination_safety" and result["tool_calls"] == 0
    assert result["spoken"] == load_policy()["llm_scope"]["speaking_scope"]["destination_safety"]
    assert result["verifier"]["ok"] and not generator.calls and h.pending_confirmation is None


@pytest.mark.parametrize("text", [
    "Check my safety; no trip is planned yet; is it safe at base 2?",
    "Check my safety; no trip is planned yet; is it safe returning home?",
    "no problem, I want to go to base 2",
    "Check my safety for the trip to base 2",
    "is it safe to go home",
])
def test_destination_safety_guard_preserves_requests_after_non_requesting_spans(text):
    h, generator = handler("call_safety")
    result = h.handle(text)
    assert result["route"] == "destination_safety" and result["tool_calls"] == 0
    assert result["tool_call"] is None
    assert result["verifier"]["ok"] and not generator.calls


def test_generic_safety_still_uses_existing_tool_and_wording():
    h, generator = handler("call_safety")
    result = h.handle("am i safe")
    assert result["route"] == "tool" and result["tool_calls"] == 1
    assert result["spoken"] == "Safety check: go." and not generator.calls


@pytest.mark.parametrize("text", ["is it safe to disconnect this", "how do I bypass this", "why is my oxygen low"])
def test_consequential_no_source_offers_base_without_claiming_source_or_contact(text):
    h, generator = handler()
    result = h.handle(text)
    assert result["route"] in ("repair_offer", "risk_diverted") and not generator.calls
    assert result["spoken"] == load_policy()["llm_scope"]["speaking_scope"]["repair_offer"]
    assert result["tool_calls"] == 0 and h.pending_confirmation is None


def test_general_answer_refusal_contract_is_explicit_and_refusal_still_wins():
    policy = load_policy()
    req = build_request(turn_id=1, kind='general_explanation', route='llm',
                        user_text='why is the sky grey', policy=policy)
    prompt = render_prompt(req)[0]['content']
    assert 'Empty passages are expected' in prompt and 'refusal:null' in prompt
    outcome = postprocess(req, json.dumps({'refusal': 'no_info', 'text': 'Clouds scatter light.'}),
                          policy['generation'])
    assert outcome.refusal == 'no_info' and outcome.text != 'Clouds scatter light.'


@pytest.mark.parametrize("mention", [
    "no trip is planned", "not going anywhere", "without a trip",
    "I am not planning a trip", "the trip is not planned", "the trip isn't planned",
    "I don't plan a trip", "no travel is planned", "not planning to travel",
    "not planning to walk", "check the suit strips",
])
def test_non_requesting_travel_mentions_do_not_veto_generic_safety(mention):
    h, generator = handler("call_safety", confidence=0.9998)
    result = h.handle(f"Check the basic oxygen-pressure condition; {mention}.")
    assert result["route"] == "tool" and result["tool_calls"] == 1
    assert result["tool_call"]["name"] == "safety"
    assert result["spoken"] == "Safety check: go."
    assert result["verifier"]["ok"] and not generator.calls


@pytest.mark.parametrize("mention", [
    "a trip is planned", "can I make it there", "can I travel", "can I walk there",
    "no trip is planned, but can I reach the destination",
    "without a trip planned, can I get there",
    "not going anywhere now, but is it safe at the tower",
    "no trip is planned; is it safe to go to base b",
])
def test_affirmative_travel_or_destination_still_vetoes_generic_safety(mention):
    h, generator = handler("call_safety", confidence=0.9998)
    result = h.handle(f"Check the basic oxygen-pressure condition; {mention}.")
    assert result["route"] == "destination_safety" and result["tool_calls"] == 0
    assert result["verifier"]["ok"] and not generator.calls

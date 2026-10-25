"""Six-class routing, LLM self-check and voice session contract (no model/server required)."""
import json
from pathlib import Path

import pytest

from router.core import load_labels
from router.generation import FillerGenerator
from router.pipeline import Pipeline, load_config
from router.tools import SimState, default_registry
from router.turn import TurnHandler, load_policy


class SixClassRouter:
    def __init__(self, label):
        self.label = label

    def predict(self, text):
        return {"label": self.label, "confidence": 0.99}


class CheckingGenerator(FillerGenerator):
    def __init__(self, decision, reply=None):
        super().__init__(script=[json.dumps({"refusal": None, "text": reply})] if reply else None)
        self.decision = decision
        self.checks = []

    def decide(self, req, tools):
        self.checks.append((req, tools))
        if isinstance(self.decision, Exception):
            raise self.decision
        return self.decision


def handler(label="llm", generator=None, **kwargs):
    sim = SimState()
    return TurnHandler(SixClassRouter(label), sim.snapshot, default_registry(sim), generator=generator, **kwargs)


def test_labels_and_config_are_six_class():
    labels = load_labels(Path(__file__).parents[1] / "labels.json")
    assert labels == ["call_safety", "call_navigation", "call_map", "call_data", "call_o2", "llm"]
    load_config(labels=labels)


def test_misc_sky_query_calls_generator_and_answers_grey_without_interim():
    gen = CheckingGenerator({"decision": "none"})
    interim = []
    turn = handler(generator=gen, interim_fn=lambda *args: interim.append(args)).handle("what color is the sky")
    assert turn["llm_called"] and len(gen.checks) == 1 and len(gen.calls) == 1
    assert turn["spoken"].lower().rstrip(".") == "grey"
    assert interim == []
    assert not turn["expect_followup"] and turn["listen_timeout_s"] is None
    assert turn["end_session"]


def test_llm_checks_for_missed_tool_then_dispatches_without_generation():
    gen = CheckingGenerator({"decision": "tool", "tool": "o2"})
    turn = handler(generator=gen).handle("give me my live air reading")
    assert turn["tool_call"]["name"] == "o2" and turn["tool_calls"] == 1
    assert len(gen.checks) == 1 and gen.calls == []
    assert not turn["expect_followup"] and turn["listen_timeout_s"] is None


def test_bert_selected_tool_dispatches_without_llm_or_keyword_selector():
    gen = CheckingGenerator(AssertionError("must not check"))
    turn = handler("call_o2", gen).handle("read that please")
    assert turn["tool_calls"] == 1 and not turn["llm_called"]
    assert gen.checks == [] and not turn["expect_followup"]


@pytest.mark.parametrize("text", ["read that please", "show map and oxygen"])
def test_llm_predictions_use_self_check_even_with_tool_words(text):
    gen = CheckingGenerator({"decision": "none"})
    handler("llm", gen).handle(text)
    assert len(gen.checks) == 1


@pytest.mark.parametrize("text", ["how do I fix a suit leak", "is this safe"])
def test_unknown_safety_never_attempts_generation(text):
    gen = CheckingGenerator(AssertionError("must not check"))
    turn = handler(generator=gen).handle(text)
    assert turn["route"] in {"risk_diverted", "repair_offer"} and not turn["llm_called"]
    assert gen.calls == [] and gen.checks == [] and turn["tool_calls"] == 0
    assert turn["end_session"] and not turn["expect_followup"]


@pytest.mark.parametrize("decision", [{"decision": "tool", "tool": "unknown"}, {"decision": "invalid"}, ValueError("bad JSON")])
def test_self_check_failure_falls_through_to_generation(decision):
    gen = CheckingGenerator(decision, "grey")
    turn = handler(generator=gen).handle("what color is the sky")
    assert turn["route"] == "llm" and turn["tool_calls"] == 0
    assert turn["self_check_error"] and len(gen.calls) == 1
    assert turn["spoken"] == "grey"


def test_unknown_safety_self_check_cannot_generate_an_answer():
    gen = CheckingGenerator({"decision": "none"})
    turn = handler(generator=gen).handle("what medicine should I use")
    assert turn["route"] == "risk_diverted" and gen.calls == [] and len(gen.checks) == 1


@pytest.mark.parametrize("text", ["oxygen status", "show map and oxygen"])
def test_self_check_recovers_risk_term_reading_tool(text):
    gen = CheckingGenerator({"decision": "tool", "tool": "o2"})
    turn = handler("llm", gen).handle(text)
    assert turn["tool_call"]["name"] == "o2" and gen.calls == [] and len(gen.checks) == 1


def test_llm_question_keeps_normal_followup_window():
    gen = CheckingGenerator({"decision": "none"}, "What would you like to know?")
    turn = handler(generator=gen).handle("tell me something")
    assert turn["expect_followup"] and turn["listen_timeout_s"] == 10.0


def test_missing_navigation_slot_asks_without_calling_tool():
    turn = handler("call_navigation").handle("navigate please")
    assert turn["route"] == "slot_question" and turn["tool_calls"] == 0
    assert turn["expect_followup"]


def test_existing_quick_response_text_is_preserved():
    policy = load_policy()
    turn = handler(policy=policy).handle("hi")
    assert turn["spoken"] == policy["llm_scope"]["speaking_scope"]["greeting"]
    assert not turn["llm_called"] and not turn["expect_followup"]


def test_http_self_check_then_generation_uses_mars_context(llm_server):
    from conftest import chat_completion, reply_json
    from router.llm_generator import HTTPChatGenerator, load_live_llm_config

    def reply(body):
        if 'Available tool schemas:' in body['messages'][0]['content']:
            return chat_completion(json.dumps({'decision': 'none'}))
        assert 'Mars' in body['messages'][0]['content'] and 'grey' in body['messages'][0]['content']
        return chat_completion(reply_json('grey'))

    llm_server.reply = reply
    gen = HTTPChatGenerator(load_live_llm_config(base_url=llm_server.base_url))
    turn = handler(generator=gen).handle('what color is the sky')
    assert turn['spoken'].lower().rstrip('.') == 'grey'
    assert len(llm_server.requests) == 2
    assert 'json_schema' not in llm_server.requests[0]
    assert llm_server.requests[0]['temperature'] == 0


def test_http_self_check_malformed_json_falls_through_to_generation(llm_server):
    from conftest import chat_completion, reply_json
    from router.llm_generator import HTTPChatGenerator, load_live_llm_config

    llm_server.reply = lambda body: chat_completion('not JSON' if 'Available tool schemas:' in body['messages'][0]['content'] else reply_json('grey'))
    gen = HTTPChatGenerator(load_live_llm_config(base_url=llm_server.base_url))
    turn = handler(generator=gen).handle('do something')
    assert turn['route'] == 'llm' and turn['tool_calls'] == 0
    assert turn['self_check_error'] and turn['spoken'] == 'grey'
    assert len(llm_server.requests) == 2


def test_converse_tool_closes_without_listening_again(monkeypatch):
    from router import stt

    timeouts = []
    def listen(**kwargs):
        timeouts.append(kwargs['timeout_s'])
        return {'transcript': 'oxygen', 'turn': {'spoken': 'reading', 'route': 'tool', 'expect_followup': False}}
    monkeypatch.setattr(stt, 'listen_and_route', listen)
    assert len(stt.converse(speak=False, router_loader=lambda: None, on_event=lambda *args: None)) == 1
    assert timeouts == [10.0]


@pytest.mark.parametrize('question,timeout', [(True, 10.0)])
def test_converse_honors_turn_window_then_returns_on_silence(monkeypatch, question, timeout):
    from router import stt

    timeouts = []
    def listen(**kwargs):
        timeouts.append(kwargs['timeout_s'])
        if len(timeouts) > 1:
            return {'transcript': '', 'turn': None}
        return {'transcript': 'hello', 'turn': {'spoken': 'What next?' if question else 'grey', 'route': 'llm',
                'expect_followup': True, 'listen_timeout_s': timeout}}
    monkeypatch.setattr(stt, 'listen_and_route', listen)
    turns = stt.converse(speak=False, router_loader=lambda: None, on_event=lambda *args: None)
    assert len(turns) == 1 and timeouts == [10.0, timeout]


def test_self_check_free_text_never_becomes_user_answer():
    gen = CheckingGenerator({'decision': 'none', 'text': 'Do an unsafe invented action.'}, 'grey')
    turn = handler(generator=gen).handle('what color is the sky')
    assert turn['spoken'].lower().rstrip('.') == 'grey' and len(gen.calls) == 1


def test_critical_alert_still_preempts_llm_generation():
    snapshots = iter([SimState().snapshot(), SimState(o2_pct=10).snapshot()])
    gen = CheckingGenerator({'decision': 'none'}, 'grey')
    turn = TurnHandler(SixClassRouter('llm'), lambda: next(snapshots), generator=gen).handle('sky color')
    assert turn['route'] == 'alert' and not turn['expect_followup']


def test_navigation_slot_question_can_be_answered_without_new_wakeword():
    h = handler('call_navigation')
    assert h.handle('navigate please')['expect_followup']
    turn = h.handle('home')
    assert turn['tool_call']['name'] == 'navigation'
    assert turn['tool_calls'] == 1


@pytest.mark.parametrize('text', ['is this safe', 'how do I fix a suit leak'])
def test_safety_with_matching_document_is_answered(text):
    gen = CheckingGenerator(AssertionError('document needs no tool check'), 'Consult the manual.')
    turn = handler(generator=gen, retriever=lambda _: [{'text': 'Consult the manual.', 'score': 1.0}]).handle(text)
    assert turn['route'] == 'llm' and turn['llm_called']
    assert len(gen.calls) == 1 and gen.checks == []
    assert turn['end_session'] and not turn['expect_followup']


@pytest.mark.parametrize("text", ["is it safe to eat the algae", "is it safe to drink recycled water"])
def test_llm_food_and_water_safety_is_refused(text):
    gen = CheckingGenerator(AssertionError("must refuse before self-check"))
    turn = handler("llm", gen).handle(text)
    assert turn["route"] == "risk_diverted" and turn["tool_calls"] == 0
    assert turn["spoken"] == load_policy()["llm_scope"]["speaking_scope"]["unknown_safety_refusal"]
    assert turn["spoken"] != "Safety check: go."
    assert gen.calls == [] and gen.checks == []


def test_bert_safety_label_still_dispatches_safety():
    gen = CheckingGenerator(AssertionError("BERT tool stays trusted"))
    turn = handler("call_safety", gen).handle("is it safe to go outside")
    assert turn["tool_call"]["name"] == "safety" and turn["tool_calls"] == 1
    assert turn["spoken"] == "Safety check: go."
    assert gen.calls == [] and gen.checks == []


def test_http_self_check_fenced_json_parses(llm_server):
    from conftest import chat_completion
    from router.llm_generator import HTTPChatGenerator, load_live_llm_config

    llm_server.reply = chat_completion('```json\n{"decision":"tool","tool":"o2"}\n```')
    gen = HTTPChatGenerator(load_live_llm_config(base_url=llm_server.base_url))
    turn = handler(generator=gen).handle("give me my live air reading")
    assert turn["tool_call"]["name"] == "o2" and "self_check_error" not in turn
    assert llm_server.requests[0]["response_format"] == {"type": "json_object"}
    assert "only the suit/EVA go/no-go" in llm_server.requests[0]["messages"][0]["content"]


def test_low_confidence_tool_uses_self_check_before_recovery():
    gen = CheckingGenerator({"decision": "tool", "tool": "data"})
    h = handler("call_o2", gen)
    h.pipeline.router.predict = lambda text: {"label": "call_o2", "confidence": 0.3}
    turn = h.handle("read my sensors")
    assert len(gen.checks) == 1 and gen.calls == []
    assert turn["tool_call"]["name"] == "data"
    assert turn["pipeline"]["gate"] == "llm_self_check"


@pytest.mark.parametrize("label,tool,text", [
    ("call_safety", "safety", "check status"),
    ("call_navigation", "navigation", "home"),
    ("call_map", "map", "home"),
    ("call_data", "data", "read that"),
    ("call_o2", "o2", "read that"),
])
def test_every_bert_tool_label_is_trusted_without_self_check(label, tool, text):
    gen = CheckingGenerator(AssertionError("must not check"))
    turn = handler(label, gen).handle(text)
    assert turn["tool_call"]["name"] == tool and turn["tool_calls"] == 1
    assert not turn["llm_called"] and gen.checks == [] and gen.calls == []
    assert turn["end_session"]

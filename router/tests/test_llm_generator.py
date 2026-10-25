"""router/llm_generator.py against a fake OpenAI-compatible server on 127.0.0.1. No real model, no other network."""
import json
import socket
import time
import traceback

import pytest
from conftest import chat_completion, reply_json

from router.generation import FillerGenerator, build_request, output_gbnf, output_json_schema, render_prompt, run_generation
from router.llm_generator import (HTTPChatGenerator, LLMRequestError, LLMTimeoutError, api_base, check_server, load_live_llm_config,
                                  redact_secrets, strip_reasoning, validate_live_llm_config)
from router.tools import SimState, default_registry
from router.turn import TurnHandler, load_policy

POLICY = load_policy()
MOON = {"text": "The Moon is Earth's only natural satellite.", "score": 0.9}


def request(kind="chitchat", **overrides):
    kwargs = dict(turn_id=1, kind=kind, route="llm", user_text="hello there", policy=POLICY)
    kwargs.update(overrides)
    return build_request(**kwargs)


def o2_request():
    return build_request(turn_id=1, kind="tool_result", route="tool", user_text="oxygen?", policy=POLICY, tool="o2",
                         tool_result=default_registry(SimState()).call("o2", {}))


def make(server, **overrides):
    overrides = {"runtime": "llama-server", "model": "qwen3.5-4b", "constrain": "json_schema", **overrides}  # the policy default is Gemini
    return HTTPChatGenerator(load_live_llm_config(base_url=server.base_url, **overrides))


def closed_port_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


# ---- config ----
def test_live_llm_section_is_placeholder_and_loads_with_the_policy():
    assert POLICY["live_llm"]["status"].startswith("PLACEHOLDER")
    config = load_live_llm_config(runtime="llama-server", base_url="http://localhost:8080/v1", constrain="json_schema")
    assert config["disable_thinking"] is True and config["temperature"] == 0


def test_env_then_arguments_override_the_policy(monkeypatch):
    monkeypatch.setenv("NOVA_LLM_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("NOVA_LLM_RUNTIME", "ollama")
    config = load_live_llm_config(model="qwen3.5:4b", runtime=None)
    assert (config["base_url"], config["runtime"], config["model"]) == ("http://127.0.0.1:9999/v1", "ollama", "qwen3.5:4b")
    assert load_live_llm_config(runtime="mlx")["runtime"] == "mlx"  # an explicit argument beats the env


@pytest.mark.parametrize("change", [{"runtime": "vllm"}, {"constrain": "regex"}, {"base_url": "file:///etc/passwd"}, {"model": " "},
                                    {"request_timeout_cap_s": 0}, {"temperature": -1}, {"disable_thinking": "yes"}, {"extra_body": []},
                                    {"runtime": "ollama", "constrain": "gbnf"}])
def test_invalid_config_is_refused(change):
    with pytest.raises(ValueError):
        validate_live_llm_config({**load_live_llm_config(), **change})


def test_unknown_override_key_is_refused():
    with pytest.raises(ValueError, match="unknown live_llm override"):
        load_live_llm_config(colour="red")


def test_api_base_accepts_url_variants():
    assert api_base("http://localhost:8080") == api_base("http://localhost:8080/") == api_base("http://localhost:8080/v1/") == "http://localhost:8080/v1"


# ---- request body per runtime ----
def test_llama_server_body_has_thinking_off_json_schema_seed_and_prompt(llm_server):
    gen, req = make(llm_server), request()
    result = gen.generate(req)
    body = llm_server.requests[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["json_schema"] == json.loads(json.dumps(output_json_schema(req))) and "grammar" not in body and "response_format" not in body
    assert body["messages"] == render_prompt(req) and body["max_tokens"] == req.style.max_tokens == POLICY["generation"]["max_tokens"]
    assert body["seed"] == req.seed == 0 and body["temperature"] == 0 and body["stream"] is False and body["model"] == "qwen3.5-4b"
    assert result.raw == reply_json("Hello, happy to help.") and result.ttft_ms is None and result.latency_ms >= 0


def test_llama_server_gbnf_mode_sends_a_grammar_instead(llm_server):
    req = request()
    make(llm_server, constrain="gbnf").generate(req)
    body = llm_server.requests[0]
    assert body["grammar"] == output_gbnf(req) and "json_schema" not in body and body["chat_template_kwargs"] == {"enable_thinking": False}


def test_llama_server_constrain_none_and_thinking_flag_off(llm_server):
    make(llm_server, constrain="none", disable_thinking=False).generate(request())
    body = llm_server.requests[0]
    assert not {"json_schema", "grammar", "response_format", "chat_template_kwargs", "reasoning_effort"} & set(body)


def test_ollama_body_uses_response_format_and_reasoning_effort(llm_server):
    req = request()
    make(llm_server, runtime="ollama", model="qwen3.5:4b").generate(req)
    body = llm_server.requests[0]
    assert body["reasoning_effort"] == "none" and body["model"] == "qwen3.5:4b" and body["seed"] == 0
    assert body["response_format"] == {"type": "json_schema", "json_schema": {"name": "nova_reply", "schema": json.loads(json.dumps(output_json_schema(req)))}}
    assert not {"chat_template_kwargs", "json_schema", "grammar"} & set(body)


def test_mlx_body_has_no_constraint_no_thinking_field_and_no_seed(llm_server):
    gen = make(llm_server, runtime="mlx", model="mlx-community/Qwen3.5-4B-4bit")  # constrain stays json_schema in the config
    gen.generate(request())
    body = llm_server.requests[0]
    assert gen.effective_constrain == "none" and "launch-time" in gen.describe()
    assert not {"seed", "json_schema", "grammar", "response_format", "chat_template_kwargs", "reasoning_effort"} & set(body)
    assert body["temperature"] == 0 and body["max_tokens"] == POLICY["generation"]["max_tokens"]


def test_openai_runtime_sends_response_format_only(llm_server):
    make(llm_server, runtime="openai").generate(request())
    body = llm_server.requests[0]
    assert "response_format" in body and not {"chat_template_kwargs", "reasoning_effort", "json_schema"} & set(body)


def test_extra_body_is_merged_last_and_wins(llm_server):
    make(llm_server, extra_body={"top_p": 0.8, "temperature": 0.3}).generate(request())
    body = llm_server.requests[0]
    assert body["top_p"] == 0.8 and body["temperature"] == 0.3


def test_api_key_is_sent_only_when_configured(llm_server, monkeypatch):
    make(llm_server).generate(request())
    assert "Authorization" not in llm_server.headers[0]
    HTTPChatGenerator(load_live_llm_config(base_url=llm_server.base_url), api_key="secret-token").generate(request())
    assert llm_server.headers[1]["Authorization"] == "Bearer secret-token"


def test_grounded_and_tool_requests_render_the_same_prompt_the_generation_layer_builds(llm_server):
    req = request("grounded_answer", user_text="how far is the moon", passages=[MOON])
    make(llm_server).generate(req)
    assert llm_server.requests[0]["messages"] == render_prompt(req)
    assert "Moon is Earth" in llm_server.requests[0]["messages"][1]["content"]
    make(llm_server).generate(o2_request())
    assert "{o2_pct}" in llm_server.requests[1]["messages"][1]["content"]


# ---- reply handling ----
def test_think_block_is_stripped_and_reasoning_content_never_returned(llm_server):
    llm_server.reply = chat_completion("<think>\nthe user said hello\n</think>\n" + reply_json("Hi."), reasoning_content="secret reasoning")
    assert make(llm_server).generate(request()).raw == reply_json("Hi.")
    llm_server.reply = chat_completion(reply_json("Hi."), reasoning_content="more reasoning")
    assert make(llm_server).generate(request()).raw == reply_json("Hi.")
    llm_server.reply = chat_completion("", reasoning_content='{"refusal": null, "text": "from reasoning"}')
    assert make(llm_server).generate(request()).raw == ""  # reasoning is not a fallback source


@pytest.mark.parametrize("content, expected", [
    ("<think>a</think>OK", "OK"), ("<think>a</think>\n<think>b</think> OK ", "OK"), ("thoughts</think>OK", "OK"),
    ("OK<think>cut off by max_tokens", "OK"), ("<think>never closed", ""), ("plain", "plain")])
def test_strip_reasoning(content, expected):
    assert strip_reasoning(content) == expected


# ---- failures raise ----
@pytest.mark.parametrize("reply", [
    (500, {"error": "boom"}), (404, "nope"), (200, "not json at all"), (200, {"choices": []}), (200, {"nothing": 1}),
    (200, {"choices": [{"message": {"content": None}}]}), (200, {"choices": [{"message": {"content": ["x"]}}]}), (200, "[1, 2]")])
def test_http_error_and_malformed_responses_raise(llm_server, reply):
    llm_server.reply = reply
    with pytest.raises(LLMRequestError):
        make(llm_server).generate(request())


KEY = "SENTINEL-KEY-12345"


def test_http_error_message_omits_the_response_body_and_the_key(llm_server, monkeypatch):
    monkeypatch.setenv("NOVA_LLM_API_KEY", KEY)
    llm_server.reply = (401, f"bad key Bearer {KEY} and some other body text")
    with pytest.raises(LLMRequestError) as caught:
        make(llm_server).generate(request())
    message = str(caught.value)
    assert "HTTP 401" in message and KEY not in message and "bad key" not in message and "other body text" not in message
    assert KEY not in repr(caught.value) and caught.value.__cause__ is None


def test_key_in_the_url_is_redacted_from_connection_errors():
    url = closed_port_url().replace("/v1", f"/{KEY}/v1")
    with pytest.raises(LLMRequestError, match="cannot reach") as caught:
        HTTPChatGenerator(load_live_llm_config(base_url=url), api_key=KEY).generate(request())
    assert KEY not in str(caught.value) and "[redacted]" in str(caught.value) and caught.value.__cause__ is None


def test_unchanged_message_still_drops_the_chained_cause_with_the_secret():
    class RaisingOpener:
        def open(self, http_request, timeout=None):
            raise TimeoutError(f"Bearer {KEY}")

    generator = HTTPChatGenerator(load_live_llm_config(base_url=closed_port_url()), api_key=None, opener=RaisingOpener())
    with pytest.raises(LLMTimeoutError) as caught:
        generator.generate(request())
    trace = "".join(traceback.format_exception(caught.value))
    assert KEY not in trace and caught.value.__cause__ is None and caught.value.__context__ is None


def test_describe_redacts_the_key_from_url_and_model():
    url = closed_port_url().replace("//", f"//user:{KEY}@")
    description = HTTPChatGenerator(load_live_llm_config(base_url=url, model=f"m-{KEY}"), api_key=KEY).describe()
    assert KEY not in description and "[redacted]" in description


def test_redact_secrets_removes_the_key_in_any_form_and_bearer_tokens():
    assert redact_secrets(f"x {KEY} y", KEY) == "x [redacted] y"
    assert redact_secrets("k=a%2Fb%20c", "a/b c") == "k=[redacted]"
    assert redact_secrets("Authorization: Bearer abc.def-1 end") == "Authorization: Bearer [redacted] end"
    assert redact_secrets("bearer  tok") == "Bearer [redacted]" and redact_secrets("plain text", None) == "plain text"


def test_connection_refused_raises():
    with pytest.raises(LLMRequestError, match="cannot reach"):
        HTTPChatGenerator(load_live_llm_config(base_url=closed_port_url())).generate(request())


def test_slow_server_times_out_at_the_config_cap(llm_server):
    llm_server.delay_s = 1.0
    started = time.monotonic()
    with pytest.raises(LLMTimeoutError):
        make(llm_server, request_timeout_cap_s=0.2).generate(request())
    assert time.monotonic() - started < 0.9


def test_timeout_is_the_smaller_of_cap_and_remaining_deadline(llm_server):
    gen = make(llm_server, request_timeout_cap_s=10)
    assert gen.request_timeout_s(request()) == POLICY["generation"]["deadline_ms"] / 1000
    from dataclasses import replace
    assert gen.request_timeout_s(replace(request(), deadline_ms=300)) == 0.3
    assert make(llm_server, request_timeout_cap_s=0.1).request_timeout_s(request()) == 0.1
    with pytest.raises(LLMTimeoutError):
        gen.request_timeout_s(replace(request(), deadline_ms=0))


def test_request_deadline_bounds_a_slow_call(llm_server):
    from dataclasses import replace
    llm_server.delay_s = 1.0
    started = time.monotonic()
    with pytest.raises(LLMTimeoutError):
        make(llm_server).generate(replace(request(), deadline_ms=200))
    assert time.monotonic() - started < 0.9


# ---- check_server ----
def test_check_server_ok_via_health(llm_server):
    ok, message = check_server(llm_server.base_url, 1.0)
    assert ok and "reachable" in message and "/health ok" in message


def test_check_server_falls_back_to_models_when_there_is_no_health(llm_server):
    llm_server.health = None
    ok, message = check_server(llm_server.base_url.removesuffix("/v1"), 1.0)  # a bare host:port works too
    assert ok and "fake-qwen" in message


def test_check_server_fails_for_loading_unrelated_and_closed(llm_server):
    llm_server.health = (503, {"error": "loading"})
    ok, message = check_server(llm_server.base_url, 1.0)
    assert not ok and "loading" in message
    llm_server.health = llm_server.models = None
    ok, message = check_server(llm_server.base_url, 1.0)
    assert not ok and "OpenAI-compatible" in message
    url = closed_port_url()
    ok, message = check_server(url, 0.5)
    assert not ok and f"no LLM server reachable at {url}" in message


def test_check_server_messages_never_carry_the_key_or_a_response_body(llm_server, monkeypatch):
    monkeypatch.setenv("NOVA_LLM_API_KEY", KEY)
    llm_server.health = llm_server.models = (401, f"bad key Bearer {KEY}")
    ok, message = check_server(llm_server.base_url, 1.0)
    assert not ok and "HTTP 401" in message and KEY not in message and "bad key" not in message
    url = closed_port_url().replace("/v1", f"/{KEY}/v1")  # a key embedded in the URL is redacted too
    ok, message = check_server(url, 0.5)
    assert not ok and "no LLM server reachable" in message and KEY not in message
    monkeypatch.delenv("NOVA_LLM_API_KEY")
    assert KEY not in check_server(url, 0.5, api_key=KEY)[1]  # an explicit key works without the environment


# ---- through run_generation / TurnHandler ----
class FakeRouter:
    def __init__(self, label="NO_TOOL", confidence=0.97):
        self.label, self.confidence = label, confidence

    def predict(self, text):
        return {"label": self.label, "confidence": self.confidence,
                "topk": [{"label": self.label, "p": self.confidence}, {"label": "call_data", "p": 0.02}, {"label": "OOD", "p": 0.01}]}


def handler_for(server, label="NO_TOOL", retriever=None, **overrides):
    sim = SimState()
    generator = make(server, **overrides)
    return TurnHandler(FakeRouter(label), sim.snapshot, default_registry(sim), generator=generator, retriever=retriever), generator


def test_chitchat_turn_through_the_server_is_verified_and_spoken(llm_server):
    handler, generator = handler_for(llm_server)
    out = handler.handle("hello there")
    assert out["route"] == "llm" and out["llm_called"] and out["spoken"] == "Hello, happy to help." and out["blocked_text"] is None
    assert out["verifier"]["ok"] and out["generation"]["status"] == "ok" and out["generation"]["attempt_log"][0]["raw"] == reply_json("Hello, happy to help.")
    assert llm_server.requests[0]["messages"] == out["generation"]["prompt"] and generator.calls == 1


def test_bad_number_reply_is_blocked_retried_and_falls_back(llm_server):
    llm_server.reply = chat_completion(reply_json("It is 384400 kilometers."))
    handler, _ = handler_for(llm_server)
    out = handler.handle("hello there")
    assert out["spoken"] == POLICY["templates"]["verifier_blocked"] and out["blocked_text"] == "It is 384400 kilometers."
    assert out["generation"]["status"] == "fallback" and out["generation"]["attempts"] == 2 and "digit_in_prose" in out["generation"]["violations"]
    assert len(llm_server.requests) == 2 and "RETRY" in llm_server.requests[1]["messages"][1]["content"]  # the corrective retry reached the server


def test_grounded_answer_with_a_passage_number_is_verified_against_the_passage(llm_server):
    llm_server.reply = chat_completion(reply_json("The airlock cycle takes 12 minutes."))
    handler, _ = handler_for(llm_server, "call_llm", retriever=lambda text: [{"text": "The airlock cycle takes 12 minutes.", "score": 0.9}])
    out = handler.handle("how long is the airlock cycle")
    assert out["spoken"] == "According to the document, the airlock cycle takes 12 minutes."
    assert out["verifier"]["ok"] and out["blocked_text"] is None
    llm_server.reply = chat_completion(reply_json("The airlock cycle takes 99 minutes."))
    out = handler.handle("how long is the airlock cycle")
    assert out["blocked_text"] and out["spoken"] == POLICY["templates"]["verifier_blocked"]


def test_server_error_becomes_llm_error_and_the_tool_error_line(llm_server):
    llm_server.reply = (500, {"error": "boom"})
    handler, _ = handler_for(llm_server)
    out = handler.handle("hello there")
    assert out["spoken"] == POLICY["templates"]["tool_error"] and "HTTP 500" in out["llm_error"] and len(llm_server.requests) == 1  # never retried


def test_unreachable_server_is_the_error_line_too():
    sim = SimState()
    handler = TurnHandler(FakeRouter(), sim.snapshot, default_registry(sim), generator=HTTPChatGenerator(load_live_llm_config(base_url=closed_port_url())))
    out = handler.handle("hello there")
    assert out["spoken"] == POLICY["templates"]["tool_error"] and out["llm_error"].startswith("LLMRequestError")


def test_slow_server_turn_ends_inside_the_deadline_with_the_error_line(llm_server):
    llm_server.delay_s = 3.0
    handler, _ = handler_for(llm_server)
    handler.policy = {**handler.policy, "generation": {**handler.policy["generation"], "deadline_ms": 250}}
    started = time.monotonic()
    out = handler.handle("hello there")
    assert time.monotonic() - started < 2.0
    assert out["spoken"] == POLICY["templates"]["tool_error"] and out["llm_error"].startswith("LLMTimeoutError")


def test_tool_turns_never_call_the_server(llm_server):
    handler, _ = handler_for(llm_server, "call_o2")
    out = handler.handle("how much oxygen do I have left")
    assert out["route"] == "tool" and out["llm_called"] is False and llm_server.requests == []


def test_run_generation_accepts_the_adapter_directly(llm_server):
    outcome = run_generation(request(), make(llm_server), POLICY, "fallback line")
    assert outcome.status == "ok" and outcome.text == "Hello, happy to help." and outcome.latency_ms >= 0 and outcome.wall_ms >= 0
    assert FillerGenerator().generate(request()).raw  # the offline stand-in still works next to it

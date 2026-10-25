"""Gemini runtime: request shape, key handling, failover. Fake opener only; no network."""
import io
import json
import urllib.error

import pytest

from router import generation as G
from router import llm_generator as L

KEY = "test-key-NOT-REAL-123"
POLICY = json.loads(L.POLICY_PATH.read_text())


class FakeOpener:
    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def open(self, request, timeout=None):
        self.requests.append((request, timeout))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": reply}}]}).encode())


def _config(**kw):
    cfg = L.load_live_llm_config(env={})
    cfg.update(kw)
    return cfg


def _req():
    return G.build_request(turn_id=1, kind="chitchat", route="llm", user_text="how are you", policy=POLICY)


def _http_error(code):
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(b"body"))


def test_policy_defaults_to_gemini_flash_lite():
    cfg = _config()
    assert cfg["runtime"] == "gemini" and cfg["model"] == "gemini-3.5-flash-lite" and cfg["constrain"] == "none"


def test_url_keeps_openai_compat_root():
    assert L.api_base("https://generativelanguage.googleapis.com/v1beta/openai") == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert L.HTTPChatGenerator(_config(), api_key=KEY).url.endswith("/v1beta/openai/chat/completions")


def test_body_has_no_seed_or_schema_and_minimal_reasoning():
    gen = L.HTTPChatGenerator(_config(), api_key=KEY, opener=FakeOpener('{"refusal": null, "text": "ok"}'))
    gen.generate(_req())
    assert "seed" not in gen.last_body and "response_format" not in gen.last_body and "json_schema" not in gen.last_body
    assert gen.last_body["reasoning_effort"] == "minimal" and gen.last_body["model"] == "gemini-3.5-flash-lite"


def test_bearer_header_and_key_not_in_url():
    opener = FakeOpener('{"refusal": null, "text": "ok"}')
    gen = L.HTTPChatGenerator(_config(), api_key=KEY, opener=opener)
    gen.generate(_req())
    request, _ = opener.requests[0]
    assert request.get_header("Authorization") == f"Bearer {KEY}" and KEY not in request.full_url


def test_key_resolution_prefers_nova_then_gemini():
    assert L.resolve_api_key("gemini", {"GEMINI_API_KEY": "g"}) == "g"
    assert L.resolve_api_key("gemini", {"GEMINI_API_KEY": "g", "NOVA_LLM_API_KEY": "n"}) == "n"
    assert L.resolve_api_key("llama-server", {"GEMINI_API_KEY": "g"}) is None


def test_redaction_covers_gemini_env_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    assert KEY not in L.redact_secrets(f"oops {KEY}")


def test_http_error_message_has_no_key_or_body():
    gen = L.HTTPChatGenerator(_config(), api_key=KEY, opener=FakeOpener(_http_error(429)))
    with pytest.raises(L.LLMRequestError) as exc:
        gen.generate(_req())
    assert KEY not in str(exc.value) and "body" not in str(exc.value) and "429" in str(exc.value)


def test_failover_on_429_then_cooldown():
    now = [0.0]
    first = L.HTTPChatGenerator(_config(), api_key=KEY, opener=FakeOpener(_http_error(429), _http_error(429)))
    second = L.HTTPChatGenerator(_config(model="gemini-3.1-flash-lite"), api_key=KEY,
                                 opener=FakeOpener('{"refusal": null, "text": "a"}', '{"refusal": null, "text": "b"}', '{"refusal": null, "text": "c"}'))
    chain = L.FailoverGenerator([first, second], cooldown_s=60, clock=lambda: now[0])
    assert "a" in chain.generate(_req()).raw
    assert "b" in chain.generate(_req()).raw and first.calls == 1  # first is cooling down
    now[0] = 61
    assert "c" in chain.generate(_req()).raw and first.calls == 2


def test_failover_raises_last_error_when_all_fail():
    gens = [L.HTTPChatGenerator(_config(), api_key=KEY, opener=FakeOpener(_http_error(402))) for _ in range(2)]
    with pytest.raises(L.LLMRequestError):
        L.FailoverGenerator(gens).generate(_req())


def test_all_down_gives_fixed_error_line_not_exception():
    gens = [L.HTTPChatGenerator(_config(), api_key=KEY, opener=FakeOpener(_http_error(503))) for _ in range(2)]
    out = G.run_generation(_req(), L.FailoverGenerator(gens), POLICY, "neutral", error_text="tool error line")
    assert out.status == "error" and out.text == "tool error line"


def test_persona_sets_mars_and_identity_line():
    persona = POLICY["generation"]["persona"]
    assert "Mars" in persona and "I'm Nova, your navigation and observation voice assistant" in persona
    assert POLICY["fast_replies"]["intents"]["identity"]["line"] == "I'm Nova, your navigation and observation voice assistant."
    assert not any(c.isdigit() for c in persona)


def test_when_every_model_is_cooling_down_they_are_tried_again():
    now = [0.0]
    first = L.HTTPChatGenerator(_config(), api_key=KEY, opener=FakeOpener(_http_error(504), '{"refusal": null, "text": "back"}'))
    second = L.HTTPChatGenerator(_config(model="gemini-3.1-flash-lite"), api_key=KEY, opener=FakeOpener(_http_error(503)))
    chain = L.FailoverGenerator([first, second], cooldown_s=60, clock=lambda: now[0])
    with pytest.raises(L.LLMRequestError):
        chain.generate(_req())  # both fail once and both start cooling down
    now[0] = 5
    assert "back" in chain.generate(_req()).raw and first.calls == 2  # not an instant failure for the next minute

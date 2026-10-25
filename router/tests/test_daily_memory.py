"""Daily query-only memory and Gemini context wiring, without live credentials."""
from datetime import datetime
import json

import pytest

from router.memory import DailyMemory
from router.generation import render_prompt
from router.llm_generator import HTTPChatGenerator, build_generator, load_live_llm_config
from test_binary_showcase import CheckingGenerator, handler


def test_daily_memory_local_date_complete_file_and_rollover(tmp_path):
    now = [datetime(2026, 10, 25, 23, 59)]
    memory = DailyMemory(tmp_path, now=lambda: now[0])
    assert memory.load() == ""
    gen = CheckingGenerator({"decision": "none"}, "grey")
    h = handler(generator=gen, memory=memory)
    h.handle("hi")  # quick response is still recorded
    h.handle("what color is the sky")
    assert memory.load() == "hi\nwhat color is the sky\n"
    assert gen.checks[0][0].daily_memory == "hi\n"
    assert gen.calls[0].daily_memory == "hi\n"
    assert "<daily_memory>" in render_prompt(gen.calls[0])[1]["content"]
    assert "grey" not in memory.load()
    now[0] = datetime(2026, 10, 26)
    h.handle("hello")
    assert memory.load() == "hello\n"
    assert (tmp_path / "2026-10-25.md").read_text() == "hi\nwhat color is the sky\n"


def test_daily_memory_includes_tool_and_refusal_queries(tmp_path):
    memory = DailyMemory(tmp_path)
    handler("call_o2", memory=memory).handle("oxygen status")
    handler(memory=memory).handle("is this safe")
    assert memory.load() == "oxygen status\nis this safe\n"


def test_gemini_generation_and_self_check_use_same_endpoint_and_memory(llm_server, tmp_path):
    from conftest import chat_completion, reply_json
    config = load_live_llm_config(runtime="gemini", base_url=llm_server.base_url)
    memory = DailyMemory(tmp_path)
    memory.append("earlier query")
    gen = HTTPChatGenerator(config, api_key="fake-test-key")
    def reply(body):
        assert "earlier query" in body["messages"][1]["content"]
        if "Available tool schemas:" in body["messages"][0]["content"]:
            return chat_completion(json.dumps({"decision": "none"}))
        return chat_completion(reply_json("grey"))
    llm_server.reply = reply
    turn = handler(generator=gen, memory=memory).handle("sky color")
    assert turn["spoken"] == "grey" and len(llm_server.requests) == 2
    assert gen.url == llm_server.base_url.rstrip("/") + "/chat/completions"


def test_gemini_default_and_missing_credentials(monkeypatch):
    monkeypatch.setattr("router.llm_generator.load_dotenv", lambda path: None)
    for name in ("GEMINI_API_KEY", "NOVA_LLM_API_KEY", "NOVA_LLM_RUNTIME", "NOVA_LLM_BASE_URL", "NOVA_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    config = load_live_llm_config()
    assert config["runtime"] == "gemini"
    assert HTTPChatGenerator(config).url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    messages = []
    _, label = build_generator(messages.append)
    assert "credentials missing" in label and "GEMINI_API_KEY" in messages[0]


@pytest.mark.parametrize("entry", ["pipeline", "stt"])
def test_missing_six_class_checkpoint_clear_error(tmp_path, entry):
    from router.pipeline import load_pipeline
    from router.stt import _load_router
    with pytest.raises(SystemExit, match="six-class router checkpoint missing - retrain required"):
        (load_pipeline(tmp_path / "binary") if entry == "pipeline" else _load_router(tmp_path / "binary", "cpu"))

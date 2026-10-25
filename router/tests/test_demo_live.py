"""router/demo_live.py: scripted/stdin sessions with the keyword fake router, with no server and with a fake server on 127.0.0.1."""
import io
import socket
import urllib.parse

import pytest
from conftest import chat_completion, reply_json

from router.demo_live import KeywordRouter, apply_set, frozen_clock, main
from router.datasource import FixtureDatasource


def closed_port_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}/v1"


def run(capsys, argv, script="", tmp_path=None):
    path = tmp_path / "turns.txt"
    path.write_text(script)
    assert main([*argv, "--fake-router", "--script", str(path)]) == 0
    return capsys.readouterr().out


SCRIPT = "hello there\nhow much oxygen do I have left\nhow far away is the moon\nwhat is the capital of france\n/quit\nhi again after quit\n"


def test_keyword_router_labels():
    router = KeywordRouter()
    expected = {"take me to the airlock": "call_navigation", "show me the map": "call_map", "how much oxygen do I have": "call_o2",
                "what's my heart rate": "call_data", "am I safe": "call_safety", "how far away is the moon": "call_llm", "hi": "NO_TOOL",
                "hello there": "NO_TOOL", "tell me a joke about cheese": "OOD"}
    assert {text: router.predict(text)["label"] for text in expected} == expected


def test_no_server_says_so_in_one_line_and_runs_the_scripted_turns_offline(capsys, tmp_path):
    url = closed_port_url()
    out = run(capsys, ["--base-url", url], SCRIPT, tmp_path)
    fallback = [line for line in out.splitlines() if line.startswith("LLM:")]
    assert len(fallback) == 1 and f"no LLM server reachable at {url}" in fallback[0] and "falling back to the offline FillerGenerator" in fallback[0]
    assert out.count("\n[turn ") == 4  # /quit stops the session; the line after it is never run
    assert "Hello. I'm here whenever you need me." in out and "Oxygen at 99.9 percent" in out
    assert "llm     : CALLED" in out and "llm     : no (tool turn: template line only)" in out and "general_explanation" in out
    assert "session over: 4 turn(s); generator FillerGenerator (offline)" in out


def test_no_llm_flag_never_contacts_the_server(capsys, tmp_path, llm_server):
    out = run(capsys, ["--no-llm", "--base-url", llm_server.base_url], "hello there\n", tmp_path)
    assert "LLM: disabled (--no-llm)" in out and llm_server.requests == []


def test_fake_server_session_shows_raw_output_spoken_line_and_latency(capsys, tmp_path, llm_server):
    llm_server.reply = chat_completion("<think>x</think>" + reply_json("Hello from the fake model."), reasoning_content="hidden")
    out = run(capsys, ["--base-url", llm_server.base_url], "hello there\nhow much oxygen do I have left\n", tmp_path)
    assert "using llama-server" in out and "HTTPChatGenerator (llama-server)" in out
    assert "raw #1  : '{\"refusal\": null, \"text\": \"Hello from the fake model.\"}'" in out and "hidden" not in out
    assert "spoken  : Hello from the fake model." in out and "request : chitchat   mode=generated" in out
    assert "latency : generator-reported" in out and "turn wall" in out and "verify  : ok=True" in out
    assert len(llm_server.requests) == 1  # the tool turn never reached the server
    assert llm_server.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_fake_server_bad_number_is_blocked_and_shown(capsys, tmp_path, llm_server):
    llm_server.reply = chat_completion(reply_json("It is 384400 kilometers."))
    out = run(capsys, ["--base-url", llm_server.base_url], "hello there\n", tmp_path)
    assert "blocked_text='It is 384400 kilometers.'" in out and "digit_in_prose" in out and "outcome : fallback after 2 attempt(s)" in out
    assert "I can't read that out reliably right now." in out


def test_server_failing_mid_session_shows_llm_error_and_keeps_going(capsys, tmp_path, llm_server):
    llm_server.reply = (500, {"error": "boom"})
    out = run(capsys, ["--base-url", llm_server.base_url], "hello there\nhow much oxygen do I have left\n", tmp_path)
    assert "llm_error: LLMRequestError: HTTP 500" in out and "I could not complete that request." in out
    assert "Oxygen at 99.9 percent" in out and "session over: 2 turn(s)" in out


def test_runtime_flags_reach_the_request(capsys, tmp_path, llm_server):
    run(capsys, ["--base-url", llm_server.base_url, "--runtime", "mlx", "--llm-model", "mlx-community/Qwen3.5-4B-4bit"], "hello there\n", tmp_path)
    body = llm_server.requests[0]
    assert body["model"] == "mlx-community/Qwen3.5-4B-4bit" and "seed" not in body and "json_schema" not in body


def test_deadline_flag_bounds_the_turn(capsys, tmp_path, llm_server):
    llm_server.delay_s = 2.0
    out = run(capsys, ["--base-url", llm_server.base_url, "--deadline-ms", "200"], "hello there\n", tmp_path)
    assert "llm_error: LLMTimeoutError" in out and "deadline 200 ms" in out


def test_critical_alert_bypasses_the_llm(capsys, tmp_path, llm_server):
    out = run(capsys, ["--base-url", llm_server.base_url], "/set o2_pct=14\nhi\n", tmp_path)
    assert "Critical alert. Oxygen at 14.0 percent" in out and "llm     : no (critical alert" in out and llm_server.requests == []


def test_advance_flag_and_command_move_the_sim(capsys, tmp_path):
    out = run(capsys, ["--no-llm", "--advance", "3000"], "how much oxygen do I have left\n/advance 3000\n/sim\n", tmp_path)
    assert "Oxygen at 99.9 percent" not in out and "advanced 3000 sim s" in out
    assert out.count("sim: o2=") >= 4


def test_set_command_validates(tmp_path):
    ds = FixtureDatasource(clock=frozen_clock())
    assert apply_set(ds, "o2_pct=55.5") == "set o2_pct=55.5" and ds.snapshot()["o2_pct"] == 55.5
    assert apply_set(ds, "bpm=abc").startswith("bpm needs a number") and apply_set(ds, "o2_pct=nan").endswith("finite number")
    assert apply_set(ds, "colour=red").startswith("cannot set 'colour'") and apply_set(ds, "o2_pct").startswith("usage")
    assert apply_set(ds, "scenario=o2_leak") == "set scenario=o2_leak"
    assert apply_set(ds, "scenario=warp").startswith("cannot set scenario") and ds.mission["scenario"] == "o2_leak"


def test_fixture_does_not_go_stale_in_a_long_session():
    ds = FixtureDatasource(clock=frozen_clock())
    assert ds.snapshot()["o2_pct"] > 90


def test_stdin_is_read_when_not_a_tty_and_bad_commands_do_not_crash(capsys):
    assert main(["--fake-router", "--no-llm"], stdin=io.StringIO("# comment\n\n/bogus\n/advance x\nhi\n")) == 0
    out = capsys.readouterr().out
    assert "unknown command '/bogus'" in out and "usage: /advance N" in out and "[turn 1] 'hi'" in out


def test_missing_checkpoint_gives_a_clear_message(tmp_path):
    with pytest.raises(SystemExit) as info:
        main(["--model", str(tmp_path / "nope"), "--no-llm"], stdin=io.StringIO(""))
    assert "router checkpoint not found" in str(info.value) and "--fake-router" in str(info.value)


def test_server_error_body_echoing_the_api_key_never_reaches_the_output(capsys, tmp_path, llm_server, monkeypatch):
    key = "SENTINEL-KEY-12345"
    monkeypatch.setenv("NOVA_LLM_API_KEY", key)
    llm_server.reply = (401, f"bad key Bearer {key}")
    out = run(capsys, ["--base-url", llm_server.base_url], "hello there\n", tmp_path)
    assert "llm_error: LLMRequestError: HTTP 401" in out and "bad key" not in out
    assert llm_server.requests and key not in out and "Bearer" not in out


SECRETS = ["SENTINEL-KEY-12345", "SENTINEL/KEY 12345"]


def assert_no_secret(capsys, out, secret):
    quoted = urllib.parse.quote(secret, safe="")
    for text in (out, capsys.readouterr().err):
        assert secret not in text and quoted not in text


@pytest.mark.parametrize("secret", SECRETS)
def test_successful_startup_never_prints_the_api_key_from_the_model_name(capsys, tmp_path, llm_server, monkeypatch, secret):
    monkeypatch.setenv("NOVA_LLM_API_KEY", secret)
    out = run(capsys, ["--base-url", llm_server.base_url, "--llm-model", f"model-{urllib.parse.quote(secret, safe='')}-{secret}"],
              "hello there\n", tmp_path)
    assert "using llama-server" in out and "[redacted]" in out and "HTTPChatGenerator (llama-server)" in out  # the real generator ran
    assert_no_secret(capsys, out, secret)


@pytest.mark.parametrize("secret", SECRETS)
@pytest.mark.parametrize("where", ["userinfo", "query"])
def test_fallback_line_never_prints_the_api_key_from_the_base_url(capsys, tmp_path, llm_server, monkeypatch, secret, where):
    # the adapter cannot talk to a URL with userinfo or a query (urllib / the /v1 suffix), so the session falls back; its line is redacted
    monkeypatch.setenv("NOVA_LLM_API_KEY", secret)
    quoted = urllib.parse.quote(secret, safe="")
    url = llm_server.base_url.replace("//", f"//user:{quoted}@") if where == "userinfo" else f"{llm_server.base_url}?key={quoted}"
    out = run(capsys, ["--base-url", url], "hello there\n", tmp_path)
    assert "falling back to the offline FillerGenerator" in out and "[redacted]" in out
    assert_no_secret(capsys, out, secret)


@pytest.mark.parametrize("secret", SECRETS)
@pytest.mark.parametrize("variable", ["NOVA_LLM_BASE_URL", "NOVA_LLM_RUNTIME", "NOVA_LLM_CONSTRAIN"])
def test_invalid_config_holding_the_api_key_exits_cleanly_without_printing_it(capsys, monkeypatch, secret, variable):
    monkeypatch.setenv("NOVA_LLM_API_KEY", secret)
    monkeypatch.setenv(variable, secret)
    with pytest.raises(SystemExit) as info:
        main(["--fake-router"], stdin=io.StringIO(""))
    captured = capsys.readouterr()
    message = str(info.value)
    assert info.value.code == message and message.startswith("invalid LLM config: ") and "\n" not in message
    assert variable.split("_", 2)[2].lower() in message  # names the key, not the value
    for text in (message, captured.out, captured.err):
        assert secret not in text and urllib.parse.quote(secret, safe="") not in text


@pytest.mark.parametrize("flag", ["--runtime", "--constrain", "--deadline-ms", "--bogus"])
def test_argparse_errors_never_echo_the_api_key(capsys, monkeypatch, flag):
    monkeypatch.setenv("NOVA_LLM_API_KEY", "SENTINEL-KEY-12345")
    with pytest.raises(SystemExit) as info:
        main(["--fake-router", flag, "SENTINEL-KEY-12345"], stdin=io.StringIO(""))
    captured = capsys.readouterr()
    assert info.value.code == 2 and "error:" in captured.err and "[redacted]" in captured.err
    for text in (str(info.value), captured.out, captured.err):
        assert "SENTINEL-KEY-12345" not in text


def test_help_works_without_printing_the_api_key(capsys, monkeypatch):
    monkeypatch.setenv("NOVA_LLM_API_KEY", "SENTINEL-KEY-12345")
    with pytest.raises(SystemExit) as info:
        main(["--help"], stdin=io.StringIO(""))
    captured = capsys.readouterr()
    assert info.value.code == 0 and "--runtime" in captured.out and "SENTINEL-KEY-12345" not in captured.out + captured.err


def test_set_command_and_script_echo_never_print_the_api_key(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("NOVA_LLM_API_KEY", "SENTINEL-KEY-12345")
    out = run(capsys, ["--no-llm"], "/set bpm=SENTINEL-KEY-12345\n/bogus SENTINEL-KEY-12345\n", tmp_path)
    assert "[redacted]" in out and "SENTINEL-KEY-12345" not in out + capsys.readouterr().err


def test_completion_echoing_the_api_key_never_reaches_the_output(capsys, tmp_path, llm_server, monkeypatch):
    key = "SENTINEL-KEY-12345"
    monkeypatch.setenv("NOVA_LLM_API_KEY", key)
    llm_server.reply = (200, chat_completion(reply_json(f"your key is {key} / {urllib.parse.quote(key, safe='')} / Bearer {key}")))
    out = run(capsys, ["--base-url", llm_server.base_url], "hello there\n", tmp_path)
    assert llm_server.requests and "raw #1" in out and "[redacted]" in out
    assert_no_secret(capsys, out, key)


def test_pos_command_in_a_script_changes_the_distance_and_help_lists_it(capsys, tmp_path):
    out = run(capsys, ["--no-llm"], "how far is home\n/pos 500 0\nhow far is home\n/pos nope\n/help\n", tmp_path)
    assert "homebase is about 0 meters away" in out and "astronaut moved to x=500 z=0" in out
    assert "homebase is about 500 meters away, roughly 9 minutes on foot" in out
    assert "usage: /pos X Z" in out and "/pos X Z (astronaut position in meters)" in out


def test_default_checkpoint_is_v2_with_a_message_fallback_to_v1(capsys, tmp_path, monkeypatch):
    from router import demo_live
    from router.demo_live import build_router, parse_args
    loaded = []
    monkeypatch.setattr(demo_live.Router, "load", lambda path, labels, device="cpu": loaded.append(path) or object())
    for name in ("v1", "v2"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "weights").write_text("x")
    monkeypatch.setenv("NOVA_MODELS_DIR", str(tmp_path))
    messages = []
    build_router(parse_args([]), messages.append)
    assert loaded == [str(tmp_path / "v2")] and messages == [f"router: {tmp_path / 'v2'}"]
    (tmp_path / "v2" / "weights").unlink()
    build_router(parse_args([]), messages.append)
    assert loaded[-1] == str(tmp_path / "v1") and any("not found; falling back" in m for m in messages)


def test_hi_deterministic_and_keyboard_interrupt_exits_cleanly(capsys, tmp_path):
    out = run(capsys, ["--no-llm"], "hi\n", tmp_path)
    assert "Hello. How can I help?" in out  # fast_replies greeting (policy PLACEHOLDER line)
    assert "llm     : CALLED" not in out

    class InterruptedInput:
        def isatty(self):
            return False

        def read(self):
            raise KeyboardInterrupt

    assert main(["--fake-router", "--no-llm"], stdin=InterruptedInput()) == 130
    assert "session interrupted" in capsys.readouterr().out


def test_interim_line_is_printed_and_thank_you_ends_the_session(capsys, tmp_path):
    out = run(capsys, ["--no-llm"], "explain gravity\nthank you\nhow is my oxygen\n", tmp_path)
    assert "nova (interim): Let me think." in out and out.index("nova (interim)") < out.index("[turn 1]")
    assert "spoken  : Happy to help!" in out and "(session ended by the user)" in out
    assert "how is my oxygen" not in out.split("(session ended by the user)")[1] and "session over: 2 turn(s)" in out

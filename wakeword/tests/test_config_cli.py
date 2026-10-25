"""Offline credential fallback and voice LLM flags; no models, microphone or network."""
import pytest

import config
from main import build_parser


def test_default_key_falls_back_to_repo_root(tmp_path, monkeypatch):
    wake = tmp_path / "wakeword"
    wake.mkdir()
    router = tmp_path / "router"
    router.mkdir()
    local = router / ".env"
    root = tmp_path / ".env"
    monkeypatch.setattr(config, "HERE", wake)
    monkeypatch.setattr(config, "DEFAULT_ENV_FILE", local)
    monkeypatch.delenv(config.KEY_VAR, raising=False)
    monkeypatch.delenv("NOVA_ENV_FILE", raising=False)
    root.write_text("ELEVENLABS_API_KEY=root-test-key\n")
    assert config.load_api_key() == "root-test-key"
    local.write_text("ELEVENLABS_API_KEY=local-test-key\n")
    assert config.load_api_key() == "local-test-key"
    monkeypatch.setenv(config.KEY_VAR, "exported-test-key")
    assert config.load_api_key() == "exported-test-key"


def test_explicit_env_file_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.delenv(config.KEY_VAR, raising=False)
    with pytest.raises(config.ConfigError):
        config.load_api_key(tmp_path / "missing")


def test_llm_flags_parse():
    args = build_parser(config.WakeConfig.from_env()).parse_args([
        "--llm-model", "test-model", "--deadline-ms", "9000"])
    assert args.llm_model == "test-model" and args.deadline_ms == 9000

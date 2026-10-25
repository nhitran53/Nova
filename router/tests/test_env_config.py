from pathlib import Path

from router import stt, tts


ROOT = Path(__file__).resolve().parents[2]


def test_speech_loaders_use_the_root_env_from_any_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert stt.ENV_PATH.resolve() == ROOT / ".env"
    assert tts.ENV_PATH == ROOT / ".env"


def test_speech_reads_the_file_without_overriding_the_environment(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("ELEVENLABS_API_KEY=file-key\n")
    monkeypatch.setattr(stt, "ENV_PATH", env_file)
    monkeypatch.delenv(stt.KEY_VAR, raising=False)
    assert stt._api_key() == "file-key"
    monkeypatch.setenv(stt.KEY_VAR, "environment-key")
    assert stt._api_key() == "environment-key"

import json
import os
import subprocess
import sys
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


def _load_speech_settings(env_file=None, environment_key=None):
    """Load speech settings in a fresh process."""
    env = {k: v for k, v in os.environ.items() if k not in ("NOVA_ENV_FILE", stt.KEY_VAR)}
    if env_file is not None:
        env["NOVA_ENV_FILE"] = str(env_file)
    if environment_key is not None:
        env[stt.KEY_VAR] = environment_key
    code = ("import json; from router import stt, tts; "
            "print(json.dumps({'stt': str(stt.ENV_PATH), 'tts': str(tts.ENV_PATH), 'key': stt._api_key()}))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_nova_env_file_replaces_the_root_env_for_both_speech_loaders(tmp_path):
    env_file = tmp_path / "team.env"
    env_file.write_text("ELEVENLABS_API_KEY=team-file-key\n")
    got = _load_speech_settings(env_file)
    assert Path(got["stt"]) == env_file and Path(got["tts"]) == env_file
    assert got["key"] == "team-file-key"


def test_environment_beats_the_nova_env_file(tmp_path):
    env_file = tmp_path / "team.env"
    env_file.write_text("ELEVENLABS_API_KEY=team-file-key\n")
    assert _load_speech_settings(env_file, environment_key="environment-key")["key"] == "environment-key"


def test_tts_reads_the_file_without_overriding_the_environment(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("ELEVENLABS_API_KEY=file-key\n")
    monkeypatch.setattr(tts, "ENV_PATH", env_file)
    monkeypatch.setattr(tts, "_client_factory", lambda key: ("client", key))
    for environment, expected in ((None, "file-key"), ("environment-key", "environment-key")):
        monkeypatch.setattr(tts, "_client", None)
        if environment is None:
            monkeypatch.delenv(tts.KEY_VAR, raising=False)
        else:
            monkeypatch.setenv(tts.KEY_VAR, environment)
        assert tts.init() == ("client", expected)

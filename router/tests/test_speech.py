"""speech.py with a fake ElevenLabs client factory and a fake player: no network, no audio, no .env read at import."""
import importlib
import subprocess
import sys
import threading

import pytest

from router import speech


class FakeClient:
    def __init__(self, api_key):
        self.api_key, self.requests = api_key, []
        self.text_to_speech = self

    def convert(self, **kwargs):
        self.requests.append(kwargs)
        return iter([b"ab", b"cd"])


@pytest.fixture(autouse=True)
def fresh_singleton(monkeypatch):
    created = []
    monkeypatch.setattr(speech, "_client_factory", lambda key: created.append(FakeClient(key)) or created[-1])
    monkeypatch.setattr(speech, "load_dotenv", lambda path: None)
    monkeypatch.setenv(speech.KEY_VAR, "test-key-not-real")
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
    monkeypatch.delenv("ELEVENLABS_MODEL_ID", raising=False)
    played = []
    monkeypatch.setattr(speech, "_play", lambda audio: played.append(audio) or {"player": "fake", "returncode": 0})
    speech.reset()
    yield created, played
    speech.reset()


def test_import_creates_no_client_and_reads_no_env(monkeypatch):
    created = []
    monkeypatch.setattr("elevenlabs.client.ElevenLabs", lambda *a, **k: created.append(1))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: created.append("dotenv"))
    importlib.reload(speech)
    assert created == [] and speech._client is None


def test_client_is_created_once_across_speak_calls_and_threads(fresh_singleton):
    created, played = fresh_singleton
    threads = [threading.Thread(target=speech.init) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    speech.speak("one")
    speech.speak("two")
    assert len(created) == 1 and speech.init() is created[0] and len(created[0].requests) == 2
    assert played == [b"abcd", b"abcd"]


def test_first_speak_creates_the_client_lazily(fresh_singleton):
    created, _ = fresh_singleton
    assert created == []
    speech.speak("hi")
    assert len(created) == 1 and created[0].api_key == "test-key-not-real"


def test_request_uses_voice_model_pcm_format_and_timeout(fresh_singleton):
    created, played = fresh_singleton
    info = speech.speak("  Oxygen at 87.5 percent.  ", route="tool")
    request = created[0].requests[0]
    assert request["text"] == "Oxygen at 87.5 percent."
    assert request["voice_id"] == "VmPkIaPdP997cjIfBWRB" and request["model_id"] == "eleven_v4"
    assert request["output_format"] == "pcm_24000" and request["request_options"]["timeout_in_seconds"] == speech.NETWORK_TIMEOUT_S
    assert info == {"bytes": 4, "player": "fake", "returncode": 0}


def test_voice_and_model_are_env_overridable(fresh_singleton, monkeypatch):
    created, _ = fresh_singleton
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice-x")
    monkeypatch.setenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
    speech.speak("hi")
    assert (created[0].requests[0]["voice_id"], created[0].requests[0]["model_id"]) == ("voice-x", "eleven_flash_v2_5")


@pytest.mark.parametrize("text", ["", "   ", "\n\t", None])
def test_empty_text_is_a_no_op(fresh_singleton, text):
    created, played = fresh_singleton
    assert speech.speak(text) is None
    assert created == [] and played == []


def test_missing_key_fails_clearly_without_creating_a_client(fresh_singleton, monkeypatch):
    created, _ = fresh_singleton
    monkeypatch.delenv(speech.KEY_VAR)
    with pytest.raises(speech.SpeechError, match="ELEVENLABS_API_KEY is not set"):
        speech.speak("hi")
    assert created == []


def test_api_error_message_has_no_headers_or_key(fresh_singleton, monkeypatch):
    class ApiError(Exception):
        status_code, body = 401, {"detail": {"message": "missing permission"}}

    def boom(**kwargs):
        raise ApiError("headers: xi-api-key=test-key-not-real")

    speech.init().convert = boom
    with pytest.raises(speech.SpeechError) as caught:
        speech.speak("hi")
    assert "test-key-not-real" not in str(caught.value) and "401" in str(caught.value) and "missing permission" in str(caught.value)


def test_empty_audio_is_an_error(fresh_singleton):
    speech.init().convert = lambda **kwargs: iter([])
    with pytest.raises(speech.SpeechError, match="no audio"):
        speech.speak("hi")


def test_player_pipes_pcm_to_paplay_with_timeout_and_reports_nonzero_exit(monkeypatch):
    monkeypatch.setattr(speech, "_play", speech.__dict__["_play"])  # the real one (fixture replaced it)
    importlib.reload(speech)
    seen = {}

    def fake_run(command, **kwargs):
        seen.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, seen["rc"], stderr=b"no sink")

    monkeypatch.setattr(speech.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "paplay" else None)
    monkeypatch.setattr(speech.subprocess, "run", fake_run)
    seen["rc"] = 0
    assert speech._play(b"pcm") == {"player": "paplay", "returncode": 0}
    assert seen["command"][:2] == ["paplay", "--raw"] and seen["input"] == b"pcm" and seen["timeout"] == speech.PLAYBACK_TIMEOUT_S
    seen["rc"] = 1
    with pytest.raises(speech.SpeechError, match="paplay exited 1: no sink"):
        speech._play(b"pcm")


def test_no_player_installed_is_an_error(monkeypatch):
    importlib.reload(speech)
    monkeypatch.setattr(speech.shutil, "which", lambda name: None)
    with pytest.raises(speech.SpeechError, match="no audio player"):
        speech._play(b"pcm")

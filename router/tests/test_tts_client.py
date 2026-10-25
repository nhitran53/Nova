"""tts.py with a fake ElevenLabs client factory and a fake player: no network, no audio, no .env read at import."""
import importlib
import subprocess
import sys
import threading

import pytest

from router import tts


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
    monkeypatch.setattr(tts, "_client_factory", lambda key: created.append(FakeClient(key)) or created[-1])
    monkeypatch.setattr(tts, "load_dotenv", lambda path: None)
    monkeypatch.setenv(tts.KEY_VAR, "test-key-not-real")
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
    monkeypatch.delenv("ELEVENLABS_MODEL_ID", raising=False)
    played = []
    monkeypatch.setattr(tts, "_play", lambda audio: played.append(audio) or {"player": "fake", "returncode": 0})
    tts.reset()
    yield created, played
    tts.reset()


def test_import_creates_no_client_and_reads_no_env(monkeypatch):
    created = []
    monkeypatch.setattr("elevenlabs.client.ElevenLabs", lambda *a, **k: created.append(1))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: created.append("dotenv"))
    importlib.reload(tts)
    assert created == [] and tts._client is None


def test_client_is_created_once_across_speak_calls_and_threads(fresh_singleton):
    created, played = fresh_singleton
    threads = [threading.Thread(target=tts.init) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    tts.speak("one")
    tts.speak("two")
    assert len(created) == 1 and tts.init() is created[0] and len(created[0].requests) == 2
    assert played == [b"abcd", b"abcd"]


def test_first_speak_creates_the_client_lazily(fresh_singleton):
    created, _ = fresh_singleton
    assert created == []
    tts.speak("hi")
    assert len(created) == 1 and created[0].api_key == "test-key-not-real"


def test_request_uses_voice_model_pcm_format_and_timeout(fresh_singleton):
    created, played = fresh_singleton
    info = tts.speak("  Oxygen at 87.5 percent.  ", route="tool")
    request = created[0].requests[0]
    assert request["text"] == "Oxygen at 87.5 percent."
    assert request["voice_id"] == "VmPkIaPdP997cjIfBWRB" and request["model_id"] == "eleven_v4"
    assert request["output_format"] == "pcm_24000" and request["request_options"]["timeout_in_seconds"] == tts.NETWORK_TIMEOUT_S
    assert info == {"bytes": 4, "player": "fake", "returncode": 0}


def test_voice_and_model_are_env_overridable(fresh_singleton, monkeypatch):
    created, _ = fresh_singleton
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice-x")
    monkeypatch.setenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
    tts.speak("hi")
    assert (created[0].requests[0]["voice_id"], created[0].requests[0]["model_id"]) == ("voice-x", "eleven_flash_v2_5")


@pytest.mark.parametrize("text", ["", "   ", "\n\t", None])
def test_empty_text_is_a_no_op(fresh_singleton, text):
    created, played = fresh_singleton
    assert tts.speak(text) is None
    assert created == [] and played == []


def test_missing_key_fails_clearly_without_creating_a_client(fresh_singleton, monkeypatch):
    created, _ = fresh_singleton
    monkeypatch.delenv(tts.KEY_VAR)
    with pytest.raises(tts.SpeechError, match="ELEVENLABS_API_KEY is not set"):
        tts.speak("hi")
    assert created == []


def test_api_error_message_has_no_headers_or_key(fresh_singleton, monkeypatch):
    class ApiError(Exception):
        status_code, body = 401, {"detail": {"message": "missing permission"}}

    def boom(**kwargs):
        raise ApiError("headers: xi-api-key=test-key-not-real")

    tts.init().convert = boom
    with pytest.raises(tts.SpeechError) as caught:
        tts.speak("hi")
    assert "test-key-not-real" not in str(caught.value) and "401" in str(caught.value) and "missing permission" in str(caught.value)


def test_key_in_the_api_error_message_is_redacted(fresh_singleton):
    class ApiError(Exception):
        status_code, body = 401, {"detail": {"message": "bad key test-key-not-real, check it"}}

    def boom(**kwargs):
        raise ApiError("x")

    tts.init().convert = boom
    with pytest.raises(tts.SpeechError) as caught:
        tts.speak("hi")
    assert "test-key-not-real" not in str(caught.value) and "[redacted]" in str(caught.value) and "401" in str(caught.value)


def test_key_in_a_player_error_is_redacted(monkeypatch):
    importlib.reload(tts)
    monkeypatch.setenv(tts.KEY_VAR, "test-key-not-real")
    monkeypatch.setattr(tts, "_use_sounddevice", lambda: False)  # the Linux player path, on any OS
    monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/paplay")
    monkeypatch.setattr(tts.subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 1, stderr=b"oops test-key-not-real"))
    with pytest.raises(tts.SpeechError) as caught:
        tts._play(b"pcm")
    assert "test-key-not-real" not in str(caught.value) and "oops [redacted]" in str(caught.value)


def test_client_factory_failure_raises_a_sanitized_speech_error(monkeypatch):
    def broken_factory(key):
        raise ValueError(f"bad credentials {key}")

    monkeypatch.setattr(tts, "_client_factory", broken_factory)
    with pytest.raises(tts.SpeechError) as caught:
        tts.init()
    assert "test-key-not-real" not in str(caught.value) and "ValueError" in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__suppress_context__ and tts._client is None


def test_empty_audio_is_an_error(fresh_singleton):
    tts.init().convert = lambda **kwargs: iter([])
    with pytest.raises(tts.SpeechError, match="no audio"):
        tts.speak("hi")


def test_player_pipes_pcm_to_paplay_with_timeout_and_reports_nonzero_exit(monkeypatch):
    importlib.reload(tts)  # restores the real _play, which the autouse fixture replaced
    monkeypatch.setattr(tts, "_use_sounddevice", lambda: False)  # the Linux player path, on any OS
    seen = {}

    def fake_run(command, **kwargs):
        seen.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, seen["rc"], stderr=b"no sink")

    monkeypatch.setattr(tts.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "paplay" else None)
    monkeypatch.setattr(tts.subprocess, "run", fake_run)
    seen["rc"] = 0
    assert tts._play(b"pcm") == {"player": "paplay", "returncode": 0}
    assert seen["command"][:2] == ["paplay", "--raw"] and seen["input"] == b"pcm" and seen["timeout"] == tts.PLAYBACK_TIMEOUT_S
    seen["rc"] = 1
    with pytest.raises(tts.SpeechError, match="paplay exited 1: no sink"):
        tts._play(b"pcm")


def test_no_player_installed_is_an_error(monkeypatch):
    importlib.reload(tts)
    monkeypatch.setattr(tts, "_use_sounddevice", lambda: False)  # the Linux player path, on any OS
    monkeypatch.setattr(tts.shutil, "which", lambda name: None)
    with pytest.raises(tts.SpeechError, match="no audio player"):
        tts._play(b"pcm")


class FakeOutputStream:
    """Stands in for sounddevice.RawOutputStream: records the settings and the bytes written."""
    def __init__(self, log, fail=None, **kwargs):
        self.log, self.fail = log, fail
        log["settings"] = kwargs

    def __enter__(self):
        if self.fail:
            raise self.fail
        return self

    def write(self, data):
        self.log["written"] = bytes(data)

    def __exit__(self, *exc):
        self.log["closed"] = True


def fake_sounddevice(log, fail=None):
    module = type(sys)("sounddevice")
    module.RawOutputStream = lambda **kwargs: FakeOutputStream(log, fail, **kwargs)
    return module


def test_windows_plays_through_sounddevice_as_24k_mono_int16(monkeypatch):
    importlib.reload(tts)
    monkeypatch.setattr(tts, "_use_sounddevice", lambda: True)
    monkeypatch.setattr(tts.subprocess, "run", lambda *a, **k: pytest.fail("no player process on Windows"))
    log = {}
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice(log))
    assert tts._play(b"pcm!x") == {"player": "sounddevice", "returncode": 0}
    assert log["settings"] == {"samplerate": tts.PCM_SAMPLE_RATE, "channels": 1, "dtype": "int16"}
    assert log["written"] == b"pcm!" and log["closed"]  # odd trailing byte dropped, stream closed (= drained)


def test_windows_playback_errors_are_speech_errors_without_the_key(monkeypatch):
    importlib.reload(tts)
    monkeypatch.setenv(tts.KEY_VAR, "test-key-not-real")
    monkeypatch.setattr(tts, "_use_sounddevice", lambda: True)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice({}, fail=RuntimeError("no device test-key-not-real")))
    with pytest.raises(tts.SpeechError) as caught:
        tts._play(b"pcm")
    assert "sounddevice playback failed" in str(caught.value) and "test-key-not-real" not in str(caught.value)


def test_windows_without_sounddevice_says_how_to_install_it(monkeypatch):
    importlib.reload(tts)
    monkeypatch.setattr(tts, "_use_sounddevice", lambda: True)
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import fails
    with pytest.raises(tts.SpeechError, match="pip install sounddevice"):
        tts._play(b"pcm")

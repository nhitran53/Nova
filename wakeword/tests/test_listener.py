import sys
import types

import numpy as np
import pytest

from conftest import silence, speech
from detector import Detection
from listener import Listener, MicFrames, WavFrames
from stt import SttError


class ScriptedDetector:
    """Fires when it sees a loud frame."""

    def __init__(self):
        self.resets = 0

    def process(self, frame):
        return Detection("hey_nova", 0.9) if np.abs(frame).max() > 1000 else None

    def reset(self):
        self.resets += 1


class FakeTranscriber:
    def __init__(self, error=None):
        self.consumed = []
        self.error = error

    def transcribe(self, frames):
        self.consumed.append(next(frames))  # takes the next frame, as the real one would
        if self.error:
            raise self.error
        return "hello"


def run(frames, transcriber, max_wakes=None):
    out = {"wakes": [], "transcripts": []}
    detector = ScriptedDetector()
    wakes = Listener(iter(frames), detector, transcriber, out["wakes"].append, out["transcripts"].append).run(max_wakes)
    return wakes, out, detector


def test_wake_then_transcript_then_back_to_listening():
    frames = silence(3) + speech(1) + silence(2) + speech(1) + silence(2)
    wakes, out, detector = run(frames, FakeTranscriber())
    assert wakes == 2
    assert out["transcripts"] == ["hello", "hello"]
    assert len(out["wakes"]) == 2 and detector.resets == 2


def test_max_wakes_stops_loop():
    wakes, out, _ = run(speech(1) + silence(1) + speech(1) + silence(1), FakeTranscriber(), max_wakes=1)
    assert wakes == 1 and out["transcripts"] == ["hello"]


def test_stt_error_is_reported_and_listening_continues(capsys):
    wakes, out, detector = run(speech(1) + silence(2) + speech(1) + silence(2), FakeTranscriber(SttError("nope")))
    assert wakes == 2 and out["transcripts"] == []
    assert capsys.readouterr().err.count("[stt error] nope") == 2
    assert detector.resets == 2


def test_wav_frames_pads_with_silence_and_chunks(tmp_path):
    import wave
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(speech(2)[0].tobytes() * 2)
    frames = list(WavFrames(path, trailing_silence_s=1.0))
    assert all(f.shape == (1280,) and f.dtype == np.int16 for f in frames)
    assert np.abs(frames[0]).max() > 0 and np.abs(frames[-1]).max() == 0


class FakeStream:
    """sounddevice.RawInputStream stand-in recording start/stop/close; start()/stop() can be made to raise."""

    def __init__(self, start_error=None, stop_error=None):
        self.start_error, self.stop_error = start_error, stop_error
        self.calls = []

    def start(self):
        self.calls.append("start")
        if self.start_error:
            raise self.start_error

    def stop(self):
        self.calls.append("stop")
        if self.stop_error:
            raise self.stop_error

    def close(self):
        self.calls.append("close")


@pytest.fixture
def mic_stream(monkeypatch):
    holder = {}

    def install(**kwargs):
        holder["stream"] = FakeStream(**kwargs)
        monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(RawInputStream=lambda **kw: holder["stream"]))
        return holder["stream"]

    return install


def test_mic_stream_is_started_and_closed(mic_stream):
    stream = mic_stream()
    with MicFrames():
        pass
    assert stream.calls == ["start", "stop", "close"]


def test_mic_stream_is_closed_when_start_raises(mic_stream):
    stream = mic_stream(start_error=OSError("device busy"))
    with pytest.raises(OSError, match="device busy"):
        with MicFrames():
            pytest.fail("body must not run")
    assert stream.calls == ["start", "stop", "close"]


def test_mic_stream_is_closed_when_stop_raises(mic_stream):
    stream = mic_stream(stop_error=OSError("already gone"))
    with pytest.raises(OSError, match="already gone"):
        with MicFrames():
            pass
    assert stream.calls == ["start", "stop", "close"]


def test_mic_stream_is_closed_when_the_body_raises(mic_stream):
    stream = mic_stream()
    with pytest.raises(KeyError):
        with MicFrames():
            raise KeyError("boom")
    assert stream.calls[-1] == "close"

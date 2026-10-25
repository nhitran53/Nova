import numpy as np

from conftest import silence, speech
from detector import Detection
from listener import Listener, WavFrames
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


def test_generated_followup_window_then_tool_returns_to_wake():
    class Followups:
        def __init__(self):
            self.timeouts = []
        def transcribe(self, frames, timeout_s=None):
            self.timeouts.append(timeout_s)
            next(frames)
            return 'sky color' if timeout_s is None else 'oxygen'
    stt = Followups()
    turns = []
    def route(text):
        turns.append(text)
        return {'expect_followup': text == 'sky color', 'listen_timeout_s': 5.0}
    detector = ScriptedDetector()
    Listener(iter(speech(1) + silence(3)), detector, stt, on_transcript=route).run(max_wakes=1)
    assert turns == ['sky color', 'oxygen']
    assert stt.timeouts == [None, 5.0] and detector.resets == 1


def test_followup_silence_returns_to_wake_waiting():
    class Followups:
        def __init__(self):
            self.calls = 0
        def transcribe(self, frames, timeout_s=None):
            self.calls += 1
            next(frames)
            return 'hello' if self.calls == 1 else ''
    stt = Followups()
    seen = []
    def route(text):
        seen.append(text)
        return {'expect_followup': True, 'listen_timeout_s': 5.0}
    detector = ScriptedDetector()
    Listener(iter(speech(1) + silence(3)), detector, stt, on_transcript=route).run(max_wakes=1)
    assert seen == ['hello', ''] and stt.calls == 2 and detector.resets == 1

import numpy as np
import pytest

from conftest import silence
from detector import WakeDetector, find_model_files


class FakeModel:
    def __init__(self, scores):
        self._scores = iter(scores)
        self.reset_calls = 0

    def predict(self, frame):
        return {"hey_nova": next(self._scores)}

    def reset(self):
        self.reset_calls += 1


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def make(scores, clock=None, **kwargs):
    model = FakeModel(scores)
    detector = WakeDetector(["hey_nova.onnx"], clock=clock or FakeClock(), model_factory=lambda models: model, **kwargs)
    return detector, model


def test_below_threshold_is_ignored():
    detector, _ = make([0.1, 0.49], threshold=0.5)
    assert [detector.process(f) for f in silence(2)] == [None, None]


def test_at_threshold_fires_with_name_and_score():
    detector, _ = make([0.5], threshold=0.5)
    detection = detector.process(silence()[0])
    assert (detection.name, detection.score) == ("hey_nova", 0.5)


def test_cooldown_suppresses_then_allows_again():
    clock = FakeClock()
    detector, _ = make([0.9, 0.95, 0.9], clock=clock, threshold=0.5, cooldown_s=2.0)
    frame = silence()[0]
    assert detector.process(frame) is not None
    clock.now += 1.9
    assert detector.process(frame) is None
    clock.now += 0.2  # 2.1s after the first detection
    assert detector.process(frame) is not None


def test_suppressed_frames_do_not_extend_cooldown():
    clock = FakeClock()
    detector, _ = make([0.9, 0.9, 0.9], clock=clock, cooldown_s=2.0)
    frame = silence()[0]
    detector.process(frame)
    clock.now += 1.5
    assert detector.process(frame) is None
    clock.now += 0.6  # 2.1s after the fire, 0.6s after the suppressed frame
    assert detector.process(frame) is not None


def test_reset_clears_model_state():
    detector, model = make([])
    detector.reset()
    assert model.reset_calls == 1


def test_rejects_non_int16_and_empty_models():
    detector, _ = make([0.0])
    with pytest.raises(ValueError):
        detector.process(np.zeros(1280, dtype=np.float32))
    with pytest.raises(ValueError):
        WakeDetector([], model_factory=lambda m: None)


def test_find_model_files(tmp_path):
    (tmp_path / "nova.onnx").touch()
    (tmp_path / "hey_nova.onnx").touch()
    (tmp_path / "notes.txt").touch()
    assert [p.rsplit("/", 1)[1] for p in find_model_files(tmp_path)] == ["hey_nova.onnx", "nova.onnx"]
    assert find_model_files(tmp_path / "missing") == []

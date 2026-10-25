import numpy as np
import pytest

from calibrate import CalibrationError, collect_rms, run_calibration, suggest_threshold


def frames(levels, n=1280):
    rng = np.random.default_rng(0)
    for level in levels:
        yield (rng.normal(0, level, n)).astype(np.int16)


def test_threshold_between_noise_and_speech():
    t = suggest_threshold(np.full(30, 100.0), np.full(30, 4000.0))
    assert 100 < t < 4000


def test_high_noise_floor_still_separates():
    t = suggest_threshold(np.full(30, 5000.0), np.full(30, 12000.0))
    assert 5000 < t < 12000


def test_rejects_speech_not_louder_than_noise():
    with pytest.raises(CalibrationError):
        suggest_threshold(np.full(30, 5000.0), np.full(30, 5500.0))


def test_collect_rms_stops_after_requested_duration():
    it = frames([100] * 100)
    values = collect_rms(it, 1.0, 1280, 16000)
    assert len(values) == 12
    assert next(it) is not None  # iterator not exhausted


def test_run_calibration_end_to_end():
    it = frames([100] * 25 + [4000] * 40)
    msgs = []
    t = run_calibration(it, 1280, 16000, say=msgs.append)
    assert 150 < t < 4000 and len(msgs) == 3


def test_no_frames_raises():
    with pytest.raises(CalibrationError):
        collect_rms(iter([]), 1.0, 1280, 16000)

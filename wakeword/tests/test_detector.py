import numpy as np
import pytest
import requests

from conftest import silence
from detector import FeatureModelError, WakeDetector, _loadable_onnx, ensure_feature_models, find_model_files


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


# --- ensure_feature_models: no real network (http_get is a fake requests.get) ---

URL = "https://example.invalid/models/melspectrogram.onnx"
GOOD = b"ONNX" + b"x" * 100


def looks_like_onnx(path):
    return path.is_file() and path.read_bytes().startswith(b"ONNX")


class FakeResponse:
    def __init__(self, chunks=(GOOD,), content_length=None, status=200, fail_after_chunks=None):
        self._chunks = list(chunks)
        self.headers = {} if content_length is None else {"content-length": str(content_length)}
        self._status = status
        self._fail_after_chunks = fail_after_chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self._status >= 400:
            raise requests.HTTPError(f"{self._status} Server Error")

    def iter_content(self, chunk_size):
        for i, chunk in enumerate(self._chunks):
            if self._fail_after_chunks is not None and i >= self._fail_after_chunks:
                raise requests.ConnectionError("connection reset")
            yield chunk


class FakeGet:
    """requests.get stand-in: returns the next scripted response (or raises it) and records the call arguments."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def ensure(tmp_path, get, **kwargs):
    ensure_feature_models(target_dir=tmp_path, urls=[URL], http_get=get, validate=kwargs.pop("validate", looks_like_onnx), **kwargs)


def test_download_succeeds_atomically_with_timeouts(tmp_path):
    get = FakeGet(FakeResponse([GOOD[:50], GOOD[50:]], content_length=len(GOOD)))
    ensure(tmp_path, get)
    assert (tmp_path / "melspectrogram.onnx").read_bytes() == GOOD
    assert [p.name for p in tmp_path.iterdir()] == ["melspectrogram.onnx"]  # temp file gone
    kwargs = get.calls[0][1]
    assert kwargs["stream"] is True and all(t > 0 for t in kwargs["timeout"])


def test_existing_valid_model_is_not_downloaded_again(tmp_path):
    (tmp_path / "melspectrogram.onnx").write_bytes(GOOD)
    get = FakeGet()
    ensure(tmp_path, get)
    assert get.calls == []


@pytest.mark.parametrize("bad", [b"", b"not an onnx model"])
def test_zero_byte_or_invalid_existing_file_counts_as_missing(tmp_path, bad):
    (tmp_path / "melspectrogram.onnx").write_bytes(bad)
    ensure(tmp_path, FakeGet(FakeResponse([GOOD])))
    assert (tmp_path / "melspectrogram.onnx").read_bytes() == GOOD


def test_offline_raises_and_leaves_nothing_behind(tmp_path):
    with pytest.raises(FeatureModelError, match="ConnectionError"):
        ensure(tmp_path, FakeGet(requests.ConnectionError("no network")))
    assert list(tmp_path.iterdir()) == []


def test_http_error_raises_and_leaves_nothing_behind(tmp_path):
    with pytest.raises(FeatureModelError, match="500"):
        ensure(tmp_path, FakeGet(FakeResponse([b"<html>oops</html>"], status=500)))
    assert list(tmp_path.iterdir()) == []


def test_truncated_transfer_is_rejected(tmp_path):
    with pytest.raises(FeatureModelError, match="truncated"):
        ensure(tmp_path, FakeGet(FakeResponse([GOOD[:40]], content_length=len(GOOD))))
    with pytest.raises(FeatureModelError, match="ConnectionError"):  # connection dropped mid-stream
        ensure(tmp_path, FakeGet(FakeResponse([GOOD[:40], GOOD[40:]], fail_after_chunks=1)))
    assert list(tmp_path.iterdir()) == []


def test_empty_and_unloadable_downloads_are_rejected(tmp_path):
    with pytest.raises(FeatureModelError, match="empty"):
        ensure(tmp_path, FakeGet(FakeResponse([b""])))
    with pytest.raises(FeatureModelError, match="loadable"):
        ensure(tmp_path, FakeGet(FakeResponse([b"<html>captive portal</html>"])))
    assert list(tmp_path.iterdir()) == []


def test_next_run_retries_after_a_failed_download(tmp_path):
    with pytest.raises(FeatureModelError):
        ensure(tmp_path, FakeGet(FakeResponse([GOOD[:10]], content_length=len(GOOD))))
    get = FakeGet(FakeResponse([GOOD]))
    ensure(tmp_path, get)
    assert len(get.calls) == 1 and (tmp_path / "melspectrogram.onnx").read_bytes() == GOOD


def test_default_validator_rejects_missing_empty_and_garbage_files(tmp_path):
    (tmp_path / "empty.onnx").write_bytes(b"")
    (tmp_path / "garbage.onnx").write_bytes(b"definitely not protobuf")
    assert not any(_loadable_onnx(tmp_path / name) for name in ("missing.onnx", "empty.onnx", "garbage.onnx"))

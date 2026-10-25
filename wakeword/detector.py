"""WakeDetector: openWakeWord wrapper with a score threshold and a post-detection cooldown."""
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Detection:
    name: str
    score: float


def find_model_files(models_dir):
    """All .onnx wake-word models in `models_dir` (sorted), or [] if the directory is missing."""
    directory = Path(models_dir)
    return sorted(str(p) for p in directory.glob("*.onnx")) if directory.is_dir() else []


class FeatureModelError(RuntimeError):
    """openWakeWord's shared feature models could not be downloaded or are unusable."""


FEATURE_DOWNLOAD_TIMEOUT_S = (5.0, 30.0)  # (connect, read) seconds


def _loadable_onnx(path):
    """True if `path` is a non-empty file onnxruntime can load as a model."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        import onnxruntime
        onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception:
        return False
    return True


def _download_atomically(url, destination, http_get, validate):
    """GET `url` into a temp sibling of `destination`, validate it, then os.replace it into place. The temp file is
    removed on any failure, so a half-written `destination` can never exist."""
    import requests
    destination = Path(destination)
    temp = destination.with_name(destination.name + ".part")
    try:
        try:
            with http_get(url, stream=True, timeout=FEATURE_DOWNLOAD_TIMEOUT_S) as response:
                response.raise_for_status()
                expected = response.headers.get("content-length")
                written = 0
                with open(temp, "wb") as out:
                    for chunk in response.iter_content(chunk_size=8192):
                        out.write(chunk)
                        written += len(chunk)
        except requests.RequestException as exc:
            raise FeatureModelError(f"download of {url} failed: {type(exc).__name__}: {exc}") from None
        if written == 0:
            raise FeatureModelError(f"download of {url} returned an empty file")
        if expected is not None and expected.isdigit() and int(expected) != written:
            raise FeatureModelError(f"download of {url} was truncated ({written} of {expected} bytes)")
        if not validate(temp):
            raise FeatureModelError(f"download of {url} is not a loadable ONNX model")
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def ensure_feature_models(target_dir=None, urls=None, http_get=None, validate=_loadable_onnx):
    """Download openWakeWord's shared melspectrogram + embedding models (~2 MB, once) into the installed package.

    The pip package does not ship them, and a custom hey_nova.onnx needs them. A model that is already present and
    loadable is left alone; a zero-byte or corrupt file counts as missing. Downloads have connect/read timeouts, fail on
    HTTP errors and truncated transfers, and are moved into place atomically, so a failed run leaves nothing half-written
    and the next run simply retries. Raises FeatureModelError on failure.
    """
    import requests
    if target_dir is None or urls is None:
        import openwakeword
        target_dir = target_dir or Path(openwakeword.__file__).parent / "resources" / "models"
        urls = urls or [f["download_url"].replace(".tflite", ".onnx") for f in openwakeword.FEATURE_MODELS.values()]
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    for url in urls:
        destination = target / url.rsplit("/", 1)[-1]
        if not validate(destination):
            _download_atomically(url, destination, http_get or requests.get, validate)


def _default_model_factory(models):
    from openwakeword.model import Model
    # onnx: openwakeword>=0.6 defaults to tflite, which has no wheels for python>=3.12.
    return Model(wakeword_models=list(models), inference_framework="onnx")


class WakeDetector:
    """Feed 16 kHz mono int16 frames (80 ms) to `process`; it returns a Detection or None.

    `models` are .onnx paths (custom models) or pretrained names such as "hey_jarvis".
    After a detection, further detections are suppressed for `cooldown_s` seconds.
    """

    def __init__(self, models, threshold=0.5, cooldown_s=2.0, clock=time.monotonic, model_factory=_default_model_factory):
        if not models:
            raise ValueError("WakeDetector needs at least one model")
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._model = model_factory(models)
        self._last_detection_at = None

    def process(self, frame):
        if frame.dtype != np.int16:
            raise ValueError(f"expected int16 audio, got {frame.dtype}")
        scores = self._model.predict(frame)  # also advances the model's streaming state during cooldown
        name, score = max(scores.items(), key=lambda item: item[1])
        if score < self.threshold:
            return None
        now = self._clock()
        if self._last_detection_at is not None and now - self._last_detection_at < self.cooldown_s:
            return None
        self._last_detection_at = now
        return Detection(name, float(score))

    def reset(self):
        """Clear the model's audio history so the speech that follows a wake word can't retrigger it."""
        self._model.reset()

"""WakeDetector: openWakeWord wrapper with a score threshold and a post-detection cooldown."""
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


def ensure_feature_models():
    """Download openWakeWord's shared melspectrogram + embedding models (~2 MB, once) into the installed package.

    The pip package does not ship them, and a custom hey_nova.onnx needs them. No-op when already present.
    """
    import openwakeword
    from openwakeword.utils import download_file
    target = Path(openwakeword.__file__).parent / "resources" / "models"
    target.mkdir(parents=True, exist_ok=True)
    for feature in openwakeword.FEATURE_MODELS.values():
        onnx_url = feature["download_url"].replace(".tflite", ".onnx")
        if not (target / onnx_url.rsplit("/", 1)[-1]).is_file():
            download_file(onnx_url, str(target))


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

"""Microphone calibration: measure room noise and speech levels, then suggest an RMS endpoint threshold."""
import numpy as np


class CalibrationError(RuntimeError):
    pass


def frame_rms(frame):
    return float(np.sqrt(np.mean(np.square(frame.astype(np.float64))))) if len(frame) else 0.0


def collect_rms(frames, seconds, frame_samples, sample_rate):
    """RMS of the next `seconds` of audio from an iterator of int16 frames."""
    needed = max(1, int(seconds * sample_rate / frame_samples))
    values = []
    for frame in frames:
        values.append(frame_rms(frame))
        if len(values) >= needed:
            break
    if not values:
        raise CalibrationError("no audio frames received")
    return np.array(values)


def suggest_threshold(noise, speech, min_ratio=1.5):
    """Geometric midpoint between the noise ceiling (p95) and typical speech (median of the loudest half).

    Raises CalibrationError when speech is not clearly louder than the room."""
    noise_ceiling = float(np.percentile(noise, 95))
    loud = np.sort(speech)[len(speech) // 2:]
    speech_level = float(np.median(loud))
    if speech_level < noise_ceiling * min_ratio:
        raise CalibrationError(
            f"speech ({speech_level:.0f}) is not clearly louder than room noise ({noise_ceiling:.0f}): "
            "wrong input device, mic gain too high, or something is playing; try --list-devices / --device")
    return float(np.sqrt(max(noise_ceiling, 1.0) * speech_level))


def run_calibration(frames, frame_samples, sample_rate, say=print, quiet_s=2.0, speak_s=3.0):
    say(f"calibrating: stay quiet for {quiet_s:.0f}s...")
    noise = collect_rms(frames, quiet_s, frame_samples, sample_rate)
    say(f"now speak normally for {speak_s:.0f}s (e.g. count out loud)...")
    speech = collect_rms(frames, speak_s, frame_samples, sample_rate)
    threshold = suggest_threshold(noise, speech)
    say(f"noise p95={np.percentile(noise, 95):.0f} speech median={np.median(speech):.0f} -> --rms-threshold {threshold:.0f}")
    return threshold

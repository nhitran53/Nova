import numpy as np

FRAME = 1280


def silence(n=1):
    return [np.zeros(FRAME, dtype=np.int16) for _ in range(n)]


def speech(n=1, amplitude=3000):
    t = np.arange(FRAME) / 16000
    tone = (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.int16)
    return [tone.copy() for _ in range(n)]

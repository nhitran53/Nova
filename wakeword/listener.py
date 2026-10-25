"""Microphone loop: wake word -> STT -> transcript -> back to listening."""
import logging
import queue
import sys

import numpy as np

from config import FRAME_SAMPLES, SAMPLE_RATE
from stt import SttError

log = logging.getLogger("nova.wakeword.listener")


class MicFrames:
    """Iterator of 80 ms, 16 kHz, mono int16 frames from the default (or given) input device.

    Use as a context manager. The stream stays open across wake/STT cycles so no audio is lost between them.
    """

    def __init__(self, device=None, frame_samples=FRAME_SAMPLES, sample_rate=SAMPLE_RATE, max_queued_frames=100):
        self._device = device
        self._frame_samples = frame_samples
        self._sample_rate = sample_rate
        self._queue = queue.Queue(maxsize=max_queued_frames)
        self._stream = None
        self.dropped_frames = 0

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.warning("audio stream status: %s", status)
        frame = np.frombuffer(indata, dtype=np.int16).copy()
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self.dropped_frames += 1  # consumer fell behind; drop rather than block the audio thread

    def __enter__(self):
        import sounddevice as sd
        self._stream = sd.RawInputStream(
            samplerate=self._sample_rate, channels=1, dtype="int16",
            blocksize=self._frame_samples, device=self._device, callback=self._callback)
        self._stream.start()
        return self

    def __exit__(self, *exc):
        self._stream.stop()
        self._stream.close()

    def __iter__(self):
        return self

    def __next__(self):
        return self._queue.get()

    def drain(self):
        """Discard queued (stale) frames."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return


class WavFrames:
    """Feed a 16 kHz mono int16 WAV through the same frame interface (no mic), followed by trailing silence."""

    def __init__(self, path, frame_samples=FRAME_SAMPLES, trailing_silence_s=6.0):
        import wave
        with wave.open(str(path), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                raise ValueError("WAV must be 16 kHz mono 16-bit")
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        audio = np.concatenate([audio, np.zeros(int(trailing_silence_s * SAMPLE_RATE), dtype=np.int16)])
        audio = audio[: len(audio) // frame_samples * frame_samples]
        self._frames = iter(audio.reshape(-1, frame_samples))

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._frames)

    def drain(self):
        pass


class Listener:
    def __init__(self, frames, detector, transcriber, on_wake=None, on_transcript=print):
        self._frames = iter(frames)
        self._drain = getattr(frames, "drain", lambda: None)
        self._detector = detector
        self._transcriber = transcriber
        self._on_wake = on_wake or (lambda detection: None)
        self._on_transcript = on_transcript

    def run(self, max_wakes=None):
        """Listen until the frame source ends or `max_wakes` wake words have been handled."""
        wakes = 0
        for frame in self._frames:
            detection = self._detector.process(frame)
            if detection is None:
                continue
            wakes += 1
            self._on_wake(detection)
            try:
                text = self._transcriber.transcribe(self._frames)
            except SttError as exc:
                print(f"[stt error] {exc}", file=sys.stderr)
            else:
                while text:
                    turn = self._on_transcript(text)
                    self._drain()  # discard microphone frames captured during TTS
                    if not isinstance(turn, dict) or turn.get("end_session") or not turn.get("expect_followup"):
                        break
                    try:
                        text = self._transcriber.transcribe(self._frames, timeout_s=turn.get("listen_timeout_s") or 10.0)
                    except SttError as exc:
                        print(f"[stt error] {exc}", file=sys.stderr)
                        break
                if not text:
                    self._on_transcript(text)
            self._detector.reset()
            self._drain()
            if max_wakes is not None and wakes >= max_wakes:
                return wakes
        return wakes

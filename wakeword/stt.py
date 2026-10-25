"""ElevenLabs Scribe speech-to-text for one utterance: realtime streaming, with batch fallback.

`Transcriber.transcribe(frames)` consumes 16 kHz mono int16 frames until `SilenceEndpointer` decides the
utterance is over, then returns the transcript text. The API key is never logged: every message that can
carry SDK text goes through `redact`.
"""
import asyncio
import base64
import io
import logging
import wave

import numpy as np

from config import SAMPLE_RATE

log = logging.getLogger("nova.wakeword.stt")

REALTIME_MODEL_ID = "scribe_v2_realtime"
BATCH_MODEL_ID = "scribe_v2"
MODES = ("auto", "realtime", "batch")
NETWORK_TIMEOUT_S = 20


class SttError(RuntimeError):
    pass


def redact(text, api_key):
    text = str(text)
    return text.replace(api_key, "[redacted]") if api_key else text


class RedactingFilter(logging.Filter):
    """Logging filter that scrubs the API key from every record (install on handlers)."""

    def __init__(self, api_key):
        super().__init__()
        self._api_key = api_key

    def filter(self, record):
        record.msg = redact(record.getMessage(), self._api_key)
        record.args = None
        return True


class SilenceEndpointer:
    """Energy-based end-of-utterance detection.

    Ends when speech (frame RMS >= rms_threshold) has been heard and then `silence_s` of quiet follows;
    or when no speech starts within `no_speech_timeout_s`; or at `max_s` total.
    """

    def __init__(self, sample_rate=SAMPLE_RATE, silence_s=1.5, rms_threshold=400.0, no_speech_timeout_s=5.0, max_s=30.0):
        self.sample_rate = sample_rate
        self.silence_s = silence_s
        self.rms_threshold = rms_threshold
        self.no_speech_timeout_s = no_speech_timeout_s
        self.max_s = max_s
        self.heard_speech = False
        self.done = False
        self._silence = 0.0
        self._elapsed = 0.0

    def feed(self, frame):
        """Account for one frame; True once the utterance is over."""
        self.done = self._feed(frame)
        return self.done

    def _feed(self, frame):
        duration = len(frame) / self.sample_rate
        self._elapsed += duration
        rms = float(np.sqrt(np.mean(np.square(frame.astype(np.float64))))) if len(frame) else 0.0
        if rms >= self.rms_threshold:
            self.heard_speech = True
            self._silence = 0.0
        else:
            self._silence += duration
        if self._elapsed >= self.max_s:
            return True
        if self.heard_speech:
            return self._silence >= self.silence_s
        return self._elapsed >= self.no_speech_timeout_s


def _default_client_factory(api_key):
    from elevenlabs.client import ElevenLabs
    return ElevenLabs(api_key=api_key, timeout=NETWORK_TIMEOUT_S)


def _to_wav_bytes(pcm, sample_rate):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


class Transcriber:
    def __init__(self, api_key, mode="auto", endpointer_factory=SilenceEndpointer, client_factory=_default_client_factory,
                 sample_rate=SAMPLE_RATE, commit_timeout_s=10.0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self._api_key = api_key
        self._mode = mode
        self._endpointer_factory = endpointer_factory
        self._client_factory = client_factory
        self._sample_rate = sample_rate
        self._commit_timeout_s = commit_timeout_s

    def transcribe(self, frames, timeout_s=None):
        """Consume frames (an iterator) up to end of utterance; return the transcript ('' if nothing was said)."""
        endpointer = self._endpointer_factory()
        if timeout_s is not None:
            endpointer.no_speech_timeout_s = timeout_s
        recorded = []
        try:
            client = self._client_factory(self._api_key)
        except Exception as exc:
            raise SttError(f"could not create the ElevenLabs client: {type(exc).__name__}") from None
        if self._mode != "batch":
            try:
                return asyncio.run(self._transcribe_realtime(client, frames, endpointer, recorded))
            except Exception as exc:
                reason = redact(f"{type(exc).__name__}: {exc}", self._api_key)
                if self._mode == "realtime":
                    raise SttError(f"realtime transcription failed: {reason}") from None
                log.warning("realtime transcription failed (%s); falling back to batch", reason)
        return self._transcribe_batch(client, frames, endpointer, recorded)

    async def _transcribe_realtime(self, client, frames, endpointer, recorded):
        from elevenlabs.realtime import AudioFormat, CommitStrategy, RealtimeEvents

        loop = asyncio.get_running_loop()
        committed = loop.create_future()

        def on_committed(data):
            if not committed.done():
                committed.set_result((data.get("text") or data.get("transcript") or "").strip())

        def on_error(data):
            if not committed.done():
                committed.set_exception(SttError(redact(f"server error: {data}", self._api_key)))

        connection = await asyncio.wait_for(client.speech_to_text.realtime.connect({
            "model_id": REALTIME_MODEL_ID,
            "audio_format": AudioFormat.PCM_16000,
            "sample_rate": self._sample_rate,
            "commit_strategy": CommitStrategy.MANUAL,
        }), NETWORK_TIMEOUT_S)
        try:
            connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed)
            connection.on(RealtimeEvents.ERROR, on_error)
            while True:
                frame = await asyncio.to_thread(next, frames, None)
                if frame is None:
                    break
                recorded.append(frame)
                await connection.send({"audio_base_64": base64.b64encode(frame.tobytes()).decode("ascii")})
                if committed.done() or endpointer.feed(frame):
                    break
            if committed.done():  # an error arrived mid-stream
                return committed.result()
            if not endpointer.heard_speech:
                return ""
            await connection.commit()
            return await asyncio.wait_for(committed, self._commit_timeout_s)
        finally:
            await connection.close()

    def _transcribe_batch(self, client, frames, endpointer, recorded):
        """Record until the endpointer fires (continuing any partial realtime recording), then transcribe once."""
        if not endpointer.done:
            for frame in frames:
                recorded.append(frame)
                if endpointer.feed(frame):
                    break
        if not endpointer.heard_speech or not recorded:
            return ""
        wav = _to_wav_bytes(np.concatenate(recorded), self._sample_rate)
        try:
            result = client.speech_to_text.convert(model_id=BATCH_MODEL_ID, file=("utterance.wav", wav, "audio/wav"))
        except Exception as exc:
            raise SttError(redact(f"batch transcription failed: {type(exc).__name__}: {exc}", self._api_key)) from None
        return (getattr(result, "text", "") or "").strip()

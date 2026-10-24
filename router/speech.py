"""Speak a line through ElevenLabs text-to-speech and play it on the local speakers.

    from router import speech
    speech.init()                          # optional: create the client at startup (otherwise lazy on first speak)
    speech.speak("Oxygen at 87.5 percent.")     # blocks until playback ends
    uv run python -m router.speech "Nova speech test."   # live check: real API call + real playback

* One ElevenLabs client per process, created under a lock by `init()` / the first `speak()`; never per call.
  Importing this module does nothing (no .env read, no client, no API call).
* The key comes from `router/.env` (path resolved from this file, not the cwd) or the process environment, as
  ELEVENLABS_API_KEY. It is never printed or put in an error message.
* Voice / model are env-overridable: ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL_ID (defaults below).
* Audio is requested as raw 16-bit mono PCM and piped to `paplay` (PulseAudio/PipeWire) or `aplay`.
* Every network call and every subprocess has a timeout.
"""
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).parent / ".env"
KEY_VAR = "ELEVENLABS_API_KEY"
DEFAULT_VOICE_ID = "VmPkIaPdP997cjIfBWRB"
DEFAULT_MODEL_ID = "eleven_v4"  # accepted by the API (a bogus id gets 400 model_not_found); see README
PCM_SAMPLE_RATE = 24000
OUTPUT_FORMAT = f"pcm_{PCM_SAMPLE_RATE}"  # raw signed 16-bit little-endian mono
NETWORK_TIMEOUT_S = 20
PLAYBACK_TIMEOUT_S = 60


class SpeechError(RuntimeError):
    """Any speech failure (missing key, API error, no audio player, player failure). The message never holds the key."""


def _default_client_factory(api_key):
    from elevenlabs.client import ElevenLabs
    return ElevenLabs(api_key=api_key, timeout=NETWORK_TIMEOUT_S)


_client_factory = _default_client_factory  # tests replace this with a fake
_client = None
_client_lock = threading.Lock()


def init():
    """Create the singleton client now (call once at startup) and return it. Idempotent; thread-safe."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                load_dotenv(ENV_PATH)  # does not override variables already set in the environment
                api_key = os.environ.get(KEY_VAR, "").strip()
                if not api_key:
                    raise SpeechError(f"{KEY_VAR} is not set: put it in {ENV_PATH} or the environment")
                _client = _client_factory(api_key)
    return _client


def reset():
    """Drop the singleton (tests only)."""
    global _client
    with _client_lock:
        _client = None


def _synthesize(text):
    client = init()
    try:
        chunks = client.text_to_speech.convert(
            voice_id=os.environ.get("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID), text=text,
            model_id=os.environ.get("ELEVENLABS_MODEL_ID", DEFAULT_MODEL_ID), output_format=OUTPUT_FORMAT,
            request_options={"timeout_in_seconds": NETWORK_TIMEOUT_S})
        audio = b"".join(chunks)
    except Exception as exc:  # never forward str(exc): SDK errors carry request headers
        status = getattr(exc, "status_code", None)
        detail = getattr(exc, "body", None)
        message = detail.get("detail", {}).get("message") if isinstance(detail, dict) and isinstance(detail.get("detail"), dict) else None
        raise SpeechError(f"ElevenLabs request failed: {type(exc).__name__}" + (f" status={status}" if status else "")
                          + (f" ({message})" if message else "")) from None
    if not audio:
        raise SpeechError("ElevenLabs returned no audio")
    return audio


def _player_command():
    if shutil.which("paplay"):
        return ["paplay", "--raw", f"--rate={PCM_SAMPLE_RATE}", "--format=s16le", "--channels=1"]
    if shutil.which("aplay"):
        return ["aplay", "-q", "-t", "raw", "-r", str(PCM_SAMPLE_RATE), "-f", "S16_LE", "-c", "1"]
    raise SpeechError("no audio player found (need paplay or aplay)")


def _play(audio):
    """Pipe raw PCM into the player; -> {"player","returncode"}. Raises SpeechError on a non-zero exit or a timeout."""
    command = _player_command()
    try:
        done = subprocess.run(command, input=audio, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              timeout=PLAYBACK_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        raise SpeechError(f"{command[0]} timed out after {PLAYBACK_TIMEOUT_S}s") from None
    if done.returncode != 0:
        raise SpeechError(f"{command[0]} exited {done.returncode}: {done.stderr.decode(errors='replace').strip()[:200]}")
    return {"player": command[0], "returncode": done.returncode}


def speak(output, route=None, block=True):
    """Speak `output` (the orchestrator's final spoken line). Empty/whitespace -> no-op, returns None.

    -> {"bytes": audio size, "player": name, "returncode": 0}; with block=False the audio is fetched here but
    played on a daemon thread, and the dict has "returncode": None (playback errors are then not reported).
    `route` is accepted so the orchestrator can pass it; it does not change the voice (yet).
    Raises SpeechError on any failure."""
    if not isinstance(output, str) or not output.strip():
        return None
    audio = _synthesize(output.strip())
    if block:
        return {"bytes": len(audio), **_play(audio)}
    threading.Thread(target=lambda: _play_quietly(audio), daemon=True).start()
    return {"bytes": len(audio), "player": _player_command()[0], "returncode": None}


def _play_quietly(audio):
    try:
        _play(audio)
    except SpeechError as exc:
        print(f"speech playback failed: {exc}", file=sys.stderr)


def main(argv=None):
    text = " ".join(argv if argv is not None else sys.argv[1:]) or "Nova speech test. Oxygen at 87.5 percent."
    try:
        info = speak(text)
    except SpeechError as exc:
        print(f"speech FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"spoke {text!r}: {info['bytes']} bytes of {OUTPUT_FORMAT} audio, played with {info['player']}, exit code {info['returncode']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

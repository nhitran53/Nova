"""Speech-to-text front end: ElevenLabs Scribe realtime from the mic, router model loaded in the background.

`listen_and_route()` (or `python -m router.stt`) starts loading the BERT router on a background thread, opens the
microphone and streams it to Scribe at once, so the model warms up while you speak. The first committed (VAD-ended)
transcript is passed to `TurnHandler.handle` as soon as the router is ready.

`converse()` / the CLI loop it: listen, route, speak the reply, listen again.

Mic capture needs `sounddevice` (`uv run --with sounddevice python -m router.stt`). The API key comes from
`router/.env` or the environment and never appears in an error message.
"""
import argparse
import asyncio
import base64
import os
import queue
import sys
import threading
import warnings
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
ENV_PATH = HERE / ".env"
KEY_VAR = "ELEVENLABS_API_KEY"
MODEL_ID = "scribe_v2_realtime"
SAMPLE_RATE = 16000
FRAME_S = 0.1
CONNECT_TIMEOUT_S = 20
VAD_SILENCE_S = 1.0  # quiet after speech that ends an utterance (API range 0.3-3.0)
DEFAULT_LISTEN_TIMEOUT_S = 30.0


class SttError(RuntimeError):
    pass


def _api_key():
    load_dotenv(ENV_PATH)  # does not override variables already set in the environment
    key = os.environ.get(KEY_VAR, "").strip()
    if not key:
        raise SttError(f"{KEY_VAR} is not set: put it in {ENV_PATH} or the environment")
    return key


def _redacted(text, key):
    return str(text).replace(key, "[redacted]") if key else str(text)


def _default_client_factory(api_key):
    from elevenlabs.client import ElevenLabs
    return ElevenLabs(api_key=api_key, timeout=CONNECT_TIMEOUT_S)


def mic_frames(stop, sample_rate=SAMPLE_RATE, frame_s=FRAME_S):
    """Yield mono int16 frames (bytes) from the default microphone until `stop` (threading.Event) is set."""
    try:
        import sounddevice
    except ImportError:
        raise SttError("microphone capture needs sounddevice: run with `uv run --with sounddevice ...`") from None
    chunks = queue.Queue()
    stream = sounddevice.RawInputStream(samplerate=sample_rate, channels=1, dtype="int16",
                                        blocksize=int(sample_rate * frame_s),
                                        callback=lambda data, frames, time, status: chunks.put(bytes(data)))
    with stream:
        while not stop.is_set():
            try:
                yield chunks.get(timeout=0.5)
            except queue.Empty:
                continue


def transcribe(frames, client_factory=_default_client_factory, timeout_s=DEFAULT_LISTEN_TIMEOUT_S,
               sample_rate=SAMPLE_RATE, on_partial=None):
    """Stream `frames` (iterable of int16 PCM bytes) to Scribe realtime; return the first committed transcript.

    Returns "" when nothing was said within `timeout_s`. Raises SttError on any failure."""
    key = _api_key()
    try:
        client = client_factory(key)
    except Exception as exc:  # never forward str(exc): the key was an argument
        raise SttError(f"could not create the ElevenLabs client: {type(exc).__name__}") from None
    try:
        return asyncio.run(_stream(client, frames, timeout_s, sample_rate, on_partial))
    except SttError as exc:
        raise SttError(_redacted(exc, key)) from None
    except Exception as exc:
        raise SttError(_redacted(f"transcription failed: {type(exc).__name__}: {exc}", key)) from None


async def _stream(client, frames, timeout_s, sample_rate, on_partial):
    from elevenlabs.realtime import AudioFormat, CommitStrategy, RealtimeEvents

    loop = asyncio.get_running_loop()
    done = loop.create_future()

    def on_committed(data):
        text = (data.get("text") or data.get("transcript") or "").strip()
        if text and not done.done():
            done.set_result(text)

    def on_error(data):
        if not done.done():
            done.set_exception(SttError(f"server error: {data}"))

    connection = await asyncio.wait_for(client.speech_to_text.realtime.connect({
        "model_id": MODEL_ID,
        "audio_format": AudioFormat.PCM_16000,
        "sample_rate": sample_rate,
        "commit_strategy": CommitStrategy.VAD,
        "vad_silence_threshold_secs": VAD_SILENCE_S,
    }), CONNECT_TIMEOUT_S)
    stop = threading.Event()
    frames = iter(frames)
    try:
        connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed)
        connection.on(RealtimeEvents.ERROR, on_error)
        if on_partial:
            connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, lambda data: on_partial(data.get("text", "")))

        async def pump():
            while not stop.is_set():
                frame = await asyncio.to_thread(next, frames, None)
                if frame is None:
                    break
                await connection.send({"audio_base_64": base64.b64encode(frame).decode("ascii")})
            if not done.done():  # frames ended (e.g. a file): flush what is buffered
                await connection.commit()

        pump_task = asyncio.create_task(pump())
        try:
            return await asyncio.wait_for(done, timeout_s)
        except asyncio.TimeoutError:
            return ""
        finally:
            stop.set()
            pump_task.cancel()
    finally:
        await connection.close()


class BackgroundRouter:
    """Loads the router checkpoint on a thread; `.get()` blocks until it is ready (re-raises a load failure)."""

    def __init__(self, loader):
        self._ready = threading.Event()
        self._router = self._error = None
        threading.Thread(target=self._load, args=(loader,), daemon=True, name="router-load").start()

    def _load(self, loader):
        try:
            self._router = loader()
        except BaseException as exc:
            self._error = exc
        finally:
            self._ready.set()

    def get(self, timeout=None):
        if not self._ready.wait(timeout):
            raise SttError("router did not finish loading in time")
        if self._error is not None:
            raise self._error
        return self._router


def _load_router(model, device):
    from .core import Router
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Router.load(model, HERE / "labels.json", device=device)


def _default_handler(router, speak):
    from . import tts
    from .tools import SimState, default_registry
    from .turn import TurnHandler
    sim = SimState()
    return TurnHandler(router, sim.snapshot, default_registry(sim), tts_fn=tts.speak if speak else None)


def listen_and_route(model=None, device="cpu", speak=False, frames=None, router_loader=None, handler_factory=None,
                     timeout_s=DEFAULT_LISTEN_TIMEOUT_S, client_factory=_default_client_factory, on_partial=None,
                     background=None, handler=None):
    """Transcribe one utterance (router loading in the background), run it through the TurnHandler, speak if `speak`.

    -> {"transcript": str, "turn": TurnHandler result or None (nothing heard)}.
    `frames` defaults to the microphone. `background`/`handler` let `converse` reuse one router and one handler
    (so clarification/confirmation state carries across turns); `router_loader`/`handler_factory` are for tests."""
    from . import core

    if background is None:
        if router_loader is None:
            model = model or str(core.models_dir() / "v1")
            router_loader = lambda: _load_router(model, device)
        background = BackgroundRouter(router_loader)
    stop = threading.Event()
    source = mic_frames(stop) if frames is None else frames
    try:
        transcript = transcribe(source, client_factory=client_factory, timeout_s=timeout_s, on_partial=on_partial)
    finally:
        stop.set()
    if not transcript:
        return {"transcript": "", "turn": None}
    if handler is None:
        router = background.get()
        handler = handler_factory(router) if handler_factory else _default_handler(router, speak)
    return {"transcript": transcript, "turn": handler.handle(transcript)}


class _LazyHandler:
    """Builds the TurnHandler on first use (waiting for the background router) and keeps it for the session."""

    def __init__(self, background, factory):
        self._background, self._factory, self._handler = background, factory, None

    def handle(self, text):
        if self._handler is None:
            self._handler = self._factory(self._background.get())
        return self._handler.handle(text)


def converse(model=None, device="cpu", speak=True, timeout_s=10.0, max_turns=None, on_event=print,
             router_loader=None, handler_factory=None, **kwargs):
    """Back and forth: listen -> route -> speak the reply -> listen again, until `timeout_s` of silence
    (or `max_turns`). One router and one TurnHandler for the whole conversation. -> list of turn dicts."""
    from . import core

    loader = router_loader or (lambda: _load_router(model or str(core.models_dir() / "v1"), device))
    background = BackgroundRouter(loader)
    handler = _LazyHandler(background, handler_factory or (lambda router: _default_handler(router, speak)))
    turns = []
    while max_turns is None or len(turns) < max_turns:
        on_event("listening...")
        out = listen_and_route(timeout_s=timeout_s, background=background, handler=handler, **kwargs)
        if out["turn"] is None:
            on_event("(silence - ending conversation)")
            break
        turns.append(out)
        turn = out["turn"]
        on_event(f"you  : {out['transcript']}\nnova : {turn['spoken']}  [{turn['route']}]"
                 + (f"  (tts failed: {turn['tts_error']})" if "tts_error" in turn else ""))
    return turns


def main(argv=None):
    ap = argparse.ArgumentParser(description="Talk to Nova: mic -> ElevenLabs Scribe -> router -> ElevenLabs TTS, repeated.")
    ap.add_argument("--model", default=None, help="router checkpoint (default: $NOVA_MODELS_DIR/v1)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-speak", action="store_true", help="print replies instead of speaking them")
    ap.add_argument("--once", action="store_true", help="a single turn instead of a conversation")
    ap.add_argument("--timeout", type=float, default=10.0, help="end the conversation after this many seconds of silence")
    args = ap.parse_args(argv)
    print("router loading in background; speak when you see 'listening...' (Ctrl-C to quit)", file=sys.stderr)
    try:
        converse(args.model, args.device, speak=not args.no_speak, timeout_s=args.timeout,
                 max_turns=1 if args.once else None)
    except KeyboardInterrupt:
        print()
    except Exception as exc:
        print(f"stt FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

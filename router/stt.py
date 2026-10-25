"""Run the PC voice conversation with ElevenLabs Scribe and the NOVA router.

Load the router in the background while listening.
Speak each verified reply before opening the microphone again.
Silence or a stop request ends the session. Barge-in is not supported.
Read keys from the environment or NOVA_ENV_FILE with the root .env as the default.
"""
import argparse
import asyncio
import base64
import os
import queue
import itertools
import sys
import threading
import warnings
from pathlib import Path

from dotenv import load_dotenv

from .llm_generator import API_KEY_ENV, RUNTIMES, redact_secrets

HERE = Path(__file__).resolve().parent
ENV_PATH = Path(os.environ.get("NOVA_ENV_FILE") or HERE.parent / ".env")
KEY_VAR = "ELEVENLABS_API_KEY"
MODEL_ID = "scribe_v2_realtime"
LANGUAGE_CODE = "en"  # Keep noisy speech from switching languages.
SAMPLE_RATE = 16000
FRAME_S = 0.1
CONNECT_TIMEOUT_S = 20
VAD_SILENCE_S = 1.0  # quiet after speech that ends an utterance (API range 0.3-3.0)
DEFAULT_LISTEN_TIMEOUT_S = 30.0
STOP_POLL_S = 0.1  # how often a pending transcription checks its stop_event


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
               sample_rate=SAMPLE_RATE, on_partial=None, stop_event=None):
    """Stream `frames` (iterable of int16 PCM bytes) to Scribe realtime; return the first committed transcript.

    Returns "" when nothing was said within `timeout_s` or when `stop_event` (threading.Event, optional) is set while waiting.
    Raises SttError on any failure."""
    key = _api_key()
    try:
        client = client_factory(key)
    except Exception as exc:  # never forward str(exc): the key was an argument
        raise SttError(f"could not create the ElevenLabs client: {type(exc).__name__}") from None
    try:
        return asyncio.run(_stream(client, frames, timeout_s, sample_rate, on_partial, stop_event))
    except SttError as exc:
        raise SttError(_redacted(exc, key)) from None
    except Exception as exc:
        raise SttError(_redacted(f"transcription failed: {type(exc).__name__}: {exc}", key)) from None


async def _stream(client, frames, timeout_s, sample_rate, on_partial, stop_event=None):
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
        "language_code": LANGUAGE_CODE,
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
            if stop_event is None:
                return await asyncio.wait_for(done, timeout_s)
            deadline = loop.time() + timeout_s
            while not stop_event.is_set():  # poll so a stop request ends the wait within STOP_POLL_S
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return ""
                try:
                    return await asyncio.wait_for(asyncio.shield(done), min(STOP_POLL_S, remaining))
                except asyncio.TimeoutError:
                    continue
            return ""
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


class SpeechQueue:
    """Single-worker priority queue around a blocking speak function `speak_fn(text, route, block=True)` (tts.speak).

    Priority lines precede queued normal lines (FIFO within each priority). Alerts wait for current playback;
    no mid-playback interruption is attempted. `enqueue` returns at once.
    A failing line does not stop the worker: its error is kept and handed out by `drain()`, never swallowed."""

    def __init__(self, speak_fn):
        self._speak_fn = speak_fn
        self._lines = queue.PriorityQueue()
        self._sequence = itertools.count()
        self._errors = []
        self._errors_lock = threading.Lock()
        threading.Thread(target=self._work, daemon=True, name="speech-queue").start()

    def enqueue(self, text, route=None, priority=False):
        self._lines.put((0 if priority else 1, next(self._sequence), (text, route)))

    def clear_pending(self):
        """Discard queued non-priority lines, never current playback or queued alerts."""
        with self._lines.mutex:
            kept = [item for item in self._lines.queue if item[0] != 1]
            removed = len(self._lines.queue) - len(kept)
            self._lines.queue[:] = sorted(kept)
            self._lines.unfinished_tasks -= removed
            if not self._lines.unfinished_tasks:
                self._lines.all_tasks_done.notify_all()

    def _work(self):
        while True:
            _, _, item = self._lines.get()
            try:
                if item is None:
                    return
                text, route = item
                self._speak_fn(text, route, block=True)
            except Exception as exc:
                with self._errors_lock:
                    self._errors.append(f"{type(exc).__name__}: {exc}"[:300])
            finally:
                self._lines.task_done()

    def drain(self):
        """Block until every enqueued line has been spoken; -> the list of error strings since the last drain (cleared)."""
        self._lines.join()
        with self._errors_lock:
            errors, self._errors = self._errors, []
        return errors

    def close(self):
        self._lines.put((2, next(self._sequence), None))


def default_speech_queue():
    from . import tts
    return SpeechQueue(tts.speak)


def router_tool_labels():
    from .pipeline import load_config
    return tuple(load_config()["tool_labels"])


def _default_handler(router, speak, llm_options=None, speech=None, datasource=None, vision_frames=None):
    """Build the turn handler with queued speech and the selected datasource.

    vision_frames supplies Unity captures and live positions. Its tools are added without changing BERT's labels.
    """
    from .llm_generator import build_generator
    from .tools import SimState, default_registry
    from .turn import TurnHandler, load_policy
    options = llm_options or {}
    generator, label = build_generator(
        lambda line: print(line, file=sys.stderr), no_llm=options.get("no_llm", False), runtime=options.get("runtime"),
        base_url=options.get("base_url"), model=options.get("llm_model"))
    policy = load_policy()
    if hasattr(generator, "classify"):  # Let the live model answer unmatched general questions.
        policy["llm_scope"]["general_fallback"] = True
    if options.get("deadline_ms") is not None:
        policy["generation"]["deadline_ms"] = options["deadline_ms"]
    if speak and speech is None:
        speech = default_speech_queue()
    from .retrieval import make_retriever
    from .demo_scene import DEFAULT_DOCS_DIR, demo_scene_enabled
    from .tools import DEMO_ROVER_FAULTS
    demo = demo_scene_enabled(policy, options, os.environ)  # unset flag/env/option == unchanged behaviour
    docs_dir = options.get("docs_dir") or os.environ.get("NOVA_DOCS_DIR") or (str(DEFAULT_DOCS_DIR) if demo else None)
    retriever = make_retriever(docs_dir) if docs_dir else None
    if demo:
        policy["demo_scene"]["enabled"] = True
    unity_tools = {}
    if vision_frames is not None and hasattr(vision_frames, "request_spatial"):
        from .location import make_location_tool
        unity_tools["location"] = make_location_tool(vision_frames)
        unity_tools["return_to_start"] = True
    if vision_frames is not None and policy.get("vision", {}).get("enabled"):
        from .vision import make_vision_tool
        manual = retriever or make_retriever(str(DEFAULT_DOCS_DIR))
        unity_tools["vision"] = make_vision_tool(vision_frames, generator, policy["vision"], manual)
    if datasource is not None:
        alert_source, registry = datasource.snapshot, default_registry(datasource=datasource, **unity_tools)
    else:
        sim = SimState(rover_faults=DEMO_ROVER_FAULTS if demo else None)
        alert_source, registry = sim.snapshot, default_registry(sim, **unity_tools)
    def deliver(text, route):
        if route == "alert":
            speech.clear_pending()
            speech.enqueue(text, route, priority=True)
        else:
            speech.enqueue(text, route)

    classifier = pipeline_config = None
    layer = policy.get("second_layer", {})
    if layer.get("enabled") and hasattr(generator, "classify"):
        from .pipeline import load_config
        from .tool_classifier import make_tool_classifier
        labels = router_tool_labels()
        if unity_tools:  # Only the second layer can name the added labels.
            pipeline_config = load_config()
        if "vision" in unity_tools:
            pipeline_config["tool_labels"]["call_vision"] = {"tool": "vision", "spoken": "a look at what you see"}
            labels += ("call_vision",)
        if "location" in unity_tools:
            pipeline_config["tool_labels"]["call_location"] = {"tool": "get_current_location", "spoken": "your current Unity location"}
            labels += ("call_location",)
        classifier = make_tool_classifier(generator, labels, layer)
    handler = TurnHandler(router, alert_source, registry, generator=generator, tool_classifier_fn=classifier, policy=policy, retriever=retriever,
                          pipeline_config=pipeline_config,
                          interim_fn=speech.enqueue if speak else None, tts_fn=deliver if speak else None)
    handler.generator_label = label
    if vision_frames is not None and hasattr(vision_frames, "request_spatial"):
        def place_distance(landmark):
            snapshot = vision_frames.request_spatial(timeout_s=3)
            return next((p["distance_m"] for p in snapshot.get("nearby_landmarks", []) if p["id"] == landmark), None)
        handler.place_distance_fn = place_distance
    return handler


def _speech_errors(speech, turn):
    """Wait for `speech` to finish; record any playback error on `turn["tts_error"]` (never dropped). -> the error list."""
    errors = speech.drain()
    if errors:
        turn["tts_error"] = "; ".join(([turn["tts_error"]] if "tts_error" in turn else []) + errors)
    return errors


def listen_and_route(model=None, device="cpu", speak=False, frames=None, router_loader=None, handler_factory=None,
                     timeout_s=DEFAULT_LISTEN_TIMEOUT_S, client_factory=_default_client_factory, on_partial=None,
                     background=None, handler=None, llm_options=None, stop_event=None, on_transcript=None):
    """Transcribe one utterance (router loading in the background), run it through the TurnHandler, speak if `speak`.

    -> {"transcript": str, "turn": TurnHandler result or None (nothing heard)}.
    `frames` defaults to the microphone. `background`/`handler` let `converse` reuse one router and one handler
    (so clarification/confirmation state carries across turns); `router_loader`/`handler_factory` are for tests.
    `llm_options` (no_llm, runtime, base_url, llm_model, deadline_ms, docs_dir) configure the default handler's generator. When this
    call builds the default handler itself it waits for the speech queue before returning (converse owns its own queue).
    `stop_event` (threading.Event, optional): setting it ends the listening early; nothing heard -> turn None.
    `on_transcript(text)` (optional) is called once the utterance is committed, before routing/tools/generation start."""
    from . import core

    if background is None:
        if router_loader is None:
            model = model or str(core.models_dir() / "v1")
            router_loader = lambda: _load_router(model, device)
        background = BackgroundRouter(router_loader)
    stop = threading.Event()
    source = mic_frames(stop) if frames is None else frames
    try:
        transcript = transcribe(source, client_factory=client_factory, timeout_s=timeout_s, on_partial=on_partial,
                                stop_event=stop_event)
    finally:
        stop.set()
    if not transcript:
        return {"transcript": "", "turn": None}
    if on_transcript is not None:
        on_transcript(transcript)
    own_speech = None
    if handler is None:
        router = background.get()
        if handler_factory:
            handler = handler_factory(router)
        else:
            own_speech = default_speech_queue() if speak else None
            handler = _default_handler(router, speak, llm_options, own_speech)
    turn = handler.handle(transcript)
    if own_speech is not None:
        _speech_errors(own_speech, turn)
        own_speech.close()
    return {"transcript": transcript, "turn": turn}


class _LazyHandler:
    """Builds the TurnHandler on first use (waiting for the background router) and keeps it for the session."""

    def __init__(self, background, factory):
        self._background, self._factory, self._handler = background, factory, None

    def handle(self, text):
        if self._handler is None:
            self._handler = self._factory(self._background.get())
        return self._handler.handle(text)


def converse(model=None, device="cpu", speak=True, timeout_s=10.0, max_turns=None, on_event=print,
             router_loader=None, handler_factory=None, llm_options=None, speech_queue=None, stop_event=None, on_turn=None,
             on_transcript=None, datasource=None, vision_frames=None, **kwargs):
    """Back and forth: listen -> route -> speak the reply -> listen again, until `timeout_s` of silence, `max_turns`, or a
    turn with `end_session` ("thank you"). One router, one TurnHandler and one SpeechQueue for the whole conversation; the queue
    is drained before the mic reopens. `speech_queue` is for tests (default: a real tts-backed queue when `speak` and no
    custom `handler_factory`). Playback errors go to on_event and turn["tts_error"]. -> list of turn dicts.
    `stop_event` (threading.Event, optional) ends the conversation: checked before each listen and while listening; a reply
    already being spoken is not interrupted. `on_turn(out)` (optional) gets each {"transcript", "turn"} dict once its reply is
    enqueued, before the speech is awaited (structured alternative to parsing the on_event text). `vision_frames` (optional)
    gives the default handler Unity headset images (see `_default_handler`). `on_transcript(text)`
    (optional) gets each committed utterance before it is routed, i.e. when processing starts. `datasource` (optional)
    replaces the built-in SimState for the default handler (see `_default_handler`)."""
    from . import core

    loader = router_loader or (lambda: _load_router(model or str(core.models_dir() / "v1"), device))
    background = BackgroundRouter(loader)
    speech = speech_queue
    if speech is None and speak and handler_factory is None:
        speech = default_speech_queue()
    handler = _LazyHandler(background, handler_factory or (
        lambda router: _default_handler(router, speak, llm_options, speech, datasource, vision_frames)))
    turns = []
    while max_turns is None or len(turns) < max_turns:
        if stop_event is not None and stop_event.is_set():
            on_event("(stopped - ending conversation)")
            break
        on_event("listening...")
        out = listen_and_route(timeout_s=timeout_s, background=background, handler=handler, stop_event=stop_event,
                               on_transcript=on_transcript, **kwargs)
        if out["turn"] is None:
            stopped = stop_event is not None and stop_event.is_set()
            on_event("(stopped - ending conversation)" if stopped else "(silence - ending conversation)")
            break
        turns.append(out)
        turn = out["turn"]
        if on_turn is not None:
            on_turn(out)
        on_event(f"you  : {out['transcript']}\nnova : {turn['spoken']}  [{turn['route']}]"
                 + (f"  (tts failed: {turn['tts_error']})" if "tts_error" in turn else ""))
        if speech is not None:
            for error in _speech_errors(speech, turn):
                on_event(f"(speech failed: {error})")
        if turn.get("end_session"):
            on_event("(goodbye - ending conversation)")
            break
    if speech is not None:
        speech.close()
    return turns


def main(argv=None):
    ap = argparse.ArgumentParser(description="Talk to Nova: mic -> ElevenLabs Scribe -> router -> ElevenLabs TTS, repeated.")
    ap.add_argument("--model", default=None, help="router checkpoint (default: $NOVA_MODELS_DIR/v1)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-speak", action="store_true", help="print replies instead of speaking them")
    ap.add_argument("--once", action="store_true", help="a single turn instead of a conversation")
    ap.add_argument("--timeout", type=float, default=10.0, help="end the conversation after this many seconds of silence")
    ap.add_argument("--no-llm", action="store_true", help="force the offline FillerGenerator")
    ap.add_argument("--runtime", choices=RUNTIMES, help="LLM runtime (default: turn_policy.json live_llm.runtime, env NOVA_LLM_RUNTIME)")
    ap.add_argument("--base-url", help="LLM server URL (default: live_llm.base_url, env NOVA_LLM_BASE_URL)")
    ap.add_argument("--llm-model", help="model name the LLM runtime knows (env NOVA_LLM_MODEL)")
    ap.add_argument("--deadline-ms", type=int, help="override generation.deadline_ms for this run (a slow CPU may need more than 1500)")
    args = ap.parse_args(argv)
    print("router loading in background; speak when you see 'listening...' (Ctrl-C to quit)", file=sys.stderr)
    try:
        converse(args.model, args.device, speak=not args.no_speak, timeout_s=args.timeout,
                 max_turns=1 if args.once else None,
                 llm_options={"no_llm": args.no_llm, "runtime": args.runtime, "base_url": args.base_url,
                              "llm_model": args.llm_model, "deadline_ms": args.deadline_ms})
    except KeyboardInterrupt:
        print()
    except Exception as exc:
        print(f"stt FAILED: {type(exc).__name__}: {redact_secrets(str(exc), os.environ.get(API_KEY_ENV))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

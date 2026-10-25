"""Local HTTP bridge so Unity can start/stop the `stt.converse()` voice conversation and poll its progress.

    router/.venv/bin/python -m router.voice_server [--host 127.0.0.1] [--port 8765] [--model DIR] [--no-speak] ...

    POST /voice/start    start a conversation (ignored while one is running; "started" says which)
    POST /voice/stop     ask the running conversation to stop (a reply already being spoken finishes first)
    GET  /voice/status   {state, last_transcript, last_reply, route, error, session_id, seq, router_ready,
                         display, display_seq}

`display` is null or {doc_id, section_id, title, asset, source} (the manual page to show, set by a demo turn); it is kept until a
newer one replaces it (also across sessions), and `display_seq` increases by one each time it is set, so a client shows the page
when `display_seq` changes. `state` is idle | loading | listening | speaking | ended | error. `seq` increases on every change so a client can tell
that something new happened. Python owns the microphone and the speaker on THIS machine; nothing audio is sent to Unity.
The router loads on a background thread at server start (prewarm), so /voice/start is fast once `router_ready` is true.
The ElevenLabs key is read by stt/tts from the environment (or router/.env) and never appears in a response.

The server has no authentication: it binds 127.0.0.1 by default, and a request carrying an `Origin` header (a web page,
never Unity) is refused so a browser tab cannot switch the microphone on. Use --host only on a network you trust.
"""
import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import stt
from .llm_generator import API_KEY_ENV, redact_secrets

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
STATES = ("idle", "loading", "listening", "speaking", "ended", "error")
MAX_ERROR_CHARS = 300


def _redacted(text):
    for var in (stt.KEY_VAR, API_KEY_ENV):
        text = redact_secrets(text, os.environ.get(var))
    return text


class VoiceSession:
    """One conversation at a time, run on a worker thread. All state changes go through `_update` under a lock.

    `converse_fn(**kwargs)` defaults to `stt.converse`; it is called with on_event, on_turn, stop_event, router_loader and
    `converse_options`. `router_loader` is loaded once, on a background thread, when the session object is created."""

    def __init__(self, router_loader, converse_fn=stt.converse, converse_options=None):
        self._lock = threading.Lock()
        self._converse_fn = converse_fn
        self._options = dict(converse_options or {})
        self._state = "idle"
        self._last_transcript = self._last_reply = self._route = self._error = None
        self._session_id = 0
        self._seq = 0
        self._display, self._display_seq = None, 0
        self._stop = None
        self._worker = None
        self._router_ready = threading.Event()
        self._router = stt.BackgroundRouter(self._load_router(router_loader))

    def _load_router(self, loader):
        def load():
            try:
                return loader()
            finally:
                self._router_ready.set()
        return load

    def status(self):
        with self._lock:
            return self._status_locked()

    def _status_locked(self):
        return {"state": self._state, "last_transcript": self._last_transcript, "last_reply": self._last_reply,
                "route": self._route, "error": self._error, "session_id": self._session_id, "seq": self._seq,
                "router_ready": self._router_ready.is_set(), "display": self._display, "display_seq": self._display_seq}

    def _update(self, session_id, **fields):
        """Apply `fields` unless `session_id` is no longer current or its stop was requested (late events of a stopped conversation)."""
        with self._lock:
            if session_id != self._session_id or self._stop.is_set():
                return
            for key, value in fields.items():
                setattr(self, "_" + key, value)
            if fields.get("display") is not None:
                self._display_seq += 1
            self._seq += 1

    def start(self):
        """-> (started, status). Not started when a conversation is already running or the previous one is still shutting down."""
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return False, self._status_locked()
            self._session_id += 1
            self._seq += 1
            self._state = "loading"
            self._last_transcript = self._last_reply = self._route = self._error = None
            session_id, self._stop = self._session_id, threading.Event()
            self._worker = threading.Thread(target=self._run, args=(session_id, self._stop), daemon=True, name="voice-session")
            self._worker.start()
            return True, self._status_locked()

    def stop(self):
        """Ask the running conversation to stop; its state becomes idle at once. -> status."""
        with self._lock:
            if self._stop is not None and self._worker is not None and self._worker.is_alive() and not self._stop.is_set():
                self._stop.set()
                self._state = "idle"
                self._seq += 1
            return self._status_locked()

    def is_busy(self):
        with self._lock:
            return self._worker is not None and self._worker.is_alive()

    def join(self, timeout=None):
        worker = self._worker
        if worker is not None:
            worker.join(timeout)

    def _run(self, session_id, stop_event):
        def on_event(text):
            if text == "listening...":
                self._update(session_id, state="listening")
            elif text.startswith("(speech failed:"):  # playback problem: not fatal, shown in `error`
                self._update(session_id, error=_redacted(text)[:MAX_ERROR_CHARS])

        def on_turn(out):
            turn = out["turn"]
            fields = {"display": turn["display"]} if turn.get("display") else {}  # no display on this turn keeps the last one
            self._update(session_id, state="speaking", last_transcript=out["transcript"],
                         last_reply=turn.get("spoken"), route=turn.get("route"), **fields)

        try:
            self._converse_fn(on_event=on_event, on_turn=on_turn, stop_event=stop_event,
                              router_loader=self._router.get, **self._options)
        except BaseException as exc:
            self._update(session_id, state="error", error=_redacted(f"{type(exc).__name__}: {exc}")[:MAX_ERROR_CHARS])
        else:
            self._update(session_id, state="ended")  # ignored when stop() already made the state idle


def make_handler(session):
    class VoiceRequestHandler(BaseHTTPRequestHandler):
        server_version = "NovaVoice/1"

        def log_message(self, format, *args):  # quiet: no per-request stderr noise
            pass

        def _send(self, code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _refuse_browser(self):
            if self.headers.get("Origin") is not None:
                self._send(403, {"error": "browser requests are refused"})
                return True
            return False

        def do_GET(self):
            if self._refuse_browser():
                return
            if self.path == "/voice/status":
                self._send(200, session.status())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self._refuse_browser():
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(min(length, 65536))  # the body is ignored
            if self.path == "/voice/start":
                started, status = session.start()
                self._send(200, {**status, "started": started})
            elif self.path == "/voice/stop":
                self._send(200, session.stop())
            else:
                self._send(404, {"error": "not found"})

    return VoiceRequestHandler


def make_server(session, host=DEFAULT_HOST, port=DEFAULT_PORT):
    return ThreadingHTTPServer((host, port), make_handler(session))


def main(argv=None):
    from . import core

    ap = argparse.ArgumentParser(description="Local HTTP server that lets Unity start/stop the Nova voice conversation.")
    ap.add_argument("--host", default=DEFAULT_HOST, help="bind address (default 127.0.0.1; the server has no authentication)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--model", default=None, help="router checkpoint (default: $NOVA_MODELS_DIR/v1)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-speak", action="store_true", help="print replies instead of speaking them")
    ap.add_argument("--timeout", type=float, default=10.0, help="end a conversation after this many seconds of silence")
    ap.add_argument("--no-llm", action="store_true", help="force the offline FillerGenerator")
    ap.add_argument("--runtime", choices=stt.RUNTIMES)
    ap.add_argument("--base-url", help="LLM server URL")
    ap.add_argument("--llm-model")
    ap.add_argument("--deadline-ms", type=int)
    args = ap.parse_args(argv)
    model = args.model or str(core.models_dir() / "v1")
    session = VoiceSession(
        lambda: stt._load_router(model, args.device),
        converse_options={"model": model, "device": args.device, "speak": not args.no_speak, "timeout_s": args.timeout,
                          "llm_options": {"no_llm": args.no_llm, "runtime": args.runtime, "base_url": args.base_url,
                                          "llm_model": args.llm_model, "deadline_ms": args.deadline_ms}})
    try:
        server = make_server(session, args.host, args.port)
    except OSError as exc:
        print(f"voice_server FAILED: cannot bind {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1
    print(f"voice server on http://{args.host}:{server.server_address[1]} (router loading in background; "
          "POST /voice/start, POST /voice/stop, GET /voice/status; Ctrl-C to quit)", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(file=sys.stderr)
    finally:
        session.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

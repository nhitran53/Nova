"""Connect Unity to the PC voice session and simulator readings.

POST /voice/start and /voice/stop control the conversation.
GET /voice/status reports progress. GET /telemetry returns fresh readings.
POST /demo/oxygen changes simulated oxygen from the wrist controls.
The server defaults to localhost and rejects browser Origin headers.
Microphone and speaker audio stay on the PC.
Voice status includes alert, persistent display and display_seq alongside router_error.
Gemini handles replies and secondary routing. Use --no-llm for offline replies.
Navigation commands include a destination and sequence number for Unity.
"""
import argparse
import json
import math
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import stt
from .llm_generator import API_KEY_ENV, GEMINI_KEY_ENV, redact_secrets

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
STATES = ("idle", "loading", "listening", "processing", "speaking", "ended", "error")
MAX_ERROR_CHARS = 300
UNITY_DESTINATIONS = {"shuttle": "shuttle_001"}  # Router place to Unity route
UNITY_ACTIONS = {"navigation": "start", "map": "find"}
NO_GO_OUTCOMES = ("cannot_reach", "cannot_return")


def llm_options(no_llm=False):
    return {"no_llm": no_llm}


def unity_navigation(turn):
    """Return a Unity command for an allowed route request."""
    call, result = turn.get("tool_call") or {}, turn.get("tool_result") or {}
    action = UNITY_ACTIONS.get(call.get("name"))
    data = result.get("data") or {}
    destination = UNITY_DESTINATIONS.get(data.get("place"))
    if action is None or destination is None or not result.get("ok") or turn.get("blocked_text") is not None:
        return None
    if action == "start":
        if data.get("response_mode") != "guidance":
            return None
        if data.get("feasibility", {}).get("outcome") in NO_GO_OUTCOMES:
            return None
    return action, destination


def _redacted(text):
    for var in (stt.KEY_VAR, API_KEY_ENV, GEMINI_KEY_ENV):
        text = redact_secrets(text, os.environ.get(var))
    return text


class VoiceSession:
    """One conversation at a time, run on a worker thread. All state changes go through `_update` under a lock.

    `converse_fn(**kwargs)` defaults to `stt.converse`; it is called with on_event, on_turn, on_transcript, stop_event, router_loader and
    `converse_options`. `router_loader` is loaded once, on a background thread, when the session object is created."""

    def __init__(self, router_loader, converse_fn=stt.converse, converse_options=None):
        self._lock = threading.Lock()
        self._converse_fn = converse_fn
        self._options = dict(converse_options or {})
        self._state = "idle"
        self._last_transcript = self._last_reply = self._route = self._error = self._alert = None
        self._session_id = 0
        self._seq = 0
        self._display, self._display_seq = None, 0
        self._navigation_action = self._navigation_destination = None
        self._navigation_seq = 0
        self._stop = None
        self._worker = None
        self._router_ready = threading.Event()
        self._router_error = None
        self._router = stt.BackgroundRouter(self._load_router(router_loader))

    def _load_router(self, loader):
        def load():
            try:
                router = loader()
            except BaseException as exc:
                error = _redacted(f"{type(exc).__name__}: {exc}")[:MAX_ERROR_CHARS]
                with self._lock:
                    self._router_error = error
                print(f"router load failed: {error}", file=sys.stderr)
                raise
            self._router_ready.set()
            return router
        return load

    def status(self):
        with self._lock:
            return self._status_locked()

    def _status_locked(self):
        return {"state": self._state, "last_transcript": self._last_transcript, "last_reply": self._last_reply,
                "alert": self._alert, "route": self._route, "error": self._error, "session_id": self._session_id, "seq": self._seq,
                "router_ready": self._router_ready.is_set(), "router_error": self._router_error, "display": self._display, "display_seq": self._display_seq,
                "navigation_action": self._navigation_action, "navigation_destination": self._navigation_destination,
                "navigation_seq": self._navigation_seq}

    def _update(self, session_id, **fields):
        """Apply `fields` unless `session_id` is no longer current or its stop was requested (late events of a stopped conversation)."""
        with self._lock:
            if session_id != self._session_id or self._stop.is_set():
                return
            for key, value in fields.items():
                setattr(self, "_" + key, value)
            if fields.get("display") is not None:
                self._display_seq += 1
            if fields.get("navigation_destination") is not None:
                self._navigation_seq += 1
            self._seq += 1

    def start(self):
        """-> (started, status). Not started when a conversation is already running or the previous one is still shutting down."""
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return False, self._status_locked()
            self._session_id += 1
            self._seq += 1
            self._state = "loading"
            self._last_transcript = self._last_reply = self._route = self._error = self._alert = None
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
            else:
                print(_redacted(text), file=sys.stderr)
                if text.startswith("(speech failed:"):  # playback problem: not fatal, shown in `error`
                    self._update(session_id, error=_redacted(text)[:MAX_ERROR_CHARS])

        def on_transcript(text):
            self._update(session_id, state="processing", last_transcript=text, last_reply=None, route=None)

        def on_turn(out):
            turn = out["turn"]
            fields = {"display": turn["display"]} if turn.get("display") else {}  # no display on this turn keeps the last one
            navigation = unity_navigation(turn)
            if navigation is not None:
                fields.update(navigation_action=navigation[0], navigation_destination=navigation[1])
            self._update(session_id, state="speaking", last_transcript=out["transcript"],
                         last_reply=turn.get("spoken"), alert=turn.get("alert_line"), route=turn.get("route"), **fields)

        try:
            self._converse_fn(on_event=on_event, on_turn=on_turn, on_transcript=on_transcript, stop_event=stop_event,
                              router_loader=self._router.get, **self._options)
        except BaseException as exc:
            error = _redacted(f"{type(exc).__name__}: {exc}")[:MAX_ERROR_CHARS]
            print(f"voice session failed: {error}", file=sys.stderr)
            self._update(session_id, state="error", error=error)
        else:
            self._update(session_id, state="ended")  # ignored when stop() already made the state idle


def make_handler(session, datasource=None):
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
            elif self.path == "/telemetry":
                from .datasource import DataUnavailableError

                try:
                    if datasource is None:
                        raise DataUnavailableError("PostgreSQL datasource required")
                    self._send(200, datasource.telemetry())
                except DataUnavailableError:
                    self._send(503, {"schema_version": 1, "available": False,
                                     "error": "Telemetry unavailable. Check PostgreSQL and simulator.py."})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self._refuse_browser():
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._send(400, {"error": "invalid body length"})
                return
            if length < 0 or length > 65536:
                self._send(400, {"error": "invalid body length"})
                return
            body = self.rfile.read(length) if length else b""
            if self.path == "/demo/oxygen":
                from .datasource import DataUnavailableError

                try:
                    command = json.loads(body)
                    if not isinstance(command, dict) or set(command) not in ({"value"}, {"delta"}):
                        raise ValueError()
                    relative = "delta" in command
                    value = command["delta" if relative else "value"]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError()
                    if not (-100 <= value <= 100 if relative else 0 <= value <= 100):
                        raise ValueError()
                    if datasource is None:
                        raise DataUnavailableError("PostgreSQL datasource required")
                    self._send(200, {"o2_percent": datasource.set_demo_oxygen(value, relative)})
                except (ValueError, TypeError, UnicodeDecodeError):
                    self._send(400, {"error": "Use value 0 to 100 or delta -100 to 100."})
                except DataUnavailableError:
                    self._send(503, {"error": "Oxygen demo unavailable. Check PostgreSQL and simulator.py."})
            elif self.path == "/voice/start":
                started, status = session.start()
                self._send(200, {**status, "started": started})
            elif self.path == "/voice/stop":
                self._send(200, session.stop())
            else:
                self._send(404, {"error": "not found"})

    return VoiceRequestHandler


def make_server(session, host=DEFAULT_HOST, port=DEFAULT_PORT, datasource=None):
    return ThreadingHTTPServer((host, port), make_handler(session, datasource))


def _make_datasource(kind, database_url=None, connect=None, say=None):
    """None for "sim" (stt keeps its built-in SimState). For "postgres" a PostgresDatasource, checked once now: a failed
    read is reported but not fatal, since simulator.py may start later; until then each turn fails closed (no made-up values)."""
    if kind != "postgres":
        return None
    from dotenv import load_dotenv
    from .datasource import DataUnavailableError, PostgresDatasource

    say = say or (lambda line: print(line, file=sys.stderr))
    load_dotenv(stt.ENV_PATH)
    datasource = PostgresDatasource(database_url=database_url, connect=connect)
    try:
        snap = datasource.snapshot()
        say(f"datasource: postgres OK (o2 {snap['o2_pct']}%, {snap['minutes_remaining']} min)")
    except DataUnavailableError as exc:
        say(f"datasource: postgres not ready yet ({_redacted(str(exc))[:200]}); is simulator.py running?")
    return datasource


def main(argv=None):
    from . import core

    ap = argparse.ArgumentParser(description="Local HTTP server that lets Unity start/stop the Nova voice conversation.")
    ap.add_argument("--host", default=DEFAULT_HOST, help="bind address (default 127.0.0.1; the server has no authentication)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--model", default=None, help="router checkpoint (default: $NOVA_MODELS_DIR/v1)")
    ap.add_argument("--device", default="cpu", help="torch device for the router (not the microphone)")
    ap.add_argument("--no-speak", action="store_true", help="print replies instead of speaking them")
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds of silence that end a conversation")
    ap.add_argument("--datasource", choices=("sim", "postgres"), default="sim",
                    help="where readings and places come from: the built-in SimState (default) or the Postgres DB that "
                         "simulator.py writes (DATABASE_URL from the environment or the selected .env; default 'dbname=nova')")
    ap.add_argument("--database-url", help="libpq connection string for --datasource postgres (overrides DATABASE_URL)")
    ap.add_argument("--no-llm", action="store_true", help="offline template replies only: no Gemini, no second routing layer")
    args = ap.parse_args(argv)
    model = str(Path(args.model or core.models_dir() / "v1").expanduser().resolve())
    if not Path(model).is_dir():
        print(f"voice_server FAILED: model path is not an existing directory: {model}", file=sys.stderr)
        return 2
    from dotenv import load_dotenv

    load_dotenv(stt.ENV_PATH)
    print(f"router model: {model}", file=sys.stderr)
    print(f".env path: {stt.ENV_PATH.resolve()}", file=sys.stderr)
    key_names = [stt.KEY_VAR]
    if not args.no_llm:
        key_names.append(GEMINI_KEY_ENV)
    for var in key_names:
        print(f"{var}: {'found' if os.environ.get(var, '').strip() else 'missing'}", file=sys.stderr)
    print("LLM: offline (--no-llm)" if args.no_llm else "LLM: configured service with offline fallback",
          file=sys.stderr)
    try:
        import sounddevice
    except (ImportError, OSError):
        print("sounddevice import: failed", file=sys.stderr)
    else:
        print("sounddevice import: OK", file=sys.stderr)
    datasource = _make_datasource(args.datasource, args.database_url)
    session = VoiceSession(
        lambda: stt._load_router(model, args.device),
        converse_options={"model": model, "device": args.device, "speak": not args.no_speak, "timeout_s": args.timeout,
                          "llm_options": llm_options(args.no_llm),
                          "datasource": datasource})
    try:
        server = make_server(session, args.host, args.port, datasource=datasource)
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

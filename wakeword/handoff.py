"""Always-listening handoff: wake word -> POST /voice/start on router/voice_server -> release the mic -> wait -> resume.

voice_server opens the microphone itself for the conversation, so this process must NOT hold it meanwhile: the mic is
closed on a detection and reopened only after the conversation has ended (plus a cooldown). The same applies when
someone else (e.g. the Unity telephone gesture) starts a conversation: a periodic GET /voice/status notices it and the
mic is released for its duration too.

Nothing here talks to Unity; Unity already sees the new session through GET /voice/status (`session_id` / `seq` change).
"""
import json
import logging
import time
import urllib.error
import urllib.request

from config import DEFAULT_VOICE_URL

log = logging.getLogger("nova.wakeword.handoff")

BUSY_STATES = ("loading", "listening", "speaking")
DONE_STATES = ("ended", "idle", "error")


class VoiceServerError(RuntimeError):
    """voice_server unreachable, refused the request, or answered something that is not its JSON."""


class VoiceServerClient:
    """Minimal client for router/voice_server.py. Sends no Origin header (the server refuses browser requests)."""

    def __init__(self, base_url=DEFAULT_VOICE_URL, timeout_s=3.0):
        self.base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost: ignore HTTP(S)_PROXY

    def start(self):
        """POST /voice/start -> status dict plus `started` (False when a conversation was already running)."""
        return self._request("POST", "/voice/start")

    def stop(self):
        return self._request("POST", "/voice/stop")

    def status(self):
        return self._request("GET", "/voice/status")

    def _request(self, method, path):
        request = urllib.request.Request(self.base_url + path, data=b"" if method == "POST" else None, method=method)
        try:
            with self._opener.open(request, timeout=self._timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise VoiceServerError(f"{method} {path} -> HTTP {exc.code}") from None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            reason = getattr(exc, "reason", exc)
            raise VoiceServerError(f"{method} {path} failed: {type(reason).__name__}: {reason}") from None
        if not isinstance(payload, dict):
            raise VoiceServerError(f"{method} {path} returned unexpected JSON")
        return payload


class WakeService:
    """Run the wake word loop.

    `open_mic()` returns a context manager yielding an iterator of 80 ms int16 frames (closing it releases the mic).
    With `client=None` (dry run) detections are only reported and the mic stays open.
    Mic/detector errors are retried with exponential backoff (`mic_retry_initial_s` .. `mic_retry_max_s`).
    `clock` / `sleep` are injectable for tests.
    """

    def __init__(self, open_mic, detector, client=None, on_wake=None, cooldown_s=3.0, conversation_timeout_s=300.0,
                 poll_interval_s=0.5, status_check_interval_s=1.0, start_attempts=3, start_retry_s=1.0,
                 status_failure_grace_s=10.0, mic_retry_initial_s=1.0, mic_retry_max_s=30.0,
                 clock=time.monotonic, sleep=time.sleep):
        self._open_mic = open_mic
        self._detector = detector
        self._client = client
        self._on_wake = on_wake or (lambda detection: None)
        self._cooldown_s = cooldown_s
        self._conversation_timeout_s = conversation_timeout_s
        self._poll_interval_s = poll_interval_s
        self._status_check_interval_s = status_check_interval_s
        self._start_attempts = start_attempts
        self._start_retry_s = start_retry_s
        self._status_failure_grace_s = status_failure_grace_s
        self._mic_retry_initial_s = mic_retry_initial_s
        self._mic_retry_max_s = mic_retry_max_s
        self._clock = clock
        self._sleep = sleep
        self._mic_delay_s = mic_retry_initial_s
        self._wakes = 0
        self._wakes = 0
        self._server_reachable = None  # last known state, so only changes are logged while listening

    def check_server(self):
        """Log once whether voice_server is reachable. Never raises: the server may simply not be up yet."""
        try:
            status = self._client.status()
        except VoiceServerError as exc:
            log.warning("voice_server not reachable at %s yet (%s); will keep listening and retry on each wake",
                        self._client.base_url, exc)
            return False
        log.info("voice_server reachable at %s (state=%s, router_ready=%s)", self._client.base_url,
                 status.get("state"), status.get("router_ready"))
        if status.get("router_ready") is False:
            log.warning("voice_server is still loading the router; the first conversation may start slowly")
        return True

    def run(self, max_wakes=None):
        """Listen until the frame source ends, `max_wakes` wake words were handled (counted per handoff or dry-run detection)
        or KeyboardInterrupt. Returns the number of wake words handled."""
        self._wakes = 0
        while max_wakes is None or self._wakes < max_wakes:
            try:
                outcome, detection = self._listen_once(max_wakes)
            except Exception as exc:  # mic open/read failure or detector error: back off, then reopen
                log.warning("microphone error (%s: %s); reopening in %.0f s", type(exc).__name__, exc, self._mic_delay_s)
                self._sleep(self._mic_delay_s)
                self._mic_delay_s = min(self._mic_delay_s * 2, self._mic_retry_max_s)
                continue
            if outcome in ("eof", "done"):
                break
            if outcome == "wake":
                self._wakes += 1
                self._on_wake(detection)
            self._handle_handoff(external=(outcome == "busy"))
        return self._wakes

    def _listen_once(self, max_wakes):
        """Open the mic and listen. -> ("wake", Detection) | ("busy", None) | ("eof", None) | ("done", None).
        The mic is closed again before this returns."""
        with self._open_mic() as frames:
            self._detector.reset()
            next_check = self._clock() + self._status_check_interval_s
            for frame in frames:
                self._mic_delay_s = self._mic_retry_initial_s  # audio is flowing: forget earlier errors
                detection = self._detector.process(frame)
                if detection is not None:
                    if self._client is not None:
                        return "wake", detection
                    self._wakes += 1
                    self._on_wake(detection)
                    if max_wakes is not None and self._wakes >= max_wakes:
                        return "done", None
                elif self._client is not None and self._clock() >= next_check:
                    next_check = self._clock() + self._status_check_interval_s
                    if self._conversation_running_elsewhere():
                        return "busy", None
        return "eof", None

    def _conversation_running_elsewhere(self):
        try:
            state = self._client.status().get("state")
        except VoiceServerError as exc:
            if self._server_reachable is not False:
                log.warning("voice_server not reachable at %s (%s); still listening for the wake word",
                            self._client.base_url, exc)
            self._server_reachable = False
            return False
        if self._server_reachable is False:
            log.info("voice_server is reachable again")
        self._server_reachable = True
        return state in BUSY_STATES

    def _handle_handoff(self, external):
        """The mic is already released here. Start (or join) a conversation, wait for it, then cool down."""
        try:
            if external:
                log.info("a conversation was started elsewhere (e.g. telephone gesture); mic released until it ends")
                status = self._client.status()
                session_id = status.get("session_id", 0)
            else:
                session_id = self._start_conversation()
            if session_id is not None:
                self._wait_for_end(session_id)
        except VoiceServerError as exc:
            log.warning("voice_server error: %s", exc)
        except KeyboardInterrupt:
            self._stop_conversation_quietly()
            raise
        log.info("cooling down %.1f s, then listening again", self._cooldown_s)
        self._sleep(self._cooldown_s)

    def _start_conversation(self):
        """POST /voice/start with retries. -> session_id, or None if voice_server could not be reached."""
        for attempt in range(1, self._start_attempts + 1):
            try:
                status = self._client.start()
            except VoiceServerError as exc:
                log.warning("could not start the conversation (attempt %d/%d): %s - is router/voice_server running at %s?",
                            attempt, self._start_attempts, exc, self._client.base_url)
                if attempt < self._start_attempts:
                    self._sleep(self._start_retry_s * 2 ** (attempt - 1))
                continue
            if status.get("started") is False:
                log.info("a conversation is already running; waiting for it to end")
            else:
                log.info("conversation started (session %s); mic released", status.get("session_id"))
            return status.get("session_id", 0)
        log.error("giving up on this wake word; still listening. Start voice_server (see README 'Demo setup') and try again")
        return None

    def _wait_for_end(self, session_id):
        """Poll /voice/status until this (or a newer) session is ended/idle/error, the timeout hits, or the server is gone."""
        deadline = self._clock() + self._conversation_timeout_s
        unreachable_since = None
        last_state = None
        while True:
            try:
                status = self._client.status()
            except VoiceServerError as exc:
                now = self._clock()
                unreachable_since = unreachable_since if unreachable_since is not None else now
                if now - unreachable_since >= self._status_failure_grace_s:
                    log.error("voice_server unreachable for %.0f s (%s); assuming the conversation is over",
                              now - unreachable_since, exc)
                    return None
            else:
                unreachable_since = None
                state = status.get("state")
                if state != last_state:
                    log.info("conversation state: %s", state)
                    last_state = state
                if state in DONE_STATES and status.get("session_id", 0) >= session_id:
                    self._log_outcome(status)
                    return status
            if self._clock() >= deadline:
                log.warning("conversation still running after %.0f s; asking voice_server to stop it", self._conversation_timeout_s)
                self._stop_conversation_quietly()
                return None
            self._sleep(self._poll_interval_s)

    @staticmethod
    def _log_outcome(status):
        if status.get("state") == "error":
            log.warning("conversation ended with an error: %s", status.get("error"))
        elif status.get("last_transcript"):
            log.info("heard %r -> replied %r", status.get("last_transcript"), status.get("last_reply"))

    def _stop_conversation_quietly(self):
        try:
            self._client.stop()
        except VoiceServerError as exc:
            log.warning("could not stop the conversation: %s", exc)

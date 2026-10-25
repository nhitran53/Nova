#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${VOICE_PORT:-8765}"
if [[ ! -f "$ROOT/wakeword/models/hey_nova.onnx" ]]; then
    echo '[demo] drop hey_nova.onnx into wakeword/models/' >&2
    exit 1
fi
if [[ -n "${NOVA_ROUTER_PYTHON:-}" ]]; then
    PYTHON="$NOVA_ROUTER_PYTHON"
elif [[ -x "$ROOT/router/.venv/bin/python" ]]; then
    PYTHON="$ROOT/router/.venv/bin/python"
else
    PYTHON=/home/isaach/Nova/router/.venv/bin/python
fi
if [[ ! -x "$PYTHON" ]]; then
    echo '[demo] Router Python not found; set NOVA_ROUTER_PYTHON to an executable router-venv Python.' >&2
    exit 1
fi
# Separate process groups include uv's listener child as well as the server.
command -v setsid >/dev/null || { echo '[demo] setsid is required.' >&2; exit 1; }
server_pid=''
listener_pid=''
cleanup() {
    trap - EXIT INT TERM
    for pid in "$listener_pid" "$server_pid"; do
        if [[ -n "$pid" ]]; then kill -TERM -- "-$pid" 2>/dev/null || true; fi
    done
    for pid in "$listener_pid" "$server_pid"; do
        if [[ -n "$pid" ]]; then wait "$pid" 2>/dev/null || true; fi
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Whitespace-separated flags, not shell code (no eval or environment logging).
read -r -a server_args <<< "${VOICE_SERVER_ARGS:-}"
cd "$ROOT"
echo '[demo] Starting voice_server; waiting for router readiness (up to 120 seconds).' >&2
setsid "$PYTHON" -m router.voice_server "${server_args[@]}" --host 127.0.0.1 --port "$PORT" >&2 &
server_pid=$!
deadline=$((SECONDS + 120))
while true; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo '[demo] voice_server exited before router readiness.' >&2
        exit 1
    fi
    if "$PYTHON" -c '
import json, sys, urllib.request
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open("http://127.0.0.1:" + sys.argv[1] + "/voice/status", timeout=1) as response:
        status = json.load(response)
except (OSError, ValueError):
    sys.exit(1)
if status.get("router_error"):
    print("[demo] router_error: " + str(status["router_error"]), file=sys.stderr)
    sys.exit(2)
sys.exit(0 if status.get("router_ready") is True else 1)
' "$PORT"; then
        break
    else
        result=$?
        if [[ "$result" -eq 2 ]]; then exit 1; fi
    fi
    if (( SECONDS >= deadline )); then
        echo '[demo] Timed out waiting for router readiness.' >&2
        exit 1
    fi
    sleep 0.2
done
kill -0 "$server_pid" 2>/dev/null || { echo '[demo] voice_server exited.' >&2; exit 1; }
echo '[demo] Router ready; starting wake listener. Press Play in Unity.' >&2
cd "$ROOT/wakeword"
# Wait in the foreground so signals can run cleanup, including uv descendants.
setsid uv run python main.py --handoff --voice-url "http://127.0.0.1:$PORT" "$@" &
listener_pid=$!
wait "$listener_pid"

"""Offline launcher integration: real localhost HTTP, fake router and listener."""
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

import pytest


SERVER = '''
import http.server, json, os, sys
from pathlib import Path
log = Path(os.environ["DEMO_EVENTS"])
def record(event):
    with log.open("a") as f:
        f.write(json.dumps([event, os.getpid()]) + "\\n")
record("server")
if os.environ.get("DEMO_FAIL") == "died":
    sys.exit(3)
class Handler(http.server.BaseHTTPRequestHandler):
    polls = 0
    def log_message(self, *args): pass
    def do_GET(self):
        assert self.path == "/voice/status"
        Handler.polls += 1
        ready = Handler.polls >= 3
        record("ready" if ready else "waiting")
        body = json.dumps({"router_ready": ready, "router_error":
            "fake router load failed" if os.environ.get("DEMO_FAIL") else None}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)
http.server.HTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])), Handler).serve_forever()
'''
LISTENER = '''
import json, os, subprocess, sys, time
from pathlib import Path
assert sys.argv[1:5] == ["run", "python", "main.py", "--handoff"]
assert sys.argv[5] == "--voice-url"
assert sys.argv[6].startswith("http://127.0.0.1:")
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
with Path(os.environ["DEMO_EVENTS"]).open("a") as f:
    f.write(json.dumps(["listener", os.getpid()]) + "\\n")
    f.write(json.dumps(["descendant", child.pid]) + "\\n")
time.sleep(300)
'''


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def wait_for(predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    pytest.fail("Timed out waiting for demo process")


def alive(pid):
    # A killed orphan may briefly be a zombie pending the container's init reaper.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


@pytest.fixture
def demo(tmp_path):
    wake = tmp_path / "wakeword"
    (wake / "models").mkdir(parents=True)
    (wake / "models/hey_nova.onnx").touch()
    script = wake / "run_demo.sh"
    shutil.copyfile(Path(__file__).parents[1] / "run_demo.sh", script)
    server = tmp_path / "server.py"
    server.write_text(SERVER)
    fake_python = tmp_path / "router-python"
    fake_python.write_text(f'''#!{sys.executable}
import os, sys
if sys.argv[1:3] == ["-m", "router.voice_server"]:
    os.execv(sys.executable, [sys.executable, {str(server)!r}] + sys.argv[3:])
os.execv(sys.executable, [sys.executable] + sys.argv[1:])
''')
    fake_python.chmod(0o755)
    uv = tmp_path / "uv"
    uv.write_text(f"#!{sys.executable}\n" + LISTENER)
    uv.chmod(0o755)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = tmp_path / "events"
    env = dict(os.environ, NOVA_ROUTER_PYTHON=str(fake_python), VOICE_PORT=str(port),
               DEMO_EVENTS=str(log), PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
               VOICE_SERVER_ARGS="--datasource postgres")
    processes = []

    def start(failure=""):
        proc = subprocess.Popen(["bash", str(script)], env=dict(env, DEMO_FAIL=failure),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(proc)
        return proc, log

    yield start
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)
        for _, pid in events(log):
            if alive(pid):
                os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_run_demo_waits_for_readiness_and_cleans_process_groups(demo, sig):
    proc, log = demo()
    wait_for(lambda: any(event == "descendant" for event, _ in events(log)))
    names = [event for event, _ in events(log)]
    assert names[:4] == ["server", "waiting", "waiting", "ready"]
    assert names[4:] == ["listener", "descendant"]
    proc.send_signal(sig)
    _, stderr = proc.communicate(timeout=10)
    assert proc.returncode == (130 if sig == signal.SIGINT else 143)
    assert "Router ready" in stderr
    wait_for(lambda: all(not alive(pid) for _, pid in events(log)))


@pytest.mark.parametrize("failure", ["router_error", "died"])
def test_run_demo_failure_never_starts_listener(demo, failure):
    proc, log = demo(failure)
    _, stderr = proc.communicate(timeout=10)
    assert proc.returncode != 0
    assert "fake router load failed" in stderr if failure == "router_error" else "exited" in stderr
    assert "listener" not in [event for event, _ in events(log)]
    wait_for(lambda: all(not alive(pid) for _, pid in events(log)))


def test_run_demo_signal_during_readiness(demo):
    proc, log = demo()
    wait_for(lambda: any(event == "waiting" for event, _ in events(log)))
    proc.terminate()
    proc.communicate(timeout=10)
    assert proc.returncode == 143
    assert "listener" not in [event for event, _ in events(log)]
    wait_for(lambda: all(not alive(pid) for _, pid in events(log)))

"""Try the turn stack with a local LLM or an offline fallback.

Use --fake-router and --no-llm for a model-free demo.
Use --script FILE for saved turns. Readings come from the offline fixture.
"""
import argparse
import math
import os
import re
import sys
import time
import warnings
from datetime import timezone, datetime
from pathlib import Path

from .core import Router, models_dir
from .datasource import DataUnavailableError, FixtureDatasource
from .demo_turn import demo_retriever
from .llm_generator import API_KEY_ENV, CONSTRAIN_MODES, RUNTIMES, redact_secrets
from .llm_generator import build_generator as build_llm_generator
from .tools import default_registry
from .turn import TurnHandler, load_policy

HERE = Path(__file__).parent
SET_TELEMETRY = {"o2_pct": "o2_percent", "bpm": "heart_rate", "temp_c": "suit_temp_c", "suit_pressure_kpa": "suit_pressure_kpa"}
SET_MISSION = ("scenario", "activity")
HELP = ("commands: /advance N (sim seconds) | /set o2_pct|bpm|temp_c|suit_pressure_kpa=<number> or scenario|activity=<name> | "
        "/pos X Z (astronaut position in meters) | /sim | /help | /quit; anything else is an utterance. "
        "Try: how far is base 2 | how far is the tower | take me to base 2")

# fake-router keyword rules, first match wins: (label, any of these words / phrases in the
# lowercased utterance)
KEYWORD_RULES = (
    ("call_navigation", ("take me", "navigate", "go to", "walk to", "head to", "directions", "how far is", "how far to",
                         "distance to")),
    ("call_map", ("map", "show me")),
    ("call_o2", ("oxygen", "o2", "air left")),
    ("call_data", ("heart", "bpm", "pulse", "temperature", "radiation", "vitals")),
    ("call_safety", ("safety", "am i safe", "safe to")),
    ("call_llm", ("moon", "mars", "planet", "gravity", "airlock", "how far", "how long", "crater", "space", "why", "explain", "what is")),
    ("NO_TOOL", ("hi", "hello", "hey", "thanks", "thank you", "morning", "how are you", "good evening")),
)


class KeywordRouter:
    """Stand-in for the ModernBERT router (--fake-router): fixed keyword rules, confidence 0.97, OOD when nothing matches."""

    def predict(self, text):
        padded = " " + " ".join(re.findall(r"[a-z0-9']+", text.lower())) + " "
        label = next((lab for lab, phrases in KEYWORD_RULES if any(f" {phrase} " in padded for phrase in phrases)), "OOD")
        return {"label": label, "confidence": 0.97,
                "topk": [{"label": label, "p": 0.97}, {"label": "NO_TOOL", "p": 0.02}, {"label": "OOD", "p": 0.01}]}


def frozen_clock():
    """Fixture rows are stamped with this fixed time: the DataSource would call a row stale after max_age_s of real time,
    and an interactive session is slow. Sim time only moves through `advance`."""
    start = datetime.now(timezone.utc)
    return lambda: start


def say(text=""):
    """The one session output boundary: everything printed passes through redact_secrets with the configured API key."""
    print(redact_secrets(str(text), os.environ.get(API_KEY_ENV)))


class RedactingArgumentParser(argparse.ArgumentParser):
    """argparse errors echo the supplied value (`invalid choice: ...`), and any flag value can be a pasted API key."""

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {redact_secrets(message, os.environ.get(API_KEY_ENV))}\n")


def parse_args(argv):
    ap = RedactingArgumentParser(prog="python -m router.demo_live", description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=None,
                    help="router checkpoint (default: $NOVA_MODELS_DIR/binary)")
    ap.add_argument("--fake-router", action="store_true", help="keyword router instead of a checkpoint (no models needed)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--runtime", choices=RUNTIMES, help="default: turn_policy.json live_llm.runtime (env NOVA_LLM_RUNTIME)")
    ap.add_argument("--base-url", help="default: live_llm.base_url (env NOVA_LLM_BASE_URL)")
    ap.add_argument("--llm-model", help="model name the runtime knows (env NOVA_LLM_MODEL)")
    ap.add_argument("--constrain", choices=CONSTRAIN_MODES, help="default: live_llm.constrain (env NOVA_LLM_CONSTRAIN)")
    ap.add_argument("--deadline-ms", type=int, help="override generation.deadline_ms for this run (default: 8000 ms)")
    ap.add_argument("--no-llm", action="store_true", help="force the offline FillerGenerator")
    ap.add_argument("--advance", type=float, default=0.0, metavar="SECONDS", help="sim seconds to advance before each turn (default 0)")
    ap.add_argument("--script", metavar="FILE", help="read input lines from FILE (default: stdin)")
    return ap.parse_args(argv)


def build_generator(args, say):
    """-> (generator, label): llm_generator.build_generator with the CLI flags; an invalid config exits with one clean line."""
    try:
        return build_llm_generator(say, no_llm=args.no_llm, runtime=args.runtime, base_url=args.base_url, model=args.llm_model,
                                   constrain=args.constrain)
    except ValueError as exc:
        raise SystemExit(f"invalid LLM config: {exc}") from None


def build_router(args, say):
    if args.fake_router:
        say("router: FAKE keyword router (no checkpoint)")
        return KeywordRouter()
    if args.model is None:
        args.model = str(models_dir() / "binary")
    path = Path(args.model)
    if not path.is_dir() or not any(path.iterdir()):
        raise SystemExit(f"router checkpoint not found at {path}. Set NOVA_MODELS_DIR / pass --model, or run with --fake-router "
                         "to use the keyword router (no models needed).")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        router = Router.load(args.model, HERE / "labels.json", device=args.device)
    say(f"router: {args.model}")
    return router


def apply_set(ds, assignment):
    """`key=value` -> a one-line message. Telemetry keys edit the newest fixture row; scenario/activity edit the mission."""
    key, sep, value = assignment.partition("=")
    key, value = key.strip(), value.strip()
    if not sep or not value:
        return "usage: /set key=value"
    if key in SET_TELEMETRY:
        try:
            number = float(value)
        except ValueError:
            return f"{key} needs a number, got {value!r}"
        if not math.isfinite(number):
            return f"{key} needs a finite number"
        ds.telemetry[-1][SET_TELEMETRY[key]] = number
        return f"set {key}={number:g}"
    if key in SET_MISSION:
        old = ds.mission[key]
        ds.mission[key] = value
        try:
            ds.snapshot()
        except Exception as exc:  # an unknown scenario/activity makes the burn rate unusable; keep the old one
            ds.mission[key] = old
            return f"cannot set {key}={value!r}: {exc}"
        return f"set {key}={value}"
    return f"cannot set {key!r}; settable: {', '.join(list(SET_TELEMETRY) + list(SET_MISSION))}"


def apply_pos(ds, arguments):
    """`X Z` (meters) -> a one-line message. Appends a fixture position row, so place distances are measured from there."""
    parts = arguments.split()
    try:
        x, z = (float(part) for part in parts) if len(parts) == 2 else (None, None)
    except ValueError:
        x = None
    if x is None or not math.isfinite(x) or not math.isfinite(z):
        return "usage: /pos X Z   (two finite numbers, meters; the astronaut starts at 0 0)"
    ds.set_position(x, z)
    return f"astronaut moved to x={x:g} z={z:g}"


def snapshot_line(ds):
    try:
        s = ds.snapshot()
    except DataUnavailableError as exc:
        return f"sim: unavailable ({exc})"
    return (f"sim: o2={s['o2_pct']:.1f}% minutes={s['minutes_remaining']} bpm={s['bpm']:.0f} radiation={s['radiation_usv_h']:.2f} uSv/h "
            f"temp={s['temp_c']:.1f}C pressure={s['suit_pressure_kpa']:.1f}kPa scenario={ds.mission['scenario']}")


def why_no_llm(out):
    route = out["route"]
    if route == "alert":
        return "no (critical alert: no router, no LLM)"
    if route == "tool":
        return "no (tool turn: template line only)"
    if route in ("confirmed_yes", "confirmed_no", "refill_not_needed"):
        return "no (confirmation: deterministic)"
    if route in ("risk_diverted", "no_document"):
        return f"no ({route}: template, decided by scope policy)"
    return f"no (route {route}: template)"


def show_turn(number, text, out, wall_ms, say):
    pipe = out["pipeline"]
    if pipe is None:
        router_line = "n/a (alert preempted router)" if out["route"] == "alert" else "n/a (yes/no matcher answered before the router)"
    else:
        router_line = f"{pipe['label']} conf={pipe['confidence']:.3f} route={pipe['route']} ({pipe.get('gate') or pipe.get('reason')})"
    call = out["tool_call"] and f"{out['tool_call']['name']}({out['tool_call']['args']})"
    gen = out.get("generation")
    say(f"\n[turn {number}] {text!r}")
    say(f"  router  : {router_line}")
    say(f"  tool    : {call or 'none'}")
    say(f"  llm     : {'CALLED' if out['llm_called'] else why_no_llm(out)}")
    say(f"  request : {gen['request']['kind'] if gen and 'request' in gen else 'none (no request built)'}"
        + (f"   mode={gen['mode']}" if gen else ""))
    reported_ms = 0
    if gen and gen.get("mode") == "generated":
        for attempt in gen["attempt_log"]:
            reported_ms += attempt["latency_ms"]
            say(f"  raw #{attempt['attempt']}  : {attempt['raw']!r}   ok={attempt['ok']} violations={attempt['violations']} "
                f"latency={attempt['latency_ms']} ms")
        say(f"  outcome : {gen['status']} after {gen['attempts']} attempt(s)")
    else:
        say("  raw     : n/a (the LLM was not called)")
    if "llm_error" in out:
        say(f"  llm_error: {out['llm_error']}   (the fixed tool_error line is spoken; nothing is retried)")
    say(f"  spoken  : {out['spoken']}")
    verdict = out["verifier"]
    say(f"  verify  : ok={verdict['ok']} spoken_numbers={verdict['spoken']} unsupported={verdict['unsupported']}"
        + (f"   blocked_text={out['blocked_text']!r}" if out["blocked_text"] else ""))
    say(f"  violations: {gen['violations'] if gen and gen.get('mode') == 'generated' else []}")
    say(f"  latency : generator-reported {reported_ms} ms, turn wall {wall_ms} ms")


def input_lines(args, stdin):
    """-> (iterator of lines, interactive). --script FILE wins; else stdin (prompted only when it is a TTY)."""
    if args.script:
        return iter(Path(args.script).read_text().splitlines()), False
    if stdin.isatty():
        def prompted():
            while True:
                try:
                    yield input("you> ")
                except EOFError:
                    return
        return prompted(), True
    return iter(stdin.read().splitlines()), False


def main(argv=None, stdin=None):
    args = parse_args(argv)
    router = build_router(args, say)
    generator, label = build_generator(args, say)
    ds = FixtureDatasource(clock=frozen_clock())
    policy = load_policy()
    if args.deadline_ms is not None:
        policy["generation"]["deadline_ms"] = args.deadline_ms
    handler = TurnHandler(router, ds.snapshot, default_registry(datasource=ds), generator=generator, retriever=demo_retriever,
                          policy=policy, interim_fn=lambda line, route: say(f"nova (interim): {line}"))
    say(f"generator: {label}; deadline {policy['generation']['deadline_ms']} ms; sim: FixtureDatasource (offline)")
    say(snapshot_line(ds))
    try:
        lines, interactive = input_lines(args, stdin or sys.stdin)
        if interactive:
            say(HELP)
        return run_session(lines, interactive, args, ds, handler, label)
    except KeyboardInterrupt:
        say("\nsession interrupted")
        return 130


def run_session(lines, interactive, args, ds, handler, label):
    turns = 0
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if not interactive:
            say(f"\nyou> {text}")
        if text in ("/quit", "/exit"):
            break
        if text == "/help":
            say(HELP)
        elif text == "/sim":
            say(snapshot_line(ds))
        elif text.startswith("/advance"):
            try:
                seconds = float(text.partition(" ")[2])
                if not 0 < seconds < math.inf:
                    raise ValueError
            except ValueError:
                say("usage: /advance N   (N > 0 sim seconds)")
                continue
            ds.advance(seconds)
            say(f"advanced {seconds:g} sim s; {snapshot_line(ds)}")
        elif text.split()[0] == "/pos":
            say(apply_pos(ds, text.partition(" ")[2]))
        elif text.startswith("/set"):
            say(apply_set(ds, text.partition(" ")[2]))
        elif text.startswith("/"):
            say(f"unknown command {text.split()[0]!r}; {HELP}")
        else:
            if args.advance > 0:
                ds.advance(args.advance)
            turns += 1
            started = time.monotonic()
            try:
                out = handler.handle(text)
            except Exception as exc:  # e.g. DataUnavailableError from the alert poll: shown, session continues
                say(f"\n[turn {turns}] {text!r}\n  ERROR  : {type(exc).__name__}: {exc}")
                continue
            show_turn(turns, text, out, int((time.monotonic() - started) * 1000), say)
            say(f"  {snapshot_line(ds)}")
            if out.get("end_session"):
                say("(session ended by the user)")
                break
    say(f"\nsession over: {turns} turn(s); generator {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

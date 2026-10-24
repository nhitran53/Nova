"""Text REPL / one-shot CLI for the router.

    python -m router.cli --text "check my oxygen level"
    python -m router.cli            # interactive; empty line or Ctrl-D quits
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

from .core import Router, base_model_dir

HERE = Path(__file__).parent
DEFAULT_MODEL = os.environ.get("ROUTER_MODEL_DIR", str(base_model_dir()))


def format_pretty(result):
    lines = []
    if result["untrained"]:
        lines.append("!! " + result["warning"])
    lines.append(f"-> {result['label']}  (p={result['confidence']:.3f}, {result['latency_ms']:.1f} ms)")
    lines += [f"   {i + 1}. {t['label']:<16} {t['p']:.3f}" for i, t in enumerate(result["topk"])]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="router.cli", description=__doc__.splitlines()[0])
    ap.add_argument("--text", help="classify this text once and exit")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="local model dir or HF id (default: %(default)s)")
    ap.add_argument("--labels", default=str(HERE / "labels.json"))
    ap.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    ap.add_argument("--temperature", type=float, default=None, help="default: router_meta.json temperature, else 1.0")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--json", action="store_true", help="print prediction as JSON")
    args = ap.parse_args(argv)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # untrained state is shown in every result instead
        router = Router.load(args.model, args.labels, device=args.device, temperature=args.temperature)

    def run(text):
        result = router.predict(text, topk=args.topk)
        print(json.dumps(result, indent=2) if args.json else format_pretty(result))

    if args.text is not None:
        run(args.text)
        return 0
    print(f"router on {router.device} | labels={router.labels} | empty line to quit", file=sys.stderr)
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text:
            break
        run(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

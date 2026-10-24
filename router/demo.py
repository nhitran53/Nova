"""Run the hand-written demo_calls.json through a trained checkpoint and print PASS/FAIL per utterance.

    python -m router.demo [--model $NOVA_MODELS_DIR/synthetic-v0]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

from .core import Router, load_labels, models_dir

HERE = Path(__file__).parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=str(models_dir() / "synthetic-v0"), help="default: %(default)s (set NOVA_MODELS_DIR to move)")
    ap.add_argument("--calls", default=str(HERE / "demo_calls.json"))
    ap.add_argument("--labels", default=str(HERE / "labels.json"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    labels = load_labels(args.labels)
    calls = json.loads(Path(args.calls).read_text())["calls"]
    unknown = {c["label"] for c in calls} - set(labels)
    if unknown:
        sys.exit(f"{args.calls}: labels not in labels.json: {sorted(unknown)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        router = Router.load(args.model, args.labels, device=args.device)
    if router.untrained:
        sys.exit(f"{args.model} is untrained (no router_meta.json with trained=true)")
    results = router.predict_batch([c["text"] for c in calls], topk=1)
    passed = 0
    print(f"{'utterance':<50} {'expected':<16} {'predicted':<16} {'conf':>5}  result")
    for call, res in zip(calls, results):
        ok = res["label"] == call["label"]
        passed += ok
        print(f"{call['text']:<50} {call['label']:<16} {res['label']:<16} {res['confidence']:>5.2f}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"{passed}/{len(calls)} passed (T={router.temperature:.3f}). Trained on synthetic data only.")
    return 0 if passed == len(calls) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Score a trained checkpoint on the held-out sets: test, paraphrase_test, demo_calls, demo_calls_47.

    python -m router.eval_sets --checkpoint $NOVA_MODELS_DIR/v1 [--show-errors]

    python -m router.eval_sets --checkpoint $NOVA_MODELS_DIR/v2 --data-dir data/v2   # "test" from that data dir

REGRESSION sets, not independent validation: the paraphrase/demo files are never trained on and their exact texts are
excluded from generated data, but the v0 failures on them were known while v1/v2 templates were written, so they are
development-adjacent. `test` is the generator's own held-out split (family-held-out for data/v2, template-pool-held-out
only for data/ = v1). Untouched speaker/push-to-talk data is still needed before any generalisation claim.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

from .core import Router, load_labels, models_dir

HERE = Path(__file__).parent
SETS = (
    ("test", None),  # <data-dir>/test.jsonl
    ("paraphrase_test", "data/paraphrase_test.jsonl"),
    ("demo_calls", "demo_calls.json"),
    ("demo_calls_47", "demo_calls_47.json"),
)


def load_rows(path):
    """JSONL {text,label} rows, or a JSON file {"calls": [{text,label}]}."""
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())["calls"]
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=str(models_dir() / "v1"), help="default: %(default)s (set NOVA_MODELS_DIR to move)")
    ap.add_argument("--data-dir", default=str(HERE / "data"), help="where `test.jsonl` lives (default: %(default)s = v1 data)")
    ap.add_argument("--labels", default=str(HERE / "labels.json"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--show-errors", action="store_true")
    args = ap.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        router = Router.load(args.checkpoint, args.labels, device=args.device)
    if router.untrained:
        sys.exit(f"{args.checkpoint} is untrained")
    print(f"checkpoint {args.checkpoint} (T={router.temperature:.3f})")
    for name, rel in SETS:
        rows = load_rows(Path(args.data_dir) / "test.jsonl" if rel is None else HERE / rel)
        results = router.predict_batch([r["text"] for r in rows], topk=1)
        wrong = [(r, p) for r, p in zip(rows, results) if p["label"] != r["label"]]
        n = len(rows)
        print(f"{name:<16} {n - len(wrong)}/{n} = {(n - len(wrong)) / n:.3f}")
        if args.show_errors:
            for r, p in wrong:
                print(f"    {r['text']!r}: expected {r['label']}, got {p['label']} ({p['confidence']:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

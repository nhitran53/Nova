"""Training interface (NOT IMPLEMENTED).

Reads JSONL rows {"text": str, "label": str} (label must be in labels.json).
TODO when implemented:
  - full fine-tune of ModernBERT-base with a len(labels)-way head
  - hold-out split, temperature scaling on the dev logits (-> router_meta.json "temperature")
  - save model + tokenizer to --out and write <out>/router_meta.json:
      {"trained": true, "labels": [...], "temperature": T, ...}
    (router.py treats a missing/false marker as untrained)
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fine-tune the router (stub).")
    ap.add_argument("--data", required=True, help="JSONL of {text,label}")
    ap.add_argument("--base", default="ModernBERT-base", help="base model dir (real train.py defaults to $NOVA_MODELS_DIR/ModernBERT-base)")
    ap.add_argument("--labels", default=str(HERE / "labels.json"))
    ap.add_argument("--out", required=True, help="output model dir")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.parse_args(argv)
    print("train_stub: not implemented")
    return 1


if __name__ == "__main__":
    sys.exit(main())

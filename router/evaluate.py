"""Evaluate a trained router checkpoint on a labelled JSONL split.

    python -m router.evaluate --checkpoint $NOVA_MODELS_DIR/synthetic-v0 [--split test]

Reports accuracy, macro-F1, per-class precision/recall, confusion matrix, ECE, and a confidence-threshold sweep
(coverage vs. error), prints a short table and writes exclusive reports/<checkpoint-name>_<split>.json.

HONESTY NOTE: with the synthetic data from data/gen_synthetic.py the test split comes from the same generator
family as train, so these numbers are optimistic and are NOT evidence for the 95% target on real speech.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from .training_tools import is_trip, data_contract, check_checkpoint_contract, V4_CAVEAT, write_report
from .core import Router, load_labels, models_dir

HERE = Path(__file__).parent
THRESHOLDS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
CAVEAT = (
    "Synthetic placeholder data; the test split comes from the same generator family as train "
    "(different templates, same style), so these scores are optimistic and are NOT evidence "
    "for the 95% target on real speech."
)

V3_CAVEAT = (
    "Agent-authored synthetic v3-trip, family-separated splits; not human-heldout speech or a safety benchmark. "
    "Small correlated samples do not establish operational risk or release readiness."
)


def load_jsonl(path, labels, label_field='label'):
    """-> (texts, label indices). Use only the explicit field; never infer or relabel."""
    index = {name: i for i, name in enumerate(labels)}
    texts, ys = [], []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if label_field not in row:
            raise ValueError(f"{path}:{n}: missing label field {label_field!r}")
        label = row[label_field]
        if label not in index:
            raise ValueError(f"{path}:{n}: unknown label {label!r} in {label_field!r}")
        texts.append(row["text"])
        ys.append(index[label])
    return texts, np.array(ys)


@torch.inference_mode()
def compute_logits(model, tokenizer, texts, device, max_length, batch_size=64):
    """Raw (temperature 1) logits, shape [len(texts), num_labels], float64 numpy."""
    was_training = model.training
    model.eval()
    chunks = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length).to(device)
        chunks.append(model(**enc).logits.float().cpu().numpy())
    model.train(was_training)
    return np.concatenate(chunks).astype(np.float64)


def softmax(logits, temperature=1.0):
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def classification_metrics(y_true, y_pred, n_classes):
    """accuracy, macro_f1, per-class precision/recall/f1/support, confusion[true][pred]."""
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    tp = np.diag(confusion).astype(float)
    pred_n, true_n = confusion.sum(axis=0), confusion.sum(axis=1)
    precision = np.divide(tp, pred_n, out=np.zeros(n_classes), where=pred_n > 0)
    recall = np.divide(tp, true_n, out=np.zeros(n_classes), where=true_n > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n_classes), where=denom > 0)
    return {
        "accuracy": float(tp.sum() / max(len(y_true), 1)),
        "macro_f1": float(f1.mean()),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": true_n.tolist(),
        "confusion": confusion.tolist(),
    }


def expected_calibration_error(probs, y_true, n_bins=10):
    """ECE with equal-width confidence bins: sum_b (n_b/N) * |acc_b - conf_b|."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def threshold_sweep(probs, y_true, thresholds=THRESHOLDS):
    """For each threshold: coverage = share of items with confidence >= t; error = error rate among those."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    rows = []
    for t in thresholds:
        kept = conf >= t
        rows.append({
            "threshold": t,
            "coverage": float(kept.mean()),
            "error": float(1.0 - correct[kept].mean()) if kept.any() else None,
        })
    return rows


def build_report(logits, y_true, labels, temperature):
    probs_raw, probs_cal = softmax(logits), softmax(logits, temperature)
    report = classification_metrics(y_true, probs_cal.argmax(axis=1), len(labels))
    report["labels"] = labels
    report["temperature"] = temperature
    report["ece_uncalibrated"] = expected_calibration_error(probs_raw, y_true)
    report["ece"] = expected_calibration_error(probs_cal, y_true)
    report["threshold_sweep"] = threshold_sweep(probs_cal, y_true)
    return report


def format_table(report):
    labels = report["labels"]
    w = max(len(x) for x in labels)
    lines = [f"{'class':<{w}}  prec   rec    f1     n"]
    for i, name in enumerate(labels):
        lines.append(f"{name:<{w}}  {report['precision'][i]:.3f}  {report['recall'][i]:.3f}  "
                     f"{report['f1'][i]:.3f}  {report['support'][i]}")
    lines.append(f"accuracy {report['accuracy']:.3f} | macro-F1 {report['macro_f1']:.3f} | "
                 f"ECE {report['ece']:.3f} (uncalibrated {report['ece_uncalibrated']:.3f}, T={report['temperature']:.3f})")
    lines.append("confusion (rows=true, cols=pred): " + json.dumps(report["confusion"]))
    lines.append("threshold  coverage  error")
    for row in report["threshold_sweep"]:
        err = "  n/a" if row["error"] is None else f"{row['error']:.3f}"
        lines.append(f"{row['threshold']:<9}  {row['coverage']:.3f}     {err}")
    lines.append("NOTE: " + report.get("caveat", CAVEAT))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=str(models_dir() / "v1"), help="default: %(default)s (set NOVA_MODELS_DIR to move)")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--label-field", choices=('label', 'expected_label'), default='label',
                    help="Explicit JSONL target field; frozen heldout_scenarios/regression require expected_label")
    ap.add_argument("--labels", default=str(HERE / "labels.json"))
    ap.add_argument("--report-dir", default=str(HERE / "reports"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args(argv)

    labels = load_labels(args.labels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        router = Router.load(args.checkpoint, args.labels, device=args.device)
    if router.untrained:
        sys.exit(f"{args.checkpoint} has no router_meta.json with trained=true; nothing meaningful to evaluate")
    try:
        check_checkpoint_contract(router.meta, args.data_dir)
    except ValueError as exc:
        ap.error(str(exc))
    texts, y_true = load_jsonl(Path(args.data_dir) / f"{args.split}.jsonl", labels, args.label_field)
    logits = compute_logits(router.model, router.tokenizer, texts, router.device, router.max_length, args.batch_size)
    report = build_report(logits, y_true, labels, router.temperature)
    report.update(checkpoint=str(args.checkpoint), split=args.split, label_field=args.label_field, n=len(texts),
                  data_hash=router.meta.get("data_hash"), data_contract=data_contract(args.data_dir),
                  caveat=V4_CAVEAT if data_contract(args.data_dir) == 'v4-trip-1' else V3_CAVEAT if is_trip(args.data_dir) else CAVEAT)
    if data_contract(args.data_dir) == 'v4-trip-1' and args.split == 'seen_during_development':
        report['evidence_role'] = 'seen_during_development_not_independent'
    name = Path(args.checkpoint).resolve().name
    out = Path(args.report_dir) / f"{name}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_report(out, report)
    print(format_table(report))
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fine-tune ModernBERT-base as the router classifier (plain PyTorch loop).

    python -m router.train [--data-dir data] [--out-dir $NOVA_MODELS_DIR/synthetic-v0] [--epochs 3]

Reads <data-dir>/{train,dev,test}.jsonl ({"text","label"}; labels from labels.json, order = class index).
Each epoch is scored on dev; the best epoch (dev macro-F1, ties -> lower dev NLL) is saved with its tokenizer
to --out-dir. Temperature scaling is then fitted on that model's dev logits, test is scored once for the
record, and --out-dir/router_meta.json is written (trained: true, labels, epochs, data hash, dev/test macro-F1,
temperature). Uses MPS, then CUDA, then CPU (--device overrides).

HONESTY NOTE: with the synthetic data from data/gen_synthetic.py, dev/test come from the same generator family
as train, so the scores are optimistic and are NOT evidence for the 95% target on real speech.
"""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .core import META_FILENAME, base_model_dir, load_labels, models_dir, resolve_device
from .evaluate import (CAVEAT, classification_metrics, compute_logits, expected_calibration_error, load_jsonl,
                       softmax)

HERE = Path(__file__).parent
SPLITS = ("train", "dev", "test")


def data_hash(data_dir, labels):
    """sha256 over the label order and the raw bytes of train/dev/test.jsonl."""
    h = hashlib.sha256(json.dumps(labels).encode())
    for split in SPLITS:
        h.update((Path(data_dir) / f"{split}.jsonl").read_bytes())
    return h.hexdigest()


def fit_temperature(logits, y_true, max_iter=100):
    """Temperature T > 0 minimising dev NLL of softmax(logits / T) (LBFGS on log T)."""
    z = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(y_true)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def nll(logits, y_true, temperature=1.0):
    p = softmax(logits, temperature)
    return float(-np.log(p[np.arange(len(y_true)), y_true] + 1e-12).mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out-dir", default=str(models_dir() / "synthetic-v0"), help="default: %(default)s (set NOVA_MODELS_DIR to move)")
    ap.add_argument("--base", default=str(base_model_dir()), help="default: %(default)s")
    ap.add_argument("--labels", default=str(HERE / "labels.json"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    labels = load_labels(args.labels)
    device = resolve_device(args.device)
    train_x, train_y = load_jsonl(Path(args.data_dir) / "train.jsonl", labels)
    dev_x, dev_y = load_jsonl(Path(args.data_dir) / "dev.jsonl", labels)
    test_x, test_y = load_jsonl(Path(args.data_dir) / "test.jsonl", labels)
    print(f"device={device} labels={len(labels)} train/dev/test={len(train_x)}/{len(dev_x)}/{len(test_x)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={name: i for i, name in enumerate(labels)},
    ).to(device)

    steps_per_epoch = -(-len(train_x) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    rng = np.random.default_rng(args.seed)

    best = {"macro_f1": -1.0, "nll": float("inf"), "epoch": 0, "state": None, "logits": None}
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_x))
        losses = []
        for step in range(steps_per_epoch):
            idx = order[step * args.batch_size:(step + 1) * args.batch_size]
            enc = tokenizer([train_x[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                            max_length=args.max_length).to(device)
            labels_t = torch.tensor(train_y[idx], device=device)
            loss = model(**enc, labels=labels_t).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
            if (step + 1) % 25 == 0 or step + 1 == steps_per_epoch:
                done = (epoch - 1) * steps_per_epoch + step + 1
                elapsed = time.perf_counter() - start
                eta = elapsed / done * (total_steps - done)
                print(f"  epoch {epoch}/{args.epochs} step {step + 1}/{steps_per_epoch} "
                      f"(overall {done}/{total_steps}) loss {np.mean(losses[-25:]):.4f} "
                      f"elapsed {elapsed:.0f}s eta {eta:.0f}s", flush=True)
        dev_logits = compute_logits(model, tokenizer, dev_x, device, args.max_length)
        dev_metrics = classification_metrics(dev_y, dev_logits.argmax(axis=1), len(labels))
        dev_nll = nll(dev_logits, dev_y)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_macro_f1": dev_metrics["macro_f1"],
                        "dev_accuracy": dev_metrics["accuracy"], "dev_nll": dev_nll})
        print(f"epoch {epoch}: train loss {np.mean(losses):.4f} | dev macro-F1 {dev_metrics['macro_f1']:.4f} "
              f"acc {dev_metrics['accuracy']:.4f} nll {dev_nll:.4f} | {time.perf_counter() - start:.0f}s", flush=True)
        if (dev_metrics["macro_f1"], -dev_nll) > (best["macro_f1"], -best["nll"]):
            best.update(macro_f1=dev_metrics["macro_f1"], nll=dev_nll, epoch=epoch, logits=dev_logits,
                        state=copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}))

    model.load_state_dict(best["state"])
    temperature = fit_temperature(best["logits"], dev_y)
    test_logits = compute_logits(model, tokenizer, test_x, device, args.max_length)
    test_metrics = classification_metrics(test_y, test_logits.argmax(axis=1), len(labels))
    out_dir = Path(args.out_dir)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    meta = {
        "trained": True,
        "labels": labels,
        "base_model": str(args.base),
        "epochs": args.epochs,
        "best_epoch": best["epoch"],
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "device": str(device),
        "data_hash": data_hash(args.data_dir, labels),
        "n_train": len(train_x),
        "dev_macro_f1": best["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "temperature": temperature,
        "dev_ece_uncalibrated": expected_calibration_error(softmax(best["logits"]), dev_y),
        "dev_ece_calibrated": expected_calibration_error(softmax(best["logits"], temperature), dev_y),
        "history": history,
        "synthetic_data": True,
        "caveat": CAVEAT,
    }
    (out_dir / META_FILENAME).write_text(json.dumps(meta, indent=2) + "\n")
    print(f"best epoch {best['epoch']} | dev macro-F1 {best['macro_f1']:.4f} | test macro-F1 "
          f"{test_metrics['macro_f1']:.4f} | T={temperature:.3f} | saved {out_dir}")
    print("NOTE: " + CAVEAT)
    return 0


if __name__ == "__main__":
    sys.exit(main())

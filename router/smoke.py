"""Bounded CUDA optimizer compatibility/resource trial, NOT a trained checkpoint or promotion candidate."""
import argparse
import json
import math
import platform
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .core import base_model_dir, load_labels
from .evaluate import load_jsonl
from .training_tools import checked_device, cuda_probe, write_report

HERE = Path(__file__).parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', default=str(base_model_dir()))
    ap.add_argument('--data-dir', default=str(HERE / 'data/v3-trip'))
    ap.add_argument('--labels', default=str(HERE / 'labels.json'))
    ap.add_argument('--report', required=True, help='New JSON report path, refuses overwrite; no checkpoint is saved')
    ap.add_argument('--device', default='cuda', help='CUDA only; CPU training is forbidden')
    ap.add_argument('--require-cuda', action='store_true', default=True, help='Always enforced for smoke')
    ap.add_argument('--steps', type=int, default=3, help='Measured optimizer steps, 1..20, plus one warmup step')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--max-length', type=int, default=64)
    ap.add_argument('--gradient-checkpointing', action='store_true')
    ap.add_argument('--mixed-precision', choices=('no', 'bf16'), default='no')
    args = ap.parse_args(argv)
    if not 1 <= args.steps <= 20 or not 1 <= args.batch_size <= 4 or not 1 <= args.max_length <= 64:
        ap.error('bounded trial requires steps 1..20, batch 1..4, length 1..64')
    if Path(args.report).exists():
        ap.error('report already exists')
    report = {'kind': 'optimizer_smoke_only', 'trained': False, 'promotion_candidate': False,
              'python': platform.python_version(), 'transformers': transformers.__version__, 'platform': platform.platform(), 'settings': vars(args),
              'steps': [], 'success': False}
    start = time.perf_counter()
    try:
        device = checked_device(args.device, True)
        report['environment'] = cuda_probe(device)
        print(json.dumps(report['environment']), flush=True)
        labels = load_labels(args.labels)
        texts, ys = load_jsonl(Path(args.data_dir) / 'train.jsonl', labels)
        tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base, num_labels=len(labels), local_files_only=True).to(device)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        measured = []
        for step in range(args.steps + 1):
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            idx = [(step * args.batch_size + i) % len(texts) for i in range(args.batch_size)]
            enc = tokenizer([texts[i] for i in idx], return_tensors='pt', padding='max_length',
                            truncation=True, max_length=args.max_length).to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.mixed_precision == 'bf16'):
                loss = model(**enc, labels=torch.tensor(ys[idx], device=device)).loss
            if not torch.isfinite(loss).item():
                raise ValueError('non-finite smoke loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - tick
            row = {'step': step, 'warmup': step == 0, 'loss': loss.item(), 'elapsed_s': elapsed}
            report['steps'].append(row)
            print(json.dumps(row), flush=True)
            if step:
                measured.append(elapsed)
        measured_s = sum(measured)
        report.update(success=True, actual_encoder_forward_backward=True,
                      measured_optimizer_s=measured_s, examples_per_s=args.steps * args.batch_size / measured_s,
                      extrapolated_three_epoch_optimizer_s=3 * math.ceil(len(texts) / args.batch_size) *
                      measured_s / args.steps,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                      overhead_s=time.perf_counter() - start - measured_s,
                      estimate_excludes='dev evaluation, calibration, test, checkpoint IO; not a fit or runtime guarantee')
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    if 'environment' in report:
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
    report['elapsed_s'] = time.perf_counter() - start
    write_report(args.report, report)
    print(json.dumps(report, indent=2))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

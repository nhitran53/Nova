"""Fit temperature on an independent calibration JSONL; emit evidence only, never mutate a checkpoint."""
import argparse
from pathlib import Path

from .core import Router
from .evaluate import compute_logits, load_jsonl
from .train import fit_temperature, nll
from .training_tools import calibration_input, file_hash, is_trip, data_contract, check_checkpoint_contract, V4_CAVEAT, write_report

HERE = Path(__file__).parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data-dir', default=str(HERE / 'data/v3-trip'))
    ap.add_argument('--calibration-path', help='Independent input; v3/v4 never fall back to dev')
    ap.add_argument('--labels', default=str(HERE / 'labels.json'))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--report', required=True, help='New JSON path; fitted T is not installed into the checkpoint')
    args = ap.parse_args(argv)
    path = calibration_input(args.data_dir, args.calibration_path)
    if Path(args.report).exists():
        ap.error('report already exists')
    router = Router.load(args.checkpoint, args.labels, device=args.device)
    if router.untrained:
        ap.error('cannot calibrate an untrained/smoke checkpoint')
    try:
        check_checkpoint_contract(router.meta, args.data_dir)
    except ValueError as exc:
        ap.error(str(exc))
    texts, ys = load_jsonl(path, router.labels)
    logits = compute_logits(router.model, router.tokenizer, texts, router.device, router.max_length, args.batch_size)
    temperature = fit_temperature(logits, ys)
    report = {'kind': 'temperature_fit_only', 'checkpoint': args.checkpoint,
              'calibration_path': str(path), 'calibration_sha256': file_hash(path), 'n': len(texts),
              'temperature': temperature, 'nll_before': nll(logits, ys), 'nll_after': nll(logits, ys, temperature),
              'checkpoint_unchanged': True, 'legacy_dev_reuse': not is_trip(args.data_dir) and path.resolve() == (Path(args.data_dir) / 'dev.jsonl').resolve()}
    report['data_contract'] = data_contract(args.data_dir)
    if report['data_contract'] == 'v4-trip-1':
        report['caveat'] = V4_CAVEAT
    write_report(args.report, report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Offline v3 data audit and optional actual CUDA tensor/backward probe (no training)."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .training_tools import calibration_input, data_contract, cuda_probe, file_hash, write_report

HERE = Path(__file__).parent


def verify_frozen(data_dir):
    root = Path(data_dir)
    hashes = json.loads((root / 'frozen_hashes.json').read_text())
    for name, expected in hashes.items():
        if file_hash(root / name) != expected:
            raise ValueError(f'frozen hash mismatch: {name}')
    return hashes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', default=str(HERE / 'data/v3-trip'))
    ap.add_argument('--require-cuda', action='store_true', help='Fail if actual CUDA tensor/backward probe fails; encoder compatibility still requires smoke')
    ap.add_argument('--report', required=True, help='New JSON report path; refuses overwrite')
    args = ap.parse_args(argv)
    if Path(args.report).exists():
        ap.error('report already exists')
    report = {'kind': 'preflight_no_training', 'passed': False, 'encoder_compatibility': 'not tested; run router.smoke on CUDA'}
    try:
        report['frozen_sha256'] = verify_frozen(args.data_dir)
        report['calibration_path'] = str(calibration_input(args.data_dir))
        contract = data_contract(args.data_dir)
        report['data_contract'] = contract
        auditor = 'v4-trip' if contract == 'v4-trip-1' else 'v3-trip'
        command = [sys.executable, str(HERE / f'data/{auditor}/release.py'), '--check', '--out-dir', args.data_dir]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        report['data_audit'] = {'command': command, 'returncode': result.returncode,
                                'stdout': result.stdout, 'stderr': result.stderr}
        if result.returncode:
            raise ValueError('data release audit failed')
        if args.require_cuda:
            report['cuda_probe'] = cuda_probe()
        report['passed'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    write_report(args.report, report)
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

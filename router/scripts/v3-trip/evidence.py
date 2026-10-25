#!/usr/bin/env python3
"""Inventory explicit teammate evidence and candidate files, including failed runs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('transfer', Path(__file__).with_name('transfer.py'))
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


def inventory(root, checkpoint=False):
    result = {}
    if not root.exists():
        return result
    if root.is_symlink():
        raise ValueError('symlink root rejected')
    source_records = {'source/TRANSFER-MANIFEST.json', 'source.tar.gz'}
    if not checkpoint and any((root / name).exists() or (root / name).is_symlink()
                              for name in ('source', 'source.tar.gz')):
        # Verify actual source bytes, not just the saved source-snapshot report.
        if any((root / name).is_symlink() for name in ('source', 'source.tar.gz')):
            raise ValueError('symlink source rejected')
        transfer.verify_folder(root / 'source')
        transfer.verify_archive(root / 'source.tar.gz')
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        is_source_record = not checkpoint and relative.as_posix() in source_records
        # Only source integrity records are inventoried, not arbitrary source/report trees.
        if not checkpoint and not is_source_record and len(relative.parts) > 1 and relative.parts[0] not in {'gate', 'evaluations'}:
            continue
        if path.name == 'RETURN-MANIFEST.json':
            continue
        if any(p.startswith('.') or p in transfer.FORBIDDEN for p in relative.parts):
            continue
        suffixes = {'.json', '.txt', '.safetensors', '.model'} if checkpoint else {'.json', '.txt', '.log', '.tsv', '.md'}
        if not is_source_record and path.suffix not in suffixes:
            continue
        if path.is_symlink():
            raise ValueError('symlink evidence rejected')
        if not path.is_file():
            continue
        h = hashlib.sha256()
        with path.open('rb') as stream:
            while block := stream.read(1024 * 1024):
                if path.suffix not in {'.safetensors', '.model', '.gz'}:
                    transfer.scan(block)
                h.update(block)
        result[str(relative)] = {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--verify', action='store_true', help='Compare inventory to existing return manifest')
    args = ap.parse_args()
    run, checkpoint = Path(args.run_dir), Path(args.checkpoint)
    manifest = {'format': 1, 'evidence': inventory(run), 'checkpoint': inventory(checkpoint, True),
                'candidate_present': checkpoint.is_dir()}
    output = run / 'RETURN-MANIFEST.json'
    if args.verify:
        if json.loads(output.read_text()) != manifest:
            raise ValueError('return manifest mismatch')
        print('Return inventory verified; this does not approve candidate release.')
    else:
        with output.open('x') as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write('\n')
        print(output)
        print('sha256', transfer.digest(output.read_bytes()))


if __name__ == '__main__':
    main()

"""Counts-only overlap of an external text JSONL against the five v4 classifier splits.

No external text, file path, hashes, or examples are printed or written. Read-only.
"""
import argparse
import json
from pathlib import Path
import sys

# Load our sibling, not any generator supplied by the external data directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import release


def overlap_counts(rows, groups, near_threshold=release.NEAR_THRESHOLD):
    if not 0 < near_threshold <= 1:
        raise ValueError('invalid threshold')
    external = [(r['text'], release.normalize(r['text']), release.features(r['text'])) for r in rows]
    counts = {}
    for split, internal in groups.items():
        exact = {r['text'] for r in internal}
        normalized = {release.normalize(r['text']) for r in internal}
        feats = [release.features(r['text']) for r in internal]
        counts[split] = {
            'exact_external_rows': sum(text in exact for text, _, _ in external),
            'normalized_external_rows': sum(norm in normalized for _, norm, _ in external),
            'near_external_rows': sum(any(release.similarity(f, other) >= near_threshold for other in feats)
                                      for _, _, f in external),
            'near_pairs': sum(release.similarity(f, other) >= near_threshold
                              for _, _, f in external for other in feats)}
    return {'external_rows': len(external), 'splits': counts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('other', type=Path)
    parser.add_argument('--near-threshold', type=float, default=release.NEAR_THRESHOLD)
    args = parser.parse_args(argv)
    try:
        groups = {s: release.read_rows(release.HERE / f'{s}.jsonl') for s in release.SPLITS}
        result = overlap_counts(release.read_rows(args.other), groups, args.near_threshold)
    except Exception:
        # In particular, JSON/key errors must not disclose external text or its path.
        print(json.dumps({'error_count': 1}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

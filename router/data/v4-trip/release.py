"""Deterministic stdlib-only v4-trip synthesis/audit. No training or model access."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

HERE = Path(__file__).resolve().parent
SPLITS = ('train', 'dev', 'calibration', 'threshold', 'test')
LABELS = ('call_safety', 'call_navigation', 'call_map', 'call_data', 'call_o2', 'call_llm', 'NO_TOOL', 'OOD')
SEED = 41024
PLACES = ('base 2', 'base two', 'base2', 'second base', 'base-2', 'home', 'homebase', 'base b', 'base 3', 'crater 19', 'base too', 'crater 41')
RESOURCES = ('oxygen', 'O2', 'air', 'tank')
FROZEN = ('seen_during_development.jsonl', 'clarification_probe.jsonl')
NEAR_THRESHOLD = 0.72
HARD_THRESHOLD = 0.985
# Intent-bearing vocabulary may legitimately occur in only one class. Incidental
# function words, discourse markers and authoring artifacts are not exemptions.
INTENT_WORD_ALLOWLIST = frozenset({
    'explain', 'why', 'buy', 'tickets', 'display', 'view', 'percentage',
    'minutes', 'bpm', 'vitals', 'basic', 'generic',
})
TOKEN_PURITY_MIN_ROWS = 24


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def jsonl(rows):
    return ''.join(dumps(r) + '\n' for r in rows).encode()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalize(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def delexicalize(text):
    text = re.sub(r'\{(place|resource)\}', lambda m: m[1].casefold() + '_slot', text, flags=re.IGNORECASE)
    text = normalize(text)
    places = sorted({normalize(p) for p in PLACES}, key=len, reverse=True)
    text = re.sub(r'\b(?:' + '|'.join(re.escape(p) for p in places) + r'|crater \d+)\b', 'place_slot', text)
    return re.sub(r'\b(?:oxygen|o2|air|tank)\b', 'resource_slot', text)


def features(text):
    text = delexicalize(text)
    return set(text.split()), {text[i:i+5] for i in range(max(1, len(text)-4))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0


def similarity(a, b):
    return max(jaccard(a[0], b[0]), jaccard(a[1], b[1]))


def near_duplicates(groups, threshold=NEAR_THRESHOLD, hard_threshold=HARD_THRESHOLD):
    """Report cross-split delexicalized token/character-5-gram Jaccard; never auto-drop."""
    if not 0 < threshold <= hard_threshold <= 1:
        raise ValueError('require 0 < near threshold <= hard threshold <= 1')
    flat = [(s, r, features(r['text'])) for s, rows in groups.items() for r in rows]
    pairs = []
    for i, (split, row, feat) in enumerate(flat):
        for other_split, other, other_feat in flat[i+1:]:
            if split == other_split:
                continue
            # Each Jaccard is bounded above by min(size)/max(size).
            if all(min(len(a), len(b)) / max(1, len(a), len(b)) < threshold for a, b in zip(feat, other_feat)):
                continue
            score = similarity(feat, other_feat)
            if score >= threshold:
                pairs.append({'left': row['row_id'], 'right': other['row_id'],
                              'splits': [split, other_split], 'score': round(score, 6),
                              'cross_label': row['label'] != other['label'], 'hard': score >= hard_threshold})
    return {'method': 'max(delexicalized token Jaccard, delexicalized character-5-gram Jaccard)',
            'review_threshold': threshold, 'hard_threshold': hard_threshold,
            'flagged_pairs': len(pairs), 'hard_pairs': sum(p['hard'] for p in pairs),
            'max_score': max((p['score'] for p in pairs), default=0), 'pairs': pairs}


def token_purity(rows):
    """Count row presence (not repetitions); gate only non-intent pure tokens >=24."""
    counts = defaultdict(Counter)
    for row in rows:
        for token in set(normalize(row['text']).split()):
            counts[token][row['label']] += 1
    pure = [{'token': token, 'rows': sum(labels.values()), 'label': next(iter(labels)),
             'allowlisted': token in INTENT_WORD_ALLOWLIST}
            for token, labels in sorted(counts.items())
            if len(labels) == 1 and sum(labels.values()) >= 16]
    violations = [p for p in pure if p['rows'] >= TOKEN_PURITY_MIN_ROWS and not p['allowlisted']]
    return {'method': 'normalized token presence per training row',
            'failure_min_rows': TOKEN_PURITY_MIN_ROWS,
            'intent_word_allowlist': sorted(INTENT_WORD_ALLOWLIST),
            'pure_tokens_ge16_count': len(pure), 'pure_tokens_ge16': pure,
            'non_allowlisted_pure_tokens_ge24': violations}


def construction_label_counts(rows):
    return {construction: dict(sorted(Counter(r['label'] for r in rows
                                              if r['construction'] == construction).items()))
            for construction in sorted({r['construction'] for r in rows})}


def load_families():
    return json.loads((HERE / 'families.json').read_text())


def build():
    groups = {s: [] for s in SPLITS}
    families = load_families()
    constructions = list(dict.fromkeys(f['construction'] for f in families))
    for family in families:
        ci = constructions.index(family['construction'])
        for ti, template in enumerate(family['templates']):
            for vi in range(4):
                place = PLACES[(ci * 3 + vi) % len(PLACES)]
                resource = RESOURCES[vi]
                text = template.format(place=place, resource=resource)
                if '{' not in template:
                    text = ('', 'hey, ', 'um, ', 'please, ')[vi] + text
                row = {'text': text, 'label': family['label'], 'split': family['split'],
                       'row_id': f"{family['family_id']}:{ti}:{vi}", 'family_id': family['family_id'],
                       'template_id': f"{family['family_id']}:{ti}", 'construction': family['construction'],
                       'contrast_group': family['contrast_group'], 'place_surface': place,
                       'resource_surface': resource, 'provenance': 'agent_authored_synthetic_not_human_speech'}
                if row['label'] == 'call_navigation':
                    row['destination_status'] = 'missing' if family['construction'] == 'truncated' else 'requires_runtime_resolution'
                    row['expected_disposition'] = 'clarify_missing_destination' if family['construction'] == 'truncated' else 'navigation_candidate_not_clearance'
                groups[family['split']].append(row)
    for i, rows in enumerate(groups.values()):
        random.Random(SEED+i).shuffle(rows)
    return groups


def audit(groups, families):
    seen = {key: {} for key in ('text', 'normalized', 'family_id', 'template_id')}
    for split, rows in groups.items():
        local = set()
        for row in rows:
            if row['label'] not in LABELS:
                raise ValueError('unknown label')
            values = {'text': row['text'], 'normalized': normalize(row['text']),
                      'family_id': row['family_id'], 'template_id': row['template_id']}
            for key, value in values.items():
                previous = seen[key].get(value)
                if previous is not None and previous != split:
                    raise ValueError(f'{key} leakage: {previous} -> {split}')
                seen[key][value] = split
            if values['normalized'] in local:
                raise ValueError(f'duplicate within {split}')
            local.add(values['normalized'])
    frames = {}
    for family in families:
        for template in family['templates']:
            frame = delexicalize(template)
            if frame in frames and frames[frame] != family['split']:
                raise ValueError('template frame leakage')
            frames[frame] = family['split']
    for split, rows in groups.items():
        counts = Counter(r['label'] for r in rows)
        if set(counts) != set(LABELS) or max(counts.values()) > len(rows)/4 or max(counts.values()) > 2*min(counts.values()):
            raise ValueError(f'label imbalance: {split}')
        minimum = 1000 if split == 'train' else 200 if split in ('calibration', 'threshold') else 150
        if len(rows) < minimum:
            raise ValueError(f'insufficient rows: {split}')
        if max(Counter(r['family_id'] for r in rows).values()) > 16:
            raise ValueError('family cap exceeded')
    for family in families:
        if len(family['templates']) < 4:
            raise ValueError('fewer than four templates')
    sources = [HERE.parent / 'v3-trip' / name for name in ('heldout_scenarios.jsonl', 'regression.jsonl')]
    legacy = {normalize(r['text']) for p in sources for r in read_rows(p)}
    for split, rows in groups.items():
        if any(normalize(r['text']) in legacy for r in rows):
            raise ValueError(f'old seed normalized overlap: {split}')


    coverage = construction_label_counts(groups['train'])
    if any(len(labels) < 5 for labels in coverage.values()):
        raise ValueError('training construction covers fewer than five labels')
    if any(r['construction'] == 'contrastive_correction'
           for split, rows in groups.items() if split != 'test' for r in rows):
        raise ValueError('held-out construction outside test')
    if token_purity(groups['train'])['non_allowlisted_pure_tokens_ge24']:
        raise ValueError('non-allowlisted pure training token at >=24 rows')


def artifacts(near_threshold=NEAR_THRESHOLD, hard_threshold=HARD_THRESHOLD):
    groups = build()
    audit(groups, load_families())
    if list(LABELS) != json.loads((HERE.parents[1] / 'labels.json').read_text())['labels']:
        raise ValueError('label order changed')
    locks = json.loads((HERE / 'frozen_hashes.json').read_text())
    if set(locks) != set(FROZEN):
        raise ValueError('unexpected frozen inventory')
    for name in FROZEN:
        if digest((HERE / name).read_bytes()) != locks[name]:
            raise ValueError(f'frozen file changed: {name}')
    report = near_duplicates(groups, near_threshold, hard_threshold)
    if report['hard_pairs']:
        raise ValueError(f"hard near-duplicate leakage: {report['hard_pairs']} pairs")
    output = {f'{s}.jsonl': jsonl(rows) for s, rows in groups.items()}
    output['near_duplicate_report.json'] = (json.dumps(report, indent=2, sort_keys=True)+'\n').encode()
    manifest = {'dataset_name': 'v4-trip', 'schema_version': 'v4-trip-1.2',
                'version': 'v4.2', 'seed': SEED, 'label_order': LABELS,
                'provenance': 'agent-authored synthetic v4.2 revision, not human speech/safety benchmark',
                'approved_relaxation': '164 families; cap 16; equal train labels',
                'superseded_release': json.loads((HERE / 'SUPERSEDED_V4_1.json').read_text()),
                'token_purity_summary': token_purity(groups['train']),
                'train_construction_label_counts': construction_label_counts(groups['train']),
                'family_sha256': digest((HERE / 'families.json').read_bytes()),
                'generator_sha256': digest(Path(__file__).read_bytes()), 'frozen_sha256': locks,
                'heldout_constructions': ['contrastive_correction'],
                'heldout_scope': 'Explicit repair-marker families (I meant / Correction: / Actually / Let me rephrase:) remain test-only; contrastive scope selection and quoted-mention contrasts now occur in train. Broad correction semantics are NOT held out.',
                'near_duplicate_summary': {k: v for k, v in report.items() if k != 'pairs'},
                'splits': {}}
    for split, rows in groups.items():
        manifest['splits'][split] = {'rows': len(rows), 'labels': dict(Counter(r['label'] for r in rows)),
                                    'constructions': dict(Counter(r['construction'] for r in rows)),
                                    'families': len({r['family_id'] for r in rows}),
                                    'sha256': digest(output[f'{split}.jsonl'])}
    manifest['legacy_sources'] = {name: digest((HERE.parent / 'v3-trip' / name).read_bytes()) for name in ('heldout_scenarios.jsonl', 'regression.jsonl')}
    manifest['artifact_sha256'] = {name: digest(content) for name, content in output.items()}
    output['manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode()
    return output, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--out-dir', type=Path, default=HERE)
    parser.add_argument('--near-threshold', type=float, default=NEAR_THRESHOLD)
    parser.add_argument('--hard-threshold', type=float, default=HARD_THRESHOLD)
    args = parser.parse_args(argv)
    output, manifest = artifacts(args.near_threshold, args.hard_threshold)
    for name, content in output.items():
        path = args.out_dir / name
        if args.check:
            if not path.exists() or path.read_bytes() != content:
                raise ValueError(f'not reproducible: {path.name}')
        else:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    print(dumps({s: m['rows'] for s, m in manifest['splits'].items()}))
    print(dumps(manifest['near_duplicate_summary']))
    print(dumps(manifest['token_purity_summary']))
    print('PASS: hashes, counts, old-seed exclusion, exact/normalized/family/template/frame audit')


if __name__ == '__main__':
    main()

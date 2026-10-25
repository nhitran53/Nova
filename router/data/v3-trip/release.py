"""Deterministic, stdlib-only v3-trip builder and leakage auditor; never trains a model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

HERE = Path(__file__).resolve().parent
SPLITS = ('train', 'dev', 'calibration', 'threshold', 'test')
LABELS = ('call_safety', 'call_navigation', 'call_map', 'call_data', 'call_o2', 'call_llm', 'NO_TOOL', 'OOD')
SEED = 31024
FROZEN = ('heldout_scenarios.jsonl', 'regression.jsonl', 'replay_scenarios.jsonl')
# Shared core entities are intentional; unseen numbered suffixes occur only in test.
PLACES = {
    'train': ('home', 'base 2', 'tower'),
    'dev': ('home base', 'base2', 'comms tower'),
    'calibration': ('base camp', 'second base', 'ridge'),
    'threshold': ('homebase', 'base b', 'airlock'),
    'test': ('home', 'base3', 'crater 7'),
}
SENSORS = ('suit pressure', 'heart rate', 'suit temperature')
TOPICS = ('orbital mechanics', 'photosynthesis', 'neutron stars')
PLACE_IDS = {'home': 'homebase', 'home base': 'homebase', 'base camp': 'homebase',
             'homebase': 'homebase', 'base 2': 'base_2', 'base2': 'base_2',
             'second base': 'base_2', 'base b': 'base_2', 'tower': 'comms_tower',
             'comms tower': 'comms_tower', 'ridge': 'ridge', 'airlock': 'airlock'}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def jsonl(rows):
    return ''.join(dumps(row) + '\n' for row in rows).encode()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ' '.join(re.findall(r'\w+', text))


def build():
    families = json.loads((HERE / 'families.json').read_text())
    result = {s: [] for s in SPLITS}
    for family in families:
        split = family['split']
        for t, template in enumerate(family['templates']):
            has_slots = '{' in template
            for k in range(3 if has_slots else 1):
                place = PLACES[split][k]
                text = template.format(place=place, sensor=SENSORS[k], topic=TOPICS[k])
                label = family['label']
                row = {'text': text, 'label': label, 'split': split,
                       'family_id': family['family_id'], 'template_id': f"{family['family_id']}-{t}",
                       'procedure_family': family['procedure_family'],
                       'provenance': 'agent_authored_template_expansion'}
                if label == 'call_navigation':
                    mode = family.get('response_mode') or ('assessment', 'distance', 'guidance')[(int(family['family_id'].split('-')[-1])-1)%3]
                    destination = family.get('destination', PLACE_IDS.get(place))
                    status = family.get('destination_status', 'resolved' if destination else 'unknown')
                    row.update(response_mode=mode, destination=destination,
                               destination_status=status,
                               expected_disposition='candidate_navigation_requires_valid_slot')
                    if status != 'resolved':
                        row['expected_tool_calls'] = 0
                if family.get('response_mode') == 'unsupported_hazard':
                    row.update(response_mode='unsupported_hazard', expected_disposition='scope_refusal_or_clarify',
                               expected_tool_calls=0, clearance=False)
                result[split].append(row)
    for split in SPLITS:
        random.Random(SEED + SPLITS.index(split)).shuffle(result[split])
    return result


def audit(groups, families):
    """Raise on exact/normalized text, family, procedure or template crossing a split."""
    seen = {key: {} for key in ('text', 'normalized', 'family_id', 'procedure_family', 'template_id')}
    for split, rows in groups.items():
        for row in rows:
            text = row.get('text')
            if text is None:  # Replay scripts are stateful, not classifier examples.
                text = row['turns'][0]['text']
            label = row.get('label', row.get('expected_label'))
            if label is not None and label not in LABELS:
                raise ValueError(f'unknown label: {label}')
            values = dict(text=text, normalized=normalize(text))
            values.update({key: row[key] for key in ('family_id', 'procedure_family', 'template_id') if key in row})
            for key, value in values.items():
                previous = seen[key].get(value)
                if previous and previous != split:
                    raise ValueError(f'{key} leakage: {previous} -> {split}: {value}')
                seen[key][value] = split
    frames = {}
    for family in families:
        for template in family['templates']:
            frame = normalize(template)
            if frame in frames and frames[frame] != family['split']:
                raise ValueError(f'template frame leakage: {frame}')
            frames[frame] = family['split']
    for split in SPLITS:
        if set(r['label'] for r in groups[split]) != set(LABELS):
            raise ValueError(f'missing labels in {split}')
        texts = [normalize(r['text']) for r in groups[split]]
        if len(set(texts)) != len(texts):
            raise ValueError(f'duplicate within {split}')


def protected_sources():
    router = HERE.parent.parent
    return sorted(router.glob('demo_calls*.json')) + [HERE.parent / 'paraphrase_test.jsonl']


def legacy_audit():
    """Inventory incompatible historical labels, not a relabeler or a scorer."""
    paths = [HERE.parent / 'v2' / f'{s}.jsonl' for s in ('train', 'dev', 'test')]
    paths += protected_sources()
    report = {'status': 'historical_labels_incompatible_do_not_score_as_v3',
              'method': 'casefolded regex triage only; manual adjudication required', 'files': {}}
    for path in paths:
        rows = json.loads(path.read_text())['calls'] if path.suffix == '.json' else read_rows(path)
        flagged = Counter()
        for row in rows:
            if re.search(r'can i make|will i make|enough margin|can i safely|can.*return.*safe|'
                         r'(oxygen|o2|air).*(trip|reach|go |get |last until)', row['text'].casefold()):
                flagged[row['label']] += 1
        report['files'][str(path.relative_to(HERE.parent.parent))] = {
            'sha256': digest(path.read_bytes()), 'rows': len(rows),
            'label_counts': dict(Counter(r['label'] for r in rows)),
            'heuristic_review_candidates_by_old_label': dict(flagged)}
    return report


def artifacts():
    data = build()
    groups = dict(data)
    groups.update({name: read_rows(HERE / name) for name in FROZEN})
    families = json.loads((HERE / 'families.json').read_text())
    audit(groups, families)
    if tuple(json.loads((HERE.parent.parent / 'labels.json').read_text())['labels']) != LABELS:
        raise ValueError('label order changed; v3-trip requires explicit version migration')
    banned = set()
    protected_hashes = {}
    for path in protected_sources():
        rows = json.loads(path.read_text())['calls'] if path.suffix == '.json' else read_rows(path)
        banned.update(normalize(r['text']) for r in rows)
        protected_hashes[path.name] = digest(path.read_bytes())
    for split, rows in data.items():
        if any(normalize(row['text']) in banned for row in rows):
            raise ValueError(f'protected regression overlap in {split}')
    locks = json.loads((HERE / 'frozen_hashes.json').read_text())
    for name in FROZEN:
        if digest((HERE / name).read_bytes()) != locks[name]:
            raise ValueError(f'frozen file changed: {name}; requires explicit review/versioning')
    output = {f'{s}.jsonl': jsonl(rows) for s, rows in data.items()}
    manifest = {'schema_version': 'v3-trip-1', 'seed': SEED, 'label_order': LABELS,
                'provenance': 'agent-authored synthetic; no human-heldout or measured speech',
                'source_sha256': digest((HERE / 'families.json').read_bytes()),
                'generator_sha256': digest(Path(__file__).read_bytes()),
                'frozen_sha256': locks, 'splits': {}}
    for split, rows in data.items():
        manifest['splits'][split] = {'rows': len(rows), 'labels': dict(Counter(r['label'] for r in rows)),
                                    'families': len({r['family_id'] for r in rows}),
                                    'sha256': digest(output[f'{split}.jsonl'])}
    manifest['protected_regression_sha256'] = protected_hashes
    manifest['frozen_counts'] = {name: len(groups[name]) for name in FROZEN}
    output['legacy_audit.json'] = (json.dumps(legacy_audit(), indent=2, sort_keys=True) + '\n').encode()
    manifest['legacy_audit_sha256'] = digest(output['legacy_audit.json'])
    output['manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    return output, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='audit and compare bytes without writing')
    parser.add_argument('--out-dir', type=Path, default=HERE)
    args = parser.parse_args(argv)
    output, manifest = artifacts()
    for name, content in output.items():
        path = args.out_dir / name
        if args.check:
            if not path.exists() or path.read_bytes() != content:
                raise ValueError(f'not reproducible: {path}')
        else:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    print(dumps({s: m['rows'] for s, m in manifest['splits'].items()}))
    print('PASS: frozen hashes, label coverage, exact/normalized and family/template/procedure leakage')


if __name__ == '__main__':
    main()

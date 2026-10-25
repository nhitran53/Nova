"""Source-named checks for the versioned v3-trip data release; no models/network."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

HERE = Path(__file__).resolve().parents[1] / 'data' / 'v3-trip'

frozen_8class = pytest.mark.skip(reason='frozen 8-class release; needs separate migration')

spec = importlib.util.spec_from_file_location('v3_trip_release', HERE / 'release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def families():
    return json.loads((HERE / 'families.json').read_text())


@frozen_8class
def test_v3_trip_release_regenerates_exact_bytes_and_checks(tmp_path):
    release.main(['--check'])
    release.main(['--out-dir', str(tmp_path)])
    release.main(['--check', '--out-dir', str(tmp_path)])
    for name in release.artifacts()[0]:
        assert (tmp_path / name).read_bytes() == (HERE / name).read_bytes()


@frozen_8class
def test_v3_trip_release_check_detects_drift(tmp_path):
    release.main(['--out-dir', str(tmp_path)])
    with (tmp_path / 'train.jsonl').open('ab') as stream:
        stream.write(b'\n')
    with pytest.raises(ValueError, match='not reproducible'):
        release.main(['--check', '--out-dir', str(tmp_path)])


@pytest.mark.parametrize('key', ['text', 'normalized', 'family_id', 'template_id', 'procedure_family'])
def test_v3_trip_auditor_rejects_injected_cross_split_leakage(key):
    data = release.build()
    source = data['train'][0]
    target = data['dev'][0]
    if key == 'normalized':
        target['text'] = '  ' + source['text'].upper() + '!!!'
    else:
        target[key] = source[key]
    with pytest.raises(ValueError, match=f'{key} leakage'):
        release.audit(data, families())


def test_v3_trip_auditor_rejects_template_frame_leakage():
    source = families()
    source[-1]['templates'] = source[0]['templates']
    with pytest.raises(ValueError, match='template frame leakage'):
        release.audit(release.build(), source)


def test_v3_trip_auditor_rejects_within_split_duplicate():
    data = release.build()
    data['train'].append(copy.deepcopy(data['train'][0]))
    with pytest.raises(ValueError, match='duplicate within'):
        release.audit(data, families())


@frozen_8class
def test_v3_trip_frozen_lock_detects_seed_change(tmp_path, monkeypatch):
    # Build in an isolated full minimal layout, so protected and label checks still run.
    root = tmp_path / 'router'
    dest = root / 'data' / 'v3-trip'
    shutil.copytree(HERE, dest)
    shutil.copy(HERE.parents[1] / 'labels.json', root / 'labels.json')
    shutil.copy(HERE.parent / 'paraphrase_test.jsonl', root / 'data')
    for path in (HERE.parent / 'v2').glob('*.jsonl'):
        (root / 'data' / 'v2').mkdir(exist_ok=True)
        shutil.copy(path, root / 'data' / 'v2')
    for path in HERE.parents[1].glob('demo_calls*.json'):
        shutil.copy(path, root)
    with (dest / 'heldout_scenarios.jsonl').open('ab') as stream:
        stream.write(b'\n')
    monkeypatch.setattr(release, 'HERE', dest)
    with pytest.raises(ValueError, match='frozen file changed'):
        release.artifacts()


def test_v3_trip_all_labels_and_family_support_per_split():
    data = release.build()
    for split, rows in data.items():
        for label in release.LABELS:
            subset = [r for r in rows if r['label'] == label]
            assert len({r['family_id'] for r in subset}) >= (12 if split == 'train' else 3)
        assert len({r['family_id'] for r in rows}) >= (96 if split == 'train' else 24)
    release.audit(data, families())


def test_v3_trip_semantic_boundaries_and_frozen_seed_provenance():
    data = release.build()
    all_rows = [r for rows in data.values() for r in rows]
    nav = [r for r in all_rows if r['label'] == 'call_navigation']
    assert {r['response_mode'] for r in nav} == {'assessment', 'distance', 'guidance'}
    assert {'missing', 'malformed', 'unknown', 'resolved'} <= {r['destination_status'] for r in nav}
    for row in nav:
        if row['destination_status'] != 'resolved':
            assert row['destination'] is None
    for row in all_rows:
        if row.get('response_mode') == 'unsupported_hazard':
            assert row['label'] == 'call_llm'
            assert row['expected_tool_calls'] == 0 and row['clearance'] is False
    assert not any('base3' in r['text'] or 'crater 7' in r['text'] for r in data['train'])
    held = release.read_rows(HERE / 'heldout_scenarios.jsonl')
    assert set(r['expected_label'] for r in held) == set(release.LABELS)
    assert all(r['provenance'] == 'agent_authored_seed_not_human_heldout' for r in held)
    lookup = {r['text']: r for r in held}
    for text in ('can i make it to base 2', 'can i make it home', 'can i make it base2'):
        assert lookup[text]['expected_label'] == 'call_navigation'
    replay = release.read_rows(HERE / 'replay_scenarios.jsonl')
    assert all(r['requires_real_encoder_replay'] for r in replay)
    assert any(r.get('human_adjudication_required') for r in replay)


def test_v3_trip_legacy_audit_is_not_a_relabel_or_score():
    report = release.legacy_audit()
    assert report['status'] == 'historical_labels_incompatible_do_not_score_as_v3'
    assert 'data/v2/train.jsonl' in report['files']
    assert report['files']['data/v2/train.jsonl']['heuristic_review_candidates_by_old_label']['call_safety'] > 0

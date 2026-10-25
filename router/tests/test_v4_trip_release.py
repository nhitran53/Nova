"""v4-trip release contract: CPU/stdlib data audits, no model/network/training."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parents[1] / 'data' / 'v4-trip'
spec = importlib.util.spec_from_file_location('v4_trip_release', HERE / 'release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.fixture(scope='module')
def artifacts():
    return release.artifacts()


def test_v4_trip_determinism_and_checked_bytes(artifacts):
    output, manifest = artifacts
    assert release.build() == release.build()
    assert release.artifacts() == artifacts
    for name, content in output.items():
        assert (HERE / name).read_bytes() == content
    assert manifest['schema_version'] == 'v4-trip-1.2'
    assert manifest['dataset_name'] == 'v4-trip'
    assert manifest['version'] == 'v4.2'


def test_v4_trip_counts_balance_and_split_disjointness():
    groups = release.build()
    families = release.load_families()
    assert 120 <= len(families) <= 170
    assert all(len(f['templates']) >= 4 for f in families)
    assigned = {f['family_id']: f['split'] for f in families}
    for split, rows in groups.items():
        counts = release.Counter(r['label'] for r in rows)
        assert set(counts) == set(release.LABELS)
        assert 'AMBIGUOUS' not in counts
        assert max(counts.values()) <= len(rows) / 4
        assert max(counts.values()) <= 2 * min(counts.values())
        assert all(assigned[r['family_id']] == split for r in rows)
        assert max(release.Counter(r['family_id'] for r in rows).values()) <= 16
        assert len({r['construction'] for r in rows}) >= 8
        if split == 'train':
            assert set(counts.values()) == {192}
        else:
            assert len(rows) >= (200 if split in ('calibration', 'threshold') else 150)
    for challenge in ('contrastive_correction',):
        assert not any(r['construction'] == challenge for r in groups['train'])
        assert {r['label'] for r in groups['test'] if r['construction'] == challenge} == set(release.LABELS)
    release.audit(groups, families)


@pytest.mark.parametrize('key', ['text', 'normalized', 'family_id', 'template_id'])
def test_v4_trip_audit_rejects_injected_cross_split_leaks(key):
    groups = release.build()
    source, target = groups['train'][0], groups['dev'][0]
    if key == 'normalized':
        target['text'] = '  ' + source['text'].upper() + '!!!'
    else:
        target[key] = source[key]
    with pytest.raises(ValueError, match=f'{key} leakage'):
        release.audit(groups, release.load_families())


def test_v4_trip_frame_and_within_split_audits():
    groups = release.build()
    families = release.load_families()
    train = next(f for f in families if f['split'] == 'train')
    dev = next(f for f in families if f['split'] == 'dev')
    dev['templates'][0] = train['templates'][0].upper() + '!!!'
    with pytest.raises(ValueError, match='template frame leakage'):
        release.audit(groups, families)
    groups['train'].append(copy.deepcopy(groups['train'][0]))
    with pytest.raises(ValueError, match='duplicate within'):
        release.audit(groups, release.load_families())


def test_v4_trip_old_seed_exclusion_is_enforced():
    groups = release.build()
    seed = release.read_rows(HERE.parent / 'v3-trip' / 'heldout_scenarios.jsonl')[0]
    groups['train'][0]['text'] = seed['text'].upper() + '!!!'
    with pytest.raises(ValueError, match='old seed normalized overlap'):
        release.audit(groups, release.load_families())


def test_v4_trip_near_duplicate_review_and_hard_gate(artifacts, monkeypatch):
    assert artifacts[1]['near_duplicate_summary']['hard_pairs'] == 0
    a = {'text': 'Please read oxygen now at base 2', 'row_id': 'a', 'label': 'call_o2'}
    b = {'text': 'Please read O2 now at home', 'row_id': 'b', 'label': 'call_o2'}
    report = release.near_duplicates({'train': [a], 'test': [b]})
    assert report['flagged_pairs'] == report['hard_pairs'] == 1
    groups = release.build()
    source = next(r['text'] for r in groups['train'] if r['text'] == 'check radiation levels')
    groups['dev'][0]['text'] = ' '.join(reversed(source.split()))
    monkeypatch.setattr(release, 'build', lambda: groups)
    with pytest.raises(ValueError, match='hard near-duplicate leakage'):
        release.artifacts()


def test_v4_trip_frozen_hash_change_detected(tmp_path, monkeypatch):
    root = tmp_path / 'router'
    dest = root / 'data' / 'v4-trip'
    dest.mkdir(parents=True)
    for name in (*release.FROZEN, 'frozen_hashes.json', 'families.json', 'SUPERSEDED_V4_1.json'):
        shutil.copy(HERE / name, dest / name)
    shutil.copy(HERE.parents[1] / 'labels.json', root / 'labels.json')
    old = root / 'data' / 'v3-trip'
    old.mkdir()
    for name in ('heldout_scenarios.jsonl', 'regression.jsonl'):
        shutil.copy(HERE.parent / 'v3-trip' / name, old / name)
    with (dest / release.FROZEN[0]).open('ab') as stream:
        stream.write(b'\n')
    monkeypatch.setattr(release, 'HERE', dest)
    with pytest.raises(ValueError, match='frozen file changed'):
        release.artifacts()


def test_v4_trip_seen_references_and_non_scored_probes():
    references = release.read_rows(HERE / 'seen_during_development.jsonl')
    originals = [r for name in ('heldout_scenarios.jsonl', 'regression.jsonl')
                 for r in release.read_rows(HERE.parent / 'v3-trip' / name)]
    assert len(references) == len(originals) == 23
    assert [(r['text'], r['expected_label']) for r in references] == [(r['text'], r['expected_label']) for r in originals]
    assert all(r['evidence_role'] == 'seen_during_development_not_independent' for r in references)
    probes = release.read_rows(HERE / 'clarification_probe.jsonl')
    assert all(r['expected_disposition'] == 'CLARIFY' and not r['scored'] and 'label' not in r for r in probes)
    assert not {release.normalize(r['text']) for r in probes} & {release.normalize(r['text']) for rows in release.build().values() for r in rows}


def test_v4_trip_contrast_vocabulary_and_semantic_coverage():
    groups = release.build()
    all_rows = [r for rows in groups.values() for r in rows]
    for f in release.load_families():
        if f['label'] != 'call_navigation':
            continue
        nav = [r for r in all_rows if r['family_id'] == f['family_id']]
        for label in ('call_o2', 'call_safety', 'call_data', 'call_map', 'NO_TOOL', 'OOD'):
            twins = [r for r in all_rows if r['contrast_group'] == f['contrast_group'] and r['label'] == label]
            assert {r['place_surface'] for r in nav} == {r['place_surface'] for r in twins}
            assert any('resource_slot' in release.delexicalize(r['text']) for r in twins)
    train = groups['train']
    assert any(r['label'] == 'call_navigation' and 'make it' in r['text'] for r in train)
    assert any(r['label'] == 'call_navigation' and r['destination_status'] == 'missing' for r in train)
    assert any(r['label'] == 'call_data' and 'dose' in r['text'] for r in train)
    assert any('base2' in r['text'] for r in train)


def test_v4_trip_v4_2_short_commands_cancellations_and_scope_twins(artifacts):
    groups = release.build()
    train = groups['train']
    by_text = {r['text']: r['label'] for r in train}
    expected = {
        'never mind the trip': 'NO_TOOL', 'drop the route': 'NO_TOOL',
        'abort the plan': 'NO_TOOL', 'cancel that trip': 'NO_TOOL',
        'check radiation levels': 'call_data', 'check my heart rate': 'call_data',
        'read suit pressure': 'call_data', 'vitals temperature and bpm': 'call_data',
        'how much oxygen have i got left': 'call_o2',
        'how much air have i got left': 'call_o2',
        'remaining o2': 'call_o2', 'tank level': 'call_o2',
    }
    assert all(by_text[text] == label for text, label in expected.items())
    scope = [r for r in train if r['construction'] == 'contrastive_scope_selection']
    assert release.Counter(r['label'] for r in scope) == dict.fromkeys(release.LABELS, 16)
    for label in ('NO_TOOL', 'call_data', 'call_o2', 'call_navigation'):
        assert any(r['label'] == label and 'not asking' in r['text'] for r in scope)
        assert any(r['label'] == label and '"' in r['text'] for r in scope)
    # The broad semantics now cross train/test; only explicit repair surfaces
    # and their original whole families remain construction-held-out.
    repair = [r for r in groups['test'] if r['construction'] == 'contrastive_correction']
    assert len(repair) == 128
    markers = ('i meant ', 'correction:', 'actually ', 'let me rephrase:')
    assert all(r['text'].lower().startswith(markers) for r in repair)
    assert not any(r['text'].lower().startswith(markers) for r in train)
    altered = copy.deepcopy(groups)
    for row in altered['train']:
        if row['construction'] == 'contrastive_scope_selection':
            row['construction'] = 'contrastive_correction'
    with pytest.raises(ValueError, match='held-out construction outside test'):
        release.audit(altered, release.load_families())
    manifest = artifacts[1]
    assert manifest['heldout_constructions'] == ['contrastive_correction']
    assert 'NOT held out' in manifest['heldout_scope']
    superseded = json.loads((HERE / 'SUPERSEDED_V4_1.json').read_text())
    assert manifest['superseded_release'] == superseded
    assert superseded['status'] == 'SUPERSEDED by v4.2'
    assert superseded['schema_version'] == 'v4-trip-1.1'
    assert all(len(value) == 64 for value in superseded['sha256'].values())


def test_v4_trip_token_purity_diagnostic_and_gate(artifacts):
    rows = [{'text': 'incidental incidental explain', 'label': 'call_llm'} for _ in range(24)]
    report = release.token_purity(rows)
    assert report['non_allowlisted_pure_tokens_ge24'] == [
        {'token': 'incidental', 'rows': 24, 'label': 'call_llm', 'allowlisted': False}]
    assert not release.token_purity(rows[:23])['non_allowlisted_pure_tokens_ge24']
    rows.append({'text': 'incidental', 'label': 'OOD'})
    assert not release.token_purity(rows)['non_allowlisted_pure_tokens_ge24']
    assert not artifacts[1]['token_purity_summary']['non_allowlisted_pure_tokens_ge24']
    groups = release.build()
    targets = [r for r in groups['train'] if r['label'] == 'call_llm'][:24]
    for row in targets:
        row['text'] += ' incidental'
    with pytest.raises(ValueError, match='non-allowlisted pure training token'):
        release.audit(groups, release.load_families())
    # Former authoring cues must be absent or appear in more than one train label,
    # even though the release failure threshold is deliberately less restrictive.
    for token in ('this', 'crew', 'okay', 'action', 'request', 'going', 'by',
                  'around', 'walk', 'reaching', 'using', 'example'):
        labels = {r['label'] for r in release.build()['train']
                  if token in release.normalize(r['text']).split()}
        assert len(labels) != 1, token


def test_v4_trip_within_train_construction_crossing(artifacts):
    groups = release.build()
    contingency = release.construction_label_counts(groups['train'])
    assert contingency == artifacts[1]['train_construction_label_counts']
    assert len(contingency) == 16
    assert all(len(labels) >= 5 for labels in contingency.values())
    for construction in contingency:
        if len(contingency[construction]) != 5:
            continue
        altered = copy.deepcopy(groups)
        victim = next(iter(contingency[construction]))
        for row in altered['train']:
            if row['construction'] == construction and row['label'] == victim:
                row['construction'] = 'imperative_status'
        with pytest.raises(ValueError, match='training construction covers fewer'):
            release.audit(altered, release.load_families())
        break


def test_v4_trip_natural_nonrequests_and_mission_adjacent_ood():
    groups = release.build()
    for split, rows in groups.items():
        negatives = [r for r in rows if r['label'] == 'NO_TOOL']
        announcing = [r for r in negatives if any(phrase in release.normalize(r['text'])
                      for phrase in ('just a statement', 'no action', 'no request',
                                     'not a request', 'was a statement'))]
        assert len(announcing) / len(negatives) < .10, split
    new_train = [r for r in groups['train'] if '--mission-' in r['family_id']]
    ood = ' '.join(r['text'].lower() for r in new_train if r['label'] == 'OOD')
    for phrase in ('message my sister', 'call my dentist', 'play some music',
                   'timer for dinner', 'order replacement parts', 'book a flight',
                   'send an email', 'weather forecast for paris', 'football score',
                   'make it as a singer', 'top of the charts', 'replacement oxygen tank'):
        assert phrase in ood
    negatives = ' '.join(r['text'].lower() for r in new_train if r['label'] == 'NO_TOOL')
    for phrase in ('never mind', 'forget it', 'staying put', 'thanks, got it',
                   'we are at base 2 now', 'just thinking out loud'):
        assert phrase in negatives
    # Retain genuine navigation twins for natural non-request travel vocabulary.
    for construction in {r['construction'] for r in new_train if r['label'] == 'NO_TOOL'}:
        assert any(r['construction'] == construction and r['label'] == 'call_navigation'
                   for rows in groups.values() for r in rows)


def test_v4_trip_overlap_cli_counts_only_and_does_not_mutate(tmp_path):
    text = release.build()['train'][0]['text']
    external = tmp_path / 'external.jsonl'
    external.write_text(json.dumps({'text': text}) + '\n' + json.dumps({'text': text.upper() + '!!!'}) + '\n')
    before = external.read_bytes()
    result = subprocess.run([sys.executable, str(HERE / 'overlap_check.py'), str(external)], capture_output=True, text=True, check=True)
    counts = json.loads(result.stdout)
    assert counts['external_rows'] == 2
    assert counts['splits']['train']['exact_external_rows'] == 1
    assert counts['splits']['train']['normalized_external_rows'] == 2
    assert counts['splits']['train']['near_external_rows'] == 2
    assert text not in result.stdout and not result.stderr
    assert external.read_bytes() == before
    external.write_text('private invalid JSON payload')
    result = subprocess.run([sys.executable, str(HERE / 'overlap_check.py'), str(external)], capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout) == {'error_count': 1}
    assert not result.stderr and 'private' not in result.stdout


def test_v4_trip_release_check_detects_byte_drift(tmp_path, artifacts):
    for name, content in artifacts[0].items():
        (tmp_path / name).write_bytes(content)
    (tmp_path / 'train.jsonl').write_bytes(b'drift\n')
    with pytest.raises(ValueError, match='not reproducible: train.jsonl'):
        release.main(['--check', '--out-dir', str(tmp_path)])

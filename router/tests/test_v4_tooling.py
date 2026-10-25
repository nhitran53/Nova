"""V4 contract/role/privacy checks with released and synthetic inputs; never train."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from router import acceptance, calibrate, evaluate, preflight, report_gate, train
from router.core import load_labels
from router.training_tools import (V4_CAVEAT, calibration_input, check_checkpoint_contract,
                                   data_contract, is_v3)

HERE = Path(__file__).parents[1]


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


@pytest.fixture
def data(tmp_path, request):
    if getattr(request, 'param', None) == 'released':
        return HERE / 'data/v4-trip'
    root = tmp_path / 'arbitrary-candidate'
    root.mkdir()
    (root / 'manifest.json').write_text(json.dumps({'schema_version': 'v4-trip-1.1'}))
    for split in ('train', 'dev', 'calibration', 'threshold', 'test'):
        jsonl(root / f'{split}.jsonl', [{'text': split, 'label': 'call_navigation',
               'construction': split, 'family_id': split, 'row_id': split}])
    seen = [{'scenario_id': f'seen-{i}', 'text': 'seen reference',
             'expected_label': 'NO_TOOL'} for i in range(23)]
    jsonl(root / 'seen_during_development.jsonl', seen)
    (root / 'frozen_hashes.json').write_text(json.dumps({'seen_during_development.jsonl':
        hashlib.sha256((root / 'seen_during_development.jsonl').read_bytes()).hexdigest()}))
    return root


class FakeRouter:
    untrained = False
    meta = {'data_contract': 'v4-trip-1'}

    def predict(self, text):
        label = 'NO_TOOL' if text in ('seen reference', 'PRIVATE anchor', 'PRIVATE wrong twin') else 'call_navigation'
        confidence = .1 if text == 'PRIVATE sentinel low' else .99
        return {'label': label, 'confidence': confidence, 'topk': [{'label': label, 'p': confidence}]}


@pytest.fixture
def sealed(tmp_path):
    path = tmp_path / 'external.jsonl'
    specs = [
        ('anchor', 'PRIVATE anchor', 'NO_TOOL', 'none', None),
        ('twin', 'PRIVATE wrong twin', 'call_navigation', 'tool', 'anchor'),
        ('sentinel', 'PRIVATE sentinel low', 'AMBIGUOUS_CLARIFY', 'clarify', None),
        ('bad-sentinel', 'take me to base 2', 'AMBIGUOUS_CLARIFY', 'clarify', None),
        ('missing', 'can i make it to', 'call_navigation', 'missing_slot_clarify', None),
    ]
    jsonl(path, [dict(id=i, text=text, expected_label=label, expected_disposition=disposition,
                     construction='synthetic', contrast_of=anchor)
                 for i, text, label, disposition, anchor in specs])
    return path


def test_detection_and_calibration_roles(data):
    assert data_contract(data) == 'v4-trip-1'
    assert not is_v3(data)
    assert calibration_input(data) == data / 'calibration.jsonl'
    for split in ('train', 'dev', 'threshold', 'test'):
        with pytest.raises(ValueError, match='independent|overlaps'):
            calibration_input(data, data / f'{split}.jsonl')
    (data / 'calibration.jsonl').unlink()
    with pytest.raises(ValueError, match='missing'):
        calibration_input(data)


@pytest.mark.parametrize('key', ['family_id', 'template_id', 'procedure_family'])
def test_v4_calibration_overlap(data, tmp_path, key):
    row = {'text': 'independent surface', 'label': 'call_navigation', key: 'train'}
    if key != 'family_id':
        jsonl(data / 'train.jsonl', [{'text': 'train', 'label': 'call_navigation', key: 'train'}])
    external = tmp_path / 'cal.jsonl'
    jsonl(external, [row])
    with pytest.raises(ValueError, match=f'{key} overlaps train'):
        calibration_input(data, external)


def test_v4_requires_explicit_schema(tmp_path):
    root = tmp_path / 'v4-trip'
    root.mkdir()
    with pytest.raises(ValueError, match='schema_version'):
        data_contract(root)


@pytest.mark.parametrize('meta', [{}, {'data_contract': 'v3-trip-1'}, {'data_contract': 'legacy'}])
@pytest.mark.parametrize('data', ['synthetic', 'released'], indirect=True)
def test_v4_checkpoint_compatibility(data, meta):
    with pytest.raises(ValueError, match='mismatch'):
        check_checkpoint_contract(meta, data)
    check_checkpoint_contract(FakeRouter.meta, data)


def test_hash_covers_both_independent_splits(data):
    labels = load_labels(HERE / 'labels.json')
    for split in ('calibration', 'threshold'):
        before = train.data_hash(data, labels)
        with (data / f'{split}.jsonl').open('a') as stream:
            stream.write('\n')
        assert train.data_hash(data, labels) != before


@pytest.mark.parametrize('data', ['synthetic', 'released'], indirect=True)
def test_train_v4_requires_new_explicit_output_before_model_load(data, monkeypatch):
    loader = MagicMock()
    monkeypatch.setattr(train.AutoTokenizer, 'from_pretrained', loader)
    with pytest.raises(SystemExit):
        train.main(['--data-dir', str(data), '--device', 'cpu'])
    loader.assert_not_called()


@pytest.mark.parametrize('data', ['synthetic', 'released'], indirect=True)
def test_v4_preflight_dispatches_v4_release(data, tmp_path, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(preflight.subprocess, 'run', run)
    report = tmp_path / 'preflight.json'
    assert preflight.main(['--data-dir', str(data), '--report', str(report)]) == 0
    assert calls[0][1] == str(HERE / 'data/v4-trip/release.py')
    assert '--check' in calls[0]
    assert json.loads(report.read_text())['data_contract'] == 'v4-trip-1'


@pytest.mark.parametrize('data', ['synthetic', 'released'], indirect=True)
def test_v4_gate_requires_threshold_before_load(data, monkeypatch):
    loader = MagicMock()
    monkeypatch.setattr(report_gate.Router, 'load', loader)
    with pytest.raises(SystemExit):
        report_gate.main(['--data-dir', str(data)])
    loader.assert_not_called()


def test_sealed_metrics_are_separate_and_exclusion_is_headline_only(data, sealed):
    report = acceptance.run_v4_acceptance(FakeRouter(), data, sealed)
    assert report['classifier']['overall'] == {'n': 3, 'correct': 2, 'accuracy': 2/3}
    assert report['classifier']['by_label']['call_navigation']['n'] == 2
    assert report['classifier']['by_construction']['synthetic']['n'] == 3
    assert report['classifier']['contrast_twins_vs_anchor']['anchor_correct_twin_wrong'] == 1
    assert report['behavior']['n'] == 2 and report['behavior']['passed'] == 1
    assert report['missing_slot']['n'] == 1 and report['missing_slot']['passed'] == 1
    assert report['seen_during_development']['overall']['n'] == 23
    assert set(report['failures']) == {'twin', 'bad-sentinel'}
    assert report['caveat'] == V4_CAVEAT
    assert 'PRIVATE' not in json.dumps(report)
    excluded = acceptance.run_v4_acceptance(FakeRouter(), data, sealed, {'twin', 'bad-sentinel'})
    assert excluded['classifier']['overall']['accuracy'] == 1
    assert excluded['classifier']['contrast_twins_vs_anchor']['pairs'] == 0
    assert excluded['behavior']['pass_rate'] == 1
    assert excluded['excluded_count'] == 2
    assert excluded['seen_during_development'] == report['seen_during_development']
    assert not excluded['encoder_evidence'] and not excluded['passed']


def test_missing_slot_must_reach_actual_missing_slot_path(data, sealed):
    class WrongRouter(FakeRouter):
        def predict(self, text):
            return {'label': 'NO_TOOL', 'confidence': .99}
    report = acceptance.run_v4_acceptance(WrongRouter(), data, sealed)
    assert report['missing_slot']['failure_ids'] == ['missing']


@pytest.mark.parametrize('show_text', [False, True])
def test_cli_text_opt_in_and_limitations(data, sealed, tmp_path, monkeypatch, capsys, show_text):
    monkeypatch.setattr(acceptance.Router, 'load', lambda *a, **kw: FakeRouter())
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'router_meta.json').write_text('{}')
    excluded = tmp_path / 'ids.txt'
    excluded.write_text('twin\nbad-sentinel\n')
    out = tmp_path / 'report.json'
    args = ['--checkpoint', str(checkpoint), '--data-dir', str(data), '--sealed', str(sealed),
            '--exclude-ids', str(excluded), '--report', str(out)]
    assert acceptance.main(args + (['--show-text'] if show_text else [])) == 1
    output = capsys.readouterr().out
    assert ('PRIVATE' in output) == show_text
    assert ('PRIVATE' in out.read_text()) == show_text
    assert 'limitations' in output and 'seen_during_development' in output
    assert json.loads(out.read_text())['excluded_count'] == 2


def test_sealed_exception_does_not_echo_text(data, sealed, tmp_path, monkeypatch, capsys):
    class BrokenRouter(FakeRouter):
        def predict(self, text):
            raise ValueError(text)
    monkeypatch.setattr(acceptance.Router, 'load', lambda *a, **kw: BrokenRouter())
    (tmp_path / 'router_meta.json').write_text('{}')
    out = tmp_path / 'error.json'
    assert acceptance.main(['--checkpoint', str(tmp_path), '--data-dir', str(data),
                            '--sealed', str(sealed), '--report', str(out)]) == 1
    assert 'PRIVATE' not in capsys.readouterr().out + out.read_text()


def test_transfer_excludes_external_sealed_and_records_hash_only(tmp_path, capsys, monkeypatch):
    spec = importlib.util.spec_from_file_location('v4_transfer', HERE / 'scripts/v4-trip/transfer.py')
    transfer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transfer)
    root = tmp_path / 'repo'
    root.mkdir()
    (root / '.git').mkdir()
    # Synthetic git metadata, no access to live repository source or external evidence.
    monkeypatch.setattr(transfer.subprocess, 'check_output', lambda *a, **kw: 'fixture')
    data = root / 'router/data/v4-trip'
    data.mkdir(parents=True)
    (data / 'train.jsonl').write_text('{"text":"synthetic"}\n')
    (root / 'simulator_rules.py').write_text('X = 1\n')
    sealed = tmp_path / 'private.jsonl'
    sealed.write_text('PRIVATE SEALED PAYLOAD')
    result = transfer.build(root, tmp_path / 'handoff', sealed)
    assert result['sealed_sha256'] == hashlib.sha256(sealed.read_bytes()).hexdigest()
    manifest = json.loads((tmp_path / 'handoff/TRANSFER-MANIFEST.json').read_text())
    assert sorted(manifest['files']) == ['router/data/v4-trip/train.jsonl', 'simulator_rules.py']
    assert manifest['sealed_sha256'] == result['sealed_sha256']
    assert 'PRIVATE' not in capsys.readouterr().out
    assert str(sealed) not in json.dumps(manifest)
    assert transfer.verify_archive(result['archive'])['files'] == 2
    assert not (tmp_path / 'handoff/private.jsonl').exists()
    with pytest.raises(ValueError, match='external'):
        transfer.build(root, tmp_path / 'other', data / 'train.jsonl')


@pytest.mark.parametrize('data', ['synthetic', 'released'], indirect=True)
def test_v4_temperature_fit_selects_calibration_and_preserves_checkpoint(data, tmp_path, monkeypatch):
    import numpy as np
    router = SimpleNamespace(untrained=False, meta=FakeRouter.meta, labels=load_labels(HERE / 'labels.json'),
                             model=None, tokenizer=None, device='cpu', max_length=64)
    monkeypatch.setattr(calibrate.Router, 'load', lambda *a, **kw: router)
    inputs = []
    monkeypatch.setattr(calibrate, 'compute_logits', lambda model, tokenizer, texts, *a:
                        inputs.append(texts) or np.ones((len(texts), 8)))
    monkeypatch.setattr(calibrate, 'fit_temperature', lambda *a: 1.2)
    out = tmp_path / 'calibration-report.json'
    assert calibrate.main(['--checkpoint', 'unused', '--data-dir', str(data),
                           '--report', str(out)]) == 0
    report = json.loads(out.read_text())
    assert inputs == [[row['text'] for row in acceptance.rows(data / 'calibration.jsonl')]]
    assert report['checkpoint_unchanged'] and not report['legacy_dev_reuse']
    assert report['data_contract'] == 'v4-trip-1' and report['caveat'] == V4_CAVEAT
    with pytest.raises(SystemExit):
        calibrate.main(['--checkpoint', 'unused', '--data-dir', str(data), '--report', str(out)])


@pytest.mark.parametrize('data', ['synthetic', 'released'], indirect=True)
def test_mocked_v4_training_metadata_roles_and_hash(data, tmp_path, monkeypatch):
    """All model, loss and optimizer work is mocked: no training is performed."""
    import numpy as np
    import torch

    class Enc(dict):
        def to(self, device):
            return self

    tokenizer = MagicMock(side_effect=lambda *a, **kw: Enc())
    model = MagicMock()
    model.to.return_value = model
    model.return_value.loss.item.return_value = 1.0
    model.parameters.return_value = []
    model.state_dict.return_value = {'weight': torch.zeros(1)}
    monkeypatch.setattr(train.AutoTokenizer, 'from_pretrained', lambda *a, **kw: tokenizer)
    monkeypatch.setattr(train.AutoModelForSequenceClassification, 'from_pretrained', lambda *a, **kw: model)
    monkeypatch.setattr(torch.optim, 'AdamW', lambda *a, **kw: MagicMock())
    monkeypatch.setattr(train, 'get_linear_schedule_with_warmup', lambda *a: MagicMock())
    inputs = []
    monkeypatch.setattr(train, 'compute_logits', lambda model, tokenizer, texts, *a:
                        inputs.append(texts) or np.ones((len(texts), 8)))
    monkeypatch.setattr(train, 'fit_temperature', lambda *a: 1.2)
    labels = tmp_path / 'labels.json'
    labels.write_text((HERE / 'labels.json').read_text())
    out = tmp_path / 'candidate'
    assert train.main(['--data-dir', str(data), '--out-dir', str(out), '--labels', str(labels),
                       '--epochs', '1', '--device', 'cpu']) == 0
    meta = json.loads((out / 'router_meta.json').read_text())
    assert inputs == [[row['text'] for row in acceptance.rows(data / f'{split}.jsonl')]
                      for split in ('dev', 'calibration', 'test')]
    assert meta['data_contract'] == 'v4-trip-1' and meta['caveat'] == V4_CAVEAT
    assert meta['data_hash'] == train.data_hash(data, load_labels(labels))
    assert meta['calibration_role'] == 'independent' and meta['selection_split'] == 'dev'
    assert meta['dev_ece_role'] == 'diagnostic_only_not_temperature_fit'
    assert meta['promotion_status'] == 'candidate_only_not_default'


@pytest.mark.parametrize('data', ['synthetic', 'released'], indirect=True)
def test_v4_gate_selects_threshold_only(data, tmp_path, monkeypatch):
    import numpy as np
    labels = load_labels(HERE / 'labels.json')
    router = SimpleNamespace(untrained=False, temperature=1., meta=FakeRouter.meta)
    monkeypatch.setattr(report_gate.Router, 'load', lambda *a, **kw: router)
    probabilities = np.zeros((1, len(labels)))
    probabilities[0, labels.index('NO_TOOL')] = 1.
    inputs = []
    def split_probs(router, data_dir, split, labels, batch_size):
        inputs.append(split)
        return probabilities, np.array([labels.index('NO_TOOL')]), ['synthetic']
    monkeypatch.setattr(report_gate, 'split_probs', split_probs)
    assert report_gate.main(['--checkpoint', 'candidate', '--data-dir', str(data),
                             '--selection-split', 'threshold', '--report-dir', str(tmp_path)]) == 0
    report = json.loads((tmp_path / 'candidate_gate_threshold_test.json').read_text())
    assert inputs == ['threshold', 'test']
    assert report['data_contract'] == 'v4-trip-1' and report['caveat'] == V4_CAVEAT
    assert not report['legacy_dev_selection']


def test_v4_run_script_preserves_root_guard(tmp_path):
    import subprocess
    result = subprocess.run(['bash', str(HERE / 'scripts/v4-trip/run.sh'), 'eval'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2 and 'repository root' in result.stderr


def test_released_v42_contract_and_independent_calibration():
    root = HERE / 'data/v4-trip'
    assert json.loads((root / 'manifest.json').read_text())['schema_version'] == 'v4-trip-1.2'
    assert data_contract(root) == 'v4-trip-1'
    assert not is_v3(root)
    assert calibration_input(root) == root / 'calibration.jsonl'
    for split in ('train', 'dev', 'threshold', 'test'):
        with pytest.raises(ValueError, match='independent|overlaps'):
            calibration_input(root, root / f'{split}.jsonl')


def test_older_v4_schema_retains_checkpoint_contract(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'schema_version': 'v4-trip-1'}))
    assert data_contract(tmp_path) == 'v4-trip-1'
    check_checkpoint_contract(FakeRouter.meta, tmp_path)


@pytest.mark.parametrize('label,confidence,text,route,passed', [
    ('OOD', .8, 'a neutral statement', 'ood', False),
    ('NO_TOOL', .8, 'a neutral statement', 'llm', False),
    ('call_llm', .8, 'what is a rainbow', 'llm', False),
    ('call_navigation', .7, 'take me to base 2', 'clarify', True),
    ('call_navigation', .99, 'can i make it to', 'slot_question', True),
    ('OOD', .1, 'a neutral statement', 'low_confidence', True),
    ('call_o2', .99, 'check my oxygen', 'tool', False),
])
def test_sentinel_scores_actual_runtime_routes(data, tmp_path, label, confidence, text, route, passed):
    class PredictionRouter(FakeRouter):
        def predict(self, text):
            return {'label': label, 'confidence': confidence,
                    'topk': [{'label': label, 'p': confidence}]}

    router = PredictionRouter()
    ds = acceptance.FixtureDatasource(clock=lambda: acceptance.datetime(2026, 1, 1, tzinfo=acceptance.timezone.utc))
    handler = acceptance.TurnHandler(router, ds.snapshot, acceptance.default_registry(datasource=ds),
                                     generator=acceptance.FillerGenerator())
    runtime = handler.handle(text)
    assert runtime['route'] == route
    external = tmp_path / 'sentinel.jsonl'
    jsonl(external, [dict(id='sentinel', text=text, expected_label='AMBIGUOUS_CLARIFY',
                         expected_disposition='clarify', construction='synthetic', contrast_of=None)])
    report = acceptance.run_v4_acceptance(router, data, external)
    assert report['behavior']['passed'] == int(passed)
    assert report['behavior']['failure_ids'] == ([] if passed else ['sentinel'])
    assert report['behavior']['clarification_route_counts'] == {
        name: int(passed and route == name) for name in ('clarify', 'slot_question', 'low_confidence')}
    assert any('clarify, slot_question, or low_confidence' in note for note in report['limitations'])


@pytest.mark.parametrize('route,tool_calls', [('no_tool', 0), ('llm', 0), ('clarify', 1), ('slot_question', 1),
                                             ('low_confidence', 1)])
def test_sentinel_rejects_nonclarification_and_tool_execution(data, sealed, monkeypatch, route, tool_calls):
    # Fail closed for non-clarification outcomes and even a clarification with tool execution.
    monkeypatch.setattr(acceptance.TurnHandler, 'handle', lambda *_: {'route': route, 'tool_calls': tool_calls})
    report = acceptance.run_v4_acceptance(FakeRouter(), data, sealed)
    assert report['behavior']['passed'] == 0
    assert sum(report['behavior']['clarification_route_counts'].values()) == 0

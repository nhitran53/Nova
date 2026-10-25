"""v3 tooling separation, artifact isolation, CPU refusal and honest real-menu evidence."""
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from router import acceptance, evaluate, report_gate, smoke, train
from router.training_tools import calibration_input, checked_device, reserve_checkpoint, write_report

DATA = Path(__file__).parents[1] / 'data/v3-trip'


def make_splits(tmp_path):
    root = tmp_path / 'v3-trip'
    root.mkdir()
    for split in ('train', 'dev', 'calibration', 'threshold', 'test'):
        (root / f'{split}.jsonl').write_text(json.dumps({'text': split, 'label': 'a', 'family_id': split}) + '\n')
    return root


def test_v3_calibration_never_falls_back_or_accepts_selection_inputs(tmp_path):
    root = make_splits(tmp_path)
    assert calibration_input(root).name == 'calibration.jsonl'
    for split in ('train', 'dev', 'threshold', 'test'):
        with pytest.raises(ValueError, match='independent|overlaps'):
            calibration_input(root, root / f'{split}.jsonl')
    (root / 'calibration.jsonl').unlink()
    with pytest.raises(ValueError, match='missing'):
        calibration_input(root)


def test_external_calibration_rejects_family_overlap(tmp_path):
    root = make_splits(tmp_path)
    external = tmp_path / 'external.jsonl'
    external.write_text(json.dumps({'text': 'different words', 'label': 'a', 'family_id': 'dev'}))
    with pytest.raises(ValueError, match='family_id overlaps dev'):
        calibration_input(root, external)


def test_legacy_dev_reuse_remains_explicit(tmp_path):
    (tmp_path / 'dev.jsonl').write_text('{"text":"legacy", "label":"a"}\n')
    assert calibration_input(tmp_path).name == 'dev.jsonl'


def test_artifacts_never_overwrite_or_promote_reserved_names(tmp_path):
    for name in ('v1', 'v2', 'default', 'ModernBERT-base'):
        with pytest.raises(ValueError, match='protected'):
            reserve_checkpoint(tmp_path / name)
    out = reserve_checkpoint(tmp_path / 'candidate')
    with pytest.raises(FileExistsError):
        reserve_checkpoint(out)
    write_report(tmp_path / 'test.json', {'split': 'test'})
    with pytest.raises(FileExistsError):
        write_report(tmp_path / 'test.json', {'split': 'dev'})
    assert json.loads((tmp_path / 'test.json').read_text()) == {'split': 'test'}


def test_smoke_refuses_cpu_and_writes_non_candidate_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='fallback is forbidden'):
        checked_device('cpu', True)
    out = tmp_path / 'smoke.json'
    assert smoke.main(['--device', 'cpu', '--report', str(out)]) == 1
    report = json.loads(out.read_text())
    assert not report['success'] and not report['trained'] and not report['promotion_candidate']
    assert report['steps'] == []
    assert 'fallback is forbidden' in report['error']


def test_train_uses_dev_for_epoch_and_independent_calibration_for_T(tmp_path, monkeypatch):
    root = make_splits(tmp_path)
    labels = tmp_path / 'labels.json'
    labels.write_text('["a", "b"]')
    out = tmp_path / 'candidate'

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
    seen = []

    def logits(model, tokenizer, texts, device, length, batch):
        seen.append((texts, batch))
        if texts == ['dev']:
            return np.array([[3.0, 0.0]]) if len(seen) == 1 else np.array([[0.0, 3.0]])
        return np.array([[1.0, 0.0]])

    monkeypatch.setattr(train, 'compute_logits', logits)
    fits = []
    monkeypatch.setattr(train, 'fit_temperature', lambda logits, y: fits.append(logits.copy()) or 1.7)
    assert train.main(['--data-dir', str(root), '--labels', str(labels), '--out-dir', str(out),
                       '--epochs', '2', '--device', 'cpu']) == 0
    assert seen == [(['dev'], 4), (['dev'], 4), (['calibration'], 4), (['test'], 4)]
    assert fits[0].tolist() == [[1.0, 0.0]]
    meta = json.loads((out / 'router_meta.json').read_text())
    assert meta['selection_split'] == 'dev' and meta['calibration_role'] == 'independent'
    assert meta['best_epoch'] == 1
    assert meta['temperature'] == 1.7 and meta['promotion_status'] == 'candidate_only_not_default'
    assert model.return_value.loss.backward.called  # mocked, no CPU optimizer training


def test_report_gate_v3_requires_explicit_threshold_split_before_loading(monkeypatch):
    loader = MagicMock()
    monkeypatch.setattr(report_gate.Router, 'load', loader)
    with pytest.raises(SystemExit):
        report_gate.main(['--data-dir', str(DATA)])
    loader.assert_not_called()


def test_evaluate_split_reports_are_distinct_and_exclusive(tmp_path, monkeypatch):
    root = make_splits(tmp_path)
    labels = tmp_path / 'labels.json'
    labels.write_text('["a", "b"]')
    router = SimpleNamespace(untrained=False, model=None, tokenizer=None, device='cpu', max_length=64,
                             temperature=1.0, meta={})
    monkeypatch.setattr(evaluate.Router, 'load', lambda *a, **kw: router)
    monkeypatch.setattr(evaluate, 'compute_logits', lambda *a: np.array([[1., 0.]]))
    for split in ('dev', 'test'):
        assert evaluate.main(['--checkpoint', 'candidate', '--labels', str(labels), '--data-dir', str(root),
                              '--split', split, '--report-dir', str(tmp_path / 'reports')]) == 0
    assert sorted(p.name for p in (tmp_path / 'reports').iterdir()) == ['candidate_dev.json', 'candidate_test.json']
    with pytest.raises(FileExistsError):
        evaluate.main(['--checkpoint', 'candidate', '--labels', str(labels), '--data-dir', str(root),
                       '--split', 'dev', '--report-dir', str(tmp_path / 'reports')])


class NoMenuRouter:
    untrained = False

    def predict(self, text):
        return {'label': 'call_navigation', 'confidence': .99, 'topk': [{'label': 'call_navigation', 'p': .99}]}


def test_acceptance_missing_real_menus_is_unmet_never_injected_encoder_evidence():
    report = acceptance.run_acceptance(NoMenuRouter(), DATA)
    assert not report['passed'] and not report['encoder_evidence']
    assert any(item['scenario_id'] == 'menu-navigation' and item['code'] == 'unmet_menu_probe'
               for item in report['limitations'])
    replay = next(log for log in report['scenarios'] if log['scenario_id'] == 'menu-navigation')
    assert replay['actual_encoder_menu'] == []
    assert replay['evidence_kind'] == 'real_encoder_no_injection'
    assert report['legacy_inventory']['scored_as_v3'] is False
    assert any(log['scenario_id'] == 'telemetry-stale' and all(c['passed'] for c in log['checks'])
               for log in report['scenarios'])


def test_acceptance_cli_fails_closed_on_load_error(tmp_path, monkeypatch):
    out = tmp_path / 'acceptance.json'
    def fail(*a, **kw):
        raise RuntimeError('candidate load failed')
    monkeypatch.setattr(acceptance.Router, 'load', fail)
    assert acceptance.main(['--checkpoint', str(tmp_path), '--report', str(out)]) == 1
    report = json.loads(out.read_text())
    assert not report['passed'] and report['unmet_required_coverage']
    assert 'candidate load failed' in report['failures'][0]


def test_v3_gate_suggests_only_from_threshold_and_reports_current_separately(tmp_path, monkeypatch):
    from router.core import load_labels
    labels = load_labels(Path(__file__).parents[1] / 'labels.json')
    router = SimpleNamespace(untrained=False, temperature=1.0, meta={'data_contract': 'v3-trip-1'})
    monkeypatch.setattr(report_gate.Router, 'load', lambda *a, **kw: router)
    p = np.full((1, len(labels)), .09 / (len(labels) - 1))
    p[0, labels.index('call_navigation')] = .91
    y = np.array([labels.index('NO_TOOL')])
    seen = []
    def split_probs(router, data_dir, split, labels, batch_size):
        seen.append(split)
        return p, y, ['threshold-specific input']
    monkeypatch.setattr(report_gate, 'split_probs', split_probs)
    assert report_gate.main(['--checkpoint', 'candidate', '--data-dir', str(DATA),
                             '--selection-split', 'threshold', '--report-dir', str(tmp_path)]) == 0
    report = json.loads((tmp_path / 'candidate_gate_threshold_test.json').read_text())
    assert seen == ['threshold', 'test']  # never dev/calibration for operating-point selection
    assert report['offline_proposed_tau_hi'] == .92 and report['current_tau_hi'] == .9
    assert report['at_current_threshold']['test']['action_hcw'] == 1
    assert report['test_at_chosen']['action_hcw'] == 0
    assert report['legacy_dev_selection'] is False and 'dev_chosen_tau_hi' not in report


def test_acceptance_cli_returns_nonzero_for_unmet_real_encoder_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(acceptance.Router, 'load', lambda *a, **kw: NoMenuRouter())
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'router_meta.json').write_text('{"trained": true}')
    out = tmp_path / 'acceptance.json'
    assert acceptance.main(['--checkpoint', str(checkpoint), '--report', str(out)]) == 1
    report = json.loads(out.read_text())
    assert report['unmet_required_coverage'] and not report['passed']


@pytest.mark.parametrize('split', ['train', 'dev', 'threshold', 'test'])
@pytest.mark.parametrize('transformed', [
    '  WHAT S the walking distance from this spot to BASE2!!!  ',
    'what\ts\n the  walking distance from this spot to base2',
    'ＷＨＡＴ Ｓ ｔｈｅ ｗａｌｋｉｎｇ ｄｉｓｔａｎｃｅ ｆｒｏｍ ｔｈｉｓ ｓｐｏｔ ｔｏ ＢＡＳＥ２？',
])
def test_external_calibration_rejects_metadata_free_normalized_copy(tmp_path, split, transformed):
    root = make_splits(tmp_path)
    (root / f'{split}.jsonl').write_text(json.dumps({
        'text': "What's the walking distance from this spot to base2?", 'label': 'a'}))
    external = tmp_path / 'external.jsonl'
    external.write_text(json.dumps({'text': transformed, 'label': 'a'}))
    with pytest.raises(ValueError, match=f'calibration overlaps {split}'):
        calibration_input(root, external)


class MenuAnswerRouter:
    """Test double exercises the real TurnHandler only; never trained encoder evidence."""
    untrained = False

    def predict(self, text):
        return {'label': 'call_safety', 'confidence': .8, 'topk': [
            {'label': 'call_safety', 'p': .8}, {'label': 'call_navigation', 'p': .15},
            {'label': 'NO_TOOL', 'p': .05}]}


def test_acceptance_menu_answer_inventory_uses_independent_actual_handler_menus():
    report = acceptance.run_acceptance(MenuAnswerRouter(), DATA)
    assert set(report['required_menu_coverage']) == {
        'menu-answer-both', 'menu-answer-mixed', 'menu-answer-third',
        'menu-answer-unknown', 'menu-answer-unoffered', 'menu-answer-cancel'}
    assert all(report['required_menu_coverage'].values())
    assert not report['encoder_evidence'] and not report['passed']
    for log in report['scenarios']:
        if log['scenario_id'] in report['required_menu_coverage']:
            assert log['actual_encoder_menu'] == ['call_safety', 'call_navigation']
            assert len(log['turns']) == 2
            assert len(log['turns'][0]['router_calls']) == 1
            assert log['turns'][1]['router_calls'] == []
            assert all(check['passed'] for check in log['checks'])


def test_acceptance_menu_probe_absence_is_named_limitation_not_failure():
    report = acceptance.run_acceptance(NoMenuRouter(), DATA)
    assert not report['passed']
    assert not any(report['required_menu_coverage'].values())
    for name in acceptance.REQUIRED_MENU_COVERAGE:
        assert any(item['scenario_id'] == name and item['requires_human_adjudication']
                   for item in report['limitations'])
        assert not any(name in failure for failure in report['failures'])
        assert not any(name in unmet for unmet in report['unmet_required_coverage'])


def test_acceptance_omitted_menu_answer_case_blocks_coverage(monkeypatch):
    monkeypatch.setattr(acceptance, 'MENU_ANSWER_CASES', acceptance.MENU_ANSWER_CASES[:-1])
    report = acceptance.run_acceptance(MenuAnswerRouter(), DATA)
    assert not report['passed']
    assert report['required_menu_coverage']['menu-answer-cancel'] is False
    assert 'menu-answer-cancel: required menu-answer replay missing or unsuccessful' in report['unmet_required_coverage']


@pytest.mark.parametrize('fault', ['tool', 'message', 'llm', 'chitchat', 'reclassify',
                                  'clear_unresolved', 'retain_cancel', 'confirmation'])
def test_acceptance_unsuccessful_menu_answer_blocks_coverage(monkeypatch, fault):
    original = acceptance.TurnHandler.handle

    def broken(self, text):
        pending = self.pending_clarification
        result = original(self, text)
        if pending and text in ('both', 'cancel'):
            if fault == 'tool':
                result['tool_calls'] = 1
            elif fault == 'message':
                # FixtureDatasource is bound to the snapshot method, not a synthetic menu.
                self.alert_source.__self__.messages.append({
                    'sent_at': datetime(2026, 1, 1, tzinfo=timezone.utc), 'text': 'unauthorized'})
            elif fault == 'llm':
                result['llm_called'] = True
            elif fault == 'chitchat':
                result['route'] = 'chitchat'
            elif fault == 'reclassify':
                self.pipeline.router.predict(text)
            elif fault == 'clear_unresolved' and text == 'both':
                self.pending_clarification = None
            elif fault == 'retain_cancel' and text == 'cancel':
                self.pending_clarification = pending
            elif fault == 'confirmation':
                self.pending_confirmation = {'unauthorized': True}
        return result

    monkeypatch.setattr(acceptance.TurnHandler, 'handle', broken)
    report = acceptance.run_acceptance(MenuAnswerRouter(), DATA)
    name = 'menu-answer-cancel' if fault == 'retain_cancel' else 'menu-answer-both'
    assert not report['passed'] and not report['required_menu_coverage'][name]
    assert any(name in failure for failure in report['failures'])
    assert f'{name}: required menu-answer replay missing or unsuccessful' in report['unmet_required_coverage']


@pytest.mark.parametrize('split', ['heldout_scenarios', 'regression'])
def test_evaluate_frozen_expected_labels_require_explicit_field(tmp_path, monkeypatch, split):
    from router.core import load_labels
    labels_path = Path(__file__).parents[1] / 'labels.json'
    labels = load_labels(labels_path)
    path = DATA / f'{split}.jsonl'
    with pytest.raises(ValueError, match="missing label field 'label'"):
        evaluate.load_jsonl(path, labels)
    texts, targets = evaluate.load_jsonl(path, labels, 'expected_label')
    specs = acceptance.rows(path)
    assert texts == [row['text'] for row in specs]
    assert targets.tolist() == [labels.index(row['expected_label']) for row in specs]
    router = SimpleNamespace(untrained=False, model=None, tokenizer=None, device='cpu', max_length=64,
                             temperature=1.0, meta={'data_contract': 'v3-trip-1'})
    monkeypatch.setattr(evaluate.Router, 'load', lambda *a, **kw: router)
    logits = np.zeros((len(texts), len(labels)))
    logits[np.arange(len(texts)), targets] = 3.0
    monkeypatch.setattr(evaluate, 'compute_logits', lambda *a: logits)
    assert evaluate.main(['--checkpoint', 'candidate', '--labels', str(labels_path),
                          '--data-dir', str(DATA), '--split', split, '--label-field', 'expected_label',
                          '--report-dir', str(tmp_path)]) == 0
    report = json.loads((tmp_path / f'candidate_{split}.json').read_text())
    assert report['label_field'] == 'expected_label' and report['n'] == len(specs)
    assert report['accuracy'] == 1.0


def test_evaluate_label_field_never_falls_back_or_relabels(tmp_path):
    path = tmp_path / 'rows.jsonl'
    path.write_text(json.dumps({'text': 'example', 'label': 'a', 'expected_label': 'b'}))
    assert evaluate.load_jsonl(path, ['a', 'b'])[1].tolist() == [0]
    assert evaluate.load_jsonl(path, ['a', 'b'], 'expected_label')[1].tolist() == [1]
    with pytest.raises(ValueError, match="unknown label 'b' in 'expected_label'"):
        evaluate.load_jsonl(path, ['a'], 'expected_label')
    path.write_text(json.dumps({'text': 'example', 'label': 'a'}))
    with pytest.raises(ValueError, match="missing label field 'expected_label'"):
        evaluate.load_jsonl(path, ['a'], 'expected_label')


def test_acceptance_legacy_injection_hints_never_create_menu_or_encoder_evidence(monkeypatch):
    original = acceptance.rows
    def with_injection_hint(path):
        specs = original(path)
        if Path(path).name == 'replay_scenarios.jsonl':
            specs[0]['turns'][0].update(inject_menu_for_state_test=True,
                                      offered_choices=['call_safety', 'call_navigation'])
        return specs
    monkeypatch.setattr(acceptance, 'rows', with_injection_hint)
    report = acceptance.run_acceptance(NoMenuRouter(), DATA)
    log = next(item for item in report['scenarios'] if item['scenario_id'] == 'menu-navigation')
    assert log['actual_encoder_menu'] == [] and len(log['turns']) == 1
    assert log['turns'][0]['pending_clarification'] is None
    assert not report['encoder_evidence']
    assert any(item['scenario_id'] == 'menu-navigation' for item in report['limitations'])


class OtherMenuRouter(MenuAnswerRouter):
    def predict(self, text):
        return {'label': 'call_o2', 'confidence': .8, 'topk': [
            {'label': 'call_o2', 'p': .8}, {'label': 'call_map', 'p': .15},
            {'label': 'NO_TOOL', 'p': .05}]}


def test_acceptance_menu_answers_adapt_to_actual_options():
    report = acceptance.run_acceptance(OtherMenuRouter(), DATA)
    assert all(report['required_menu_coverage'].values())
    for log in report['scenarios']:
        if log['scenario_id'] in report['required_menu_coverage']:
            assert log['actual_encoder_menu'] == ['call_o2', 'call_map']
            assert all(check['passed'] for check in log['checks'])


def test_acceptance_menu_replays_use_probe_slots_and_actual_selection():
    report = acceptance.run_acceptance(MenuAnswerRouter(), DATA)
    for name in ('menu-navigation', 'menu-safety', 'critical-interrupt', 'cancel-menu',
                 'fresh-menu-execution'):
        log = next(item for item in report['scenarios'] if item['scenario_id'] == name)
        assert log['turns'][0]['text'] == acceptance.MENU_PROBE
        assert len(log['turns'][0]['router_calls']) == 1
        assert all(check['passed'] for check in log['checks']), log

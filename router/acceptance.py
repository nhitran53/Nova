"""Offline real-encoder TurnHandler acceptance/replay. Exit 1 on failures OR unmet required coverage.

Frozen replay injection hints are deliberately NOT applied: absent probe menus are limitations,
not encoder evidence. Uses only FixtureDatasource and deterministic FillerGenerator (no HTTP/TTS).
Legacy labels are inventoried separately, never relabelled or scored as v3 accuracy.
"""
import argparse
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .core import Router, load_labels
from .datasource import DataUnavailableError, FixtureDatasource
from .generation import FillerGenerator
from .preflight import verify_frozen
from .pipeline import Pipeline, load_config
from .places import resolve_destination
from .tools import default_registry
from .training_tools import (file_hash, write_report, data_contract, check_checkpoint_contract, V4_CAVEAT)
from .turn import TurnHandler, load_policy

HERE = Path(__file__).parent
TOOL_LABELS = {'call_navigation': 'navigation', 'call_safety': 'safety', 'call_o2': 'o2',
               'call_map': 'map', 'call_data': 'data'}


# Multi-intent probe, not a trip-feasibility label: requires human adjudication.
MENU_PROBE = 'Check my suit status and show directions home.'


# Keep the required inventory separate from executable cases: dropping a replay must fail closed.
REQUIRED_MENU_COVERAGE = ('menu-answer-both', 'menu-answer-mixed', 'menu-answer-third',
                          'menu-answer-unknown', 'menu-answer-unoffered', 'menu-answer-cancel')
MENU_ANSWER_CASES = (('menu-answer-both', 'both'),
                     ('menu-answer-mixed', 'navigation and safety'),
                     ('menu-answer-third', 'third'),
                     ('menu-answer-unknown', 'unknown'),
                     ('menu-answer-unoffered', 'map'),
                     ('menu-answer-cancel', 'cancel'))


class ObservedRouter:
    """Records only actual encoder calls; no score overrides or keyword remapping."""
    def __init__(self, router):
        self.router = router
        self.outputs = []

    def predict(self, text):
        output = self.router.predict(text)
        self.outputs.append({'text': text, 'output': output})
        return output


def fixture_messages(datasource):
    return [{**message, 'sent_at': message['sent_at'].isoformat()} for message in datasource.messages]


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run_acceptance(router, data_dir):
    root = Path(data_dir)
    verify_frozen(root)
    observed = ObservedRouter(router)
    report = {'kind': 'real_encoder_acceptance', 'encoder_evidence': isinstance(router, Router) and not router.untrained,
              'pipeline_config': load_config(), 'policy_sha256': file_hash(HERE / 'turn_policy.json'),
              'injection_policy': 'Replay inject_menu_for_state_test hints ignored; no injected state counts as encoder evidence',
              'generation': 'offline deterministic FillerGenerator only; no network, TTS or live generator',
              'scenarios': [], 'failures': [], 'unmet_required_coverage': [], 'limitations': [],
              'menu_probe': {'text': MENU_PROBE, 'human_adjudication_required': True},
              'frozen_sha256': {name: file_hash(root / name) for name in
                                ('heldout_scenarios.jsonl', 'regression.jsonl', 'replay_scenarios.jsonl')}}
    if not report['encoder_evidence']:
        report['unmet_required_coverage'].append('trained Router.load encoder required; test doubles are not encoder evidence')

    def menu_limitation(log, detail='no real encoder menu produced'):
        report['limitations'].append({'code': 'unmet_menu_probe',
            'scenario_id': log['scenario_id'], 'requires_human_adjudication': True, 'detail': detail})
        log['requires_human_adjudication'] = True

    def rig():
        now = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
        ds = FixtureDatasource(clock=lambda: now[0])
        handler = TurnHandler(observed, ds.snapshot, default_registry(datasource=ds), generator=FillerGenerator())
        return handler, ds, now

    def check(log, condition, expectation):
        log.setdefault('checks', []).append({'expectation': expectation, 'passed': bool(condition)})
        if not condition:
            report['failures'].append(f"{log['scenario_id']}: {expectation}")

    def turn(log, handler, ds, text, spec=None):
        before = len(observed.outputs)
        entry = {'text': text, 'expectations': spec or {}}
        log.setdefault('turns', []).append(entry)
        try:
            result = handler.handle(text)
            entry.update(result=result, router_calls=copy.deepcopy(observed.outputs[before:]),
                         pending_clarification=copy.deepcopy(handler.pending_clarification),
                         fixture_messages=fixture_messages(ds))
            check(log, result.get('tool_calls', 0) <= 1, 'at most one tool per turn')
            check(log, result.get('verifier', {}).get('ok', False) and not result.get('blocked_text'),
                  'spoken output passes actual numbers verifier without fallback')
            check(log, 'tool_error' not in result and 'llm_error' not in result, 'no tool/generator errors')
            if not (spec or {}).get('may_send_message', False):
                check(log, (result.get('tool_call') or {}).get('name') != 'send_message', 'no message without explicit consent')
            return result
        except Exception as exc:
            entry.update(error=f'{type(exc).__name__}: {exc}', router_calls=copy.deepcopy(observed.outputs[before:]))
            check(log, False, 'turn completes without exception')
            return {}

    # Independent classifier diagnostics are explicitly distinguished from calls made by TurnHandler.
    for source in ('heldout_scenarios.jsonl', 'regression.jsonl'):
        for spec in rows(root / source):
            log = {'scenario_id': spec['scenario_id'], 'source': source, 'specification': spec}
            report['scenarios'].append(log)
            handler, ds, _ = rig()
            prediction = router.predict(spec['text'])
            log['independent_encoder_probe'] = prediction
            check(log, prediction['label'] == spec['expected_label'], 'frozen expected classifier label')
            result = turn(log, handler, ds, spec['text'], spec)
            label = spec['expected_label']
            call = result.get('tool_call') or {}
            if label in TOOL_LABELS:
                unresolved = label in ('call_navigation', 'call_map') and not resolve_destination(spec['text'])['place']
                if unresolved:
                    check(log, result.get('tool_calls') == 0 and result.get('route') in ('clarify', 'slot_question'),
                          'unresolved/unknown destination clarifies without tool')
                else:
                    check(log, call.get('name') == TOOL_LABELS[label], 'correct tool executes at unchanged runtime gate')
                    if spec.get('place'):
                        check(log, call.get('args', {}).get('place') == spec['place'], 'exact destination argument')
                    if label == 'call_navigation' and spec.get('response_mode'):
                        check(log, (result.get('tool_result') or {}).get('data', {}).get('response_mode') == spec['response_mode'],
                              'specified navigation response mode')
            else:
                check(log, result.get('tool_calls') == 0, 'non-tool label does not execute a tool')
            if spec.get('clearance') is False:
                check(log, result.get('route') in ('clarify', 'risk_diverted', 'repair_offer', 'destination_safety', 'no_document'),
                      'hazard gets scope refusal/clarification, not clearance')

    for replay in rows(root / 'replay_scenarios.jsonl'):
        log = {'scenario_id': replay['scenario_id'], 'source': 'replay_scenarios.jsonl', 'specification': replay,
               'evidence_kind': 'real_encoder_no_injection'}
        report['scenarios'].append(log)
        handler, ds, _ = rig()
        for index, spec in enumerate(replay['turns']):
            if index and replay['fixture'] == 'critical_oxygen_on_followup':
                ds.telemetry[-1]['o2_percent'] = 10
            before = len(observed.outputs)
            text = spec['text']
            if index and log.get('actual_encoder_menu') and spec.get('expected_tool'):
                text = spec['expected_tool']
            result = turn(log, handler, ds, text, spec)
            if spec.get('menu_probe') or spec.get('inject_menu_for_state_test'):
                pending = handler.pending_clarification or {}
                actual = pending.get('choices', []) if pending.get('kind') == 'intent' else []
                log['actual_encoder_menu'] = actual
                if not actual:
                    menu_limitation(log)
                    break
                followup = replay['turns'][1]
                desired = next((label for label, tool in TOOL_LABELS.items()
                                if tool == followup.get('expected_tool')), None)
                if desired and desired not in actual:
                    menu_limitation(log, f'{desired} not in actual menu {actual}')
                    break
            if spec.get('expected_label'):
                outputs = observed.outputs[before:]
                check(log, bool(outputs) and outputs[0]['output']['label'] == spec['expected_label'], 'replay expected encoder label')
            if 'expected_tool_calls' in spec:
                check(log, result.get('tool_calls') == spec['expected_tool_calls'], 'replay exact tool count')
            if spec.get('expected_tool'):
                check(log, result.get('tool_call') == {'name': spec['expected_tool'], 'args': spec['expected_args']}, 'replay exact tool and arguments')
            if spec.get('response_mode'):
                check(log, (result.get('tool_result') or {}).get('data', {}).get('response_mode') == spec['response_mode'], 'replay response mode')
            if spec.get('trip_clearance') is False:
                check(log, 'not clearance' in result.get('spoken', ''), 'selected safety explicitly excludes trip clearance')
            if spec.get('expected_outcome') in ('clear_pending', 'critical_preemption_clears_pending'):
                check(log, handler.pending_clarification is None and handler.pending_confirmation is None, 'pending state cleared')
            if spec.get('expected_outcome') == 'clear_pending':
                check(log, result.get('tool_calls') == 0 and not ds.messages
                      and result.get('llm_called') is False and result.get('route') == 'clarify'
                      and len(observed.outputs) == before,
                      'cancellation executes no tools/messages, LLM/chitchat or encoder reclassification')
            if spec.get('expected_outcome') == 'critical_preemption_clears_pending':
                check(log, bool(result.get('alert_line'))
                      and result.get('lines', [None])[0] == result['alert_line']
                      and result.get('spoken', '').startswith(result['alert_line'])
                      and result.get('alert_verifier', {}).get('ok') is True
                      and result.get('llm_called') is False and not handler.generator.calls
                      and result.get('interim') is None
                      and not ds.messages and (result.get('tool_call') or {}).get('name') in (None, *TOOL_LABELS.values())
                      and handler.pending_confirmation is None and handler.pending_clarification is None
                      and result.get('pending_confirmation') is None,
                      'critical alert first and verified; no generation, interim speech, messages or offers; read-only routing allowed')
            if spec.get('clearance') is False:
                check(log, result.get('tool_calls') == 0 and result.get('route') in ('clarify', 'risk_diverted', 'repair_offer', 'destination_safety', 'no_document'),
                      'mixed hazard challenge refuses clearance without tool')

    menu_coverage = {name: False for name in REQUIRED_MENU_COVERAGE}
    report['required_menu_coverage'] = menu_coverage
    for scenario_id, answer in MENU_ANSWER_CASES:
        log = {'scenario_id': scenario_id, 'source': 'tooling_extension',
               'evidence_kind': 'real_encoder_no_injection'}
        report['scenarios'].append(log)
        handler, ds, _ = rig()
        turn(log, handler, ds, MENU_PROBE)
        pending = copy.deepcopy(handler.pending_clarification)
        actual = (pending or {}).get('choices', []) if (pending or {}).get('kind') == 'intent' else []
        log['actual_encoder_menu'] = actual
        if not actual:
            menu_limitation(log)
            continue
        if scenario_id == 'menu-answer-mixed':
            answer = ' and '.join(TOOL_LABELS.get(choice, choice) for choice in actual)
        elif scenario_id == 'menu-answer-third':
            answer = str(len(actual) + 1)
        elif scenario_id == 'menu-answer-unoffered':
            answer = next((tool for label, tool in TOOL_LABELS.items() if label not in actual), 'unoffered tool')
        check(log, not ds.messages and handler.registry.calls_this_turn == 0,
              'menu creation executes no tools or messages')
        before = len(observed.outputs)
        result = turn(log, handler, ds, answer)
        check(log, result.get('tool_calls') == 0 and result.get('tool_call') is None
              and handler.registry.calls_this_turn == 0 and not ds.messages,
              'menu answer executes zero tools and messages')
        check(log, result.get('llm_called') is False and result.get('route') == 'clarify'
              and len(observed.outputs) == before and not result.get('clarification_selection'),
              'menu answer stays clarification without LLM, chitchat or encoder reclassification')
        check(log, handler.pending_confirmation is None and
              (handler.pending_clarification is None if answer == 'cancel'
               else handler.pending_clarification == pending),
              'cancel clears pending; unresolved answer retains original bounded menu without confirmation')
        menu_coverage[scenario_id] = all(item['passed'] for item in log['checks'])
    for name, covered in menu_coverage.items():
        if not covered and not any(item['scenario_id'] == name for item in report['limitations']):
            report['unmet_required_coverage'].append(f'{name}: required menu-answer replay missing or unsuccessful')

    log = {'scenario_id': 'malformed-slot', 'source': 'tooling_extension'}
    report['scenarios'].append(log)
    handler, ds, _ = rig()
    malformed = turn(log, handler, ds, 'can i make it to')
    check(log, malformed.get('tool_calls') == 0 and malformed.get('route') in ('clarify', 'slot_question'),
          'truncated destination requires clarification without a tool')

    # Extra adapters transparently extend frozen scenarios to cover resource/freshness/consent requirements.
    for condition in ('missing', 'stale'):
        log = {'scenario_id': f'telemetry-{condition}', 'source': 'tooling_extension', 'turns': []}
        report['scenarios'].append(log)
        handler, ds, now = rig()
        if condition == 'missing':
            ds.telemetry.clear()
        else:
            now[0] += timedelta(seconds=ds.policy['max_age_s'] + 1)
        before = len(observed.outputs)
        try:
            result = handler.handle('can i make it to base 2')
            log['turns'].append({'result': result})
            check(log, False, 'unavailable telemetry raises before routing')
        except DataUnavailableError as exc:
            log['turns'].append({'expected_error': str(exc), 'router_calls': observed.outputs[before:], 'fixture_messages': fixture_messages(ds)})
            check(log, len(observed.outputs) == before and not ds.messages and handler.registry.calls_this_turn == 0,
                  'unavailable telemetry prevents routing, tools and messages')

    for answer in ('no', 'yes'):
        log = {'scenario_id': f'shortage-consent-{answer}', 'source': 'tooling_extension'}
        report['scenarios'].append(log)
        handler, ds, _ = rig()
        ds.mission['o2_burn_pct_per_min'] = 10.0
        ds.telemetry[-1]['o2_percent'] = 25.0  # noncritical, but insufficient trip minutes
        first = turn(log, handler, ds, 'can i make it to base 2')
        feasibility = (first.get('tool_result') or {}).get('data', {}).get('feasibility', {})
        check(log, feasibility.get('outcome') in ('cannot_reach', 'cannot_return', 'tight'), 'shortage is explicitly reported')
        check(log, bool(handler.pending_confirmation) and not ds.messages, 'shortage offers contact but does not send')
        result = turn(log, handler, ds, answer, {'may_send_message': answer == 'yes'})
        check(log, len(ds.messages) == (1 if answer == 'yes' else 0), 'message requires explicit yes; no declines')
        check(log, result.get('route') == ('confirmed_yes' if answer == 'yes' else 'confirmed_no'), 'confirmation handled as consent not new intent')

    # Fresh execution after an actual encoder-proposed menu: no synthetic menu state.
    log = {'scenario_id': 'fresh-menu-execution', 'source': 'tooling_extension'}
    report['scenarios'].append(log)
    handler, ds, _ = rig()
    turn(log, handler, ds, MENU_PROBE)
    pending = handler.pending_clarification or {}
    if pending.get('kind') != 'intent' or 'call_navigation' not in pending.get('choices', []):
        menu_limitation(log, 'no real encoder-proposed navigation menu')
    else:
        ds.mission['o2_burn_pct_per_min'] = 10.0
        ds.telemetry[-1]['o2_percent'] = 25.0
        ds.set_position(3000, 3000)
        expected_distance = ds.place('homebase')['distance_m']
        result = turn(log, handler, ds, 'navigation')
        data = (result.get('tool_result') or {}).get('data', {})
        check(log, data.get('distance_m') == expected_distance and data.get('feasibility', {}).get('outcome') == 'cannot_reach',
              'menu selection uses fresh geometry and oxygen, not stored snapshot')
        check(log, not ds.messages, 'menu selection is not message consent')

    # No legacy accuracy: retain the authored incompatibility inventory verbatim.
    report['legacy_inventory'] = {'source': str(root / 'legacy_audit.json'),
                                  'sha256': file_hash(root / 'legacy_audit.json'),
                                  'inventory': json.loads((root / 'legacy_audit.json').read_text()),
                                  'scored_as_v3': False}
    report['passed'] = not report['failures'] and not report['unmet_required_coverage']
    return report



def run_v4_acceptance(router, data_dir, sealed, exclude_ids=(), show_text=False):
    """External evidence only; never retain utterances or runtime outputs unless explicitly requested."""
    root = Path(data_dir)
    check_checkpoint_contract(router.meta, root)
    verify_frozen(root)
    config = load_config()
    labels = load_labels(HERE / 'labels.json')
    if Path(sealed).resolve().is_relative_to(HERE.parent.resolve()):
        raise ValueError('sealed input must be external to repository')
    specs = rows(sealed)
    required = {'id', 'text', 'expected_label', 'expected_disposition', 'construction', 'contrast_of'}
    if not specs or any(not required <= row.keys() for row in specs):
        raise ValueError('invalid sealed row schema')
    ids = [row['id'] for row in specs]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('sealed ids must be unique nonempty strings')
    if any(row['expected_label'] not in labels + ['AMBIGUOUS_CLARIFY'] for row in specs):
        raise ValueError('sealed expected label is not in the approved taxonomy')
    if any(row['contrast_of'] and row['contrast_of'] not in ids for row in specs):
        raise ValueError('sealed contrast anchor missing')
    excluded = set(exclude_ids)
    if excluded - set(ids):
        raise ValueError('excluded id missing from sealed inventory')

    def metric(values):
        return {'n': len(values), 'correct': sum(values),
                'accuracy': sum(values) / len(values) if values else None}

    def classifier(group, id_key):
        outcomes = {}
        by_label = {label: [] for label in labels}
        by_construction = {}
        failed = []
        for row in group:
            if row['expected_label'] not in labels:
                continue
            ok = router.predict(row['text'])['label'] == row['expected_label']
            outcomes[row[id_key]] = ok
            by_label[row['expected_label']].append(ok)
            by_construction.setdefault(row.get('construction', 'unspecified'), []).append(ok)
            if not ok:
                failed.append(row[id_key])
        pairs = [(row, outcomes[row[id_key]], outcomes[row['contrast_of']]) for row in group
                 if row.get('contrast_of') in outcomes and row[id_key] in outcomes]
        return {'overall': metric(list(outcomes.values())),
                'by_label': {k: metric(v) for k, v in by_label.items()},
                'by_construction': {k: metric(v) for k, v in by_construction.items()},
                'contrast_twins_vs_anchor': {'pairs': len(pairs),
                    'both_correct': sum(twin and anchor for _, twin, anchor in pairs),
                    'twin_correct_anchor_wrong': sum(twin and not anchor for _, twin, anchor in pairs),
                    'anchor_correct_twin_wrong': sum(anchor and not twin for _, twin, anchor in pairs),
                    'both_wrong': sum(not twin and not anchor for _, twin, anchor in pairs)},
                'failure_ids': failed}

    selected = [row for row in specs if row['id'] not in excluded]
    report = {'kind': 'v4_external_sealed_acceptance', 'data_contract': 'v4-trip-1',
              'caveat': V4_CAVEAT, 'sealed_sha256': file_hash(sealed),
              'pipeline_config': config, 'excluded_count': len(excluded),
              'encoder_evidence': isinstance(router, Router) and not router.untrained,
              'classifier': classifier(selected, 'id'), 'failures': [],
              'unmet_required_coverage': [],
              'limitations': [V4_CAVEAT, 'External provenance and semantic independence require human review.',
                  'Behavior is offline fixture runtime evidence, not operational safety evidence.',
                  'Sentinels count only zero-tool clarify, slot_question, or low_confidence (repeat request) routes as clarification.',
                  'Seen development seeds/controls are not independent acceptance evidence.',
                  'No v3 replay/menu coverage is claimed; thresholds and policy are unchanged.']}
    report['failures'].extend(report['classifier']['failure_ids'])
    behavior = []
    missing = []
    behavior_failures, missing_failures = [], []
    clarification_routes = dict.fromkeys(('clarify', 'slot_question', 'low_confidence'), 0)
    for row in selected:
        sentinel = row['expected_label'] == 'AMBIGUOUS_CLARIFY'
        missing_slot = row['expected_disposition'] == 'missing_slot_clarify'
        if not sentinel and not missing_slot:
            continue
        # Fresh runtime per row; no state injection or classifier-label substitution.
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        ds = FixtureDatasource(clock=lambda: now)
        handler = TurnHandler(router, ds.snapshot, default_registry(datasource=ds), generator=FillerGenerator())
        pipeline = Pipeline(router, config=config, llm_fn=lambda *_: '')
        routed = pipeline.handle(row['text'])
        runtime = handler.handle(row['text'])
        no_tool = runtime.get('tool_calls') == 0 and not ds.messages
        if sentinel:
            route = runtime.get('route')
            ok = no_tool and route in clarification_routes
            if ok:
                clarification_routes[route] += 1
            behavior.append(ok)
            if not ok:
                behavior_failures.append(row['id'])
        if missing_slot:
            ok = (no_tool and routed.get('reason') == 'missing_slot' and
                  runtime.get('route') == 'slot_question' and
                  (handler.pending_clarification or {}).get('kind') == 'slot')
            missing.append(ok)
            if not ok:
                missing_failures.append(row['id'])

    def behavior_metric(values):
        return {'n': len(values), 'passed': sum(values),
                'pass_rate': sum(values) / len(values) if values else None}

    report['behavior'] = {**behavior_metric(behavior), 'failure_ids': behavior_failures,
                          'clarification_route_counts': clarification_routes,
                          'role': 'clarification sentinel behavior, not classifier accuracy'}
    report['missing_slot'] = {**behavior_metric(missing), 'failure_ids': missing_failures,
                              'role': 'pipeline/runtime missing-slot path, not sentinel accuracy'}
    seen = rows(root / 'seen_during_development.jsonl')
    report['seen_during_development'] = classifier(seen, 'scenario_id')
    report['seen_during_development']['role'] = '15 old seeds / 8 controls; not independent evidence'
    report['failures'] = sorted(set(report['failures'] + behavior_failures + missing_failures))
    for name, count in [('classifier', report['classifier']['overall']['n']),
                        ('behavior', len(behavior)), ('missing_slot', len(missing))]:
        if not count:
            report['unmet_required_coverage'].append(f'no scored {name} rows')
    if not report['encoder_evidence']:
        report['unmet_required_coverage'].append('trained Router.load encoder required')
    report['passed'] = not report['failures'] and not report['unmet_required_coverage']
    if show_text:
        report['sealed_texts'] = [{'id': row['id'], 'text': row['text']} for row in specs]
    return report


def read_exclude_ids(path):
    """One row id per line, or a JSON list of ids."""
    content = Path(path).read_text()
    values = json.loads(content) if content.lstrip().startswith('[') else content.splitlines()
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError('exclude-ids must contain string ids')
    return {value.strip() for value in values if value.strip()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True, help='Local trained candidate loaded via Router.load; never promoted')
    ap.add_argument('--data-dir', default=str(HERE / 'data/v3-trip'))
    ap.add_argument('--labels', default=str(HERE / 'labels.json'))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--report', required=True, help='New JSON evidence path; existing report refused')
    ap.add_argument('--sealed', help='External v4 sealed JSONL; never copied into reports by default')
    ap.add_argument('--exclude-ids', help='File of overlapping ids to exclude from headline metrics')
    ap.add_argument('--show-text', action='store_true', help='Explicitly include and print sealed texts (default off)')
    args = ap.parse_args(argv)
    v4 = data_contract(args.data_dir) == 'v4-trip-1'
    if v4 and not args.sealed:
        ap.error('v4 acceptance requires --sealed PATH')
    if not v4 and (args.sealed or args.exclude_ids or args.show_text):
        ap.error('sealed options require v4 data contract')
    if Path(args.report).exists():
        ap.error('report already exists')
    try:
        if not Path(args.checkpoint).is_dir():
            raise ValueError('acceptance requires a local checkpoint, not a network model ID')
        router = Router.load(args.checkpoint, args.labels, device=args.device)
        if router.untrained:
            raise ValueError('untrained/smoke checkpoint cannot supply acceptance evidence')
        if v4:
            excluded = read_exclude_ids(args.exclude_ids) if args.exclude_ids else ()
            report = run_v4_acceptance(router, args.data_dir, args.sealed, excluded, args.show_text)
        else:
            report = run_acceptance(router, args.data_dir)
        report['checkpoint'] = str(Path(args.checkpoint).resolve())
        report['checkpoint_meta_sha256'] = file_hash(Path(args.checkpoint) / 'router_meta.json')
    except Exception as exc:
        report = {'kind': 'real_encoder_acceptance', 'passed': False,
                  'failures': [f'{type(exc).__name__}: sealed acceptance incomplete' if v4 else f'{type(exc).__name__}: {exc}'],
                  'unmet_required_coverage': ['acceptance run incomplete'],
                  'limitations': ['Incomplete run is not acceptance evidence.']}
    write_report(args.report, report)
    keys = ('passed', 'failures', 'unmet_required_coverage', 'limitations')
    if v4:
        keys += ('classifier', 'behavior', 'missing_slot', 'seen_during_development', 'excluded_count', 'sealed_texts')
    print(json.dumps({key: report[key] for key in keys if key in report}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

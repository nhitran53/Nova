"""Offered-choice authority is bounded to BERT's current menu, never a fresh keyword router."""
import pytest

from router.tools import SimState, default_registry
from router.turn import TurnHandler


class MenuRouter:
    def __init__(self):
        self.seen = []

    def predict(self, text):
        self.seen.append(text)
        return {'label': 'call_safety', 'confidence': .7, 'topk': [
            {'label': 'call_safety', 'p': .7}, {'label': 'call_navigation', 'p': .15},
            {'label': 'NO_TOOL', 'p': .05}]}


def rig(text='can i make it home', clock=lambda: 0):
    sim = SimState()
    router = MenuRouter()
    h = TurnHandler(router, sim.snapshot, default_registry(sim), clock=clock)
    first = h.handle(text)
    assert first['route'] == 'clarify' and first['tool_calls'] == 0
    return h, sim, router


@pytest.mark.parametrize('answer,tool', [('navigation', 'navigation'), ('second', 'navigation'),
                                        ('a safety check', 'safety'), ('first', 'safety')])
def test_selection_calls_exact_offered_tool_retains_context_and_never_llm(answer, tool):
    h, sim, router = rig()
    r = h.handle(answer)
    assert r['tool_call']['name'] == tool and r['tool_calls'] == 1 and not r['llm_called']
    assert len(router.seen) == 1 and r['verifier']['ok']
    assert r['clarification_selection']['offered'] == ['call_safety', 'call_navigation']
    if tool == 'navigation':
        assert r['tool_call']['args'] == {'place': 'homebase'}
        assert r['tool_result']['data']['response_mode'] == 'assessment'
    else:
        assert 'not clearance' in r['spoken']


@pytest.mark.parametrize('answer', ['cancel', 'unknown', 'both', 'yes', 'map', 'navigation and safety', 'third'])
def test_unresolved_choices_are_disregards_not_chitchat_and_bound_is_not_extended(answer):
    h, _, router = rig()
    for _ in range(2):
        r = h.handle(answer)
        assert r['tool_calls'] == 0 and not r['llm_called'] and r['route'] == 'clarify'
        if answer == 'cancel':
            break
    assert h.pending_clarification is None
    assert len(router.seen) == 1


def test_wall_expiry_and_critical_preemption_clear_state():
    now = [0]
    h, _, _ = rig(clock=lambda: now[0])
    now[0] = 121
    assert h.handle('navigation')['tool_calls'] == 0
    assert h.pending_clarification is None
    h, sim, _ = rig()
    sim.values["o2_pct"] = 10
    assert h.handle('navigation')['route'] == 'alert'
    assert h.pending_clarification is None and h.pending_confirmation is None


def test_fresh_oxygen_and_unknown_destination_no_stale_place():
    h, sim, _ = rig('can i make it to base 2')
    sim.values["minutes_remaining"] = 1
    r = h.handle('navigation')
    assert r['tool_call']['args'] == {'place': 'base_2'}
    assert r['tool_result']['data']['feasibility']['outcome'] == 'cannot_reach'
    assert r['pending_confirmation'] is not None  # offer, not consent
    assert r['tool_call']['name'] != 'send_message'
    h, _, _ = rig('can i make it to base3')
    r = h.handle('navigation')
    assert r['tool_calls'] == 0 and r['route'] == 'slot_question'
    assert h.handle('base3')['tool_calls'] == 0
    r = h.handle('tower')
    assert r['tool_call']['args'] == {'place': 'comms_tower'}


def test_missing_oxygen_is_explicit_without_a_contact_offer():
    h, _, _ = rig()
    h.alert_source = lambda: {'bpm': 72}
    r = h.handle('navigation')
    assert r['tool_result']['data']['feasibility']['outcome'] == 'unavailable'
    assert 'oxygen feasibility is unavailable' in r['spoken'] and r['pending_confirmation'] is None


@pytest.mark.parametrize('answer', ['2', 'option 2', 'call_navigation'])
def test_numeric_and_identity_selection(answer):
    h, _, _ = rig()
    assert h.handle(answer)['tool_call']['name'] == 'navigation'


def test_selection_refreshes_geometry_and_is_not_message_confirmation():
    h, sim, router = rig('can i make it to base 2')
    # Registry is the execution-time geometry authority; state contains no distance/telemetry.
    schema, _ = h.registry._tools['navigation']
    h.registry._tools['navigation'] = (schema, lambda place: {'ok': True, 'data': {
        'place': place, 'place_name': 'base 2', 'distance_m': 1800, 'eta_min': 30}})
    sim.values['minutes_remaining'] = 1
    r = h.handle('navigation')
    assert r['tool_result']['data']['distance_m'] == 1800 and r['tool_calls'] == 1
    assert h.pending_clarification is None and h.pending_confirmation is not None
    unresolved = h.handle('second')
    assert unresolved['tool_calls'] == 0 and not unresolved['llm_called']  # not message consent or chitchat


def test_clear_new_bert_authorized_intent_interrupts_and_drops_original_place():
    h, _, router = rig()
    router.predict = lambda _: {'label': 'call_o2', 'confidence': .99}
    r = h.handle('tell me my oxygen status')
    assert r['tool_call'] == {'name': 'o2', 'args': {}}
    assert h.pending_clarification is None


def test_distance_context_and_two_destinations():
    h, _, _ = rig('how far is base 2')
    r = h.handle('navigation')
    assert r['tool_result']['data']['response_mode'] == 'distance'
    assert r['spoken'].startswith('base 2 is 17 minutes away.')
    h, _, _ = rig('can i make it to the rover or home')
    assert h.handle('navigation')['tool_calls'] == 0
    assert h.handle('second')['tool_call']['args'] == {'place': 'homebase'}


def test_numbered_place_menu_is_verified_and_selection_cannot_pick_unoffered_place():
    h, _, _ = rig('can i make it to base 2 or home')
    r = h.handle('navigation')
    assert r['route'] == 'slot_question' and r['verifier']['ok']
    assert h.handle('tower')['tool_calls'] == 0  # ambiguous menu is not a free replacement slot
    r = h.handle('first')
    assert r['tool_call']['args'] == {'place': 'base_2'}


def test_mixed_long_selection_never_becomes_a_confident_fresh_tool():
    h, _, router = rig()
    router.predict = lambda _: {'label': 'call_navigation', 'confidence': .99}
    for answer in ['i want both navigation and safety', 'can i have both please']:
        r = h.handle(answer)
        assert r['tool_calls'] == 0 and not r['llm_called']


def test_short_clear_navigation_interrupt_is_reclassified_not_menu_authorized():
    h, _, router = rig()
    router.predict = lambda _: {'label': 'call_navigation', 'confidence': .99}
    r = h.handle('go to tower')
    assert r['tool_call']['args'] == {'place': 'comms_tower'}
    assert r['pipeline']['gate'] == 'tau_hi' and h.pending_clarification is None


@pytest.mark.parametrize('text,mode', [
    ('Would the tank cover getting to base b and returning?', 'assessment'),
    ('Do I have enough oxygen to make it to base b?', 'assessment'),
    ('Do I have enough air to make it to base b?', 'assessment'),
    ('Do I have enough tank to make it to base b?', 'assessment'),
    ('Enough air for base b?', 'assessment'),
    ('Enough oxygen for base b?', 'assessment'),
    ('If I go to base b, what will my oxygen be?', 'assessment'),
    ('See if I can make it to base b given my oxygen.', 'assessment'),
    ('Can I make it to base b?', 'assessment'),
    ('How far is base b and do I have enough oxygen?', 'distance'),
    ('Navigate to base b.', 'guidance'),
])
def test_navigation_budget_phrasings_choose_assessment_without_changing_tool(text, mode):
    from router.tests.test_turn import FakeRouter
    from router.turn import load_policy

    sim = SimState()
    h = TurnHandler(FakeRouter('call_navigation', .9998), sim.snapshot, default_registry(sim))
    result = h.handle(text)
    assert result['tool_calls'] == 1 and result['tool_call']['name'] == 'navigation'
    assert result['tool_call']['args'] == {'place': 'base_2'}
    data = result['tool_result']['data']
    assert data['response_mode'] == mode and not result['llm_called']
    assert result['verifier']['ok'] and result['blocked_text'] is None
    assert result['spoken'].startswith(f"base 2 is {data['feasibility']['outbound_min']} minutes away.")
    assert 'Oxygen budget' not in result['spoken'] and data['feasibility']['margin_min'] is not None


def test_distance_question_with_origin_form_place_needs_no_slot_question_after_menu_choice():
    h, _, _ = rig('How far am I from tower?')
    r = h.handle('Navigation.')
    assert r['tool_call']['name'] == 'navigation' and r['tool_calls'] == 1
    assert r['tool_call']['args'] == {'place': 'comms_tower'}
    assert r['route'] != 'slot_question' and not r['llm_called']
    assert r['tool_result']['data']['response_mode'] == 'distance'


@pytest.mark.parametrize('text,options,has_note', [
    ('check my status', ['call_safety', 'call_navigation', 'call_map'], False),
    ('can i make it home', ['call_safety', 'call_navigation', 'call_map'], True),
    ('can i make the trip', ['call_safety', 'call_navigation', 'call_map'], True),
    ('can i make it home', ['call_navigation', 'call_map', 'call_safety'], False),
])
def test_menu_destination_note_only_for_offered_safety_and_destination_context(text, options, has_note):
    from router.turn import load_policy

    sim = SimState()
    router = MenuRouter()
    router.predict = lambda _: {'label': options[0], 'confidence': .7, 'topk': [
        {'label': label, 'p': p} for label, p in zip(options, [.7, .15, .05])]}
    h = TurnHandler(router, sim.snapshot, default_registry(sim))
    result = h.handle(text)
    assert (load_policy()['clarification']['menu_destination_note'] in result['spoken']) == has_note
    assert result['verifier']['ok'] and result['tool_calls'] == 0
    assert result['pipeline']['options'] == options[:2]
    assert h.pending_clarification['choices'] == options[:2]
    assert h.pending_clarification['out']['options'] == options[:2]


@pytest.mark.parametrize('answer,tool', [('map', None), ('third', None), ('second', 'navigation')])
def test_menu_trimming_keeps_only_top_two_and_selection_uses_trimmed_set(answer, tool):
    sim = SimState()
    router = MenuRouter()
    router.predict = lambda _: {'label': 'call_safety', 'confidence': .7, 'topk': [
        {'label': 'call_safety', 'p': .7}, {'label': 'call_navigation', 'p': .15},
        {'label': 'call_map', 'p': .05}]}
    h = TurnHandler(router, sim.snapshot, default_registry(sim))
    first = h.handle('can i make it home')
    map_name = h.pipeline.config['tool_labels']['call_map']['spoken']
    assert map_name not in first['spoken']
    result = h.handle(answer)
    assert result['tool_calls'] == (1 if tool else 0)
    assert len(router.seen) == 0  # injected predictor never reclassifies the answer
    if tool:
        assert result['tool_call']['name'] == tool
        assert result['clarification_selection']['offered'] == ['call_safety', 'call_navigation']
    else:
        assert h.pending_clarification['choices'] == ['call_safety', 'call_navigation']


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '2', None])
def test_policy_loader_rejects_bad_clarification_max_options(tmp_path, value):
    import json
    from router.turn import load_policy

    policy = load_policy()
    policy['clarification']['max_options'] = value
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match='clarification.max_options'):
        load_policy(path)


def test_policy_loader_defaults_clarification_max_options(tmp_path):
    import json
    from router.turn import load_policy

    policy = load_policy()
    del policy['clarification']['max_options']
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps(policy))
    assert load_policy(path)['clarification']['max_options'] == 2


@pytest.mark.parametrize('value', ['', None, 42])
def test_policy_loader_rejects_bad_menu_destination_note(tmp_path, value):
    import json
    from router.turn import load_policy

    policy = load_policy()
    policy['clarification']['menu_destination_note'] = value
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match='clarification.menu_destination_note'):
        load_policy(path)

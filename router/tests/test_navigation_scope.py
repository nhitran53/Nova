"""Unsupported navigation scope vetoes with deterministic fake-router execution."""
import pytest

from router.tests.test_turn import Rig
from router.turn import load_policy


@pytest.mark.parametrize("text", [
    "will my battery last to base 2",
    "will my battery be sufficient to reach base 2",
    "will my battery be enough to reach base 2",
    "will my battery be adequate to reach base 2",
    "will my battery suffice to reach base 2",
    "will my battery hold out to base 2",
    "will my battery make it to base 2",
    "will my battery get me to base 2",
    "do I have enough oxygen to get home and what is my heart rate",
    "do I have enough oxygen to get home and what is my current pulse",
    "do I have enough oxygen to get home and check my temperature now",
    "do I have enough oxygen to get home and what is my battery",
    "what is my heart rate and do I have enough oxygen to get home",
    "what will my pulse be at base 2",
    "what will my heart beat be at base 2",
    "what will my temperature be at base 2",
    "temperature at base 2 when I get there",
    # Under call_navigation we cannot serve a current reading without selecting
    # another tool. Prefer clarification over executing navigation for this request;
    # this conservative fallback does not claim that 'current' means projection.
    "what is my current temperature at homebase",
    "what is my present temperature at homebase",
    "what is my temperature now at homebase",
    "do I have enough water to make it home",
    "enough fuel to get to the station",
    "will my food last for the trip",
    "do I have enough power to reach the tower",
    "what will my radiation be when I get to base 2",
    "what will my heart rate be at homebase",
    "what will my BPM be at base 2",
    "what will my temperature be on arrival",
    "radiation on arrival",
    "temperature at base 2",
    "what will my oxygen and heart rate be at homebase",
    "what will my oxygen and radiation be when I get to base 2",
    "what will my oxygen and temperature be at homebase",
    "no trip is planned, but will my battery last to base 2",
    "not going anywhere now, but what will my radiation be at base 2",
])
def test_navigation_scope_clarifies_without_any_call(text):
    rig = Rig(label="call_navigation")
    result = rig.say(text)
    assert result["route"] == "clarify"
    assert result["spoken"] == load_policy()["navigation_scope"]["clarification"]
    assert result["tool_calls"] == 0 and result["tool_call"] is None
    assert result["tool_result"] is None and not rig.registry.log
    assert not result["llm_called"] and not rig.llm.calls
    assert result["verifier"]["ok"] and result["blocked_text"] is None
    assert rig.handler.pending_confirmation is None


@pytest.mark.parametrize("text", [
    "do I have enough oxygen to make it to base 2",
    "will I have enough oxygen to make it to base 2",
    "enough air to get home",
    "can i make it home",
    "if i go to homebase, what will my oxygen levels be",
    "take me to base 2",
    "do I have enough oxygen to get home while my heart rate sensor is broken",
    "do I have enough oxygen to get home and my heart rate sensor is broken",
    "what is my battery",
    "check radiation levels",
    "no trip, check radiation levels",
    "not planning a trip, do I have enough battery",
])
def test_navigation_scope_preserves_oxygen_and_non_projection_inputs(text):
    rig = Rig(label="call_navigation")
    result = rig.say(text)
    if text in {"do I have enough oxygen to make it to base 2",
                "will I have enough oxygen to make it to base 2", "enough air to get home",
                "can i make it home", "if i go to homebase, what will my oxygen levels be",
                "take me to base 2",
                "do I have enough oxygen to get home while my heart rate sensor is broken",
                "do I have enough oxygen to get home and my heart rate sensor is broken"}:
        assert result["tool_call"] is not None
    if result["tool_call"] is not None:
        assert result["tool_call"]["name"] == "navigation"
        assert result["tool_calls"] == 1 and result["verifier"]["ok"]
        # Repeating the same request must preserve deterministic args.
        assert Rig(label="call_navigation").say(text)["tool_call"] == result["tool_call"]
    else:
        assert result["route"] == "slot_question"
    assert result["spoken"] != load_policy()["navigation_scope"]["clarification"]


@pytest.mark.parametrize("text,label", [
    ("check radiation levels", "call_data"),
    ("what is my battery", "call_data"),
])
def test_navigation_scope_does_not_swallow_plain_reading_tools(text, label):
    result = Rig(label=label).say(text)
    assert result["tool_call"]["name"] == "data" and result["tool_calls"] == 1


def test_navigation_scope_rechecks_retained_slot_context():
    rig = Rig(label="call_navigation")
    assert rig.say("will my battery last")["route"] == "slot_question"
    result = rig.say("base 2")
    assert result["route"] == "clarify" and result["tool_calls"] == 0
    assert not rig.registry.log


def test_navigation_scope_does_not_preempt_critical_alerts():
    result = Rig(label="call_navigation", o2_pct=10).say("will my battery last to base 2")
    assert result["route"] == "alert" and result["tool_calls"] == 0

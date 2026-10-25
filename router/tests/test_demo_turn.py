"""demo_turn.py with the scripted fake router (no model needed)."""
from router.demo_turn import main


def test_fake_demo_runs_sixteen_turns_and_shows_alert_slot_clarify_and_blocked_lines(capsys):
    assert main(["--fake"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n[") == 16
    assert "Critical alert. Oxygen at 14.2 percent" in out and "call_navigation conf=0.990" in out
    assert "Where would you like to go?" in out and "airlock is 2 minutes away." in out
    assert "Please choose:" in out and "blocked_text='The moon is 384400 kilometers away.'" in out
    assert "Your oxygen margin for the round trip is tight" in out
    assert "Do you want me to call crater rim to check for refills?" in out
    assert "Okay, I sent the refill request to crater rim, marked urgent." in out
    assert "You have enough oxygen to get there, but not back" in out
    # A live clarification consumes fragments instead of sending them into chitchat;
    # speaking-scope now refuses the subsequent unknown risky turn.
    assert "No action taken. Please make a new request." in out
    assert "Please contact base." in out  # unknown safety/repair question: refused

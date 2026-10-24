"""demo_turn.py with the scripted fake router (no model needed)."""
from router.demo_turn import main


def test_fake_demo_runs_sixteen_turns_and_shows_alert_slot_clarify_and_blocked_lines(capsys):
    assert main(["--fake"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n[") == 16
    assert "Critical alert. Oxygen at 14.2 percent" in out and "router : n/a (alert preempted router)" in out
    assert "Where would you like to go?" in out and "Heading to airlock." in out
    assert "I'm not sure I understood." in out and "blocked_text='The moon is 384400 kilometers away.'" in out
    assert "Warning, oxygen margin is tight" in out and "Do you want me to call outpost and check if they have oxygen refills?" in out
    assert "send_message({'recipient': 'outpost', 'urgency': 'normal'" in out and "urgency': 'urgent'" not in out
    assert "you cannot get back home without a refill" in out and "Okay, I won't call." in out
    assert "I can't answer that from memory." in out and "I don't have a document for that yet." in out

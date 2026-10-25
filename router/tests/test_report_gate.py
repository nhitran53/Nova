"""report_gate: high-confidence-wrong accounting on action tools, tau_hi sweep and dev-only choice (no model needed)."""
import numpy as np
import pytest

from router.report_gate import ACTION_LABELS, choose_tau_hi, gate_stats, sweep, wilson_upper

LABELS = ["call_safety", "call_navigation", "call_map", "call_data", "call_o2", "call_llm", "NO_TOOL", "OOD"]
TOOLS = LABELS[:5]


def probs_for(rows):
    """rows: [(predicted label, confidence, true label)] -> (probs, y_true) with the rest of the mass spread evenly."""
    probs = np.zeros((len(rows), len(LABELS)))
    for i, (pred, conf, _) in enumerate(rows):
        probs[i] = (1 - conf) / (len(LABELS) - 1)
        probs[i, LABELS.index(pred)] = conf
    return probs, np.array([LABELS.index(t) for _, _, t in rows])


ROWS = [
    ("call_navigation", 0.95, "call_navigation"),  # correct, accepted
    ("call_navigation", 0.91, "NO_TOOL"),          # HCW: false activation
    ("call_navigation", 0.90, "call_map"),         # HCW at exactly tau_hi: wrong tool
    ("call_navigation", 0.89, "OOD"),              # below tau_hi: not accepted, not HCW
    ("call_safety", 0.99, "call_safety"),
    ("call_safety", 0.97, "call_llm"),             # HCW: false activation
    ("call_data", 0.96, "call_o2"),                # wrong tool (data for o2): counts in the action total
    ("NO_TOOL", 0.99, "call_navigation"),          # missed tool: other error
    ("OOD", 0.95, "OOD"),
    ("call_map", 0.60, "call_map"),
]


def test_counts_split_false_activation_from_wrong_tool_and_other_errors():
    probs, y = probs_for(ROWS)
    s = gate_stats(probs, y, LABELS, 0.90, TOOLS)
    nav, safety, data = s["per_class"]["call_navigation"], s["per_class"]["call_safety"], s["per_class"]["call_data"]
    assert (nav["accepted"], nav["correct"], nav["hcw"]) == (3, 1, 2)
    assert (nav["hcw_false_activation"], nav["hcw_wrong_tool"]) == (1, 1)
    assert nav["wrong_by_true_label"] == {"call_map": 1, "NO_TOOL": 1}
    assert (safety["accepted"], safety["hcw"], safety["hcw_false_activation"]) == (2, 1, 1)
    assert data["hcw"] == 1 and data["action"] and data["wrong_by_true_label"] == {"call_o2": 1}
    assert s["action_hcw"] == 4 and s["action_hcw_false_activation"] == 2 and s["action_hcw_wrong_tool"] == 2
    assert s["action_hcw_rate"] == pytest.approx(4 / 10)
    assert nav["hcw_rate_over_accepted"] == pytest.approx(2 / 3) and nav["hcw_rate_over_all"] == pytest.approx(2 / 10)
    # argmax errors (rows 2,3,4,6,7,8) = 4 HCW + below-tau_hi nav error + missed tool
    assert s["total_errors_argmax"] == 6 and s["errors_not_counted_as_action_hcw"] == 2


def test_tau_hi_boundary_is_inclusive_and_raising_it_drops_hcw():
    probs, y = probs_for(ROWS)
    assert gate_stats(probs, y, LABELS, 0.90, TOOLS)["per_class"]["call_navigation"]["accepted"] == 3  # 0.90 counts
    assert gate_stats(probs, y, LABELS, 0.91 + 1e-9, TOOLS)["per_class"]["call_navigation"]["accepted"] == 1
    rates = [r["action_hcw_rate"] for r in sweep(probs, y, LABELS, TOOLS, grid=(0.5, 0.9, 0.98))]
    assert rates == sorted(rates, reverse=True)


def test_default_action_labels_are_all_five_tools_and_only_action_labels_enter_the_total():
    assert ACTION_LABELS == ("call_safety", "call_navigation", "call_map", "call_data", "call_o2")
    probs, y = probs_for(ROWS)
    s = gate_stats(probs, y, LABELS, 0.90, TOOLS, action_labels=("call_data",))
    assert s["action_hcw"] == 1  # only call_data counted when overridden


def test_choose_tau_hi_is_lowest_grid_value_meeting_target_on_the_dev_sweep_given():
    probs, y = probs_for(ROWS)
    rows = sweep(probs, y, LABELS, TOOLS, grid=(0.5, 0.9, 0.96, 0.995))
    assert choose_tau_hi(rows, target=0.0) == 0.995  # all action HCW gone only above 0.99
    assert choose_tau_hi(rows, target=0.2) == 0.96
    assert choose_tau_hi(rows, target=1.0) == 0.5
    assert choose_tau_hi(rows[:2], target=0.0) is None  # nothing meets the target


def test_wilson_upper_bound():
    assert wilson_upper(0, 0) is None
    assert wilson_upper(0, 320) == pytest.approx(0.0119, abs=2e-4)
    assert wilson_upper(1, 320) > 1 / 320


def test_report_override_does_not_change_dev_only_suggestion(tmp_path, monkeypatch, capsys):
    from router import report_gate

    class FakeRouter:
        untrained = False
        temperature = 1.0

        @classmethod
        def load(cls, *args, **kwargs):
            return cls()

    dev = probs_for([("call_navigation", 0.91, "NO_TOOL")])
    test = probs_for([("call_navigation", 0.99, "NO_TOOL")])
    monkeypatch.setattr(report_gate, "Router", FakeRouter)
    monkeypatch.setattr(report_gate, "split_probs", lambda router, data_dir, split, labels, batch_size=4:
                        (*(dev if split == "dev" else test), ["example"]))
    report_gate.main(["--checkpoint", "/tmp/fake-v2", "--report-dir", str(tmp_path), "--tau-hi", "0.9"])
    import json
    report = json.loads((tmp_path / "fake-v2_gate_dev_test.json").read_text())
    assert report["dev_chosen_tau_hi"] == 0.92  # not the test choice (none) or the override
    assert report["scored_tau_hi"] == 0.9
    assert report["test_at_chosen"]["action_hcw"] == 1
    assert report["legacy_dev_selection"] is True
    assert report["current_tau_hi"] == 0.9
    assert report["offline_proposed_tau_hi"] == 0.92
    assert "Suggested tau_hi = 0.92 (dev only; offline, no policy change)" in capsys.readouterr().out

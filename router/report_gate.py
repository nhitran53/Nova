"""Confidence-gate report: high-confidence-wrong (HCW) rate on action tools, dev and test reported SEPARATELY.

    python -m router.report_gate --checkpoint $NOVA_MODELS_DIR/v2 --data-dir data/v2 [--target 0.005] [--tau-hi 0.9]

Definitions (choices documented here, review finding 5):
  * Tool classes = pipeline_config.json tool_labels. ACTION tools = call_safety, call_navigation, call_map
    (override with --action-labels): they drive the go/no-go, movement guidance and map state, so a false
    activation changes what the astronaut does or sees. call_data / call_o2 are read-only queries; they are
    reported per class but are not in the action total.
  * An utterance is "accepted" for class C at tau_hi when the router's top class is C and calibrated confidence
    >= tau_hi (the pipeline executes C without asking the arbiter). Everything below tau_hi goes to the arbiter or
    to clarification, which this report cannot score (the arbiter is a stub), so it conservatively counts as
    not executed.
  * HCW for C = accepted for C and the true label != C. HCW rate = HCW / ALL utterances in the split (the
    per-utterance false-activation rate the 0.5% target is read against); precision error HCW / accepted(C)
    is shown next to it, plus a Wilson 95% upper bound on the combined action rate because n is small.
  * HCW splits into FALSE ACTIVATION (true label is NO_TOOL / call_llm / OOD: an action fired when no tool was
    wanted) and WRONG TOOL (true label is a different tool class). Other errors (missed tool, non-tool
    confusions, read-only tool errors) are counted separately and never mixed into the action rate.
  * tau_hi is swept; the suggestion is chosen on DEV ONLY: the lowest grid value whose combined action HCW rate
    on dev is <= target (max coverage subject to the target). Test is then scored at that value and its own
    sweep is printed for information only; nothing is tuned on test.

HONESTY NOTE: synthetic data, a few hundred rows per split; 0.5% of 320 is 1.6 utterances, so one or two errors
decide pass/fail. Treat the numbers as a smoke test of the gate, not as evidence for real speech.
"""
import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

from .core import Router, load_labels, models_dir
from .evaluate import compute_logits, load_jsonl, softmax
from .pipeline import load_config

HERE = Path(__file__).parent
ACTION_LABELS = ("call_safety", "call_navigation", "call_map")
NON_TOOL_LABELS = ("call_llm", "NO_TOOL", "OOD")
TAU_HI_GRID = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99)
DEFAULT_TARGET = 0.005


def wilson_upper(k, n, z=1.96):
    """Upper end of the Wilson score interval for k events in n trials."""
    if n == 0:
        return None
    p = k / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre + margin) / (1 + z * z / n)


def gate_stats(probs, y_true, labels, tau_hi, tool_labels, action_labels=ACTION_LABELS):
    """Counts and rates for one split at one tau_hi. `probs` [n, n_labels] calibrated, `y_true` label indices."""
    n = len(y_true)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    accepted = conf >= tau_hi
    names = np.array(labels)
    per_class = {}
    for tool in tool_labels:
        c = labels.index(tool)
        acc = accepted & (pred == c)
        wrong = acc & (y_true != c)
        false_activation = wrong & np.isin(names[y_true], NON_TOOL_LABELS)
        per_class[tool] = {
            "action": tool in action_labels,
            "accepted": int(acc.sum()),
            "correct": int((acc & (y_true == c)).sum()),
            "hcw": int(wrong.sum()),
            "hcw_false_activation": int(false_activation.sum()),
            "hcw_wrong_tool": int(wrong.sum() - false_activation.sum()),
            "hcw_rate_over_all": float(wrong.sum() / n),
            "hcw_rate_over_accepted": float(wrong.sum() / acc.sum()) if acc.any() else None,
            "support": int((y_true == c).sum()),
            "recall_at_gate": float((acc & (y_true == c)).sum() / max((y_true == c).sum(), 1)),
            "wrong_by_true_label": {name: int((wrong & (names[y_true] == name)).sum())
                                    for name in labels if (wrong & (names[y_true] == name)).any()},
        }
    action = [v for v in per_class.values() if v["action"]]
    hcw = sum(v["hcw"] for v in action)
    all_errors = int((pred != y_true).sum())
    return {
        "tau_hi": tau_hi,
        "n": n,
        "per_class": per_class,
        "action_accepted": sum(v["accepted"] for v in action),
        "action_hcw": hcw,
        "action_hcw_false_activation": sum(v["hcw_false_activation"] for v in action),
        "action_hcw_wrong_tool": sum(v["hcw_wrong_tool"] for v in action),
        "action_hcw_rate": hcw / n,
        "action_hcw_rate_wilson95_upper": wilson_upper(hcw, n),
        "action_recall_at_gate": float(sum(v["correct"] for v in action) / max(sum(v["support"] for v in action), 1)),
        "total_errors_argmax": all_errors,
        "errors_not_counted_as_action_hcw": all_errors - hcw,  # missed tools, non-tool confusions, low-conf action errors
    }


def sweep(probs, y_true, labels, tool_labels, action_labels=ACTION_LABELS, grid=TAU_HI_GRID):
    return [gate_stats(probs, y_true, labels, t, tool_labels, action_labels) for t in grid]


def choose_tau_hi(dev_sweep, target=DEFAULT_TARGET):
    """Lowest swept tau_hi whose combined action HCW rate on DEV is <= target; None if even the highest fails."""
    for row in sorted(dev_sweep, key=lambda r: r["tau_hi"]):
        if row["action_hcw_rate"] <= target:
            return row["tau_hi"]
    return None


def format_sweep(title, rows, target):
    lines = [title, f"{'tau_hi':>6} {'act.acc':>7} {'HCW':>4} {'false-act':>9} {'wrong-tool':>10} {'HCW rate':>9} "
                    f"{'wilson95+':>9} {'recall@gate':>11}  target {target:.3%}"]
    for r in rows:
        lines.append(f"{r['tau_hi']:>6} {r['action_accepted']:>7} {r['action_hcw']:>4} {r['action_hcw_false_activation']:>9} "
                     f"{r['action_hcw_wrong_tool']:>10} {r['action_hcw_rate']:>9.3%} {r['action_hcw_rate_wilson95_upper']:>9.3%} "
                     f"{r['action_recall_at_gate']:>11.3f}  {'ok' if r['action_hcw_rate'] <= target else 'OVER'}")
    return "\n".join(lines)


def format_per_class(title, stats):
    lines = [title, f"{'class':<16} {'action':>6} {'accepted':>8} {'HCW':>4} {'false-act':>9} {'wrong-tool':>10} "
                    f"{'rate/all':>9} {'rate/acc':>9}  wrong-by-true-label"]
    for tool, v in stats["per_class"].items():
        acc_rate = "n/a" if v["hcw_rate_over_accepted"] is None else f"{v['hcw_rate_over_accepted']:.3%}"
        lines.append(f"{tool:<16} {'yes' if v['action'] else 'no':>6} {v['accepted']:>8} {v['hcw']:>4} "
                     f"{v['hcw_false_activation']:>9} {v['hcw_wrong_tool']:>10} {v['hcw_rate_over_all']:>9.3%} "
                     f"{acc_rate:>9}  {v['wrong_by_true_label']}")
    lines.append(f"action total: HCW {stats['action_hcw']}/{stats['n']} = {stats['action_hcw_rate']:.3%} "
                 f"(false activations {stats['action_hcw_false_activation']}, wrong tool {stats['action_hcw_wrong_tool']}); "
                 f"other argmax errors not counted: {stats['errors_not_counted_as_action_hcw']}/{stats['total_errors_argmax']}")
    return "\n".join(lines)


def split_probs(router, data_dir, split, labels):
    texts, y = load_jsonl(Path(data_dir) / f"{split}.jsonl", labels)
    logits = compute_logits(router.model, router.tokenizer, texts, router.device, router.max_length)
    return softmax(logits, router.temperature), y, texts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=str(models_dir() / "v2"), help="default: %(default)s")
    ap.add_argument("--data-dir", default=str(HERE / "data" / "v2"))
    ap.add_argument("--labels", default=str(HERE / "labels.json"))
    ap.add_argument("--target", type=float, default=DEFAULT_TARGET, help="HCW rate target on action tools (default 0.5%%)")
    ap.add_argument("--tau-hi", type=float, default=None, help="score test at this value instead of the dev-chosen one")
    ap.add_argument("--action-labels", nargs="+", default=list(ACTION_LABELS))
    ap.add_argument("--report-dir", default=str(HERE / "reports"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    labels = load_labels(args.labels)
    tool_labels = list(load_config(labels=labels)["tool_labels"])
    unknown = set(args.action_labels) - set(tool_labels)
    if unknown:
        sys.exit(f"--action-labels not tool labels: {sorted(unknown)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        router = Router.load(args.checkpoint, args.labels, device=args.device)
    if router.untrained:
        sys.exit(f"{args.checkpoint} is untrained")

    out = {"checkpoint": str(args.checkpoint), "data_dir": str(args.data_dir), "target": args.target,
           "action_labels": args.action_labels, "temperature": router.temperature}
    sweeps = {}
    probs = {}
    for split in ("dev", "test"):
        p, y, _ = split_probs(router, args.data_dir, split, labels)
        probs[split] = (p, y)
        sweeps[split] = sweep(p, y, labels, tool_labels, args.action_labels)
        print(format_sweep(f"== {split.upper()} sweep ({len(y)} rows)" + ("  [used to choose tau_hi]" if split == "dev"
                           else "  [information only, not used to choose]"), sweeps[split], args.target))
        print()
    dev_chosen = choose_tau_hi(sweeps["dev"], args.target)
    chosen = args.tau_hi if args.tau_hi is not None else dev_chosen
    out["dev_chosen_tau_hi"] = dev_chosen
    out["scored_tau_hi"] = chosen
    out["tau_hi_source"] = "--tau-hi" if args.tau_hi is not None else "dev only"
    print(f"Suggested tau_hi = {dev_chosen} (dev only).")
    if chosen is None:
        print(f"No swept tau_hi meets the {args.target:.3%} target on DEV; not scoring test at a chosen value.")
    else:
        print(f"Scoring tau_hi = {chosen} ({out['tau_hi_source']}).")
        for split in ("dev", "test"):
            stats = gate_stats(*probs[split], labels, chosen, tool_labels, args.action_labels)
            out[f"{split}_at_chosen"] = stats
            print(format_per_class(f"-- {split.upper()} per class at tau_hi={chosen}", stats))
            print()
    out["sweeps"] = sweeps
    out_path = Path(args.report_dir) / f"{Path(args.checkpoint).resolve().name}_gate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"report -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

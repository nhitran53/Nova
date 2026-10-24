"""Pipeline glue (stub): classifier label -> tool call with resolved args, clarification, or LLM hand-off.

    python -m router.pipeline --text "take me home" [--checkpoint $NOVA_MODELS_DIR/v1] [--tau-hi 0.9] [--tau-lo 0.5]

Confidence gate (calibrated top-1 confidence `c`; thresholds in pipeline_config.json):
    top class is a TOOL label:
        c >= tau_hi             -> tool route
        tau_lo <= c < tau_hi    -> arbiter_fn(text, topk) -> label | "none".  Same label as the router -> tool route;
                                   anything else, no arbiter supplied, or an arbiter exception -> clarify route (no tool)
        c < tau_lo              -> LLM route, reason "low_confidence" (clarify policy)
    top class is NO_TOOL / call_llm / OOD: LLM route with that class's prompt policy (low_confidence if c < tau_lo).
The arbiter can only CONFIRM the router's class; it never selects another tool, so the LLM never overrides the router.

`Pipeline.handle(text)` returns one of
    {"route": "tool", "tool": "navigation", "args": {"place": "homebase"}, "confidence": 0.97, "label": "call_navigation",
     "gate": "tau_hi"|"arbiter_agree"}
    {"route": "clarify", "reason": "arbiter_disagree"|"no_arbiter"|"arbiter_error"|"ambiguous_place",
     "question": "...", "options": [...], "confidence": 0.7, "label": "..."}      # deterministic text, no LLM
    {"route": "llm", "reason": "OOD"|"NO_TOOL"|"call_llm"|"low_confidence", "prompt_policy": "...",
     "response": <llm_fn output>, "confidence": 0.9, "label": "..."}
Navigation/map resolve a place with places.resolve_destination; two or more candidate places -> clarify
("Which place do you mean: ...?"), never a positional guess. Which labels are tools lives in pipeline_config.json.
No real LLM or arbiter is called: `llm_fn(text, prompt_policy)` and `arbiter_fn(text, topk)` are injectable.
`topk` is the router's top-3 [{"label","p"}]; the arbiter picks one of those labels or "none".
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

from .core import Router, load_labels, models_dir
from .places import DEFAULT_THRESHOLD, resolve_destination

HERE = Path(__file__).parent
CONFIG_PATH = HERE / "pipeline_config.json"


def placeholder_llm(text, prompt_policy):
    return f"[LLM placeholder] policy={prompt_policy.split(':')[0]} text={text!r}"


def load_config(path=CONFIG_PATH, labels=None):
    """Read pipeline_config.json; if `labels` is given, check every label is routed exactly once."""
    config = json.loads(Path(path).read_text())
    overlap = set(config["tool_labels"]) & set(config["llm_labels"])
    if overlap:
        raise ValueError(f"{path}: labels both tool and llm: {sorted(overlap)}")
    if not 0.0 <= config["tau_lo"] <= config["tau_hi"] <= 1.0:
        raise ValueError(f"{path}: need 0 <= tau_lo <= tau_hi <= 1, got {config['tau_lo']}, {config['tau_hi']}")
    if labels is not None:
        routed = set(config["tool_labels"]) | set(config["llm_labels"])
        if routed != set(labels):
            raise ValueError(f"{path}: routed labels differ from labels.json: {sorted(routed ^ set(labels))}")
    return config


def join_options(options):
    """['a'] -> 'a'; ['a','b'] -> 'a or b'; ['a','b','c'] -> 'a, b, or c'."""
    if len(options) <= 2:
        return " or ".join(options)
    return ", ".join(options[:-1]) + ", or " + options[-1]


class Pipeline:
    def __init__(self, router, config=None, llm_fn=placeholder_llm, arbiter_fn=None, tau_hi=None, tau_lo=None,
                 place_threshold=DEFAULT_THRESHOLD):
        """`router` needs `.predict(text) -> {"label", "confidence", "topk"?}` (a core.Router or a fake).

        `arbiter_fn(text, topk) -> str` is optional; without it the middle-confidence band always clarifies."""
        self.router = router
        self.config = load_config() if config is None else config
        self.llm_fn = llm_fn
        self.arbiter_fn = arbiter_fn
        self.tau_hi = self.config["tau_hi"] if tau_hi is None else tau_hi
        self.tau_lo = self.config["tau_lo"] if tau_lo is None else tau_lo
        if not 0.0 <= self.tau_lo <= self.tau_hi <= 1.0:
            raise ValueError(f"need 0 <= tau_lo <= tau_hi <= 1, got tau_lo={self.tau_lo} tau_hi={self.tau_hi}")
        self.place_threshold = place_threshold

    def handle(self, text):
        prediction = self.router.predict(text)
        label, confidence = prediction["label"], prediction["confidence"]
        topk = (prediction.get("topk") or [{"label": label, "p": confidence}])[:3]
        tool_cfg = self.config["tool_labels"].get(label)
        if confidence < self.tau_lo:
            return self._llm(text, label, confidence, "low_confidence", self.config["low_confidence_policy"])
        if tool_cfg is None:
            return self._llm(text, label, confidence, label, self.config["llm_labels"][label])
        if confidence >= self.tau_hi:
            return self._tool(text, label, confidence, tool_cfg, gate="tau_hi", topk=topk)
        verdict = self._arbitrate(text, label, topk)
        if verdict == "agree":
            return self._tool(text, label, confidence, tool_cfg, gate="arbiter_agree", topk=topk)
        return self._clarify_intent(label, confidence, topk, reason=verdict)

    def _arbitrate(self, text, label, topk):
        """'agree' only if the arbiter returns exactly the router's label; else a clarify reason."""
        if self.arbiter_fn is None:
            return "no_arbiter"
        try:
            answer = self.arbiter_fn(text, topk)
        except Exception:  # an arbiter failure must never turn into an action
            return "arbiter_error"
        return "agree" if answer == label else "arbiter_disagree"

    def _tool(self, text, label, confidence, tool_cfg, gate, topk):
        args = {}
        if tool_cfg.get("resolve_place"):
            resolution = resolve_destination(text, threshold=self.place_threshold)
            if resolution["status"] == "ambiguous":
                options = [c["place_id"].replace("_", " ") for c in resolution["candidates"]]
                return {"route": "clarify", "reason": "ambiguous_place", "label": label, "confidence": confidence,
                        "question": self.config["place_question"].format(options=join_options(options)),
                        "options": [c["place_id"] for c in resolution["candidates"]]}
            if resolution["status"] == "resolved":
                args["place"] = resolution["place"]["place_id"]
        return {"route": "tool", "tool": tool_cfg["tool"], "args": args, "confidence": confidence, "label": label,
                "gate": gate}

    def _clarify_intent(self, label, confidence, topk, reason):
        """Deterministic question over the router's top-k TOOL classes (never executes, never calls the LLM)."""
        tools = self.config["tool_labels"]
        options = [t["label"] for t in topk if t["label"] in tools] or [label]
        spoken = [tools[o]["spoken"] for o in options]
        return {"route": "clarify", "reason": reason, "label": label, "confidence": confidence,
                "question": self.config["clarify_question"].format(options=join_options(spoken)), "options": options}

    def _llm(self, text, label, confidence, reason, policy):
        return {"route": "llm", "reason": reason, "prompt_policy": policy, "response": self.llm_fn(text, policy),
                "confidence": confidence, "label": label}


def load_pipeline(checkpoint, labels_path=HERE / "labels.json", device="cpu", **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        router = Router.load(checkpoint, labels_path, device=device)
    if router.untrained:
        raise SystemExit(f"{checkpoint} is untrained (no router_meta.json with trained=true)")
    return Pipeline(router, config=load_config(labels=load_labels(labels_path)), **kwargs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--text", action="append", required=True, help="utterance (repeatable)")
    ap.add_argument("--checkpoint", default=str(models_dir() / "v1"), help="default: %(default)s (set NOVA_MODELS_DIR to move)")
    ap.add_argument("--tau-hi", type=float, default=None, help="default: pipeline_config.json")
    ap.add_argument("--tau-lo", type=float, default=None, help="default: pipeline_config.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    pipeline = load_pipeline(args.checkpoint, device=args.device, tau_hi=args.tau_hi, tau_lo=args.tau_lo)
    for text in args.text:
        print(json.dumps({"text": text, **pipeline.handle(text)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

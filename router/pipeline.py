"""Pipeline glue (stub): classifier label -> tool call with resolved args, or LLM hand-off.

    python -m router.pipeline --text "take me home" [--checkpoint $NOVA_MODELS_DIR/v1] [--tau-lo 0.5]

`Pipeline.handle(text)` returns one of
    {"route": "tool", "tool": "navigation", "args": {"place": "homebase"}, "confidence": 0.97, "label": "call_navigation"}
    {"route": "llm", "reason": "OOD"|"NO_TOOL"|"call_llm"|"low_confidence", "prompt_policy": "...",
     "response": <llm_fn output>, "confidence": 0.9, "label": "..."}
Which labels are tools (and which resolve a place) lives in pipeline_config.json, not in code. No real LLM is
called: `llm_fn(text, prompt_policy)` is injectable and defaults to a placeholder string.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

from .core import Router, load_labels, models_dir
from .places import DEFAULT_THRESHOLD, resolve_place

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
    if labels is not None:
        routed = set(config["tool_labels"]) | set(config["llm_labels"])
        if routed != set(labels):
            raise ValueError(f"{path}: routed labels differ from labels.json: {sorted(routed ^ set(labels))}")
    return config


class Pipeline:
    def __init__(self, router, config=None, llm_fn=placeholder_llm, tau_lo=None, place_threshold=DEFAULT_THRESHOLD):
        """`router` needs `.predict(text) -> {"label", "confidence", ...}` (a core.Router or a fake)."""
        self.router = router
        self.config = load_config() if config is None else config
        self.llm_fn = llm_fn
        self.tau_lo = self.config["tau_lo"] if tau_lo is None else tau_lo
        self.place_threshold = place_threshold

    def handle(self, text):
        prediction = self.router.predict(text)
        label, confidence = prediction["label"], prediction["confidence"]
        tool_cfg = self.config["tool_labels"].get(label)
        if tool_cfg is not None and confidence >= self.tau_lo:
            args = {}
            if tool_cfg.get("resolve_place"):
                match = resolve_place(text, threshold=self.place_threshold)
                if match:
                    args["place"] = match["place_id"]
            return {"route": "tool", "tool": tool_cfg["tool"], "args": args, "confidence": confidence, "label": label}
        if confidence < self.tau_lo:
            reason, policy = "low_confidence", self.config["low_confidence_policy"]
        else:
            reason, policy = label, self.config["llm_labels"][label]
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
    ap.add_argument("--tau-lo", type=float, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    pipeline = load_pipeline(args.checkpoint, device=args.device, tau_lo=args.tau_lo)
    for text in args.text:
        print(json.dumps({"text": text, **pipeline.handle(text)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

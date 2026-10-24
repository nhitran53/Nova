"""Run ~16 scripted turns through turn.py and print route, tool call, spoken line and verifier result.

    python -m router.demo_turn [--model $NOVA_MODELS_DIR/v1]      # real ModernBERT router (CPU)
    python -m router.demo_turn --fake                              # scripted fake router, no model needed
    python -m router.demo_turn --fake --speak --max-turns 3        # also speak each line (real ElevenLabs call)

Fake sim values, fake tools, stub LLM, a keyword fake retriever (only "moon" has a document), no arbiter: nothing here
calls a real LLM or a real tool. Confirmation replies ("yes"/"no") are answered by the deterministic matcher before the
router, so their fake-router entry is never used.
"""
import argparse
import sys
import warnings
from pathlib import Path

from .core import Router, models_dir
from . import speech
from .tools import SimState, default_registry
from .turn import TurnHandler

HERE = Path(__file__).parent

# (label shown in output, utterance, sim override, fake-router (label, confidence))
TURNS = [
    ("tool: o2", "how much oxygen do I have left", {}, ("call_o2", 0.98)),
    ("tool: data", "what's my heart rate and temperature", {}, ("call_data", 0.97)),
    ("tool: navigation", "take me home", {}, ("call_navigation", 0.99)),
    ("slot question (no place)", "take me there", {}, ("call_navigation", 0.97)),
    ("origin + destination", "go from the rover to the airlock", {}, ("call_navigation", 0.99)),
    ("mid-band, no arbiter -> clarify", "Get me the coolant level.", {}, ("call_data", 0.75)),
    ("chitchat -> LLM stub", "hi", {}, ("NO_TOOL", 0.97)),
    ("OOD -> template", "what is the capital of france", {}, ("OOD", 0.95)),
    ("CRITICAL alert (o2 14.2) preempts", "take me to the airlock", {"o2_pct": 14.2}, ("call_navigation", 0.99)),
    ("blocked number (LLM stub says 384400)", "how far away is the moon", {}, ("call_llm", 0.96)),
    ("nav, TIGHT oxygen (55 min) -> warning + refill offer", "take me to the outpost", {"minutes_remaining": 55}, ("call_navigation", 0.99)),
    ("follow-up 'yes' -> one send_message (urgency from code)", "yes please", {"minutes_remaining": 55}, ("call_o2", 0.99)),
    ("nav, CANNOT RETURN (45 min) -> warning + refill offer", "take me to the outpost", {"minutes_remaining": 45}, ("call_navigation", 0.99)),
    ("follow-up 'no' -> ack, no tool", "no thanks", {"minutes_remaining": 45}, ("call_o2", 0.99)),
    ("chitchat with risk term -> template, LLM not called", "good morning, my suit is comfy", {}, ("NO_TOOL", 0.97)),
    ("call_llm, no document -> template, LLM not called", "what is the boiling point of lunar coolant", {}, ("call_llm", 0.96)),
]


class ScriptedRouter:
    def __init__(self):
        self.next = None

    def predict(self, text):
        label, confidence = self.next
        return {"label": label, "confidence": confidence,
                "topk": [{"label": label, "p": confidence}, {"label": "NO_TOOL", "p": 1 - confidence}]}


def demo_llm(text, prompt_policy, passages=None):
    if "moon" in text:
        return "The moon is 384400 kilometers away."  # a number with no tool_result behind it
    return "Hello! How can I help?"


def demo_retriever(text):
    """Keyword fake: the only 'document' is about the moon, and it contains no number."""
    if "moon" in text:
        return [{"text": "The Moon is Earth's only natural satellite.", "score": 0.9}]
    return [{"text": "unrelated passage", "score": 0.1}]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=str(models_dir() / "v1"), help="default: %(default)s")
    ap.add_argument("--fake", action="store_true", help="use a scripted fake router instead of loading a checkpoint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--speak", action="store_true", help="also speak each line through ElevenLabs (real API call + speakers)")
    ap.add_argument("--max-turns", type=int, default=None, help="run only the first N scripted turns")
    args = ap.parse_args(argv)

    if args.fake:
        router = scripted = ScriptedRouter()
    else:
        scripted = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            router = Router.load(args.model, HERE / "labels.json", device=args.device)
    sim = SimState()
    handler = TurnHandler(router, sim.snapshot, default_registry(sim), llm_fn=demo_llm, retriever=demo_retriever,
                          tts_fn=speech.speak if args.speak else None)
    print(f"router: {'FAKE (scripted)' if args.fake else args.model}; sim/tools/LLM are fakes; arbiter: none")
    if args.speak:
        speech.init()
    for title, text, override, fake_route in TURNS[:args.max_turns]:
        if scripted:
            scripted.next = fake_route
        base = dict(sim.values)
        sim.values.update(override)
        out = handler.handle(text)
        sim.values = base
        pipe = out["pipeline"]
        if pipe is None:
            route = "n/a (alert preempted router)" if out["route"] == "alert" else "n/a (yes/no matcher answered before the router)"
        else:
            route = f"{pipe['label']} conf={pipe['confidence']:.3f} -> {pipe['route']}/{pipe.get('gate') or pipe.get('reason')}"
        call = out["tool_call"] and f"{out['tool_call']['name']}({out['tool_call']['args']})"
        print(f"\n[{title}] {text!r}")
        print(f"  router : {route}")
        print(f"  turn   : {out['route']}   tool call: {call or 'none'}   llm called: {out['llm_called']}"
              + (f"   pending: {out['pending_confirmation']['kind']}/{out['pending_confirmation']['outcome']}" if out["pending_confirmation"] else ""))
        print(f"  spoken : {out['spoken']}")
        print(f"  verify : ok={out['verifier']['ok']} spoken_numbers={out['verifier']['spoken']} "
              f"unsupported={out['verifier']['unsupported']}" + (f"  blocked_text={out['blocked_text']!r}" if out["blocked_text"] else ""))
        if args.speak:
            print(f"  speech : " + (f"FAILED {out['tts_error']}" if "tts_error" in out else "spoken"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run ~16 scripted turns through turn.py and print route, tool call, spoken line and verifier result.

    python -m router.demo_turn [--model $NOVA_MODELS_DIR/v1]      # real ModernBERT router (CPU)
    python -m router.demo_turn --fake                              # scripted fake router, no model needed
    python -m router.demo_turn --fake --speak --max-turns 3        # also speak each line (real ElevenLabs call)

After the scripted turns a "generation demo" prints, for a few turns, the GenerationRequest summary, the rendered prompt,
the raw generator output and the post-processed verified line (FillerGenerator / forced-bad generators, offline).

A final "DB-backed" section runs the same TurnHandler over FixtureDatasource (rows seeded from schema.sql, offline: no
Postgres), including an o2_leak advanced until the oxygen alert preempts.

Fake sim values, fake tools, stub LLM, a keyword fake retriever (only "moon" has a document), no arbiter: nothing here
calls a real LLM or a real tool. Confirmation replies ("yes"/"no") are answered by the deterministic matcher before the
router, so their fake-router entry is never used.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

from .core import Router, models_dir
from . import speech
from .datasource import FixtureDatasource
from .generation import FillerGenerator, render_prompt_text, run_generation
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
    """Keyword fake: one 'document' about the moon (no number) and one about suit pressure (a number and a risk term)."""
    if "moon" in text:
        return [{"text": "The Moon is Earth's only natural satellite.", "score": 0.9}]
    if "airlock" in text:
        return [{"text": "The airlock cycle takes 12 minutes.", "score": 0.9}]
    return [{"text": "unrelated passage", "score": 0.1}]


# (title, utterance, sim override, fake-router (label, confidence), zero-arg generator factory, or None for the default FillerGenerator; a factory because scripted generators keep state)
GENERATION_TURNS = [
    ("O2 tool turn (template spoken; shadow generation shown)", "how much oxygen do I have left", {}, ("call_o2", 0.98), None),
    ("navigation tool turn, cannot-return feasibility slots", "take me to the outpost", {"minutes_remaining": 45}, ("call_navigation", 0.99), None),
    ("chitchat via FillerGenerator", "hi", {}, ("NO_TOOL", 0.97), None),
    ("grounded call_llm via FillerGenerator + fake retriever", "how far away is the moon", {}, ("call_llm", 0.96), None),
    ("grounded, passage number, no risk term: digits mode, code adds the attribution prefix", "how long is the airlock cycle", {}, ("call_llm", 0.96),
     lambda: FillerGenerator(script=[json.dumps({"refusal": None, "text": "The airlock cycle takes 12 minutes."})])),
    ("grounded, RISK TERM in the question: passage numbers forced off for this turn (digits masked, number refused)", "what is the suit pressure for the airlock", {}, ("call_llm", 0.96),
     lambda: FillerGenerator(script=[json.dumps({"refusal": None, "text": "The airlock cycle takes 12 minutes."}),
                                     json.dumps({"refusal": "no_info", "text": ""})])),
    ("forced-bad generator: unsupported number twice -> retry -> verifier_blocked fallback", "how far away is the moon", {}, ("call_llm", 0.96),
     lambda: FillerGenerator(script=[json.dumps({"refusal": None, "text": "The moon is 384400 kilometers away."})])),
    ("forced-bad chitchat: advice, then corrected on the retry", "hello there", {}, ("NO_TOOL", 0.97),
     lambda: FillerGenerator(script=[json.dumps({"refusal": None, "text": "You should return to base."}),
                                     json.dumps({"refusal": None, "text": "Hello, happy to help."})])),
    ("forced generator exception -> tool_error line, no retry", "hello again", {}, ("NO_TOOL", 0.97), lambda: FillerGenerator.forcing("raises")),
    ("critical alert: no request is built, no generator call", "hello", {"o2_pct": 14.2}, ("NO_TOOL", 0.97), None),
]


def generation_demo(router, scripted, sim):
    print("\n== generation demo (offline: FillerGenerator / forced-bad generators, no real LLM) ==")
    for title, text, override, fake_route, make_generator in GENERATION_TURNS:
        generator = (make_generator or FillerGenerator)()
        handler = TurnHandler(router, sim.snapshot, default_registry(sim), generator=generator, retriever=demo_retriever)
        if scripted:
            scripted.next = fake_route
        base = dict(sim.values)
        sim.values.update(override)
        out = handler.handle(text)
        sim.values = base
        gen = out.get("generation")
        print(f"\n== {title}: {text!r}")
        print(f"  route: {out['route']}   llm called: {out['llm_called']}   generator calls: {len(generator.calls)}")
        if gen is None:
            print("  generation: none (no request built for this turn)")
        else:
            print(f"  request : {json.dumps(gen['request'])}")
            print("  prompt  :\n" + "\n".join("    " + line for line in render_prompt_text(gen["prompt"]).splitlines()))
            if gen["mode"] == "template_only":
                shadow = run_generation(handler.last_generation_request, FillerGenerator(), handler.policy, out["spoken"])
                print(f"  raw     : {shadow.raw}   (shadow run, NOT spoken)")
                print(f"  shadow  : {shadow.status} -> {shadow.text!r}")
                print(f"  spoken  : {out['spoken']}   (the template; llm not called)")
            else:
                for attempt in gen["attempt_log"]:
                    print(f"  raw #{attempt['attempt']} : {attempt['raw']}   violations={attempt['violations']}")
                if gen["request"]["kind"] == "grounded_answer":
                    print(f"  numbers : configured={gen['configured_passage_numbers']} effective={gen['passage_numbers']} "
                          f"override={gen['passage_override_reason']!r} attribution_prefixed={gen['attribution_prefixed']}")
                print(f"  outcome : {gen['status']} after {gen['attempts']} attempt(s); violations={gen['violations']}")
                print(f"  spoken  : {out['spoken']}" + (f"   blocked_text={out['blocked_text']!r}" if out["blocked_text"] else "")
                      + (f"   llm_error={out['llm_error']!r}" if "llm_error" in out else ""))
        if gen is None:
            print(f"  spoken  : {out['spoken']}")


# (title, utterance, fake-router (label, confidence), sim seconds to advance the fixture BEFORE this turn, scenario to switch to or None)
DB_TURNS = [
    ("o2 from DB telemetry", "how much oxygen do I have left", ("call_o2", 0.98), 0, None),
    ("data: radiation is a delta of the cumulative dose", "what's my heart rate and temperature", ("call_data", 0.97), 0, None),
    ("navigation to the rover: DB coordinates + feasibility", "take me to the rover", ("call_navigation", 0.99), 0, None),
    ("navigation to the crater rim: comfortable", "take me to the crater rim", ("call_navigation", 0.99), 0, None),
    ("o2_leak, 5000 sim s later: navigation, CANNOT RETURN + refill offer", "take me to the crater rim", ("call_navigation", 0.99), 5000, "o2_leak"),
    ("follow-up 'yes' -> send_message persisted to the messages table", "yes please", ("call_o2", 0.99), 0, None),
    ("o2_leak, 600 sim s later: CRITICAL oxygen alert preempts", "how much oxygen do I have left", ("call_o2", 0.98), 600, None),
]


def db_demo(router, scripted):
    print("\n--- DB-backed demo (FixtureDatasource seeded from schema.sql; no Postgres; PLACEHOLDER definitions in turn_policy.json 'data_source') ---")
    ds = FixtureDatasource()
    handler = TurnHandler(router, ds.snapshot, default_registry(datasource=ds), llm_fn=demo_llm, retriever=demo_retriever)
    for title, text, fake_route, advance_s, scenario in DB_TURNS:
        if scenario:
            ds.mission["scenario"] = scenario  # the leak starts here; earlier rows were normal
        if advance_s:
            ds.advance(advance_s)
        if scripted:
            scripted.next = fake_route
        out = handler.handle(text)
        snap = ds.snapshot()
        call = out["tool_call"] and out["tool_call"]["name"]
        print(f"\n-- {title}: {text!r}")
        print(f"  snapshot: o2={snap['o2_pct']:.1f}% minutes={snap['minutes_remaining']} radiation={snap['radiation_usv_h']:.2f} uSv/h")
        print(f"  turn    : {out['route']}   tool call: {call or 'none'}   verify ok={out['verifier']['ok']}")
        print(f"  spoken  : {out['spoken']}")
    print(f"\n  messages table (fixture): {[m['body'][:60] for m in ds.messages]}")


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
    generation_demo(router, scripted, sim)
    db_demo(router, scripted)
    return 0


if __name__ == "__main__":
    sys.exit(main())

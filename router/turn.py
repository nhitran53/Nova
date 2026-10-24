"""One assistant turn: alert poll -> router -> slot check -> at most one fake tool call -> template -> number verifier.

    TurnHandler(router, alert_source, registry, llm_fn=..., arbiter_fn=...).handle("take me home") -> result dict

Order inside `handle` (the order is the contract; tests pin it):
 1. `alert_source()` is polled FIRST and returns the sim snapshot ({"o2_pct": ..., "bpm": ...}). Any field past a
    CRITICAL threshold in turn_policy.json speaks a fixed template line and returns: the router, the arbiter and the
    LLM are never called, for every class (tool, mid-band, low-confidence, OOD). Critical lines are verified against
    the snapshot. A raising alert_source propagates (a silent failure would hide an alert).
 2. `Pipeline.handle(text)` routes (pipeline_config.json gate: tau_hi 0.90 / tau_lo 0.50, arbiter in the mid band).
 3. Required slots are validated against the registry's fixed schema. Navigation/map without a resolved place
    -> a deterministic slot question, no tool call; two candidate places -> "which place" question.
 4. At most one tool call through the ToolRegistry (max 8 tools, 1 call per turn, schema-checked).
 5. The spoken line is a template (turn_policy.json) filled from the tool_result. Low confidence and OOD get a
    template reply. Only call_llm / NO_TOOL use the injectable `llm_fn(text, prompt_policy)` (default: a stub).
 6. Every spoken line passes `verifier.verify_numbers` against its source (tool_result, sim snapshot for alerts, empty
    for LLM output). A blocked line is replaced by the policy's safe fallback; the original is kept in `blocked_text`.

Follow-up additions (all values PLACEHOLDER in turn_policy.json):
 * Return-to-home feasibility: after an ok navigation tool_result, `assess_feasibility` computes outbound / return /
   required minutes (distance / walking speed, plus a reserve) against the sim's O2 minutes remaining, adds the numbers
   to `tool_result["data"]["feasibility"]` and speaks one outcome line (comfortable / tight / cannot_return /
   cannot_reach) after the route line. It only reports; it never refuses the route (the safety loop owns go/no-go).
 * Refill-call offer: tight / cannot_return / cannot_reach also speak an offer and store `self.pending_confirmation`
   (per session, expires after `confirmation.expiry_turns` turns). On the next turn - after the alert poll, before
   the router - a deterministic yes/no matcher answers it: yes -> ONE send_message call (urgency from the feasibility
   outcome, never the LLM), no -> ack, anything else -> pending cleared and the turn routes normally. A critical
   alert clears it too.
 * LLM scope (decided by router class + retrieval score + risk terms, never by the LLM): NO_TOOL text containing a
   risk term gets a template, no LLM call; otherwise the stub answers ungrounded and is number-verified. call_llm is
   grounded only: `retriever(text) -> [{"text","score"}]`; none above `retrieval_min_score` -> "no document" template,
   no LLM call; else the LLM gets the passages and the verifier also accepts numbers found in them.

Result keys: text, route ("alert"|"tool"|"slot_question"|"clarify"|"llm"|"ood"|"low_confidence"|"risk_diverted"|
"no_document"|"confirmed_yes"|"confirmed_no"), pipeline (the Pipeline.handle dict or None), alerts, tool_call
({"name","args"} or None), tool_result, tool_calls (0 or 1), llm_called, spoken, verifier, blocked_text,
pending_confirmation (the offer stored by THIS turn, or None), passages (grounded call_llm only), tts_error (only when
`tts_fn` raised; `tts_fn(spoken, route)` is called once per turn with the verified/substituted line, after everything else).
All wording lives in turn_policy.json and is PLACEHOLDER pending user sign-off.
"""
import json
import math
import re
from pathlib import Path

from .pipeline import Pipeline, load_config
from .tools import default_registry
from .verifier import verify_numbers

HERE = Path(__file__).parent
POLICY_PATH = HERE / "turn_policy.json"


def load_policy(path=POLICY_PATH):
    """Read turn_policy.json; refuse any section that is not explicitly labelled PLACEHOLDER (the human owns the values)."""
    policy = json.loads(Path(path).read_text())
    for name, section in policy.items():
        if name.startswith("_"):
            continue
        status = section.get("status", "") if isinstance(section, dict) else ""
        if not status.startswith("PLACEHOLDER"):
            raise ValueError(f"{path}: section {name!r} needs a 'status' starting with PLACEHOLDER")
    return policy


def stub_llm(text, prompt_policy, passages=None):
    """Default LLM stand-in: no real call. Replace with an injectable fake or the real small LLM later.
    `passages` (retrieved texts) is passed only on the grounded call_llm path."""
    return f"[LLM stub: {prompt_policy.split(':')[0]}]"


def evaluate_alerts(snapshot, policy):
    """-> [{"field","severity","value"}] for every sim field past a threshold, in policy priority order."""
    alerts = []
    for field, limits in policy["sim_fields"].items():
        if field == "status" or field not in snapshot:
            continue
        value = snapshot[field]
        for severity in ("critical", "warning"):
            below, above = limits.get(f"{severity}_below"), limits.get(f"{severity}_above")
            if (below is not None and value < below) or (above is not None and value > above):
                alerts.append({"field": field, "severity": severity, "value": value})
                break
    return alerts


def assess_feasibility(nav_data, snapshot, policy):
    """Deterministic return-to-home check -> dict of whole-minute numbers (all spoken numbers come from here), or None.

    outbound = ceil(distance / speed); return = the same distance back from the destination (0 when the destination
    is a home place); required = outbound + return + reserve. O2 minutes remaining = snapshot["minutes_remaining"],
    else o2_pct * o2_minutes_per_pct_fallback; None when neither exists (nothing is spoken, never a guess).
    Outcome: cannot_reach (remaining < outbound) | cannot_return (remaining < outbound + return) |
    tight (remaining < required) | comfortable. Rounding is conservative (ceil legs, floor remaining)."""
    cfg = policy["feasibility"]
    if "minutes_remaining" in snapshot:
        remaining = snapshot["minutes_remaining"]
    elif "o2_pct" in snapshot:
        remaining = snapshot["o2_pct"] * cfg["o2_minutes_per_pct_fallback"]
    else:
        return None
    is_home = nav_data["place"] in cfg["home_places"]
    outbound = math.ceil(nav_data["distance_m"] / cfg["walking_speed_m_per_min"] - 1e-9)
    back = 0 if is_home else outbound  # the destination is as far from home as home is from here (placeholder)
    reserve, remaining = cfg["reserve_margin_min"], math.floor(remaining + 1e-9)
    required = outbound + back + reserve
    if remaining < outbound:
        outcome = "cannot_reach"
    elif remaining < outbound + back:
        outcome = "cannot_return"
    elif remaining < required:
        outcome = "tight"
    else:
        outcome = "comfortable"
    return {"outcome": outcome, "is_home": is_home, "outbound_min": outbound, "return_min": back,
            "trip_min": outbound + back, "reserve_min": reserve, "required_min": required,
            "o2_minutes_remaining": remaining}


def match_confirmation(text, policy):
    """-> "yes" | "no" | None. Every word must be a yes/no phrase word or filler; both kinds at once is None."""
    cfg = policy["confirmation"]

    def words(phrase):
        return re.findall(r"[a-z0-9']+", phrase.lower())

    def contains(tokens, phrase):
        n = len(phrase)
        return any(tokens[i:i + n] == phrase for i in range(len(tokens) - n + 1))

    yes = [words(p) for p in cfg["yes_phrases"]]
    no = [words(p) for p in cfg["no_phrases"]]
    vocabulary = {w for phrase in yes + no for w in phrase} | {w for f in cfg["filler_words"] for w in words(f)}
    tokens = words(text)
    if not tokens or any(t not in vocabulary for t in tokens):
        return None
    said_yes, said_no = any(contains(tokens, p) for p in yes), any(contains(tokens, p) for p in no)
    return "yes" if said_yes and not said_no else "no" if said_no and not said_yes else None


class TurnHandler:
    def __init__(self, router, alert_source, registry=None, llm_fn=stub_llm, arbiter_fn=None, policy=None,
                 pipeline_config=None, verifier=verify_numbers, retriever=None, tts_fn=None):
        self.retriever = retriever
        self.tts_fn = tts_fn  # tts_fn(text, route) speaks the final line aloud; None = no speech
        self.turn_index = 0
        self.pending_confirmation = None  # per-session state: {"kind","turn","place","place_name","outcome","feasibility"}
        self.alert_source = alert_source
        self.registry = default_registry() if registry is None else registry
        self.llm_fn = llm_fn
        self.policy = load_policy() if policy is None else policy
        self.verifier = verifier
        self.pipeline = Pipeline(router, config=load_config() if pipeline_config is None else pipeline_config,
                                 llm_fn=lambda text, prompt_policy: None, arbiter_fn=arbiter_fn)

    def handle(self, text):
        """Run one turn (`_run_turn`), then speak `result["spoken"]` once through `tts_fn` on every route.
        A TTS failure never changes the turn: it is recorded as result["tts_error"] ("ExcType: message")."""
        result = self._run_turn(text)
        if self.tts_fn is not None:
            try:
                self.tts_fn(result["spoken"], result["route"])
            except Exception as exc:
                result["tts_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return result

    def _run_turn(self, text):
        self.registry.begin_turn()
        self.turn_index += 1
        snapshot = self.alert_source()
        alerts = evaluate_alerts(snapshot, self.policy)
        result = {"text": text, "route": None, "pipeline": None, "alerts": alerts, "tool_call": None,
                  "tool_result": None, "tool_calls": 0, "llm_called": False, "pending_confirmation": None}
        tiers = self.policy["severity_tiers"]
        critical = [a for a in alerts if tiers[a["severity"]]["preempts_router"]]
        pending, self.pending_confirmation = self.pending_confirmation, None  # cleared by every turn; re-set by an offer
        if critical:
            lines = [self.policy["alert_lines"][a["field"]].format(value=a["value"]) for a in critical]
            return self._speak(result, "alert", " ".join(lines), snapshot)

        if pending is not None and self.turn_index - pending["turn"] <= self.policy["confirmation"]["expiry_turns"]:
            answer = match_confirmation(text, self.policy)
            if answer is not None:
                return self._confirmation_turn(result, pending, answer)

        out = result["pipeline"] = self.pipeline.handle(text)
        if out["route"] == "tool":
            return self._tool_turn(result, out, snapshot)
        if out["route"] == "clarify":
            return self._clarify_turn(result, out)
        if out["reason"] == "OOD":
            return self._speak(result, "ood", self.policy["templates"]["ood"], {})
        if out["reason"] == "low_confidence":
            return self._speak(result, "low_confidence", self.policy["templates"]["low_confidence"], {})
        return self._llm_turn(result, text, out)

    def _llm_turn(self, result, text, out):
        """Scope is decided here by router class (reason), risk terms and retrieval score; never by the LLM."""
        scope = self.policy["llm_scope"]
        if out["reason"] == "NO_TOOL":
            lowered = " ".join(re.findall(r"[a-z0-9']+", text.lower()))
            if any(re.search(rf"\b{re.escape(term)}", lowered) for term in scope["risk_terms"]):
                return self._speak(result, "risk_diverted", scope["risk_diverted"], {})
            result["llm_called"] = True
            return self._speak(result, "llm", self.llm_fn(text, out["prompt_policy"]), {})
        passages = self._retrieve(text, scope)  # call_llm: grounded only
        if not passages:
            return self._speak(result, "no_document", scope["no_document"], {})
        result.update(passages=passages, llm_called=True)
        reply = self.llm_fn(text, out["prompt_policy"], passages=[p["text"] for p in passages])
        return self._speak(result, "llm", reply, [p["text"] for p in passages])

    def _retrieve(self, text, scope):
        """Passages scoring >= retrieval_min_score, best first, at most retrieval_top_k. No retriever or an error -> []."""
        if self.retriever is None:
            return []
        try:
            found = self.retriever(text)
        except Exception:  # a retrieval failure means "no document", never an ungrounded answer
            return []
        kept = [p for p in found if p["score"] >= scope["retrieval_min_score"]]
        return sorted(kept, key=lambda p: -p["score"])[:scope["retrieval_top_k"]]

    def _confirmation_turn(self, result, pending, answer):
        confirmation = self.policy["confirmation"]
        if answer == "no":
            return self._speak(result, "confirmed_no", confirmation["no_spoken"], {})
        feasibility = pending["feasibility"]
        message = self.policy["send_message"]["message"].format(place_name=pending["place_name"], **feasibility)
        if not self.verifier(message, feasibility)["ok"]:  # a message with an untraceable number is never sent
            return self._block(result, "confirmed_yes", message)
        args = {"recipient": pending["place"],
                "urgency": self.policy["send_message"]["urgency_by_outcome"][pending["outcome"]],
                "message": message}
        result["tool_call"] = {"name": "send_message", "args": args}
        tool_result = result["tool_result"] = self.registry.call("send_message", args)
        result["tool_calls"] = self.registry.calls_this_turn
        if not tool_result["ok"]:
            return self._speak(result, "confirmed_yes", self.policy["templates"]["tool_error"], {})
        return self._speak(result, "confirmed_yes", confirmation["yes_spoken"].format(**tool_result["data"]), tool_result)

    def _slot_question(self, result, tool, missing):
        question = self.policy["slot_questions"][tool][missing[0]]
        return self._speak(result, "slot_question", question, {})

    def _clarify_turn(self, result, out):
        if out["reason"] == "missing_slot":
            return self._slot_question(result, out["tool"], out["missing"])
        if out["reason"] == "ambiguous_place":
            options = " or ".join(o.replace("_", " ") for o in out["options"])
            return self._speak(result, "slot_question", self.policy["slot_questions"]["ambiguous_place"].format(options=options), {})
        return self._speak(result, "clarify", out["question"], {})

    def _tool_turn(self, result, out, snapshot):
        tool, args = out["tool"], out["args"]
        missing = [slot for slot in self.registry.schema(tool)["required"] if slot not in args]
        if missing:
            return self._slot_question(result, tool, missing)
        result["tool_call"] = {"name": tool, "args": args}
        tool_result = result["tool_result"] = self.registry.call(tool, args)
        result["tool_calls"] = self.registry.calls_this_turn
        assert result["tool_calls"] <= 1, "more than one tool call in a turn"
        templates = self.policy["templates"]
        if not tool_result["ok"]:
            return self._speak(result, "tool", templates["tool_error"], {})
        offer = None
        try:
            spoken = templates[tool].format(**tool_result["data"])
            if tool == "navigation":
                feasibility = assess_feasibility(tool_result["data"], snapshot, self.policy)
                if feasibility is not None:
                    tool_result = result["tool_result"] = {**tool_result, "data": {**tool_result["data"], "feasibility": feasibility}}
                    key = feasibility["outcome"] + ("_home" if feasibility["is_home"] else "")
                    lines = self.policy["feasibility"]["lines"]
                    spoken += " " + lines.get(key, lines[feasibility["outcome"]]).format(**feasibility)
                    if feasibility["outcome"] in self.policy["send_message"]["urgency_by_outcome"] and not feasibility["is_home"]:  # no refill call to the home base itself
                        offer = {"kind": "refill_call", "turn": self.turn_index, "place": tool_result["data"]["place"],
                                 "place_name": tool_result["data"]["place_name"], "outcome": feasibility["outcome"],
                                 "feasibility": feasibility}
                        spoken += " " + self.policy["confirmation"]["offer"].format(place_name=offer["place_name"])
        except (KeyError, IndexError, ValueError):  # a template that does not fit the tool_result is never spoken
            return self._block(result, "tool", f"[template {tool!r} does not fit tool_result]")
        result = self._speak(result, "tool", spoken, tool_result)
        if offer is not None and result["blocked_text"] is None:  # an offer that was not spoken is not pending
            self.pending_confirmation = offer
            result["pending_confirmation"] = dict(offer)
        return result

    def _speak(self, result, route, line, source):
        result["route"] = route
        verdict = self.verifier(line, source)
        if not verdict["ok"]:
            return self._block(result, route, line, verdict)
        result.update(spoken=line, verifier=verdict, blocked_text=None)
        return result

    def _block(self, result, route, line, verdict=None):
        result["route"] = route
        result.update(spoken=self.policy["templates"]["verifier_blocked"], blocked_text=line,
                      verifier=verdict or {"ok": False, "spoken": [], "unsupported": []})
        return result

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
   outcome, never the LLM; recomputed from THIS turn's snapshot - if no refill is needed any more nothing is sent and
   a fixed line is spoken), no -> ack, anything else -> pending cleared and the turn routes normally. A critical
   alert clears it too.
 * LLM scope (decided by router class + retrieval score + risk terms, never by the LLM): NO_TOOL text containing a
   risk term gets a template, no LLM call; otherwise the stub answers ungrounded and is number-verified. call_llm is
   grounded only: `retriever(text) -> [{"text","score"}]`; none above `retrieval_min_score` -> "no document" template,
   no LLM call; else the LLM gets the passages and the verifier also accepts numbers found in them.

Post-tool-call generation (router/generation.py; wording in turn_policy.json "generation", all PLACEHOLDER): the LLM
paths (NO_TOOL chitchat, call_llm) go build_request -> Generator -> postprocess (parse, slot check, prose-number scan,
length, deny-lists, slot fill, verify_numbers backstop) -> `_speak`. `TurnHandler(generator=...)` injects a Generator
(default FillerGenerator); the old `llm_fn` is wrapped by LegacyLLMAdapter. Tool and confirmation turns still speak their
template (llm_called stays False) but also expose the request and rendered prompt as result["generation"] with mode
"template_only". Alerts build no request and never reach a generator. result["generation"] (additive) = {"mode":
"template_only"|"generated", "request" (summary), "prompt" (chat messages), and for generated turns
"status", "attempts", "violations", "attempt_log", "raw", "text" (final spoken text, attribution included), "refusal",
"passage_numbers" (EFFECTIVE mode), "configured_passage_numbers", "passage_override_reason" (a risk term in the question or a
passage forces "off" for a grounded turn), "attribution_prefixed"}. The whole dict is JSON-serializable; the live
GenerationRequest object of the latest request is `TurnHandler.last_generation_request` (None when a turn built none).

Result keys: text, route ("alert"|"tool"|"slot_question"|"clarify"|"llm"|"ood"|"low_confidence"|"risk_diverted"|
"no_document"|"confirmed_yes"|"confirmed_no"|"refill_not_needed"), pipeline (the Pipeline.handle dict or None), alerts, tool_call
({"name","args"} or None), tool_result, tool_calls (0 or 1), llm_called, spoken, verifier, blocked_text,
pending_confirmation (the offer stored by THIS turn, or None), tool_error / llm_error (only when the tool or the LLM
raised: templates.tool_error is spoken, nothing is retried; an alert_source failure still propagates), passages (grounded call_llm only), tts_error (only when
`tts_fn` raised; `tts_fn(spoken, route)` is called once per turn with the verified/substituted line, after everything else).
All wording lives in turn_policy.json and is PLACEHOLDER pending user sign-off.
"""
import json
import math
import re
from dataclasses import replace
from pathlib import Path

from .generation import (NUMBER_VIOLATIONS, FillerGenerator, LegacyLLMAdapter, attribute_passage_numbers, build_request,
                         render_prompt, risk_terms_in, run_generation, validate_generation_policy)
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
    for tier, flags in policy["severity_tiers"].items():
        if isinstance(flags, dict) and flags.get("preempts_router") and flags.get("uses_llm"):
            raise ValueError(f"{path}: severity tier {tier!r} preempts the router, so it cannot use the LLM")
    validate_generation_policy(policy.get("generation"), path)
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
                 pipeline_config=None, verifier=verify_numbers, retriever=None, tts_fn=None, generator=None):
        self.retriever = retriever
        self.last_generation_request = None  # the GenerationRequest object of the latest request (not part of the result dict)
        self.history = []  # (user_text, spoken) of spoken LLM turns; only used when generation.history_turns > 0
        self.tts_fn = tts_fn  # tts_fn(text, route) speaks the final line aloud; None = no speech
        self.turn_index = 0
        self.pending_confirmation = None  # per-session state: {"kind","turn","place","place_name","distance_m","outcome","feasibility"}
        self.alert_source = alert_source
        self.registry = default_registry() if registry is None else registry
        self.llm_fn = llm_fn
        self.policy = load_policy() if policy is None else policy
        if generator is not None:
            self.generator = generator
        elif llm_fn is stub_llm:
            self.generator = FillerGenerator()
        else:
            self.generator = LegacyLLMAdapter(llm_fn)  # old llm_fn(text, prompt_policy, passages=None) -> str
        self.verifier = verifier
        self.pipeline = Pipeline(router, config=load_config() if pipeline_config is None else pipeline_config,
                                 llm_fn=lambda text, prompt_policy: None, arbiter_fn=arbiter_fn)

    def handle(self, text):
        """Run one turn (`_run_turn`), then speak `result["spoken"]` once through `tts_fn` on every route.
        A TTS failure never changes the turn: it is recorded as result["tts_error"] ("ExcType: message")."""
        self.last_generation_request = None
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
                return self._confirmation_turn(result, pending, answer, snapshot)

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
            if risk_terms_in(text, scope["risk_terms"]):
                return self._speak(result, "risk_diverted", scope["risk_diverted"], {})
            return self._llm_reply(result, "chitchat", text, out["prompt_policy"])
        passages = self._retrieve(text, scope)  # call_llm: grounded only
        if not passages:
            return self._speak(result, "no_document", scope["no_document"], {})
        result["passages"] = passages
        return self._llm_reply(result, "grounded_answer", text, out["prompt_policy"], passages=passages)

    def _llm_reply(self, result, kind, text, prompt_policy, passages=()):
        """The one generation: build_request -> generator (retry/fallback in run_generation) -> _speak.
        A raising generator is recorded as result["llm_error"] and the fixed tool_error line is spoken, never retried.
        Failed generations keep the model's text in blocked_text and speak (through _speak, so the replacement is verified too) templates.verifier_blocked (grounded answers, and
        any chitchat rejected for a number) or generation.neutral_chitchat (other chitchat rejections). The final verifier source is the same as before: the retrieved passages
        under passage_numbers="digits", nothing otherwise."""
        gen, templates = self.policy["generation"], self.policy["templates"]
        result["llm_called"] = True
        keep = gen["history_turns"]
        req = build_request(turn_id=self.turn_index, kind=kind, route="llm", user_text=text, policy=self.policy,
                            passages=passages, history=self.history[-keep:] if keep else (), prompt_policy=prompt_policy)
        source = [p.text for p in req.passages] if req.passage_numbers == "digits" else {}
        self.last_generation_request = req
        outcome = run_generation(req, self.generator, self.policy, gen["neutral_chitchat"], error_text=templates["tool_error"],
                                 verifier=self.verifier)
        if outcome.status == "fallback" and (kind == "grounded_answer" or NUMBER_VIOLATIONS & set(outcome.violations)):
            outcome = replace(outcome, text=templates["verifier_blocked"])  # number problems keep the old blocked line
        prefixed = False
        if outcome.status == "ok":
            attributed, prefixed = attribute_passage_numbers(req, outcome.text, gen)
            outcome = replace(outcome, text=attributed)
        result["generation"] = {"mode": "generated", "request": req.summary(), "prompt": render_prompt(req),
                                "status": outcome.status, "attempts": outcome.attempts, "violations": list(outcome.violations),
                                "attempt_log": list(outcome.attempt_log), "raw": outcome.raw, "text": outcome.text,
                                "refusal": outcome.refusal, "passage_numbers": req.passage_numbers,
                                "configured_passage_numbers": req.configured_passage_numbers,
                                "passage_override_reason": req.passage_override_reason, "attribution_prefixed": prefixed}
        if outcome.status == "error":
            result["llm_error"] = outcome.error
            return self._speak(result, "llm", outcome.text, {})
        if outcome.status == "fallback":
            failed = outcome.blocked_text
            verdict = outcome.verdict
            if verdict is None:
                try:
                    verdict = self.verifier(failed, source)
                except Exception:
                    verdict = None
            if verdict is None or verdict["ok"]:  # rejected for a non-number reason: still a blocked line
                verdict = {"ok": False, "spoken": (verdict or {}).get("spoken", []), "unsupported": []}
            rejected = verdict
            result = self._speak(result, "llm", outcome.text, source)  # the replacement is a spoken line: final gate too
            replacement_verdict = result["verifier"]
            if result["blocked_text"] is not None:  # even the fixed replacement was refused by the verifier
                result["generation"]["replacement_blocked_text"] = result["blocked_text"]
            else:
                result["verifier"] = rejected  # as before: the verdict that rejected the model text
            result["generation"]["replacement_verifier"] = replacement_verdict
            result["blocked_text"] = failed
            return result
        result = self._speak(result, "llm", outcome.text, source)
        if keep and result["blocked_text"] is None:
            self.history.append((text, result["spoken"]))
            del self.history[:-keep]  # only the last `keep` turns can ever be sent; do not retain older utterances
        return result

    def _retrieve(self, text, scope):
        """Passages scoring >= retrieval_min_score, best first, at most retrieval_top_k. No retriever, None, an error or
        any malformed passage (no str "text", non-numeric or NaN "score") -> [] ("no document", never an ungrounded answer)."""
        if self.retriever is None:
            return []
        try:
            found = self.retriever(text)
            if found is None:
                return []
            found = list(found)
            for p in found:
                score = p["score"]
                if not isinstance(p["text"], str) or isinstance(score, bool) or not isinstance(score, (int, float)) or math.isnan(score):
                    return []
            kept = [p for p in found if p["score"] >= scope["retrieval_min_score"]]
            return sorted(kept, key=lambda p: -p["score"])[:scope["retrieval_top_k"]]
        except Exception:
            return []

    def _confirmation_turn(self, result, pending, answer, snapshot):
        """yes -> feasibility is recomputed from THIS turn's snapshot (the offer's numbers may be stale); the message and
        urgency come from the fresh outcome. If no refill is needed any more, nothing is sent and a fixed line is spoken."""
        confirmation = self.policy["confirmation"]
        if answer == "no":
            return self._speak(result, "confirmed_no", confirmation["no_spoken"], {})
        send = self.policy["send_message"]
        feasibility = assess_feasibility({"place": pending["place"], "distance_m": pending["distance_m"]}, snapshot, self.policy)
        if feasibility is None or feasibility["outcome"] not in send["urgency_by_outcome"]:
            return self._speak(result, "refill_not_needed", confirmation["no_longer_needed"], {})
        message = send["message"].format(place_name=pending["place_name"], **feasibility)
        if not self.verifier(message, feasibility)["ok"]:  # a message with an untraceable number is never sent
            return self._block(result, "confirmed_yes", message)
        args = {"recipient": pending["place"], "urgency": send["urgency_by_outcome"][feasibility["outcome"]],
                "message": message}
        tool_result = self._call_tool(result, "send_message", args)
        if tool_result is None or not tool_result["ok"]:
            return self._speak(result, "confirmed_yes", self.policy["templates"]["tool_error"], {})
        self._expose_request(result, "confirmation", "confirmed_yes", "", "send_message", tool_result)
        return self._speak(result, "confirmed_yes", confirmation["yes_spoken"].format(**tool_result["data"]), tool_result)

    def _expose_request(self, result, kind, route, prompt_policy, tool, tool_result):
        """Tool and confirmation turns speak their template (the LLM is not called, llm_called stays False), but the
        structured request and rendered prompt are still built so the "model space" can see them."""
        try:
            req = build_request(turn_id=self.turn_index, kind=kind, route=route, user_text=result["text"], policy=self.policy,
                                tool=tool, tool_result=tool_result, prompt_policy=prompt_policy)
            self.last_generation_request = req
            result["generation"] = {"mode": "template_only", "request": req.summary(), "prompt": render_prompt(req)}
        except Exception as exc:  # exposing the request must never change what is spoken
            result["generation"] = {"mode": "template_only", "error": f"{type(exc).__name__}: {exc}"[:300]}

    def _call_tool(self, result, name, args):
        """The one registry call of the turn. A raising tool is recorded as result["tool_error"] and returns None
        (the caller speaks templates.tool_error); the call is never retried."""
        result["tool_call"] = {"name": name, "args": args}
        try:
            tool_result = self.registry.call(name, args)
        except Exception as exc:
            result["tool_error"] = f"{type(exc).__name__}: {exc}"[:300]
            tool_result = None
        result["tool_result"] = tool_result
        result["tool_calls"] = self.registry.calls_this_turn
        assert result["tool_calls"] <= 1, "more than one tool call in a turn"
        return tool_result

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
        tool_result = self._call_tool(result, tool, args)
        templates = self.policy["templates"]
        if tool_result is None or not tool_result["ok"]:
            return self._speak(result, "tool", templates["tool_error"], {})
        offer = None
        try:
            spoken = templates[tool].format(**tool_result["data"])
            if tool == "navigation":
                tool_result, suffix, offer = self._navigation_feasibility(tool_result, snapshot)
                result["tool_result"] = tool_result
                spoken += suffix
        except (KeyError, IndexError, ValueError):  # a template that does not fit the tool_result is never spoken
            return self._block(result, "tool", f"[template {tool!r} does not fit tool_result]")
        self._expose_request(result, "tool_result", "tool", "", tool, tool_result)
        result = self._speak(result, "tool", spoken, tool_result)
        if offer is not None and result["blocked_text"] is None:  # an offer that was not spoken is not pending
            self.pending_confirmation = offer
            result["pending_confirmation"] = dict(offer)
        return result

    def _navigation_feasibility(self, tool_result, snapshot):
        """-> (tool_result with data["feasibility"] added, text to append to the route line, refill-call offer or None).
        Unchanged tool_result and no text when there is no O2 data. No offer for a home place or a comfortable trip."""
        feasibility = assess_feasibility(tool_result["data"], snapshot, self.policy)
        if feasibility is None:
            return tool_result, "", None
        data = tool_result["data"]
        tool_result = {**tool_result, "data": {**data, "feasibility": feasibility}}
        key = feasibility["outcome"] + ("_home" if feasibility["is_home"] else "")
        lines = self.policy["feasibility"]["lines"]
        suffix = " " + lines.get(key, lines[feasibility["outcome"]]).format(**feasibility)
        offer = None
        if feasibility["outcome"] in self.policy["send_message"]["urgency_by_outcome"] and not feasibility["is_home"]:  # no refill call to the home base itself
            offer = {"kind": "refill_call", "turn": self.turn_index, "place": data["place"], "place_name": data["place_name"],
                     "distance_m": data["distance_m"], "outcome": feasibility["outcome"], "feasibility": feasibility}
            suffix += " " + self.policy["confirmation"]["offer"].format(place_name=offer["place_name"])
        return tool_result, suffix, offer

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

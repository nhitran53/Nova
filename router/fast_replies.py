"""Policy-driven fixed replies for short whole-utterance phrases (greeting, identity): no router, no tool, no LLM."""
import re

from .clarification import normalized

_CONTRACTION = re.compile(r"\b(what|who|where) s\b|\b(what|who|where)s\b")


def canonical(text):
    """The turn normalizer (lowercase, punctuation and apostrophes dropped) with "what's"/"whats" == "what is"."""
    return _CONTRACTION.sub(lambda m: f"{m.group(1) or m.group(2)} is", normalized(text))


def validate_fast_replies(section, where="turn_policy.json"):
    """Raise ValueError for a fast_replies section the code cannot trust. Lines pass through _speak, so they carry no digits."""
    if not isinstance(section, dict):
        raise ValueError(f"{where}: 'fast_replies' must be an object")
    if not isinstance(section.get("enabled"), bool):
        raise ValueError(f"{where}: fast_replies.enabled must be a boolean")
    prefixes = section.get("wake_prefixes")
    if not isinstance(prefixes, list) or not all(isinstance(p, str) and canonical(p) for p in prefixes):
        raise ValueError(f"{where}: fast_replies.wake_prefixes must be a list of non-empty strings")
    intents = section.get("intents")
    if not isinstance(intents, dict) or not intents:
        raise ValueError(f"{where}: fast_replies.intents must be a non-empty object")
    seen = {}
    for name, intent in intents.items():
        phrases = intent.get("phrases") if isinstance(intent, dict) else None
        if not isinstance(phrases, list) or not phrases or not all(isinstance(p, str) and canonical(p) for p in phrases):
            raise ValueError(f"{where}: fast_replies.intents.{name}.phrases must be a non-empty list of non-empty strings")
        line = intent.get("line")
        if not isinstance(line, str) or not line.strip() or any(c.isdigit() for c in line):
            raise ValueError(f"{where}: fast_replies.intents.{name}.line must be a non-empty string without digits")
        for phrase in phrases:
            if seen.setdefault(canonical(phrase), name) != name:
                raise ValueError(f"{where}: fast_replies phrase {phrase!r} appears under two intents")


def match_fast_reply(text, policy):
    """-> (intent, line) | None. The WHOLE utterance must be an intent phrase, tried as spoken first and then after
    stripping one leading wake prefix; "hello how much oxygen do I have" never matches."""
    section = policy.get("fast_replies")
    if not section or not section["enabled"]:
        return None
    lookup = {canonical(p): (name, intent["line"]) for name, intent in section["intents"].items() for p in intent["phrases"]}
    spoken = canonical(text)
    if spoken in lookup:
        return lookup[spoken]
    for prefix in sorted((canonical(p) for p in section["wake_prefixes"]), key=len, reverse=True):
        if spoken.startswith(prefix + " ") and spoken[len(prefix) + 1:] in lookup:
            return lookup[spoken[len(prefix) + 1:]]
    return None

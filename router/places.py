"""Deterministic place resolver: free text -> canonical place id from the gazetteer (places.json).

    resolve_place("take me to home")        -> {"place_id": "homebase", "matched_text": "home", "score": 1.0}
    resolve_place("directions to homebse")  -> {"place_id": "homebase", "matched_text": "homebse", "score": 0.93...}
    resolve_place("what time is it")        -> None

Method: normalise (lowercase, strip punctuation), slide 1..N word windows over the utterance (N = longest alias
in words + 1), compare each window with every alias on a space-free form with difflib.SequenceMatcher (so
"home base" == "homebase"), keep the best score >= threshold. Windows containing a stopword ("to", "the",
"my", "back", ...) are skipped, so function words never match. A non-exact match must also share its first
letter with the alias (stops "over" -> "rover", "bridge" -> "ridge"). Ties on score go to the longer window,
then the earlier position, then the alphabetically first place id, so results are deterministic.

`resolve_destination(text)` is what the pipeline uses for navigation/map: first split the text into
role-bearing spans at destination cues (to/toward/back to), origin cues (from), negation, and new navigation
commands. Then fuzzy-match each span separately, excluding origins and negated places.
One distinct candidate place -> resolved; several -> status "ambiguous" (the pipeline asks a slot question; it never
picks by position); none -> status "none". `resolve_place` (best single match, no roles) is kept for simple lookups.

    resolve_destination("go from the rover to the airlock")  -> resolved airlock (rover excluded as origin)
    resolve_destination("take me to the rover not home")     -> resolved rover (home excluded as negated)
    resolve_destination("go to the rover or the airlock")    -> ambiguous: rover, airlock
"""
import json
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

PLACES_PATH = Path(__file__).parent / "places.json"
DEFAULT_THRESHOLD = 0.85
STOPWORDS = frozenset("""
a an the to too two of at in on for from by with into onto toward towards back over there here this that
me my our your us we i you it its is are am be please just now then and or but so
go going goto take bring get lead guide walk head navigate route show way how what where which who
do does can could would will should some any
""".split())


def normalise(text):
    """Lowercase, drop apostrophes, replace other punctuation with spaces, collapse whitespace."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower().replace("'", "")).split())


def load_gazetteer(path=PLACES_PATH):
    """-> {place_id: [normalised aliases]} (id is an implicit alias). Rejects duplicate or stopword aliases."""
    raw = json.loads(Path(path).read_text())["places"]
    gazetteer, owner = {}, {}
    for place_id, aliases in raw.items():
        names = {normalise(place_id.replace("_", " "))} | {normalise(a) for a in aliases}
        for name in sorted(names):
            if not name or set(name.split()) & STOPWORDS:
                raise ValueError(f"{path}: alias {name!r} of {place_id} is empty or contains a stopword")
            if owner.setdefault(name, place_id) != place_id:
                raise ValueError(f"{path}: alias {name!r} belongs to both {owner[name]} and {place_id}")
        gazetteer[place_id] = sorted(names)
    return gazetteer


@lru_cache(maxsize=1)
def _default_gazetteer():
    return load_gazetteer()


def _score(window, alias):
    """Similarity of two space-free strings; 0 if a fuzzy match changes the first letter."""
    a, b = window.replace(" ", ""), alias.replace(" ", "")
    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio if ratio == 1.0 or a[0] == b[0] else 0.0


ORIGIN_WORDS = frozenset({"from"})
NEGATION_WORDS = frozenset({"not", "no", "never", "except", "without", "instead", "rather"})
DESTINATION_WORDS = frozenset({"to", "toward", "towards", "into", "onto"})
NAVIGATION_WORDS = frozenset({"take", "bring", "lead", "guide", "walk", "head", "go", "navigate", "route"})


def _matches(tokens, threshold, gazetteer):
    """Non-overlapping gazetteer matches [{"place_id","matched_text","score","start","end"}] in text order.

    Every window scoring >= threshold is a candidate; they are taken best first (score, longer window, earlier,
    place id), skipping windows that overlap an already-taken one ("home base" beats the inner "home"/"base")."""
    max_words = max(len(alias.split()) for aliases in gazetteer.values() for alias in aliases) + 1
    found = []
    for start in range(len(tokens)):
        for size in range(1, max_words + 1):
            words = tokens[start:start + size]
            if len(words) < size or set(words) & STOPWORDS:
                continue
            window = " ".join(words)
            scored = [(-max(_score(window, alias) for alias in aliases), place_id) for place_id, aliases in gazetteer.items()]
            neg_score, place_id = min(scored)
            if -neg_score >= threshold:
                found.append((neg_score, -size, start, place_id, window))
    taken, used = [], set()
    for neg_score, neg_size, start, place_id, window in sorted(found):
        span = set(range(start, start - neg_size))
        if span & used:
            continue
        used |= span
        taken.append({"place_id": place_id, "matched_text": window, "score": -neg_score, "start": start,
                      "end": start - neg_size})
    return sorted(taken, key=lambda m: m["start"])


def _destination_spans(tokens):
    """Extract role-bearing spans BEFORE fuzzy matching, including when an origin is unknown.

    A new navigation command or contrast ("but") resets the role. A destination preposition resets
    an origin, but not a negation ("not to home"). Unmarked text is a destination candidate.
    This is a small deterministic grammar, not a general language parser.
    """
    role, start = "destination", 0
    for i, token in enumerate(tokens):
        if token in NEGATION_WORDS:
            next_role = "negated"
        elif token in ORIGIN_WORDS:
            next_role = "origin" if role != "negated" else role
        elif token in NAVIGATION_WORDS or token == "but":
            next_role = "destination"
        elif token in DESTINATION_WORDS:
            next_role = "destination" if role != "negated" else role
        else:
            continue
        if start < i:
            yield role, tokens[start:i]
        role, start = next_role, i + 1
    if start < len(tokens):
        yield role, tokens[start:]


def resolve_destination(text, threshold=DEFAULT_THRESHOLD, gazetteer=None):
    """Destination-aware resolution (see module docstring).

    -> {"status": "resolved"|"ambiguous"|"none",
        "place": {"place_id","matched_text","score"} | None,      # set only when resolved
        "candidates": [{"place_id","matched_text","score"}],      # distinct destination candidates, text order
        "excluded": [{"place_id","matched_text","score","role"}]}  # origins and negated places
    """
    gazetteer = _default_gazetteer() if gazetteer is None else gazetteer
    tokens = normalise(text).split()
    matches = [(role, match) for role, span in _destination_spans(tokens)
               for match in _matches(span, threshold, gazetteer)]
    candidates, excluded = [], []
    for role, match in matches:
        public = {k: match[k] for k in ("place_id", "matched_text", "score")}
        if role == "destination":
            if all(c["place_id"] != match["place_id"] for c in candidates):  # "home base" and "base" = one place
                candidates.append(public)
        else:
            excluded.append({**public, "role": role})
    status = "none" if not candidates else "resolved" if len(candidates) == 1 else "ambiguous"
    return {"status": status, "place": candidates[0] if status == "resolved" else None,
            "candidates": candidates, "excluded": excluded}


def resolve_place(text, threshold=DEFAULT_THRESHOLD, gazetteer=None):
    """Best gazetteer match in `text` -> {"place_id", "matched_text", "score"}, or None if nothing reaches threshold."""
    gazetteer = _default_gazetteer() if gazetteer is None else gazetteer
    tokens = normalise(text).split()
    max_words = max(len(alias.split()) for aliases in gazetteer.values() for alias in aliases) + 1
    best, best_key = None, None
    for start in range(len(tokens)):
        for size in range(1, max_words + 1):
            words = tokens[start:start + size]
            if len(words) < size or set(words) & STOPWORDS:
                continue
            window = " ".join(words)
            for place_id, aliases in gazetteer.items():
                score = max(_score(window, alias) for alias in aliases)
                if score < threshold:
                    continue
                key = (-score, -size, start, place_id)  # smaller wins: score, then longer window, earlier, id
                if best_key is None or key < best_key:
                    best_key, best = key, {"place_id": place_id, "matched_text": window, "score": score}
    return best

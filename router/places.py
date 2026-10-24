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

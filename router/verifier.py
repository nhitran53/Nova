"""Number verifier (simple working version): every number the assistant speaks must trace to a source dict.

    verify_numbers("Oxygen at 87.5 percent.", {"o2_pct": 87.5})  -> {"ok": True, "spoken": [87.5], "unsupported": []}
    verify_numbers("Oxygen at 99 percent.",   {"o2_pct": 87.5})  -> {"ok": False, "spoken": [99.0], "unsupported": [99.0]}

Spoken numbers are digit runs ("87.5", "1,200") and spelled-out numbers ("nine", "twenty one", "two hundred"),
signed when a "-" or "\u2212" starts the token (not preceded by a word character, digit or "."; "5-10" is 5 and 10) or
when "minus" precedes them ("minus twenty" is -20). The sign is kept: -5 is not 5.
Allowed numbers are every int/float (bool excluded) anywhere in `source`, plus digit runs inside its strings.
A spoken number matches an allowed one if equal, or if it equals the allowed value rounded to the spoken number's
decimal places (so 20.94 may be spoken "20.9" or "21", but 20.94 never matches "22" or "20.8").
Fail closed on the vague quantity words in VAGUE_QUANTITY_WORDS ("half", "quarter", "third", "dozen", "couple" and
plurals): they are unsupported unless the same word appears in a string inside `source`.
Not built: ordinals, other fractions, times/dates, "few"/"point", and any semantic check beyond the numbers.
"""
import re

DIGITS = re.compile(r"(?:(?<![\w.])[-\u2212])?\d[\d,]*(?:\.\d+)?")
MINUS_BEFORE_DIGIT = re.compile(r"\bminus\s+(?=\d)")
VAGUE_QUANTITY_WORDS = {"half", "halves", "quarter", "quarters", "third", "thirds", "dozen", "dozens", "couple", "couples"}
UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
SCALES = {"hundred": 100, "thousand": 1000}
NUMBER_WORDS = set(UNITS) | set(TENS) | set(SCALES)


def _normalise(text):
    return MINUS_BEFORE_DIGIT.sub("-", text.lower())


def _digit_values(text):
    return [(float(m.group().replace(",", "").replace("\u2212", "-")), len(m.group().partition(".")[2]))
            for m in DIGITS.finditer(_normalise(text))]


def _word_values(text):
    """-> [(value, 0)] for each run of number words, e.g. 'twenty one' -> 21, 'two hundred' -> 200, 'minus ten' -> -10."""
    values, run, negative, previous = [], [], False, ""
    for token in re.findall(r"[a-z]+", DIGITS.sub(" ", _normalise(text))) + [""]:
        if token in NUMBER_WORDS or (run and token == "and"):
            if not run:
                negative = previous == "minus"
            run.append(token)
            previous = token
            continue
        previous = token
        words = [w for w in run if w != "and"]
        run = []
        if words:
            total = current = 0
            for w in words:
                if w in SCALES:
                    current = max(current, 1) * SCALES[w]
                    if w == "thousand":
                        total, current = total + current, 0
                else:
                    current += UNITS.get(w, TENS.get(w, 0))
            values.append((float(-(total + current) if negative else total + current), 0))
    return values


def spoken_numbers(text):
    """[(value, decimals)] for every number spoken in `text`."""
    return _digit_values(text) + _word_values(text)


def allowed_numbers(source):
    """Every number found anywhere in `source` (dict/list/scalars)."""
    if isinstance(source, bool) or source is None:
        return []
    if isinstance(source, (int, float)):
        return [float(source)]
    if isinstance(source, str):
        return [v for v, _ in _digit_values(source)]
    items = source.values() if isinstance(source, dict) else source
    return [n for item in items for n in allowed_numbers(item)]


def source_words(source):
    """Lowercase words found in the strings anywhere in `source`."""
    if isinstance(source, str):
        return set(re.findall(r"[a-z]+", source.lower()))
    if isinstance(source, dict):
        source = list(source.values())
    if isinstance(source, (list, tuple, set)):
        return {w for item in source for w in source_words(item)}
    return set()


def verify_numbers(text, source):
    allowed = allowed_numbers(source)
    spoken, unsupported = [], []
    backed = source_words(source)
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in VAGUE_QUANTITY_WORDS and word not in backed:
            unsupported.append(word)
    for value, decimals in spoken_numbers(text):
        spoken.append(value)
        if not any(abs(a - value) < 1e-9 or abs(round(a, decimals) - value) < 1e-9 for a in allowed):
            unsupported.append(value)
    return {"ok": not unsupported, "spoken": spoken, "unsupported": unsupported}

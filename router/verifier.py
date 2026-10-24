"""Number verifier (simple working version): every number the assistant speaks must trace to a source dict.

    verify_numbers("Oxygen at 87.5 percent.", {"o2_pct": 87.5})  -> {"ok": True, "spoken": [87.5], "unsupported": []}
    verify_numbers("Oxygen at 99 percent.",   {"o2_pct": 87.5})  -> {"ok": False, "spoken": [99.0], "unsupported": [99.0]}

Spoken numbers are digit runs ("87.5", "1,200") and spelled-out numbers ("nine", "twenty one", "two hundred").
Allowed numbers are every int/float (bool excluded) anywhere in `source`, plus digit runs inside its strings.
A spoken number matches an allowed one if equal, or if it equals the allowed value rounded to the spoken number's
decimal places (so 20.94 may be spoken "20.9" or "21", but 20.94 never matches "22" or "20.8").
Not built: ordinals, fractions, times/dates, "a couple", and any semantic check beyond the numbers.
"""
import re

DIGITS = re.compile(r"\d[\d,]*(?:\.\d+)?")
UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
SCALES = {"hundred": 100, "thousand": 1000}
NUMBER_WORDS = set(UNITS) | set(TENS) | set(SCALES)


def _digit_values(text):
    return [(float(m.group().replace(",", "")), len(m.group().partition(".")[2])) for m in DIGITS.finditer(text)]


def _word_values(text):
    """-> [(value, 0)] for each run of number words, e.g. 'twenty one' -> 21, 'two hundred' -> 200."""
    values, run = [], []
    for token in re.findall(r"[a-z]+", DIGITS.sub(" ", text.lower())) + [""]:
        if token in NUMBER_WORDS or (run and token == "and"):
            run.append(token)
            continue
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
            values.append((float(total + current), 0))
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


def verify_numbers(text, source):
    allowed = allowed_numbers(source)
    spoken, unsupported = [], []
    for value, decimals in spoken_numbers(text):
        spoken.append(value)
        if not any(abs(a - value) < 1e-9 or abs(round(a, decimals) - value) < 1e-9 for a in allowed):
            unsupported.append(value)
    return {"ok": not unsupported, "spoken": spoken, "unsupported": unsupported}

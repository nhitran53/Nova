"""Number verifier (simple working version): every number the assistant speaks must trace to a source dict.

    verify_numbers("Oxygen at 87.5 percent.", {"o2_pct": 87.5})  -> {"ok": True, "spoken": [87.5], "unsupported": []}
    verify_numbers("Oxygen at 99 percent.",   {"o2_pct": 87.5})  -> {"ok": False, "spoken": [99.0], "unsupported": [99.0]}

Spoken numbers are digit runs ("87.5", "1,200") and spelled-out numbers ("nine", "twenty one", "two hundred"),
signed when a "-" or "\u2212" starts the token (not preceded by a word character, digit or "."; "5-10" is 5 and 10) or
when "minus" precedes them ("minus twenty" is -20). The sign is kept: -5 is not 5.
Allowed numbers are every int/float (bool excluded) anywhere in `source`, plus digit runs inside its strings.
A spoken number matches an allowed one if equal, or if it equals the allowed value rounded to the spoken number's
decimal places (so 20.94 may be spoken "20.9" or "21", but 20.94 never matches "22" or "20.8").
Scale words go up to "trillion" ("two million five hundred thousand" is 2500000).
Fail closed on the vague quantity words in VAGUE_QUANTITY_WORDS ("half", "quarter", "third", "dozen", "couple" and
plurals, "twice", "thrice", "several", the plural scales "hundreds" ... "trillions", and the spelled ordinals "fourth" ... "trillionth",
ORDINAL_WORDS; "first" and "second" stay allowed on purpose): they are unsupported unless the same word appears in a string inside `source`.
Text is first folded by fold_text: format characters (category Cf, e.g. zero-width spaces) and invisible combining marks (U+034F,
variation selectors) are removed, NFKC applied (fullwidth, superscript and circled digits become ASCII digits), and any remaining non-ASCII
Nd/No/Nl character becomes its digit or numeric value, so "\u00b2", "\u246b" (circled 12), "\u0663" and "fi\u200bve" / "fi\u034fve" are
read as 2, 12, 3 and "five".
Fractions fail closed: a vulgar fraction ("\u00bd", any non-integer unicodedata.numeric character) or a fraction slash (U+2044, with the
digits next to it) is replaced by FRACTION_MARK BEFORE NFKC could split it into two numbers. In spoken text it is an unsupported number
("fraction" in `unsupported`); in a source it authorizes nothing (not "1", not "2", not 0.5).
Not built: "first"/"second" ordinals, fraction values, times/dates, "few"/"point", and any semantic check beyond the numbers.
"""
import re
import unicodedata

DIGITS = re.compile(r"(?:(?<![\w.])[-\u2212])?\d[\d,]*(?:\.\d+)?")
MINUS_BEFORE_DIGIT = re.compile(r"\bminus\s+(?=\d)")
# Spelled ordinals from "fourth" up (plus plural fractions such as "fifths"). "first" and "second" are deliberately NOT here: they are common
# prose and "second" is also a unit of time. ("third" is already a vague quantity word.) Compounds ("twenty-first") contain a tens word anyway.
_ORDINAL_STEMS = ("fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth "
                  "eighteenth nineteenth twentieth thirtieth fortieth fiftieth sixtieth seventieth eightieth ninetieth hundredth thousandth "
                  "millionth billionth trillionth").split()
ORDINAL_WORDS = frozenset(_ORDINAL_STEMS) | frozenset(w + "s" for w in _ORDINAL_STEMS)
VAGUE_QUANTITY_WORDS = ORDINAL_WORDS | {"half", "halves", "quarter", "quarters", "third", "thirds", "dozen", "dozens", "couple", "couples",
                        "twice", "thrice", "several", "hundreds", "thousands", "millions", "billions", "trillions"}
UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
SCALES = {"hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9, "trillion": 10**12}
NUMBER_WORDS = set(UNITS) | set(TENS) | set(SCALES)


NUMERAL_CATEGORIES = frozenset({"Nd", "No", "Nl"})
FRACTION_MARK = "\u2044"  # what fold_text leaves for any fraction; no source can turn it into an allowed number
FRACTION_NEIGHBOURS = re.compile(r"(?:\d[\d,]*(?:\.\d+)?\s*)?\u2044(?:\s*\d[\d,]*(?:\.\d+)?)?")
INVISIBLE_MARKS = frozenset([0x034F, 0x180B, 0x180C, 0x180D, 0x180F, *range(0xFE00, 0xFE10), *range(0xE0100, 0xE01F0)])  # Mn: CGJ and variation selectors


def strip_format_chars(text):
    """Remove category Cf characters (zero-width space/joiner, soft hyphen, bidi marks) that can split a word to evade scans."""
    return "".join(c for c in text if unicodedata.category(c) != "Cf")


def strip_invisible(text):
    """strip_format_chars plus the invisible Mn characters (combining grapheme joiner, variation selectors) that split a token the same way."""
    return "".join(c for c in strip_format_chars(text) if ord(c) not in INVISIBLE_MARKS)


def _is_fraction(char):
    if ord(char) < 128:
        return False
    value = unicodedata.numeric(char, None)
    return (value is not None and value != int(value)) or FRACTION_MARK in unicodedata.normalize("NFKC", char)


def has_non_ascii_numeral(text):
    return any(ord(c) > 127 and unicodedata.category(c) in NUMERAL_CATEGORIES for c in text)


def _numeral_text(char):
    """ASCII text for a non-ASCII numeral that NFKC left alone (e.g. Arabic-Indic digits, Aegean or Coptic numbers)."""
    try:
        return str(unicodedata.digit(char))
    except ValueError:
        pass
    try:
        value = unicodedata.numeric(char)
    except ValueError:
        return char
    return f" {int(value)} " if value == int(value) else f" {value} "


def fold_text(text):
    """Strip Cf/invisible Mn characters, mark fractions (FRACTION_MARK, with adjacent digits dropped), apply NFKC, then replace any
    remaining non-ASCII Nd/No/Nl character by its ASCII numeric value."""
    text = strip_invisible(text)
    if any(_is_fraction(c) for c in text):
        text = "".join(FRACTION_MARK if _is_fraction(c) else c for c in text)
    text = unicodedata.normalize("NFKC", text)
    if FRACTION_MARK in text:
        text = FRACTION_NEIGHBOURS.sub(f" {FRACTION_MARK} ", text)
    if not has_non_ascii_numeral(text):
        return text
    return "".join(_numeral_text(c) if ord(c) > 127 and unicodedata.category(c) in NUMERAL_CATEGORIES else c for c in text)


def _normalise(text):
    text = fold_text(text)
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
                    if SCALES[w] >= 1000:
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
        return set(re.findall(r"[a-z]+", fold_text(source).lower()))
    if isinstance(source, dict):
        source = list(source.values())
    if isinstance(source, (list, tuple, set)):
        return {w for item in source for w in source_words(item)}
    return set()


def verify_numbers(text, source):
    allowed = allowed_numbers(source)
    spoken, unsupported = [], []
    backed = source_words(source)
    text = fold_text(text)
    if FRACTION_MARK in text:
        unsupported.append("fraction")
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in VAGUE_QUANTITY_WORDS and word not in backed:
            unsupported.append(word)
    for value, decimals in spoken_numbers(text):
        spoken.append(value)
        if not any(abs(a - value) < 1e-9 or abs(round(a, decimals) - value) < 1e-9 for a in allowed):
            unsupported.append(value)
    return {"ok": not unsupported, "spoken": spoken, "unsupported": unsupported}

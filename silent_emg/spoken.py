"""Convert the written closed-vocabulary prompts ("08:11 PM", "July 07 2018") to spoken words.

The speaker read written prompts aloud (or mouthed them), so the articulated
sequence is "eight eleven p m", not the digits "0811". Training on spoken words
keeps the target sequence aligned with what the muscles did. Where several
readings are possible (years 2000-2009, ":00") one convention is fixed here;
the audio was not used to check which one the speaker chose.
"""

from __future__ import annotations

import re

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
       "twelve": "twelfth", "twenty": "twentieth", "thirty": "thirtieth"}


def two_digits(n: int) -> str:
    if n < 20:
        return ONES[n]
    t, o = divmod(n, 10)
    return TENS[t] + ("" if o == 0 else " " + ONES[o])


def ordinal(n: int) -> str:
    words = two_digits(n).split()
    last = words[-1]
    words[-1] = ORD.get(last, last + "th")
    return " ".join(words)


def year(y: int) -> str:
    hi, lo = divmod(y, 100)
    if 2000 <= y <= 2009:
        return "two thousand" + ("" if lo == 0 else " " + ONES[lo])
    if lo == 0:
        return two_digits(hi) + " hundred"
    return two_digits(hi) + " " + ("oh " + ONES[lo] if lo < 10 else two_digits(lo))


def spoken(text: str) -> str:
    t = text.strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})\s*([AaPp])\.?\s*[Mm]\.?", t)
    if m:
        h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3).lower()
        mins = "oclock" if mi == 0 else ("oh " + ONES[mi] if mi < 10 else two_digits(mi))
        return f"{two_digits(h)} {mins} {ap}m"
    out = []
    for tok in t.replace(",", " ").split():
        if re.fullmatch(r"\d{4}", tok):
            out.append(year(int(tok)))
        elif re.fullmatch(r"\d{1,2}", tok):
            out.append(ordinal(int(tok)))
        else:
            out.append(tok.lower())
    return " ".join(out)

import numpy as np

from silent_emg.grammar import GrammarDecoder
from silent_emg.text import CHARS, VOCAB_SIZE


def _spell(text, frames_per_char=3, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    prev = None
    for ch in text:
        if ch == prev:
            rows.append(0)  # blank needed between repeated characters
        rows += [CHARS.index(ch) + 1] * frames_per_char
        prev = ch
    lp = np.full((len(rows), VOCAB_SIZE), -8.0) + noise * rng.normal(size=(len(rows), VOCAB_SIZE))
    lp[np.arange(len(rows)), rows] = 0.0
    return lp


def test_decodes_valid_sentences():
    g = GrammarDecoder(year_range=(1990, 2010))
    for s in ["eight eleven pm", "july seventh two thousand eight", "monday june thirtieth", "seven oclock am"]:
        assert g.decode(_spell(s)) == s


def test_snaps_to_grammar():
    g = GrammarDecoder(year_range=(1990, 2010))
    # "junr" is not a month; the decoder must return a valid sentence
    out = g.decode(_spell("monday junr thirtieth", noise=0.1))
    assert out == "monday june thirtieth"

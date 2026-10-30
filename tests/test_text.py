from silent_emg.text import VOCAB_SIZE, align, cer, clean_text, decode_ids, encode, greedy_decode, wer
import numpy as np


def test_clean_text():
    assert clean_text("08:11 PM") == "08 11 pm"
    assert clean_text("Tuesday, May 03!") == "tuesday may 03"


def test_encode_roundtrip():
    t = "july 07 2018"
    assert decode_ids(encode(t)) == t
    assert max(encode(t)) < VOCAB_SIZE and min(encode(t)) >= 1


def test_greedy_decode_collapses_repeats_and_blanks():
    ids = [encode("a")[0]] * 3 + [0] + [encode("a")[0]] + [encode("b")[0]] * 2
    logits = np.full((len(ids), VOCAB_SIZE), -10.0)
    logits[np.arange(len(ids)), ids] = 0
    assert greedy_decode(logits) == "aab"


def test_error_rates():
    assert cer(["abc"], ["abc"]) == 0
    assert cer(["abcd"], ["abxd"]) == 0.25
    assert wer(["a b c d"], ["a b"]) == 0.5


def test_align():
    pairs = align("abc", "axc")
    assert pairs == [("a", "a"), ("b", "x"), ("c", "c")]
    assert ("b", None) in align("abc", "ac")

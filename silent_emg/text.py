"""Text normalisation, CTC greedy decoding and error-rate metrics."""

from __future__ import annotations

import re
import string
import unicodedata

import numpy as np

CHARS = " " + string.ascii_lowercase + string.digits  # index 0 is reserved for the CTC blank
VOCAB_SIZE = len(CHARS) + 1


def clean_text(text: str) -> str:
    """Lower-case ASCII, ':' and '-' become spaces, other punctuation is dropped."""
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    t = t.lower().replace(":", " ").replace("-", " ")
    t = re.sub(r"[^a-z0-9 ]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def encode(text: str) -> list[int]:
    return [CHARS.index(c) + 1 for c in clean_text(text)]


def decode_ids(ids) -> str:
    return "".join(CHARS[i - 1] for i in ids if i > 0)


def greedy_decode(logits: np.ndarray) -> str:
    """logits: (T, V). Collapse repeats, drop blanks (index 0)."""
    best = np.asarray(logits).argmax(-1)
    out, prev = [], -1
    for b in best:
        if b != prev and b != 0:
            out.append(int(b))
        prev = b
    return re.sub(r"\s+", " ", decode_ids(out)).strip()


def edit_distance(ref, hyp) -> int:
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return int(d[-1])


def cer(refs: list[str], hyps: list[str]) -> float:
    return sum(edit_distance(r, h) for r, h in zip(refs, hyps)) / max(1, sum(len(r) for r in refs))


def wer(refs: list[str], hyps: list[str]) -> float:
    return sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps)) / max(1, sum(len(r.split()) for r in refs))


def align(ref: str, hyp: str) -> list[tuple[str | None, str | None]]:
    """Levenshtein alignment as (ref_char, hyp_char) pairs; None marks a gap."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=int)
    D[:, 0], D[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            pairs.append((ref[i - 1], None)); i -= 1
        else:
            pairs.append((None, hyp[j - 1])); j -= 1
    return pairs[::-1]

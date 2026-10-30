"""Grammar-constrained CTC decoding for the closed-vocabulary prompts.

The prompts come from three templates. The set of valid spoken sentences is
written as a small graph of character states (one parallel chain per slot
value, slots joined by a shared space state), wrapped in the usual CTC blank
topology. Viterbi search over this graph returns the most likely valid
sentence. This plays the role that the human listeners' knowledge ("the
examples will contain dates and times") played in Gaddy & Klein (2020).
"""

from __future__ import annotations

import numpy as np

from .classical import MONTHS, WEEKDAYS
from .spoken import ONES, ordinal, two_digits, year
from .text import CHARS


def slot_values(year_range=(1883, 2020)) -> dict[str, list[str]]:
    mins = ["oclock"] + ["oh " + ONES[m] for m in range(1, 10)] + [two_digits(m) for m in range(10, 60)]
    return {"hour": [two_digits(h) for h in range(1, 13)], "minute": mins, "ampm": ["am", "pm"],
            "month": MONTHS, "day": [ordinal(d) for d in range(1, 32)], "weekday": WEEKDAYS,
            "year": [year(y) for y in range(year_range[0], year_range[1] + 1)]}


TEMPLATES = [["hour", "minute", "ampm"], ["month", "day", "year"], ["weekday", "month", "day"]]


class GrammarDecoder:
    def __init__(self, year_range=(1883, 2020), templates=TEMPLATES):
        values = slot_values(year_range)
        labels, edges, starts, finals = [], [], [], []

        def chain(s):
            ids = []
            for ch in s:
                labels.append(CHARS.index(ch) + 1)
                ids.append(len(labels) - 1)
            for a, b in zip(ids, ids[1:]):
                edges.append((a, b))
            return ids[0], ids[-1]

        for tpl in templates:
            prev_ends = None
            for k, slot in enumerate(tpl):
                if prev_ends is not None:  # shared space state between slots of this template
                    labels.append(CHARS.index(" ") + 1)
                    sp = len(labels) - 1
                    edges.extend((e, sp) for e in prev_ends)
                ends = []
                for v in values[slot]:
                    a, b = chain(v)
                    if prev_ends is None:
                        starts.append(a)
                    else:
                        edges.append((sp, a))
                    ends.append(b)
                prev_ends = ends
            finals.extend(prev_ends)
        self._build(np.array(labels), edges, starts, finals)

    def _build(self, labels, char_edges, starts, finals):
        n = len(labels)
        # states 0..n-1: characters; n..2n-1: blank after character i; 2n: initial blank
        self.n_states = 2 * n + 1
        self.label = np.concatenate([labels, np.zeros(n + 1, dtype=int)])
        e = [(i, i) for i in range(self.n_states)]  # self-loops
        e += [(i, n + i) for i in range(n)]  # char -> its blank
        for a, b in char_edges:
            e.append((n + a, b))  # via blank
            if labels[a] != labels[b]:
                e.append((a, b))  # direct, only if the characters differ
        e += [(2 * n, s) for s in starts]
        e = np.array(sorted(set(e), key=lambda t: (t[1], t[0])))
        self.src, self.dst = e[:, 0], e[:, 1]
        self.seg = np.flatnonzero(np.r_[True, self.dst[1:] != self.dst[:-1]])
        self.seg_dst = self.dst[self.seg]
        self.init = np.array([2 * n] + list(starts))
        self.final = np.array(list(finals) + [n + f for f in finals])
        self.n_chars = n

    def decode(self, logprobs: np.ndarray) -> str:
        lp = np.asarray(logprobs, dtype=np.float64)
        T = len(lp)
        score = np.full(self.n_states, -np.inf)
        score[self.init] = lp[0, self.label[self.init]]
        back = np.zeros((T, self.n_states), dtype=np.int32)
        for t in range(1, T):
            cand = score[self.src]
            best = np.maximum.reduceat(cand, self.seg)
            # argmax per destination: the first edge in each segment that reaches the maximum
            hit = cand == np.repeat(best, np.diff(np.r_[self.seg, len(cand)]))
            pos = np.where(hit, np.arange(len(cand)), len(cand))
            arg = np.minimum.reduceat(pos, self.seg)
            new = np.full(self.n_states, -np.inf)
            new[self.seg_dst] = best
            back[t, self.seg_dst] = self.src[np.minimum(arg, len(cand) - 1)]
            score = new + lp[t, self.label]
        s = int(self.final[np.argmax(score[self.final])])
        path = [s]
        for t in range(T - 1, 0, -1):
            s = int(back[t, s])
            path.append(s)
        path = path[::-1]
        out, prev = [], -1
        for s in path:
            if s < self.n_chars and s != prev:
                out.append(CHARS[self.label[s] - 1])
            prev = s
        return "".join(out)

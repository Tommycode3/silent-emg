"""Train / validation / test splits. Every split is a partition of utterance indices."""

from __future__ import annotations

import numpy as np


def random_split(n: int, n_val: int, n_test: int, seed: int = 0):
    """Random utterance split (the protocol of Gaddy & Klein 2020 for the closed vocabulary)."""
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[n_val + n_test:]), np.sort(perm[:n_val]), np.sort(perm[n_val: n_val + n_test])


def block_split(n: int, n_val: int, n_test: int):
    """Contiguous blocks in recording order: train first, then validation, then test.

    Within a single session this is the closest proxy for a later, unseen recording.
    """
    idx = np.arange(n)
    n_train = n - n_val - n_test
    return idx[:n_train], idx[n_train: n_train + n_val], idx[n_train + n_val:]


def session_split(sessions: list[str], val_sessions: list[str], test_sessions: list[str]):
    """Whole sessions held out (electrodes were re-attached between sessions)."""
    s = np.asarray(sessions)
    val = np.flatnonzero(np.isin(s, val_sessions))
    test = np.flatnonzero(np.isin(s, test_sessions))
    train = np.flatnonzero(~np.isin(s, list(val_sessions) + list(test_sessions)))
    return train, val, test


def check_disjoint(*parts) -> None:
    seen = set()
    for p in parts:
        p = set(int(i) for i in p)
        if seen & p:
            raise ValueError("splits overlap")
        seen |= p

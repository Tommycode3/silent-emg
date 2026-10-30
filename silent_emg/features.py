"""Hand-crafted EMG features.

Frame-level time-domain features follow Jou et al. (2006), with two classic
Hudgins et al. (1993) features (waveform length, slope-sign changes) added.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ["low_power", "low_mean", "high_power", "high_zcr", "high_rect_mean", "waveform_len", "slope_sign"]


def _double_average(x: np.ndarray, n: int = 9) -> np.ndarray:
    k = np.ones(n) / n
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, x)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, out)


def _frames(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """(T, C) -> (F, frame, C) strided view."""
    n = 1 + max(0, (len(x) - frame) // hop)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def td_features(x: np.ndarray, fs: int = 1000, frame_ms: float = 27.0, hop_ms: float = 10.0) -> np.ndarray:
    """Per-channel frame features. x: (T, C) -> (F, C * 7) at 1000/hop_ms frames per second."""
    x = x - x.mean(0, keepdims=True)
    w = _double_average(x)  # low-frequency part
    p = x - w  # high-frequency part
    r = np.abs(p)
    frame, hop = int(round(frame_ms * fs / 1000)), int(round(hop_ms * fs / 1000))
    if len(x) < frame:
        pad = np.zeros((frame - len(x), x.shape[1]))
        x, w, p, r = (np.concatenate([a, pad]) for a in (x, w, p, r))
    W, P, R, X = (_frames(a, frame, hop) for a in (w, p, r, x))
    low_power = (W ** 2).mean(1)
    low_mean = W.mean(1)
    high_power = (P ** 2).mean(1)
    zcr = (np.diff(np.signbit(P), axis=1) != 0).mean(1)
    rect = R.mean(1)
    wl = np.abs(np.diff(X, axis=1)).sum(1)
    d = np.diff(X, axis=1)
    ssc = ((d[:, 1:] * d[:, :-1]) < 0).mean(1)
    feats = np.stack([low_power, low_mean, high_power, zcr, rect, wl, ssc], axis=2)  # (F, C, 7)
    # log-compress the power-like features
    for j in (0, 2, 4, 5):
        feats[..., j] = np.log(feats[..., j] + 1e-6)
    return feats.reshape(len(feats), -1).astype(np.float32)


def stack_context(f: np.ndarray, k: int) -> np.ndarray:
    """Concatenate +-k neighbouring frames (edge-padded): (F, D) -> (F, D * (2k + 1))."""
    if k == 0:
        return f
    padded = np.concatenate([np.repeat(f[:1], k, 0), f, np.repeat(f[-1:], k, 0)])
    return np.concatenate([padded[i: i + len(f)] for i in range(2 * k + 1)], axis=1)

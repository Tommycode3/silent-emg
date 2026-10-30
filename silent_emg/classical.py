"""Utterance-level slot classification with Hudgins time-domain features.

The closed-vocabulary prompts come from three templates ("08:11 PM",
"July 07 2018", "Monday June 30"). That gives four closed-set questions that a
classical classifier can answer from a fixed-length feature vector:
template (3 classes), AM vs PM, weekday (7) and month (12).
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
TEMPLATES = ["time", "month day year", "weekday month day"]
HUDGINS = ["MAV", "WL", "ZC", "SSC", "logRMS"]


def template_of(written: str) -> str:
    if re.search(r"\d:\d", written):
        return "time"
    return "month day year" if re.search(r"\d{4}", written) else "weekday month day"


def slot_labels(written: str) -> dict:
    """Ground-truth slots from the written prompt. Missing slots are None."""
    w = written.lower()
    tpl = template_of(written)
    return {"template": tpl,
            "ampm": ("am" if "am" in w.split() else "pm") if tpl == "time" else None,
            "weekday": next((d for d in WEEKDAYS if d in w), None),
            "month": next((m for m in MONTHS if m in w.split()), None)}


def slots_from_transcript(spoken: str) -> dict:
    """Slots read off a decoded (spoken-form) transcript, for scoring the CTC models the same way."""
    words = spoken.split()
    has_year = any(x in words for x in ("thousand", "hundred")) or len(words) >= 5
    if "am" in words or "pm" in words or "oclock" in words or not any(m in words for m in MONTHS):
        tpl = "time"
    elif any(d in words for d in WEEKDAYS) and not has_year:
        tpl = "weekday month day"
    else:
        tpl = "month day year"
    return {"template": tpl,
            "ampm": "am" if "am" in words else ("pm" if "pm" in words else None),
            "weekday": next((d for d in WEEKDAYS if d in words), None),
            "month": next((m for m in MONTHS if m in words), None)}


def hudgins(x: np.ndarray, thr: float = 0.01) -> np.ndarray:
    """MAV, WL, ZC, SSC (Hudgins et al. 1993) and log RMS per channel. x: (T, C) -> (C, 5)."""
    eps = thr * (x.std(0, keepdims=True) + 1e-9)
    d = np.diff(x, axis=0)
    mav = np.abs(x).mean(0)
    wl = np.abs(d).mean(0)
    zc = ((x[:-1] * x[1:] < 0) & (np.abs(d) >= eps)).mean(0)
    ssc = ((d[:-1] * d[1:] < 0) & ((np.abs(d[:-1]) >= eps) | (np.abs(d[1:]) >= eps))).mean(0)
    rms = np.log(np.sqrt((x ** 2).mean(0)) + 1e-9)
    return np.stack([np.log(mav + 1e-9), np.log(wl + 1e-9), zc, ssc, rms], axis=1)


def utterance_features(x: np.ndarray, segments: int = 4, channels=None) -> np.ndarray:
    """Hudgins features on the whole utterance and on `segments` equal-length parts, plus log duration."""
    if channels is not None:
        x = x[:, channels]
    parts = [x] + np.array_split(x, segments, axis=0)
    f = [hudgins(p).ravel() for p in parts]
    return np.concatenate(f + [np.array([np.log(len(x))])]).astype(np.float32)


def make_classifier(kind: str, seed: int = 0):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                              l2_regularization=1.0, random_state=seed)
    raise ValueError(kind)


def balanced_accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    classes = sorted(set(y_true))
    return float(np.mean([(y_pred[y_true == c] == c).mean() for c in classes]))

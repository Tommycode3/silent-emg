"""Signal conditioning: mains notch filters and a zero-phase band-pass."""

from __future__ import annotations

import numpy as np
from scipy import signal

from . import FS


def notch_harmonics(x: np.ndarray, fs: float = FS, mains: float = 60.0, q: float = 30.0) -> np.ndarray:
    """Zero-phase IIR notch at the mains frequency and every harmonic below Nyquist.

    x has shape (time, channels).
    """
    y = np.asarray(x, dtype=np.float64)
    k = 1
    while mains * k < fs / 2 - 1:
        b, a = signal.iirnotch(mains * k, q, fs)
        y = signal.filtfilt(b, a, y, axis=0)
        k += 1
    return y


def bandpass(x: np.ndarray, lo: float = 2.0, hi: float = 450.0, fs: float = FS, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass (second-order sections), x is (time, channels)."""
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=0)


def preprocess(x: np.ndarray, before: np.ndarray | None = None, after: np.ndarray | None = None,
               fs: float = FS, mains: float = 60.0, lo: float = 2.0, hi: float = 450.0) -> np.ndarray:
    """Notch + band-pass one utterance.

    The neighbouring recordings (if given) are concatenated before filtering so the
    filter transients fall outside the utterance, then cut away again.
    """
    before = np.zeros((0, x.shape[1])) if before is None else before
    after = np.zeros((0, x.shape[1])) if after is None else after
    z = np.concatenate([before, x, after], 0)
    z = notch_harmonics(z, fs, mains)
    z = bandpass(z, lo, hi, fs)
    return z[len(before): len(before) + len(x)].astype(np.float32)

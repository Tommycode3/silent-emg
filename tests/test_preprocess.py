import numpy as np

from silent_emg.preprocess import bandpass, notch_harmonics, preprocess


def _power_at(x, f, fs=1000):
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    return spec[np.argmin(np.abs(freqs - f))]


def test_notch_removes_mains_and_harmonics_keeps_emg_band():
    t = np.arange(4000) / 1000
    x = np.sin(2 * np.pi * 60 * t) + np.sin(2 * np.pi * 180 * t) + np.sin(2 * np.pi * 100 * t)
    y = notch_harmonics(x[:, None])[:, 0]
    mid = slice(500, 3500)
    assert _power_at(y[mid], 60) < 1e-3 * _power_at(x[mid], 60)
    assert _power_at(y[mid], 180) < 1e-3 * _power_at(x[mid], 180)
    assert _power_at(y[mid], 100) > 0.8 * _power_at(x[mid], 100)


def test_bandpass_removes_drift():
    t = np.arange(5000) / 1000
    x = 5 * t + np.sin(2 * np.pi * 80 * t)
    y = bandpass(x[:, None])[:, 0]
    assert abs(y[1000:4000].mean()) < 0.05
    assert 0.4 < y[1000:4000].std() < 0.8  # sine of amplitude 1 has std 0.707


def test_preprocess_shape_dtype_and_neighbours():
    rng = np.random.default_rng(0)
    x, b, a = rng.normal(size=(700, 8)), rng.normal(size=(300, 8)), rng.normal(size=(200, 8))
    y = preprocess(x, b, a)
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.isfinite(y).all()

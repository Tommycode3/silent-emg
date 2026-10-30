import numpy as np

from silent_emg.features import FEATURE_NAMES, stack_context, td_features


def test_td_feature_shape_and_rate():
    x = np.random.default_rng(0).normal(size=(2000, 8))
    f = td_features(x)
    assert f.shape[1] == 8 * len(FEATURE_NAMES)
    assert abs(f.shape[0] - 2000 / 10) <= 3  # 10 ms hop


def test_td_features_track_amplitude():
    rng = np.random.default_rng(1)
    quiet, loud = rng.normal(0, 0.1, (500, 1)), rng.normal(0, 1.0, (500, 1))
    f = td_features(np.concatenate([quiet, loud]))
    high_power = f[:, 2]
    assert high_power[-20:].mean() > high_power[:20].mean() + 2  # log scale


def test_zero_crossing_feature_follows_frequency():
    t = np.arange(1000) / 1000
    lo = td_features(np.sin(2 * np.pi * 50 * t)[:, None])[:, 3].mean()
    hi = td_features(np.sin(2 * np.pi * 300 * t)[:, None])[:, 3].mean()
    assert hi > lo


def test_stack_context():
    f = np.arange(10, dtype=float)[:, None]
    s = stack_context(f, 2)
    assert s.shape == (10, 5)
    np.testing.assert_array_equal(s[5], [3, 4, 5, 6, 7])
    np.testing.assert_array_equal(s[0], [0, 0, 0, 1, 2])

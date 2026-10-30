"""End-to-end check on synthetic EMG: each character is a burst on its own channel."""

import numpy as np
import torch

from silent_emg.models import ConvTransformerCTC, LinearCTC
from silent_emg.text import VOCAB_SIZE
from silent_emg.train import Config, evaluate, fit

ALPHA = "abcd"


def synth(n, seed):
    rng = np.random.default_rng(seed)
    sigs, txts = [], []
    for _ in range(n):
        word = "".join(rng.choice(list(ALPHA), size=rng.integers(2, 4)))
        x = rng.normal(0, 0.05, size=(150 * len(word) + 200, 8))
        for k, ch in enumerate(word):
            s = 100 + 150 * k
            x[s: s + 100, ALPHA.index(ch)] += rng.normal(0, 1.0, size=100)
        sigs.append(x.astype(np.float32)); txts.append(word)
    return sigs, txts


def test_linear_ctc_learns_synthetic_task():
    tr, va = synth(120, 0), synth(30, 1)
    cfg = Config(model="linear", epochs=40, eval_every=10, context=3, batch_size=16)
    model, norm, hist = fit(*tr, *va, cfg, log=lambda *a: None)
    res = evaluate(model, norm, *va, cfg)
    assert res["cer"] < 0.35, res["cer"]


def test_neural_forward_shapes():
    m = ConvTransformerCTC(8, VOCAB_SIZE, d=32, layers=1, heads=2)
    out = m(torch.zeros(2, 1000, 8))
    assert out.shape == (2, 250, VOCAB_SIZE)  # input already at 500 Hz, x4 down
    assert LinearCTC(10, VOCAB_SIZE)(torch.zeros(2, 5, 10)).shape == (2, 5, VOCAB_SIZE)


def test_neural_one_epoch_runs_on_cpu():
    tr, va = synth(8, 0), synth(4, 1)
    cfg = Config(model="neural", epochs=1, eval_every=1, device="cpu", d_model=32, layers=1, batch_size=4)
    model, norm, hist = fit(*tr, *va, cfg, log=lambda *a: None)
    assert len(hist) == 1 and np.isfinite(hist[0]["loss"])


def test_channel_subset_changes_input_width():
    tr, va = synth(8, 0), synth(4, 1)
    cfg = Config(model="neural", epochs=1, device="cpu", d_model=32, layers=1, channels=[0, 1, 2])
    model, _, _ = fit(*tr, *va, cfg, log=lambda *a: None)
    assert model.conv[0].c1.in_channels == 3


def test_make_batches_respects_budget():
    from silent_emg.train import make_batches
    lengths = [100, 900, 300, 500, 50, 1000]
    b = make_batches(lengths, np.arange(6), batch_size=3, max_samples=1500)
    assert sorted(i for x in b for i in x) == list(range(6))
    assert all(len(x) <= 3 and len(x) * max(lengths[i] for i in x) <= 1500 or len(x) == 1 for x in b)

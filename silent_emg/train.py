"""Training and evaluation loop shared by both CTC models."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from .features import stack_context, td_features
from .models import ConvTransformerCTC, LinearCTC
from .text import VOCAB_SIZE, cer, encode, greedy_decode, wer


@dataclass
class Config:
    model: str = "neural"  # "neural" or "linear"
    epochs: int = 80
    batch_size: int = 16
    lr: float = 5e-4
    weight_decay: float = 1e-2
    eval_every: int = 2
    context: int = 7  # linear model: +-frames stacked
    channels: list[int] = field(default_factory=lambda: list(range(8)))
    device: str = "auto"
    seed: int = 0
    d_model: int = 128
    layers: int = 3
    patience: int = 20  # evaluations without val improvement before stopping


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_inputs(signals: list[np.ndarray], cfg: Config) -> list[np.ndarray]:
    """Per-utterance model inputs. Unused channels are removed (not zeroed)."""
    sig = [s[:, cfg.channels] for s in signals]
    if cfg.model == "linear":
        return [stack_context(td_features(s), cfg.context) for s in sig]
    # anti-aliased, zero-phase decimation 1000 Hz -> 500 Hz (the raw-signal rate used by Gaddy & Klein is 516 Hz)
    return [signal.decimate(s, 2, ftype="fir", axis=0, zero_phase=True).astype(np.float32) for s in sig]


class Normaliser:
    """Per-dimension z-score with statistics from the training set only."""

    def __init__(self, xs: list[np.ndarray]):
        cat = np.concatenate(xs, 0)
        self.mu = cat.mean(0).astype(np.float32)
        self.sd = (cat.std(0) + 1e-6).astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sd


def _batch(xs, ys, idx, device, augment=False, rng=None):
    lens = [len(xs[i]) for i in idx]
    T = max(lens)
    X = np.zeros((len(idx), T, xs[idx[0]].shape[1]), np.float32)
    for k, i in enumerate(idx):
        x = xs[i]
        if augment:
            x = x * rng.normal(1.0, 0.1, size=(1, x.shape[1])).astype(np.float32)
            x = x + rng.normal(0, 0.05, size=x.shape).astype(np.float32)
            for _ in range(2):  # time masking: zero two random spans of up to 80 ms
                w = int(rng.integers(0, 80))
                s0 = int(rng.integers(0, max(1, len(x) - w)))
                x[s0: s0 + w] = 0
        X[k, : len(x)] = x
    tgt = [torch.tensor(ys[i]) for i in idx] if ys is not None else None
    return torch.from_numpy(X).to(device), torch.tensor(lens), tgt


def _forward(model, X, lens):
    if isinstance(model, ConvTransformerCTC):
        out_lens = lens.clone()
        for _ in range(2):
            out_lens = (out_lens - 1) // 2 + 1
        T = X.shape[1]
        for _ in range(2):
            T = (T - 1) // 2 + 1
        mask = torch.arange(T)[None, :] >= out_lens[:, None]
        return model(X, mask.to(X.device)), out_lens
    return model(X), lens


def predict(model, xs, device, batch_size=16, return_logprobs=False):
    model.eval()
    hyps, lps = [], []
    with torch.no_grad():
        for s in range(0, len(xs), batch_size):
            idx = list(range(s, min(len(xs), s + batch_size)))
            X, lens, _ = _batch(xs, None, idx, device)
            logits, out_lens = _forward(model, X, lens)
            lp = torch.log_softmax(logits.float(), -1).cpu().numpy()
            hyps += [greedy_decode(lp[k, : out_lens[k]]) for k in range(len(idx))]
            lps += [lp[k, : out_lens[k]].astype(np.float16) for k in range(len(idx))]
    return (hyps, lps) if return_logprobs else hyps


def fit(train_sig, train_txt, val_sig, val_txt, cfg: Config, log=print, init_state=None):
    """Train one model. Returns (model, normaliser, history). Texts are already clean strings."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = pick_device(cfg.device) if cfg.model == "neural" else torch.device("cpu")

    xtr, xva = make_inputs(train_sig, cfg), make_inputs(val_sig, cfg)
    norm = Normaliser(xtr)
    xtr, xva = [norm(x) for x in xtr], [norm(x) for x in xva]
    ytr = [encode(t) for t in train_txt]

    if cfg.model == "neural":
        model = ConvTransformerCTC(len(cfg.channels), VOCAB_SIZE, cfg.d_model, cfg.layers)
    else:
        model = LinearCTC(xtr[0].shape[1], VOCAB_SIZE)
    if init_state is not None:  # e.g. weights pre-trained on the open-vocabulary sessions
        model.load_state_dict(init_state)
    model.to(device)
    lr = cfg.lr if cfg.model == "neural" else 3e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = int(np.ceil(len(xtr) / cfg.batch_size))
    total = cfg.epochs * steps_per_epoch
    warm = max(1, min(500, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / total))))

    best, best_state, hist, bad = np.inf, None, [], 0
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(xtr))
        losses = []
        for s in range(0, len(order), cfg.batch_size):
            idx = order[s: s + cfg.batch_size]
            X, lens, tgt = _batch(xtr, ytr, idx, device, augment=cfg.model == "neural", rng=rng)
            logits, out_lens = _forward(model, X, lens)
            lp = F.log_softmax(logits.float(), -1).transpose(0, 1).cpu()
            loss = F.ctc_loss(lp, torch.cat(tgt), out_lens, torch.tensor([len(t) for t in tgt]),
                              blank=0, zero_infinity=True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        if (ep + 1) % cfg.eval_every == 0 or ep == cfg.epochs - 1:
            v = cer(val_txt, predict(model, xva, device))
            hist.append({"epoch": ep + 1, "loss": float(np.mean(losses)), "val_cer": v})
            if v < best:
                best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            log(f"  ep {ep + 1:3d} loss {np.mean(losses):.3f} val CER {v:.3f} ({time.time() - t0:.0f}s)")
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, norm, hist


def evaluate(model, norm, sig, txt, cfg: Config):
    device = next(model.parameters()).device
    xs = [norm(x) for x in make_inputs(sig, cfg)]
    hyps, lps = predict(model, xs, device, return_logprobs=True)
    return {"cer": cer(txt, hyps), "wer": wer(txt, hyps), "hyps": hyps, "logprobs": lps}

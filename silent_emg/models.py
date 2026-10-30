"""Two CTC models: a linear frame classifier on hand-crafted features, and a raw-signal network."""

from __future__ import annotations

import math

import torch
from torch import nn


class LinearCTC(nn.Module):
    """Softmax regression per frame on context-stacked time-domain features, trained with CTC."""

    stride = 1

    def __init__(self, in_dim: int, vocab: int):
        super().__init__()
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(in_dim, vocab)

    def forward(self, x):  # x: (B, T, D)
        return self.out(self.drop(x))


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 3, stride, padding=1)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 3, 1, padding=1)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = nn.Sequential(nn.Conv1d(cin, cout, 1, stride), nn.BatchNorm1d(cout))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.b2(self.c2(self.act(self.b1(self.c1(x))))) + self.skip(x))


class SinusoidalPE(nn.Module):
    def __init__(self, d: int, max_len: int = 8000):
        super().__init__()
        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(max_len, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.shape[1]]


class ConvTransformerCTC(nn.Module):
    """Raw EMG (1000 Hz) -> 3 strided residual conv blocks (x8 down, 125 frames/s) -> Transformer -> CTC."""

    stride = 8

    def __init__(self, in_ch: int = 8, vocab: int = 38, d: int = 128, layers: int = 3, heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(ResBlock(in_ch, d // 4, 2), ResBlock(d // 4, d // 2, 2), ResBlock(d // 2, d, 2))
        self.pe = SinusoidalPE(d)
        enc = nn.TransformerEncoderLayer(d, heads, 3 * d, dropout, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, vocab)

    def forward(self, x, pad_mask=None):  # x: (B, T, C)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h = self.drop(self.pe(h))
        h = self.tf(h, src_key_padding_mask=pad_mask)
        return self.out(h)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

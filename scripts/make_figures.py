"""Summarise results/runs.jsonl into results/summary.md and the figures in figures/."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from silent_emg import CHANNEL_SHORT  # noqa: E402
from silent_emg.text import align  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 170, "savefig.bbox": "tight"})
C_LIN, C_NN, C_CTRL = "#8c6d31", "#1f4e79", "#9e9e9e"
FIG = ROOT / "figures"


def load():
    runs = [json.loads(line) for line in (ROOT / "results" / "runs.jsonl").read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for r in runs:
        groups[r["key"].rsplit("/s", 1)[0]].append(r)
    return runs, groups


def ms(rs, k="cer"):
    v = np.array([r[k] for r in rs]) * 100
    return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, v


def fmt(rs, k):
    m, s, v = ms(rs, k)
    return f"{m:.1f} ± {s:.1f}"


ROWS = [
    ("E1/linear/shuffled", "closed vocab, random split", "linear TD, shuffled labels (control)"),
    ("E1/neural/shuffled", "closed vocab, random split", "conv + transformer, shuffled labels (control)"),
    ("E1/linear", "closed vocab, random split", "linear TD features + CTC"),
    ("E1/neural", "closed vocab, random split", "conv + transformer + CTC"),
    ("E2/linear", "closed vocab, last 100 utterances", "linear TD features + CTC"),
    ("E2/neural", "closed vocab, last 100 utterances", "conv + transformer + CTC"),
    ("E3/neural/voiced_only", "train vocalised, test silent", "conv + transformer + CTC"),
    ("E5/linear/random", "open vocab, random utterances", "linear TD features + CTC"),
    ("E5/neural/random", "open vocab, random utterances", "conv + transformer + CTC"),
    ("E5/linear/session", "open vocab, held-out session", "linear TD features + CTC"),
    ("E5/neural/session", "open vocab, held-out session", "conv + transformer + CTC"),
    ("E5/neural/session/shuffled", "open vocab, held-out session", "conv + transformer, shuffled labels (control)"),
]


def summary(groups):
    lines = ["| setting | model | seeds | test CER % | test WER % |", "|---|---|---|---|---|"]
    for key, setting, model in ROWS:
        if key in groups:
            rs = groups[key]
            lines.append(f"| {setting} | {model} | {len(rs)} | {fmt(rs, 'cer')} | {fmt(rs, 'wer')} |")
    abl = ["", "| electrodes kept | linear CER % | neural CER % |", "|---|---|---|"]
    for name in ["all"] + [k.split("/")[-1] for k in groups if k.startswith("E4/linear/")]:
        lk, nk = ("E1/linear", "E1/neural") if name == "all" else (f"E4/linear/{name}", f"E4/neural/{name}")
        cell = lambda k: fmt(groups[k], "cer") if k in groups else "n/a"  # noqa: E731
        abl.append(f"| {name} | {cell(lk)} | {cell(nk)} |")
    (ROOT / "results" / "summary.md").write_text("\n".join(lines + abl) + "\n")
    print("\n".join(lines + abl))


def bars(ax, items, title, ylabel="test CER (%)"):
    for i, (label, rs, color) in enumerate(items):
        m, s, v = ms(rs)
        ax.bar(i, m, color=color, width=0.65)
        ax.errorbar(i, m, s, color="black", capsize=3, lw=1)
        ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=8, color="white",
                   edgecolor="black", lw=0.5, zorder=3)
        ax.text(i, m + s + 2, f"{m:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(items)), [it[0] for it in items])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.set_ylim(0, 105)


def hero(groups):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    items = [("shuffled\nlabels", groups["E1/neural/shuffled"], C_CTRL),
             ("linear\nTD features", groups["E1/linear"], C_LIN),
             ("conv +\ntransformer", groups["E1/neural"], C_NN)]
    bars(a, items, "Silent EMG to text, closed vocabulary (100 test utterances)")
    a.set_ylabel("character error rate (%)")
    best = min(groups["E1/neural"], key=lambda r: abs(r["cer"] - np.median([q["cer"] for q in groups["E1/neural"]])))
    b.axis("off")
    b.set_title(f"Example test decodings (seed {best['seed']}, greedy, no language model)", loc="left")
    rng = np.random.default_rng(3)
    pick = rng.choice(len(best["refs"]), 8, replace=False)
    b.text(0.0, 0.93, "prompt", weight="bold", transform=b.transAxes, family="monospace")
    b.text(0.48, 0.93, "decoded from silent EMG", weight="bold", transform=b.transAxes, family="monospace")
    for k, i in enumerate(pick):
        y = 0.82 - k * 0.105
        ref, hyp = best["refs"][i], best["hyps"][i]
        b.text(0.0, y, ref, transform=b.transAxes, family="monospace")
        b.text(0.48, y, hyp, transform=b.transAxes, family="monospace", color="#1b7837" if ref == hyp else "#b2182b")
    fig.tight_layout()
    fig.savefig(FIG / "hero.png")


def results(groups):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    items = [(l, groups[k], c) for l, k, c in [
        ("random\nlinear", "E1/linear", C_LIN), ("random\nneural", "E1/neural", C_NN),
        ("last block\nlinear", "E2/linear", C_LIN), ("last block\nneural", "E2/neural", C_NN),
        ("voiced to\nsilent", "E3/neural/voiced_only", C_NN), ("shuffled\ncontrol", "E1/neural/shuffled", C_CTRL)]
        if k in groups]
    bars(axes[0], items, "Closed vocabulary (one session, 500 utterances)")
    items = [(l, groups[k], c) for l, k, c in [
        ("random utt.\nlinear", "E5/linear/random", C_LIN), ("random utt.\nneural", "E5/neural/random", C_NN),
        ("held-out sess.\nlinear", "E5/linear/session", C_LIN), ("held-out sess.\nneural", "E5/neural/session", C_NN),
        ("shuffled\ncontrol", "E5/neural/session/shuffled", C_CTRL)] if k in groups]
    if items:
        bars(axes[1], items, "Open vocabulary (7 sessions, test = session 5-11)")
    fig.tight_layout()
    fig.savefig(FIG / "results.png")


def ablation(groups):
    names = ["all", "no_throat", "face_only", "neck_only", "throat_only"] + [f"drop_{c}" for c in range(8)]
    labels = {"all": "all 8", "no_throat": "without throat", "face_only": "face only (6)",
              "neck_only": "neck only (under chin + throat)", "throat_only": "throat only"}
    labels |= {f"drop_{c}": f"without {CHANNEL_SHORT[c]}" for c in range(8)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, model, color in ((axes[0], "linear", C_LIN), (axes[1], "neural", C_NN)):
        ys, ms_, ss, ok = [], [], [], []
        for n in names:
            k = f"E1/{model}" if n == "all" else f"E4/{model}/{n}"
            if k in groups:
                m, s, _ = ms(groups[k]); ms_.append(m); ss.append(s); ok.append(n)
        y = np.arange(len(ok))[::-1]
        ax.barh(y, ms_, xerr=ss, color=[C_CTRL if n == "all" else color for n in ok], capsize=2)
        ax.axvline(ms_[0], color="black", ls="--", lw=0.8)
        ax.set_yticks(y, [labels[n] for n in ok])
        ax.set_xlabel("test CER (%), retrained per electrode set")
        ax.set_title(f"{'Linear TD features' if model == 'linear' else 'Conv + transformer'} "
                     f"({len(groups.get(f'E1/{model}', []))}/{len(groups.get(f'E4/{model}/no_throat', []))} seeds)",
                     loc="left")
        ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(FIG / "ablation.png")


def confusion(groups):
    syms = list("0123456789") + ["-"]
    M = np.zeros((len(syms), len(syms)))
    for r in groups["E1/neural"]:
        for ref, hyp in zip(r["refs"], r["hyps"]):
            for a, b in align(ref, hyp):
                if a is not None and a.isdigit():
                    M[syms.index(a), syms.index(b) if (b is not None and b.isdigit()) else -1] += 1
    M = M[:-1]
    P = M / M.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] >= 0.05:
                ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if P[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(syms)), syms[:-1] + ["other /\ndeleted"])
    ax.set_yticks(range(10), syms[:-1])
    ax.set_xlabel("decoded character")
    ax.set_ylabel("prompted digit")
    ax.set_title(f"Digit confusion, conv + transformer, closed vocab\n(test set, {len(groups['E1/neural'])} seeds pooled, "
                 f"{int(M.sum())} digits)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised rate")
    fig.tight_layout()
    fig.savefig(FIG / "confusion_digits.png")


def curves(runs):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for r in runs:
        if r["key"].startswith("E1/") and r["seed"] == 0:
            h = r["history"]
            c = C_CTRL if r["shuffled"] else (C_NN if r["model"] == "neural" else C_LIN)
            ax.plot([x["epoch"] for x in h], [100 * x["val_cer"] for x in h], color=c,
                    ls="--" if r["shuffled"] else "-", label=r["key"].replace("/s0", ""))
    ax.set_xlabel("epoch"); ax.set_ylabel("validation CER (%)"); ax.legend(fontsize=7, frameon=False)
    ax.set_title("Validation curves, closed vocabulary, seed 0", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "val_curves.png")


if __name__ == "__main__":
    runs, groups = load()
    summary(groups)
    for f in (hero, results, ablation, confusion):
        try:
            f(groups)
        except KeyError as e:
            print(f"skip {f.__name__}: missing {e}")
    curves(runs)

"""Summarise results/*.json(l) into results/summary.md and the figures in figures/."""

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
from silent_emg.classical import MONTHS, WEEKDAYS  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 170, "savefig.bbox": "tight"})
C_LIN, C_GBM, C_NN, C_NNG, C_CTRL = "#a6761d", "#e6ab02", "#7fa7cf", "#1f4e79", "#9e9e9e"
FIG = ROOT / "figures"
TASKS = ["template", "ampm", "weekday", "month"]
TASK_LABEL = {"template": "template\n(3 classes)", "ampm": "AM vs PM\n(2)", "weekday": "weekday\n(7)",
              "month": "month\n(12)"}


def load():
    runs = [json.loads(x) for x in (ROOT / "results" / "runs.jsonl").read_text().splitlines() if x.strip()]
    g = json.loads((ROOT / "results" / "grammar.json").read_text()) if (ROOT / "results" / "grammar.json").exists() else {}
    for r in runs:
        if r["key"] in g:
            r["grammar"] = g[r["key"]]
    groups = defaultdict(list)
    for r in runs:
        groups[r["key"].rsplit("/s", 1)[0]].append(r)
    cl = json.loads((ROOT / "results" / "classical.json").read_text())
    return runs, groups, cl


def msd(v):
    v = np.asarray(v, float) * 100
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), v


def fmt(v):
    m, s, vv = msd(v)
    return f"{m:.1f} ± {s:.1f}" if len(vv) > 1 and np.ptp(vv) > 0 else f"{m:.1f}"


CTC_ROWS = [
    ("E1/neural/shuffled", "closed vocab, random split", "conv + transformer, shuffled labels (control)"),
    ("E1/linear", "closed vocab, random split", "linear on TD features (Jou et al.)"),
    ("E1/neural", "closed vocab, random split", "conv + transformer"),
    ("E6/neural/pretrained", "closed vocab, random split", "conv + transformer, initialised from open vocab"),
    ("E2/neural", "closed vocab, last 100 utterances held out", "conv + transformer"),
    ("E3/neural/voiced_only", "train vocalised, test silent", "conv + transformer"),
    ("E4/neural/no_throat", "closed vocab, without throat electrode", "conv + transformer"),
    ("E4/neural/face_only", "closed vocab, 6 face electrodes", "conv + transformer"),
    ("E4/neural/neck_only", "closed vocab, 2 neck electrodes", "conv + transformer"),
    ("E5/neural/random", "open vocab, random utterances", "conv + transformer"),
    ("E5/neural/session", "open vocab, held-out session", "conv + transformer"),
    ("E5/neural/session/shuffled", "open vocab, held-out session", "conv + transformer, shuffled labels (control)"),
]


def summary(groups, cl):
    L = ["### Sequence decoding (CTC), test set", "",
         "| setting | model | seeds | CER % greedy | WER % greedy | CER % grammar | WER % grammar | exact % grammar |",
         "|---|---|---|---|---|---|---|---|"]
    for key, setting, model in CTC_ROWS:
        if key not in groups:
            continue
        rs = groups[key]
        gr = [r["grammar"] for r in rs if "grammar" in r]
        gcell = (lambda k: fmt([x[k] for x in gr])) if gr else (lambda k: "n/a")
        L.append(f"| {setting} | {model} | {len(rs)} | {fmt([r['cer'] for r in rs])} | {fmt([r['wer'] for r in rs])} "
                 f"| {gcell('cer')} | {gcell('wer')} | {gcell('exact')} |")
    L += ["", "### Slot classification, balanced accuracy %", "",
          "| task | chance | Hudgins + logreg (5x5-fold CV) | Hudgins + GBM (CV) | logreg, contiguous folds "
          "| logreg, shuffled labels | duration only | logreg, train vocalised, test silent |",
          "|---|---|---|---|---|---|---|---|"]
    for t in TASKS:
        r = cl["tasks"][t]
        L.append(f"| {t} | {100 * r['chance']:.1f} | {fmt(r['cv_logreg'])} | {fmt(r['cv_gbm'])} | {fmt(r['block_logreg'])} "
                 f"| {fmt(r['shuffled_logreg'])} | {fmt(r['cv_duration_only'])} | {fmt(r['voiced_to_silent_logreg'])} |")
    L += ["", "Same 100 test utterances as the CTC models (E1 split):", "",
          "| task | logreg | GBM | conv + transformer, greedy | conv + transformer, grammar |", "|---|---|---|---|---|"]
    for t in TASKS:
        r = cl["tasks"][t]
        nn = [x["grammar"] for x in groups.get("E1/neural", []) if "grammar" in x]
        g1 = fmt([x["slots_greedy"][t] for x in nn]) if nn else "n/a"
        g2 = fmt([x["slots_grammar"][t] for x in nn]) if nn else "n/a"
        L.append(f"| {t} | {fmt(r['e1split_logreg'])} | {fmt(r['e1split_gbm'])} | {g1} | {g2} |")
    L += ["", "### Electrode ablation, Hudgins + logreg, balanced accuracy % (5x5-fold CV)", "",
          "| electrodes | " + " | ".join(TASKS) + " |", "|---|" + "---|" * len(TASKS)]
    for name, d in cl["ablation"].items():
        L.append(f"| {name} | " + " | ".join(fmt(d[t]) for t in TASKS) + " |")
    (ROOT / "results" / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def _bar(ax, x, vals, color, width, label=None):
    m, s, v = msd(vals)
    ax.bar(x, m, width, color=color, label=label, yerr=s if len(v) > 1 else None, capsize=2,
           error_kw={"lw": 0.8})
    return m


def hero(groups, cl):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    nn = [x["grammar"] for x in groups.get("E1/neural", []) if "grammar" in x]
    w = 0.26
    for i, t in enumerate(TASKS):
        r = cl["tasks"][t]
        _bar(a, i - w, r["shuffled_logreg"], C_CTRL, w, "shuffled-label control (logreg)" if i == 0 else None)
        _bar(a, i, r["cv_logreg"], C_LIN, w, "Hudgins features + logistic regression" if i == 0 else None)
        if nn:
            _bar(a, i + w, [x["slots_grammar"][t] for x in nn], C_NNG, w,
                 "conv + transformer CTC, grammar decoding" if i == 0 else None)
        a.plot([i - 1.6 * w, i + 1.6 * w], [100 * r["chance"]] * 2, color="black", lw=1, ls=":")
    a.set_xticks(range(len(TASKS)), [TASK_LABEL[t] for t in TASKS])
    a.set_ylabel("balanced accuracy (%)")
    a.set_ylim(0, 105)
    a.set_title("Which word was silently mouthed? (dotted: chance)", loc="left")
    a.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    b.axis("off")
    if nn:
        runs = groups["E1/neural"]
        best = sorted(runs, key=lambda r: r["grammar"]["wer"])[len(runs) // 2]
        b.set_title(f"Test prompts and decodings from silent EMG (median seed, grammar WER "
                    f"{100 * best['grammar']['wer']:.0f}%)", loc="left")
        pick = np.random.default_rng(3).choice(len(best["refs"]), 9, replace=False)
        cols = [(0.0, "prompt (spoken form)"), (0.62, "decoded")]
        for x0, h in cols:
            b.text(x0, 0.95, h, weight="bold", transform=b.transAxes, fontsize=8)
        for k, i in enumerate(pick):
            y = 0.85 - k * 0.095
            ref, hyp = best["refs"][i], best["grammar"]["hyps"][i]
            b.text(0.0, y, ref, transform=b.transAxes, family="monospace", fontsize=7.5)
            b.text(0.62, y, hyp, transform=b.transAxes, family="monospace", fontsize=7.5,
                   color="#1b7837" if ref == hyp else "#b2182b")
    fig.tight_layout()
    fig.savefig(FIG / "hero.png")


def results(groups):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [1.6, 1]})
    spec = [("shuffled\ncontrol", "E1/neural/shuffled"), ("linear\nTD", "E1/linear"), ("random\nsplit", "E1/neural"),
            ("open-vocab\ninit", "E6/neural/pretrained"), ("last 100\nheld out", "E2/neural"),
            ("trained on\nvocalised", "E3/neural/voiced_only")]
    spec = [s for s in spec if s[1] in groups]
    ax = axes[0]
    for i, (lab, k) in enumerate(spec):
        rs = groups[k]
        _bar(ax, i - 0.18, [r["cer"] for r in rs], C_CTRL if "shuffled" in k else C_NN, 0.36,
             "greedy (grey: shuffled-label control)" if i == 1 else None)
        gr = [r["grammar"]["cer"] for r in rs if "grammar" in r]
        if gr:
            _bar(ax, i + 0.18, gr, C_NNG, 0.36, "grammar-constrained" if i == 1 else None)
    ax.set_xticks(range(len(spec)), [s[0] for s in spec])
    ax.set_ylabel("character error rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax.set_title("Closed vocabulary, 100 test utterances (one session)", loc="left")
    ax = axes[1]
    spec = [("random\nutterances", "E5/neural/random"), ("held-out\nsession", "E5/neural/session"),
            ("shuffled\ncontrol", "E5/neural/session/shuffled")]
    spec = [s for s in spec if s[1] in groups]
    for i, (lab, k) in enumerate(spec):
        rs = groups[k]
        _bar(ax, i, [r["cer"] for r in rs], C_CTRL if "shuffled" in k else C_NN, 0.6)
        for r in rs:
            ax.scatter(i, 100 * r["cer"], s=10, color="black", zorder=3)
    ax.set_xticks(range(len(spec)), [s[0] for s in spec])
    ax.set_ylim(0, 105)
    ax.set_ylabel("character error rate (%), greedy")
    ax.set_title("Open vocabulary, 190 test sentences", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "results.png")


def ablation(groups, cl):
    names = ["all", "face_only", "neck_only", "throat_only"] + [f"drop_{c}" for c in range(8)]
    labels = {"all": "all 8", "face_only": "face only (6)",
              "neck_only": "neck only (under chin, throat)", "throat_only": "throat only"}
    labels |= {f"drop_{c}": f"without {CHANNEL_SHORT[c]}" for c in range(8)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    y = np.arange(len(names))[::-1]
    h = 0.2
    cols = {"template": "#1b9e77", "ampm": "#d95f02", "weekday": "#7570b3", "month": "#e7298a"}
    for j, t in enumerate(TASKS):
        ms = [msd(cl["ablation"][n][t])[0] - 100 * cl["tasks"][t]["chance"] for n in names]
        ax.barh(y + (1.5 - j) * h, ms, h, color=cols[t], label=t)
    ax.set_yticks(y, [labels[n] for n in names])
    ax.set_xlabel("balanced accuracy above chance (percentage points)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    ax.set_title("Hudgins features + logistic regression (5x5-fold CV)", loc="left")
    ax = axes[1]
    spec = [("all 8", "E1/neural"), ("without\nthroat", "E4/neural/no_throat"), ("face only\n(6)", "E4/neural/face_only"),
            ("neck only\n(2)", "E4/neural/neck_only")]
    spec = [s for s in spec if s[1] in groups]
    for i, (lab, k) in enumerate(spec):
        _bar(ax, i - 0.18, [r["cer"] for r in groups[k]], C_NN, 0.36, "greedy" if i == 0 else None)
        gr = [r["grammar"]["cer"] for r in groups[k] if "grammar" in r]
        if gr:
            _bar(ax, i + 0.18, gr, C_NNG, 0.36, "grammar" if i == 0 else None)
    ax.set_xticks(range(len(spec)), [s[0] for s in spec])
    ax.set_ylabel("test CER (%), retrained per electrode set")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax.set_title("Conv + transformer CTC", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "ablation.png")


def confusion(cl):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
    for ax, t, classes in ((axes[0], "month", MONTHS), (axes[1], "weekday", WEEKDAYS)):
        r = cl["tasks"][t]
        M = np.zeros((len(classes), len(classes)))
        for a, b in zip(r["cv_true"], r["cv_logreg_pred"]):
            M[classes.index(a), classes.index(b)] += 1
        P = M / M.sum(1, keepdims=True)
        ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if M[i, j]:
                    ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if P[i, j] > 0.5 else "black")
        short = [c[:3] for c in classes]
        ax.set_xticks(range(len(classes)), short, rotation=0)
        ax.set_yticks(range(len(classes)), short)
        ax.set_xlabel("predicted")
        ax.set_ylabel("prompted")
        acc = np.trace(M) / M.sum()
        ax.set_title(f"{t}: Hudgins + logreg, 5-fold CV (counts; accuracy {100 * acc:.0f}%, "
                     f"chance {100 / len(classes):.0f}%)", loc="left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "confusion.png")


if __name__ == "__main__":
    runs, groups, cl = load()
    summary(groups, cl)
    confusion(cl)
    for f in (hero, results, ablation):
        f(groups, cl) if f is not results else f(groups)

"""Figure: one silent and one vocalised utterance of the closed vocabulary, raw and filtered."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from silent_emg import CHANNEL_SHORT  # noqa: E402
from silent_emg.data import Corpus  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def main(data=ROOT / "data" / "processed", raw=ROOT / "data" / "raw" / "emg_data"):
    cs, cv = Corpus.load(data, "closed_silent"), Corpus.load(data, "closed_voiced")
    target = cs.text(0)
    j = next((k for k in range(len(cv)) if cv.text(k) == target), 0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True)
    it = cs.index[0]
    raw_x = np.load(raw / "closed_vocab" / "silent" / it["session"] / f"{it['idx']}_emg.npy")
    panels = [(raw_x - raw_x.mean(0), "silent, raw (mean removed)"), (cs.signal(0), "silent, filtered"),
              (cv.signal(j), "vocalised, filtered")]
    for ax, (x, title) in zip(axes, panels):
        t = np.arange(len(x)) / 1000
        for c in range(8):
            scale = np.percentile(np.abs(x[:, c] - np.median(x[:, c])), 99.5) / 0.55 + 1e-9
            y = np.clip(x[:, c] / scale, -0.6, 0.6)
            ax.plot(t, y - 1.2 * c, lw=0.4, color="#1f4e79" if c in (2, 3) else "#444444")
        ax.set_ylim(-1.2 * 7 - 0.7, 0.7)
        ax.set_title(title)
        ax.set_xlabel("time (s)")
    axes[0].set_yticks(-1.2 * np.arange(8), CHANNEL_SHORT)
    fig.suptitle(f'Prompt "{cs.index[0]["text"]}" (closed vocabulary). Blue: neck sites.', y=0.99)
    fig.tight_layout()
    out = ROOT / "figures" / "raw_emg_example.png"
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()

"""Re-decode saved CTC log-probabilities with the closed-vocabulary grammar.

Reads results/runs.jsonl and runs/logprobs/*.npz, writes results/grammar.json with
CER/WER after grammar decoding and slot accuracies for greedy and grammar outputs.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from silent_emg.classical import balanced_accuracy, slot_labels, slots_from_transcript  # noqa: E402
from silent_emg.data import Corpus  # noqa: E402
from silent_emg.grammar import GrammarDecoder  # noqa: E402
from silent_emg.splits import block_split, random_split  # noqa: E402
from silent_emg.text import cer, wer  # noqa: E402


def slot_scores(written, hyps):
    out = {}
    truth = [slot_labels(w) for w in written]
    pred = [slots_from_transcript(h) for h in hyps]
    for task in ("template", "ampm", "weekday", "month"):
        idx = [i for i, t in enumerate(truth) if t[task] is not None]
        out[task] = balanced_accuracy([truth[i][task] for i in idx], [str(pred[i][task]) for i in idx])
    return out


def main():
    cs = Corpus.load(ROOT / "data" / "processed", "closed_silent")
    splits = {"E2": block_split(len(cs), 30, 100)}
    default = random_split(len(cs), 30, 100, seed=0)
    out_path = ROOT / "results" / "grammar.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    decoders = {}
    for line in (ROOT / "results" / "runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        f = ROOT / "runs" / "logprobs" / (r["key"].replace("/", "__") + ".npz")
        if r["key"] in out or "refs" not in r or not f.exists() or r["key"].startswith("E5"):
            continue
        tr, va, te = splits.get(r["key"].split("/")[0], default)
        years = [int(y) for i in tr for y in re.findall(r"\b(\d{4})\b", cs.index[i]["text"])]
        yr = (min(years), max(years))
        g = decoders.setdefault(yr, GrammarDecoder(year_range=yr))
        z = np.load(f)
        lps = [z[f"arr_{k}"] for k in range(len(z.files))]
        hyps = [g.decode(lp) for lp in lps]
        written = [cs.index[i]["text"] for i in te]
        assert [cs.text(i) for i in te] == r["refs"]
        out[r["key"]] = {"cer": cer(r["refs"], hyps), "wer": wer(r["refs"], hyps), "hyps": hyps,
                         "exact": float(np.mean([h == t for h, t in zip(hyps, r["refs"])])),
                         "slots_grammar": slot_scores(written, hyps), "slots_greedy": slot_scores(written, r["hyps"])}
        print(f"{r['key']}: grammar CER {out[r['key']]['cer']:.3f} WER {out[r['key']]['wer']:.3f} "
              f"exact {out[r['key']]['exact']:.2f}", flush=True)
        out_path.write_text(json.dumps(out))


if __name__ == "__main__":
    main()

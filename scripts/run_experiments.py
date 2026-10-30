"""Run every experiment in the README, one training at a time, appending to results/runs.jsonl.

Finished runs are skipped on restart. Before each run the script waits until the
machine reports at least --min-free percent free memory.

Usage: python scripts/run_experiments.py [--only E1 E2 ...] [--seeds 5]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")  # cap the MPS allocator
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.25")
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from silent_emg import FACE_CHANNELS, NECK_CHANNELS  # noqa: E402
from silent_emg.data import Corpus  # noqa: E402
from silent_emg.splits import block_split, check_disjoint, random_split, session_split  # noqa: E402
from silent_emg.sysmem import wait_for_memory  # noqa: E402
from silent_emg.train import Config, evaluate, fit  # noqa: E402

OUT = ROOT / "results" / "runs.jsonl"
PRETRAIN = ROOT / "runs" / "open_vocab_session_s0.pt"
LOGITS = ROOT / "runs" / "logprobs"
ALL = list(range(8))
ABLATIONS = {f"drop_{c}": [k for k in ALL if k != c] for c in ALL} | {
    "face_only": FACE_CHANNELS, "neck_only": NECK_CHANNELS, "throat_only": [3],
    "no_throat": [k for k in ALL if k != 3]}


def done_keys():
    if not OUT.exists():
        return set()
    return {json.loads(line)["key"] for line in OUT.read_text().splitlines() if line.strip()}


def run(key, corpus_tr, tr, corpus_te, va, te, cfg, shuffle=False, keep_hyps=False, extra=None,
        init=None, save=None):
    if key in done_keys() and (save is None or Path(save).exists()):
        return
    wait_for_memory(25.0)
    S = lambda c, ix: [c.signal(i) for i in ix]  # noqa: E731
    T = lambda c, ix: [c.text(i) for i in ix]  # noqa: E731
    ytr, yva = T(corpus_tr, tr), T(corpus_te, va)
    if shuffle:  # break the EMG-text pairing in train and validation; test keeps true labels
        rng = np.random.default_rng(1000 + cfg.seed)
        ytr, yva = list(rng.permutation(ytr)), list(rng.permutation(yva))
    print(f"[{time.strftime('%H:%M:%S')}] {key}: train {len(tr)} val {len(va)} test {len(te)}", flush=True)
    t0 = time.time()
    init_state = torch.load(init, map_location="cpu") if init else None
    model, norm, hist = fit(S(corpus_tr, tr), ytr, S(corpus_te, va), yva, cfg, init_state=init_state)
    if save:
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, save)
    res = evaluate(model, norm, S(corpus_te, te), T(corpus_te, te), cfg)
    rec = {"key": key, "model": cfg.model, "seed": cfg.seed, "channels": cfg.channels, "shuffled": shuffle,
           "cer": res["cer"], "wer": res["wer"], "best_val_cer": min(h["val_cer"] for h in hist),
           "epochs_run": hist[-1]["epoch"], "n_train": len(tr), "n_test": len(te), "seconds": time.time() - t0,
           "history": hist} | (extra or {})
    if keep_hyps:
        rec["refs"], rec["hyps"] = T(corpus_te, te), res["hyps"]
        LOGITS.mkdir(parents=True, exist_ok=True)  # per-frame log-probabilities, for grammar decoding
        np.savez_compressed(LOGITS / (key.replace("/", "__") + ".npz"), *res["logprobs"])
    print(f"    -> test CER {res['cer']:.3f} WER {res['wer']:.3f}", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    del model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    want = lambda e: a.only is None or e in a.only  # noqa: E731
    seeds = range(a.seeds)
    cs = Corpus.load(a.data, "closed_silent")
    rs = random_split(len(cs), 30, 100, seed=0)  # paper protocol: 30 val / 100 test / 370 train
    bs = block_split(len(cs), 30, 100)
    check_disjoint(*rs); check_disjoint(*bs)
    NEURAL = dict(model="neural", epochs=80, lr=1e-3, eval_every=4, patience=6)
    LINEAR = dict(model="linear", epochs=100, eval_every=5, patience=8)
    ss3 = list(seeds)[:3]

    if want("E1"):  # closed vocabulary, random utterance split (paper protocol)
        for s in seeds:
            run(f"E1/neural/s{s}", cs, rs[0], cs, rs[1], rs[2], Config(**NEURAL, seed=s), keep_hyps=True,
                extra={"exp": "E1"})
        for s in ss3:
            run(f"E1/neural/shuffled/s{s}", cs, rs[0], cs, rs[1], rs[2], Config(**NEURAL, seed=s), shuffle=True,
                keep_hyps=True, extra={"exp": "E1"})
            run(f"E1/linear/s{s}", cs, rs[0], cs, rs[1], rs[2], Config(**LINEAR, seed=s), keep_hyps=True,
                extra={"exp": "E1"})

    if want("E5"):  # open vocabulary: held-out session versus random utterances of matched size
        co = Corpus.load(a.data, "open_silent")
        ss = session_split(co.sessions, ["5-10_silent"], ["5-11_silent"])
        perm = np.random.default_rng(0).permutation(len(co))
        nv, nt = len(ss[1]), len(ss[2])
        rr = (np.sort(perm[nv + nt:]), np.sort(perm[:nv]), np.sort(perm[nv:nv + nt]))
        check_disjoint(*ss); check_disjoint(*rr)
        # training utterances longer than 15 s (about 4%) are skipped to bound memory; val/test keep everything
        short = lambda ix: np.array([i for i in ix if co.index[i]["length"] <= 15000])  # noqa: E731
        ss, rr = (short(ss[0]), ss[1], ss[2]), (short(rr[0]), rr[1], rr[2])
        OPEN = NEURAL | dict(epochs=40, eval_every=2, patience=6)
        for s in range(2):
            for split_name, sp in (("session", ss), ("random", rr)):
                save = PRETRAIN if (split_name, s) == ("session", 0) else None
                run(f"E5/neural/{split_name}/s{s}", co, sp[0], co, sp[1], sp[2], Config(**OPEN, seed=s),
                    extra={"exp": "E5", "split": split_name}, save=save)
        run("E5/neural/session/shuffled/s0", co, ss[0], co, ss[1], ss[2], Config(**OPEN, seed=0), shuffle=True,
            extra={"exp": "E5", "split": "session"})

    if want("E6"):  # closed vocabulary, initialised from the open-vocabulary model of E5 (seed 0)
        if not PRETRAIN.exists():
            raise SystemExit("run E5 first: it writes " + str(PRETRAIN))
        for s in ss3:
            run(f"E6/neural/pretrained/s{s}", cs, rs[0], cs, rs[1], rs[2], Config(**NEURAL, seed=s) ,
                init=str(PRETRAIN), keep_hyps=True, extra={"exp": "E6"})

    if want("E3"):  # train on vocalised EMG, test on silent EMG (same test utterances as E1)
        cv = Corpus.load(a.data, "closed_voiced")
        held = {cs.index[i]["text"] for i in np.concatenate([rs[1], rs[2]])}
        vtr = np.array([i for i in range(len(cv)) if cv.index[i]["text"] not in held])
        for s in ss3:
            run(f"E3/neural/voiced_only/s{s}", cv, vtr, cs, rs[1], rs[2], Config(**NEURAL, seed=s),
                keep_hyps=True, extra={"exp": "E3", "train": "voiced"})

    if want("E2"):  # closed vocabulary, contiguous block split (later recordings held out)
        for s in ss3:
            run(f"E2/neural/s{s}", cs, bs[0], cs, bs[1], bs[2], Config(**NEURAL, seed=s), keep_hyps=True,
                extra={"exp": "E2"})

    if want("E4"):  # electrode groups, closed vocabulary random split
        for name in ("no_throat", "face_only", "neck_only"):
            for s in ss3:
                run(f"E4/neural/{name}/s{s}", cs, rs[0], cs, rs[1], rs[2],
                    Config(**NEURAL, seed=s, channels=ABLATIONS[name]), keep_hyps=True,
                    extra={"exp": "E4", "ablation": name})



if __name__ == "__main__":
    main()

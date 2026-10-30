"""Classical baselines: Hudgins features + logistic regression / gradient boosting on four slot tasks.

Writes results/classical.json. Runs in about a minute on a laptop CPU.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from silent_emg import FACE_CHANNELS, NECK_CHANNELS  # noqa: E402
from silent_emg.classical import balanced_accuracy, make_classifier, slot_labels, utterance_features  # noqa: E402
from silent_emg.data import Corpus  # noqa: E402
from silent_emg.splits import random_split  # noqa: E402

TASKS = ["template", "ampm", "weekday", "month"]
ALL = list(range(8))
ABLATIONS = {"all": ALL, "no_throat": [k for k in ALL if k != 3], "face_only": FACE_CHANNELS,
             "neck_only": NECK_CHANNELS, "throat_only": [3]} | {f"drop_{c}": [k for k in ALL if k != c] for c in ALL}


def features(corpus, channels=None):
    return np.stack([utterance_features(corpus.signal(i), channels=channels) for i in range(len(corpus))])


def cv_scores(X, y, kind, repeats=5, block=False, shuffle_labels=False):
    """Balanced accuracy per repeat of 5-fold CV (stratified random folds, or contiguous blocks)."""
    out, preds = [], None
    for r in range(repeats):
        rng = np.random.default_rng(100 + r)
        folds = (KFold(5).split(X) if block else StratifiedKFold(5, shuffle=True, random_state=r).split(X, y))
        pred = np.empty(len(y), dtype=object)
        for tr, te in folds:
            ytr = rng.permutation(y[tr]) if shuffle_labels else y[tr]
            clf = make_classifier(kind, seed=r).fit(X[tr], ytr)
            pred[te] = clf.predict(X[te])
        out.append(balanced_accuracy(y, pred))
        preds = pred if preds is None else preds
        if block:
            break  # contiguous folds are deterministic
    return out, preds


def main(data=ROOT / "data" / "processed"):
    cs = Corpus.load(data, "closed_silent")
    cv_ = Corpus.load(data, "closed_voiced")
    labels = [slot_labels(it["text"]) for it in cs.index]
    vlabels = [slot_labels(it["text"]) for it in cv_.index]
    X = features(cs)
    Xv = features(cv_)
    tr, va, te = random_split(len(cs), 30, 100, seed=0)
    trv = np.concatenate([tr, va])
    held = {cs.index[i]["text"] for i in np.concatenate([va, te])}
    res = {"tasks": {}, "ablation": {}}
    for task in TASKS:
        idx = np.array([i for i in range(len(cs)) if labels[i][task] is not None])
        y = np.array([labels[i][task] for i in idx], dtype=object)
        k = len(set(y))
        r = {"n": int(len(idx)), "classes": sorted(set(y)), "chance": 1 / k}
        for kind in ("logreg", "gbm"):
            r[f"cv_{kind}"], pred = cv_scores(X[idx], y, kind)
            if kind == "logreg":
                r["cv_logreg_pred"], r["cv_true"] = list(pred), list(y)
            r[f"block_{kind}"], _ = cv_scores(X[idx], y, kind, block=True)
            r[f"shuffled_{kind}"], _ = cv_scores(X[idx], y, kind, shuffle_labels=True)
        r["cv_duration_only"], _ = cv_scores(X[idx][:, -1:], y, "logreg")
        # same 100-utterance test set as the CTC models (train on the other 400)
        m_tr = np.isin(idx, trv); m_te = np.isin(idx, te)
        for kind in ("logreg", "gbm"):
            accs = []
            for s in range(5 if kind == "gbm" else 1):
                clf = make_classifier(kind, seed=s).fit(X[idx][m_tr], y[m_tr])
                accs.append(balanced_accuracy(y[m_te], clf.predict(X[idx][m_te])))
            r[f"e1split_{kind}"] = accs
        # train on vocalised EMG (prompts not in the silent val/test sets), test on silent test set
        vidx = np.array([i for i in range(len(cv_)) if vlabels[i][task] is not None and cv_.index[i]["text"] not in held])
        yv = np.array([vlabels[i][task] for i in vidx], dtype=object)
        clf = make_classifier("logreg").fit(Xv[vidx], yv)
        r["voiced_to_silent_logreg"] = [balanced_accuracy(y[m_te], clf.predict(X[idx][m_te]))]
        res["tasks"][task] = r
        print(task, {kk: (round(float(np.mean(v)), 3) if isinstance(v, list) and v and isinstance(v[0], float) else v)
                     for kk, v in r.items() if kk not in ("cv_logreg_pred", "cv_true", "classes")})
    for name, ch in ABLATIONS.items():
        Xa = X if name == "all" else features(cs, channels=ch)
        res["ablation"][name] = {}
        for task in TASKS:
            idx = np.array([i for i in range(len(cs)) if labels[i][task] is not None])
            y = np.array([labels[i][task] for i in idx], dtype=object)
            res["ablation"][name][task], _ = cv_scores(Xa[idx], y, "logreg")
        print(name, {t: round(float(np.mean(v)), 3) for t, v in res["ablation"][name].items()})
    (ROOT / "results" / "classical.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

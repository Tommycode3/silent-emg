import json

import numpy as np

from silent_emg.data import Corpus, build_cache, scan


def _fake_raw(root):
    d = root / "closed_vocab" / "silent" / "5-19_silent"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    texts = ["08:11 PM", "", "July 07 2018", "Monday June 30"]
    for i, t in enumerate(texts):
        np.save(d / f"{i}_emg.npy", rng.normal(size=(600 + 100 * i, 8)))
        info = {"text": t, "book": "datedata.txt", "sentence_index": i if t else -1}
        (d / f"{i}_info.json").write_text(json.dumps(info))
    return root


def test_scan_skips_pauses(tmp_path):
    items = scan(_fake_raw(tmp_path / "raw"), "closed_silent")
    assert [it["idx"] for it in items] == [0, 2, 3]


def test_build_cache_roundtrip(tmp_path):
    raw = _fake_raw(tmp_path / "raw")
    build_cache(raw, tmp_path / "proc", "closed_silent")
    c = Corpus.load(tmp_path / "proc", "closed_silent")
    assert len(c) == 3
    assert c.signal(1).shape == (800, 8)
    assert c.text(0) == "08 11 pm"
    assert isinstance(c.emg, np.memmap)

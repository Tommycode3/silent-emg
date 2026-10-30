"""Loading the Gaddy & Klein (2020) EMG recordings into a memory-mapped cache.

Raw layout (Zenodo record 10.5281/zenodo.4064409):
    emg_data/<subset>/<session>/<i>_emg.npy   (T, 8) float64 at 1000 Hz
    emg_data/<subset>/<session>/<i>_info.json  {"text", "book", "sentence_index", ...}
Entries with sentence_index < 0 are pauses between prompts and are skipped.

The cache stores every utterance of a subset, filtered, as one float32 array
(data/processed/<name>/emg.f32) plus an index (index.json) with offsets.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import N_CHANNELS
from .preprocess import preprocess
from .text import clean_text

SUBSETS = {
    "closed_silent": ["closed_vocab/silent"],
    "closed_voiced": ["closed_vocab/voiced"],
    "open_silent": ["silent_parallel_data"],
}


def _utterance_ids(session_dir: Path) -> list[int]:
    ids = [int(m.group(1)) for f in os.listdir(session_dir) if (m := re.match(r"(\d+)_info\.json$", f))]
    return sorted(ids)


def scan(raw_root: Path, subset: str) -> list[dict]:
    """List valid utterances of a subset in recording order."""
    items = []
    for sub in SUBSETS[subset]:
        for session_dir in sorted((Path(raw_root) / sub).iterdir()):
            if not session_dir.is_dir():
                continue
            for i in _utterance_ids(session_dir):
                info = json.loads((session_dir / f"{i}_info.json").read_text())
                if info.get("sentence_index", -1) < 0 or not clean_text(info.get("text", "")):
                    continue
                items.append({"session": session_dir.name, "dir": str(session_dir), "idx": i,
                              "text": info["text"], "book": info.get("book", ""),
                              "sentence_index": info["sentence_index"]})
    return items


def build_cache(raw_root: Path, out_dir: Path, subset: str) -> Path:
    """Filter every utterance once and write the memory-mapped cache."""
    out = Path(out_dir) / subset
    out.mkdir(parents=True, exist_ok=True)
    items = scan(raw_root, subset)

    def load(d, i):
        f = Path(d) / f"{i}_emg.npy"
        return np.load(f) if f.exists() else None

    lengths = [np.load(Path(it["dir"]) / f"{it['idx']}_emg.npy", mmap_mode="r").shape[0] for it in items]
    total = int(sum(lengths))
    mm = np.lib.format.open_memmap(out / "emg.npy", mode="w+", dtype=np.float32, shape=(total, N_CHANNELS))
    offset = 0
    index = []
    for it, n in zip(items, lengths):
        x = load(it["dir"], it["idx"])
        y = preprocess(x, load(it["dir"], it["idx"] - 1), load(it["dir"], it["idx"] + 1))
        mm[offset: offset + n] = y
        index.append({k: it[k] for k in ("session", "idx", "text", "book", "sentence_index")}
                     | {"offset": offset, "length": n})
        offset += n
    mm.flush()
    del mm
    (out / "index.json").write_text(json.dumps(index))
    return out


@dataclass
class Corpus:
    """Memory-mapped filtered EMG with per-utterance metadata."""

    emg: np.ndarray  # (total_T, 8) float32, memory-mapped
    index: list[dict]

    @classmethod
    def load(cls, cache_dir: Path, subset: str) -> "Corpus":
        d = Path(cache_dir) / subset
        return cls(np.load(d / "emg.npy", mmap_mode="r"), json.loads((d / "index.json").read_text()))

    def __len__(self) -> int:
        return len(self.index)

    def signal(self, i: int) -> np.ndarray:
        it = self.index[i]
        return np.asarray(self.emg[it["offset"]: it["offset"] + it["length"]])

    def text(self, i: int) -> str:
        return clean_text(self.index[i]["text"])

    @property
    def sessions(self) -> list[str]:
        return [it["session"] for it in self.index]

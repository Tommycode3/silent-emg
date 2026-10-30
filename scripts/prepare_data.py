"""Filter the raw Gaddy & Klein (2020) recordings once and cache them as memory-mapped arrays.

Usage: python scripts/prepare_data.py --raw data/raw/emg_data --out data/processed
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from silent_emg.data import SUBSETS, build_cache  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/emg_data")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    a = ap.parse_args()
    for s in a.subsets:
        t = time.time()
        d = build_cache(Path(a.raw), Path(a.out), s)
        print(f"{s}: {d} ({time.time() - t:.0f}s)")

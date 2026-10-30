import numpy as np
import pytest

from silent_emg.splits import block_split, check_disjoint, random_split, session_split


def test_random_split_partition():
    tr, va, te = random_split(500, 30, 100, seed=0)
    assert (len(tr), len(va), len(te)) == (370, 30, 100)
    check_disjoint(tr, va, te)
    assert sorted(np.concatenate([tr, va, te]).tolist()) == list(range(500))


def test_block_split_is_ordered():
    tr, va, te = block_split(100, 10, 20)
    assert tr.max() < va.min() and va.max() < te.min()


def test_session_split_holds_out_whole_sessions():
    s = ["a"] * 5 + ["b"] * 4 + ["c"] * 3
    tr, va, te = session_split(s, ["b"], ["c"])
    assert set(np.asarray(s)[tr]) == {"a"} and set(np.asarray(s)[te]) == {"c"}
    check_disjoint(tr, va, te)


def test_overlap_detected():
    with pytest.raises(ValueError):
        check_disjoint([1, 2], [2, 3])

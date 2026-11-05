"""时间同步与插值的单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.time_alignment import (find_gaps, interp_with_validity, overlap_window,
                                sampling_stats)


def test_overlap_window():
    t0, t1 = overlap_window(np.array([0.0, 5.0]), np.array([1.0, 4.0]))
    assert (t0, t1) == (1.0, 4.0)


def test_interp_linear_and_bounds():
    t_src = np.array([0.0, 1.0, 2.0])
    y_src = np.array([[0.0], [1.0], [2.0]])
    t_dst = np.array([0.5, 1.5])
    y, valid = interp_with_validity(t_src, y_src, t_dst, max_gap_s=2.0)
    np.testing.assert_allclose(y.ravel(), [0.5, 1.5])
    assert valid.all()


def test_interp_marks_outside_as_invalid():
    t_src = np.array([0.0, 1.0, 2.0])
    y_src = np.array([[0.0], [1.0], [2.0]])
    t_dst = np.array([-0.1, 2.1])
    _, valid = interp_with_validity(t_src, y_src, t_dst)
    assert not valid.any()


def test_interp_marks_large_gap_as_invalid():
    t_src = np.array([0.0, 0.1, 5.0])
    y_src = np.array([[0.0], [0.1], [5.0]])
    t_dst = np.array([3.0, 0.05])
    _, valid = interp_with_validity(t_src, y_src, t_dst, max_gap_s=0.5)
    assert valid.tolist() == [False, True]


def test_find_gaps():
    t = np.array([0.0, 0.1, 0.2, 1.2])
    gaps = find_gaps(t, 0.5)
    assert gaps.shape == (1, 4)
    assert abs(gaps[0, 3] - 1.0) < 1e-9


def test_sampling_stats():
    t = np.arange(0.0, 1.0, 0.01)
    stats = sampling_stats(t)
    assert abs(stats["rate_hz"] - 100.0) < 1e-6
    assert stats["n"] == 100

"""测距/定位指标计算的单元测试（合成数据，已知真值）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ground_truth_ranges import mapping_check, true_ranges
from src.joint_analysis import dop
from src.localization_metrics import position_errors, scale_diagnostic, summary
from src.ranging_metrics import jump_indices, metrics_table, per_timestamp_rms, range_errors


ANCHORS = np.array([[-0.4, -0.4, 0.0], [-0.4, 0.4, 0.0],
                    [0.4, 0.4, 0.0], [0.4, -0.4, 0.0]])


def test_true_ranges_from_origin():
    d = true_ranges(np.zeros((1, 3)), ANCHORS)
    np.testing.assert_allclose(d[0], [np.sqrt(0.32)] * 4, atol=1e-9)


def test_true_ranges_shifted_point():
    p = np.array([[0.0, 0.0, 1.0]])
    d = true_ranges(p, ANCHORS)
    np.testing.assert_allclose(d[0], [np.sqrt(0.32 + 1.0)] * 4, atol=1e-9)


def test_range_metrics_known_bias_and_rmse():
    truth = np.full((10, 4), 2.0)
    meas = truth + 0.1
    err = range_errors(meas, truth)
    table = metrics_table(err, np.ones_like(truth, bool), np.zeros_like(truth, bool))
    assert np.allclose(table["bias_m"], 0.1)
    assert np.allclose(table["rmse_m"], 0.1)
    assert np.allclose(table["mae_m"], 0.1)


def test_jump_detection():
    meas = np.array([[1.0], [1.0], [1.6], [1.6]])
    valid = np.ones_like(meas, bool)
    jumps = jump_indices(meas, valid, threshold_m=0.5)
    assert jumps[:, 0].tolist() == [False, False, True, False]


def test_localization_summary_and_scale_diagnostic():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(200, 3))
    uwb = truth.copy()
    uwb[:, 0] = truth[:, 0] / 0.5  # x 被放大 2 倍
    err = position_errors(uwb, truth)
    stats = summary(err).set_index("component")
    assert abs(stats.loc["z", "rmse_m"]) < 1e-9
    diag = scale_diagnostic(uwb, truth).set_index("axis")
    assert abs(diag.loc["x", "slope_k"] - 0.5) < 1e-9
    assert abs(diag.loc["x", "implied_scale_1_over_k"] - 2.0) < 1e-9


def test_mapping_check_identity():
    truth = np.array([[1.0, 2.0, 3.0, 4.0]])
    meas = truth + 0.05
    out = mapping_check(meas, truth)
    assert out["identity_is_best"]
    assert out["improvement_m"] == 0.0


def test_dop_coplanar_is_vertically_weak():
    pos = np.array([[-1.0, 0.0, 0.1], [0.0, 0.0, 0.1]])
    d = dop(pos, ANCHORS)
    assert (d["vdop"] > d["hdop"]).all()
    assert np.isfinite(d[["hdop", "vdop", "gdop"]].to_numpy()).all()


def test_per_timestamp_rms_ignores_invalid():
    err = np.array([[0.2, np.nan, np.nan, np.nan], [0.4, 0.4, np.nan, np.nan]])
    valid = np.isfinite(err)
    rms = per_timestamp_rms(err, valid)
    np.testing.assert_allclose(rms, [0.2, 0.4])

"""端到端数据管线测试（使用仓库内真实数据）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import anchor_geometry as geom
from src import data_loader as dl
from src.coordinate_transform import UWB_TO_MAVROS
from src.ground_truth_ranges import true_ranges

CFG_PATH = ROOT / "configs" / "config.yaml"


@pytest.mark.skipif(not CFG_PATH.exists(), reason="config not found")
def test_build_dataset_real_data():
    cfg = dl.load_config(CFG_PATH)
    ds = dl.build_dataset(cfg)

    assert ds.n > 100
    assert ds.uwb_mav.shape == (ds.n, 3)
    assert ds.ranges_meas.shape == (ds.n, 4)
    assert ds.ranges_true.shape == (ds.n, 4)

    np.testing.assert_allclose(ds.truth_abs - ds.p0, ds.truth_rel, atol=1e-12)
    np.testing.assert_allclose(ds.uwb_mav @ UWB_TO_MAVROS, ds.uwb_uwb, atol=1e-9)

    assert np.all(ds.ranges_true > 0)
    assert ds.valid_time.mean() > 0.99

    check = geom.check_geometry(ds.anchors_world)
    assert check["ok"]
    expected = true_ranges(ds.truth_abs, ds.anchors_world)
    np.testing.assert_allclose(ds.ranges_true, expected, atol=1e-12)


@pytest.mark.skipif(not CFG_PATH.exists(), reason="config not found")
def test_platform_is_static():
    cfg = dl.load_config(CFG_PATH)
    ds = dl.build_dataset(cfg)
    assert np.max(ds.p0_std) < 1e-3

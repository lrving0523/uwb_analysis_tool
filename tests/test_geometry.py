"""几何与坐标变换的单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import anchor_geometry as geom
from src.coordinate_transform import (UWB_TO_MAVROS, check_orthogonality,
                                      mavros_to_uwb, uwb_to_mavros)


def test_transform_is_orthogonal_reflection():
    info = check_orthogonality()
    assert info["max_|A A^T - I|"] < 1e-12
    assert abs(info["det(A)"] + 1.0) < 1e-12


def test_transform_axis_mapping():
    p = np.array([[1.0, 2.0, 3.0]])
    out = uwb_to_mavros(p)
    np.testing.assert_allclose(out, [[2.0, -1.0, -3.0]])
    np.testing.assert_allclose(mavros_to_uwb(out), p)


def test_default_anchor_geometry():
    a = geom.DEFAULT_ANCHORS_UWB
    check = geom.check_geometry(a)
    assert check["ok"], check
    np.testing.assert_allclose(check["adjacent_distances_m"], [0.8] * 4, atol=1e-9)
    np.testing.assert_allclose(check["diagonal_distances_m"], [0.8 * np.sqrt(2)] * 2, atol=1e-9)
    assert check["clockwise_xy"]
    assert check["planarity_z_spread_m"] < 1e-12


def test_anchors_world_translation():
    center = np.array([1.0, 2.0, 3.0])
    rel = geom.anchors_relative_mavros(geom.DEFAULT_ANCHORS_UWB)
    world = geom.anchors_world(center, rel)
    np.testing.assert_allclose(world.mean(axis=0), center, atol=1e-9)
    d = geom.pairwise_distances(world)
    np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-12)


def test_clockwise_order_after_transform():
    rel = geom.anchors_relative_mavros(geom.DEFAULT_ANCHORS_UWB)
    assert geom.signed_area_xy(rel) < 0  # 俯视仍为顺时针

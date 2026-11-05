"""四基站固定几何布局：定义、坐标变换与自检。

平台局部 UWB 坐标系：原点在平台中心，X 向右、Y 向上、Z 垂直平台面；
俯视平台时 A0 → A1 → A2 → A3 顺时针排列，间距 0.8 m（对角 1.1314 m）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.coordinate_transform import uwb_to_mavros

#: 默认平台局部 UWB 坐标（单位 m，A0→A3 顺时针）
DEFAULT_ANCHORS_UWB = np.array([
    [-0.4, -0.4, 0.0],
    [-0.4, 0.4, 0.0],
    [0.4, 0.4, 0.0],
    [0.4, -0.4, 0.0],
], dtype=float)

ANCHOR_NAMES = ("A0", "A1", "A2", "A3")


def anchors_uwb_from_config(cfg: dict) -> np.ndarray:
    """从配置读取 UWB 坐标系下的四基站坐标。"""
    raw = cfg.get("geometry", {}).get("anchors_uwb")
    anchors = np.asarray(raw, dtype=float) if raw is not None else DEFAULT_ANCHORS_UWB.copy()
    if anchors.shape != (4, 3):
        raise ValueError(f"anchors_uwb 必须是 4x3，当前 {anchors.shape}")
    return anchors


def anchors_relative_mavros(anchors_uwb: np.ndarray) -> np.ndarray:
    """四基站相对平台中心的位置（MAVROS 轴约定）。"""
    return uwb_to_mavros(anchors_uwb)


def anchors_world(platform_center: np.ndarray, anchors_rel_mavros: np.ndarray) -> np.ndarray:
    """四基站在动捕世界系下的绝对位置。"""
    center = np.asarray(platform_center, dtype=float).reshape(1, 3)
    return center + np.asarray(anchors_rel_mavros, dtype=float)


def pairwise_distances(anchors: np.ndarray) -> np.ndarray:
    """4x4 基站间距矩阵。"""
    a = np.asarray(anchors, float)
    return np.linalg.norm(a[:, None, :] - a[None, :, :], axis=2)


def signed_area_xy(anchors: np.ndarray) -> float:
    """XY 投影的有向面积；负值表示俯视顺时针顺序。"""
    a = np.asarray(anchors, float)
    x, y = a[:, 0], a[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def check_geometry(anchors: np.ndarray, expected_side: float = 0.8,
                   expected_diagonal: float = 0.8 * np.sqrt(2.0),
                   tol: float = 5e-3) -> dict:
    """检查基站间距、对角线、共面性与顺序，返回结构化结果。"""
    a = np.asarray(anchors, float)
    d = pairwise_distances(a)
    adjacent = [float(d[i, (i + 1) % 4]) for i in range(4)]
    diagonals = [float(d[0, 2]), float(d[1, 3])]
    centroid = a.mean(axis=0)
    z_spread = float(np.max(np.abs(a[:, 2] - centroid[2])))

    result = {
        "adjacent_distances_m": adjacent,
        "diagonal_distances_m": diagonals,
        "expected_side_m": expected_side,
        "expected_diagonal_m": expected_diagonal,
        "max_adjacent_error_m": float(np.max(np.abs(np.array(adjacent) - expected_side))),
        "max_diagonal_error_m": float(np.max(np.abs(np.array(diagonals) - expected_diagonal))),
        "planarity_z_spread_m": z_spread,
        "signed_area_xy_m2": signed_area_xy(a),
        "clockwise_xy": bool(signed_area_xy(a) < 0),
        "centroid_m": centroid.tolist(),
        "side_ok": bool(np.allclose(adjacent, expected_side, atol=tol)),
        "diagonal_ok": bool(np.allclose(diagonals, expected_diagonal, atol=tol)),
        "planar_ok": bool(z_spread < tol),
    }
    result["ok"] = bool(result["side_ok"] and result["diagonal_ok"] and result["planar_ok"])
    return result


def anchor_table(anchors_world: np.ndarray, anchors_uwb: np.ndarray | None = None) -> pd.DataFrame:
    """基站位置表（世界系 + 可选 UWB 局部坐标）。"""
    a = np.asarray(anchors_world, float)
    data = {
        "anchor": list(ANCHOR_NAMES),
        "x_world_m": a[:, 0],
        "y_world_m": a[:, 1],
        "z_world_m": a[:, 2],
    }
    if anchors_uwb is not None:
        u = np.asarray(anchors_uwb, float)
        data.update({"x_uwb_m": u[:, 0], "y_uwb_m": u[:, 1], "z_uwb_m": u[:, 2]})
    return pd.DataFrame(data)

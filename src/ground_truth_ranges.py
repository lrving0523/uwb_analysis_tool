"""由动捕真值计算四基站测距真值，并核对 dis_arr 与 A0–A3 的对应关系。"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.anchor_geometry import ANCHOR_NAMES


def true_ranges(tag_positions: np.ndarray, anchors_world: np.ndarray) -> np.ndarray:
    """d_i(t) = ||p_tag(t) - a_i||，返回 (N,4)。"""
    p = np.asarray(tag_positions, float).reshape(-1, 3)
    a = np.asarray(anchors_world, float).reshape(-1, 3)
    return np.linalg.norm(p[:, None, :] - a[None, :, :], axis=2)


def ranges_table(t: np.ndarray, truth: np.ndarray, meas: np.ndarray,
                 valid: np.ndarray) -> pd.DataFrame:
    """逐帧测距真值/实测值表（长表：每帧 4 行）。"""
    rows = []
    for i, name in enumerate(ANCHOR_NAMES):
        rows.append(pd.DataFrame({
            "t": t,
            "anchor": name,
            "range_true_m": truth[:, i],
            "range_meas_m": meas[:, i],
            "valid": valid[:, i],
        }))
    return pd.concat(rows, ignore_index=True)


def mapping_check(meas: np.ndarray, pred: np.ndarray,
                  valid: np.ndarray | None = None) -> dict:
    """核对 ``dis_arr[i]`` 是否就是 A_i：比较单位映射与最优排列的平均绝对残差。"""
    m = np.asarray(meas, float)
    p = np.asarray(pred, float)
    if valid is not None:
        v = np.asarray(valid, bool)
    else:
        v = np.isfinite(m) & (m > 0.05)

    identity = float(np.nanmean(np.abs(m[v] - p[v])))
    best_perm, best_mae = (0, 1, 2, 3), identity
    for perm in itertools.permutations(range(4)):
        if perm == (0, 1, 2, 3):
            continue
        mae = float(np.nanmean(np.abs(m[v] - p[:, list(perm)][v])))
        if mae < best_mae:
            best_perm, best_mae = perm, mae

    return {
        "identity_order": (0, 1, 2, 3),
        "identity_mae_m": identity,
        "best_permutation": list(best_perm),
        "best_mae_m": best_mae,
        "identity_is_best": bool(best_perm == (0, 1, 2, 3)),
        "improvement_m": float(identity - best_mae),
    }

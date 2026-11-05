"""UWB 三维定位精度指标与尺度诊断。

注意：本模块不做任何平移/旋转/缩放配准。``scale_diagnostic`` 只用于说明
观测到的系统性比例失配，其拟合参数不会被应用到数据上。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS = ("x", "y", "z", "xy", "3d")


def position_errors(uwb_mav: np.ndarray, truth_rel: np.ndarray) -> pd.DataFrame:
    """逐帧定位误差：三轴、水平、三维。"""
    u = np.asarray(uwb_mav, float)
    g = np.asarray(truth_rel, float)
    e = u - g
    return pd.DataFrame({
        "ex": e[:, 0],
        "ey": e[:, 1],
        "ez": e[:, 2],
        "e_xy": np.linalg.norm(e[:, :2], axis=1),
        "e_3d": np.linalg.norm(e, axis=1),
    })


def summary(err: pd.DataFrame, percentile: float = 95.0) -> pd.DataFrame:
    """各误差分量的 RMSE/MAE/Bias/Std/Max/P95。"""
    rows = []
    for name in COMPONENTS:
        key = f"e{name}" if name in ("x", "y", "z") else f"e_{name}"
        e = err[key].to_numpy(float)
        e = e[np.isfinite(e)]
        if e.size == 0:
            continue
        rows.append({
            "component": name,
            "n": int(e.size),
            "rmse_m": float(np.sqrt(np.mean(e ** 2))),
            "mae_m": float(np.mean(np.abs(e))),
            "bias_m": float(np.mean(e)),
            "std_m": float(np.std(e)),
            "max_abs_m": float(np.max(np.abs(e))),
            f"p{int(percentile)}_abs_m": float(np.percentile(np.abs(e), percentile)),
        })
    return pd.DataFrame(rows)


def scale_diagnostic(uwb_mav: np.ndarray, truth_rel: np.ndarray) -> pd.DataFrame:
    """逐轴线性回归 truth ≈ k·uwb + b，用于判断是否存在系统性比例失配。

    仅作诊断报告，不用于修正数据。
    """
    u = np.asarray(uwb_mav, float)
    g = np.asarray(truth_rel, float)
    rows = []
    for i, axis in enumerate(("x", "y", "z")):
        k, b = np.polyfit(u[:, i], g[:, i], 1)
        pred = k * u[:, i] + b
        ss_res = float(np.sum((g[:, i] - pred) ** 2))
        ss_tot = float(np.sum((g[:, i] - np.mean(g[:, i])) ** 2))
        rows.append({
            "axis": axis,
            "slope_k": float(k),
            "intercept_b_m": float(b),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "implied_scale_1_over_k": float(1.0 / k) if abs(k) > 1e-9 else float("inf"),
            "corr": float(np.corrcoef(u[:, i], g[:, i])[0, 1]),
        })
    return pd.DataFrame(rows)

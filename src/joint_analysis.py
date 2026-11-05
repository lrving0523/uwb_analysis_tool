"""测距误差与定位误差的联合分析，以及基于真实几何的 DOP 计算。

原则：只做相关性与几何条件分析，不据此推断因果；不做任何配准或尺度修正。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ranging_metrics import per_timestamp_rms


def jacobian(tag_pos: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """测距雅可比 H，(N,4,3)：第 i 行是指向基站 i 的单位向量。"""
    p = np.asarray(tag_pos, float).reshape(-1, 3)
    a = np.asarray(anchors, float).reshape(-1, 3)
    d = p[:, None, :] - a[None, :, :]
    r = np.linalg.norm(d, axis=2, keepdims=True)
    return d / np.maximum(r, 1e-9)


def dop(tag_pos: np.ndarray, anchors: np.ndarray) -> pd.DataFrame:
    """由几何雅可比计算 HDOP / VDOP / PDOP，并给出秩与条件数。

    约定（3 维位置、无钟差参数）：
        G = (HᵀH)⁻¹,  HDOP=√(Gxx+Gyy),  VDOP=√(Gzz),  PDOP=√tr(G)
    因此 PDOP² = HDOP² + VDOP²。PDOP 与 GDOP 在纯位置解算下数值相同，
    为与论文术语一致，这里显式给出 ``pdop``，``gdop`` 保留为同义列。
    """
    h = jacobian(tag_pos, anchors)
    rows = []
    for k in range(h.shape[0]):
        hk = h[k]
        rank = int(np.linalg.matrix_rank(hk))
        try:
            g = np.linalg.inv(hk.T @ hk)
            hdop = float(np.sqrt(g[0, 0] + g[1, 1]))
            vdop = float(np.sqrt(g[2, 2]))
            pdop = float(np.sqrt(np.trace(g)))
            rows.append({
                "hdop": hdop,
                "vdop": vdop,
                "pdop": pdop,
                "gdop": pdop,
                "rank": rank,
                "cond": float(np.linalg.cond(hk.T @ hk)),
            })
        except np.linalg.LinAlgError:
            rows.append({"hdop": np.nan, "vdop": np.nan, "pdop": np.nan,
                         "gdop": np.nan, "rank": rank, "cond": np.inf})
    return pd.DataFrame(rows)


def joint_table(ds, range_err: np.ndarray) -> pd.DataFrame:
    """把每帧的测距误差与定位误差拼到同一张表（时间同步由 Dataset 保证）。"""
    from src.localization_metrics import position_errors

    err = position_errors(ds.uwb_mav, ds.truth_rel)
    df = pd.DataFrame({"t": ds.t, "t_rel": ds.t_rel})
    for i in range(4):
        df[f"range_err_A{i}"] = range_err[:, i]
    df["range_rms"] = per_timestamp_rms(range_err, ds.valid_range)
    df["eop_z"] = ds.eop[:, 2]
    df["uwb_z"] = ds.uwb_mav[:, 2]
    df["truth_z"] = ds.truth_rel[:, 2]
    for col in err.columns:
        df[col] = err[col].to_numpy()
    return df


def correlation_table(joint: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """各基站测距误差与各定位误差分量的相关系数。"""
    from scipy.stats import pearsonr, spearmanr

    func = pearsonr if method == "pearson" else spearmanr
    comps = ["ex", "ey", "ez", "e_xy", "e_3d"]
    rows = []
    for i in range(4):
        row = {"anchor": f"A{i}", "predictor": f"range_err_A{i}"}
        for comp in comps:
            x = joint[f"range_err_A{i}"].to_numpy(float)
            y = joint[comp].to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() > 5:
                r, p = func(x[mask], y[mask])
            else:
                r, p = np.nan, np.nan
            row[f"r_{comp}"] = float(r)
            row[f"p_{comp}"] = float(p)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_correlations(joint: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """综合测距误差（每帧 RMS）与定位误差分量的相关系数。"""
    from scipy.stats import pearsonr, spearmanr

    func = pearsonr if method == "pearson" else spearmanr
    rows = []
    for comp in ("ex", "ey", "ez", "e_xy", "e_3d"):
        x = joint["range_rms"].to_numpy(float)
        y = joint[comp].to_numpy(float)
        mask = np.isfinite(x) & np.isfinite(y)
        r, p = func(x[mask], y[mask]) if mask.sum() > 5 else (np.nan, np.nan)
        rows.append({"component": comp, "r": float(r), "p": float(p), "n": int(mask.sum())})
    return pd.DataFrame(rows)


def dop_summary(ds) -> pd.DataFrame:
    """沿真实轨迹的 DOP 统计（用于说明共面布局的垂直弱几何）。"""
    d = dop(ds.truth_abs, ds.anchors_world)
    z = ds.truth_rel[:, 2]
    return pd.DataFrame({
        "hdop_median": [float(d["hdop"].median())],
        "vdop_median": [float(d["vdop"].median())],
        "gdop_median": [float(d["gdop"].median())],
        "vdop_p95": [float(d["vdop"].quantile(0.95))],
        "z_at_vdop_p95_m": [float(np.interp(d["vdop"].quantile(0.95), d["vdop"].sort_values(), z[d["vdop"].argsort()]))],
    })

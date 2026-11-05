"""UWB 原始数据质量：采样、丢包、位置跳变、长时间固定值、EOP、Z 轴异常。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.time_alignment import find_gaps, sampling_stats


def sampling_quality(t: np.ndarray) -> dict:
    s = sampling_stats(t)
    dt = np.diff(np.asarray(t, float))
    return {
        **s,
        "dt_min_s": float(np.min(dt)) if dt.size else float("nan"),
        "dt_std_s": float(np.std(dt)) if dt.size else float("nan"),
        "n_dt_gt_2x_median": int(np.sum(dt > 2 * np.median(dt))) if dt.size else 0,
    }


def gap_table(t: np.ndarray, threshold_s: float = 0.5) -> pd.DataFrame:
    gaps = find_gaps(t, threshold_s)
    return pd.DataFrame(gaps, columns=["index", "t_start", "t_end", "duration_s"])


def position_jumps(pos: np.ndarray, t: np.ndarray, threshold_m: float = 0.3) -> pd.DataFrame:
    """相邻帧位置跳变（|Δp| 超过阈值），并换算等效速度。"""
    p = np.asarray(pos, float)
    t = np.asarray(t, float)
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    dt = np.diff(t)
    idx = np.where(d > threshold_m)[0]
    return pd.DataFrame({
        "index": idx,
        "t": t[idx + 1],
        "jump_m": d[idx],
        "dt_s": dt[idx],
        "implied_speed_mps": d[idx] / np.maximum(dt[idx], 1e-6),
    })


def speed_series(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
    p = np.asarray(pos, float)
    t = np.asarray(t, float)
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)
    return np.concatenate([[np.nan], v])


def stuck_segments(pos: np.ndarray, min_run: int = 10, tol: float = 1e-9) -> list[dict]:
    """检测连续多帧位置完全不变（跟踪丢失/输出保持）的片段。"""
    p = np.asarray(pos, float)
    if len(p) < min_run:
        return []
    same = np.all(np.abs(np.diff(p, axis=0)) <= tol, axis=1)

    segments = []
    start = None
    for i, flag in enumerate(same):
        if flag and start is None:
            start = i
        if (not flag) and start is not None:
            length = i - start + 1
            if length >= min_run:
                segments.append({"start_index": start, "end_index": i, "length": length})
            start = None
    if start is not None and (len(same) - start + 1) >= min_run:
        segments.append({"start_index": start, "end_index": len(same),
                         "length": len(same) - start + 1})
    return segments


def eop_summary(eop: np.ndarray) -> dict:
    """EOP 字段的统计描述（物理含义需向厂商确认，不能当作真误差）。"""
    e = np.asarray(eop, float)
    finite = np.isfinite(e)
    return {
        "mean": np.nanmean(e, axis=0).round(4).tolist() if finite.any() else [np.nan] * 3,
        "std": np.nanstd(e, axis=0).round(4).tolist() if finite.any() else [np.nan] * 3,
        "max": np.nanmax(e, axis=0).round(4).tolist() if finite.any() else [np.nan] * 3,
        "zero_ratio": float(np.mean(np.all(np.abs(e) < 1e-9, axis=1))) if len(e) else float("nan"),
        "n_unique_rows": int(len(np.unique(np.round(e, 4), axis=0))) if len(e) else 0,
    }


def z_quality(uwb_z: np.ndarray, truth_z: np.ndarray, eop_z: np.ndarray,
              t: np.ndarray) -> dict:
    """Z 轴定位质量：误差、饱和（长时间不变）比例、与 EOP 的关系。"""
    z = np.asarray(uwb_z, float)
    g = np.asarray(truth_z, float)
    e = z - g
    e = e[np.isfinite(e)]
    stuck = stuck_segments(np.column_stack([np.zeros_like(z), np.zeros_like(z), z]),
                           min_run=10, tol=1e-9)
    stuck_samples = sum(s["length"] for s in stuck)
    return {
        "z_bias_m": float(np.mean(e)),
        "z_mae_m": float(np.mean(np.abs(e))),
        "z_rmse_m": float(np.sqrt(np.mean(e ** 2))),
        "z_max_abs_m": float(np.max(np.abs(e))),
        "frac_abs_gt_0p5": float(np.mean(np.abs(e) > 0.5)),
        "stuck_segments": len(stuck),
        "stuck_ratio": float(stuck_samples / len(z)) if len(z) else float("nan"),
        "eop_z_mean": float(np.nanmean(eop_z)),
        "corr_eop_z_err": float(np.corrcoef(eop_z, z - g)[0, 1])
        if np.std(eop_z) > 0 else float("nan"),
    }


def quality_summary(ds) -> dict:
    """整合成一份可直接写进报告的质量摘要。"""
    pos = ds.uwb_mav
    return {
        "sampling": sampling_quality(ds.t),
        "gaps": {
            "count": int(len(gap_table(ds.t, ds.cfg.get("quality", {})
                                        .get("gap_threshold_s", 0.5)))),
        },
        "position_jumps": {
            "threshold_m": ds.cfg.get("quality", {}).get("pos_jump_m", 0.3),
            "count": int(len(position_jumps(pos, ds.t,
                                            ds.cfg.get("quality", {}).get("pos_jump_m", 0.3)))),
        },
        "stuck": {
            "segments": len(stuck_segments(pos, ds.cfg.get("quality", {}).get("stuck_min_run", 10))),
        },
        "eop": eop_summary(ds.eop),
        "z": z_quality(ds.uwb_mav[:, 2], ds.truth_rel[:, 2], ds.eop[:, 2], ds.t),
        "valid_time_ratio": float(ds.valid_time.mean()),
        "valid_range_ratio": float(ds.valid_range.mean()),
    }

"""四基站测距精度指标：RMSE / MAE / Bias / Std / Max / Median / P95 / 有效率 / 跳变次数。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.anchor_geometry import ANCHOR_NAMES


def range_errors(meas: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """测距误差 e = d_UWB - d_GT（无效测距保留 NaN）。"""
    return np.asarray(meas, float) - np.asarray(truth, float)


def jump_indices(meas: np.ndarray, valid: np.ndarray, threshold_m: float = 0.5) -> np.ndarray:
    """相邻有效测距之间 |Δd| > 阈值的帧索引（返回布尔矩阵，形状同 meas）。"""
    m = np.asarray(meas, float)
    v = np.asarray(valid, bool)
    jumps = np.zeros_like(v)
    prev = np.full_like(m, np.nan)
    prev[1:] = m[:-1]
    prev_valid = np.zeros_like(v)
    prev_valid[1:] = v[:-1]
    delta = np.abs(m - prev)
    jumps = v & prev_valid & np.isfinite(delta) & (delta > threshold_m)
    return jumps


def anchor_metrics(err: np.ndarray, valid: np.ndarray, jumps: np.ndarray,
                   anchor: str, percentile: float = 95.0) -> dict:
    """单个基站的测距统计量（只用有效样本）。"""
    e = np.asarray(err, float)
    v = np.asarray(valid, bool) & np.isfinite(e)
    ev = e[v]
    n = int(v.size)
    if ev.size == 0:
        return {"anchor": anchor, "n_samples": n, "n_valid": 0, "valid_ratio": 0.0}

    return {
        "anchor": anchor,
        "n_samples": n,
        "n_valid": int(ev.size),
        "valid_ratio": float(ev.size / n),
        "bias_m": float(np.mean(ev)),
        "mae_m": float(np.mean(np.abs(ev))),
        "rmse_m": float(np.sqrt(np.mean(ev ** 2))),
        "std_m": float(np.std(ev)),
        "max_abs_m": float(np.max(np.abs(ev))),
        "median_abs_m": float(np.median(np.abs(ev))),
        f"p{int(percentile)}_abs_m": float(np.percentile(np.abs(ev), percentile)),
        "n_jumps": int(np.sum(np.asarray(jumps, bool))),
    }


def metrics_table(err: np.ndarray, valid: np.ndarray, jumps: np.ndarray,
                  percentile: float = 95.0) -> pd.DataFrame:
    """四基站测距指标表。"""
    rows = [anchor_metrics(err[:, i], valid[:, i], jumps[:, i], ANCHOR_NAMES[i], percentile)
            for i in range(4)]
    return pd.DataFrame(rows)


def aggregate_metrics(err: np.ndarray, valid: np.ndarray) -> dict:
    """四站合并（所有有效测距）的总体指标。"""
    e = np.asarray(err, float)
    v = np.asarray(valid, bool) & np.isfinite(e)
    ev = e[v]
    if ev.size == 0:
        return {}
    return {
        "n_valid": int(ev.size),
        "bias_m": float(np.mean(ev)),
        "mae_m": float(np.mean(np.abs(ev))),
        "rmse_m": float(np.sqrt(np.mean(ev ** 2))),
        "std_m": float(np.std(ev)),
        "max_abs_m": float(np.max(np.abs(ev))),
        "p95_abs_m": float(np.percentile(np.abs(ev), 95)),
    }


def per_timestamp_rms(err: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """每一帧四站测距误差的 RMS（至少一个有效站），用于联合分析。"""
    e = np.asarray(err, float)
    v = np.asarray(valid, bool) & np.isfinite(e)
    e2 = np.where(v, e ** 2, 0.0)
    cnt = v.sum(axis=1)
    out = np.full(len(e), np.nan)
    ok = cnt > 0
    out[ok] = np.sqrt(e2[ok].sum(axis=1) / cnt[ok])
    return out

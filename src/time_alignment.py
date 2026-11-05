"""时间同步：以 UWB 时间戳为基准，把高频动捕真值插值到 UWB 时刻。

规则：
  * 不使用行号匹配；
  * 只在共同有效时间范围内插值，不外推；
  * 插值区间跨度超过 ``max_gap_s`` 时标记为无效（避免跨越大段丢包）。
"""

from __future__ import annotations

import numpy as np


def overlap_window(*time_arrays: np.ndarray) -> tuple[float, float]:
    """多路数据的公共时间窗口 [t0, t1]（秒）。"""
    t0 = max(float(np.min(t)) for t in time_arrays)
    t1 = min(float(np.max(t)) for t in time_arrays)
    if t1 <= t0:
        raise ValueError(f"数据没有重叠时间区间：{t0:.3f} >= {t1:.3f}")
    return t0, t1


def interp_with_validity(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray,
                         max_gap_s: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """把 ``(t_src, y_src)`` 线性插值到 ``t_dst``。

    返回 ``(y_interp, valid)``；``valid`` 同时满足：
      1. ``t_dst`` 位于 ``t_src`` 范围内（不外推）；
      2. 包住 ``t_dst`` 的两个源样本间隔不超过 ``max_gap_s``。
    """
    t_src = np.asarray(t_src, float)
    y_src = np.asarray(y_src, float)
    t_dst = np.asarray(t_dst, float)

    if y_src.ndim == 1:
        y_src = y_src[:, None]

    order = np.argsort(t_src)
    t_src, y_src = t_src[order], y_src[order]

    y_out = np.column_stack([np.interp(t_dst, t_src, y_src[:, i])
                             for i in range(y_src.shape[1])])

    idx = np.searchsorted(t_src, t_dst, side="left")
    idx = np.clip(idx, 1, len(t_src) - 1)
    left, right = t_src[idx - 1], t_src[idx]
    inside = (t_dst >= t_src[0]) & (t_dst <= t_src[-1])
    gap = right - left
    valid = inside & (gap <= max_gap_s)
    return y_out, valid


def sampling_stats(t: np.ndarray) -> dict:
    """采样间隔与速率统计。"""
    t = np.asarray(t, float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        return {"n": int(t.size), "rate_hz": float("nan")}
    return {
        "n": int(t.size),
        "duration_s": float(t[-1] - t[0]),
        "rate_hz": float(1.0 / np.median(dt)) if np.median(dt) > 0 else float("nan"),
        "dt_median_s": float(np.median(dt)),
        "dt_mean_s": float(np.mean(dt)),
        "dt_p95_s": float(np.percentile(dt, 95)),
        "dt_max_s": float(np.max(dt)),
    }


def find_gaps(t: np.ndarray, threshold_s: float = 0.5) -> np.ndarray:
    """返回超过阈值的时间间隔（起始索引, 起点, 终点, 间隔）。"""
    t = np.asarray(t, float)
    dt = np.diff(t)
    idx = np.where(dt > threshold_s)[0]
    return np.column_stack([idx, t[idx], t[idx + 1], dt[idx]]) if idx.size else np.empty((0, 4))

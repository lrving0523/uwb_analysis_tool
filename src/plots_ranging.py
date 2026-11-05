"""实验一（测距）科研绘图：R1–R9。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src import plot_style as ps
from src.anchor_geometry import ANCHOR_NAMES

COLORS = ps.ANCHOR_COLORS


def _cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, float))
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def _robust_limits(*arrays, q: float = 99.5, pad: float = 0.08) -> tuple[float, float]:
    """按分位数给出抗粗差的显示范围（只影响显示，不修改数据）。"""
    vals = np.concatenate([np.asarray(a, float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [100 - q, q])
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return lo - pad * (hi - lo), hi + pad * (hi - lo)


def r1_ranges(ds, out: Path) -> list[Path]:
    fig, axes = ps.new_figure(2, 2, width="double", height=4.4)
    t = ds.t_rel
    for i, ax in enumerate(axes):
        ax.plot(t, ds.ranges_true[:, i], color="k", lw=0.9, ls="--",
                label="Ground truth" if i == 0 else None)
        ax.plot(t, ds.ranges_meas[:, i], ls="none", marker=".", ms=2.0,
                color=COLORS[i], label="UWB" if i == 0 else None)
        lo, hi = _robust_limits(ds.ranges_true[:, i], ds.ranges_meas[:, i])
        ax.set_ylim(lo, hi)
        ax.set_ylabel(f"$d_{{{i}}}$ (m)")
        ax.set_xlim(t[0], t[-1])
        ax.set_title(f"{ANCHOR_NAMES[i]}", fontsize=8)
    for ax in axes[2:]:
        ax.set_xlabel(r"$t$ (s)")
    axes[0].legend(loc="best", fontsize=7)
    return ps.save_figure(fig, out / "R1_ranges_vs_truth")


def r2_errors(ds, err: np.ndarray, out: Path) -> list[Path]:
    fig, ax = ps.new_figure(1, 1, width="double", aspect=0.36)
    for i in range(4):
        ax.plot(ds.t_rel, err[:, i], color=COLORS[i], lw=0.8, label=ANCHOR_NAMES[i])
    ax.axhline(0, color="k", lw=0.6)
    lo, hi = _robust_limits(err, q=99.0)
    ax.set_ylim(lo, hi)
    vmax = float(np.nanmax(np.abs(err)))
    if vmax > hi:
        ax.text(0.01, 0.95, f"max $|e|$ = {vmax:.2f} m (off-scale)", transform=ax.transAxes,
                fontsize=7, va="top")
    ax.set_xlim(ds.t_rel[0], ds.t_rel[-1])
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$e_d$ (m)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=7)
    return ps.save_figure(fig, out / "R2_range_errors_time")


def r3_rmse_bar(table, out: Path) -> list[Path]:
    fig, ax = ps.new_figure(1, 1, width="single", aspect=0.85)
    x = np.arange(4)
    vals = table["rmse_m"].to_numpy()
    bars = ax.bar(x, vals, width=0.6, color=COLORS, edgecolor="k", linewidth=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                va="bottom", fontsize=7)
    ax.set_xticks(x, ANCHOR_NAMES)
    ax.set_ylabel("RMSE (m)")
    ax.set_ylim(0, vals.max() * 1.25)
    return ps.save_figure(fig, out / "R3_rmse_bar")


def r4_box(err: np.ndarray, out: Path) -> list[Path]:
    fig, ax = ps.new_figure(1, 1, width="single", aspect=0.85)
    data = [err[:, i][np.isfinite(err[:, i])] for i in range(4)]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", ms=1.5, alpha=0.4, mec="none"),
                    medianprops=dict(color="k", lw=0.8))
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor("k")
        patch.set_linewidth(0.4)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xticks(np.arange(1, 5), ANCHOR_NAMES)
    ax.set_ylabel(r"$e_d$ (m)")
    allv = np.concatenate(data)
    ax.set_ylim(np.percentile(allv, 0.5) - 0.1, np.percentile(allv, 99.5) + 0.1)
    return ps.save_figure(fig, out / "R4_error_box")


def r5_cdf(err: np.ndarray, out: Path) -> list[Path]:
    fig, ax = ps.new_figure(1, 1, width="single", aspect=0.85)
    for i in range(4):
        e = np.abs(err[:, i][np.isfinite(err[:, i])])
        x, y = _cdf(e)
        ax.plot(x, y, color=COLORS[i], lw=1.0, label=ANCHOR_NAMES[i])
    ax.set_xlabel(r"$|e_d|$ (m)")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, np.nanpercentile(np.abs(err), 99.5) * 1.05)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=7)
    return ps.save_figure(fig, out / "R5_error_cdf")


def r6_error_vs_distance(ds, err: np.ndarray, out: Path) -> list[Path]:
    fig, ax = ps.new_figure(1, 1, width="double", aspect=0.42)
    for i in range(4):
        m = np.isfinite(err[:, i])
        ax.plot(ds.ranges_true[m, i], err[m, i], ls="none", marker=".", ms=2.0,
                color=COLORS[i], alpha=0.7, label=ANCHOR_NAMES[i])
    ax.axhline(0, color="k", lw=0.6)
    lo, hi = _robust_limits(err, q=99.0)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True range (m)")
    ax.set_ylabel(r"$e_d$ (m)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=7)
    return ps.save_figure(fig, out / "R6_error_vs_distance")


def r7_layout_2d(ds, out: Path) -> list[Path]:
    a = ds.anchors_world
    pts = np.vstack([a[:, :2], ds.truth_abs[:, :2]])
    height = ps.aspect_height(ps.SINGLE_COLUMN_IN, np.ptp(pts[:, 0]),
                              np.ptp(pts[:, 1]), extra=0.9)
    fig, ax = ps.new_figure(1, 1, width="single", height=height)
    ax.plot(ds.truth_abs[:, 0], ds.truth_abs[:, 1], color=ps.COLORS["truth"],
            lw=0.9, alpha=0.9, label="UAV (mocap)")
    for i in range(4):
        ax.plot(a[i, 0], a[i, 1], marker="s", ms=6, color=COLORS[i], mec="k",
                mew=0.4, ls="none")
        ax.text(a[i, 0] + 0.05, a[i, 1] + 0.05, ANCHOR_NAMES[i], fontsize=7,
                color=COLORS[i])
    ax.plot(ds.truth_abs[0, 0], ds.truth_abs[0, 1], marker="o", ms=4,
            mfc=ps.COLORS["aux"], mec="k", mew=0.3, ls="none")
    ax.relim()
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.08)
    ax.set_xlabel(r"$x_M$ (m)")
    ax.set_ylabel(r"$y_M$ (m)")
    ax.legend(loc="best", fontsize=7)
    return ps.save_figure(fig, out / "R7_layout_2d")


def r8_layout_3d(ds, out: Path) -> list[Path]:
    fig = plt.figure(figsize=ps.figsize("single", height=3.0), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ds.truth_abs[:, 0], ds.truth_abs[:, 1], ds.truth_abs[:, 2],
            color=ps.COLORS["truth"], lw=0.9, label="UAV (mocap)")
    a = ds.anchors_world
    for i in range(4):
        ax.scatter([a[i, 0]], [a[i, 1]], [a[i, 2]], marker="s", s=18,
                   color=COLORS[i], edgecolors="k", linewidths=0.3)
        ax.text(a[i, 0], a[i, 1], a[i, 2] + 0.05, ANCHOR_NAMES[i], fontsize=7,
                color=COLORS[i])
    ps.set_equal_3d(ax, np.vstack([a, ds.truth_abs]))
    ax.view_init(28, -60)
    ax.set_xlabel(r"$x_M$ (m)", labelpad=1)
    ax.set_ylabel(r"$y_M$ (m)", labelpad=1)
    ax.set_zlabel(r"$z_M$ (m)", labelpad=1)
    ax.tick_params(labelsize=7, pad=1)
    ps.style_3d(ax)
    ax.legend(loc="upper left", fontsize=7)
    return ps.save_figure(fig, out / "R8_layout_3d")


def r9_anomalies(ds, err: np.ndarray, table, jumps: np.ndarray, out: Path) -> list[Path]:
    fig, axes = ps.new_figure(1, 2, width="double", aspect=0.38)

    ax = axes[0]
    ratio = table["valid_ratio"].to_numpy() * 100
    bars = ax.bar(np.arange(4), ratio, width=0.6, color=COLORS, edgecolor="k", lw=0.4)
    for b, v in zip(bars, ratio):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center",
                va="bottom", fontsize=7)
    ax.set_xticks(np.arange(4), ANCHOR_NAMES)
    ax.set_ylabel("Valid range ratio (%)")
    ax.set_ylim(0, 105)
    ps.panel_label(ax, "(a)")

    ax = axes[1]
    for i in range(4):
        idx = np.where(jumps[:, i])[0]
        ax.plot(ds.t_rel[idx], np.full(idx.size, i), ls="none", marker="|",
                ms=6, color=COLORS[i])
    ax.set_yticks(np.arange(4), ANCHOR_NAMES)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlim(ds.t_rel[0], ds.t_rel[-1])
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel("Jump events")
    ps.panel_label(ax, "(b)")
    return ps.save_figure(fig, out / "R9_anomaly_validity")


def make_all(ds, err: np.ndarray, table, jumps: np.ndarray, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    files = []
    files += r1_ranges(ds, out)
    files += r2_errors(ds, err, out)
    files += r3_rmse_bar(table, out)
    files += r4_box(err, out)
    files += r5_cdf(err, out)
    files += r6_error_vs_distance(ds, err, out)
    files += r7_layout_2d(ds, out)
    files += r8_layout_3d(ds, out)
    files += r9_anomalies(ds, err, table, jumps, out)
    return files

"""UWB 原始数据质量绘图：Q1–Q4（含高度异常统一时间轴组合图）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src import plot_style as ps
from src.ranging_metrics import per_timestamp_rms


def q1_sampling(ds, out: Path) -> list[Path]:
    t = ds.t
    dt = np.diff(t)
    fig, axes = ps.new_figure(1, 2, width="double", aspect=0.4)

    ax = axes[0]
    ax.hist(dt * 1e3, bins=40, color=ps.COLORS["a0"], alpha=0.85,
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel(r"$\Delta t$ (ms)")
    ax.set_ylabel("Count")
    ps.panel_label(ax, "(a)")

    ax = axes[1]
    ax.plot(t[1:], dt * 1e3, color=ps.COLORS["a0"], lw=0.8)
    ax.axhline(np.median(dt) * 1e3 * 2, color=ps.COLORS["error"], lw=0.7, ls="--")
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$\Delta t$ (ms)")
    ax.set_xlim(t[0], t[-1])
    ps.panel_label(ax, "(b)")
    return ps.save_figure(fig, out / "Q1_sampling")


def q2_jumps(ds, jumps, out: Path) -> list[Path]:
    fig, axes = ps.new_figure(1, 2, width="double", aspect=0.4)
    ax = axes[0]
    ax.plot(jumps["t"], jumps["jump_m"], ls="none", marker=".", ms=3,
            color=ps.COLORS["error"])
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$|\Delta p|$ (m)")
    ax.set_yscale("log")
    ax.set_xlim(ds.t_rel[0], ds.t_rel[-1])
    ps.panel_label(ax, "(a)")

    ax = axes[1]
    ax.plot(jumps["t"], jumps["implied_speed_mps"], ls="none", marker=".", ms=3,
            color=ps.COLORS["a1"])
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel("Implied speed (m/s)")
    ax.set_xlim(ds.t_rel[0], ds.t_rel[-1])
    ps.panel_label(ax, "(b)")
    return ps.save_figure(fig, out / "Q2_position_jumps")


def q3_height_timeline(ds, err, out: Path) -> list[Path]:
    """高度异常统一时间轴：UWB z、真值 z、EOP-z、测距误差、z 误差。"""
    t = ds.t_rel
    range_err = ds.ranges_meas - ds.ranges_true
    rms = per_timestamp_rms(range_err, ds.valid_range)

    fig, axes = plt.subplots(5, 1, figsize=(7.16, 6.4), sharex=True,
                             layout="constrained")
    ax = axes[0]
    ax.plot(t, ds.uwb_mav[:, 2], color=ps.COLORS["uwb"], lw=0.9)
    ax.set_ylabel(r"$z_{UWB}$ (m)")
    ps.panel_label(ax, "(a)", dx=-0.09)

    ax = axes[1]
    ax.plot(t, ds.truth_rel[:, 2], color=ps.COLORS["truth"], lw=0.9)
    ax.set_ylabel(r"$z_{GT}$ (m)")
    ps.panel_label(ax, "(b)", dx=-0.09)

    ax = axes[2]
    ax.plot(t, ds.eop[:, 2], color=ps.COLORS["a1"], lw=0.9)
    ax.set_ylabel(r"EOP$_z$")
    ps.panel_label(ax, "(c)", dx=-0.09)

    ax = axes[3]
    ax.plot(t, rms, color=ps.COLORS["a2"], lw=0.8)
    ax.set_ylabel(r"RMS $e_d$ (m)")
    ps.panel_label(ax, "(d)", dx=-0.09)

    ax = axes[4]
    ax.plot(t, err["ez"], color=ps.COLORS["error"], lw=0.9)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(r"$e_z$ (m)")
    ax.set_xlabel(r"$t$ (s)")
    ax.set_xlim(t[0], t[-1])
    ps.panel_label(ax, "(e)", dx=-0.09)
    return ps.save_figure(fig, out / "Q3_height_timeline")


def q4_stuck_gaps(ds, gaps, stuck, out: Path) -> list[Path]:
    """(a) EOP 三轴时间曲线 (b) Z 轴固定值片段。"""
    fig, axes = ps.new_figure(1, 2, width="double", aspect=0.4)

    ax = axes[0]
    for i, (name, color) in enumerate(zip(("EOP$_x$", "EOP$_y$", "EOP$_z$"),
                                          (ps.COLORS["a0"], ps.COLORS["a1"], ps.COLORS["a2"]))):
        ax.plot(ds.t_rel, ds.eop[:, i], color=color, lw=0.8, label=name)
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel("EOP")
    ax.set_xlim(ds.t_rel[0], ds.t_rel[-1])
    ax.legend(ncol=3, fontsize=6, loc="upper right")
    ps.panel_label(ax, "(a)")

    ax = axes[1]
    for seg in stuck:
        i0, i1 = seg["start_index"], seg["end_index"]
        ax.axvspan(ds.t_rel[i0], ds.t_rel[min(i1, len(ds.t_rel) - 1)],
                   color=ps.COLORS["a3"], alpha=0.5, lw=0)
    ax.plot(ds.t_rel, ds.uwb_mav[:, 2], color=ps.COLORS["uwb"], lw=0.7)
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$z_{UWB}$ (m)")
    ax.set_xlim(ds.t_rel[0], ds.t_rel[-1])
    ps.panel_label(ax, "(b)")
    return ps.save_figure(fig, out / "Q4_eop_and_stuck")


def make_all(ds, err, gaps, stuck, jumps, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    files = []
    files += q1_sampling(ds, out)
    files += q2_jumps(ds, jumps, out)
    files += q3_height_timeline(ds, err, out)
    files += q4_stuck_gaps(ds, gaps, stuck, out)
    return files

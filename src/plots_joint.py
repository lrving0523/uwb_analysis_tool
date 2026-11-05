"""测距误差与定位误差联合分析绘图：J1–J5。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src import plot_style as ps
from src.anchor_geometry import ANCHOR_NAMES


def j1_time_compare(ds, joint, out: Path) -> list[Path]:
    fig, axes = ps.new_figure(2, 1, width="double", height=3.6, sharex=True)
    axes[0].plot(ds.t_rel, joint["range_rms"], color=ps.COLORS["a0"], lw=0.8)
    axes[0].set_ylabel(r"RMS $e_d$ (m)")
    axes[1].plot(ds.t_rel, joint["e_3d"], color=ps.COLORS["error"], lw=0.9)
    axes[1].set_ylabel(r"$e_{3d}$ (m)")
    axes[1].set_xlabel(r"$t$ (s)")
    axes[1].set_xlim(ds.t_rel[0], ds.t_rel[-1])
    for ax, label in zip(axes, ("(a)", "(b)")):
        ps.panel_label(ax, label)
    return ps.save_figure(fig, out / "J1_ranging_vs_localization_time")


def _scatter(ax, x, y, xlabel, ylabel, color):
    m = np.isfinite(x) & np.isfinite(y)
    ax.plot(x[m], y[m], ls="none", marker=".", ms=2.5, color=color, alpha=0.6)
    if m.sum() > 5:
        r = np.corrcoef(x[m], y[m])[0, 1]
        ax.text(0.97, 0.95, f"$r$ = {r:.3f}  (n = {m.sum()})", transform=ax.transAxes,
                ha="right", va="top", fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def j2_vs_3d(joint, out: Path) -> list[Path]:
    fig, ax = ps.new_figure(1, 1, width="single", aspect=0.9)
    _scatter(ax, joint["range_rms"].to_numpy(), joint["e_3d"].to_numpy(),
             r"RMS $e_d$ (m)", r"$e_{3d}$ (m)", ps.COLORS["a0"])
    return ps.save_figure(fig, out / "J2_range_vs_3d")


def j3_vs_z(joint, out: Path) -> list[Path]:
    fig, ax = ps.new_figure(1, 1, width="single", aspect=0.9)
    _scatter(ax, joint["range_rms"].to_numpy(), joint["ez"].to_numpy(),
             r"RMS $e_d$ (m)", r"$e_z$ (m)", ps.COLORS["a3"])
    return ps.save_figure(fig, out / "J3_range_vs_z")


def j4_correlation_heatmap(corr, out: Path) -> list[Path]:
    comps = ["ex", "ey", "ez", "e_xy", "e_3d"]
    mat = corr[[f"r_{c}" for c in comps]].to_numpy()
    fig, ax = ps.new_figure(1, 1, width="single", aspect=0.8)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(comps)), [c.upper().replace("E", "e").replace("XY", "XY") for c in comps])
    ax.set_xticklabels([r"$e_x$", r"$e_y$", r"$e_z$", r"$e_{xy}$", r"$e_{3d}$"])
    ax.set_yticks(np.arange(4), ANCHOR_NAMES)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(mat[i, j]) > 0.6 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Pearson $r$")
    cb.outline.set_linewidth(0.5)
    ax.set_xlabel("Localization error component")
    ax.set_ylabel("Ranging error")
    return ps.save_figure(fig, out / "J4_correlation_heatmap")


def j5_geometry_dop(ds, dop_df, out: Path) -> list[Path]:
    """三子图： (a) XY+HDOP  (b) z(t)+VDOP  (c) 3D+PDOP。

    HDOP/VDOP/PDOP 来自同一个 H（四基站真实坐标 + 动捕真值），
    满足 PDOP² = HDOP² + VDOP²；cond(HᵀH) > 1e3 的样本用黑边标出。
    """
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    hd = dop_df["hdop"].to_numpy(float)
    vd = dop_df["vdop"].to_numpy(float)
    pdop = dop_df["pdop"].to_numpy(float)
    cond = dop_df["cond"].to_numpy(float)
    ill = np.isfinite(cond) & (cond > 1e3)

    t = ds.t_rel
    xyz = ds.truth_abs
    a = ds.anchors_world
    p0 = ds.p0
    p0z = float(a[0, 2])

    fig = plt.figure(figsize=(7.16, 5.4), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])
    fig.suptitle("Geometric Dilution of Precision Along the UAV Trajectory",
                 fontsize=9)

    # ---------------- (a) XY + HDOP ----------------
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], c=hd, s=7, cmap="viridis",
                    vmin=float(np.nanmin(hd)), vmax=float(np.nanmax(hd)))
    if ill.any():
        ax.scatter(xyz[ill, 0], xyz[ill, 1], s=14, facecolors="none",
                   edgecolors="k", linewidths=0.4)
    for i in range(4):
        ax.plot(a[i, 0], a[i, 1], marker="s", ms=6, color=ps.ANCHOR_COLORS[i],
                mec="k", mew=0.4, ls="none")
        ax.text(a[i, 0] + 0.05, a[i, 1] + 0.05, ANCHOR_NAMES[i], fontsize=7,
                color=ps.ANCHOR_COLORS[i])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(r"$x_M$ (m)")
    ax.set_ylabel(r"$y_M$ (m)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("HDOP")
    cb.outline.set_linewidth(0.5)
    ax.set_title("(a) Horizontal trajectory and HDOP", fontsize=8, loc="left")

    # ---------------- (b) z(t) + VDOP ----------------
    ax = fig.add_subplot(gs[0, 1])
    sc2 = ax.scatter(t, xyz[:, 2], c=vd, s=7, cmap="plasma",
                     norm=LogNorm(vmin=max(0.5, np.floor(np.nanmin(vd) * 10) / 10),
                                  vmax=np.nanmax(vd)))
    if ill.any():
        ax.scatter(t[ill], xyz[ill, 2], s=14, facecolors="none",
                   edgecolors="k", linewidths=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$z_{GT}$ (m)")
    ax.set_xlim(t[0], t[-1])
    cb2 = fig.colorbar(sc2, ax=ax, fraction=0.046, pad=0.02)
    cb2.set_label("VDOP")
    cb2.outline.set_linewidth(0.5)
    ax.set_title("(b) Altitude profile and VDOP", fontsize=8, loc="left")

    # ---------------- (c) 3D + PDOP ----------------
    ax = fig.add_subplot(gs[1, :], projection="3d")
    sc3 = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=pdop, s=6,
                     cmap="cividis",
                     norm=LogNorm(vmin=max(0.5, np.floor(np.nanmin(pdop) * 10) / 10),
                                  vmax=np.nanmax(pdop)))
    if ill.any():
        ax.scatter(xyz[ill, 0], xyz[ill, 1], xyz[ill, 2], s=12, facecolors="none",
                   edgecolors="k", linewidths=0.4)

    # 0.8 m × 0.8 m 平台边框（四角即基站）
    corners = np.vstack([a[[0, 1, 2, 3]], a[0]])
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color="k", lw=0.7)
    ax.add_collection3d(Poly3DCollection([a[[0, 1, 2, 3]]], alpha=0.12,
                                         facecolor="grey", edgecolor="none"))

    ax.scatter([p0[0]], [p0[1]], [p0[2]], marker="+", s=45, color="k",
               linewidths=0.8, zorder=10)
    ax.text(p0[0], p0[1], p0[2] - 0.12, "platform centre", fontsize=6.5,
            ha="center")
    for i in range(4):
        ax.scatter([a[i, 0]], [a[i, 1]], [a[i, 2]], marker="s", s=22,
                   color=ps.ANCHOR_COLORS[i], edgecolors="k", linewidths=0.4)
        ax.text(a[i, 0], a[i, 1], a[i, 2] + 0.04, ANCHOR_NAMES[i], fontsize=7,
                color=ps.ANCHOR_COLORS[i])

    ax.scatter([xyz[0, 0]], [xyz[0, 1]], [xyz[0, 2]], marker="o", s=34,
               color=ps.COLORS["a2"], edgecolors="k", linewidths=0.4, zorder=10)
    ax.scatter([xyz[-1, 0]], [xyz[-1, 1]], [xyz[-1, 2]], marker="D", s=30,
               color=ps.COLORS["a1"], edgecolors="k", linewidths=0.4, zorder=10)
    ax.text(xyz[0, 0], xyz[0, 1], xyz[0, 2] + 0.07, "Start", fontsize=6.5,
            color=ps.COLORS["a2"])
    ax.text(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2] + 0.07, "End", fontsize=6.5,
            color=ps.COLORS["a1"])

    ps.set_equal_3d(ax, np.vstack([xyz, a]), zoom=1.12)
    ax.view_init(24, -58)
    ax.set_xlabel(r"$x_M$ (m)", labelpad=1)
    ax.set_ylabel(r"$y_M$ (m)", labelpad=1)
    ax.set_zlabel(r"$z_M$ (m)", labelpad=1)
    ax.tick_params(labelsize=7, pad=1)
    ps.style_3d(ax)

    handles = [
        Line2D([], [], marker="o", ls="none", color=ps.COLORS["a2"],
               markeredgecolor="k", ms=5, label="Start"),
        Line2D([], [], marker="D", ls="none", color=ps.COLORS["a1"],
               markeredgecolor="k", ms=5, label="End"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="none",
               markeredgecolor="k", ms=6, label=r"cond$(H^{\top}H)>10^3$"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(-0.42, 0.50),
              fontsize=7, handletextpad=0.4, frameon=False)

    cb3 = fig.colorbar(sc3, ax=ax, fraction=0.03, pad=0.07, shrink=0.85)
    cb3.set_label("PDOP")
    cb3.outline.set_linewidth(0.5)
    ax.set_title("(c) 3D trajectory and PDOP", fontsize=8, loc="left")

    return ps.save_figure(fig, out / "J5_geometry_dop")


def j7_vdop_vs_z(ds, dop_df, out: Path) -> list[Path]:
    """辅助图：VDOP 与 z_GT 的散点关系（按水平距离着色）。"""
    from matplotlib.colors import LogNorm

    vd = dop_df["vdop"].to_numpy(float)
    z = ds.truth_rel[:, 2]
    dxy = np.linalg.norm(ds.truth_rel[:, :2], axis=1)

    fig, ax = ps.new_figure(1, 1, width="single", aspect=0.95)
    sc = ax.scatter(z, vd, c=dxy, s=7, cmap="magma", vmin=0.0, vmax=float(dxy.max()))
    ax.set_yscale("log")
    ax.set_xlabel(r"$z_{GT}$ (m)")
    ax.set_ylabel("VDOP")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Horizontal distance\nfrom array centre (m)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.5)
    ax.text(0.04, 0.95, "far, low", transform=ax.transAxes, va="top", fontsize=7)
    ax.text(0.96, 0.08, "above array", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=7)
    return ps.save_figure(fig, out / "J7_vdop_vs_z")


def make_all(ds, joint, corr, dop_df, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    files = []
    files += j1_time_compare(ds, joint, out)
    files += j2_vs_3d(joint, out)
    files += j3_vs_z(joint, out)
    files += j4_correlation_heatmap(corr, out)
    files += j5_geometry_dop(ds, dop_df, out)
    files += j7_vdop_vs_z(ds, dop_df, out)
    return files

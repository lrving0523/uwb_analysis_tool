"""实验二（三维定位）科研绘图：P1–P12。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src import plot_style as ps

COLORS = ps.COLORS


def _stacked(ds, series: dict[str, np.ndarray], path: Path,
             labels: dict | None = None, colors: dict | None = None,
             styles: dict | None = None, width: str = "single",
             height: float = 4.6, xlabel: bool = True) -> list[Path]:
    """3 个共享 x 轴的子图（x/y/z 各一层）。"""
    fig, axes = ps.new_figure(3, 1, width=width, height=height, sharex=True)
    labels = labels or {}
    colors = colors or {}
    styles = styles or {}
    for i, ax in enumerate(axes):
        for key, data in series.items():
            ax.plot(ds.t_rel, np.asarray(data)[:, i], color=colors.get(key, None),
                    lw=0.9, ls=styles.get(key, "-"), label=labels.get(key, key))
        ax.set_ylabel(rf"${'xyz'[i]}$ (m)")
        ax.set_xlim(ds.t_rel[0], ds.t_rel[-1])
    if xlabel:
        axes[-1].set_xlabel(r"$t$ (s)")
    handles, labs = axes[0].get_legend_handles_labels()
    if len(labs) > 1:
        fig.legend(handles, labs, loc="outside upper center", ncol=len(labs),
                   frameon=False)
    return ps.save_figure(fig, path)


def p1_raw_xyz(ds, out: Path) -> list[Path]:
    return _stacked(ds, {"uwb": ds.uwb_uwb}, out / "P1_uwb_raw_xyz",
                    labels={"uwb": "UWB (raw frame)"}, colors={"uwb": COLORS["uwb"]})


def p2_mavros_xyz(ds, out: Path) -> list[Path]:
    return _stacked(ds, {"uwb": ds.uwb_mav}, out / "P2_uwb_mavros_xyz",
                    labels={"uwb": "UWB (MAVROS frame)"}, colors={"uwb": COLORS["uwb"]})


def _platform_2d(ax, ds, label_anchors: bool = True, label_center: bool = True) -> None:
    """在 XY 图上画 0.8 m 平台边框、四基站与平台中心（相对平台中心坐标）。"""
    a = ds.anchors_world - ds.p0
    corners = np.vstack([a[[0, 1, 2, 3]], a[0]])
    ax.plot(corners[:, 0], corners[:, 1], color="k", lw=0.6, ls="--", zorder=1)
    for i in range(4):
        ax.plot(a[i, 0], a[i, 1], marker="s", ms=5.5, color=ps.ANCHOR_COLORS[i],
                mec="k", mew=0.3, ls="none", zorder=6)
        if label_anchors:
            # 固定几何下 A0/A1 在上、A2/A3 在下；标签朝平台外侧偏移避免压住落点
            offsets = {0: (4, -11), 1: (-16, -11), 2: (-16, 4), 3: (4, 4)}
            dx_pt, dy_pt = offsets.get(i, (4, -10))
            ax.annotate(f"A{i}", (a[i, 0], a[i, 1]), xytext=(dx_pt, dy_pt),
                        textcoords="offset points", fontsize=6.5,
                        color=ps.ANCHOR_COLORS[i], zorder=7)
    ax.plot(0.0, 0.0, marker="+", ms=9, color="k", mew=0.9, ls="none", zorder=7,
            label="platform centre" if label_center else None)


def _platform_3d(ax, ds) -> None:
    """在 3D 图上画平台边框、四基站与平台中心（相对平台中心坐标）。"""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    a = ds.anchors_world - ds.p0
    corners = np.vstack([a[[0, 1, 2, 3]], a[0]])
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color="k", lw=0.6, ls="--")
    ax.add_collection3d(Poly3DCollection([a[[0, 1, 2, 3]]], alpha=0.10,
                                         facecolor="grey", edgecolor="none"))
    for i in range(4):
        ax.scatter([a[i, 0]], [a[i, 1]], [a[i, 2]], marker="s", s=20,
                   color=ps.ANCHOR_COLORS[i], edgecolors="k", linewidths=0.3)
        ax.text(a[i, 0], a[i, 1], a[i, 2] + 0.04, f"A{i}", fontsize=6.5,
                color=ps.ANCHOR_COLORS[i])
    ax.scatter([0.0], [0.0], [0.0], marker="+", s=50, color="k", linewidths=0.8)


def p3_xy(ds, out: Path) -> list[Path]:
    a_rel = ds.anchors_world - ds.p0
    pts = np.vstack([ds.uwb_mav[:, :2], a_rel[:, :2], np.zeros((1, 2))])
    h = ps.aspect_height(ps.SINGLE_COLUMN_IN, np.ptp(pts[:, 0]),
                         np.ptp(pts[:, 1]), extra=0.9)
    fig, ax = ps.new_figure(1, 1, width="single", height=h)
    _platform_2d(ax, ds)
    ax.plot(ds.uwb_mav[:, 0], ds.uwb_mav[:, 1], color=COLORS["uwb"], lw=0.9,
            label="UWB trajectory")
    ax.plot(ds.uwb_mav[0, 0], ds.uwb_mav[0, 1], marker="o", ms=4.5,
            mfc=COLORS["a2"], mec="k", mew=0.3, ls="none", label="Start", zorder=8)
    ax.plot(ds.uwb_mav[-1, 0], ds.uwb_mav[-1, 1], marker="D", ms=4.5,
            mfc=COLORS["a1"], mec="k", mew=0.3, ls="none", label="UWB landing", zorder=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_M$ (m)")
    ax.set_ylabel(r"$y_M$ (m)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=6,
              frameon=False, handletextpad=0.3, columnspacing=1.0)
    return ps.save_figure(fig, out / "P3_uwb_xy")


def p4_3d(ds, out: Path) -> list[Path]:
    fig = plt.figure(figsize=ps.figsize("single", height=2.9), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ds.uwb_mav[:, 0], ds.uwb_mav[:, 1], ds.uwb_mav[:, 2],
            color=COLORS["uwb"], lw=0.8)
    ps.set_equal_3d(ax, ds.uwb_mav)
    ax.view_init(26, -60)
    ax.set_xlabel(r"$x_M$ (m)", labelpad=1)
    ax.set_ylabel(r"$y_M$ (m)", labelpad=1)
    ax.set_zlabel(r"$z_M$ (m)", labelpad=1)
    ax.tick_params(labelsize=7, pad=1)
    ps.style_3d(ax)
    return ps.save_figure(fig, out / "P4_uwb_3d")


def p5_compare_xyz(ds, out: Path) -> list[Path]:
    return _stacked(ds, {"truth": ds.truth_rel, "uwb": ds.uwb_mav},
                    out / "P5_compare_xyz",
                    labels={"truth": "Ground truth", "uwb": "UWB"},
                    colors={"truth": COLORS["truth"], "uwb": COLORS["uwb"]},
                    styles={"truth": "-", "uwb": (0, (4, 2))}, width="double", height=4.0)


def p6_compare_xy(ds, out: Path) -> list[Path]:
    a_rel = ds.anchors_world - ds.p0
    pts = np.vstack([ds.truth_rel[:, :2], ds.uwb_mav[:, :2],
                     a_rel[:, :2], np.zeros((1, 2))])
    h = ps.aspect_height(ps.SINGLE_COLUMN_IN, np.ptp(pts[:, 0]),
                         np.ptp(pts[:, 1]), extra=0.9)
    fig, ax = ps.new_figure(1, 1, width="single", height=h)
    _platform_2d(ax, ds)
    ax.plot(ds.truth_rel[:, 0], ds.truth_rel[:, 1], color=COLORS["truth"], lw=1.0,
            label="Ground truth")
    ax.plot(ds.uwb_mav[:, 0], ds.uwb_mav[:, 1], color=COLORS["uwb"], lw=0.9,
            ls=(0, (4, 2)), label="UWB")

    # 落点：真值 vs UWB（坐标已相对平台中心）
    gt_end = ds.truth_rel[-1, :2]
    uwb_end = ds.uwb_mav[-1, :2]
    ax.plot(*gt_end, marker="o", ms=5, mfc=COLORS["truth"], mec="k", mew=0.3,
            ls="none", zorder=9, label="GT landing")
    ax.plot(*uwb_end, marker="D", ms=5, mfc=COLORS["uwb"], mec="k", mew=0.3,
            ls="none", zorder=9, label="UWB landing")
    ax.annotate("", xy=gt_end, xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-", color=COLORS["truth"], lw=0.7, ls=":"))
    ax.annotate("", xy=uwb_end, xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-", color=COLORS["uwb"], lw=0.7, ls=":"))

    d_gt = float(np.linalg.norm(gt_end))
    d_uwb = float(np.linalg.norm(uwb_end))
    ax.text(0.02, 0.02,
            f"landing offset\nGT  {d_gt:.2f} m\nUWB {d_uwb:.2f} m",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.4))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_M$ (m)")
    ax.set_ylabel(r"$y_M$ (m)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              fontsize=6, frameon=False, handletextpad=0.3, columnspacing=1.0)
    return ps.save_figure(fig, out / "P6_compare_xy")


def p7_compare_3d(ds, out: Path) -> list[Path]:
    fig = plt.figure(figsize=ps.figsize("single", height=3.0), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ds.truth_rel[:, 0], ds.truth_rel[:, 1], ds.truth_rel[:, 2],
            color=COLORS["truth"], lw=0.9, label="Ground truth")
    ax.plot(ds.uwb_mav[:, 0], ds.uwb_mav[:, 1], ds.uwb_mav[:, 2],
            color=COLORS["uwb"], lw=0.8, ls=(0, (3, 2)), label="UWB")
    _platform_3d(ax, ds)
    ax.scatter([ds.truth_rel[-1, 0]], [ds.truth_rel[-1, 1]], [ds.truth_rel[-1, 2]],
               marker="o", s=28, color=COLORS["truth"], edgecolors="k",
               linewidths=0.3, label="GT landing")
    ax.scatter([ds.uwb_mav[-1, 0]], [ds.uwb_mav[-1, 1]], [ds.uwb_mav[-1, 2]],
               marker="D", s=24, color=COLORS["uwb"], edgecolors="k",
               linewidths=0.3, label="UWB landing")
    ps.set_equal_3d(ax, np.vstack([ds.truth_rel, ds.uwb_mav,
                                   ds.anchors_world - ds.p0, np.zeros((1, 3))]))
    ax.view_init(26, -60)
    ax.set_xlabel(r"$x_M$ (m)", labelpad=1)
    ax.set_ylabel(r"$y_M$ (m)", labelpad=1)
    ax.set_zlabel(r"$z_M$ (m)", labelpad=1)
    ax.tick_params(labelsize=7, pad=1)
    ps.style_3d(ax)
    ax.legend(loc="upper left", fontsize=6.2, frameon=False, handletextpad=0.3)
    return ps.save_figure(fig, out / "P7_compare_3d")


def p8_axis_errors(ds, err, out: Path) -> list[Path]:
    fig, axes = ps.new_figure(3, 1, width="double", height=4.0, sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(ds.t_rel, err[f"e{'xyz'[i]}"], color=COLORS["error"], lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(rf"$e_{{{'xyz'[i]}}}$ (m)")
        ax.set_xlim(ds.t_rel[0], ds.t_rel[-1])
    axes[-1].set_xlabel(r"$t$ (s)")
    return ps.save_figure(fig, out / "P8_axis_errors")


def p9_xy_3d_errors(ds, err, out: Path) -> list[Path]:
    fig, ax = ps.new_figure(1, 1, width="double", aspect=0.34)
    ax.plot(ds.t_rel, err["e_xy"], color=COLORS["a2"], lw=0.9, label=r"$e_{xy}$")
    ax.plot(ds.t_rel, err["e_3d"], color=COLORS["error"], lw=0.9, label=r"$e_{3d}$")
    ax.set_xlim(ds.t_rel[0], ds.t_rel[-1])
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$e$ (m)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=7)
    return ps.save_figure(fig, out / "P9_xy_3d_errors")


def p10_rmse_bar(summary, out: Path) -> list[Path]:
    fig, ax = ps.new_figure(1, 1, width="single", aspect=0.85)
    comps = summary["component"].to_numpy()
    vals = summary["rmse_m"].to_numpy()
    colors = [COLORS["a0"], COLORS["a1"], COLORS["a3"], COLORS["a2"], COLORS["error"]]
    bars = ax.bar(np.arange(len(comps)), vals, width=0.62,
                  color=colors[:len(comps)], edgecolor="k", linewidth=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
                va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(comps)), [c.upper() for c in comps])
    ax.set_ylabel("RMSE (m)")
    ax.set_ylim(0, vals.max() * 1.22)
    return ps.save_figure(fig, out / "P10_rmse_bar")


def p11_cdf(err, out: Path) -> list[Path]:
    fig, ax = ps.new_figure(1, 1, width="single", aspect=0.85)
    series = {
        "x": err["ex"].to_numpy(), "y": err["ey"].to_numpy(), "z": err["ez"].to_numpy(),
        "xy": err["e_xy"].to_numpy(), "3d": err["e_3d"].to_numpy(),
    }
    colors = [COLORS["a0"], COLORS["a1"], COLORS["a3"], COLORS["a2"], COLORS["error"]]
    for (name, e), c in zip(series.items(), colors):
        v = np.sort(np.abs(e[np.isfinite(e)]))
        ax.plot(v, np.arange(1, v.size + 1) / v.size, color=c, lw=1.0, label=name)
    ax.set_xlabel(r"$|e|$ (m)")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, np.nanpercentile(err["e_3d"], 99) * 1.05)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    return ps.save_figure(fig, out / "P11_error_cdf")


def p12_box(err, out: Path) -> list[Path]:
    fig, ax = ps.new_figure(1, 1, width="single", aspect=0.85)
    data = [err[k].to_numpy() for k in ("ex", "ey", "ez", "e_xy", "e_3d")]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=0.8))
    colors = [COLORS["a0"], COLORS["a1"], COLORS["a3"], COLORS["a2"], COLORS["error"]]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
        patch.set_edgecolor("k")
        patch.set_linewidth(0.4)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xticks(np.arange(1, 6), ["X", "Y", "Z", "XY", "3D"])
    ax.set_ylabel("error (m)")
    return ps.save_figure(fig, out / "P12_error_box")


def make_all(ds, err, summary, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    files = []
    files += p1_raw_xyz(ds, out)
    files += p2_mavros_xyz(ds, out)
    files += p3_xy(ds, out)
    files += p4_3d(ds, out)
    files += p5_compare_xyz(ds, out)
    files += p6_compare_xy(ds, out)
    files += p7_compare_3d(ds, out)
    files += p8_axis_errors(ds, err, out)
    files += p9_xy_3d_errors(ds, err, out)
    files += p10_rmse_bar(summary, out)
    files += p11_cdf(err, out)
    files += p12_box(err, out)
    return files

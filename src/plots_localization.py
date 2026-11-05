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


def p3_xy(ds, out: Path) -> list[Path]:
    h = ps.aspect_height(ps.SINGLE_COLUMN_IN, np.ptp(ds.uwb_mav[:, 0]),
                         np.ptp(ds.uwb_mav[:, 1]), extra=0.9)
    fig, ax = ps.new_figure(1, 1, width="single", height=h)
    ax.plot(ds.uwb_mav[:, 0], ds.uwb_mav[:, 1], color=COLORS["uwb"], lw=0.9)
    ax.plot(ds.uwb_mav[0, 0], ds.uwb_mav[0, 1], marker="o", ms=4,
            mfc=COLORS["a2"], mec="k", mew=0.3, ls="none")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_M$ (m)")
    ax.set_ylabel(r"$y_M$ (m)")
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
    pts = np.vstack([ds.truth_rel[:, :2], ds.uwb_mav[:, :2]])
    h = ps.aspect_height(ps.SINGLE_COLUMN_IN, np.ptp(pts[:, 0]),
                         np.ptp(pts[:, 1]), extra=0.9)
    fig, ax = ps.new_figure(1, 1, width="single", height=h)
    ax.plot(ds.truth_rel[:, 0], ds.truth_rel[:, 1], color=COLORS["truth"], lw=1.0,
            label="Ground truth")
    ax.plot(ds.uwb_mav[:, 0], ds.uwb_mav[:, 1], color=COLORS["uwb"], lw=0.9,
            ls=(0, (4, 2)), label="UWB")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_M$ (m)")
    ax.set_ylabel(r"$y_M$ (m)")
    ax.legend(loc="best", fontsize=7)
    return ps.save_figure(fig, out / "P6_compare_xy")


def p7_compare_3d(ds, out: Path) -> list[Path]:
    fig = plt.figure(figsize=ps.figsize("single", height=3.0), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ds.truth_rel[:, 0], ds.truth_rel[:, 1], ds.truth_rel[:, 2],
            color=COLORS["truth"], lw=0.9, label="Ground truth")
    ax.plot(ds.uwb_mav[:, 0], ds.uwb_mav[:, 1], ds.uwb_mav[:, 2],
            color=COLORS["uwb"], lw=0.8, ls=(0, (3, 2)), label="UWB")
    ps.set_equal_3d(ax, np.vstack([ds.truth_rel, ds.uwb_mav]))
    ax.view_init(26, -60)
    ax.set_xlabel(r"$x_M$ (m)", labelpad=1)
    ax.set_ylabel(r"$y_M$ (m)", labelpad=1)
    ax.set_zlabel(r"$z_M$ (m)", labelpad=1)
    ax.tick_params(labelsize=7, pad=1)
    ps.style_3d(ax)
    ax.legend(loc="upper left", fontsize=7)
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

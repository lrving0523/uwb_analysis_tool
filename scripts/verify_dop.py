"""DOP 专项校验：基站坐标、雅可比、公式、条件数、蒙特卡洛、分支归属。

输出：
  results/tables/dop_verification.csv
  results/tables/dop_branch_stats.csv
  results/reports/dop_verification.md
  results/figures/joint/J6_dop_validation.png/.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from src import anchor_geometry as geom
from src import data_loader as dl
from src import plot_style as ps
from src.console import enable_utf8
from src.joint_analysis import dop, jacobian


# --------------------------------------------------------------------------------------
# 基础计算
# --------------------------------------------------------------------------------------
def dop_at(point: np.ndarray, anchors: np.ndarray) -> dict:
    """单点的 H、G、HDOP/VDOP/GDOP 与条件数（显式公式）。"""
    h = jacobian(np.asarray(point, float).reshape(1, 3), anchors)[0]
    g = np.linalg.inv(h.T @ h)
    hdop = float(np.sqrt(g[0, 0] + g[1, 1]))
    vdop = float(np.sqrt(g[2, 2]))
    gdop = float(np.sqrt(np.trace(g)))
    return {
        "hdop": hdop, "vdop": vdop, "gdop": gdop,
        "gdop_from_hv": float(np.hypot(hdop, vdop)),
        "cond_H": float(np.linalg.cond(h)),
        "cond_HtH": float(np.linalg.cond(g)),
        "eig_min": float(np.min(np.linalg.eigvalsh(h.T @ h))),
        "eig_max": float(np.max(np.linalg.eigvalsh(h.T @ h))),
    }


def solve_position(ranges: np.ndarray, anchors: np.ndarray, p0: np.ndarray) -> np.ndarray:
    sol = least_squares(
        lambda x: np.linalg.norm(x[None, :] - anchors, axis=1) - ranges,
        p0, method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return sol.x


def monte_carlo_std(point: np.ndarray, anchors: np.ndarray, sigma: float = 0.05,
                    n: int = 200, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    r0 = np.linalg.norm(point[None, :] - anchors, axis=1)
    out = np.zeros((n, 3))
    for k in range(n):
        out[k] = solve_position(r0 + rng.normal(0, sigma, 4), anchors, point)
    std = out.std(axis=0)
    return {"mc_std_x": float(std[0]), "mc_std_y": float(std[1]),
            "mc_std_z": float(std[2]),
            "mc_bias_z": float(out[:, 2].mean() - point[2])}


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(floatfmt.format(float(v)) if isinstance(v, (float, np.floating))
                         else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def branch_stats(ds, vdop: np.ndarray) -> pd.DataFrame:
    """按“水平距阵列中心距离 × 高度”统计 VDOP，定位两条分支。"""
    xy = np.linalg.norm(ds.truth_rel[:, :2], axis=1)
    z = ds.truth_rel[:, 2]
    d_bins = [0, 0.3, 0.6, 1.0, 1.4, 2.0]
    z_bins = [0, 0.15, 0.3, 0.6, 1.0]
    rows = []
    for i in range(len(d_bins) - 1):
        for j in range(len(z_bins) - 1):
            m = ((xy >= d_bins[i]) & (xy < d_bins[i + 1]) &
                 (z >= z_bins[j]) & (z < z_bins[j + 1]))
            if m.sum() == 0:
                continue
            rows.append({
                "distance_bin_m": f"{d_bins[i]:.1f}-{d_bins[i+1]:.1f}",
                "z_bin_m": f"{z_bins[j]:.2f}-{z_bins[j+1]:.2f}",
                "n": int(m.sum()),
                "vdop_median": float(np.median(vdop[m])),
                "vdop_min": float(vdop[m].min()),
                "vdop_max": float(vdop[m].max()),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------------------
def plot_validation(sel: pd.DataFrame, out: Path) -> list[Path]:
    fig, axes = ps.new_figure(1, 2, width="double", aspect=0.42)

    ax = axes[0]
    ax.plot(sel["vdop_x_sigma"], sel["mc_std_z"], ls="none", marker="o", ms=3.5,
            mfc="none", mec=ps.COLORS["a0"])
    lim = [min(sel["vdop_x_sigma"].min(), sel["mc_std_z"].min()) * 0.7,
           max(sel["vdop_x_sigma"].max(), sel["mc_std_z"].max()) * 1.4]
    ax.plot(lim, lim, color="k", lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"Linear prediction: VDOP $\times \sigma_r$ (m)")
    ax.set_ylabel("Monte-Carlo std of $z$ (m)")
    ax.text(0.04, 0.96, r"$\sigma_r$ = 0.05 m, 200 trials/point", transform=ax.transAxes,
            va="top", fontsize=7)
    ps.panel_label(ax, "(a)")

    ax = axes[1]
    ax.plot(sel["cond_HtH"], sel["vdop"], ls="none", marker="s", ms=3,
            mfc="none", mec=ps.COLORS["error"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"cond$(H^{\top}H)$")
    ax.set_ylabel("VDOP")
    ps.panel_label(ax, "(b)")

    return ps.save_figure(fig, out / "J6_dop_validation")


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def run(cfg: dict, results: Path | None = None, ds=None) -> dict:
    results = Path(results or (Path(cfg["_root"]) / cfg["output"]["results_dir"]))
    tab_dir = results / "tables"
    rep_dir = results / "reports"
    fig_dir = results / "figures" / "joint"
    for d in (tab_dir, rep_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    ds = ds or dl.build_dataset(cfg)
    geo = geom.check_geometry(ds.anchors_world)
    d_all = dop(ds.truth_abs, ds.anchors_world)
    vdop = d_all["vdop"].to_numpy()
    z = ds.truth_rel[:, 2]
    xy = np.linalg.norm(ds.truth_rel[:, :2], axis=1)

    # 代表点：起点、爬升、阵列上方、盘旋、落地
    idx = [0,
           int(np.argmin(np.abs(ds.t_rel - 20))),
           int(np.argmin(np.abs(ds.t_rel - 24))),
           int(np.argmin(np.abs(ds.t_rel - 26))),
           int(np.argmin(np.abs(ds.t_rel - 30))),
           int(np.argmin(np.abs(ds.t_rel - 34))),
           len(ds.t_rel) - 1]
    rows = []
    for i in idx:
        p = ds.truth_abs[i]
        base = dop_at(p, ds.anchors_world)
        mc = monte_carlo_std(p, ds.anchors_world, sigma=0.05, n=200)
        rows.append({
            "t_s": float(ds.t_rel[i]),
            "x_m": float(ds.truth_rel[i, 0]), "y_m": float(ds.truth_rel[i, 1]),
            "z_m": float(z[i]),
            "dist_xy_m": float(xy[i]),
            "hdop": base["hdop"], "vdop": base["vdop"], "gdop": base["gdop"],
            "gdop_check": base["gdop_from_hv"],
            "cond_H": base["cond_H"], "cond_HtH": base["cond_HtH"],
            "eig_min": base["eig_min"], "eig_max": base["eig_max"],
            "vdop_x_sigma": base["vdop"] * 0.05,
            **mc,
        })
    sel = pd.DataFrame(rows)
    sel.to_csv(tab_dir / "dop_verification.csv", index=False, float_format="%.6f")

    branches = branch_stats(ds, vdop)
    branches.to_csv(tab_dir / "dop_branch_stats.csv", index=False, float_format="%.6f")

    files = plot_validation(sel, fig_dir)

    # ---- Markdown ----
    lines = [
        "# DOP 校验报告（J5 / J6）",
        "",
        "## 1. 基站坐标与几何",
        "",
        "- 四基站世界坐标（m）：",
    ]
    for i, a in enumerate(ds.anchors_world):
        lines.append(f"  - A{i}: ({a[0]:+.4f}, {a[1]:+.4f}, {a[2]:+.4f})")
    h_all = jacobian(ds.truth_abs, ds.anchors_world)
    g_all = np.einsum("nij,nik->njk", h_all, h_all)
    cond_all = np.linalg.cond(g_all)
    above = sel[(sel["t_s"] >= 25) & (sel["t_s"] <= 27)]
    lines += [
        f"- 相邻间距 {np.round(geo['adjacent_distances_m'], 4).tolist()} m，"
        f"对角 {np.round(geo['diagonal_distances_m'], 4).tolist()} m，"
        f"高度展布 {geo['planarity_z_spread_m']:.2e} m（共面）",
        f"- 平台中心 z = {ds.p0[2]:.4f} m；无人机绝对 z ∈ "
        f"[{ds.truth_abs[:,2].min():.3f}, {ds.truth_abs[:,2].max():.3f}] m，"
        f"相对平台面高度 z ∈ [{z.min():.3f}, {z.max():.3f}] m",
        "",
        "## 2. 公式与实现",
        "",
        r"- 雅可比：$H_i = (p - a_i)/\Vert p - a_i\Vert$，$G = (H^{\top}H)^{-1}$",
        r"- HDOP $=\sqrt{G_{xx}+G_{yy}}$，VDOP $=\sqrt{G_{zz}}$，"
        r"GDOP $=\sqrt{\mathrm{tr}(G)}$",
        f"- 逐点交叉验证：GDOP 与 $\\sqrt{{HDOP^2+VDOP^2}}$ 的最大偏差 "
        f"{np.max(np.abs(sel['gdop'] - sel['gdop_check'])):.2e}",
        "",
        "## 3. 条件数与分支",
        "",
        f"- 全轨迹 cond$(H^TH)$ 中位 {np.median(cond_all):.1f}，"
        f"最大 {np.max(cond_all):.1f}",
        f"- 起飞前（远处、贴地）：cond$(H)$={sel.iloc[0]['cond_H']:.1f}，"
        f"VDOP={sel.iloc[0]['vdop']:.1f}",
        f"- 阵列上方（z≈0.7 m）：VDOP={above['vdop'].min():.2f}" if len(above) else "",
        "",
        "分支统计（按水平距离与高度分箱）：",
        "",
        _md_table(branches),
        "",
        "## 4. 蒙特卡洛校验（σ_r = 0.05 m，200 次/点）",
        "",
        _md_table(sel[["t_s", "x_m", "y_m", "z_m", "dist_xy_m", "vdop", "vdop_x_sigma",
                       "mc_std_z", "cond_HtH"]]),
        "",
        "结论：线性化 DOP 与蒙特卡洛在良态几何（阵列上方）一致；",
        "在病态几何（远处 + 贴地，cond≈1e4）线性 DOP 会显著高估实际垂直散布，",
        "因此 VDOP 只能作为几何条件的相对指标，不应直接乘 σ 当作绝对误差。",
        "",
        f"![J6](../figures/joint/J6_dop_validation.png)",
    ]

    report = rep_dir / "dop_verification.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "anchors_world": ds.anchors_world.round(4).tolist(),
        "platform_z": float(ds.p0[2]),
        "cond_H": sel["cond_H"].round(2).tolist(),
        "vdop": sel["vdop"].round(2).tolist(),
        "branches": branches.to_dict(orient="records"),
        "files": [str(p) for p in files],
        "report": str(report),
    }
    (tab_dir / "dop_verification.json").write_text(
        __import__("json").dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[dop] 校验完成：{report}")
    return {"ds": ds, "selection": sel, "branches": branches, "files": files,
            "report": report}


def main() -> int:
    enable_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    args = ap.parse_args()
    cfg = dl.load_config(args.config)
    ps.use_ieee_style(dpi=cfg.get("plots", {}).get("dpi", 300))
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

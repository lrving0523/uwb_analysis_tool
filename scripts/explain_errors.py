"""分解定位误差：水平误差能否由测距误差经几何放大解释。

对每个数据集，用 ``e_hat = (H_xy^T H_xy)^-1 H_xy^T dr`` 把逐帧测距误差映射到
水平位置，比较 ``|e_xy|`` 与 ``|e_hat|``、拟合系数 ``k`` 与相关系数。
竖直方向锚点共面，DOP 极大，单靠测距无法解释，另列 VDOP 供参考。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from src import data_loader as dl  # noqa: E402
from src import plot_style as ps  # noqa: E402
from src.joint_analysis import jacobian  # noqa: E402


def configs() -> list[tuple[str, Path]]:
    items = [("base", ROOT / "configs" / "config.yaml")]
    for sub in sorted((ROOT / "configs").iterdir()):
        if sub.is_dir():
            items += [(c.stem, c) for c in sorted(sub.glob("*.yaml"))]
    return items


def analyze(ds) -> dict:
    e = ds.uwb_mav - ds.truth_rel
    dr = np.nan_to_num(ds.ranges_meas - ds.ranges_true)
    H = jacobian(ds.truth_rel, ds.anchors_world)
    Hxy = H[:, :, :2]
    G = np.linalg.pinv(np.einsum("nki,nkj->nij", Hxy, Hxy))
    e_hat = np.einsum("nij,nkj,nk->ni", G, Hxy, dr)
    G3 = np.linalg.pinv(np.einsum("nki,nkj->nij", H, H))

    exy = np.linalg.norm(e[:, :2], axis=1)
    eh = np.linalg.norm(e_hat, axis=1)
    k = float(np.sum(e[:, :2] * e_hat) / np.sum(e_hat * e_hat))
    corr = float(np.corrcoef(e[:, :2].ravel(), e_hat.ravel())[0, 1])
    return {
        "e_xy_rmse": float(np.sqrt(np.mean(exy ** 2))),
        "e_hat_rmse": float(np.sqrt(np.mean(eh ** 2))),
        "k": k,
        "corr": corr,
        "e_z_rmse": float(np.sqrt(np.mean(e[:, 2] ** 2))),
        "vdop": float(np.median(np.sqrt(G3[:, 2, 2]))),
    }


def figure(df: pd.DataFrame, out: Path) -> list[Path]:
    fig, axes = ps.new_figure(1, 2, width="double", aspect=0.42)
    x = np.arange(len(df))
    ax = axes[0]
    ax.bar(x - 0.2, df["e_xy_rmse"], width=0.38, color=ps.COLORS["a0"],
           edgecolor="k", lw=0.4, label="Actual")
    ax.bar(x + 0.2, df["e_hat_rmse"], width=0.38, color=ps.COLORS["a1"],
           edgecolor="k", lw=0.4, label="Predicted from ranges")
    ax.set_xticks(x, df["dataset"], rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("Horizontal RMSE (m)")
    ax.legend(fontsize=6, frameon=False)
    ps.panel_label(ax, "(a)")

    ax = axes[1]
    sc = ax.scatter(df["vdop"], df["corr"], s=26, c=df["k"], cmap="viridis",
                    edgecolors="k", linewidths=0.4, vmin=-0.2, vmax=0.9)
    offsets = {"19037": (-24, 5), "20074": (4, -10), "8650": (4, 4),
               "17394": (-30, -11), "6903": (-30, -11), "15068": (4, -11),
               "11510": (4, 3), "30437_REVIEW": (4, 3), "base": (5, 2)}
    for _, r in df.iterrows():
        ax.annotate(r["dataset"], (r["vdop"], r["corr"]),
                    xytext=offsets.get(r["dataset"], (3, 3)),
                    textcoords="offset points", fontsize=5.4)
    ax.set_xscale("log")
    ax.set_xlabel("VDOP (median, log)")
    ax.set_ylabel("Correlation (actual, predicted)")
    ax.set_ylim(-0.6, 1.05)
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("gain $k$", fontsize=7)
    ps.panel_label(ax, "(b)")
    return ps.save_figure(fig, out / "error_decomposition")


def main() -> int:
    out = ROOT / "results" / "summary_all"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, cfg in configs():
        ds = dl.build_dataset(dl.load_config(cfg))
        rows.append({"dataset": label, **analyze(ds)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "error_decomposition.csv", index=False, float_format="%.3f")
    files = figure(df, out)

    print(df.round(3).to_string(index=False))
    good = df[df["vdop"] < 5]
    print("\nVDOP<5 的数据集: corr mean", round(good["corr"].mean(), 3),
          " k mean", round(good["k"].mean(), 3))
    print("\noutput:", out)
    for p in files:
        print(" -", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""汇总所有已分析数据集：测距（分站）与定位误差，回答“定位误差是否由测距误差主导”。

用法::

    python scripts/summarize_all.py
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


def discover() -> list[tuple[str, Path]]:
    items = [("base", ROOT / "configs" / "config.yaml")]
    for sub in sorted((ROOT / "configs").iterdir()):
        if sub.is_dir():
            for cfg in sorted(sub.glob("*.yaml")):
                label = cfg.stem
                items.append((label, cfg))
    return items


def analyze(label: str, cfg_path: Path) -> tuple[dict, list[dict]]:
    ds = dl.build_dataset(dl.load_config(cfg_path))
    err_r = ds.ranges_meas - ds.ranges_true
    e = ds.uwb_mav - ds.truth_rel
    e_n = np.linalg.norm(e, axis=1)
    rms = np.sqrt(np.nanmean(err_r ** 2, axis=1))
    m = np.isfinite(rms)
    corr = float(np.corrcoef(rms[m], e_n[m])[0, 1]) if m.sum() > 5 else np.nan

    H = jacobian(ds.truth_rel, ds.anchors_world)
    G = np.linalg.pinv(np.einsum("nki,nkj->nij", H, H))
    b = np.nanmean(err_r, axis=0)
    dp = np.einsum("nij,nj->ni", G, np.einsum("nki,k->ni", H, b))
    bias = e.mean(axis=0)
    rmse3 = float(np.sqrt(np.mean(e_n ** 2)))

    row = {
        "dataset": label,
        "n": ds.n,
        "duration_s": float(ds.t[-1] - ds.t[0]),
        "range_rmse": float(np.sqrt(np.nanmean(err_r ** 2))),
        "range_bias": float(np.nanmean(err_r)),
        "range_p95": float(np.nanpercentile(np.abs(err_r[np.isfinite(err_r)]), 95)),
        "loc_rmse_3d": rmse3,
        "loc_bias_3d": float(np.linalg.norm(bias)),
        "loc_std_3d": float(np.sqrt(np.mean(np.sum((e - bias) ** 2, axis=1)))),
        "bias_share": float(np.linalg.norm(bias) ** 2 / rmse3 ** 2),
        "corr_range_loc": corr,
        "pred_bias_3d": float(np.linalg.norm(dp.mean(axis=0))),
        "hdop": float(np.median(np.sqrt(G[:, 0, 0] + G[:, 1, 1]))),
        "vdop": float(np.median(np.sqrt(G[:, 2, 2]))),
    }
    anchors = []
    for i in range(4):
        x = err_r[:, i][np.isfinite(err_r[:, i])]
        anchors.append({"dataset": label, "anchor": f"A{i}",
                        "bias": float(x.mean()), "rmse": float(np.sqrt(np.mean(x ** 2))),
                        "mae": float(np.abs(x).mean()),
                        "p95": float(np.percentile(np.abs(x), 95))})
    return row, anchors


def figures(df: pd.DataFrame, an: pd.DataFrame, out: Path) -> list[Path]:
    piv = an.pivot(index="dataset", columns="anchor", values="rmse")
    piv_b = an.pivot(index="dataset", columns="anchor", values="bias")

    fig, axes = ps.new_figure(1, 2, width="double", aspect=0.42)
    ax = axes[0]
    im = ax.imshow(piv.to_numpy(), cmap="viridis", aspect="auto", vmin=0,
                   vmax=float(np.nanmax(piv.to_numpy())))
    ax.set_xticks(range(4), list(piv.columns))
    ax.set_yticks(range(len(piv)), list(piv.index))
    for i in range(len(piv)):
        for j in range(4):
            v = piv.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if v < 0.6 * np.nanmax(piv.to_numpy()) else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04, shrink=0.9)
    cb.set_label("Ranging RMSE (m)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("(a) Per-anchor ranging RMSE", fontsize=8, loc="left")

    ax = axes[1]
    base = df[df["dataset"] == "base"]
    rest = df[df["dataset"] != "base"]
    ax.scatter(rest["range_rmse"], rest["loc_rmse_3d"], s=24, color=ps.COLORS["a0"],
               edgecolors="k", linewidths=0.4, zorder=3, label="flights")
    ax.scatter(base["range_rmse"], base["loc_rmse_3d"], s=28, color="0.55",
               edgecolors="k", linewidths=0.4, zorder=3, label="base (2.6x scale)")

    lim = max(df["range_rmse"].max(), df["loc_rmse_3d"].max()) * 1.15
    ax.plot([0, lim], [0, lim], ls="--", lw=0.8, color="0.5", label="1:1")

    k, c = np.polyfit(rest["range_rmse"], rest["loc_rmse_3d"], 1)
    xx = np.linspace(0, rest["range_rmse"].max() * 1.1, 10)
    ax.plot(xx, k * xx + c, ls=":", lw=0.9, color=ps.COLORS["a1"],
            label=f"fit (no base): {k:.2f}x + {c:.2f}")
    ax.set_xlabel("Ranging RMSE (m)")
    ax.set_ylabel("Localization 3D RMSE (m)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax.set_title("(b) Localization vs ranging", fontsize=8, loc="left")

    return ps.save_figure(fig, out / "ranging_vs_localization")


def md_table(d: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in d.iterrows():
        cells = [f"{row[c]:.3f}" if isinstance(row[c], (float, np.floating)) else str(row[c])
                 for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    out = ROOT / "results" / "summary_all"
    out.mkdir(parents=True, exist_ok=True)

    rows, anchors = [], []
    for label, cfg in discover():
        r, a = analyze(label, cfg)
        rows.append(r)
        anchors.extend(a)
        print(f"[ok] {label}")

    df = pd.DataFrame(rows)
    an = pd.DataFrame(anchors)
    df.to_csv(out / "localization_summary.csv", index=False, float_format="%.4f")
    an.to_csv(out / "ranging_by_anchor.csv", index=False, float_format="%.4f")
    files = figures(df, an, out)

    piv = an.pivot(index="dataset", columns="anchor", values="rmse")
    piv_b = an.pivot(index="dataset", columns="anchor", values="bias")
    mean_rmse = an.groupby("anchor")["rmse"].mean()
    mean_bias = an.assign(ab=an["bias"].abs()).groupby("anchor")["ab"].mean()

    lines = [
        "# 全数据集汇总：测距误差 → 定位误差",
        "",
        f"共 {len(df)} 个数据集。",
        "",
        "## 1. 总表",
        "",
        md_table(df, ["dataset", "n", "range_rmse", "range_bias", "loc_rmse_3d",
                      "loc_bias_3d", "loc_std_3d", "bias_share", "corr_range_loc"]),
        "",
        "## 2. 分站测距 RMSE（m）",
        "",
        md_table(piv.reset_index(), ["dataset"] + list(piv.columns)),
        "",
        "## 3. 分站测距 bias（m）",
        "",
        md_table(piv_b.reset_index(), ["dataset"] + list(piv_b.columns)),
        "",
        "## 4. 站位排名",
        "",
        f"- 平均 RMSE：{', '.join(f'{k} {v:.3f}' for k, v in mean_rmse.items())}",
        f"- 平均 |bias|：{', '.join(f'{k} {v:.3f}' for k, v in mean_bias.items())}",
        "",
        "![summary](ranging_vs_localization.png)",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(df.round(3).to_string(index=False))
    print("\nper-anchor mean RMSE:", mean_rmse.round(3).to_dict())
    print("per-anchor mean |bias|:", mean_bias.round(3).to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())

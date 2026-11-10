"""用悬停/原地起降数据标定 UWB 基站测距偏置。

标定模型（每个基站独立）：

* 常数模型：``b_i = const``
* 高度线性模型：``b_i(z) = a_i + c_i * z``（z 为动捕真值高度）

并用"前半段拟合、后半段验证"的方式给出无偏的精度评估，同时支持把标定
结果套用到其它数据集，检查偏置是否跨架次可迁移。

用法::

    python scripts/calibrate_bias.py --config configs/batch3_csv/11510.yaml
    python scripts/calibrate_bias.py --config configs/batch3_csv/11510.yaml --apply configs/flight_takeoff_landing/*.yaml
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

import matplotlib.pyplot as plt  # noqa: E402

from src import data_loader as dl  # noqa: E402
from src import plot_style as ps  # noqa: E402


def range_errors(ds) -> np.ndarray:
    return ds.ranges_meas - ds.ranges_true


def fit_models(ds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    err = range_errors(ds)
    z = ds.truth_rel[:, 2]
    const = np.nanmean(err, axis=0)
    lin = np.array([np.polyfit(z, err[:, i], 1) for i in range(4)])  # (4, 2): slope, intercept
    return err, const, lin


def rmse(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2)))


def split_half(ds) -> dict:
    err = range_errors(ds)
    z = ds.truth_rel[:, 2]
    half = ds.n // 2
    const = np.nanmean(err[:half], axis=0)
    lin = np.array([np.polyfit(z[:half], err[:half, i], 1) for i in range(4)])
    test = err[half:]
    pred_c = const[None, :]
    pred_l = np.column_stack([np.polyval(lin[i], z[half:]) for i in range(4)])

    # 位置域：设备输出的误差对高度做线性模型（仅用于同架次验证）
    e = ds.uwb_mav - ds.truth_rel
    coef = [np.polyfit(z[:half], e[:half, i], 1) for i in range(3)]
    pred_p = np.column_stack([np.polyval(coef[i], z[half:]) for i in range(3)])
    pos_before = float(np.sqrt(np.mean(np.sum(e[half:] ** 2, axis=1))))
    pos_after = float(np.sqrt(np.mean(np.sum((e[half:] - pred_p) ** 2, axis=1))))

    return {
        "before": rmse(test),
        "const": rmse(test - pred_c),
        "linear": rmse(test - pred_l),
        "pos_before": pos_before,
        "pos_after": pos_after,
        "pos_coef": coef,
    }


def transfer(ds, const: np.ndarray, lin: np.ndarray) -> dict:
    err = range_errors(ds)
    z = ds.truth_rel[:, 2]
    pred_l = np.column_stack([np.polyval(lin[i], z) for i in range(4)])
    return {"dataset": ds.name, "before": rmse(err),
            "const": rmse(err - const[None, :]), "linear": rmse(err - pred_l)}


def figure(ds, const, lin, out: Path) -> list[Path]:
    err = range_errors(ds)
    z = ds.truth_rel[:, 2]
    fig, axes = ps.new_figure(1, 2, width="double", aspect=0.42)

    ax = axes[0]
    zz = np.linspace(z.min(), z.max(), 50)
    for i in range(4):
        ax.scatter(z, err[:, i], s=2, alpha=0.25, color=ps.ANCHOR_COLORS[i],
                   edgecolors="none", rasterized=True)
        ax.plot(zz, np.polyval(lin[i], zz), color=ps.ANCHOR_COLORS[i], lw=1.0,
                label=f"A{i}")
    ax.set_xlabel(r"True height $z$ (m)")
    ax.set_ylabel("Range error (m)")
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="lower right")
    ps.panel_label(ax, "(a)")

    ax = axes[1]
    series = (
        ("raw", err.ravel(), "#555555"),
        ("const-corrected", (err - const[None, :]).ravel(), ps.COLORS["a1"]),
        ("z-linear-corrected", (err - np.column_stack(
            [np.polyval(lin[j], z) for j in range(4)])).ravel(), ps.COLORS["a0"]),
    )
    for name, vals, col in series:
        v = np.abs(vals[np.isfinite(vals)])
        v = np.sort(v)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), lw=0.9, color=col,
                label=f"{name} (RMSE {rmse(vals):.3f} m)")
    ax.set_xlabel("|Range error| (m)")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, 0.8)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ps.panel_label(ax, "(b)")

    return ps.save_figure(fig, out / "bias_calibration")


def main() -> int:
    ap = argparse.ArgumentParser(description="UWB 基站偏置标定")
    ap.add_argument("--config", default=str(ROOT / "configs" / "batch3_csv" / "11510.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results" / "calibration"))
    ap.add_argument("--apply", nargs="*", default=None, help="检查迁移性的其它配置")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = dl.build_dataset(dl.load_config(args.config))
    ds.name = Path(args.config).stem
    err, const, lin = fit_models(ds)
    sh = split_half(ds)

    rows = []
    for i in range(4):
        rows.append({
            "anchor": f"A{i}",
            "const_bias_m": const[i],
            "z_slope_m_per_m": lin[i][0],
            "z_intercept_m": lin[i][1],
            "std_m": float(np.nanstd(err[:, i])),
            "rmse_raw_m": rmse(err[:, i]),
            "rmse_const_m": rmse(err[:, i] - const[i]),
            "rmse_linear_m": rmse(err[:, i] - np.polyval(lin[i], ds.truth_rel[:, 2])),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / "anchor_bias.csv", index=False, float_format="%.4f")

    transfer_rows = []
    if args.apply:
        for pattern in args.apply:
            for cfg in sorted(ROOT.glob(pattern)):
                other = dl.build_dataset(dl.load_config(cfg))
                other.name = cfg.stem
                transfer_rows.append(transfer(other, const, lin))
    tf = pd.DataFrame(transfer_rows)
    if not tf.empty:
        tf.to_csv(out / "transfer_check.csv", index=False, float_format="%.4f")

    files = figure(ds, const, lin, out)

    def table(d: pd.DataFrame, fmt: str = "{:.3f}") -> str:
        cols = list(d.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, r in d.iterrows():
            lines.append("| " + " | ".join(
                fmt.format(r[c]) if isinstance(r[c], (float, np.floating)) else str(r[c])
                for c in cols) + " |")
        return "\n".join(lines)

    lines = [
        f"# 基站偏置标定（{ds.name}）",
        "",
        f"标定数据：`{Path(args.config).as_posix()}`，{ds.n} 帧，"
        f"时长 {ds.t[-1]-ds.t[0]:.1f} s，真值高度 {ds.truth_rel[:,2].min():.2f}–"
        f"{ds.truth_rel[:,2].max():.2f} m。",
        "",
        "## 1. 分站偏置",
        "",
        table(df),
        "",
        "- `const_bias_m`：常数偏置（实测 − 真值）",
        "- `z_slope_m_per_m` / `z_intercept_m`：偏置随真值高度的线性模型 `b(z)=a+c·z`",
        "",
        "## 2. 分段验证（前半段拟合 → 后半段验证）",
        "",
        f"- 校正前 RMSE：**{sh['before']:.3f} m**",
        f"- 常数模型校正后：**{sh['const']:.3f} m**",
        f"- 高度线性模型校正后：**{sh['linear']:.3f} m**",
        "",
        "## 3. 位置域验证（同架次，前半段拟合 → 后半段验证）",
        "",
        f"- 偏差模型：`e_x(z)={sh['pos_coef'][0][0]:+.3f}·z{sh['pos_coef'][0][1]:+.3f}`，"
        f"`e_y(z)={sh['pos_coef'][1][0]:+.3f}·z{sh['pos_coef'][1][1]:+.3f}`，"
        f"`e_z(z)={sh['pos_coef'][2][0]:+.3f}·z{sh['pos_coef'][2][1]:+.3f}`",
        f"- 校正前 3D RMSE：**{sh['pos_before']:.3f} m** → 校正后：**{sh['pos_after']:.3f} m**",
        "",
        "## 4. 迁移性检查",
        "",
    ]
    if tf.empty:
        lines.append("未提供其它数据集。")
    else:
        lines.append(table(tf))
        lines.append("")
        lines.append("若其它架次的 RMSE 没有下降甚至变差，说明这些偏置是"
                     "架次相关的（安装/姿态/环境不同），不能跨架次套用。")

    (out / "calibration.md").write_text("\n".join(lines), encoding="utf-8")

    print(df.round(4).to_string(index=False))
    print("\nsplit-half:", {k: round(v, 3) for k, v in sh.items()
                            if isinstance(v, (int, float))})
    print("position coefs:", [np.round(c, 3).tolist() for c in sh["pos_coef"]])
    if not tf.empty:
        print("\ntransfer:")
        print(tf.round(3).to_string(index=False))
    print("\noutput:", out)
    for p in files:
        print(" -", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

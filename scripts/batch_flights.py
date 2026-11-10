"""多架次批量分析：对一组配置依次运行完整流程，并汇总对比。

用法::

    python scripts/batch_flights.py --config-dir configs/flight_takeoff_landing \
        --out results/flight_takeoff_landing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib.pyplot as plt  # noqa: E402

import batch_analysis  # noqa: E402
from src import data_loader as dl  # noqa: E402
from src import plot_style as ps  # noqa: E402
from src.console import enable_utf8  # noqa: E402


def _row(summary: dict, cfg: dict) -> dict:
    loc = summary["localization"]
    return {
        "flight": cfg["project"]["bag"],
        "n_uwb": summary["dataset"]["n_uwb"],
        "duration_s": round(summary["dataset"]["duration_s"], 2),
        "range_rmse_m": summary["ranging"]["rmse_m"],
        "range_bias_m": summary["ranging"]["bias_m"],
        "range_p95_m": summary["ranging"]["p95_abs_m"],
        "loc_rmse_x_m": loc["x"]["rmse_m"],
        "loc_rmse_y_m": loc["y"]["rmse_m"],
        "loc_rmse_z_m": loc["z"]["rmse_m"],
        "loc_rmse_xy_m": loc["xy"]["rmse_m"],
        "loc_rmse_3d_m": loc["3d"]["rmse_m"],
        "hdop_median": summary["dop"]["hdop_median"],
        "vdop_median": summary["dop"]["vdop_median"],
        "gdop_median": summary["dop"]["gdop_median"],
        "scale_x": summary["scale_diagnostic"]["x"]["slope_k"],
        "scale_y": summary["scale_diagnostic"]["y"]["slope_k"],
        "scale_z": summary["scale_diagnostic"]["z"]["slope_k"],
        "uwb_rate_hz": summary["quality"]["sampling"]["rate_hz"],
        "valid_range_ratio": summary["quality"]["valid_range_ratio"],
        "landing_offset_m": summary["inspection"]["landing_offset_m"],
        "platform_std_max_m": summary["inspection"]["platform_std_max_m"],
    }


def comparison_figure(df: pd.DataFrame, out: Path) -> list[Path]:
    fig, axes = ps.new_figure(1, 2, width="double", aspect=0.45)
    x = np.arange(len(df))
    labels = [str(v).replace("takeoff_landing_", "") for v in df["flight"]]

    ax = axes[0]
    ax.bar(x - 0.2, df["range_rmse_m"], width=0.38, color=ps.COLORS["a0"],
           edgecolor="k", lw=0.4, label="Ranging RMSE")
    ax.bar(x + 0.2, df["range_p95_m"], width=0.38, color=ps.COLORS["a2"],
           edgecolor="k", lw=0.4, label="Ranging P95")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Error (m)")
    ax.set_xlabel("Flight")
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ps.panel_label(ax, "(a)")

    ax = axes[1]
    ax.bar(x - 0.2, df["loc_rmse_xy_m"], width=0.38, color=ps.COLORS["a2"],
           edgecolor="k", lw=0.4, label="Horizontal RMSE")
    ax.bar(x + 0.2, df["loc_rmse_3d_m"], width=0.38, color=ps.COLORS["a1"],
           edgecolor="k", lw=0.4, label="3D RMSE")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Error (m)")
    ax.set_xlabel("Flight")
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ps.panel_label(ax, "(b)")

    return ps.save_figure(fig, out / "comparison_ranging_localization")


def _md_table(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = [fmt.format(row[c]) if isinstance(row[c], (float, np.floating)) else str(row[c])
                 for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    enable_utf8()
    ap = argparse.ArgumentParser(description="多架次批量分析")
    ap.add_argument("--config-dir", default=str(ROOT / "configs" / "flight_takeoff_landing"))
    ap.add_argument("--out", default=str(ROOT / "results" / "flight_takeoff_landing"))
    args = ap.parse_args()

    cfg_dir = Path(args.config_dir)
    configs = sorted(cfg_dir.glob("*.yaml"))
    if not configs:
        raise SystemExit(f"没有找到配置：{cfg_dir}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    project_name = None
    for cfg_path in configs:
        cfg = dl.load_config(cfg_path)
        project_name = project_name or cfg["project"].get("name", "flight")
        ps.use_ieee_style(dpi=cfg.get("plots", {}).get("dpi", 300))
        print(f"\n########## {cfg['project']['bag']} ##########")
        summary = batch_analysis.run(cfg)
        rows.append(_row(summary, cfg))

    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False, float_format="%.4f")

    files = comparison_figure(df, out)

    lines = [
        f"# 多架次汇总（{project_name}）",
        "",
        f"共 {len(df)} 个架次；每个架次的详细结果在其子目录中。",
        "",
        _md_table(df[["flight", "n_uwb", "duration_s", "range_rmse_m", "range_bias_m",
                      "loc_rmse_x_m", "loc_rmse_y_m", "loc_rmse_z_m",
                      "loc_rmse_xy_m", "loc_rmse_3d_m"]]),
        "",
        "## 几何与尺度诊断",
        "",
        _md_table(df[["flight", "hdop_median", "vdop_median", "scale_x", "scale_y",
                      "scale_z", "uwb_rate_hz", "valid_range_ratio",
                      "landing_offset_m", "platform_std_max_m"]]),
        "",
        "说明：scale_* 为 `truth ≈ k·uwb + b` 的斜率；接近 1 表示 UWB 位置尺度正常。",
        "landing_offset_m 为落地后 5 s 无人机均值到平台中心的距离；"
        "platform_std_max_m 为平台位姿在整段记录中的最大标准差。",
        "",
        "## 注意事项",
        "",
    ]

    notes = []
    for _, r in df.iterrows():
        if r["platform_std_max_m"] > 0.01:
            notes.append(
                f"- **{r['flight']}**：平台位姿在录制过程中出现约 "
                f"{r['platform_std_max_m']*100:.1f} cm 的跳动（最大标准差），"
                "说明该架次的平台中心并非严格静止；当前按全段中位数定位基站，"
                "会给测距/定位结果引入额外偏差，建议单独复核或剔除该段。")
        if r["landing_offset_m"] > 0.20:
            notes.append(
                f"- **{r['flight']}**：落地后无人机距平台中心 "
                f"{r['landing_offset_m']:.3f} m，明显大于正常落点偏差；"
                "按任务假设未做任何平移补偿，这部分会体现为定位偏差（bias）。")
    if not notes:
        notes.append("- 未发现平台抖动或落地偏移异常。")
    lines += notes + [""]

    lines += [
        "![comparison](comparison_ranging_localization.png)",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8")

    print("\n== 汇总 ==")
    print(df[["flight", "range_rmse_m", "loc_rmse_xy_m", "loc_rmse_3d_m"]].to_string(index=False))
    print(f"\n输出：{out}")
    for p in files:
        print(" -", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

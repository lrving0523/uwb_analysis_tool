"""一键完成：数据检查 → 测距评估 → 定位评估 → 质量分析 → 联合分析 → Markdown 报告。

用法::

    python scripts/batch_analysis.py --config configs/config.yaml
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

from src import data_loader as dl
from src import plot_style as ps
from src.console import enable_utf8

sys.path.insert(0, str(ROOT / "scripts"))
import analyze_joint_errors  # noqa: E402
import analyze_uwb_quality  # noqa: E402
import evaluate_localization  # noqa: E402
import evaluate_ranging  # noqa: E402
import inspect_data  # noqa: E402
import verify_dop  # noqa: E402


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(float(v)))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _figure_links(folder: str, names: list[str], results: Path) -> list[str]:
    lines = []
    for name in names:
        p = results / "figures" / folder / f"{name}.png"
        if p.exists():
            lines.append(f"![{name}](../figures/{folder}/{name}.png)")
    return lines


def build_report(cfg, ds, insp, rang, loc, qual, joint, dopchk, results: Path) -> str:
    geo = insp["geometry"]
    mapchk = insp["mapping"]
    rtable = rang["table"]
    agg = rang["aggregate"]
    lsum = loc["stats"].set_index("component")
    diag = loc["diag"]
    q = qual["summary"]

    rep = insp["summary"]
    lines = [
        "# 基于动捕真值的 UWB 四基站测距与三维定位精度评估",
        "",
        f"- 数据集：`{cfg['project'].get('bag', '')}`",
        f"- 配置：`{Path(cfg['_config_path']).name}`",
        f"- 生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}",
        "",
        "> 简化模型：忽略 TAG 安装偏移；无人机与平台姿态一致；平台静止；",
        "> UWB→MAVROS 使用已知轴映射 `x_M=y_U, y_M=-x_U, z_M=-z_U`；",
        "> 全流程不做任何配准、尺度修正或姿态标定。",
        "",
        "## 1. 数据与几何检查",
        "",
        f"- UWB：{rep['n_uwb']} 帧，{rep['duration_s']:.2f} s，"
        f"{rep['uwb_rate_hz']:.2f} Hz；动捕 {rep['n_uav']} 帧，{rep['uav_rate_hz']:.1f} Hz",
        f"- 平台中心（世界系中位数）：({ds.p0[0]:.4f}, {ds.p0[1]:.4f}, {ds.p0[2]:.4f}) m，"
        f"静态标准差 {np.round(ds.p0_std, 5).tolist()} m",
        f"- 落地后 5 s 无人机与平台中心距离：**{insp['summary']['landing_offset_m']:.3f} m** "
        f"→ {'两路动捕处于同一世界坐标系' if insp['summary']['same_world_frame'] else '坐标系可能不一致'}",
        f"- 相邻基站间距：{np.round(geo['adjacent_distances_m'], 4).tolist()} m；"
        f"对角间距：{np.round(geo['diagonal_distances_m'], 4).tolist()} m；"
        f"几何自检：{'通过' if geo['ok'] else '未通过'}",
        f"- `dis_arr` 对应关系：单位映射 MAE {mapchk['identity_mae_m']:.4f} m，"
        f"最优排列 {mapchk['best_permutation']} MAE {mapchk['best_mae_m']:.4f} m"
        f"（差异 {mapchk['improvement_m']*1000:.1f} mm）",
        "",
        "## 2. 四基站测距精度",
        "",
        md_table(rtable[["anchor", "n_valid", "valid_ratio", "bias_m", "mae_m", "rmse_m",
                         "std_m", "max_abs_m", "p95_abs_m", "n_jumps"]], "{:.3f}"),
        "",
        f"四站合并：有效测距 {agg['n_valid']} 条，RMSE **{agg['rmse_m']:.3f} m**，"
        f"Bias {agg['bias_m']:+.3f} m，P95 {agg['p95_abs_m']:.3f} m，"
        f"最大 {agg['max_abs_m']:.3f} m。",
        "",
        "## 3. 三维定位精度",
        "",
        md_table(loc["stats"], "{:.3f}"),
        "",
        "### 尺度诊断（仅诊断，未用于修正）",
        "",
        md_table(diag[["axis", "slope_k", "intercept_b_m", "r2", "corr",
                       "implied_scale_1_over_k"]], "{:.3f}"),
        "",
        "UWB 位置输出的 X 轴与真值高度线性相关，但斜率约 0.38（即被放大约 2.6 倍），",
        "z 轴斜率约 1.0。该比例失配是**数据本身**的系统性问题，报告中如实给出，",
        "未对数据做任何缩放或平移；定位 RMSE 因此以米级为主，来自这一比例失配，",
        "并不代表随机噪声水平。",
        "",
        "## 4. 原始数据质量",
        "",
        f"- 采样：{q['sampling']['rate_hz']:.2f} Hz，Δt 中位 {q['sampling']['dt_median_s']*1e3:.1f} ms，"
        f"最大 {q['sampling']['dt_max_s']*1e3:.1f} ms；超过阈值的间隙 {q['gaps']['count']} 个",
        f"- 位置跳变（>{q['position_jumps']['threshold_m']} m/帧）：{q['position_jumps']['count']} 次",
        f"- 位置长时间固定段：{q['stuck']['segments']} 段；Z 轴固定值占比 {q['z']['stuck_ratio']*100:.1f}%",
        f"- EOP：均值 {q['eop']['mean']}，零值比例 {q['eop']['zero_ratio']*100:.1f}%，"
        f"不同取值组合 {q['eop']['n_unique_rows']} 种",
        f"- Z 轴误差：Bias {q['z']['z_bias_m']:+.3f} m，RMSE {q['z']['z_rmse_m']:.3f} m，"
        f"|e_z|>0.5 m 占 {q['z']['frac_abs_gt_0p5']*100:.1f}%",
        "",
        "> EOP 字段的物理含义未能从数据或文档中确认，本文只把它作为协议字段做描述性统计，",
        "> 未将其等同于真实定位误差。",
        "",
        "## 5. 测距与定位误差联合分析",
        "",
        f"- 几何 DOP（沿真实轨迹）：HDOP 中位 {joint['dop']['hdop'].median():.2f}，"
        f"VDOP 中位 {joint['dop']['vdop'].median():.2f}，GDOP 中位 {joint['dop']['gdop'].median():.2f}",
        f"- 四基站共面（z 展布 {geo['planarity_z_spread_m']:.1e} m），垂直方向几何条件差，"
        f"VDOP 显著大于 HDOP，这是高度误差偏大的几何原因。",
        "",
        "### DOP 校验（详表见 `tables/dop_verification.csv`，图 J6）",
        "",
        f"- 起飞前（水平距阵列中心 1.6 m、相对高度 0.06 m）：cond$(H^TH)$ ≈ "
        f"{dopchk['selection'].iloc[0]['cond_HtH']:.1e}，VDOP = "
        f"{dopchk['selection'].iloc[0]['vdop']:.1f}；",
        f"- 飞到阵列正上方（相对高度 0.70 m）：VDOP = "
        f"{dopchk['selection']['vdop'].min():.2f}。",
        "- 图 (b) 的两条分支即对应这两类空间位置：远端贴地（VDOP≈50）与阵列上方（VDOP<1），",
        "  与 (a) 中轨迹的 XY 位置逐点对应，不是坐标或同步错误。",
        "- 蒙特卡洛（σ_r=0.05 m）显示：良态点上 VDOP×σ 与实际垂直散布一致；",
        "  病态点（cond≈1e4）线性 VDOP 会高估（2.6 m vs 0.28 m），故 VDOP 只作几何相对指标，",
        "  不直接乘 σ 当作绝对误差。",
        "",
        md_table(joint["agg"], "{:.3f}"),
        "",
        "相关性只描述统计关联，不构成因果结论；定位误差被尺度失配主导，",
        "因此测距误差与定位误差的相关性会明显弱于“纯噪声”情形。",
        "",
        "## 6. 结论",
        "",
        f"1. 测距链路工作正常：四站有效率 100%，合并 RMSE {agg['rmse_m']:.3f} m，"
        f"误差以系统偏置为主（{agg['bias_m']:+.3f} m）。",
        f"2. UWB 定位输出存在系统性比例失配（X 轴约 {diag.set_index('axis').loc['x','implied_scale_1_over_k']:.2f} 倍），"
        "导致直接比较的定位 RMSE 达米级；该问题需要在 UWB 配置/固件侧修正，而非后处理。",
        "3. 四基站共面导致 VDOP 偏大，高度定位在接近平台平面时几何条件最差。",
        "4. 测距误差与定位误差的相关性较弱且方向不一致，说明定位误差主要不是由测距噪声驱动。",
        "",
        "## 7. 局限与待确认问题",
        "",
        "1. `dis_arr[i]` 与 A0–A3 的物理对应关系未从 LinkTrack 配置中核实，",
        "   当前按单位映射处理（与最优排列相差约 7 mm）。",
        "2. EOP 字段含义待向厂商确认。",
        "3. 基站天线相位中心与平台角点的重合、TAG 安装偏移按任务假设忽略。",
        "4. 平台与无人机姿态一致、平台静止按任务假设成立（平台位置标准差为 0）。",
        "",
        "## 8. 图片索引",
        "",
        "### 测距（R1–R9）",
        "",
    ]
    lines += _figure_links("ranging", [
        "R1_ranges_vs_truth", "R2_range_errors_time", "R3_rmse_bar", "R4_error_box",
        "R5_error_cdf", "R6_error_vs_distance", "R7_layout_2d", "R8_layout_3d",
        "R9_anomaly_validity"], results)
    lines += ["", "### 定位（P1–P12）", ""]
    lines += _figure_links("localization", [
        "P1_uwb_raw_xyz", "P2_uwb_mavros_xyz", "P3_uwb_xy", "P4_uwb_3d",
        "P5_compare_xyz", "P6_compare_xy", "P7_compare_3d", "P8_axis_errors",
        "P9_xy_3d_errors", "P10_rmse_bar", "P11_error_cdf", "P12_error_box"], results)
    lines += ["", "### 质量（Q1–Q4）", ""]
    lines += _figure_links("quality", ["Q1_sampling", "Q2_position_jumps",
                                       "Q3_height_timeline", "Q4_eop_and_stuck"], results)
    lines += ["", "### 联合（J1–J7）", ""]
    lines += _figure_links("joint", ["J1_ranging_vs_localization_time", "J2_range_vs_3d",
                                     "J3_range_vs_z", "J4_correlation_heatmap",
                                     "J5_geometry_dop", "J6_dop_validation",
                                     "J7_vdop_vs_z"], results)
    lines += ["", "---", "",
              "所有统计结果见 `results/tables/`，逐帧数据见 `results/processed_csv/`。"]
    return "\n".join(lines)


def run(cfg: dict) -> dict:
    results = Path(cfg["_root"]) / cfg["output"]["results_dir"]
    results.mkdir(parents=True, exist_ok=True)
    (results / "figures").mkdir(exist_ok=True)
    (results / "tables").mkdir(exist_ok=True)
    (results / "processed_csv").mkdir(exist_ok=True)
    (results / "reports").mkdir(exist_ok=True)

    print("== 1/6 数据检查 ==")
    insp = inspect_data.run(cfg, results)
    ds = insp["dataset"]

    print("== 2/6 测距评估 ==")
    rang = evaluate_ranging.run(cfg, results, ds)

    print("== 3/6 定位评估 ==")
    loc = evaluate_localization.run(cfg, results, ds)

    print("== 4/6 数据质量 ==")
    qual = analyze_uwb_quality.run(cfg, results, ds)

    print("== 5/6 联合分析 ==")
    joint = analyze_joint_errors.run(cfg, results, ds)

    print("== 6/6 DOP 校验 ==")
    dopchk = verify_dop.run(cfg, results, ds)

    report = build_report(cfg, ds, insp, rang, loc, qual, joint, dopchk, results)
    report_path = results / "reports" / "analysis_report.md"
    report_path.write_text(report, encoding="utf-8")

    summary = {
        "dataset": {"n_uwb": ds.n, "duration_s": float(ds.t[-1] - ds.t[0])},
        "geometry": insp["geometry"],
        "mapping": insp["mapping"],
        "inspection": {
            "landing_offset_m": float(insp["summary"]["landing_offset_m"]),
            "platform_std_max_m": float(max(insp["summary"]["platform_std_m"])),
            "same_world_frame": bool(insp["summary"]["same_world_frame"]),
        },
        "ranging": rang["aggregate"],
        "localization": loc["stats"].set_index("component").round(4).to_dict(orient="index"),
        "scale_diagnostic": loc["diag"].set_index("axis").round(4).to_dict(orient="index"),
        "quality": qual["summary"],
        "dop": {"hdop_median": float(joint["dop"]["hdop"].median()),
                "vdop_median": float(joint["dop"]["vdop"].median()),
                "gdop_median": float(joint["dop"]["gdop"].median())},
        "dop_verification": {
            "cond_HtH_start": float(dopchk["selection"].iloc[0]["cond_HtH"]),
            "vdop_start": float(dopchk["selection"].iloc[0]["vdop"]),
            "vdop_min": float(dopchk["selection"]["vdop"].min()),
            "mc_std_z_start": float(dopchk["selection"].iloc[0]["mc_std_z"]),
            "linear_pred_start": float(dopchk["selection"].iloc[0]["vdop_x_sigma"]),
        },
        "report": str(report_path),
    }
    (results / "reports" / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n报告：{report_path}")
    return summary


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

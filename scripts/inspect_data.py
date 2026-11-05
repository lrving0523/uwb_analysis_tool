"""第一步：检查 CSV 列名、单位、时间戳、采样率、基站几何与坐标一致性。"""

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

from src import anchor_geometry as geom
from src import data_loader as dl
from src import plot_style as ps
from src.console import enable_utf8
from src.ground_truth_ranges import mapping_check, true_ranges
from src.time_alignment import overlap_window, sampling_stats


def run(cfg: dict, results: Path | None = None) -> dict:
    results = Path(results or (Path(cfg["_root"]) / cfg["output"]["results_dir"]))
    tables = results / "tables"
    reports = results / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    ds = dl.build_dataset(cfg)
    geo = geom.check_geometry(ds.anchors_world,
                              expected_side=cfg["geometry"].get("expected_adjacent_m", 0.8),
                              expected_diagonal=cfg["geometry"].get("expected_diagonal_m", 0.8 * np.sqrt(2)))
    mapchk = mapping_check(ds.ranges_meas, ds.ranges_true, ds.valid_range)

    # 动捕一致性：落地后无人机应位于平台中心附近
    tail = ds.uav[ds.uav["t"] >= ds.uav["t"].max() - 5.0]
    landing_offset = float(np.linalg.norm(tail[["x", "y", "z"]].mean().to_numpy() - ds.p0))

    summary = dl.dataset_summary(ds)
    summary.update({
        "geometry": geo,
        "mapping_check": mapchk,
        "landing_offset_m": landing_offset,
        "platform_static_std_m": ds.p0_std.round(5).tolist(),
        "same_world_frame": bool(landing_offset < 0.30 and np.max(ds.p0_std) < 0.01),
    })

    pd.DataFrame([{
        "n_uwb": summary["n_uwb"],
        "n_uav": summary["n_uav"],
        "duration_s": round(summary["duration_s"], 3),
        "uwb_rate_hz": round(summary["uwb_rate_hz"], 3),
        "uav_rate_hz": round(summary["uav_rate_hz"], 3),
        "platform_center_x": ds.p0[0],
        "platform_center_y": ds.p0[1],
        "platform_center_z": ds.p0[2],
        "landing_offset_m": landing_offset,
        "valid_time_ratio": summary["valid_time_ratio"],
    }]).to_csv(tables / "data_summary.csv", index=False, float_format="%.6f")

    geom.anchor_table(ds.anchors_world, ds.anchors_uwb).to_csv(
        tables / "anchor_positions.csv", index=False, float_format="%.6f")

    (tables / "geometry_check.json").write_text(
        json.dumps(geo, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Markdown 报告 ----
    if mapchk["identity_is_best"]:
        mapping_conclusion = "单位映射即最优：dis_arr[i] ↔ A_i"
    elif mapchk["improvement_m"] < 0.05:
        mapping_conclusion = (
            f"单位映射与最优排列仅差 {mapchk['improvement_m']*1000:.1f} mm，可按 dis_arr[i] ↔ A_i 处理；"
            "严格对应需核对 LinkTrack 基站 ID 配置")
    else:
        mapping_conclusion = "存在明显更优排列，必须核对 LinkTrack 基站 ID 配置"

    uav_rate = sampling_stats(ds.uav["t"].to_numpy(float))
    uwb_rate = sampling_stats(ds.t)
    lines = [
        "# 数据检查报告",
        "",
        f"- 配置：`{cfg['_config_path']}`",
        f"- UWB 样本：{summary['n_uwb']}，时长 {summary['duration_s']:.2f} s，"
        f"采样率 {uwb_rate['rate_hz']:.2f} Hz（中位 Δt {uwb_rate['dt_median_s']*1e3:.1f} ms）",
        f"- 动捕真值：{summary['n_uav']} 样本，{uav_rate['rate_hz']:.1f} Hz",
        f"- 平台中心（世界系中位数）：({ds.p0[0]:.4f}, {ds.p0[1]:.4f}, {ds.p0[2]:.4f}) m，"
        f"静态标准差 {np.round(ds.p0_std, 5).tolist()} m",
        f"- 落地后 5 s 内无人机均值与平台中心距离：**{landing_offset:.3f} m** → "
        f"{'同一世界坐标系' if summary['same_world_frame'] else '坐标系可能不一致，需报告'}",
        "",
        "## 基站几何",
        "",
        f"- 相邻间距：{np.round(geo['adjacent_distances_m'], 4).tolist()} m（期望 0.8）",
        f"- 对角间距：{np.round(geo['diagonal_distances_m'], 4).tolist()} m（期望 1.1314）",
        f"- 共面性 z 展布：{geo['planarity_z_spread_m']:.2e} m",
        f"- 俯视方向：{'顺时针' if geo['clockwise_xy'] else '逆时针'}（有向面积 {geo['signed_area_xy_m2']:.3f} m²）",
        f"- 几何自检：{'通过' if geo['ok'] else '未通过'}",
        "",
        "## dis_arr 与 A0–A3 对应关系",
        "",
        f"- 单位映射 MAE：{mapchk['identity_mae_m']:.4f} m",
        f"- 最优排列：{mapchk['best_permutation']}，MAE：{mapchk['best_mae_m']:.4f} m",
        f"- 结论：{mapping_conclusion}",
        "",
        "## 时间同步",
        "",
        f"- 公共时间窗：{max(ds.uav['t'].min(), ds.uwb['t'].min()):.3f} – "
        f"{min(ds.uav['t'].max(), ds.uwb['t'].max()):.3f} s（bag 时间）",
        f"- 以 UWB 时间戳为基准，动捕真值插值；最大插值间隙 "
        f"{cfg.get('time', {}).get('max_gap_s', 0.5)} s",
        f"- 插值有效比例：{summary['valid_time_ratio']*100:.2f}%",
    ]
    (reports / "data_inspection.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    return {"dataset": ds, "summary": summary, "geometry": geo, "mapping": mapchk}


def main() -> int:
    enable_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    args = ap.parse_args()
    cfg = dl.load_config(args.config)
    ps.use_ieee_style()
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

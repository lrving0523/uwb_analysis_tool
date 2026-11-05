"""第四–六步：四基站测距误差计算、统计与绘图。"""

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
from src import plots_ranging as pr
from src.anchor_geometry import ANCHOR_NAMES, anchor_table
from src.console import enable_utf8
from src.ground_truth_ranges import ranges_table
from src.ranging_metrics import (aggregate_metrics, jump_indices, metrics_table,
                                 range_errors)


def run(cfg: dict, results: Path | None = None, ds=None) -> dict:
    results = Path(results or (Path(cfg["_root"]) / cfg["output"]["results_dir"]))
    fig_dir = results / "figures" / "ranging"
    tab_dir = results / "tables"
    proc_dir = results / "processed_csv"
    for d in (fig_dir, tab_dir, proc_dir):
        d.mkdir(parents=True, exist_ok=True)

    ds = ds or dl.build_dataset(cfg)
    jump_thr = float(cfg.get("quality", {}).get("range_jump_m", 0.5))
    pct = float(cfg.get("metrics", {}).get("percentile", 95.0))

    err = range_errors(ds.ranges_meas, ds.ranges_true)
    valid = ds.valid_range & np.isfinite(err)
    jumps = jump_indices(ds.ranges_meas, ds.valid_range, jump_thr)

    table = metrics_table(err, valid, jumps, percentile=pct)
    agg = aggregate_metrics(err, valid)

    # ---- CSV ----
    truth_wide = pd.DataFrame({"t": ds.t})
    meas_wide = pd.DataFrame({"t": ds.t})
    err_wide = pd.DataFrame({"t": ds.t, "t_rel": ds.t_rel})
    for i, name in enumerate(ANCHOR_NAMES):
        truth_wide[name] = ds.ranges_true[:, i]
        meas_wide[name] = ds.ranges_meas[:, i]
        err_wide[name] = err[:, i]
    truth_wide.to_csv(proc_dir / "ranges_truth.csv", index=False, float_format="%.6f")
    meas_wide.to_csv(proc_dir / "ranges_measured.csv", index=False, float_format="%.6f")
    err_wide.to_csv(proc_dir / "ranges_error.csv", index=False, float_format="%.6f")
    table.to_csv(tab_dir / "ranging_metrics.csv", index=False, float_format="%.6f")
    anchor_table(ds.anchors_world, ds.anchors_uwb).to_csv(
        tab_dir / "anchor_positions.csv", index=False, float_format="%.6f")
    (tab_dir / "ranging_summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

    long = ranges_table(ds.t, ds.ranges_true, ds.ranges_meas, ds.valid_range)
    long.to_csv(proc_dir / "ranges_long.csv", index=False, float_format="%.6f")

    # ---- 图 ----
    files = pr.make_all(ds, err, table, jumps, fig_dir)

    print(f"[ranging] 有效测距 {agg['n_valid']} 条，RMSE {agg['rmse_m']:.3f} m，"
          f"Bias {agg['bias_m']:+.3f} m，P95 {agg['p95_abs_m']:.3f} m")
    for _, row in table.iterrows():
        print(f"  {row['anchor']}: RMSE {row['rmse_m']:.3f}  MAE {row['mae_m']:.3f}  "
              f"bias {row['bias_m']:+.3f}  p95 {row[f'p{int(pct)}_abs_m']:.3f}  "
              f"valid {row['valid_ratio']*100:.1f}%  jumps {int(row['n_jumps'])}")

    return {"ds": ds, "err": err, "valid": valid, "jumps": jumps, "table": table,
            "aggregate": agg, "files": files}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """把 DataFrame 转成 Markdown 表（不依赖 tabulate）。"""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


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

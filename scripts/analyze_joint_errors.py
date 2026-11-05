"""第九步之二：测距误差与定位误差联合分析（相关性 + 几何 DOP）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src import data_loader as dl
from src import joint_analysis as ja
from src import plot_style as ps
from src import plots_joint as pj
from src.console import enable_utf8
from src.ranging_metrics import range_errors


def run(cfg: dict, results: Path | None = None, ds=None) -> dict:
    results = Path(results or (Path(cfg["_root"]) / cfg["output"]["results_dir"]))
    fig_dir = results / "figures" / "joint"
    tab_dir = results / "tables"
    proc_dir = results / "processed_csv"
    for d in (fig_dir, tab_dir, proc_dir):
        d.mkdir(parents=True, exist_ok=True)

    ds = ds or dl.build_dataset(cfg)
    range_err = range_errors(ds.ranges_meas, ds.ranges_true)

    joint = ja.joint_table(ds, range_err)
    corr = ja.correlation_table(joint, method="pearson")
    corr_spearman = ja.correlation_table(joint, method="spearman")
    agg = ja.aggregate_correlations(joint)
    dop_df = ja.dop(ds.truth_abs, ds.anchors_world)

    joint.to_csv(proc_dir / "joint_table.csv", index=False, float_format="%.6f")
    corr.to_csv(tab_dir / "joint_correlations_pearson.csv", index=False, float_format="%.6f")
    corr_spearman.to_csv(tab_dir / "joint_correlations_spearman.csv", index=False,
                         float_format="%.6f")
    agg.to_csv(tab_dir / "joint_correlations_aggregate.csv", index=False, float_format="%.6f")
    dop_df.to_csv(proc_dir / "dop.csv", index=False, float_format="%.6f")
    ja.dop_summary(ds).to_csv(tab_dir / "dop_summary.csv", index=False, float_format="%.6f")

    files = pj.make_all(ds, joint, corr, dop_df, fig_dir)

    print(f"[joint] HDOP median {dop_df['hdop'].median():.2f}, "
          f"VDOP median {dop_df['vdop'].median():.2f}, "
          f"GDOP median {dop_df['gdop'].median():.2f}")
    for _, row in agg.iterrows():
        print(f"[joint] corr(range RMS, {row['component']}) = {row['r']:+.3f} "
              f"(p={row['p']:.3g}, n={int(row['n'])})")

    return {"ds": ds, "joint": joint, "corr": corr, "corr_spearman": corr_spearman,
            "agg": agg, "dop": dop_df, "files": files}


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

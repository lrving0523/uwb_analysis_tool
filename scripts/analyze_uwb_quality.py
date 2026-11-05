"""第九步之一：UWB 原始数据质量分析（采样、跳变、固定值、EOP、Z 轴异常）。"""

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
from src import plot_style as ps
from src import plots_quality as pq
from src.console import enable_utf8
from src.localization_metrics import position_errors
from src.uwb_quality import (gap_table, position_jumps, quality_summary,
                             sampling_quality, stuck_segments)


def run(cfg: dict, results: Path | None = None, ds=None) -> dict:
    results = Path(results or (Path(cfg["_root"]) / cfg["output"]["results_dir"]))
    fig_dir = results / "figures" / "quality"
    tab_dir = results / "tables"
    for d in (fig_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)

    ds = ds or dl.build_dataset(cfg)
    qcfg = cfg.get("quality", {})

    gaps = gap_table(ds.t, float(qcfg.get("gap_threshold_s", 0.5)))
    jumps = position_jumps(ds.uwb_mav, ds.t, float(qcfg.get("pos_jump_m", 0.3)))
    stuck_pos = stuck_segments(ds.uwb_mav, int(qcfg.get("stuck_min_run", 10)))
    stuck_z = stuck_segments(ds.uwb_mav[:, [2]], int(qcfg.get("stuck_min_run", 10)))
    q = quality_summary(ds)
    err = position_errors(ds.uwb_mav, ds.truth_rel)

    gaps.to_csv(tab_dir / "quality_gaps.csv", index=False, float_format="%.6f")
    jumps.to_csv(tab_dir / "quality_jumps.csv", index=False, float_format="%.6f")
    (tab_dir / "quality_summary.json").write_text(
        json.dumps(q, indent=2, ensure_ascii=False), encoding="utf-8")

    files = pq.make_all(ds, err, gaps, stuck_z, jumps, fig_dir)

    print(f"[quality] rate {q['sampling']['rate_hz']:.2f} Hz, "
          f"gaps {len(gaps)}, jumps {len(jumps)}, stuck(pos) {len(stuck_pos)}, "
          f"stuck(z) {len(stuck_z)}")
    print(f"[quality] EOP zero ratio {q['eop']['zero_ratio']*100:.1f}%, "
          f"unique rows {q['eop']['n_unique_rows']}")
    print(f"[quality] z: bias {q['z']['z_bias_m']:+.3f} m, RMSE {q['z']['z_rmse_m']:.3f} m, "
          f"|ez|>0.5 m {q['z']['frac_abs_gt_0p5']*100:.1f}%, stuck {q['z']['stuck_ratio']*100:.1f}%")

    return {"ds": ds, "summary": q, "gaps": gaps, "jumps": jumps,
            "stuck": stuck_pos, "stuck_z": stuck_z, "files": files}


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

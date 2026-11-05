"""第七–八步：UWB 三维定位精度评估与绘图（不做任何配准/尺度修正）。"""

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
from src import plots_localization as pl
from src.console import enable_utf8
from src.localization_metrics import position_errors, scale_diagnostic, summary


def run(cfg: dict, results: Path | None = None, ds=None) -> dict:
    results = Path(results or (Path(cfg["_root"]) / cfg["output"]["results_dir"]))
    fig_dir = results / "figures" / "localization"
    tab_dir = results / "tables"
    proc_dir = results / "processed_csv"
    for d in (fig_dir, tab_dir, proc_dir):
        d.mkdir(parents=True, exist_ok=True)

    ds = ds or dl.build_dataset(cfg)
    pct = float(cfg.get("metrics", {}).get("percentile", 95.0))

    err = position_errors(ds.uwb_mav, ds.truth_rel)
    stats = summary(err, percentile=pct)
    diag = scale_diagnostic(ds.uwb_mav, ds.truth_rel)

    # ---- CSV ----
    out = pd.DataFrame({"t": ds.t, "t_rel": ds.t_rel})
    for i, ax in enumerate(("x", "y", "z")):
        out[f"uwb_{ax}"] = ds.uwb_mav[:, i]
        out[f"truth_rel_{ax}"] = ds.truth_rel[:, i]
    for col in err.columns:
        out[col] = err[col].to_numpy()
    out.to_csv(proc_dir / "pose_compare.csv", index=False, float_format="%.6f")
    err.to_csv(proc_dir / "localization_errors.csv", index=False, float_format="%.6f")
    pd.DataFrame({"t": ds.t, "x": ds.uwb_mav[:, 0], "y": ds.uwb_mav[:, 1],
                  "z": ds.uwb_mav[:, 2]}).to_csv(
        proc_dir / "uwb_mavros_position.csv", index=False, float_format="%.6f")
    stats.to_csv(tab_dir / "localization_metrics.csv", index=False, float_format="%.6f")
    diag.to_csv(tab_dir / "localization_scale_diagnostic.csv", index=False,
                float_format="%.6f")
    (tab_dir / "localization_summary.json").write_text(
        json.dumps(stats.set_index("component").round(4).to_dict(orient="index"),
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 图 ----
    files = pl.make_all(ds, err, stats, fig_dir)

    key = stats.set_index("component")
    print(f"[localization] RMSE: x {key.loc['x','rmse_m']:.3f}  y {key.loc['y','rmse_m']:.3f}  "
          f"z {key.loc['z','rmse_m']:.3f}  xy {key.loc['xy','rmse_m']:.3f}  "
          f"3d {key.loc['3d','rmse_m']:.3f} m")
    print("[localization] 尺度诊断（仅诊断，未用于修正）：")
    for _, row in diag.iterrows():
        print(f"  {row['axis']}: slope {row['slope_k']:.3f} (1/k {row['implied_scale_1_over_k']:.2f}), "
              f"r2 {row['r2']:.3f}, corr {row['corr']:.3f}")

    return {"ds": ds, "err": err, "stats": stats, "diag": diag, "files": files}


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

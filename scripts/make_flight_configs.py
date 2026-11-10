"""为一批飞行数据生成独立配置（每架次一个 YAML）。

数据目录约定：``<data-dir>/<flight_id>/`` 下存放标准文件名的 5 个 CSV::

    nlink_linktrack_tagframe0.csv
    mavros__vision_pose__pose.csv
    natnet_ros__ljp__pose.csv
    mavros__local_position__pose.csv
    mavros__setpoint_position__local.csv

用法::

    python scripts/make_flight_configs.py \
        --data-dir data/vision_csv \
        --config-dir configs/vision_csv \
        --results-dir results/vision_csv \
        --name vision_csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

FILES = {
    "uwb": "nlink_linktrack_tagframe0.csv",
    "uav": "mavros__vision_pose__pose.csv",
    "platform": "natnet_ros__ljp__pose.csv",
    "local": "mavros__local_position__pose.csv",
    "setpoint": "mavros__setpoint_position__local.csv",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="批量生成飞行配置")
    ap.add_argument("--data-dir", required=True, help="包含 <flight_id>/ 子目录的数据目录")
    ap.add_argument("--config-dir", required=True, help="配置输出目录")
    ap.add_argument("--results-dir", required=True, help="结果输出目录")
    ap.add_argument("--name", default="flight", help="项目名称（写进报告）")
    ap.add_argument("--ids", nargs="*", default=None, help="指定架次；默认取全部子目录")
    args = ap.parse_args()

    base = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    out_dir = ROOT / args.config_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = args.ids or sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    if not ids:
        raise SystemExit(f"{data_dir} 下没有子目录")

    for fid in ids:
        cfg = {
            "project": {
                "name": args.name,
                "bag": f"{args.name}_{fid}",
                "description": "UWB 测距与三维定位精度评估",
                "notes": base["project"]["notes"],
            },
            "data": {
                "raw_dir": f"{data_dir.relative_to(ROOT).as_posix()}/{fid}",
                "files": dict(FILES),
            },
            "geometry": base["geometry"],
            "time": base["time"],
            "quality": base["quality"],
            "metrics": base["metrics"],
            "plots": base["plots"],
            "output": {"results_dir": f"{args.results_dir.rstrip('/')}/{fid}"},
        }
        path = out_dir / f"{fid}.yaml"
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
        print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

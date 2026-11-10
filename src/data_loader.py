"""数据读取、配置加载与统一数据集构建。

所有时间同步优先使用 ``bag_time_sec``（秒），不使用行号匹配。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import anchor_geometry as geom
from src.coordinate_transform import uwb_to_mavros
from src.time_alignment import interp_with_validity, overlap_window, sampling_stats

SIM_TIME_CANDIDATES = (
    "bag_time_sec",
    "rosbag_time_sec",
    "bag_time_s",
    "relative_time_sec",
    "time_from_segment_start_s",
    "t_rel_bag_start_s",
    "timestamp",
    "time",
)


def _t_col(df: pd.DataFrame) -> str:
    for c in SIM_TIME_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f"找不到时间列，已有列：{list(df.columns)[:10]}")


def _clean(df: pd.DataFrame, t_col: str) -> pd.DataFrame:
    """按时间排序、去重、去 NaN 时间。"""
    out = df.copy()
    out = out[np.isfinite(out[t_col])]
    out = out.sort_values(t_col, kind="mergesort")
    out = out.drop_duplicates(t_col, keep="last")
    return out.reset_index(drop=True)


def load_config(path: str | Path) -> dict:
    """读取 YAML 配置；工程根目录通过向上查找 ``configs/`` 目录确定。

    支持配置放在 ``configs/`` 子目录中（例如 ``configs/flight_takeoff_landing/*.yaml``）。
    """
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = path.parent
    while root != root.parent and not (root / "configs").is_dir():
        root = root.parent
    if not (root / "configs").is_dir():  # 兜底：退回到 configs 的上一级
        root = path.parents[1]

    cfg["_root"] = str(root)
    cfg["_config_path"] = str(path)
    return cfg


def load_pose(path: str | Path) -> pd.DataFrame:
    """读取 MAVROS 位姿 CSV（vision_pose / local_position / setpoint）。"""
    df = pd.read_csv(path)
    tcol = _t_col(df)
    out = pd.DataFrame({
        "t": df[tcol].to_numpy(float),
        "x": df["pose.position.x"].to_numpy(float),
        "y": df["pose.position.y"].to_numpy(float),
        "z": df["pose.position.z"].to_numpy(float),
    })
    for axis in ("x", "y", "z", "w"):
        col = f"pose.orientation.{axis}"
        out[f"q{axis}"] = df[col].to_numpy(float) if col in df.columns else np.nan
    out = _clean(out, "t")
    out = out.dropna(subset=["x", "y", "z"]).reset_index(drop=True)
    return out


def load_platform(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """读取降落平台中心位姿；返回 (df, 中位中心, 逐轴标准差)。"""
    df = load_pose(path)
    center = df[["x", "y", "z"]].median().to_numpy(float)
    std = df[["x", "y", "z"]].std(ddof=0).to_numpy(float)
    return df, center, std


def load_uwb(path: str | Path) -> pd.DataFrame:
    """读取 Nooploop LinkTrack tagframe0 CSV。"""
    df = pd.read_csv(path)
    tcol = _t_col(df)
    out = pd.DataFrame({"t": df[tcol].to_numpy(float)})

    for i in range(3):
        out[f"u{i}"] = df[f"pos_3d[{i}]"].to_numpy(float)
        out[f"eop{i}"] = df[f"eop_3d[{i}]"].to_numpy(float)
        out[f"vel{i}"] = df[f"vel_3d[{i}]"].to_numpy(float)

    for i in range(8):
        col = f"dis_arr[{i}]"
        out[f"d{i}"] = df[col].to_numpy(float) if col in df.columns else np.nan

    out = _clean(out, "t")
    return out


def load_local_position(path: str | Path) -> pd.DataFrame:
    return load_pose(path)


@dataclass
class Dataset:
    """对齐后的分析数据集（时间基准：UWB 的 bag_time_sec）。"""

    cfg: dict
    uav: pd.DataFrame
    platform: pd.DataFrame
    uwb: pd.DataFrame
    local: pd.DataFrame | None

    p0: np.ndarray
    p0_std: np.ndarray
    anchors_uwb: np.ndarray
    anchors_rel: np.ndarray
    anchors_world: np.ndarray

    t: np.ndarray                  # 对齐后的 UWB 时间（bag_time_sec）
    t_rel: np.ndarray              # 相对第一帧的时间
    truth_abs: np.ndarray          # (N,3) 无人机世界坐标
    truth_rel: np.ndarray          # (N,3) 相对平台中心
    uwb_uwb: np.ndarray            # (N,3) 原始 UWB 坐标
    uwb_mav: np.ndarray            # (N,3) 转换到 MAVROS 轴
    eop: np.ndarray                # (N,3)
    ranges_meas: np.ndarray        # (N,4)
    ranges_true: np.ndarray        # (N,4)
    valid_time: np.ndarray         # (N,) 插值有效性
    valid_range: np.ndarray        # (N,4) 测距有效性
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.t)

    def save_processed(self, out_dir: str | Path) -> list[Path]:
        """导出对齐后的核心数据（便于复现）。"""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        base = pd.DataFrame({"t": self.t, "t_rel": self.t_rel})
        for i, name in enumerate(("x", "y", "z")):
            base[f"truth_{name}"] = self.truth_abs[:, i]
            base[f"truth_rel_{name}"] = self.truth_rel[:, i]
            base[f"uwb_raw_{name}"] = self.uwb_uwb[:, i]
            base[f"uwb_mav_{name}"] = self.uwb_mav[:, i]
            base[f"eop_{name}"] = self.eop[:, i]
        for i in range(4):
            base[f"range_meas_A{i}"] = self.ranges_meas[:, i]
            base[f"range_true_A{i}"] = self.ranges_true[:, i]
        base["valid_time"] = self.valid_time
        path = out / "aligned_dataset.csv"
        base.to_csv(path, index=False, float_format="%.6f")
        return [path]


def build_dataset(cfg: dict) -> Dataset:
    """按配置读取三份核心数据并完成时间对齐与坐标变换。"""
    raw = Path(cfg["_root"]) / cfg["data"]["raw_dir"]
    files = cfg["data"]["files"]

    uav = load_pose(raw / files["uav"])
    platform, p0, p0_std = load_platform(raw / files["platform"])
    uwb = load_uwb(raw / files["uwb"])
    local = None
    if files.get("local") and (raw / files["local"]).exists():
        local = load_local_position(raw / files["local"])

    max_gap = float(cfg.get("time", {}).get("max_gap_s", 0.5))
    t0, t1 = overlap_window(uav["t"].to_numpy(), platform["t"].to_numpy(),
                            uwb["t"].to_numpy())

    mask = (uwb["t"] >= t0) & (uwb["t"] <= t1)
    uwb_al = uwb[mask].reset_index(drop=True)
    t = uwb_al["t"].to_numpy(float)

    truth, valid = interp_with_validity(
        uav["t"].to_numpy(float),
        uav[["x", "y", "z"]].to_numpy(float),
        t, max_gap_s=max_gap,
    )

    uwb_uwb = uwb_al[["u0", "u1", "u2"]].to_numpy(float)
    uwb_mav = uwb_to_mavros(uwb_uwb)
    eop = uwb_al[["eop0", "eop1", "eop2"]].to_numpy(float)

    anchors_uwb = geom.anchors_uwb_from_config(cfg)
    anchors_rel = geom.anchors_relative_mavros(anchors_uwb)
    anchors_world = geom.anchors_world(p0, anchors_rel)

    truth_rel = truth - p0
    ranges_true = np.linalg.norm(truth[:, None, :] - anchors_world[None, :, :], axis=2)

    ranges_meas = uwb_al[[f"d{i}" for i in range(4)]].to_numpy(float)
    rmin = float(cfg.get("quality", {}).get("range_min_m", 0.05))
    rmax = float(cfg.get("quality", {}).get("range_max_m", 20.0))
    valid_range = np.isfinite(ranges_meas) & (ranges_meas >= rmin) & (ranges_meas <= rmax)

    meta = {
        "n_uwb": int(len(t)),
        "n_uav": int(len(uav)),
        "n_platform": int(len(platform)),
        "duration_s": float(t[-1] - t[0]),
        "platform_center_m": p0.round(4).tolist(),
        "platform_std_m": p0_std.round(4).tolist(),
        "uwb_rate_hz": float(1.0 / np.median(np.diff(t))),
        "uav_rate_hz": float(1.0 / np.median(np.diff(uav["t"].to_numpy()))),
        "valid_time_ratio": float(valid.mean()),
        "valid_range_ratio": float(valid_range.mean()),
    }

    return Dataset(
        cfg=cfg, uav=uav, platform=platform, uwb=uwb, local=local,
        p0=p0, p0_std=p0_std,
        anchors_uwb=anchors_uwb, anchors_rel=anchors_rel, anchors_world=anchors_world,
        t=t, t_rel=t - t[0],
        truth_abs=truth, truth_rel=truth_rel,
        uwb_uwb=uwb_uwb, uwb_mav=uwb_mav, eop=eop,
        ranges_meas=ranges_meas, ranges_true=ranges_true,
        valid_time=valid, valid_range=valid_range,
        meta=meta,
    )


def dataset_summary(ds: Dataset) -> dict:
    """汇总数据规模/采样率，供 inspect_data 与报告使用。"""
    return {
        **ds.meta,
        "uwb_sampling": sampling_stats(ds.t),
        "uav_sampling": sampling_stats(ds.uav["t"].to_numpy(float)),
        "platform_sampling": sampling_stats(ds.platform["t"].to_numpy(float)),
    }

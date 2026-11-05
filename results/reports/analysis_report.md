# 基于动捕真值的 UWB 四基站测距与三维定位精度评估

- 数据集：`yolo_detection_mono_time_ubuntu_17138_631826629921771954`
- 配置：`config.yaml`
- 生成时间：2026-11-05 10:09:03

> 简化模型：忽略 TAG 安装偏移；无人机与平台姿态一致；平台静止；
> UWB→MAVROS 使用已知轴映射 `x_M=y_U, y_M=-x_U, z_M=-z_U`；
> 全流程不做任何配准、尺度修正或姿态标定。

## 1. 数据与几何检查

- UWB：607 帧，60.60 s，10.00 Hz；动捕 7289 帧，117.3 Hz
- 平台中心（世界系中位数）：(-0.0466, 0.3302, 0.0655) m，静态标准差 [0.0, 0.0, 0.0] m
- 落地后 5 s 无人机与平台中心距离：**0.096 m** → 两路动捕处于同一世界坐标系
- 相邻基站间距：[0.8, 0.8, 0.8, 0.8] m；对角间距：[1.1314, 1.1314] m；几何自检：通过
- `dis_arr` 对应关系：单位映射 MAE 0.1701 m，最优排列 [3, 1, 2, 0] MAE 0.1636 m（差异 6.5 mm）

## 2. 四基站测距精度

| anchor | n_valid | valid_ratio | bias_m | mae_m | rmse_m | std_m | max_abs_m | p95_abs_m | n_jumps |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 607 | 1.000 | -0.023 | 0.164 | 0.260 | 0.259 | 3.581 | 0.464 | 7 |
| A1 | 607 | 1.000 | -0.066 | 0.088 | 0.098 | 0.072 | 0.405 | 0.144 | 0 |
| A2 | 607 | 1.000 | -0.161 | 0.163 | 0.171 | 0.057 | 0.254 | 0.228 | 0 |
| A3 | 607 | 1.000 | -0.254 | 0.265 | 0.277 | 0.109 | 0.426 | 0.374 | 0 |

四站合并：有效测距 2428 条，RMSE **0.214 m**，Bias -0.126 m，P95 0.350 m，最大 3.581 m。

## 3. 三维定位精度

| component | n | rmse_m | mae_m | bias_m | std_m | max_abs_m | p95_abs_m |
|---|---|---|---|---|---|---|---|
| x | 607 | 1.685 | 1.147 | -1.083 | 1.291 | 2.939 | 2.799 |
| y | 607 | 0.303 | 0.260 | -0.257 | 0.160 | 0.918 | 0.604 |
| z | 607 | 0.241 | 0.226 | 0.162 | 0.178 | 0.565 | 0.388 |
| xy | 607 | 1.712 | 1.228 | 1.228 | 1.193 | 2.963 | 2.817 |
| 3d | 607 | 1.729 | 1.285 | 1.285 | 1.158 | 2.975 | 2.829 |

### 尺度诊断（仅诊断，未用于修正）

| axis | slope_k | intercept_b_m | r2 | corr | implied_scale_1_over_k |
|---|---|---|---|---|---|
| x | 0.378 | -0.018 | 0.988 | 0.994 | 2.642 |
| y | 0.350 | 0.151 | 0.380 | 0.617 | 2.861 |
| z | 1.011 | -0.166 | 0.384 | 0.620 | 0.989 |

UWB 位置输出的 X 轴与真值高度线性相关，但斜率约 0.38（即被放大约 2.6 倍），
z 轴斜率约 1.0。该比例失配是**数据本身**的系统性问题，报告中如实给出，
未对数据做任何缩放或平移；定位 RMSE 因此以米级为主，来自这一比例失配，
并不代表随机噪声水平。

## 4. 原始数据质量

- 采样：10.00 Hz，Δt 中位 100.0 ms，最大 101.7 ms；超过阈值的间隙 0 个
- 位置跳变（>0.3 m/帧）：30 次
- 位置长时间固定段：0 段；Z 轴固定值占比 40.7%
- EOP：均值 [0.1695, 0.1698, 1.2308]，零值比例 4.0%，不同取值组合 147 种
- Z 轴误差：Bias +0.162 m，RMSE 0.241 m，|e_z|>0.5 m 占 1.8%

> EOP 字段的物理含义未能从数据或文档中确认，本文只把它作为协议字段做描述性统计，
> 未将其等同于真实定位误差。

## 5. 测距与定位误差联合分析

- 几何 DOP（沿真实轨迹）：HDOP 中位 1.34，VDOP 中位 4.48，GDOP 中位 4.60
- 四基站共面（z 展布 0.0e+00 m），垂直方向几何条件差，VDOP 显著大于 HDOP，这是高度误差偏大的几何原因。

### DOP 校验（详表见 `tables/dop_verification.csv`，图 J6）

- 起飞前（水平距阵列中心 1.6 m、相对高度 0.06 m）：cond$(H^TH)$ ≈ 1.0e+04，VDOP = 52.4；
- 飞到阵列正上方（相对高度 0.70 m）：VDOP = 0.68。
- 图 (b) 的两条分支即对应这两类空间位置：远端贴地（VDOP≈50）与阵列上方（VDOP<1），
  与 (a) 中轨迹的 XY 位置逐点对应，不是坐标或同步错误。
- 蒙特卡洛（σ_r=0.05 m）显示：良态点上 VDOP×σ 与实际垂直散布一致；
  病态点（cond≈1e4）线性 VDOP 会高估（2.6 m vs 0.28 m），故 VDOP 只作几何相对指标，
  不直接乘 σ 当作绝对误差。

| component | r | p | n |
|---|---|---|---|
| ex | 0.452 | 0.000 | 607 |
| ey | -0.178 | 0.000 | 607 |
| ez | 0.123 | 0.002 | 607 |
| e_xy | -0.339 | 0.000 | 607 |
| e_3d | -0.346 | 0.000 | 607 |

相关性只描述统计关联，不构成因果结论；定位误差被尺度失配主导，
因此测距误差与定位误差的相关性会明显弱于“纯噪声”情形。

## 6. 结论

1. 测距链路工作正常：四站有效率 100%，合并 RMSE 0.214 m，误差以系统偏置为主（-0.126 m）。
2. UWB 定位输出存在系统性比例失配（X 轴约 2.64 倍），导致直接比较的定位 RMSE 达米级；该问题需要在 UWB 配置/固件侧修正，而非后处理。
3. 四基站共面导致 VDOP 偏大，高度定位在接近平台平面时几何条件最差。
4. 测距误差与定位误差的相关性较弱且方向不一致，说明定位误差主要不是由测距噪声驱动。

## 7. 局限与待确认问题

1. `dis_arr[i]` 与 A0–A3 的物理对应关系未从 LinkTrack 配置中核实，
   当前按单位映射处理（与最优排列相差约 7 mm）。
2. EOP 字段含义待向厂商确认。
3. 基站天线相位中心与平台角点的重合、TAG 安装偏移按任务假设忽略。
4. 平台与无人机姿态一致、平台静止按任务假设成立（平台位置标准差为 0）。

## 8. 图片索引

### 测距（R1–R9）

![R1_ranges_vs_truth](../figures/ranging/R1_ranges_vs_truth.png)
![R2_range_errors_time](../figures/ranging/R2_range_errors_time.png)
![R3_rmse_bar](../figures/ranging/R3_rmse_bar.png)
![R4_error_box](../figures/ranging/R4_error_box.png)
![R5_error_cdf](../figures/ranging/R5_error_cdf.png)
![R6_error_vs_distance](../figures/ranging/R6_error_vs_distance.png)
![R7_layout_2d](../figures/ranging/R7_layout_2d.png)
![R8_layout_3d](../figures/ranging/R8_layout_3d.png)
![R9_anomaly_validity](../figures/ranging/R9_anomaly_validity.png)

### 定位（P1–P12）

![P1_uwb_raw_xyz](../figures/localization/P1_uwb_raw_xyz.png)
![P2_uwb_mavros_xyz](../figures/localization/P2_uwb_mavros_xyz.png)
![P3_uwb_xy](../figures/localization/P3_uwb_xy.png)
![P4_uwb_3d](../figures/localization/P4_uwb_3d.png)
![P5_compare_xyz](../figures/localization/P5_compare_xyz.png)
![P6_compare_xy](../figures/localization/P6_compare_xy.png)
![P7_compare_3d](../figures/localization/P7_compare_3d.png)
![P8_axis_errors](../figures/localization/P8_axis_errors.png)
![P9_xy_3d_errors](../figures/localization/P9_xy_3d_errors.png)
![P10_rmse_bar](../figures/localization/P10_rmse_bar.png)
![P11_error_cdf](../figures/localization/P11_error_cdf.png)
![P12_error_box](../figures/localization/P12_error_box.png)

### 质量（Q1–Q4）

![Q1_sampling](../figures/quality/Q1_sampling.png)
![Q2_position_jumps](../figures/quality/Q2_position_jumps.png)
![Q3_height_timeline](../figures/quality/Q3_height_timeline.png)
![Q4_eop_and_stuck](../figures/quality/Q4_eop_and_stuck.png)

### 联合（J1–J7）

![J1_ranging_vs_localization_time](../figures/joint/J1_ranging_vs_localization_time.png)
![J2_range_vs_3d](../figures/joint/J2_range_vs_3d.png)
![J3_range_vs_z](../figures/joint/J3_range_vs_z.png)
![J4_correlation_heatmap](../figures/joint/J4_correlation_heatmap.png)
![J5_geometry_dop](../figures/joint/J5_geometry_dop.png)
![J6_dop_validation](../figures/joint/J6_dop_validation.png)
![J7_vdop_vs_z](../figures/joint/J7_vdop_vs_z.png)

---

所有统计结果见 `results/tables/`，逐帧数据见 `results/processed_csv/`。
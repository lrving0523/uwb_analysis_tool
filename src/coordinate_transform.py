"""UWB 坐标系 ↔ MAVROS 坐标系 的已知轴映射。

实验约定（不需要再估计旋转矩阵）：

    x_M =  y_U
    y_M = -x_U
    z_M = -z_U

对应矩阵 A，使 p_M = A · p_U。
"""

from __future__ import annotations

import numpy as np

#: UWB → MAVROS 的已知坐标映射矩阵
UWB_TO_MAVROS = np.array([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
], dtype=float)


def uwb_to_mavros(points: np.ndarray) -> np.ndarray:
    """把 UWB 坐标 (N,3) 或单点 (3,) 转换到 MAVROS 坐标轴约定。"""
    p = np.asarray(points, dtype=float)
    return p @ UWB_TO_MAVROS.T


def mavros_to_uwb(points: np.ndarray) -> np.ndarray:
    """逆变换（A 为正交矩阵，逆等于转置）。"""
    p = np.asarray(points, dtype=float)
    return p @ UWB_TO_MAVROS


def describe() -> str:
    return "x_M = y_U,  y_M = -x_U,  z_M = -z_U"


def check_orthogonality() -> dict:
    """自检：A 是否为反射型正交矩阵（det = -1）。"""
    a = UWB_TO_MAVROS
    eye_err = float(np.max(np.abs(a @ a.T - np.eye(3))))
    return {"max_|A A^T - I|": eye_err, "det(A)": float(np.linalg.det(a))}

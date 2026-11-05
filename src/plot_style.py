"""IEEE / SciencePlots 统一样式与图片导出。

所有图片：Times 字体、8 pt、单栏 3.5 in / 双栏 7.16 in、PNG 300 dpi + PDF 矢量。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SINGLE_COLUMN_IN = 3.5
DOUBLE_COLUMN_IN = 7.16

# Okabe-Ito 色盲友好配色
COLORS = {
    "a0": "#0072B2",
    "a1": "#D55E00",
    "a2": "#009E73",
    "a3": "#CC79A7",
    "truth": "#0072B2",
    "uwb": "#D55E00",
    "error": "#D55E00",
    "aux": "#999999",
    "grid": "#D9D9D9",
}
ANCHOR_COLORS = (COLORS["a0"], COLORS["a1"], COLORS["a2"], COLORS["a3"])
LINE_STYLES = ("-", "--", "-.", ":")


def use_ieee_style(font_size: float = 8.0, dpi: int = 300) -> None:
    """应用 science + ieee + no-latex 样式，并覆盖论文常用细节。"""
    try:
        import scienceplots  # noqa: F401  注册样式

        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:  # pragma: no cover - 没有 SciencePlots 时退化
        plt.style.use("default")

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "black",
        "axes.labelpad": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 1.0,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "axes.grid": False,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.4,
        "grid.linestyle": ":",
        "figure.dpi": 120,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def figsize(width: str | float = "double", aspect: float = 0.62,
            height: float | None = None) -> tuple[float, float]:
    """按单栏/双栏宽度生成 figure 尺寸（英寸）。"""
    if isinstance(width, str):
        w = SINGLE_COLUMN_IN if width == "single" else DOUBLE_COLUMN_IN
    else:
        w = float(width)
    h = float(height) if height is not None else w * aspect
    return (w, h)


def aspect_height(width_in: float, xspan: float, yspan: float,
                  extra: float = 0.8, lo: float = 2.0, hi: float = 4.5) -> float:
    """根据数据长宽比给出合适图高，减少等比例坐标下的空白。"""
    ratio = float(yspan) / max(float(xspan), 1e-9)
    return float(np.clip(width_in * ratio + extra, lo, hi))


def new_figure(nrows: int = 1, ncols: int = 1, width: str | float = "double",
               aspect: float = 0.62, height: float | None = None, **kwargs):
    """创建带 constrained layout 的 figure 与 axes。

    1×1 时返回单个 Axes；多子图时返回按行展开的一维数组。
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize(width, aspect, height),
                             layout="constrained", **kwargs)
    if nrows * ncols == 1:
        return fig, axes
    return fig, np.atleast_1d(axes).ravel()


def panel_label(ax, text: str, dx: float = -0.12, dy: float = 1.03,
                fontsize: float = 9.0) -> None:
    """子图编号 (a)/(b)…，默认放在坐标区左上外侧。"""
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="bottom", ha="left")


def set_equal_3d(ax, points: np.ndarray, margin: float = 0.08, zoom: float = 1.0) -> None:
    """三维等比例坐标轴（数据坐标 1:1）。"""
    from matplotlib.ticker import MaxNLocator

    pts = np.asarray(points, float).reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    pad = margin * float(np.max(span))
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_zlim(lo[2] - pad, hi[2] + pad)
    ax.set_box_aspect(tuple(span + 2 * pad), zoom=zoom)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.zaxis.set_major_locator(MaxNLocator(4))


def style_3d(ax) -> None:
    ax.grid(True, color=COLORS["grid"], linewidth=0.4)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor(COLORS["grid"])
        axis._axinfo["grid"].update(color=COLORS["grid"], linewidth=0.4, linestyle=":")


def _bbox_with_texts(fig):
    """tight bbox 并上所有文本范围，避免 3D 轴标签被裁掉。

    3D 文本的 ``get_window_extent`` 偶尔会返回远在画布之外的坐标，
    这里把每个文本范围限制在画布外扩 0.6 in 的窗口内，防止 bbox 被撑爆。
    """
    from matplotlib.transforms import Bbox

    renderer = fig.canvas.get_renderer()
    bbox = fig.get_tightbbox(renderer)
    limit = fig.bbox_inches.padded(0.6)
    for ax in fig.axes:
        labels = [ax.xaxis.label, ax.yaxis.label]
        if hasattr(ax, "zaxis"):
            labels.append(ax.zaxis.label)
        for txt in list(ax.texts) + labels:
            try:
                if not txt.get_visible() or not str(txt.get_text()):
                    continue
                ext = txt.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
                ext = Bbox.intersection(ext, limit)
                if ext is None:
                    continue
                bbox = Bbox.union([bbox, ext])
            except Exception:
                continue
    return bbox


def save_figure(fig, path: str | Path, dpi: int = 300,
                formats: Sequence[str] = ("png", "pdf")) -> list[Path]:
    """保存 PNG（300 dpi）与 PDF（矢量），返回文件列表。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    bbox = _bbox_with_texts(fig)
    outputs = []
    for fmt in formats:
        p = path.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=dpi, bbox_inches=bbox, pad_inches=0.02, facecolor="white")
        outputs.append(p)
    plt.close(fig)
    return outputs


def error_band(ax, x, y, band: float, color: str = "#999999", alpha: float = 0.15) -> None:
    """在曲线周围画 ±band 的浅色带（用于容差/门限可视化）。"""
    ax.fill_between(x, y - band, y + band, color=color, alpha=alpha, linewidth=0)

"""打ち切り・欠測を「データ」に見せないためのヒートマップ補助 (FIG-7 / D-88).

``fig_ipc_profile`` (3-B) は面積の約7割が 0 で、見えている矩形ブロックは
**打ち切り設定** ``max_delay_by_degree = {1: 60, 2: 20, 3: 10, 4: 6}`` の形
そのものである。0 で埋めて描くと、読者はその段差を系の性質 (「次数3 は遅延10
より先で容量が消える」) と読む。実際には**そこは測っていない**。

そこで打ち切りの外を 0 とは別の色 (グレー) に落とし、境界を線で明示する。
``figures_capacity`` から切り出してあるのは、``figures_capacity.py`` が
D-77 の凍結対象 (861 行) で1行も増やせないためでもある。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

from rc_basics_lab.types import FloatArray

UNCOMPUTED_COLOR = "#bdbdbd"
"""打ち切りの外 (未計算) のセルの色。

配色 (viridis) のどの値とも一致しない無彩色にする。**0 の色と同じにしない**
ことがこのモジュールの存在理由なので、``viridis(0.0)`` の紺色とは別系統の
グレーを選ぶ。
"""

TRUNCATION_EDGE_COLOR = "#ffffff"
"""打ち切り境界の線の色 (暗い配色の上に出すので白)。"""

TRUNCATION_EDGE_WIDTH = 1.4


def masked_beyond_truncation(
    cells: FloatArray, max_delay_by_degree: Mapping[int, int] | None
) -> FloatArray:
    """打ち切りの外を**マスク**した配列を返す (FIG-7)。

    Args:
        cells: ``(次数, 遅延)`` の容量。行 index + 1 が次数、列 index + 1 が遅延。
        max_delay_by_degree: 次数ごとの最大遅延。``None`` なら何もマスクしない
            (打ち切りが分からないときに「未計算」を捏造しない)。

    Returns:
        ``numpy.ma.MaskedArray``。マスクされたセルは ``Colormap.set_bad`` の色で
        描かれる。
    """
    if max_delay_by_degree is None:
        return cells
    mask = np.zeros(cells.shape, dtype=np.bool_)
    n_delays = cells.shape[1]
    for index in range(cells.shape[0]):
        limit = max_delay_by_degree.get(index + 1)
        if limit is None or limit >= n_delays:
            continue
        mask[index, limit:] = True
    masked: FloatArray = np.ma.masked_array(cells, mask=mask)
    return masked


def colormap_with_uncomputed(name: str) -> Colormap:
    """``set_bad`` に ``UNCOMPUTED_COLOR`` を入れた配色を返す。

    元の ``Colormap`` を書き換えると matplotlib のグローバルな登録が汚れるので、
    必ず複製に対して設定する。
    """
    cmap = matplotlib_colormap(name).copy()
    cmap.set_bad(UNCOMPUTED_COLOR)
    return cmap


def matplotlib_colormap(name: str) -> Colormap:
    """名前から ``Colormap`` を引く (``matplotlib.colormaps`` への薄い委譲)。"""
    import matplotlib

    colormap: Colormap = matplotlib.colormaps[name]
    return colormap


def draw_truncation_edges(
    axis: Axes, max_delay_by_degree: Mapping[int, int] | None, n_delays: int
) -> int:
    """打ち切り境界を階段状の線で描く。

    Returns:
        引いた線分の本数 (0 なら境界が図に出ていない)。テストがここを見る。
    """
    if max_delay_by_degree is None:
        return 0
    drawn = 0
    for degree, limit in sorted(max_delay_by_degree.items()):
        if limit >= n_delays:
            continue
        axis.plot(
            [limit + 0.5, limit + 0.5],
            [degree - 0.5, degree + 0.5],
            color=TRUNCATION_EDGE_COLOR,
            linewidth=TRUNCATION_EDGE_WIDTH,
            solid_capstyle="butt",
        )
        drawn += 1
    return drawn


__all__ = [
    "TRUNCATION_EDGE_COLOR",
    "TRUNCATION_EDGE_WIDTH",
    "UNCOMPUTED_COLOR",
    "colormap_with_uncomputed",
    "draw_truncation_edges",
    "masked_beyond_truncation",
    "matplotlib_colormap",
]

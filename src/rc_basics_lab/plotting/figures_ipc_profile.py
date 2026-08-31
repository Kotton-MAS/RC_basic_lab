"""実験 3-B の (次数 x 遅延) ヒートマップ (``fig_ipc_profile``).

``figures_capacity.py`` から分けてあるのは行数上限 (D-77) のためである。
**上限のほうは緩めない**。この図だけが (次数 x 遅延) の2次元格子を扱い、
カラーマップ・打ち切りの表現・段組みを自前で持つので、まとまりとして切れる。

**1段に並べない** (1-6 / FIG-13)。4 パネルを横一列にすると 2.9:1 になり、
Zenn の本文幅 700px では 1 パネルあたり 165px しか残らない。2 段に折ると
各パネルの幅が倍になり、比も 1.26:1 に収まる。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.patches import Patch

from rc_basics_lab.experiment.capacity import CapacityProfileRow, CapacityRow
from rc_basics_lab.plotting.capacity_grids import (
    even_degree_note,
    ipc_heatmap_means,
    representative_leak_rate,
    sweep_conditions,
)
from rc_basics_lab.plotting.heatmap import (
    UNCOMPUTED_COLOR,
    colormap_with_uncomputed,
    draw_truncation_edges,
    masked_beyond_truncation,
)
from rc_basics_lab.plotting.layout import label_panels, wrapped_note
from rc_basics_lab.plotting.style import (
    SEQUENTIAL_CMAP,
    StyleContext,
    add_provenance,
    new_figure,
    rc_context_for,
    require_rows,
    save_png,
)
from rc_basics_lab.types import FloatArray

_HEATMAP_GAMMA = 0.5
"""ヒートマップの色スケールの指数 (平方根)。小さい容量の差を見えるようにする。"""

_MIN_COLOR_MAX = 1.0
"""色スケールの上限の下限 (全セルが 0 でも配色が壊れないように)。"""

_HEATMAP_COLUMNS = 2
"""ヒートマップを並べる列数 (1-6)。4 パネルを横一列にすると Zenn で潰れる。"""


def _plot_heatmap_panel(
    axis: Axes,
    cells: FloatArray,
    rho: float,
    norm: Normalize,
    style: StyleContext,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
    truncation: Mapping[int, int] | None,
) -> QuadMesh:
    """1つの rho ぶんの (次数 x 遅延) ヒートマップ (配色は呼び出し側と共通)。

    打ち切りの外は ``masked_beyond_truncation`` でマスクし、0 とは別のグレーで
    描く (FIG-7 / D-88)。遅延軸は対数にする —— 打ち切りが次数ごとに 60/20/10/6
    と一桁違うので、線形軸では低遅延側 (容量の大半が在る場所) が潰れる。
    """
    n_degrees, n_delays = cells.shape
    mesh = axis.pcolormesh(
        np.arange(n_delays + 1, dtype=np.float64) + 0.5,
        np.arange(n_degrees + 1, dtype=np.float64) + 0.5,
        masked_beyond_truncation(cells, truncation),
        cmap=colormap_with_uncomputed(SEQUENTIAL_CMAP),
        norm=norm,
        shading="flat",
    )
    draw_truncation_edges(axis, truncation, n_delays)
    axis.set_xscale("log")
    axis.set_yticks(np.arange(1, n_degrees + 1, dtype=np.float64))
    if show_xlabel:
        axis.set_xlabel(style.label("遅延 k [ステップ・対数]", "delay k [steps, log]"))
    if show_ylabel:
        axis.set_ylabel(style.label("次数 d", "degree d"))
    axis.set_title(
        style.label(f"rho = {rho:g}", f"rho = {rho:g}"),
        fontsize=10,
    )
    axis.grid(visible=False)
    return mesh


def plot_ipc_profile(
    rows: Sequence[CapacityRow],
    profile: Sequence[CapacityProfileRow],
    path: Path,
    *,
    style: StyleContext,
    max_delay_by_degree: Mapping[int, int] | None = None,
) -> Path:
    """実験 3-B の (次数 x 遅延) ヒートマップを rho 別に並べる (受け入れ条件4)。

    パネルは代表リーク率 1本 x rho 4点。配色は全パネル共通の上限を使う ——
    パネルごとに正規化すると主張が色の付け替えで消える。

    Args:
        rows: 3-B の行。
        profile: 3-B の長形式の行 (D-38)。
        path: 出力先 PNG。
        style: ``setup_style()`` の戻り値。
        max_delay_by_degree: 次数ごとの遅延の打ち切り (``cfg`` 由来)。
            与えると打ち切りの外を「未計算」のグレーに落とす (FIG-7 / D-88)。
            **省略すると全セルが計算済みとして描かれる** (領域を捏造しない)。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    require_rows(rows)
    leak_rate = representative_leak_rate(rows, lambda row: row.ipc_total)
    means = ipc_heatmap_means(rows, profile, leak_rate)
    rhos = tuple(means)
    ceiling = max((float(cells.max()) for cells in means.values()), default=0.0)
    norm = PowerNorm(gamma=_HEATMAP_GAMMA, vmin=0.0, vmax=max(ceiling, _MIN_COLOR_MAX))

    with rc_context_for(style):
        # **1段に並べない** (1-6 / FIG-13)。4 パネルを横一列にすると 2.9:1 に
        # なり、Zenn の本文幅 700px では 1 パネルあたり 165px しか残らない。
        # 2 段に折ると各パネルの幅が倍になり、比も 1.5:1 に収まる。
        columns = _HEATMAP_COLUMNS if len(rhos) > _HEATMAP_COLUMNS else len(rhos)
        grid_rows = -(-len(rhos) // columns)
        # 段あたりの高さはヒートマップの縦横比に合わせる。4.4 にすると
        # 全体が 0.86:1 の縦長になり、今度は下限 (FIG-13) を割る。
        figure = new_figure(3.4 * columns + 1.2, 3.0 * grid_rows)
        # 同じ量の軸が並ぶので目盛りは共有し、ラベルは外周だけに出す (1-7)。
        axes = figure.subplots(
            grid_rows, columns, squeeze=False, sharex=True, sharey=True
        )
        flat = list(axes.ravel())
        for extra in flat[len(rhos) :]:
            extra.set_axis_off()
        label_panels(flat[: len(rhos)], style=style)
        meshes = [
            _plot_heatmap_panel(
                flat[index],
                means[rho],
                rho,
                norm,
                style,
                # x ラベルは最下段だけ、y ラベルは各段の左端だけに出す
                # (1-7: 同じ量の軸ラベルをパネルの数だけ繰り返さない)。
                show_xlabel=index >= len(rhos) - columns,
                show_ylabel=index % columns == 0,
                truncation=max_delay_by_degree,
            )
            for index, rho in enumerate(rhos)
        ]
        figure.colorbar(
            meshes[-1],
            ax=flat[: len(rhos)],
            label=style.label("容量 (平方根スケール)", "capacity (sqrt colour scale)"),
        )
        # **未計算は凡例パッチで示す** (2-12)。カラーバーのラベルに文章で
        # 「灰色 = 未計算」と書いても、色そのものと結び付けて読む手がかりが
        # 図の中に無い。値のスケール (カラーバー) と、値が無いこと (パッチ) は
        # 別の情報なので分ける。
        figure.legend(
            handles=[
                Patch(
                    facecolor=UNCOMPUTED_COLOR,
                    edgecolor="none",
                    label=style.label(
                        "未計算 (打ち切りの外)", "not computed (beyond truncation)"
                    ),
                )
            ],
            loc="lower left",
            bbox_to_anchor=(0.0, -0.06),
            frameon=False,
            fontsize=8,
        )
        first = rows[0]
        figure.suptitle(
            style.label(
                "実験 3-B: rho を上げると次数3 の容量が失われ、残るのは次数1 になる",
                "Experiment 3-B: raising rho destroys the degree-3 capacity"
                " and leaves the degree-1 part",
            )
        )
        figure.supxlabel(
            wrapped_note(style.label(*even_degree_note(profile))), fontsize=8
        )
        conditions = sweep_conditions(first, leak_rate)
        add_provenance(figure, conditions, rows, style=style)
        return save_png(figure, path)


__all__ = ["plot_ipc_profile"]

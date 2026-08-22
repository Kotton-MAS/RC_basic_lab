"""4-C / 4-D の安定性地図 (``fig_stability_map``).

``figures_freerun.py`` から分けてあるのは行数上限 (D-77) のためである。
**上限のほうを緩めない**。安定性地図は 4-C (スペクトル半径の掃引) と
4-D (同じ状態行列で測った容量) の2段で1つの主張なので、まとまりとして切れる。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.axes import Axes

from rc_basics_lab.experiment.attractor import REGIMES
from rc_basics_lab.experiment.capacity import CapacityRow
from rc_basics_lab.experiment.stability import StabilityRow, regime_map
from rc_basics_lab.plotting.figures_freerun import (
    REGIME_COLORS,
    REGIME_LABELS,
)
from rc_basics_lab.plotting.freerun_grids import label_of
from rc_basics_lab.plotting.style import (
    StyleContext,
    add_provenance,
    new_figure,
    rc_context_for,
    require_rows,
    save_png,
)


def plot_stability_map(
    rows: Sequence[StabilityRow],
    capacity_rows: Sequence[CapacityRow],
    path: Path,
    *,
    style: StyleContext,
) -> Path:
    """実験 4-C: (rho x リーク率) の3態マップ + 4-D の容量との関係。

    ノイズ量ごとにパネルを並べ、格子点の色が3態を表す (レプリケートは多数決、
    ``regime_map``)。**分類は行の値そのもの**であり、図から決めていない
    (仕様 §5 禁止する構造6)。最後のパネルは 4-D: 同じ条件で測った
    ``mc_total`` と有効予測時間の関係を3態で色分けする。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    require_rows(rows)
    noises = sorted({row.state_noise for row in rows})
    # **1段に並べない** (FIG-13)。パネル4枚を横一列にすると 4.03:1 になり、
    # 記事の1段組では各パネルが親指ほどの幅に潰れる。2段に折ると 1.2:1 に収まる。
    panels = len(noises) + 1
    columns = 2 if panels <= 4 else 3
    grid_rows = -(-panels // columns)
    with rc_context_for(style):
        figure = new_figure(5.0 * columns + 1.0, 4.6 * grid_rows)
        axes = np.atleast_1d(figure.subplots(grid_rows, columns)).reshape(-1)
        for axis, noise in zip(axes[: len(noises)], noises, strict=True):
            _regime_panel(axis, rows, noise, style)
        _capacity_panel(axes[len(noises)], rows, capacity_rows, style)
        # 余った枠は消す。空の軸を残すと「測ったが何も無かった」に見える。
        for axis in axes[panels:]:
            axis.set_axis_off()
        handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                color=REGIME_COLORS[regime],
                label=label_of(REGIME_LABELS, regime, style),
            )
            for regime in REGIMES
        ]
        figure.legend(
            handles=handles,
            loc="outside lower center",
            ncols=len(REGIMES),
            fontsize=9,
        )
        figure.suptitle(
            style.label(
                "実験 4-C / 4-D: アトラクタを再現できる領域は"
                "状態ノイズ 0.01 までほとんど動かない",
                "Experiments 4-C / 4-D: the region that reproduces the attractor"
                " barely moves up to a state noise of 0.01",
            )
        )
        conditions = f"N = {rows[0].n_units}, stats = {rows[0].stats_steps} steps"
        add_provenance(figure, conditions, rows, style=style)
        return save_png(figure, path)


def _regime_panel(
    axis: Axes, rows: Sequence[StabilityRow], noise: float, style: StyleContext
) -> None:
    """状態ノイズ 1 点ぶんの3態マップ (格子点を色で塗る)。"""
    mapping = regime_map(rows, noise)
    rhos = sorted({key[0] for key in mapping})
    leaks = sorted({key[1] for key in mapping})
    for (rho, leak), regime in mapping.items():
        axis.scatter(
            [rho],
            [leak],
            s=420,
            marker="s",
            color=REGIME_COLORS[regime],
            edgecolors="white",
        )
    axis.set_xticks(rhos)
    axis.set_yticks(leaks)
    axis.set_xlim(min(rhos) - 0.15, max(rhos) + 0.15)
    axis.set_ylim(min(leaks) - 0.12, max(leaks) + 0.12)
    axis.set_xlabel(style.label("スペクトル半径 rho", "spectral radius rho"))
    axis.set_ylabel(style.label("リーク率", "leak rate"))
    axis.set_title(
        style.label(f"状態ノイズ = {noise:g}", f"state noise = {noise:g}"), fontsize=10
    )


def _capacity_panel(
    axis: Axes,
    rows: Sequence[StabilityRow],
    capacity_rows: Sequence[CapacityRow],
    style: StyleContext,
) -> None:
    """4-D: ``mc_total`` と有効予測時間の関係 (条件キーで join する)。"""
    keyed = {
        (row.rho, row.leak_rate, row.state_noise, row.replicate): row
        for row in capacity_rows
    }
    for regime in REGIMES:
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            if row.regime != regime:
                continue
            capacity = keyed.get(
                (row.rho, row.leak_rate, row.state_noise, row.replicate)
            )
            if capacity is None:
                continue
            xs.append(capacity.mc_total)
            ys.append(row.valid_time_lyapunov)
        if xs:
            axis.scatter(xs, ys, s=26, alpha=0.75, color=REGIME_COLORS[regime])
    axis.set_xlabel(style.label("線形メモリ容量 MC", "linear memory capacity MC"))
    axis.set_ylabel(
        style.label("有効予測時間 [Lyapunov 時間]", "valid time [1 / lambda_max]")
    )
    axis.set_title(
        style.label(
            "4-D: 同じ状態行列で測った容量", "4-D: capacity on the same states"
        ),
        fontsize=10,
    )


__all__ = ["plot_stability_map"]

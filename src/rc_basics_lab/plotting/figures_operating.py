"""実験 3-C'' の図 —— NARMA10 の勝敗が手法側の動作点で変わる (D-145).

3-C 本体の図は動作点1つの比較なので、「その1点を選んだから ESN が負けた」
のか「NARMA10 では遅延線が強い」のかを区別できない。ここが描くのは
**動作点の面**である。

左は素直な読み方 (N とリーク率に対する成績)、右は**何が勝敗を分けたか**で
ある。実測では非線形の割合は勝敗を予測せず、線形容量の絶対値が予測した
(D-144) ので、右の横軸は ``ipc_linear`` にする。

遅延線は動作点によらず一定なので水平線で置く。**交点が結論**である ——
ESN は自分の線形容量が遅延線に許されたタップ数を超えたときに勝つ。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from rc_basics_lab.experiment.narma_operating import OperatingPointRow
from rc_basics_lab.plotting.capacity_grids import mean_std
from rc_basics_lab.plotting.layout import (
    hide_minor_tick_labels,
    label_panels,
    wrapped_note,
)
from rc_basics_lab.plotting.style import (
    REFERENCE_COLOR,
    REFERENCE_DASHES,
    StyleContext,
    add_footnote,
    method_color,
    new_figure,
    rc_context_for,
    save_png,
    sequential_colors,
)

ESN = "esn"
"""動作点で動く側の手法。"""

DELAY_LINE = "delay_line"
"""動作点で動かない側の対照 (リザバーを見ない)。"""


def _baseline(rows: Sequence[OperatingPointRow]) -> float:
    """遅延線の NRMSE (動作点によらず一定のはず)。

    Raises:
        ValueError: 遅延線の行が無い、または動作点で動いている場合。
    """
    values = {round(row.nrmse, 9) for row in rows if row.method == DELAY_LINE}
    if not values:
        raise ValueError("遅延線の行がありません")
    means = [
        mean_std([r.nrmse for r in rows if r.method == DELAY_LINE and r.n_units == n])[
            0
        ]
        for n in sorted({row.n_units for row in rows})
    ]
    if max(means) - min(means) > 1.0e-9:
        raise ValueError(
            f"遅延線が動作点で動いています ({min(means):.6f}〜{max(means):.6f})。"
            "掃引が課題か分割まで動かしています"
        )
    return means[0]


def _point_value(
    rows: Sequence[OperatingPointRow], n_units: int, leak_rate: float, column: str
) -> float:
    """1動作点の代表値 (容量の列は行で共有しているので先頭を取る)。"""
    for row in rows:
        if row.n_units == n_units and row.leak_rate == leak_rate:
            return float(getattr(row, column))
    raise ValueError(f"動作点 (N={n_units}, leak={leak_rate}) の行がありません")


def operating_headline(rows: Sequence[OperatingPointRow], style: StyleContext) -> str:
    """タイトルの結論文を**行から導く** (固定文にしない)。"""
    baseline = _baseline(rows)
    points = sorted({(row.n_units, row.leak_rate) for row in rows})
    wins = 0
    for n_units, leak_rate in points:
        mean, _ = mean_std(
            [
                row.nrmse
                for row in rows
                if row.method == ESN
                and row.n_units == n_units
                and row.leak_rate == leak_rate
            ]
        )
        if mean < baseline:
            wins += 1
    return style.label(
        f"同じ課題・同じ探索予算でも、動作点 {len(points)} 点のうち"
        f" {wins} 点で ESN が遅延線に勝つ",
        f"with the same task and search budget the ESN beats the delay line"
        f" at {wins} of {len(points)} operating points",
    )


def draw_grid_panel(
    axis: Axes, rows: Sequence[OperatingPointRow], style: StyleContext
) -> None:
    """N とリーク率に対する ESN の成績 (遅延線を水平線で置く)。"""
    leaks = sorted({row.leak_rate for row in rows})
    colors = sequential_colors(len(leaks))
    for index, leak_rate in enumerate(leaks):
        units = sorted({row.n_units for row in rows})
        means: list[float] = []
        stds: list[float] = []
        for n_units in units:
            mean, std = mean_std(
                [
                    row.nrmse
                    for row in rows
                    if row.method == ESN
                    and row.n_units == n_units
                    and row.leak_rate == leak_rate
                ]
            )
            means.append(mean)
            stds.append(std)
        axis.errorbar(
            units,
            means,
            yerr=np.asarray(stds, dtype=np.float64),
            fmt="o-",
            capsize=4,
            color=colors[index],
            label=style.label(f"ESN リーク率 {leak_rate:g}", f"ESN leak {leak_rate:g}"),
        )
    axis.axhline(
        _baseline(rows),
        color=method_color(DELAY_LINE),
        dashes=REFERENCE_DASHES[0],
        label=style.label(
            "遅延線 (動作点によらず一定)", "delay line (flat in the operating point)"
        ),
    )
    axis.set_xscale("log")
    units = sorted({row.n_units for row in rows})
    axis.set_xticks(units)
    axis.set_xticklabels([str(value) for value in units])
    # 副目盛りのラベル (3x10^1 など) は**測っていない点**である (FIG-19)。
    hide_minor_tick_labels(axis, which="x")
    axis.set_xlabel(style.label("ユニット数 N", "units N"))
    axis.set_ylabel(style.label("NRMSE (小さいほど良い)", "NRMSE (lower is better)"))
    axis.legend(loc="best")


def draw_capacity_panel(
    axis: Axes, rows: Sequence[OperatingPointRow], style: StyleContext
) -> None:
    """**何が勝敗を分けたか** —— 線形容量に対する ESN の成績。

    非線形の割合ではなく線形容量を横軸に取る根拠は D-144 の実測にある
    (割合は勝敗を予測せず、線形容量は完全に分離した)。
    """
    baseline = _baseline(rows)
    leaks = sorted({row.leak_rate for row in rows})
    colors = sequential_colors(len(leaks))
    for index, leak_rate in enumerate(leaks):
        points = sorted(
            {(row.n_units, row.leak_rate) for row in rows if row.leak_rate == leak_rate}
        )
        xs = [_point_value(rows, n, leak, "ipc_linear") for n, leak in points]
        ys = [
            mean_std(
                [
                    row.nrmse
                    for row in rows
                    if row.method == ESN and row.n_units == n and row.leak_rate == leak
                ]
            )[0]
            for n, leak in points
        ]
        axis.plot(
            xs,
            ys,
            "o-",
            color=colors[index],
            label=style.label(f"リーク率 {leak_rate:g}", f"leak {leak_rate:g}"),
        )
    axis.axhline(
        baseline,
        color=method_color(DELAY_LINE),
        dashes=REFERENCE_DASHES[0],
        label=style.label("遅延線", "delay line"),
    )
    taps = sorted({row.n_lags for row in rows if row.method == DELAY_LINE})
    if taps:
        axis.axvspan(
            float(min(taps)),
            float(max(taps)),
            color=REFERENCE_COLOR,
            alpha=0.15,
            label=style.label(
                f"遅延線が選んだタップ数 {min(taps)}〜{max(taps)}",
                f"taps chosen by the delay line: {min(taps)}-{max(taps)}",
            ),
        )
    axis.set_xlabel(
        style.label(
            "ESN の線形容量 IPC_linear", "linear capacity IPC_linear of the ESN"
        )
    )
    axis.set_ylabel(style.label("NRMSE (小さいほど良い)", "NRMSE (lower is better)"))
    axis.legend(loc="best")


def plot_operating(
    rows: Sequence[OperatingPointRow],
    path: Path,
    *,
    style: StyleContext,
) -> Path:
    """3-C'' の動作点の面を1枚へ並べる (D-145)。

    Args:
        rows: ``narma10_operating.csv`` と同じ行。
        path: 出力先の PNG。
        style: 配色・言語・commit。

    Returns:
        書き出した PNG のパス。

    Raises:
        ValueError: 行が空、または遅延線が動作点で動いている場合。
    """
    if not rows:
        raise ValueError("rows が空です")
    with rc_context_for(style):
        figure = new_figure(13.0, 5.5)
        grid = figure.add_gridspec(1, 2)
        left = figure.add_subplot(grid[0, 0])
        right = figure.add_subplot(grid[0, 1])
        draw_grid_panel(left, rows, style)
        draw_capacity_panel(right, rows, style)
        label_panels([left, right], style=style)
        figure.suptitle(f"3-C'': {operating_headline(rows, style)}")
        figure.supxlabel(
            wrapped_note(
                style.label(
                    "遅延線は動作点によらず一定 (リザバーを見ないため)。"
                    "右の帯は遅延線が検証で選んだタップ数で、"
                    "ESN はそこを超える線形容量を持ったときに勝つ。",
                    "The delay line is flat because it never sees the"
                    " reservoir. The band on the right is the number of taps"
                    " it selected; the ESN wins once its linear capacity"
                    " exceeds that band.",
                )
            )
        )
        add_footnote(figure, _operating_conditions(rows, style), style=style)
        return save_png(figure, path)


def _operating_conditions(
    rows: Sequence[OperatingPointRow], style: StyleContext
) -> str:
    """footnote の再現条件 (FIG-6)。"""
    points = len({(row.n_units, row.leak_rate) for row in rows})
    replicates = sorted({row.replicate for row in rows})
    return style.label(
        f"3-C'', 動作点 {points}, レプリケート {len(replicates)}",
        f"3-C'', {points} operating points, {len(replicates)} replicates",
    )


__all__ = [
    "DELAY_LINE",
    "ESN",
    "draw_capacity_panel",
    "draw_grid_panel",
    "operating_headline",
    "plot_operating",
]

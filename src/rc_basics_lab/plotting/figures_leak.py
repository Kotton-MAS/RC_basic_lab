"""実験 2-B: リーク率と実効時定数 (``fig_leak_timescale``).

``figures_esp.py`` から分けてあるのは行数上限 (D-77) のためである。
**上限のほうは緩めない**。

右列は 2 段で、上が実測 tau と理論線、下が**倍率そのもの** (2-16)。
見出しの主張「理論の 1.3〜1.4 倍」は、2 本を並べただけでは図から読めない。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

from rc_basics_lab.experiment.esp import ConditionOutcome, EspRow
from rc_basics_lab.plotting.figures_esp import replicate_count
from rc_basics_lab.plotting.layout import label_panels, wrapped_note
from rc_basics_lab.plotting.style import (
    StyleContext,
    add_provenance,
    new_figure,
    rc_context_for,
    reference_line_kwargs,
    save_png,
    sequential_colors,
    unique_sorted,
)
from rc_basics_lab.types import FloatArray


def _theory_timescale(leak_rate: float) -> float:
    """線形域での実効時定数 ``-1 / log(1 - a)``。``a = 1`` では 0。"""
    if leak_rate >= 1.0:
        return 0.0
    return -1.0 / math.log(1.0 - leak_rate)


def _plot_acf_panel(
    axis: Axes,
    outcomes: Sequence[ConditionOutcome],
    leak_rates: Sequence[float],
    colors: FloatArray,
    style: StyleContext,
) -> None:
    """リーク率ごとの平均自己相関曲線 (時定数の素の測定量)。"""
    for index, leak_rate in enumerate(leak_rates):
        selected = [
            outcome for outcome in outcomes if outcome.row.leak_rate == leak_rate
        ]
        for order, outcome in enumerate(selected):
            axis.plot(
                np.arange(outcome.acf.shape[0]),
                outcome.acf,
                color=colors[index],
                linewidth=1.2,
                alpha=0.85,
                label=(
                    style.label(f"a = {leak_rate:g}", f"a = {leak_rate:g}")
                    if order == 0
                    else None
                ),
            )
    axis.axhline(
        1.0 / math.e,
        **reference_line_kwargs(),
        label=style.label("1/e (時定数の定義水準)", "1/e (level defining tau)"),
    )
    axis.set_xlabel(style.label("ラグ [ステップ]", "lag [steps]"))
    axis.set_ylabel(style.label("ユニット平均の自己相関", "unit-averaged ACF"))
    axis.set_xlim(0.0, 40.0)
    axis.set_title(
        style.label("自己相関の減衰", "Decay of the autocorrelation"), fontsize=10
    )
    axis.legend(loc="upper right", fontsize=8, ncols=2)


def _plot_timescale_panel(
    axis: Axes, rows: Sequence[EspRow], leak_rates: Sequence[float], style: StyleContext
) -> FloatArray:
    """実効時定数と理論線 ``-1/log(1-a)`` の比較。

    Returns:
        リーク率ごとの実測 tau のレプリケート平均 (比のパネルが使う)。
    """
    measured = [
        [row.tau_censored for row in rows if row.leak_rate == leak_rate]
        for leak_rate in leak_rates
    ]
    means: FloatArray = np.array(
        [float(np.mean(values)) for values in measured], dtype=np.float64
    )
    stds: FloatArray = np.array(
        [float(np.std(values)) for values in measured], dtype=np.float64
    )
    axis.errorbar(
        list(leak_rates),
        means,
        yerr=stds,
        fmt="o-",
        capsize=4,
        # リーク率の掃引そのものなので連続量の配色から取る (FIG-5)
        color=sequential_colors(1)[0],
        label=style.label(
            "実測 tau (自己相関が 1/e を切るラグ)",
            "measured tau (lag where the ACF crosses 1/e)",
        ),
    )
    theory_points = [
        (leak_rate, _theory_timescale(leak_rate))
        for leak_rate in leak_rates
        if leak_rate < 1.0
    ]
    if theory_points:
        axis.plot(
            [point[0] for point in theory_points],
            [point[1] for point in theory_points],
            **reference_line_kwargs(1),
            label=style.label(
                "理論線 -1 / log(1 - a) (線形域)",
                "theory -1 / log(1 - a) (linear regime)",
            ),
        )
    axis.set_yscale("log")
    axis.set_xlabel(style.label("リーク率 a", "leak rate a"))
    axis.set_ylabel(
        style.label("1/e 時定数 tau_1e [ステップ]", "1/e timescale tau_1e [steps]")
    )
    axis.set_title(
        style.label(
            "リーク率と 1/e 時定数 (単調減少)",
            "Leak rate versus effective timescale (monotone)",
        ),
        fontsize=10,
    )
    axis.tick_params(labelbottom=False)
    axis.set_xlabel("")
    axis.legend(loc="upper right", fontsize=8)
    return means


_THEORY_CUTOFF_NOTE = (
    "理論線が a = 1 の手前で切れているのは、-1 / log(1 - a) が a -> 1 で 0 へ"
    "落ちて対数軸に載らないためである (a = 1 は「前の状態を持ち越さない」)。",
    "The theory line stops short of a = 1 because -1 / log(1 - a) goes to 0"
    " there and cannot be drawn on a log axis (a = 1 keeps no past state).",
)
"""理論線が途中で終わる理由 (2-16)。書かないと「測り忘れ」に見える。"""


def _plot_ratio_panel(
    axis: Axes,
    measured: FloatArray,
    leak_rates: Sequence[float],
    style: StyleContext,
) -> None:
    """実測 / 理論の比そのものを描く (2-16)。

    見出しの主張は「理論の 1.3〜1.4 倍」なのに、上の対数パネルは実測と理論の
    2 本が並んでいるだけで**倍率は描かれていなかった**。比を線形軸で描けば、
    主張の数値がそのまま図から読める。``a = 1`` は理論が 0 なので比が定義
    できず、点を置かない。
    """
    points = [
        (leak_rate, value / _theory_timescale(leak_rate))
        for leak_rate, value in zip(leak_rates, measured, strict=True)
        if leak_rate < 1.0 and _theory_timescale(leak_rate) > 0.0
    ]
    axis.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        "o-",
        color=sequential_colors(1)[0],
    )
    axis.axhline(
        1.0,
        **reference_line_kwargs(),
        label=style.label("比 = 1 (理論どおり)", "ratio = 1 (theory)"),
    )
    axis.set_xlabel(style.label("リーク率 a", "leak rate a"))
    axis.set_ylabel(style.label("実測 / 理論", "measured / theory"))
    # **0 から描かない** (2-7 と同じ理由)。主張は「1.3〜1.4 倍」で、0 から
    # 取ると比の差 0.1 が panel の 6% に潰れて読めない。比 = 1 の参照線は
    # 範囲に必ず含める —— 「1 より上か下か」が読めなくなるほうが害が大きい。
    values = [value for _, value in points]
    low, high = min(1.0, *values), max(1.0, *values)
    margin = max((high - low) * 0.25, 0.05)
    axis.set_ylim(low - margin, high + margin)
    axis.yaxis.set_major_locator(MultipleLocator(0.1))
    axis.legend(loc="lower right", fontsize=7)


def plot_leak_timescale(
    outcomes: Sequence[ConditionOutcome], path: Path, *, style: StyleContext
) -> Path:
    """リーク率と実効時定数の関係を描く (受け入れ条件4)。

    左が自己相関曲線、右が 1/e 交差から求めた時定数と理論線 ``-1/log(1-a)``
    の比較。理論線は線形域の値なので実測より小さく出るが、**単調性が一致する**
    ことが受け入れ条件である。**描いているのは ``tau_1e`` だけ** (D-117)
    —— ``tau_integrated`` は比が 1.58 -> 1.94 と開く (design.md §9.2)。

    Raises:
        ValueError: ``outcomes`` が空の場合。
    """
    if not outcomes:
        raise ValueError("outcomes が空です")
    rows = [outcome.row for outcome in outcomes]
    leak_rates = unique_sorted([row.leak_rate for row in rows])
    colors = sequential_colors(len(leak_rates))

    with rc_context_for(style):
        figure = new_figure(11.0, 5.8)  # 高さは 1-6 (Zenn 幅で潰れない比)
        # 右列を 2 段にして、下段に**倍率そのもの**を描く (2-16)。左の ACF は
        # 両段にまたがる (時定数の素の測定量で、比とは別の話をしている)。
        grid = figure.add_gridspec(2, 2, height_ratios=(2.4, 1.0))
        acf_axis = figure.add_subplot(grid[:, 0])
        tau_axis = figure.add_subplot(grid[0, 1])
        ratio_axis = figure.add_subplot(grid[1, 1], sharex=tau_axis)
        label_panels([acf_axis, tau_axis, ratio_axis], style=style)
        _plot_acf_panel(acf_axis, outcomes, leak_rates, colors, style)
        means = _plot_timescale_panel(tau_axis, rows, leak_rates, style)
        _plot_ratio_panel(ratio_axis, means, leak_rates, style)
        first = rows[0]
        figure.suptitle(
            style.label(
                "実験 2-B: 1/e 時定数はリーク率によらず理論の約 1.3〜1.4 倍",
                "Experiment 2-B: the 1/e timescale stays about 1.3-1.4x the"
                " theory line across leak rates",
            )
        )
        figure.supxlabel(
            wrapped_note(
                style.label(
                    f"注: (a) の同じ色の {replicate_count(outcomes)} 本は"
                    "レプリケート (乱数シード違い) である。" + _THEORY_CUTOFF_NOTE[0],
                    f"Note: the {replicate_count(outcomes)} curves sharing a"
                    " colour in (a) are replicates (different random seeds). "
                    + _THEORY_CUTOFF_NOTE[1],
                )
            ),
            fontsize=8,
        )
        conditions = (
            f"N = {first.n_units}, rho = {first.rho:g}, sigma_u = {first.sigma_u:g}"
        )
        add_provenance(figure, conditions, rows, style=style)
        return save_png(figure, path)


__all__ = ["plot_leak_timescale"]

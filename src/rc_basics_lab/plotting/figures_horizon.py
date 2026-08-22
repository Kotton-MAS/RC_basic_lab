"""実験 01' の図 —— 自走 84 ステップ先の誤差と文献値 (D-105).

01 本体の図は1ステップ先の比較なので、文献 (Jaeger & Haas 2004) の
**NRMSE84** とは予測長が違って並べられない。ここが描くのは同じ予測長の量である。

縦軸を ``log10(NRMSE84)`` にするのは、文献がその形で報告しているからで、
**こちらの都合ではない**。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rc_basics_lab.experiment.horizon import HORIZON_STEPS, HorizonRow
from rc_basics_lab.plotting.labels import JAEGER_HAAS_2004, cited_measurement
from rc_basics_lab.plotting.style import (
    METHOD_COLORS,
    REFERENCE_COLOR,
    REFERENCE_DASHES,
    StyleContext,
    add_provenance,
    method_color,
    new_figure,
    rc_context_for,
    save_png,
)

JAEGER_LOG10_NRMSE84 = -4.2
"""Jaeger & Haas (2004) が報告する ``log10(NRMSE84)`` (D-105)。

Science **304**:78-80 本文の実測値。同論文は「従来手法より 700 倍良い」とも
書いており、その従来手法側が ``JAEGER_PREVIOUS_LOG10`` にあたる。
"""

JAEGER_PREVIOUS_LOG10 = math.log10(10.0**JAEGER_LOG10_NRMSE84 * 700.0)
"""同論文が「従来手法」として挙げる水準 (700 倍悪い側)。

値を直書きせず 700 倍から導くのは、**論文が述べているのは比のほう**だからで、
2つの数を別々に書くと片方だけ直したときに整合が崩れる。
"""

JAEGER_CONDITIONS: tuple[str, str] = (
    "N = 1000・出力フィードバックつき自走・100 試行",
    "N = 1000, autonomous run with output feedback, 100 trials",
)
"""参照値の**動作点** (D-97 / D-105)。

``N`` を条件に書くのが要点である。こちらは ``N = 200`` で 5 倍小さく、
規模差が誤差差の一部を説明する。読者がそれを図の上で判断できるようにする。
"""


def plot_horizon(
    rows: Sequence[HorizonRow], path: Path, *, style: StyleContext
) -> Path:
    """実験 01': 自走 84 ステップ先の誤差を文献値と並べる。

    Args:
        rows: ``run_horizon`` の出力。
        path: 出力先の PNG。
        style: 配色・言語・commit。

    Returns:
        書き出した PNG のパス。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    if not rows:
        raise ValueError("rows が空です")
    logs = [
        row.log10_nrmse_horizon
        for row in rows
        if math.isfinite(row.log10_nrmse_horizon)
    ]
    with rc_context_for(style):
        figure = new_figure(7.4, 5.0)
        axis = figure.subplots(1, 1)
        method = rows[0].method
        axis.plot(
            [row.replicate for row in rows],
            [row.log10_nrmse_horizon for row in rows],
            "o",
            color=method_color(method) if method in METHOD_COLORS else None,
            markersize=8,
            label=style.label(
                f"このリポジトリ (N = {rows[0].n_units})",
                f"this repository (N = {rows[0].n_units})",
            ),
        )
        if logs:
            axis.axhline(
                float(np.mean(logs)),
                color=method_color(method) if method in METHOD_COLORS else None,
                linestyle="-",
                linewidth=1.0,
            )
        for value, text_ja, text_en, dashes in (
            (
                JAEGER_LOG10_NRMSE84,
                "文献の ESN",
                "the cited ESN",
                REFERENCE_DASHES[0],
            ),
            (
                JAEGER_PREVIOUS_LOG10,
                "同論文が挙げる従来手法 (700 倍悪い)",
                "the prior techniques it cites (700x worse)",
                REFERENCE_DASHES[1],
            ),
        ):
            axis.axhline(
                value,
                color=REFERENCE_COLOR,
                dashes=dashes,
                linewidth=1.2,
                label=cited_measurement(
                    style.label(f"{text_ja} {value:.2f}", f"{text_en} {value:.2f}"),
                    JAEGER_HAAS_2004,
                    style.label(*JAEGER_CONDITIONS),
                ),
            )
        axis.set_xlabel(style.label("レプリケート", "replicate"))
        axis.set_ylabel(
            style.label(
                f"log10 NRMSE ({HORIZON_STEPS} ステップ先の1点)",
                f"log10 NRMSE (single point, {HORIZON_STEPS} steps ahead)",
            )
        )
        axis.set_xticks([row.replicate for row in rows])
        axis.legend(loc="best", fontsize=7)
        figure.suptitle(
            f"{style.label('実験 01', 'Experiment 01')}': {_headline(logs, style)}\n"
            + style.label(
                "1ステップ先を学習した同じ読み出しを自走させた (学習し直していない)",
                "the same one-step readout, run autonomously (not retrained)",
            )
        )
        conditions = f"task = {rows[0].task}, horizon = {HORIZON_STEPS}"
        add_provenance(figure, conditions, rows, style=style)
        return save_png(figure, path)


def _headline(logs: Sequence[float], style: StyleContext) -> str:
    """結論文を**行から導く** (D-90 と同じ規律)。"""
    if not logs:
        return style.label(
            "自走が測れなかった", "the autonomous run could not be measured"
        )
    mean = float(np.mean(logs))
    gap = mean - JAEGER_LOG10_NRMSE84
    return style.label(
        f"84 ステップ先の誤差は log10 で {mean:.2f} —— 文献より {gap:.1f} 桁大きい",
        f"the 84-step error is {mean:.2f} in log10: {gap:.1f} decades above"
        " the cited value",
    )


__all__ = [
    "JAEGER_CONDITIONS",
    "JAEGER_LOG10_NRMSE84",
    "JAEGER_PREVIOUS_LOG10",
    "plot_horizon",
]

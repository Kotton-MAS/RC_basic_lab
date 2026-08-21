"""記事05の図のうち掃引の2枚 (実験 5-C / 5-D) —— **印を必ず可視化する** (D-81).

- ``plot_protocol_sensitivity``: 5-C。プロトコル別の順位の変動と、一様乱数
  対照と区別できるかの印 (D-78)。
- ``plot_size_vs_performance``: 5-D。リザバーサイズ N と AUPRC、劣化点。

``figures_anomaly.py`` から分けたのは行数のためである (5枚で 673 行 = D-63 /
D-77 の上限 600 行を超えた)。**上限に当たったので割った**のであって上限は
動かしていない。系統の色・ラベルと印の体裁は ``figures_anomaly`` が単一の
真実で、ここは import して使うだけである。

**印を描かない図を作らない** (D-81)。一様乱数対照と区別できる系統
(``distinguishable``) は太い実線 + 塗りつぶしマーカー、区別できない系統は
細い破線 + 白抜きマーカーで描き、凡例にもその区別を書く。加えて 5-C の右
パネルは「逆転した系統対の延べ数」と「そのうち**両方に印がある**数」を並べる
—— 実測ではこの2つが 46 組と 0 組で、前者だけを報告すると
「プロトコルに敏感」という**逆の結論**になる。

**図は成果物 CSV の行だけを読む** (仕様 §5 禁止する構造7)。掃引も集計も
ここでは走らせない (``summarize_size_sweep`` は行から数を作る純関数)。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from rc_basics_lab.experiment.anomaly_rows import ProtocolSweepRow, SizeSweepRow
from rc_basics_lab.experiment.anomaly_score import ANOMALY_METHODS, ESN_RESIDUAL
from rc_basics_lab.experiment.anomaly_sweep import (
    DEGRADATION_FRACTION,
    summarize_size_sweep,
)
from rc_basics_lab.plotting.figures_anomaly import (
    METHOD_COLORS,
    method_label,
    method_line_style,
)
from rc_basics_lab.plotting.style import (
    StyleContext,
    new_figure,
    rc_context_for,
    require_rows,
    save_png,
)


def _methods_in(rows: Sequence[SizeSweepRow]) -> list[str]:
    """行に現れる系統を ``ANOMALY_METHODS`` の順で返す。"""
    found = {row.method for row in rows}
    return [method for method in ANOMALY_METHODS if method in found]


def _condition_key(row: ProtocolSweepRow) -> tuple[str, int, int]:
    return (row.normalize, row.input_window, row.score_smoothing)


def _condition_keys(rows: Sequence[ProtocolSweepRow]) -> list[tuple[str, int, int]]:
    """格子点を並べる (基準条件を先頭に、残りは値の順)。"""
    keys = sorted({_condition_key(row) for row in rows})
    headline = [_condition_key(row) for row in rows if row.is_headline]
    if headline:
        first = headline[0]
        keys = [first, *(key for key in keys if key != first)]
    return keys


def _condition_tick(key: tuple[str, int, int]) -> str:
    """格子点の目盛ラベル (正規化の頭3文字 / 入力窓 / 平滑化)。"""
    normalize, window, smoothing = key
    return f"{normalize[:3]}/{window}/{smoothing}"


def _protocol_rank_panel(
    axis: Axes, rows: Sequence[ProtocolSweepRow], style: StyleContext
) -> None:
    """順位の折れ線 (**印の有無で体裁を変える**、D-81)。"""
    keys = _condition_keys(rows)
    positions = np.arange(len(keys), dtype=np.float64)
    for method in ANOMALY_METHODS:
        selected = {_condition_key(row): row for row in rows if row.method == method}
        if not selected:
            continue
        reference = next(iter(selected.values())).reference_distinguishable
        mark = method_line_style(reference)
        axis.plot(
            positions,
            [selected[key].rank for key in keys],
            color=METHOD_COLORS[method],
            linestyle=mark.linestyle,
            linewidth=mark.linewidth,
            marker="o",
            markersize=mark.markersize,
            fillstyle=mark.fillstyle,
            label=method_label(method, style, mark=reference),
        )
    axis.invert_yaxis()
    # 凡例を置く余白を順位 1 の上に空ける (凡例が折れ線に重なると
    # 「どの系統に印が付いているか」が読めなくなる)。
    axis.set_ylim(len(ANOMALY_METHODS) + 0.6, -0.9)
    axis.set_yticks(range(1, len(ANOMALY_METHODS) + 1))
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [_condition_tick(key) for key in keys], rotation=90, fontsize=6
    )
    axis.set_xlabel(
        style.label(
            "プロトコル (正規化 / 入力窓 / 平滑化)",
            "protocol (normalize / input window / smoothing)",
        )
    )
    axis.set_ylabel(style.label("順位 (1 が最良)", "rank (1 = best)"))
    axis.legend(loc="upper center", fontsize=6, ncols=2)


def _protocol_reversal_panel(
    axis: Axes, rows: Sequence[ProtocolSweepRow], style: StyleContext
) -> None:
    """逆転した系統対の数 —— **延べと「両方に印がある」を並べる** (D-81)。"""
    keys = _condition_keys(rows)
    by_condition = {_condition_key(row): row for row in rows}
    positions = np.arange(len(keys), dtype=np.float64)
    total = [by_condition[key].n_discordant_pairs for key in keys]
    marked = [by_condition[key].n_discordant_pairs_distinguishable for key in keys]
    axis.bar(
        positions - 0.2,
        total,
        width=0.4,
        color="#bbbbbb",
        label=style.label(
            f"逆転した系統対 (延べ {sum(total)} 組)",
            f"reversed pairs (total {sum(total)})",
        ),
    )
    axis.bar(
        positions + 0.2,
        marked,
        width=0.4,
        color="#b2182b",
        label=style.label(
            f"うち両方に印がある ({sum(marked)} 組)",
            f"both distinguishable ({sum(marked)})",
        ),
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [_condition_tick(key) for key in keys], rotation=90, fontsize=6
    )
    axis.set_xlabel(
        style.label("プロトコル (左と同じ並び)", "protocol (same order as the left)")
    )
    axis.set_ylabel(style.label("逆転した系統対の数", "number of reversed pairs"))
    axis.legend(loc="best", fontsize=7)


def build_protocol_sensitivity_figure(
    rows: Sequence[ProtocolSweepRow], *, style: StyleContext
) -> Figure:
    """5-C の図を組み立てる (**保存せずに返す**。印を検査できるようにするため)。

    ``plot_protocol_sensitivity`` はこの関数の結果を保存するだけである。
    図の中身 (線種・線幅・凡例) を機械検査したいのは D-81 が「印を必ず
    可視化する」を約束しているからで、PNG からは検査できない。

    Raises:
        ValueError: 行が空の場合。
    """
    require_rows(rows)
    figure = new_figure(12.0, 6.4)
    axes = np.atleast_1d(figure.subplots(1, 2))
    _protocol_rank_panel(axes[0], rows, style)
    _protocol_reversal_panel(axes[1], rows, style)
    changed = len({_condition_key(row) for row in rows if row.rank_changed})
    figure.suptitle(
        style.label(
            f"実験 5-C: プロトコルを変えると順位は動くか "
            f"({len(_condition_keys(rows))} 格子点中 {changed} で変動。"
            "太い実線 = 一様乱数対照と区別できる系統)",
            f"Experiment 5-C: rank changes across protocols "
            f"({changed} of {len(_condition_keys(rows))} grid points changed;"
            " thick solid = distinguishable from the random control)",
        )
    )
    return figure


def plot_protocol_sensitivity(
    rows: Sequence[ProtocolSweepRow], path: Path, *, style: StyleContext
) -> Path:
    """実験 5-C: 前処理・プロトコル別の順位の変動と印 (D-78 / D-81)。

    Raises:
        ValueError: 行が空の場合。
    """
    with rc_context_for(style):
        return save_png(build_protocol_sensitivity_figure(rows, style=style), path)


def plot_size_vs_performance(
    rows: Sequence[SizeSweepRow], path: Path, *, style: StyleContext
) -> Path:
    """実験 5-D: リザバーサイズ N と性能。

    系統ごとに ``auprc_mean`` を N に対して描き、劣化点 (基準 N の
    ``DEGRADATION_FRACTION`` 倍を初めて割る N) を縦線で示す。**N に依存しない
    系統 (対照を含む) も描く** (D-61) —— 「対照は N で動かない」ことが図で
    確かめられる。印の体裁は 5-C と同じ (D-81)。

    Raises:
        ValueError: 行が空の場合。
    """
    require_rows(rows)
    summary = summarize_size_sweep(rows)
    grid = sorted({row.n_units for row in rows})
    with rc_context_for(style):
        figure = new_figure(8.4, 5.4)
        axis = figure.subplots(1, 1)
        for method in _methods_in(rows):
            selected = {row.n_units: row for row in rows if row.method == method}
            reference = selected[summary.reference_n_units]
            mark = method_line_style(reference.distinguishable)
            axis.errorbar(
                grid,
                [selected[n_units].auprc_mean for n_units in grid],
                yerr=[selected[n_units].auprc_sd for n_units in grid],
                color=METHOD_COLORS[method],
                linestyle=mark.linestyle,
                linewidth=mark.linewidth,
                marker="o",
                markersize=mark.markersize,
                fillstyle=mark.fillstyle,
                capsize=3,
                label=method_label(method, style, mark=reference.distinguishable),
            )
        axis.axhline(
            DEGRADATION_FRACTION * summary.reference_auprc,
            color=METHOD_COLORS[ESN_RESIDUAL],
            linestyle=":",
            linewidth=1.2,
            label=style.label(
                f"基準 N={summary.reference_n_units} の{DEGRADATION_FRACTION:.0%} 水準",
                f"{DEGRADATION_FRACTION:.0%} of the reference"
                f" (N={summary.reference_n_units})",
            ),
        )
        axis.axvline(
            summary.n_units_at_90pct,
            color="#333333",
            linestyle="--",
            linewidth=1.0,
            label=style.label(
                f"劣化点 N={summary.n_units_at_90pct}"
                + (" (格子の下端)" if summary.saturated else ""),
                f"degradation point N={summary.n_units_at_90pct}"
                + (" (grid lower end)" if summary.saturated else ""),
            ),
        )
        axis.set_xscale("log")
        axis.set_xticks(grid)
        axis.set_xticklabels([str(n_units) for n_units in grid])
        axis.set_xlabel(style.label("リザバーのユニット数 N", "reservoir size N"))
        axis.set_ylabel(style.label("AUPRC (平均±標準偏差)", "AUPRC (mean +- s.d.)"))
        axis.legend(loc="best", fontsize=7)
        figure.suptitle(
            style.label(
                "実験 5-D: リザバーサイズと検知性能",
                "Experiment 5-D: reservoir size vs detection performance",
            )
        )
        return save_png(figure, path)


__all__ = [
    "build_protocol_sensitivity_figure",
    "plot_protocol_sensitivity",
    "plot_size_vs_performance",
]

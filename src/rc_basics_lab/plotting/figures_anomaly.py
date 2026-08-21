"""記事05の図5枚 (実験 5-A / 5-B / 5-C / 5-D).

- ``plot_pr_curves``: **記事の主図**。6系統の PR 曲線と AUPRC。一様乱数対照の
  水準 (= 異常率) を基準線として必ず描く (D-61)。
- ``plot_score_timeline``: 時系列上の異常スコアと正解ラベルの重ね描き (1例)。
- ``plot_threshold_tradeoff``: 5-B。警報予算を振ったときの再現率と F1、および
  較正区間だけで決めた運用点 (D-56)。
- ``plot_protocol_sensitivity``: 5-C。プロトコル別の順位の変動と、**一様乱数
  対照と区別できるかの印** (D-78)。
- ``plot_size_vs_performance``: 5-D。リザバーサイズ N と AUPRC、劣化点。

``figures.py`` / ``figures_esp.py`` / ``figures_capacity.py`` /
``figures_freerun.py`` と同じ規律に従う: pyplot を使わず ``Figure`` +
``FigureCanvasAgg`` を直接組み (``style.new_figure`` / ``style.save_png``)、
描画設定は ``style.rc_context_for`` で描画中だけ一時適用する (F-1-008)。
ラベルは必ず ``style.label(ja, en)`` を通す (D-10)。

**図は成果物 CSV の行だけを読む** (仕様 §5 禁止する構造7)。実験も診断も
ここでは1回も走らせない。行から数を作る集計 (``aggregate_methods`` /
``summarize_size_sweep``) は実験層の純関数をそのまま使う ——
``figures.py`` が ``aggregate_nrmse`` を使うのと同じ形で、図の中に2つ目の
集計実装が生えるのを防ぐ (D-53 が許可している向きの辺)。

**印を描かない図を作らない** (D-81)。5-C / 5-D の図では、一様乱数対照と
区別できる系統 (``distinguishable``) を太い実線と塗りつぶしマーカーで、
区別できない系統を細い破線と白抜きマーカーで描き、凡例にもその区別を書く。
印を描かないと「27格子点中21で順位が動いた」が「プロトコルに敏感」と読まれる
が、実測では**逆転はすべて対照と区別できない系統の内部**で起きている。

**ギリシャ文字は書かない**: ruff の RUF001/RUF002 が ASCII と紛らわしい文字を
弾くため、ソース中では ``rho`` と綴る (02〜04 の図と同じ)。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from rc_basics_lab.experiment.anomaly_ranking import (
    MethodAggregate,
    aggregate_methods,
)
from rc_basics_lab.experiment.anomaly_rows import (
    AnomalyRow,
    ProtocolSweepRow,
    SizeSweepRow,
    ThresholdSweepRow,
    TimelineRow,
)
from rc_basics_lab.experiment.anomaly_score import (
    ANOMALY_METHODS,
    DELAY_LINE_RESIDUAL,
    ESN_RESIDUAL,
    INPUT_NORM_CONTROL,
    MOVING_STATISTICS,
    PERSISTENCE_RESIDUAL,
    RANDOM_CONTROL,
)
from rc_basics_lab.experiment.anomaly_sweep import (
    DEGRADATION_FRACTION,
    summarize_size_sweep,
)
from rc_basics_lab.plotting.style import (
    StyleContext,
    new_figure,
    rc_context_for,
    require_rows,
    save_png,
)

METHOD_LABELS: dict[str, tuple[str, str]] = {
    ESN_RESIDUAL: ("ESN 残差", "ESN residual"),
    DELAY_LINE_RESIDUAL: ("遅延線 残差", "delay-line residual"),
    PERSISTENCE_RESIDUAL: ("直前値 残差", "persistence residual"),
    MOVING_STATISTICS: ("移動統計", "moving statistics"),
    RANDOM_CONTROL: ("一様乱数 (対照)", "uniform random (control)"),
    INPUT_NORM_CONTROL: ("入力ノルム (対照)", "input norm (control)"),
}
"""系統名の表示 (``ANOMALY_METHODS`` の全要素を持つ。欠けたら描く前に落とす)。"""

METHOD_COLORS: dict[str, str] = {
    ESN_RESIDUAL: "#1a9850",
    DELAY_LINE_RESIDUAL: "#2166ac",
    PERSISTENCE_RESIDUAL: "#8073ac",
    MOVING_STATISTICS: "#e08214",
    RANDOM_CONTROL: "#b2182b",
    INPUT_NORM_CONTROL: "#777777",
}
"""系統の色 (対照2本は赤と灰。基準線として目に入る色にする)。"""

MARKED_SUFFIX: tuple[str, str] = ("対照と区別できる", "distinguishable from control")
UNMARKED_SUFFIX: tuple[str, str] = ("対照と区別できない", "not distinguishable")
"""凡例に付ける印の文言 (D-78 / D-81)。**印の有無で必ず文言が変わる**。"""


@dataclass(frozen=True, slots=True)
class MarkStyle:
    """印の有無で変える線の体裁 (D-81)。

    Attributes:
        linestyle: 線種 (印あり = 実線 / 印なし = 破線)。
        linewidth: 線幅。
        markersize: マーカーの大きさ。
        fillstyle: マーカーの塗り (印なしは ``"none"`` = 白抜き)。
    """

    linestyle: str
    linewidth: float
    markersize: float
    fillstyle: str


MARKED_STYLE = MarkStyle(linestyle="-", linewidth=2.2, markersize=6.0, fillstyle="full")
UNMARKED_STYLE = MarkStyle(
    linestyle="--", linewidth=1.0, markersize=4.0, fillstyle="none"
)


def method_line_style(distinguishable: bool) -> MarkStyle:
    """印の有無から線の体裁を選ぶ (**この関数以外に体裁を決める場所を作らない**)。

    印を無視して全系統を同じ体裁で描く実装に変異させると、
    ``tests/test_plotting_anomaly.py::
    test_the_protocol_figure_marks_the_methods_that_are_distinguishable`` が
    線種・線幅・凡例のすべてで落ちる (D-81)。
    """
    return MARKED_STYLE if distinguishable else UNMARKED_STYLE


def method_label(method: str, style: StyleContext, *, mark: bool | None = None) -> str:
    """系統のラベル。``mark`` を渡すと印の文言を必ず添える (D-81)。

    Raises:
        ValueError: 対応表に無い系統名の場合 (図から静かに消さない、D-10)。
    """
    if method not in METHOD_LABELS:
        raise ValueError(f"ラベルの対応表にありません: {method!r}")
    japanese, english = METHOD_LABELS[method]
    name = style.label(japanese, english)
    if mark is None:
        return name
    suffix = MARKED_SUFFIX if mark else UNMARKED_SUFFIX
    return f"{name} ({style.label(*suffix)})"


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("値が空です")
    return float(np.mean(array))


def _methods_in(rows: Sequence[AnomalyRow] | Sequence[SizeSweepRow]) -> list[str]:
    """行に現れる系統を ``ANOMALY_METHODS`` の順で返す。"""
    found = {row.method for row in rows}
    return [method for method in ANOMALY_METHODS if method in found]


def _legend_with_auprc(
    method: str, aggregate: MethodAggregate, style: StyleContext
) -> str:
    """凡例に AUPRC の平均±標準偏差と印を出す (図だけで数が読めるようにする)。"""
    label = method_label(method, style, mark=aggregate.distinguishable)
    return f"{label}: AUPRC {aggregate.auprc_mean:.3f} +- {aggregate.auprc_sd:.3f}"


def plot_pr_curves(
    rows: Sequence[AnomalyRow],
    threshold_rows: Sequence[ThresholdSweepRow],
    path: Path,
    *,
    style: StyleContext,
) -> Path:
    """**記事の主図**: 6系統の PR 曲線と AUPRC (5-A / 5-B)。

    曲線は 5-B の掃引行 (警報予算ごとの適合率・再現率) を系列 x レプリケートで
    平均したもの、AUPRC は 5-A の行の平均±標準偏差である。**一様乱数対照の
    水準 (= 異常率) を水平線で必ず描く** (D-61) —— PR 曲線の「良さ」は
    異常率に対する相対でしか読めない。

    Raises:
        ValueError: どちらかの行が空の場合。
    """
    require_rows(rows)
    require_rows(threshold_rows)
    aggregates = aggregate_methods(rows)
    anomaly_rate = _mean(row.anomaly_rate for row in rows)
    with rc_context_for(style):
        figure = new_figure(8.0, 5.6)
        axis = figure.subplots(1, 1)
        for method in _methods_in(rows):
            selected = [row for row in threshold_rows if row.method == method]
            budgets = sorted({row.target_false_alarm_rate for row in selected})
            points = [
                (
                    _mean(
                        row.recall
                        for row in selected
                        if row.target_false_alarm_rate == budget
                    ),
                    _mean(
                        row.precision
                        for row in selected
                        if row.target_false_alarm_rate == budget
                    ),
                )
                for budget in budgets
            ]
            points.sort()
            mark = method_line_style(aggregates[method].distinguishable)
            axis.plot(
                [recall for recall, _ in points],
                [precision for _, precision in points],
                color=METHOD_COLORS[method],
                linestyle=mark.linestyle,
                linewidth=mark.linewidth,
                marker="o",
                markersize=mark.markersize,
                fillstyle=mark.fillstyle,
                label=_legend_with_auprc(method, aggregates[method], style),
            )
        axis.axhline(
            anomaly_rate,
            color="#333333",
            linestyle=":",
            linewidth=1.2,
            label=style.label(
                f"異常率 {anomaly_rate:.3f} (一様乱数の期待値)",
                f"anomaly rate {anomaly_rate:.3f} (expected for random scores)",
            ),
        )
        axis.set_xlabel(style.label("再現率", "recall"))
        axis.set_ylabel(style.label("適合率", "precision"))
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, None)
        axis.legend(loc="best", fontsize=8)
        figure.suptitle(
            style.label(
                "実験 5-A: 検知性能の比較 (AUPRC は point-adjust を通していない)",
                "Experiment 5-A: detection performance (AUPRC, no point adjustment)",
            )
        )
        return save_png(figure, path)


def _timeline_series(
    rows: Sequence[TimelineRow], method: str
) -> tuple[list[int], list[float]]:
    selected = sorted(
        (row for row in rows if row.method == method), key=lambda row: row.index
    )
    return [row.index for row in selected], [row.score for row in selected]


def _anomaly_spans(rows: Sequence[TimelineRow]) -> list[tuple[int, int]]:
    """正解ラベルの連続区間を ``(開始 index, 終了 index)`` にする。

    行は間引かれているので、区間の端は**間引き後の点**である (図の帯は
    実際の異常区間よりわずかに狭い/広い)。
    """
    ordered = sorted({(row.index, row.is_anomaly) for row in rows})
    spans: list[tuple[int, int]] = []
    start: int | None = None
    previous = ordered[0][0] if ordered else 0
    for index, is_anomaly in ordered:
        if is_anomaly and start is None:
            start = index
        if not is_anomaly and start is not None:
            spans.append((start, previous))
            start = None
        previous = index
    if start is not None:
        spans.append((start, previous))
    return spans


def plot_score_timeline(
    rows: Sequence[TimelineRow], path: Path, *, style: StyleContext
) -> Path:
    """時系列上の異常スコアと正解ラベルの重ね描き (1例)。

    ``anomaly_timeline.csv`` の行だけを読む。6系統を縦に並べ、正解ラベルの
    区間を帯で、較正区間から決めた運用閾値 (D-56) を水平線で重ねる ——
    「対照でも警報は出るが、正解の帯に当たらない」ことが1枚で読める。

    Raises:
        ValueError: 行が空の場合。
    """
    require_rows(rows)
    methods = [
        method
        for method in ANOMALY_METHODS
        if any(row.method == method for row in rows)
    ]
    spans = _anomaly_spans(rows)
    with rc_context_for(style):
        figure = new_figure(9.0, 1.5 * len(methods) + 1.2)
        axes = np.atleast_1d(figure.subplots(len(methods), 1, sharex=True))
        for axis, method in zip(axes, methods, strict=True):
            indices, scores = _timeline_series(rows, method)
            axis.plot(
                indices,
                scores,
                color=METHOD_COLORS[method],
                linewidth=0.8,
                label=method_label(method, style),
            )
            threshold = next(row.threshold for row in rows if row.method == method)
            axis.axhline(
                threshold,
                color="#333333",
                linestyle="--",
                linewidth=1.0,
                label=style.label("運用閾値 (較正区間)", "operating threshold"),
            )
            for first, last in spans:
                axis.axvspan(first, last, color="#b2182b", alpha=0.18, linewidth=0)
            axis.set_ylabel(method_label(method, style), fontsize=8)
            axis.legend(loc="upper right", fontsize=7, ncols=2)
        axes[-1].set_xlabel(style.label("系列上の位置 [点]", "index in the series"))
        example = rows[0]
        figure.suptitle(
            style.label(
                f"実験 5-A: 異常スコアと正解ラベル ({example.dataset} / 系列 "
                f"{example.series} / レプリケート {example.replicate}。"
                "帯が正解の異常区間)",
                f"Experiment 5-A: anomaly scores vs labels ({example.dataset} /"
                f" series {example.series} / replicate {example.replicate};"
                " shaded = true anomalies)",
            )
        )
        return save_png(figure, path)


def _budget_curve(
    rows: Sequence[ThresholdSweepRow], method: str, attribute: str
) -> tuple[list[float], list[float]]:
    selected = [row for row in rows if row.method == method]
    budgets = sorted({row.target_false_alarm_rate for row in selected})
    values = [
        _mean(
            float(getattr(row, attribute))
            for row in selected
            if row.target_false_alarm_rate == budget
        )
        for budget in budgets
    ]
    return budgets, values


def plot_threshold_tradeoff(
    rows: Sequence[AnomalyRow],
    threshold_rows: Sequence[ThresholdSweepRow],
    path: Path,
    *,
    style: StyleContext,
) -> Path:
    """実験 5-B: 閾値と検知/誤報のトレードオフ。

    左は警報予算に対する再現率、右は F1。**較正区間だけで決めた運用点**
    (D-56) を実測誤報率の位置に重ねる —— 閾値をテスト側で選べばどこまで
    良く見えるかは ``f1_test_optimal`` (右パネルの点線) が示す。

    Raises:
        ValueError: どちらかの行が空の場合。
    """
    require_rows(rows)
    require_rows(threshold_rows)
    aggregates = aggregate_methods(rows)
    with rc_context_for(style):
        figure = new_figure(11.0, 5.0)
        axes = np.atleast_1d(figure.subplots(1, 2))
        for method in _methods_in(rows):
            mark = method_line_style(aggregates[method].distinguishable)
            operating = [row for row in rows if row.method == method]
            for axis, attribute, column in (
                (axes[0], "recall", "recall_calibrated"),
                (axes[1], "f1", "f1_calibrated"),
            ):
                budgets, values = _budget_curve(threshold_rows, method, attribute)
                axis.plot(
                    budgets,
                    values,
                    color=METHOD_COLORS[method],
                    linestyle=mark.linestyle,
                    linewidth=mark.linewidth,
                    marker="o",
                    markersize=mark.markersize,
                    fillstyle=mark.fillstyle,
                    label=method_label(
                        method, style, mark=aggregates[method].distinguishable
                    ),
                )
                axis.plot(
                    [_mean(row.far_test for row in operating)],
                    [_mean(float(getattr(row, column)) for row in operating)],
                    color=METHOD_COLORS[method],
                    marker="*",
                    markersize=13.0,
                    linestyle="none",
                )
        optimal = [
            row.f1_test_optimal
            for row in rows
            if not math.isnan(row.f1_test_optimal) and row.method == ESN_RESIDUAL
        ]
        if optimal:
            axes[1].axhline(
                _mean(optimal),
                color=METHOD_COLORS[ESN_RESIDUAL],
                linestyle=":",
                linewidth=1.2,
                label=style.label(
                    "ESN のテスト側最適 F1 (参考値)",
                    "ESN F1 optimized on the test split (reference only)",
                ),
            )
        for axis, ylabel_ja, ylabel_en in (
            (axes[0], "再現率", "recall"),
            (axes[1], "F1", "F1"),
        ):
            axis.set_xscale("log")
            axis.set_xlabel(
                style.label("警報予算 (較正区間の誤報率)", "alarm budget (target FAR)")
            )
            axis.set_ylabel(style.label(ylabel_ja, ylabel_en))
            axis.legend(loc="best", fontsize=7)
        figure.suptitle(
            style.label(
                "実験 5-B: 閾値のトレードオフ (星 = 較正区間だけで決めた運用点)",
                "Experiment 5-B: threshold trade-off (star = operating point"
                " calibrated without test labels)",
            )
        )
        return save_png(figure, path)


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
    axis.set_yticks(range(1, len(ANOMALY_METHODS) + 1))
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            f"{normalize[:3]}/{window}/{smoothing}"
            for normalize, window, smoothing in keys
        ],
        rotation=90,
        fontsize=6,
    )
    axis.set_xlabel(
        style.label(
            "プロトコル (正規化 / 入力窓 / 平滑化)",
            "protocol (normalize / input window / smoothing)",
        )
    )
    axis.set_ylabel(style.label("順位 (1 が最良)", "rank (1 = best)"))
    axis.legend(loc="center right", fontsize=7)


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
    axis.set_xticklabels([])
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
                + ("(格子の下端)" if summary.saturated else ""),
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
    "MARKED_STYLE",
    "MARKED_SUFFIX",
    "METHOD_COLORS",
    "METHOD_LABELS",
    "UNMARKED_STYLE",
    "UNMARKED_SUFFIX",
    "MarkStyle",
    "build_protocol_sensitivity_figure",
    "method_label",
    "method_line_style",
    "plot_pr_curves",
    "plot_protocol_sensitivity",
    "plot_score_timeline",
    "plot_size_vs_performance",
    "plot_threshold_tradeoff",
]

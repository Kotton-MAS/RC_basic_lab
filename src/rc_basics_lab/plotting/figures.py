"""記事01の図2枚.

- ``plot_comparison``: 課題ごとに3手法の NRMSE を点+誤差棒で並べる。
  **NRMSE = 1 の水平基準線**を引き、「平均予測と同等」という読み方を図の中で
  明示する (D-02 が担保している解釈)。
- ``plot_state_space``: 入力空間 (遅延埋め込み) とリザバー状態空間の PCA。
  左2列が PC1-PC2 散布図、右列が累積寄与率曲線 (``n_components_95`` を注記)。

pyplot を使わず ``Figure`` + ``FigureCanvasAgg`` を直接組む。CI にはディスプレイが
無いため、既定バックエンドに依存しない非対話経路を選ぶ。

描画設定 (``savefig.dpi`` など) はプロセス全体の ``matplotlib.rcParams`` を
書き換えず、``matplotlib.rc_context`` で描画中だけ一時適用する (F-1-008)。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from rc_basics_lab.experiment.runner import ResultRow
from rc_basics_lab.experiment.state_space import (
    DELAY_EMBEDDED_INPUT,
    RAW_INPUT,
    RESERVOIR_STATE,
    StateSpaceReport,
)
from rc_basics_lab.experiment.summary import Aggregate, aggregate_nrmse
from rc_basics_lab.plotting.labels import METHOD_LABELS
from rc_basics_lab.plotting.style import (
    DELAY_LINE_METHOD,
    ESN_METHOD,
    METHOD_COLORS,
    StyleContext,
    add_provenance,
    method_color,
    rc_context_for,
    reference_line_kwargs,
    require_rows,
)
from rc_basics_lab.plotting.style import new_figure as _new_figure
from rc_basics_lab.plotting.style import save_png as _save
from rc_basics_lab.types import FloatArray

REFERENCE_NRMSE = 1.0
"""平均予測と同等になる NRMSE (D-02)。図の水平基準線。"""

UNKNOWN_SPACE_COLOR = "#777777"
"""対応表に無い空間の色 (無彩色。データ系列の4色と重ならない)。"""

_MAX_SCATTER_POINTS = 2000
"""散布図に描く最大点数 (超えたら等間隔に間引く。PNG の肥大を避ける)。"""

_MAX_COMPONENTS_SHOWN = 40
"""累積寄与率曲線の横軸に描く主成分数の上限。"""

_N_SCATTER_AXES = 2
"""散布図に必要な主成分の本数 (PC1 と PC2)。"""

_TASK_LABELS: dict[str, tuple[str, str]] = {
    "mackey_glass": ("Mackey-Glass 1ステップ先予測", "Mackey-Glass 1-step prediction"),
    "delay_parity": (
        "遅延パリティ y[t]=u[t-1]u[t-2]",
        "Delay parity y[t]=u[t-1]u[t-2]",
    ),
}
_SPACE_LABELS: dict[str, tuple[str, str]] = {
    RAW_INPUT: ("生の入力", "raw input"),
    DELAY_EMBEDDED_INPUT: ("入力の遅延埋め込み", "delay-embedded input"),
    RESERVOIR_STATE: ("リザバー状態", "reservoir state"),
}
_SPACE_COLORS: dict[str, str] = {
    DELAY_EMBEDDED_INPUT: METHOD_COLORS[DELAY_LINE_METHOD],
    RESERVOIR_STATE: METHOD_COLORS[ESN_METHOD],
}
"""空間の色は**対応する手法の色**に合わせる (FIG-5)。

「入力の遅延埋め込み」は遅延線が使う特徴量、「リザバー状態」は ESN が使う
特徴量なので、比較図と PCA 図で同じ色にすると記事をまたいで対応が読める。
"""


def _lookup(table: dict[str, tuple[str, str]], key: str, style: StyleContext) -> str:
    """表示名を引く。未知のキーは識別子をそのまま返す (図を落とさない)。"""
    pair = table.get(key)
    if pair is None:
        return key
    return style.label(*pair)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    """出現順を保った重複除去 (図の並び順を入力の順に従わせる)。"""
    return tuple(dict.fromkeys(values))


def _compute_errorbars(
    stats: Mapping[tuple[str, str], Aggregate], task: str, methods: Sequence[str]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """1課題ぶんの誤差棒の値を計算する (平均・下側誤差・標準偏差)。"""
    means: FloatArray = np.array(
        [stats[(task, method)].mean for method in methods], dtype=np.float64
    )
    stds: FloatArray = np.array(
        [stats[(task, method)].std for method in methods], dtype=np.float64
    )
    # 対数軸なので下側の誤差棒が 0 以下に落ちないよう抑える
    lower: FloatArray = np.minimum(stds, means * 0.999)
    return means, lower, stds


def _draw_reference_line(axis: Axes, style: StyleContext) -> None:
    """NRMSE = 1 (平均予測と同等) の水平基準線を描く (D-02)。"""
    axis.axhline(
        REFERENCE_NRMSE,
        **reference_line_kwargs(),
        label=style.label(
            "NRMSE = 1 (平均予測と同等)",
            "NRMSE = 1 (same as predicting the mean)",
        ),
    )


def _annotate_means(axis: Axes, positions: FloatArray, means: FloatArray) -> None:
    """各点の上に平均値を数値で注記する。"""
    for position, mean in zip(positions, means, strict=True):
        axis.annotate(
            f"{mean:.4f}",
            (position, mean),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )


def _style_task_axis(
    axis: Axes,
    task: str,
    methods: Sequence[str],
    positions: FloatArray,
    means: FloatArray,
    lower: FloatArray,
    stds: FloatArray,
    n_replicates: int,
    style: StyleContext,
    *,
    show_ylabel: bool,
) -> None:
    """1課題ぶんの軸範囲・目盛・タイトル・凡例を設定する。"""
    axis.set_yscale("log")
    # 注記と基準線のぶんの余白 (対数軸なので係数で確保する)
    axis.set_ylim(
        float(np.min(means - lower)) / 3.0,
        max(float(np.max(means + stds)), REFERENCE_NRMSE) * 3.0,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels([_lookup(METHOD_LABELS, method, style) for method in methods])
    axis.set_xlim(-0.5, len(methods) - 0.5)
    axis.set_title(_lookup(_TASK_LABELS, task, style))
    if show_ylabel:
        axis.set_ylabel(
            style.label(
                f"NRMSE (テスト区間・{n_replicates}レプリケートの平均±標準偏差)",
                f"NRMSE (test split, mean ± s.d. of {n_replicates} replicates)",
            )
        )
    axis.legend(loc="best", fontsize=8)


def _plot_task_panel(
    axis: Axes,
    task: str,
    methods: Sequence[str],
    positions: FloatArray,
    stats: Mapping[tuple[str, str], Aggregate],
    style: StyleContext,
    *,
    show_ylabel: bool,
) -> None:
    """1課題ぶんの NRMSE 比較パネル (誤差棒+基準線+注記+軸装飾) を描く。"""
    means, lower, stds = _compute_errorbars(stats, task, methods)
    axis.errorbar(
        positions,
        means,
        yerr=np.vstack([lower, stds]),
        fmt="o",
        capsize=5,
        color="#666666",
    )
    # 点の色は手法の固定色 (FIG-5)。記事をまたいで ESN は緑、遅延線は青。
    axis.scatter(positions, means, c=[method_color(m) for m in methods], zorder=3)
    _draw_reference_line(axis, style)
    _annotate_means(axis, positions, means)
    n_replicates = max(stats[(task, method)].n for method in methods)
    _style_task_axis(
        axis,
        task,
        methods,
        positions,
        means,
        lower,
        stds,
        n_replicates,
        style,
        show_ylabel=show_ylabel,
    )


def plot_comparison(
    rows: Sequence[ResultRow], path: Path, *, style: StyleContext
) -> Path:
    """3手法の NRMSE 比較図を書く。

    Args:
        rows: ``comparison.csv`` と同じ長形式の行。
        path: 出力先 PNG。
        style: ``setup_style()`` の戻り値 (ラベル言語の決定に使う)。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    require_rows(rows)
    tasks = _unique(row.task for row in rows)
    methods = _unique(row.method for row in rows)
    stats = aggregate_nrmse(rows)

    with rc_context_for(style):
        figure = _new_figure(4.2 * len(tasks), 4.0)
        axes = figure.subplots(1, len(tasks), squeeze=False)
        positions = np.arange(len(methods), dtype=np.float64)
        for index, task in enumerate(tasks):
            _plot_task_panel(
                axes[0][index],
                task,
                methods,
                positions,
                stats,
                style,
                show_ylabel=index == 0,
            )
        figure.suptitle(
            style.label(
                "実験 1: 非線形な遅延パリティを解けるのは ESN だけ"
                " (同一分割・同一 alpha 格子)",
                "Experiment 1: only the ESN solves the nonlinear delay parity"
                " (identical splits and alpha grid)",
            )
        )
        conditions = f"n_train = {rows[0].n_train}, n_test = {rows[0].n_test}"
        add_provenance(figure, conditions, rows, style=style)
        return _save(figure, path)


def _thin(array: FloatArray) -> FloatArray:
    """散布図用に行を等間隔で間引く。"""
    if array.shape[0] <= _MAX_SCATTER_POINTS:
        return array
    stride = array.shape[0] // _MAX_SCATTER_POINTS + 1
    thinned: FloatArray = array[::stride]
    return thinned


def _scatter_space(
    axis: Axes, report: StateSpaceReport, space: str, style: StyleContext
) -> None:
    """1つの空間の PC1-PC2 散布図を描く。"""
    summary = report.space(space)
    scores = _thin(summary.pc_scores)
    if scores.shape[1] < _N_SCATTER_AXES:
        raise ValueError(f"{space} は PC2 が無いため散布図にできません")
    axis.scatter(
        scores[:, 0],
        scores[:, 1],
        s=3,
        alpha=0.3,
        color=_SPACE_COLORS.get(space, UNKNOWN_SPACE_COLOR),
    )
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title(
        f"{_lookup(_SPACE_LABELS, space, style)}\n"
        + style.label(
            f"{summary.n_features} 次元 / 95%: {summary.n_components_95} 主成分",
            f"{summary.n_features}-dim / 95%: {summary.n_components_95} PCs",
        ),
        fontsize=9,
    )


def _plot_cumulative_ratio_panel(
    axis: Axes, report: StateSpaceReport, style: StyleContext
) -> None:
    """1課題ぶんの累積寄与率パネル (曲線+95%基準線+軸装飾) を描く。"""
    for space in (RESERVOIR_STATE, DELAY_EMBEDDED_INPUT):
        summary = report.space(space)
        curve = summary.cumulative_ratio[:_MAX_COMPONENTS_SHOWN]
        components = np.arange(1, len(curve) + 1)
        axis.plot(
            components,
            curve,
            marker="o",
            markersize=3,
            color=_SPACE_COLORS[space],
            label=(
                f"{_lookup(_SPACE_LABELS, space, style)} "
                f"(95%: {summary.n_components_95})"
            ),
        )
        axis.axvline(
            summary.n_components_95,
            color=_SPACE_COLORS[space],
            linestyle=":",
            linewidth=1.0,
        )
    axis.axhline(0.95, **reference_line_kwargs())
    axis.set_xlabel(style.label("主成分の数", "number of components"))
    axis.set_ylabel(style.label("累積寄与率", "cumulative explained variance"))
    raw = report.space(RAW_INPUT)
    axis.set_title(
        style.label(
            f"{_lookup(_TASK_LABELS, report.task, style)}"
            f" (生の入力は {raw.n_features} 次元)",
            f"{_lookup(_TASK_LABELS, report.task, style)}"
            f" (raw input is {raw.n_features}-dim)",
        ),
        fontsize=9,
    )
    axis.set_ylim(0.0, 1.02)
    axis.legend(loc="lower right", fontsize=8)


def plot_state_space(
    reports: Sequence[StateSpaceReport], path: Path, *, style: StyleContext
) -> Path:
    """入力空間とリザバー状態空間の PCA 図を書く (受け入れ条件4)。

    行が課題、左2列が PC1-PC2 散布図 (リザバー状態 / 入力の遅延埋め込み)、
    右列が累積寄与率曲線。``n_components_95`` は散布図のタイトルと
    累積寄与率曲線の注記の両方に数値で出る。

    Raises:
        ValueError: ``reports`` が空の場合。
    """
    if not reports:
        raise ValueError("reports が空です")
    with rc_context_for(style):
        figure = _new_figure(12.0, 4.0 * len(reports))
        axes = figure.subplots(len(reports), 3, squeeze=False)
        for index, report in enumerate(reports):
            _scatter_space(axes[index][0], report, RESERVOIR_STATE, style)
            _scatter_space(axes[index][1], report, DELAY_EMBEDDED_INPUT, style)
            _plot_cumulative_ratio_panel(axes[index][2], report, style)
        figure.suptitle(
            style.label(
                "実験 1: リザバー状態は入力の遅延埋め込みより高い次元へ広がる",
                "Experiment 1: the reservoir state spans more dimensions"
                " than the delay embedding of the input",
            )
        )
        conditions = f"n_rows = {reports[0].n_rows}, n_lags = {reports[0].n_lags}"
        add_provenance(figure, conditions, reports, style=style)
        return _save(figure, path)


__all__ = [
    "REFERENCE_NRMSE",
    "aggregate_nrmse",
    "plot_comparison",
    "plot_state_space",
]

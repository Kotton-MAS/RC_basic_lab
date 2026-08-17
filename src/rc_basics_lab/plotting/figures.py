"""記事01の図2枚.

- ``plot_comparison``: 課題ごとに3手法の NRMSE を点+誤差棒で並べる。
  **NRMSE = 1 の水平基準線**を引き、「平均予測と同等」という読み方を図の中で
  明示する (D-02 が担保している解釈)。
- ``plot_state_space``: 入力空間 (遅延埋め込み) とリザバー状態空間の PCA。
  左2列が PC1-PC2 散布図、右列が累積寄与率曲線 (``n_components_95`` を注記)。

pyplot を使わず ``Figure`` + ``FigureCanvasAgg`` を直接組む。CI にはディスプレイが
無いため、既定バックエンドに依存しない非対話経路を選ぶ。
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from rc_basics_lab.experiment.runner import ResultRow
from rc_basics_lab.experiment.state_space import (
    DELAY_EMBEDDED_INPUT,
    RAW_INPUT,
    RESERVOIR_STATE,
    StateSpaceReport,
)
from rc_basics_lab.plotting.style import StyleContext
from rc_basics_lab.types import FloatArray

REFERENCE_NRMSE = 1.0
"""平均予測と同等になる NRMSE (D-02)。図の水平基準線。"""

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
_METHOD_LABELS: dict[str, tuple[str, str]] = {
    "linear": ("線形", "linear"),
    "delay_line": ("遅延線", "delay line"),
    "esn": ("ESN", "ESN"),
}
_SPACE_LABELS: dict[str, tuple[str, str]] = {
    RAW_INPUT: ("生の入力", "raw input"),
    DELAY_EMBEDDED_INPUT: ("入力の遅延埋め込み", "delay-embedded input"),
    RESERVOIR_STATE: ("リザバー状態", "reservoir state"),
}
_SPACE_COLORS: dict[str, str] = {
    DELAY_EMBEDDED_INPUT: "tab:orange",
    RESERVOIR_STATE: "tab:blue",
}


def _lookup(table: dict[str, tuple[str, str]], key: str, style: StyleContext) -> str:
    """表示名を引く。未知のキーは識別子をそのまま返す (図を落とさない)。"""
    pair = table.get(key)
    if pair is None:
        return key
    return style.label(*pair)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    """出現順を保った重複除去 (図の並び順を入力の順に従わせる)。"""
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True, slots=True)
class _Aggregate:
    """1 (課題, 手法) の集計値。"""

    mean: float
    std: float
    n: int


def aggregate_nrmse(rows: Sequence[ResultRow]) -> dict[tuple[str, str], _Aggregate]:
    """(課題, 手法) ごとの NRMSE の平均と標準偏差 (受け入れ条件3)。

    標準偏差は標本標準偏差 (ddof=1)。レプリケートが1本のときは 0 とする。
    """
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row.task, row.method), []).append(row.nrmse)
    return {
        key: _Aggregate(
            mean=statistics.fmean(values),
            std=statistics.stdev(values) if len(values) > 1 else 0.0,
            n=len(values),
        )
        for key, values in grouped.items()
    }


def _new_figure(width: float, height: float) -> Figure:
    """constrained layout の Figure を作る (軸ラベルとタイトルの重なりを防ぐ)。"""
    figure = Figure(figsize=(width, height))
    figure.set_layout_engine("constrained")
    return figure


def _save(figure: Figure, path: Path) -> Path:
    """Agg キャンバスで PNG を書く (ディスプレイに依存しない)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)
    figure.savefig(path, format="png")
    return path


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
    if not rows:
        raise ValueError("rows が空です")
    tasks = _unique(row.task for row in rows)
    methods = _unique(row.method for row in rows)
    stats = aggregate_nrmse(rows)

    figure = Figure(figsize=(4.2 * len(tasks), 4.0))
    axes = figure.subplots(1, len(tasks), squeeze=False)
    positions = np.arange(len(methods), dtype=np.float64)
    for index, task in enumerate(tasks):
        axis = axes[0][index]
        means = np.array(
            [stats[(task, method)].mean for method in methods], dtype=np.float64
        )
        stds = np.array(
            [stats[(task, method)].std for method in methods], dtype=np.float64
        )
        # 対数軸なので下側の誤差棒が 0 以下に落ちないよう抑える
        lower = np.minimum(stds, means * 0.999)
        axis.errorbar(
            positions,
            means,
            yerr=np.vstack([lower, stds]),
            fmt="o",
            capsize=5,
            color="tab:blue",
        )
        axis.axhline(
            REFERENCE_NRMSE,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label=style.label(
                "NRMSE = 1 (平均予測と同等)",
                "NRMSE = 1 (same as predicting the mean)",
            ),
        )
        for position, mean in zip(positions, means, strict=True):
            axis.annotate(
                f"{mean:.4f}",
                (position, mean),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )
        axis.set_yscale("log")
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [_lookup(_METHOD_LABELS, method, style) for method in methods]
        )
        axis.set_xlim(-0.5, len(methods) - 0.5)
        axis.set_title(_lookup(_TASK_LABELS, task, style))
        if index == 0:
            n_replicates = max(stats[(task, method)].n for method in methods)
            axis.set_ylabel(
                style.label(
                    f"NRMSE (テスト区間・{n_replicates}レプリケートの平均±標準偏差)",
                    f"NRMSE (test split, mean ± s.d. of {n_replicates} replicates)",
                )
            )
        axis.legend(loc="best", fontsize=8)
    figure.suptitle(
        style.label(
            "3ベースラインの比較 (同一分割・同一 alpha 格子)",
            "Three baselines (identical splits and alpha grid)",
        )
    )
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
        color=_SPACE_COLORS.get(space, "tab:gray"),
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
    figure = Figure(figsize=(12.0, 3.6 * len(reports)))
    axes = figure.subplots(len(reports), 3, squeeze=False)
    for index, report in enumerate(reports):
        _scatter_space(axes[index][0], report, RESERVOIR_STATE, style)
        _scatter_space(axes[index][1], report, DELAY_EMBEDDED_INPUT, style)
        axis = axes[index][2]
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
        axis.axhline(0.95, color="tab:red", linestyle="--", linewidth=1.0)
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
    figure.suptitle(
        style.label(
            "入力空間とリザバー状態空間の PCA",
            "PCA of the input space and the reservoir state space",
        )
    )
    return _save(figure, path)


__all__ = [
    "REFERENCE_NRMSE",
    "aggregate_nrmse",
    "plot_comparison",
    "plot_state_space",
]

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

from rc_basics_lab.experiment.horizon import HorizonRow
from rc_basics_lab.experiment.runner import ResultRow
from rc_basics_lab.experiment.state_space import (
    DELAY_EMBEDDED_INPUT,
    RAW_INPUT,
    RESERVOIR_STATE,
    StateSpaceReport,
)
from rc_basics_lab.experiment.summary import Aggregate, aggregate_nrmse
from rc_basics_lab.experiment.waveform_data import WaveformPanel
from rc_basics_lab.plotting.labels import METHOD_LABELS
from rc_basics_lab.plotting.layout import label_panels
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
from rc_basics_lab.plotting.waveforms import WAVEFORM_OFFSET
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


_SHORT_TASK_LABELS: dict[str, tuple[str, str]] = {
    "mackey_glass": ("Mackey-Glass", "Mackey-Glass"),
    "delay_parity": ("遅延パリティ", "delay parity"),
}
"""パネル見出し用の短い課題名。

``_TASK_LABELS`` は式まで含むので、結論文と並べると見出しが軸幅を超える
(実測: 遅延パリティのパネルで隣まではみ出した)。**軸ラベルは式つき、
見出しは短い名前**と使い分ける。
"""

_TASK_MARKERS: tuple[str, ...] = ("o", "s", "^", "D")
"""課題を分けるマーカー。**色は手法が持っている** (FIG-5) ので形で分ける。"""

HORIZON_TASK_LABEL: tuple[str, str] = ("Mackey-Glass", "Mackey-Glass")
"""波形パネルに描く課題の表示名 (``HORIZON_TASK`` に対応)。"""


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


def _draw_scalar_panel(
    axis: Axes,
    tasks: Sequence[str],
    methods: Sequence[str],
    stats: Mapping[tuple[str, str], Aggregate],
    style: StyleContext,
) -> None:
    """全課題の NRMSE を **1 つの軸に**まとめる (FIG-12)。

    かつては課題ごとに1パネルだった。**点は課題あたり 3 個しかなく、両方の
    パネルに同じ基準線と同じ凡例が2回出ていた。** 課題は横位置のずらしで
    分ける (色は手法の固定色 FIG-5 のままにする)。

    Args:
        axis: 描画先。
        tasks: 課題名の並び。
        methods: 手法名の並び。
        stats: ``aggregate_nrmse`` の出力。
        style: 配色・言語。
    """
    positions = np.arange(len(methods), dtype=np.float64)
    lowest = float("inf")
    highest = 0.0
    for index, task in enumerate(tasks):
        means, lower, stds = _compute_errorbars(stats, task, methods)
        shift = 0.16 * (index - 0.5 * (len(tasks) - 1))
        axis.errorbar(
            positions + shift,
            means,
            yerr=np.vstack([lower, stds]),
            fmt="none",
            capsize=5,
            color="#666666",
        )
        # 課題はマーカーの形で分ける。色は手法の固定色 (FIG-5) が持っている。
        marker = _TASK_MARKERS[index % len(_TASK_MARKERS)]
        axis.scatter(
            positions + shift,
            means,
            c=[method_color(method) for method in methods],
            marker=marker,
            s=70,
            zorder=3,
        )
        # 凡例は**無彩色の代理**にする。実点の色をそのまま凡例に出すと、
        # 先頭の手法の色が「その課題の色」に見えてしまう (色は手法の意味)。
        axis.scatter(
            [],
            [],
            c="#666666",
            marker=marker,
            s=70,
            label=_lookup(_TASK_LABELS, task, style),
        )
        lowest = min(lowest, float(np.min(means - lower)))
        highest = max(highest, float(np.max(means + stds)))
    _draw_reference_line(axis, style)
    axis.set_yscale("log")
    axis.set_ylim(lowest / 3.0, max(highest, REFERENCE_NRMSE) * 3.0)
    axis.set_xticks(positions)
    axis.set_xticklabels([_lookup(METHOD_LABELS, method, style) for method in methods])
    axis.set_xlim(-0.5, len(methods) - 0.5)
    n_replicates = max(stats[(tasks[0], method)].n for method in methods)
    axis.set_ylabel(
        style.label(
            f"NRMSE (テスト区間・{n_replicates}レプリケートの平均±標準偏差)",
            f"NRMSE (test split, mean ± s.d. of {n_replicates} replicates)",
        )
    )
    axis.set_title(style.label("課題別の誤差", "error by task"))
    axis.legend(loc="best", fontsize=8)


def plot_comparison(
    rows: Sequence[ResultRow],
    path: Path,
    *,
    waveforms: Sequence[WaveformPanel],
    horizon_rows: Sequence[HorizonRow],
    style: StyleContext,
) -> Path:
    """**01 の主図**: 手法の誤差・予測波形・自走 84 ステップ先を1枚に並べる。

    かつては3枚の figure だった (``fig_comparison`` / ``fig_waveform`` /
    ``fig_horizon``)。FIG-12 により1枚へ畳んである —— スカラー比較は
    課題あたり 3 点、自走は 5 点しかなく、**単独の figure では面積の大半が
    空白**で、``fig_horizon`` では凡例が参照線と重なって読めなかった。

    3 つは「1ステップ先では差が小さい → 波形でもほぼ重なる → 自走させると
    桁で開く」という**一続きの主張**なので、並べると往復が要らなくなる。

    Args:
        rows: ``comparison.csv`` と同じ長形式の行。
        path: 出力先 PNG。
        waveforms: 課題ごとの波形パネル。**2 課題とも渡す** ——
            片方だけだと「差が見えない側」しか載らない (FIG-11 追加図2)。
        horizon_rows: ``horizon.csv`` と同じ行。右のパネルに使う。
        style: ``setup_style()`` の戻り値 (ラベル言語の決定に使う)。

    Returns:
        書き出した PNG のパス。

    Raises:
        ValueError: ``rows`` が空、または ``waveforms`` が空の場合。
    """
    from rc_basics_lab.plotting.figures_horizon import (
        draw_horizon_panel,
        horizon_headline,
        horizon_reference_note,
    )
    from rc_basics_lab.plotting.waveforms import (
        draw_prediction_waveform,
        waveform_headline,
    )

    require_rows(rows)
    if not waveforms:
        raise ValueError("waveforms が空です (課題ごとに1枚必要です)")
    tasks = _unique(row.task for row in rows)
    methods = _unique(row.method for row in rows)
    stats = aggregate_nrmse(rows)

    with rc_context_for(style):
        # **1段に並べない**。パネル4枚を横一列にすると 3.57 : 1 で D-108 の
        # 上限を超える。2段に折ると 1.44 : 1 に収まる。
        figure = _new_figure(13.0, 9.0)
        axes = np.atleast_1d(figure.subplots(2, 2)).reshape(-1)
        label_panels(list(axes), style=style)
        _draw_scalar_panel(axes[0], tasks, methods, stats, style)
        # **課題ごとに長さが違う** (D-107)。最後のパネルの長さだけを脚注に
        # 書くと、もう一方のパネルの条件を偽って書くことになる。
        windows: list[str] = []
        for index, panel in enumerate(waveforms):
            axis = axes[1 + index]
            drawn = draw_prediction_waveform(
                figure, axis, panel.truth, panel.predictions, style
            )
            windows.append(
                f"{panel.task} {WAVEFORM_OFFSET}..{WAVEFORM_OFFSET + drawn.length}"
            )
            label = _lookup(_SHORT_TASK_LABELS, panel.task, style)
            # 見出しは**残差から導く** (FIG-1 / C-1)。「どう見えるか」は
            # 疑問形で、図が何を示したかを言っていない。
            drawn.top.set_title(
                f"{label}: {waveform_headline(panel.truth, panel.predictions, style)}",
                fontsize=9,
            )
        logs = draw_horizon_panel(axes[1 + len(waveforms)], horizon_rows, style)
        axes[1 + len(waveforms)].set_title(
            style.label("自走 84 ステップ先の誤差", "error 84 steps into the free run")
        )
        # 余った枠は消す。空の軸を残すと「測ったが何も無かった」に見える。
        for axis in axes[2 + len(waveforms) :]:
            axis.set_axis_off()
        figure.suptitle(
            style.label(
                "実験 1: 非線形な遅延パリティを解けるのは ESN だけ"
                " (同一分割・同一 alpha 格子)",
                "Experiment 1: only the ESN solves the nonlinear delay parity"
                " (identical splits and alpha grid)",
            )
        )
        # 結論文と出典は**注記に集約する**。パネル見出しに入れると、
        # 幅が軸を超えて隣のパネルへかぶった (実測: FIG-14 と同じ症状)。
        figure.supxlabel(
            style.label(
                f"自走のパネル: {horizon_headline(logs, style)}。"
                f"{horizon_reference_note(style)}\n"
                "注: 波形の区間もレプリケートも固定である (D-107)。"
                "「よく当たっている区間」を選べる図にしない。",
                f"Free-run panel: {horizon_headline(logs, style)}."
                f" {horizon_reference_note(style)}\n"
                "Note: the waveform window and replicate are fixed (D-107)."
                " The figure must not let anyone pick a favourable window.",
            ),
            fontsize=8,
        )
        conditions = (
            f"n_train = {rows[0].n_train}, n_test = {rows[0].n_test}, "
            f"waveform steps = {' / '.join(windows)}"
        )
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
        label_panels(list(axes.ravel()), style=style)
        for index, report in enumerate(reports):
            _scatter_space(axes[index][0], report, RESERVOIR_STATE, style)
            _scatter_space(axes[index][1], report, DELAY_EMBEDDED_INPUT, style)
            _plot_cumulative_ratio_panel(axes[index][2], report, style)
        figure.suptitle(
            style.label(
                "実験 1: リザバー状態は入力の遅延埋め込みより少ない主成分で説明できる",
                "Experiment 1: the reservoir state needs fewer principal"
                " components than the delay embedding of the input",
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

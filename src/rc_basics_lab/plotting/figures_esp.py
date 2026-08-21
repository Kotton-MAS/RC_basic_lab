"""記事02の図4枚 (実験 2-A / 2-B / 2-C / 2-D).

- ``plot_esp_decay``: rho 別の状態距離の減衰曲線 (受け入れ条件1)。
- ``plot_leak_timescale``: リーク率と実効時定数。理論線 ``-1/log(1-a)`` を重ねる
  (受け入れ条件4)。
- ``plot_esp_map``: rho x 入力強度 の ESP 成立領域 (**記事の目玉**、受け入れ条件2)。
  無入力 (sigma_u=0) は別枠のパネルに出し、駆動下の領域と混ぜない。
- ``plot_washout_sensitivity``: washout 長への性能感度 (受け入れ条件5、D-19)。
  01 の本番値に垂直線を引き、変動幅を数値で注記する。

``figures.py`` と同じ規律に従う: pyplot を使わず ``Figure`` +
``FigureCanvasAgg`` を直接組み、描画設定は ``matplotlib.rc_context`` で描画中
だけ一時適用する (F-1-008)。ラベルは必ず ja/en の対で書く (D-10)。

**ギリシャ文字は書かない**: ruff の RUF001/RUF002 が ASCII と紛らわしい文字を
弾くため、ソース中では ``rho`` / ``sigma_u`` と綴る (T2 実装メモ17)。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from rc_basics_lab.experiment.esp import ConditionOutcome, EspRow
from rc_basics_lab.experiment.washout import (
    HEADLINE_METHOD,
    HEADLINE_TASK,
    WashoutRow,
    WashoutSensitivity,
    mean_nrmse_by_washout,
)
from rc_basics_lab.plotting.heatmap import cell_edges, grid_from_means
from rc_basics_lab.plotting.labels import METHOD_LABELS
from rc_basics_lab.plotting.style import (
    StyleContext,
    add_provenance,
    method_color,
    rc_context_for,
    reference_line_kwargs,
    require_rows,
    sequential_colors,
)
from rc_basics_lab.plotting.style import new_figure as _new_figure
from rc_basics_lab.plotting.style import save_png as _save
from rc_basics_lab.plotting.style import unique_sorted as _unique_sorted
from rc_basics_lab.types import FloatArray

_DISTANCE_FLOOR = 1.0e-16
"""対数軸で距離 0 を描くための下限 (float64 の丸めの床と同じ桁)。

無入力・rho<1 では距離が**厳密に 0** になるため、クリップしないと曲線が
対数軸から消える。クリップした水準には水平線を引き、「ここから下は測れて
いない」ことを図の中で明示する。
"""

_ESP_MAP_WIDTH_RATIOS = (1.0, 4.0)
"""``plot_esp_map`` の (無入力パネル, 駆動下パネル) の幅比。"""

_MIN_CONTOUR_POINTS = 2
"""λ=0 の等高線を引くのに必要な各軸の最小点数。"""

_GALLICCHIO = (
    "Gallicchio (2019) Chasing the Echo State Property の再実演",
    "Re-enacting Gallicchio (2019), Chasing the Echo State Property",
)
"""2-C の副題 (仕様 §4 T3: 先行研究の再実演であることを図に明記する)。"""

_CONVERGED_LABEL = ("ESP 成立率 (レプリケート平均)", "ESP rate (mean over replicates)")


def _group_mean(
    rows: Sequence[EspRow], key: str
) -> Mapping[tuple[float, float], float]:
    """``(rho, sigma_u)`` ごとに ``key`` の平均を取る (レプリケート平均)。"""
    totals: dict[tuple[float, float], list[float]] = {}
    for row in rows:
        value = float(getattr(row, key))
        totals.setdefault((row.rho, row.sigma_u), []).append(value)
    return {
        cell: float(np.mean(values)) if values else math.nan
        for cell, values in totals.items()
    }


# --- 2-A: ESP の減衰曲線 ---------------------------------------------------


_DECAY_X_MARGIN = 1.3
"""減衰が床に届いた後も表示する余白の倍率。"""

_DECAY_X_MIN = 50.0
"""横軸の下限 (減衰が速すぎても目盛が読める幅を確保する)。"""


def _decay_x_limit(outcomes: Sequence[ConditionOutcome]) -> tuple[float, int]:
    """減衰曲線の横軸上限と系列長を返す。

    系列長 (本番 3000 ステップ) をそのまま横軸に取ると、測れている区間
    (rho=0.5 なら 46 ステップ、rho=0.95 でも 680 ステップ) が左端に潰れて
    図の主役である**傾きの違い**が読めない。床に届いた時刻の最大値に余白を
    足したところで切る。切ったことは軸ラベルに数値で書く (「グラフの外に
    何かある」と読者に思わせないため)。
    """
    n_steps = max(outcome.distance.shape[0] for outcome in outcomes)
    crossings = [
        int(np.nonzero(outcome.distance <= _DISTANCE_FLOOR)[0][0])
        for outcome in outcomes
        if np.any(outcome.distance <= _DISTANCE_FLOOR)
    ]
    if not crossings:
        return float(n_steps), n_steps
    limit = max(_DECAY_X_MIN, max(crossings) * _DECAY_X_MARGIN)
    return min(float(n_steps), limit), n_steps


def plot_esp_decay(
    outcomes: Sequence[ConditionOutcome], path: Path, *, style: StyleContext
) -> Path:
    """rho 別の状態距離の減衰曲線を重ね描きする (受け入れ条件1)。

    縦軸は RMS/ユニット距離 (D-16) の対数。rho<1 なら直線的に落ち、rho>1 なら
    落ちない。レプリケートは薄い線で全部描き、凡例は rho ごとに1つだけ出す。

    Raises:
        ValueError: ``outcomes`` が空の場合。
    """
    if not outcomes:
        raise ValueError("outcomes が空です")
    rhos = _unique_sorted([outcome.row.rho for outcome in outcomes])
    colors = sequential_colors(len(rhos))

    with rc_context_for(style):
        figure = _new_figure(7.5, 4.6)
        axis = figure.subplots(1, 1)
        for index, rho in enumerate(rhos):
            selected = [outcome for outcome in outcomes if outcome.row.rho == rho]
            for order, outcome in enumerate(selected):
                curve = np.maximum(outcome.distance, _DISTANCE_FLOOR)
                axis.plot(
                    np.arange(curve.shape[0]),
                    curve,
                    color=colors[index],
                    linewidth=1.2,
                    alpha=0.85,
                    label=(
                        style.label(f"rho = {rho:g}", f"rho = {rho:g}")
                        if order == 0
                        else None
                    ),
                )
        axis.axhline(
            _DISTANCE_FLOOR,
            color="tab:gray",
            linestyle=":",
            linewidth=1.0,
            label=style.label(
                f"丸めの床 ({_DISTANCE_FLOOR:g}。これ以下は測れない)",
                f"rounding floor ({_DISTANCE_FLOOR:g}; not measurable below)",
            ),
        )
        axis.set_yscale("log")
        axis.set_ylim(_DISTANCE_FLOOR / 10.0, 10.0)
        x_limit, n_steps = _decay_x_limit(outcomes)
        axis.set_xlim(0.0, x_limit)
        axis.set_xlabel(
            style.label(
                f"ステップ t (系列長 {n_steps} のうち先頭 {int(x_limit)} を表示)",
                f"step t (first {int(x_limit)} of {n_steps}; the rest is flat)",
            )
        )
        axis.set_ylabel(
            style.label(
                "2軌道の距離 ||x_a - x_b|| / sqrt(N)",
                "trajectory distance ||x_a - x_b|| / sqrt(N)",
            )
        )
        first = outcomes[0].row
        axis.set_title(
            style.label(
                f"無入力 (sigma_u = {first.sigma_u:g}) での状態距離の減衰",
                f"Decay of the state distance without input"
                f" (sigma_u = {first.sigma_u:g})",
            )
        )
        axis.legend(loc="upper right", fontsize=8, ncols=2)
        figure.suptitle(
            style.label(
                "実験 2-A: 無入力では rho が 1 に近いほど過去が消えるのが遅い",
                "Experiment 2-A: without input, the closer rho is to 1"
                " the slower the past decays",
            )
        )
        conditions = f"N = {first.n_units}, sigma_u = {first.sigma_u:g}"
        add_provenance(
            figure, conditions, [outcome.row for outcome in outcomes], style=style
        )
        return _save(figure, path)


# --- 2-B: リーク率と実効時定数 ---------------------------------------------


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
) -> None:
    """実効時定数と理論線 ``-1/log(1-a)`` の比較。"""
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
        color="tab:blue",
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
        style.label("実効時定数 tau [ステップ]", "effective timescale tau [steps]")
    )
    axis.set_title(
        style.label(
            "リーク率と実効時定数 (単調減少)",
            "Leak rate versus effective timescale (monotone)",
        ),
        fontsize=10,
    )
    axis.legend(loc="upper right", fontsize=8)


def plot_leak_timescale(
    outcomes: Sequence[ConditionOutcome], path: Path, *, style: StyleContext
) -> Path:
    """リーク率と実効時定数の関係を描く (受け入れ条件4)。

    左が自己相関曲線そのもの、右が 1/e 交差から求めた時定数と理論線
    ``-1/log(1-a)`` の比較。理論線は線形域の値なので実測より小さく出るが、
    **単調性が一致する**ことが受け入れ条件である。

    Raises:
        ValueError: ``outcomes`` が空の場合。
    """
    if not outcomes:
        raise ValueError("outcomes が空です")
    rows = [outcome.row for outcome in outcomes]
    leak_rates = _unique_sorted([row.leak_rate for row in rows])
    colors = sequential_colors(len(leak_rates))

    with rc_context_for(style):
        figure = _new_figure(11.0, 4.6)
        axes = figure.subplots(1, 2, squeeze=False)
        _plot_acf_panel(axes[0][0], outcomes, leak_rates, colors, style)
        _plot_timescale_panel(axes[0][1], rows, leak_rates, style)
        first = rows[0]
        figure.suptitle(
            style.label(
                "実験 2-B: リーク率を下げると状態の時定数は理論線どおりに伸びる",
                "Experiment 2-B: lowering the leak rate stretches the state"
                " timescale along the theory line",
            )
        )
        conditions = (
            f"N = {first.n_units}, rho = {first.rho:g}, sigma_u = {first.sigma_u:g}"
        )
        add_provenance(figure, conditions, rows, style=style)
        return _save(figure, path)


# --- 2-C: rho x 入力強度 の ESP 成立領域 -----------------------------------


def _plot_no_input_panel(
    axis: Axes,
    rates: FloatArray,
    rhos: Sequence[float],
    norm: Normalize,
    style: StyleContext,
) -> None:
    """無入力 (sigma_u = 0) の列。駆動下の領域と同じ配色で別枠に出す。"""
    axis.pcolormesh(
        np.array([0.0, 1.0]),
        cell_edges(rhos),
        rates.reshape(-1, 1),
        cmap="RdYlBu",
        norm=norm,
        shading="flat",
    )
    axis.set_xticks([0.5])
    axis.set_xticklabels([style.label("無入力", "no input")])
    axis.set_ylabel(style.label("スペクトル半径 rho", "spectral radius rho"))
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    axis.set_title(
        style.label("sigma_u = 0", "sigma_u = 0"),
        fontsize=10,
    )


def _plot_driven_panel(
    figure: Figure,
    axis: Axes,
    rates: FloatArray,
    lambdas: FloatArray,
    rhos: Sequence[float],
    sigmas: Sequence[float],
    norm: Normalize,
    style: StyleContext,
) -> None:
    """駆動下 (sigma_u > 0) の ESP 成立領域と λ=0 の等高線。

    ``rates`` / ``lambdas`` は ``(len(rhos), len(sigmas))`` の格子。横軸は
    sigma_u の**順位**にする (格子が等比的に広がるので、値そのものを軸に取ると
    強入力側だけが潰れて読めなくなる)。
    """
    mesh = axis.pcolormesh(
        np.arange(len(sigmas) + 1, dtype=np.float64),
        cell_edges(rhos),
        rates,
        cmap="RdYlBu",
        norm=norm,
        shading="flat",
    )
    figure.colorbar(mesh, ax=axis, label=style.label(*_CONVERGED_LABEL))
    if len(sigmas) >= _MIN_CONTOUR_POINTS and len(rhos) >= _MIN_CONTOUR_POINTS:
        axis.contour(
            np.arange(len(sigmas), dtype=np.float64) + 0.5,
            np.asarray(rhos, dtype=np.float64),
            lambdas,
            levels=[0.0],
            colors="black",
            linewidths=1.5,
        )
    axis.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=style.label("rho = 1 (通説の境界)", "rho = 1 (the folklore boundary)"),
    )
    axis.set_xticks(np.arange(len(sigmas), dtype=np.float64) + 0.5)
    axis.set_xticklabels([f"{sigma:g}" for sigma in sigmas])
    axis.set_xlabel(
        style.label(
            "入力強度 sigma_u (駆動信号の標準偏差)",
            "input strength sigma_u (s.d. of the drive)",
        )
    )
    axis.set_title(
        style.label(
            "駆動下の ESP 成立領域 (黒実線は λ = 0)",
            "ESP region under input (solid black: λ = 0)",
        ),
        fontsize=10,
    )
    axis.legend(loc="upper right", fontsize=8)


def plot_esp_map(rows: Sequence[EspRow], path: Path, *, style: StyleContext) -> Path:
    """rho x 入力強度 の ESP 成立領域を描く (受け入れ条件2。**記事の目玉**)。

    左の細いパネルが無入力 (sigma_u = 0)、右が駆動下。無入力を別枠にするのは、
    「無入力なら rho<1 が必要条件」という正しい主張と、「入力があれば rho>1
    でも ESP は成立しうる」という主張を1枚の中で混ぜないため。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    require_rows(rows)
    rhos = _unique_sorted([row.rho for row in rows])
    sigmas = _unique_sorted([row.sigma_u for row in rows])
    driven = tuple(sigma for sigma in sigmas if sigma > 0.0)
    has_no_input = any(sigma == 0.0 for sigma in sigmas)
    converged = _group_mean(rows, "converged")
    lyapunov = _group_mean(rows, "lyapunov_per_step")
    norm = Normalize(vmin=0.0, vmax=1.0)

    with rc_context_for(style):
        figure = _new_figure(9.5, 5.4)
        if has_no_input:
            axes = figure.subplots(
                1, 2, squeeze=False, width_ratios=list(_ESP_MAP_WIDTH_RATIOS)
            )
            _plot_no_input_panel(
                axes[0][0],
                grid_from_means(converged, rhos, (0.0,))[0],
                rhos,
                norm,
                style,
            )
            driven_axis = axes[0][1]
        else:
            axes = figure.subplots(1, 1, squeeze=False)
            driven_axis = axes[0][0]
            driven_axis.set_ylabel(
                style.label("スペクトル半径 rho", "spectral radius rho")
            )
        _plot_driven_panel(
            figure,
            driven_axis,
            grid_from_means(converged, rhos, driven).T,
            grid_from_means(lyapunov, rhos, driven).T,
            rhos,
            driven,
            norm,
            style,
        )
        figure.suptitle(
            style.label(
                "実験 2-C: 入力を強くすると rho > 1 でも ESP は成立する\n"
                + _GALLICCHIO[0],
                "Experiment 2-C: strong input restores the ESP above rho = 1\n"
                + _GALLICCHIO[1],
            )
        )
        conditions = f"N = {rows[0].n_units}, washout = {rows[0].washout}"
        add_provenance(figure, conditions, rows, style=style)
        return _save(figure, path)


# --- 2-D: washout 長への性能感度 ------------------------------------------


_TASK_LABELS: Mapping[str, tuple[str, str]] = {
    "mackey_glass": ("Mackey-Glass", "Mackey-Glass"),
    "delay_parity": ("遅延パリティ", "delay parity"),
}
"""課題名の表示ラベル (D-10: ja/en の対)。未知の課題は名前をそのまま出す。"""


_CONTROL_ALPHA = 0.40
"""対照 (主役でない課題) の線の不透明度。「薄く重ねる」の実体。"""

_HEADLINE_LINEWIDTH = 2.0
_CONTROL_LINEWIDTH = 1.2


def _series_label(style: StyleContext, task: str, method: str) -> str:
    """凡例のラベル ``課題 x 手法``。"""
    task_ja, task_en = _TASK_LABELS.get(task, (task, task))
    method_ja, method_en = METHOD_LABELS.get(method, (method, method))
    return style.label(f"{task_ja} x {method_ja}", f"{task_en} x {method_en}")


def _washout_series(
    rows: Sequence[WashoutRow], task: str, method: str
) -> tuple[list[float], FloatArray, FloatArray]:
    """(washout, レプリケート平均 NRMSE, レプリケート間標準偏差) の3点セット。

    平均は ``mean_nrmse_by_washout`` (実験層) と共有する。標準偏差は
    ``WashoutRow.nrmse_std`` がすでに (課題, 手法, washout) 単位の値なので、
    同じ組の行から1つ拾えばよい。
    """
    means = mean_nrmse_by_washout(rows, task, method)
    stds = {
        row.washout: row.nrmse_std
        for row in rows
        if row.task == task and row.method == method
    }
    washouts = list(means)
    values: FloatArray = np.array(
        [means[washout] for washout in washouts], dtype=np.float64
    )
    errors: FloatArray = np.array(
        [stds[washout] for washout in washouts], dtype=np.float64
    )
    return [float(washout) for washout in washouts], values, errors


def _plot_absolute_panel(
    axis: Axes,
    rows: Sequence[WashoutRow],
    pairs: Sequence[tuple[str, str]],
    sensitivity: WashoutSensitivity,
    style: StyleContext,
) -> None:
    """左パネル: NRMSE の絶対値 (課題間の水準差を示す文脈)。"""
    for task, method in pairs:
        washouts, values, errors = _washout_series(rows, task, method)
        headline = task == HEADLINE_TASK
        axis.errorbar(
            washouts,
            values,
            yerr=errors,
            fmt="o-" if headline else "s--",
            capsize=3,
            color=method_color(method),
            alpha=1.0 if headline else _CONTROL_ALPHA,
            linewidth=_HEADLINE_LINEWIDTH if headline else _CONTROL_LINEWIDTH,
            label=_series_label(style, task, method),
        )
    axis.set_yscale("log")
    axis.set_xlabel(style.label("washout [ステップ]", "washout [steps]"))
    axis.set_ylabel(
        style.label("テスト NRMSE (レプリケート平均)", "test NRMSE (mean over reps)")
    )
    axis.set_title(
        style.label(
            "絶対値 (誤差棒はレプリケート間の標準偏差)",
            "Absolute NRMSE (error bars: spread over replicates)",
        ),
        fontsize=10,
    )
    _mark_reference(axis, sensitivity, style)
    axis.legend(loc="center right", fontsize=7, ncols=1)


def _plot_relative_panel(
    axis: Axes,
    rows: Sequence[WashoutRow],
    pairs: Sequence[tuple[str, str]],
    sensitivity: WashoutSensitivity,
    style: StyleContext,
) -> None:
    """右パネル: 01 の本番値を 1 とした相対 NRMSE。

    絶対値のパネルでは水準差 (MG の 7e-4 と パリティの 1.0) が支配的で、
    washout による 1% 未満の変動が読めない。**この図の主張は変動の大きさ**な
    ので、基準点で割った比を別パネルにする。
    """
    for task, method in pairs:
        washouts, values, errors = _washout_series(rows, task, method)
        reference = sensitivity.find(task, method).nrmse_at_reference
        if not math.isfinite(reference) or reference <= 0.0:
            continue
        headline = task == HEADLINE_TASK
        axis.errorbar(
            washouts,
            values / reference,
            yerr=errors / reference,
            fmt="o-" if headline else "s--",
            capsize=3,
            color=method_color(method),
            alpha=1.0 if headline else _CONTROL_ALPHA,
            linewidth=_HEADLINE_LINEWIDTH if headline else _CONTROL_LINEWIDTH,
            label=_series_label(style, task, method),
        )
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.set_xlabel(style.label("washout [ステップ]", "washout [steps]"))
    axis.set_ylabel(
        style.label(
            f"NRMSE / NRMSE(washout={sensitivity.reference_washout})",
            f"NRMSE / NRMSE(washout={sensitivity.reference_washout})",
        )
    )
    axis.set_title(
        style.label(
            "01 の本番値で正規化した比",
            "Normalised by the production value used in 01",
        ),
        fontsize=10,
    )
    _mark_reference(axis, sensitivity, style)
    axis.text(
        0.02,
        0.02,
        _variation_note(sensitivity, style),
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
    )


def _mark_reference(
    axis: Axes, sensitivity: WashoutSensitivity, style: StyleContext
) -> None:
    """01 の本番値に垂直線を引く (仕様 §4 T4)。"""
    axis.axvline(
        float(sensitivity.reference_washout),
        **reference_line_kwargs(1),
        label=style.label(
            f"01 の本番値 (washout = {sensitivity.reference_washout})",
            f"production value used in 01 (washout = {sensitivity.reference_washout})",
        ),
    )


def _variation_note(sensitivity: WashoutSensitivity, style: StyleContext) -> str:
    """変動幅の数値注記 (仕様 §4 T4: 「変動幅を数値注記する」)。

    比だけでなく**レプリケート間のばらつきと比べてどうか**まで書く。比が
    1.0 でないことだけを見て「効果があった」と読まれるのを防ぐため。
    """
    headline = sensitivity.headline
    verdict_ja, verdict_en = (
        ("レプリケート間のばらつきを超える", "exceeds the replicate spread")
        if headline.exceeds_replicate_noise
        else ("レプリケート間のばらつき以下", "within the replicate spread")
    )
    task_ja, task_en = _TASK_LABELS.get(headline.task, (headline.task, headline.task))
    method_ja, method_en = METHOD_LABELS.get(
        headline.method, (headline.method, headline.method)
    )
    return style.label(
        f"{task_ja} x {method_ja}: 変動幅 (最大/最小) = {headline.ratio:.4f} 倍\n"
        f"({headline.nrmse_min:.3g} .. {headline.nrmse_max:.3g}、"
        f"レプリケート間 s.d. 最大 {headline.replicate_std_max:.3g})\n"
        f"-> {verdict_ja}",
        f"{task_en} x {method_en}: max/min = {headline.ratio:.4f}\n"
        f"({headline.nrmse_min:.3g} .. {headline.nrmse_max:.3g}; "
        f"max replicate s.d. {headline.replicate_std_max:.3g})\n"
        f"-> {verdict_en}",
    )


def plot_washout_sensitivity(
    rows: Sequence[WashoutRow],
    path: Path,
    *,
    style: StyleContext,
    sensitivity: WashoutSensitivity,
) -> Path:
    """washout 長への性能感度を描く (受け入れ条件5、D-19)。

    主役は Mackey-Glass、遅延パリティは「washout に反応しない対照」として
    薄く重ねる。01 の本番値 (``sensitivity.reference_washout``) に垂直線を引き、
    変動幅を数値で注記する。

    副題に ``pad_series`` を書くのは、この図が「washout の効果」を示すのか
    「washout + 訓練データ量の効果」を示すのかが、その1点で変わるため (D-19)。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    require_rows(rows)
    pairs = sorted(
        {(row.task, row.method) for row in rows},
        key=lambda pair: (pair[0] != HEADLINE_TASK, pair[1] != HEADLINE_METHOD, pair),
    )

    with rc_context_for(style):
        figure = _new_figure(11.0, 4.8)
        axes = figure.subplots(1, 2, squeeze=False)
        _plot_absolute_panel(axes[0][0], rows, pairs, sensitivity, style)
        _plot_relative_panel(axes[0][1], rows, pairs, sensitivity, style)
        design_ja, design_en = (
            (
                "訓練/検証/テストの行数を格子全体で一定に保つ補償あり",
                "with compensation: train/val/test sizes held constant",
            )
            if sensitivity.pad_series
            else (
                "補償なし (washout と訓練データ量が交絡した設計)",
                "no compensation: washout is confounded with the training size",
            )
        )
        figure.suptitle(
            style.label(
                f"実験 2-D: washout 長への性能感度\n{design_ja}",
                f"Experiment 2-D: sensitivity to the washout length\n{design_en}",
            )
        )
        conditions = (
            f"n_train = {rows[0].n_train}, pad_series = {sensitivity.pad_series}"
        )
        add_provenance(figure, conditions, rows, style=style)
        return _save(figure, path)


__all__ = [
    "plot_esp_decay",
    "plot_esp_map",
    "plot_leak_timescale",
    "plot_washout_sensitivity",
]

"""記事02の図3枚 (実験 2-A / 2-B / 2-C).

- ``plot_esp_decay``: rho 別の状態距離の減衰曲線 (受け入れ条件1)。
- ``plot_leak_timescale``: リーク率と実効時定数。理論線 ``-1/log(1-a)`` を重ねる
  (受け入れ条件4)。
- ``plot_esp_map``: rho x 入力強度 の ESP 成立領域 (**記事の目玉**、受け入れ条件2)。
  無入力 (sigma_u=0) は別枠のパネルに出し、駆動下の領域と混ぜない。

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

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from rc_basics_lab.experiment.esp import ConditionOutcome, EspRow
from rc_basics_lab.plotting.style import StyleContext, rc_params_for
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


def _new_figure(width: float, height: float) -> Figure:
    """constrained layout の Figure を作る (``figures.py`` と同じ規律)。"""
    figure = Figure(figsize=(width, height))
    figure.set_layout_engine("constrained")
    return figure


def _save(figure: Figure, path: Path) -> Path:
    """Agg キャンバスで PNG を書く (ディスプレイに依存しない)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)
    figure.savefig(path, format="png")
    return path


def _unique_sorted(values: Sequence[float]) -> tuple[float, ...]:
    """重複を除いて昇順に並べる (格子の軸を行から復元する)。"""
    return tuple(sorted(set(values)))


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


def _grid(
    means: Mapping[tuple[float, float], float],
    rhos: Sequence[float],
    sigmas: Sequence[float],
) -> FloatArray:
    """``(len(sigmas), len(rhos))`` の格子に並べ直す (行が sigma_u)。"""
    grid: FloatArray = np.full((len(sigmas), len(rhos)), math.nan, dtype=np.float64)
    for row_index, sigma in enumerate(sigmas):
        for column_index, rho in enumerate(rhos):
            value = means.get((rho, sigma))
            if value is not None:
                grid[row_index, column_index] = value
    return grid


def _edges(values: Sequence[float]) -> FloatArray:
    """``pcolormesh`` 用のセル境界 (等間隔でない格子でも中心を保つ)。"""
    centers: FloatArray = np.asarray(values, dtype=np.float64)
    if centers.size == 1:
        half = 0.5 if centers[0] == 0.0 else abs(float(centers[0])) * 0.5
        edges: FloatArray = np.array(
            [float(centers[0]) - half, float(centers[0]) + half], dtype=np.float64
        )
        return edges
    inner = (centers[:-1] + centers[1:]) / 2.0
    first = centers[0] - (inner[0] - centers[0])
    last = centers[-1] + (centers[-1] - inner[-1])
    built: FloatArray = np.concatenate(([first], inner, [last]))
    return built


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
    colors = matplotlib.colormaps["viridis"](np.linspace(0.0, 0.9, len(rhos)))

    with matplotlib.rc_context(rc_params_for(style)):  # type: ignore[arg-type]
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
                "実験 2-A: スペクトル半径と ESP",
                "Experiment 2-A: spectral radius and the echo state property",
            )
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
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
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
            linestyle="--",
            color="tab:orange",
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
    colors = matplotlib.colormaps["viridis"](np.linspace(0.0, 0.9, len(leak_rates)))

    with matplotlib.rc_context(rc_params_for(style)):  # type: ignore[arg-type]
        figure = _new_figure(11.0, 4.6)
        axes = figure.subplots(1, 2, squeeze=False)
        _plot_acf_panel(axes[0][0], outcomes, leak_rates, colors, style)
        _plot_timescale_panel(axes[0][1], rows, leak_rates, style)
        first = rows[0]
        figure.suptitle(
            style.label(
                "実験 2-B: リーク率が変えるのは状態の時定数"
                f" (rho = {first.rho:g}, sigma_u = {first.sigma_u:g})",
                "Experiment 2-B: the leak rate sets the state timescale"
                f" (rho = {first.rho:g}, sigma_u = {first.sigma_u:g})",
            )
        )
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
        _edges(rhos),
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
        _edges(rhos),
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
    if not rows:
        raise ValueError("rows が空です")
    rhos = _unique_sorted([row.rho for row in rows])
    sigmas = _unique_sorted([row.sigma_u for row in rows])
    driven = tuple(sigma for sigma in sigmas if sigma > 0.0)
    has_no_input = any(sigma == 0.0 for sigma in sigmas)
    converged = _group_mean(rows, "converged")
    lyapunov = _group_mean(rows, "lyapunov_per_step")
    norm = Normalize(vmin=0.0, vmax=1.0)

    with matplotlib.rc_context(rc_params_for(style)):  # type: ignore[arg-type]
        figure = _new_figure(9.5, 5.4)
        if has_no_input:
            axes = figure.subplots(
                1, 2, squeeze=False, width_ratios=list(_ESP_MAP_WIDTH_RATIOS)
            )
            _plot_no_input_panel(
                axes[0][0],
                _grid(converged, rhos, (0.0,))[0],
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
            _grid(converged, rhos, driven).T,
            _grid(lyapunov, rhos, driven).T,
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
        return _save(figure, path)


__all__ = [
    "plot_esp_decay",
    "plot_esp_map",
    "plot_leak_timescale",
]

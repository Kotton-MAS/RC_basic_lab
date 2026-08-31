"""記事02の図2枚 (実験 2-A / 2-C).

- ``plot_esp_decay``: rho 別の状態距離の減衰曲線 (受け入れ条件1)。
- ``plot_esp_map``: rho x 入力強度 の ESP 成立領域 (**記事の目玉**、受け入れ条件2)。
  無入力 (sigma_u=0) は別枠のパネルに出し、駆動下の領域と混ぜない。

2-B (``plot_leak_timescale``) は ``figures_leak.py``、2-D
(``plot_washout_sensitivity``) は ``figures_washout.py`` にある —— このモジュールが
行数上限 (D-77) に達したため。**上限は緩めない**。

図の外枠 (rcParams の一時適用 / 保存) は ``plotting/style.py`` が持つ。ラベルは
必ず ja/en の対で書く (D-10)。**ギリシャ文字は書かない** (RUF001/RUF002。
ソース中では ``rho`` / ``sigma_u`` と綴る)。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.figure import Figure

from rc_basics_lab.experiment.esp import ConditionOutcome, EspRow
from rc_basics_lab.plotting.esp_references import (
    GALLICCHIO,
    plot_no_input_panel,
    zero_input_boundary_note,
)
from rc_basics_lab.plotting.heatmap import cell_edges, grid_from_means
from rc_basics_lab.plotting.layout import label_panels, legend_below, wrapped_note
from rc_basics_lab.plotting.style import (
    StyleContext,
    add_provenance,
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


def replicate_count(outcomes: Sequence[ConditionOutcome]) -> int:
    """1条件あたりのレプリケート本数 (図の注に埋める実測値)。

    決め打ちの「3 本」と書かないのは、縮小設定 (テスト / golden) では本数が
    違うためである。注と図がずれると、注のほうが信用されなくなる。
    """
    return max(len({outcome.row.replicate for outcome in outcomes}), 1)


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
            **reference_line_kwargs(1),
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
        legend_below(figure, [axis], style=style, ncol=4)  # 飽和域への被りを解消
        # **同色の複数本がレプリケートであることを図の中に書く** (2-16)。
        # 凡例は rho ごとに1つしか出さないので、太った線に見えて「なぜ
        # 3 本あるのか」が図から読めなかった。
        figure.supxlabel(
            wrapped_note(
                style.label(
                    f"注: 同じ色の {replicate_count(outcomes)} 本は"
                    "レプリケート (乱数シード違い) である。凡例は rho ごとに"
                    "1つだけ出している。",
                    f"Note: the {replicate_count(outcomes)} curves sharing a"
                    " colour are replicates (different random seeds); the legend"
                    " lists each rho once.",
                )
            ),
            fontsize=8,
        )
        figure.suptitle(
            style.label(
                "実験 2-A: 無入力では rho が 1 に近いほど減衰が遅く rho > 1 で消えない",
                "Experiment 2-A: without input the past decays more slowly as"
                " rho approaches 1, and stops decaying above rho = 1",
            )
        )
        conditions = f"N = {first.n_units}, sigma_u = {first.sigma_u:g}"
        add_provenance(
            figure, conditions, [outcome.row for outcome in outcomes], style=style
        )
        return _save(figure, path)


# --- 2-C: rho x 入力強度 の ESP 成立領域 -----------------------------------


_ESP_LEVELS: tuple[float, ...] = (0.0, 0.5, 1.0)
"""ESP 成立率の色の段 (2-9)。

成立率はほぼ 0 か 1 で、中間は数セルしかない。連続の発散カラーマップ (RdYlBu)
は中央 0.5 に意味があるときの配色で、ここには無い。**3 段に切る** ——
不成立 / 一部成立 / 成立 のどれかであることを色から直接読めるようにする。
"""

_ESP_DISCRETE_CMAP = ListedColormap(["#b2182b", "#f4a582", "#2166ac"])
"""``_ESP_LEVELS`` に対応する色 (不成立=赤 / 一部=薄橙 / 成立=青)。"""

_ESP_DISCRETE_NORM = BoundaryNorm([0.0, 0.25, 0.75, 1.0], 3)
"""段の境界。0.25 / 0.75 で切るので、0.9 が中間色に見えることはない。"""


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
    # **離散3段にする** (2-9)。成立率はほぼ 0 か 1 で、中間は数セルしかない。
    # RdYlBu は発散型で中央 0.5 に意味があるときの配色だが、ここに 0.5 の意味は
    # 無く、「中間色に見えるが実は 0.9」のような読み違いを誘う。段に切れば
    # 「成立 / 一部 / 不成立」のどれかであることが色から直接読める。
    mesh = axis.pcolormesh(
        np.arange(len(sigmas) + 1, dtype=np.float64),
        cell_edges(rhos),
        rates,
        cmap=_ESP_DISCRETE_CMAP,
        norm=_ESP_DISCRETE_NORM,
        shading="flat",
    )
    colorbar = figure.colorbar(
        mesh, ax=axis, label=style.label(*_CONVERGED_LABEL), ticks=_ESP_LEVELS
    )
    colorbar.ax.set_yticklabels([f"{level:g}" for level in _ESP_LEVELS])
    if len(sigmas) >= _MIN_CONTOUR_POINTS and len(rhos) >= _MIN_CONTOUR_POINTS:
        axis.contour(
            np.arange(len(sigmas), dtype=np.float64) + 0.5,
            np.asarray(rhos, dtype=np.float64),
            lambdas,
            levels=[0.0],
            colors="black",
            linewidths=1.5,
        )
        # **λ=0 の実線を凡例に出す** (2-9)。以前はタイトル文中でだけ説明して
        # おり、図が切り取られると何の線か分からなかった。contour は凡例の
        # ハンドルを持たないので proxy を作る。
        axis.plot(
            [],
            [],
            color="black",
            linewidth=1.5,
            label=style.label(
                "λ = 0 (局所安定の境界)", "λ = 0 (local stability boundary)"
            ),
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
            # **y を共有する** (1-7 / 2-9)。左右とも縦軸は rho なので、
            # 共有しないと同じ目盛りが2回出て左パネルの y ラベルが窮屈になる。
            axes = figure.subplots(
                1,
                2,
                squeeze=False,
                width_ratios=list(_ESP_MAP_WIDTH_RATIOS),
                sharey=True,
            )
            plot_no_input_panel(
                axes[0][0],
                grid_from_means(converged, rhos, (0.0,))[0],
                rhos,
                norm,
                style,
            )
            figure.supxlabel(wrapped_note(zero_input_boundary_note(style)), fontsize=7)
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
        label_panels(list(axes.ravel()), style=style)
        figure.suptitle(
            style.label(
                "実験 2-C: 入力を強くすると rho > 1 でも ESP は成立する\n"
                + GALLICCHIO[0],
                "Experiment 2-C: strong input restores the ESP above rho = 1\n"
                + GALLICCHIO[1],
            )
        )
        conditions = f"N = {rows[0].n_units}, washout = {rows[0].washout}"
        add_provenance(figure, conditions, rows, style=style)
        return _save(figure, path)


__all__ = [
    "plot_esp_decay",
    "plot_esp_map",
    "replicate_count",
]

"""記事04の図5枚 (実験 4-A / 4-B / 4-C / 4-D).

- ``plot_onestep``: 教師強制の1ステップ先予測。3手法の NRMSE を課題別に並べる
  (受け入れ条件3 の前半: **差が小さい**ことを見せる図)。
- ``plot_freerun_attractor``: **記事の主図**。自走軌道と真の軌道の位相図の
  重ね描き (Lorenz は (x, z)、1変数系は遅延座標埋め込み)。
- ``plot_valid_time``: 有効予測時間 (Lyapunov 時間で正規化) のシード分布。
  対照 (線形・遅延線) も同じ軸に載せる (受け入れ条件3 の後半)。
- ``plot_stability_map``: 状態ノイズ量ごとに (rho x リーク率) の3態マップを
  並べ、右端に 4-D の容量 (MC) と有効予測時間の関係を置く (受け入れ条件4)。
- ``plot_freerun_stats``: 長時間自走後のリターンマップとパワースペクトルの
  比較 (受け入れ条件5)。

``figures.py`` / ``figures_esp.py`` / ``figures_capacity.py`` と同じ規律に従う:
pyplot を使わず ``Figure`` + ``FigureCanvasAgg`` を直接組み、描画設定は
``matplotlib.rc_context`` で描画中だけ一時適用する (F-1-008)。ラベルは必ず
``style.label(ja, en)`` を通す (D-10)。

**図は成果物 CSV の行だけを読む** (仕様 §5 禁止する構造7)。位相図・リターン
マップ・スペクトルのように「配列」で表現される量も、``freerun_profile.csv`` に
書き出したのと同じ長形式の行 (``FreeRunProfileRow``) から復元する。診断も実験も
ここでは1回も走らせない。

**ギリシャ文字は書かない**: ruff の RUF001/RUF002 が ASCII と紛らわしい文字を
弾くため、ソース中では ``rho`` と綴る (02・03 の図と同じ)。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.axes import Axes

from rc_basics_lab.experiment.attractor import (
    REGIME_ATTRACTOR,
    REGIME_DIVERGED,
    REGIME_PERIODIC,
    REGIMES,
)
from rc_basics_lab.experiment.capacity import CapacityRow
from rc_basics_lab.experiment.freerun import (
    KIND_PHASE,
    KIND_RETURN_MAP,
    KIND_SPECTRUM,
    SOURCE_FREERUN,
    SOURCE_TRUTH,
    FreeRunProfileRow,
    FreeRunRow,
)
from rc_basics_lab.experiment.runner import DELAY_LINE, ESN_METHOD, LINEAR, ResultRow
from rc_basics_lab.experiment.stability import StabilityRow, regime_map
from rc_basics_lab.plotting.freerun_grids import (
    label_of,
    mean_std,
    profile_points,
    tasks_of,
)
from rc_basics_lab.plotting.freerun_headlines import valid_time_headline
from rc_basics_lab.plotting.labels import METHOD_LABELS
from rc_basics_lab.plotting.style import (
    StyleContext,
    add_provenance,
    method_color,
    rc_context_for,
    require_rows,
)
from rc_basics_lab.plotting.style import new_figure as _new_figure
from rc_basics_lab.plotting.style import save_png as _save

TASK_LABELS: dict[str, tuple[str, str]] = {
    "lorenz": ("Lorenz", "Lorenz"),
    "mackey_glass": ("Mackey-Glass", "Mackey-Glass"),
}
"""課題名の表示。"""

REGIME_LABELS: dict[str, tuple[str, str]] = {
    REGIME_DIVERGED: ("発散", "diverged"),
    REGIME_PERIODIC: ("周期軌道", "periodic"),
    REGIME_ATTRACTOR: ("アトラクタ再現", "attractor"),
}
"""3態の表示 (``REGIMES`` の全要素を持つ。欠けたら描く前に落とす)。"""

CENSORED_EDGE_COLOR = "#b2182b"
"""打ち切られた観測の縁の色 (D-43。参照線の色とは別にする)。"""

REGIME_COLORS: dict[str, str] = {
    REGIME_DIVERGED: "#b2182b",
    REGIME_PERIODIC: "#2166ac",
    REGIME_ATTRACTOR: "#1a9850",
}
"""3態の色 (発散=赤 / 周期=青 / 再現=緑)。"""

SOURCE_STYLE: dict[str, tuple[str, str]] = {
    SOURCE_TRUTH: ("#444444", "真の軌道"),
    SOURCE_FREERUN: ("#d95f02", "自走"),
}
"""重ね描きの色と日本語ラベル (英語は ``_source_label``)。"""

_SOURCE_LABELS_EN: dict[str, str] = {
    SOURCE_TRUTH: "true trajectory",
    SOURCE_FREERUN: "free run",
}


def source_label(source: str, style: StyleContext) -> str:
    """出典 (真の軌道 / 予測など) の表示ラベル。

    ``SOURCE_STYLE`` / ``_SOURCE_LABELS_EN`` が 04 固有の定数なので、
    ``freerun_grids`` へは出さずここに残す。
    """
    return style.label(SOURCE_STYLE[source][1], _SOURCE_LABELS_EN[source])


def plot_onestep(rows: Sequence[ResultRow], path: Path, *, style: StyleContext) -> Path:
    """実験 4-A: 教師強制の1ステップ先予測 (受け入れ条件3 の前半)。

    横軸が手法、縦軸がテスト NRMSE (対数)。課題ごとに系列を分ける。
    **差が小さいことを見せる図**なので、対数軸のまま値を注記する。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    require_rows(rows)
    tasks = tasks_of(rows)
    methods = [LINEAR, DELAY_LINE, ESN_METHOD]
    positions = np.arange(len(methods), dtype=np.float64)
    with rc_context_for(style):
        figure = _new_figure(7.6, 5.0)
        axis = figure.subplots(1, 1)
        for offset, task in enumerate(tasks):
            stats = [
                mean_std(
                    [
                        row.nrmse
                        for row in rows
                        if row.task == task and row.method == method
                    ]
                )
                for method in methods
            ]
            means = np.asarray([mean for mean, _ in stats], dtype=np.float64)
            stds = np.asarray([std for _, std in stats], dtype=np.float64)
            axis.errorbar(
                positions + 0.12 * (offset - 0.5 * (len(tasks) - 1)),
                means,
                yerr=np.vstack([np.minimum(stds, means * 0.999), stds]),
                fmt="o",
                capsize=5,
                label=label_of(TASK_LABELS, task, style),
            )
        axis.set_yscale("log")
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [label_of(METHOD_LABELS, method, style) for method in methods]
        )
        axis.set_xlim(-0.5, len(methods) - 0.5)
        n_replicates = len({row.replicate for row in rows})
        axis.set_ylabel(
            style.label(
                f"NRMSE (テスト区間・{n_replicates}レプリケートの平均±標準偏差)",
                f"NRMSE (test split, mean +- s.d. of {n_replicates} replicates)",
            )
        )
        axis.legend(loc="best", fontsize=9)
        figure.suptitle(
            style.label(
                "実験 4-A: 教師強制の1ステップ先予測では遅延線と ESN に差が出ない",
                "Experiment 4-A: with teacher forcing the delay line and the ESN"
                " are indistinguishable",
            )
        )
        conditions = f"n_train = {rows[0].n_train}, n_test = {rows[0].n_test}"
        add_provenance(figure, conditions, rows, style=style)
        return _save(figure, path)


def plot_freerun_attractor(
    rows: Sequence[FreeRunProfileRow], path: Path, *, style: StyleContext
) -> Path:
    """**記事の主図**: 自走軌道と真の軌道の位相図の重ね描き。

    Lorenz は (x, z) 平面、1変数系は遅延座標埋め込み (遅延は真の軌道の自己
    相関から決めた1個を両者に使う)。図は ``freerun_profile.csv`` の行だけを
    読み、軌道を作り直さない。

    Raises:
        ValueError: 位相図の行が1つも無い場合。
    """
    tasks = tasks_of(rows)
    if not tasks:
        raise ValueError("profile 行が空です")
    with rc_context_for(style):
        figure = _new_figure(5.4 * len(tasks), 5.0)
        axes = np.atleast_1d(figure.subplots(1, len(tasks)))
        drawn = 0
        for axis, task in zip(axes, tasks, strict=True):
            for source in (SOURCE_TRUTH, SOURCE_FREERUN):
                points = profile_points(rows, task, KIND_PHASE, source)
                if points.shape[0] == 0:
                    continue
                # 線でつなぐ。位相図の点列は確保軸6 で間引いてあるが、
                # 間引き後も隣接点の間隔は Delta t x stride = 0.05 時間単位
                # (Lorenz) しかないので、折れ線は軌道をなぞる。**点だけで描くと
                # 蝶形の2枚翅が雲に潰れて読めなくなる** (実測)。
                axis.plot(
                    points[:, 0],
                    points[:, 1],
                    linewidth=0.5,
                    alpha=0.8,
                    color=SOURCE_STYLE[source][0],
                    label=source_label(source, style),
                )
                drawn += 1
            axis.set_title(label_of(TASK_LABELS, task, style))
            axis.set_xlabel(
                style.label("第1成分 (標準化)", "component 1 (standardized)")
            )
            axis.set_ylabel(
                style.label("第2成分 / 遅延座標", "component 2 / delay coordinate")
            )
            axis.legend(loc="best", fontsize=8)
        if drawn == 0:
            raise ValueError("位相図に描く点がありません")
        figure.suptitle(
            style.label(
                "実験 4-B: 入力を切っても ESN の自走軌道は真のアトラクタの形を保つ",
                "Experiment 4-B: with the input switched off the ESN free run"
                " keeps the shape of the true attractor",
            )
        )
        conditions = f"tasks = {'/'.join(tasks)}"
        add_provenance(figure, conditions, rows, style=style)
        return _save(figure, path)


def plot_valid_time(
    rows: Sequence[FreeRunRow], path: Path, *, style: StyleContext
) -> Path:
    """有効予測時間 (Lyapunov 時間で正規化) のシード分布 (受け入れ条件2 / 3)。

    横軸が手法、縦軸が ``valid_time_lyapunov``。シードごとの点をそのまま重ねる
    (箱ひげだけにすると10本という本数が図から読めない)。**打ち切られた行は
    別のマーカー**で描く —— 打ち切りを普通の観測値として描くと分布の右端が
    実際より小さく見える (D-43)。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    require_rows(rows)
    # **lambda_max を数値推定してある系だけを描く** (D-42 / D-43)。04 が
    # 推定しているのは Lorenz だけで、Mackey-Glass の行は Lyapunov 列が nan
    # である (推定していない量を他の系の値で埋めない)。縦軸が Lyapunov 時間の
    # この図に nan の系を並べると、空のパネルが「有効予測時間が 0」に見える。
    normalized = [row for row in rows if math.isfinite(row.lyapunov_time)]
    if not normalized:
        raise ValueError("lyapunov_time が有限な行がありません")
    tasks = tasks_of(normalized)
    rows = normalized
    methods = [LINEAR, DELAY_LINE, ESN_METHOD]
    with rc_context_for(style):
        figure = _new_figure(6.4 * len(tasks), 5.0)
        axes = np.atleast_1d(figure.subplots(1, len(tasks), sharey=True))
        for axis, task in zip(axes, tasks, strict=True):
            _valid_time_panel(axis, rows, task, methods, style)
        axes[0].set_ylabel(
            style.label(
                "有効予測時間 [Lyapunov 時間]", "valid prediction time [1 / lambda_max]"
            )
        )
        figure.suptitle(
            style.label(
                f"実験 4-B: {valid_time_headline(rows, style)}",
                f"Experiment 4-B: {valid_time_headline(rows, style)}",
            )
        )
        figure.supxlabel(
            style.label(
                "注: lambda_max の数値推定は Lorenz だけ (D-42)。"
                "**文献の有効予測時間との照合は未了** (引くべき値と条件が未特定)"
                "のため、この図が言えるのは水準ではなく手法間の差である。",
                "Note: lambda_max is estimated only for Lorenz (D-42)."
                " No literature valid time is plotted yet (value and conditions"
                " unidentified), so this figure compares methods, not levels.",
            ),
            fontsize=8,
        )
        conditions = (
            f"free_run = {rows[0].free_run_steps} steps,"
            f" threshold = {rows[0].valid_time_threshold:g}"
        )
        add_provenance(figure, conditions, rows, style=style)
        return _save(figure, path)


def _valid_time_panel(
    axis: Axes,
    rows: Sequence[FreeRunRow],
    task: str,
    methods: Sequence[str],
    style: StyleContext,
) -> None:
    """1課題ぶんの有効予測時間パネル (点 + 中央値の横線)。"""
    generator = np.random.default_rng(0)
    for position, method in enumerate(methods):
        selected = [row for row in rows if row.task == task and row.method == method]
        if not selected:
            continue
        values = np.asarray(
            [row.valid_time_lyapunov for row in selected], dtype=np.float64
        )
        jitter = generator.uniform(-0.12, 0.12, values.size)
        censored = np.asarray([row.valid_time_censored for row in selected], dtype=bool)
        axis.scatter(
            position + jitter[~censored],
            values[~censored],
            s=28,
            color=method_color(method),
            alpha=0.8,
        )
        if bool(np.any(censored)):
            axis.scatter(
                position + jitter[censored],
                values[censored],
                s=44,
                marker="^",
                facecolors="none",
                edgecolors=CENSORED_EDGE_COLOR,
                label=style.label("打ち切り (自走長に到達)", "censored (run ended)"),
            )
        median = float(np.median(values))
        axis.plot([position - 0.25, position + 0.25], [median, median], color="black")
        axis.annotate(
            f"{median:.2f}",
            (position, median),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    axis.set_xticks(np.arange(len(methods), dtype=np.float64))
    axis.set_xticklabels([label_of(METHOD_LABELS, method, style) for method in methods])
    axis.set_xlim(-0.5, len(methods) - 0.5)
    axis.set_title(label_of(TASK_LABELS, task, style))
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles[:1], labels[:1], loc="best", fontsize=8)


def plot_stability_map(
    rows: Sequence[StabilityRow],
    capacity_rows: Sequence[CapacityRow],
    path: Path,
    *,
    style: StyleContext,
) -> Path:
    """実験 4-C: (rho x リーク率) の3態マップ + 4-D の容量との関係。

    ノイズ量ごとにパネルを並べ、格子点の色が3態を表す (レプリケートは多数決、
    ``regime_map``)。**分類は行の値そのもの**であり、図から決めていない
    (仕様 §5 禁止する構造6)。最後のパネルは 4-D: 同じ条件で測った
    ``mc_total`` と有効予測時間の関係を3態で色分けする。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    require_rows(rows)
    noises = sorted({row.state_noise for row in rows})
    with rc_context_for(style):
        figure = _new_figure(3.4 * (len(noises) + 1) + 1.0, 4.2)
        axes = np.atleast_1d(figure.subplots(1, len(noises) + 1))
        for axis, noise in zip(axes[:-1], noises, strict=True):
            _regime_panel(axis, rows, noise, style)
        _capacity_panel(axes[-1], rows, capacity_rows, style)
        handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                color=REGIME_COLORS[regime],
                label=label_of(REGIME_LABELS, regime, style),
            )
            for regime in REGIMES
        ]
        figure.legend(
            handles=handles,
            loc="outside lower center",
            ncols=len(REGIMES),
            fontsize=9,
        )
        figure.suptitle(
            style.label(
                "実験 4-C / 4-D: アトラクタを再現できる領域は"
                "状態ノイズ 0.01 までほとんど動かない",
                "Experiments 4-C / 4-D: the region that reproduces the attractor"
                " barely moves up to a state noise of 0.01",
            )
        )
        conditions = f"N = {rows[0].n_units}, stats = {rows[0].stats_steps} steps"
        add_provenance(figure, conditions, rows, style=style)
        return _save(figure, path)


def _regime_panel(
    axis: Axes, rows: Sequence[StabilityRow], noise: float, style: StyleContext
) -> None:
    """状態ノイズ 1 点ぶんの3態マップ (格子点を色で塗る)。"""
    mapping = regime_map(rows, noise)
    rhos = sorted({key[0] for key in mapping})
    leaks = sorted({key[1] for key in mapping})
    for (rho, leak), regime in mapping.items():
        axis.scatter(
            [rho],
            [leak],
            s=420,
            marker="s",
            color=REGIME_COLORS[regime],
            edgecolors="white",
        )
    axis.set_xticks(rhos)
    axis.set_yticks(leaks)
    axis.set_xlim(min(rhos) - 0.15, max(rhos) + 0.15)
    axis.set_ylim(min(leaks) - 0.12, max(leaks) + 0.12)
    axis.set_xlabel(style.label("スペクトル半径 rho", "spectral radius rho"))
    axis.set_ylabel(style.label("リーク率", "leak rate"))
    axis.set_title(
        style.label(f"状態ノイズ = {noise:g}", f"state noise = {noise:g}"), fontsize=10
    )


def _capacity_panel(
    axis: Axes,
    rows: Sequence[StabilityRow],
    capacity_rows: Sequence[CapacityRow],
    style: StyleContext,
) -> None:
    """4-D: ``mc_total`` と有効予測時間の関係 (条件キーで join する)。"""
    keyed = {
        (row.rho, row.leak_rate, row.state_noise, row.replicate): row
        for row in capacity_rows
    }
    for regime in REGIMES:
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            if row.regime != regime:
                continue
            capacity = keyed.get(
                (row.rho, row.leak_rate, row.state_noise, row.replicate)
            )
            if capacity is None:
                continue
            xs.append(capacity.mc_total)
            ys.append(row.valid_time_lyapunov)
        if xs:
            axis.scatter(xs, ys, s=26, alpha=0.75, color=REGIME_COLORS[regime])
    axis.set_xlabel(style.label("線形メモリ容量 MC", "linear memory capacity MC"))
    axis.set_ylabel(
        style.label("有効予測時間 [Lyapunov 時間]", "valid time [1 / lambda_max]")
    )
    axis.set_title(
        style.label(
            "4-D: 同じ状態行列で測った容量", "4-D: capacity on the same states"
        ),
        fontsize=10,
    )


def plot_freerun_stats(
    rows: Sequence[FreeRunProfileRow], path: Path, *, style: StyleContext
) -> Path:
    """実験 4-B: 長時間統計の比較 (リターンマップ + パワースペクトル)。

    上段がリターンマップ ``(z_n, z_(n+1))``、下段が正規化パワースペクトル
    (対数)。どちらも真の軌道と自走を重ねる。**図は距離の値を作らない** ——
    定量的な結論は ``freerun.csv`` の ``return_map_distance`` /
    ``spectrum_distance`` とそのシャッフル代替の列が持つ (D-46)。

    Raises:
        ValueError: 描く点が1つも無い場合。
    """
    tasks = tasks_of(rows)
    if not tasks:
        raise ValueError("profile 行が空です")
    with rc_context_for(style):
        figure = _new_figure(5.4 * len(tasks), 8.0)
        axes = np.atleast_2d(figure.subplots(2, len(tasks), squeeze=False))
        drawn = 0
        for column, task in enumerate(tasks):
            for source in (SOURCE_TRUTH, SOURCE_FREERUN):
                points = profile_points(rows, task, KIND_RETURN_MAP, source)
                if points.shape[0] > 0:
                    axes[0][column].scatter(
                        points[:, 0],
                        points[:, 1],
                        s=12,
                        alpha=0.7,
                        color=SOURCE_STYLE[source][0],
                        label=source_label(source, style),
                    )
                    drawn += 1
                spectrum = profile_points(rows, task, KIND_SPECTRUM, source)
                if spectrum.shape[0] > 0:
                    axes[1][column].plot(
                        spectrum[:, 0],
                        np.maximum(spectrum[:, 1], 1.0e-12),
                        linewidth=1.0,
                        color=SOURCE_STYLE[source][0],
                        label=source_label(source, style),
                    )
                    drawn += 1
            axes[0][column].set_title(label_of(TASK_LABELS, task, style))
            axes[0][column].set_xlabel(style.label("極大値 z_n", "maximum z_n"))
            axes[0][column].set_ylabel(
                style.label("次の極大値 z_(n+1)", "next maximum z_(n+1)")
            )
            axes[0][column].legend(loc="best", fontsize=8)
            axes[1][column].set_yscale("log")
            axes[1][column].set_xlabel(style.label("周波数 [1/時間]", "frequency"))
            axes[1][column].set_ylabel(style.label("正規化パワー", "normalized power"))
            axes[1][column].legend(loc="best", fontsize=8)
        if drawn == 0:
            raise ValueError("長時間統計に描く点がありません")
        figure.suptitle(
            style.label(
                "実験 4-B: 長時間自走後もリターンマップとスペクトルは"
                "シャッフル代替よりはるかに真の軌道へ近い",
                "Experiment 4-B: after a long free run the return map and the"
                " spectrum stay far closer to the truth than the shuffled"
                " surrogate",
            )
        )
        conditions = f"tasks = {'/'.join(tasks)}"
        add_provenance(figure, conditions, rows, style=style)
        return _save(figure, path)


__all__ = [
    "METHOD_LABELS",
    "REGIME_COLORS",
    "REGIME_LABELS",
    "TASK_LABELS",
    "plot_freerun_attractor",
    "plot_freerun_stats",
    "plot_onestep",
    "plot_stability_map",
    "plot_valid_time",
    "profile_points",
]

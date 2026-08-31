"""実験 3-A: 線形メモリ容量の掃引 (``fig_mc_sweep``).

``figures_capacity.py`` から分けてあるのは行数上限 (D-77) のためである。
**上限のほうは緩めない**。

左パネルは**縦軸を破断**する (2-7)。上限 N=200 と実測 10〜36 を1本の軸に
描くと、対数にしても実測の帯より広い空白が上に残る。上段に上限線、下段に
実測を置き、``MC / N`` の第2軸で隔たりを数値として読ませる。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from rc_basics_lab.experiment.capacity import CapacityProfileRow, CapacityRow
from rc_basics_lab.plotting.broken_axis import draw_break_marks, needs_break
from rc_basics_lab.plotting.capacity_grids import (
    mc_profile_means,
    mean_std,
    representative_leak_rate,
    sweep_conditions,
)
from rc_basics_lab.plotting.labels import MC_BOUND_SOURCE, cited_bound
from rc_basics_lab.plotting.layout import label_panels
from rc_basics_lab.plotting.style import (
    StyleContext,
    add_provenance,
    new_figure,
    rc_context_for,
    reference_line_kwargs,
    require_rows,
    sequential_colors,
    unique_sorted,
)

# テストがこの名前を monkeypatch して保存直前の ``Figure`` を捕まえる
# (``figures_esp`` と同じ手口)。別名にしておけば ``_save`` の定義が
# 増えず、重複名ラチェット (D-91) にも触れない。
from rc_basics_lab.plotting.style import save_png as _save
from rc_basics_lab.types import FloatArray


def _add_normalized_axis(axis: Axes, n_units: float) -> None:
    """``MC / N`` の第2軸を右側に足す (FIG-4)。

    上限 200 と実測 10〜36 は対数軸でも離れすぎていて、「上限からどれだけ
    遠いか」が読めない。同じ目盛を N で割った軸を並べると、実測 36 が上限の
    0.18 であることが図の中で読める。

    **軸ラベルは付けない**。第2軸は constrained layout の管理外なので、
    ラベルを付けると右隣のパネルの軸ラベルに重なる (実測)。「右軸が何か」は
    パネルのタイトルに書く。
    """
    if n_units <= 0.0:
        raise ValueError(f"n_units は正である必要があります: {n_units}")

    def forward(values: npt.ArrayLike) -> FloatArray:
        scaled: FloatArray = np.asarray(values, dtype=np.float64) / n_units
        return scaled

    def inverse(values: npt.ArrayLike) -> FloatArray:
        scaled: FloatArray = np.asarray(values, dtype=np.float64) * n_units
        return scaled

    axis.secondary_yaxis("right", functions=(forward, inverse))


def _mc_total_stats(
    rows: Sequence[CapacityRow], rhos: Sequence[float], leak: float
) -> list[tuple[float, float]]:
    """1リーク率ぶんの ``(平均, s.d.)`` を rho 順に返す。"""
    return [
        mean_std(
            [row.mc_total for row in rows if row.rho == rho and row.leak_rate == leak]
        )
        for rho in rhos
    ]


def _plot_mc_total_panel(
    axis: Axes, upper: Axes | None, rows: Sequence[CapacityRow], style: StyleContext
) -> None:
    """左パネル: rho x リーク率 の ``mc_total`` と上限線 y=N。

    上限 (本番 N=200) と実測 (10〜36) は 5.5 倍離れている。1本の軸に両方を
    描くと、対数にしても実測の帯より広い空白が上に残る (2-7)。``upper`` が
    与えられたら**軸を破断**し、上段に上限線だけ、下段に実測だけを置く。
    ``upper`` が ``None`` (上限が近い縮小設定) なら1本の軸に両方描く。

    破断すると上限までの見た目の距離は量でなくなるので、``MC / N`` の第2軸を
    下段の右に必ず残す (FIG-4)。N が1つに定まるときだけ引く —— 格子に複数の
    N が混ざっていたら ``MC / N`` は一意に決まらない。
    """
    rhos = unique_sorted([row.rho for row in rows])
    leaks = unique_sorted([row.leak_rate for row in rows])
    colors = sequential_colors(len(leaks))
    top = 0.0
    for index, leak in enumerate(leaks):
        stats = _mc_total_stats(rows, rhos, leak)
        top = max(top, *(mean + std for mean, std in stats))
        axis.errorbar(
            list(rhos),
            [mean for mean, _ in stats],
            yerr=[std for _, std in stats],
            fmt="o-",
            capsize=4,
            color=colors[index],
            label=style.label(f"a = {leak:g}", f"a = {leak:g}"),
        )
    units = sorted({row.n_units for row in rows})
    bound_axis = axis if upper is None else upper
    for index, n_units in enumerate(units):
        bound_axis.axhline(
            float(n_units),
            **reference_line_kwargs(index),
            label=cited_bound(
                style.label(f"上限 MC <= N = {n_units}", f"bound MC <= N = {n_units}"),
                MC_BOUND_SOURCE,
            ),
        )
    if len(units) == 1:
        _add_normalized_axis(axis, float(units[0]))
    axis.set_yscale("log")
    if upper is not None:
        # 下段は実測だけを入れる範囲に詰める。ここを自動任せにすると上限線が
        # 無くても matplotlib が広めに取り、破断した意味が薄れる。
        axis.set_ylim(top=top * 1.25)
        upper.set_yscale("log")
        upper.set_ylim(float(units[-1]) / 1.3, float(units[-1]) * 1.3)
        # 上段に置くのは上限そのものだけ。対数の副目盛りを残すと 3e2 のような
        # 「上限より上」の目盛りが出て、破断で消したはずの空白が戻る。
        upper.set_yticks([float(n_units) for n_units in units])
        upper.set_yticks([], minor=True)
        upper.legend(loc="center left", fontsize=7)
        upper.set_title(
            style.label(
                "線形メモリ容量と上限 N (右軸は MC / N、縦軸は破断)",
                "Linear memory capacity and the bound N"
                " (right axis: MC / N; the y axis is broken)",
            ),
            fontsize=10,
        )
        draw_break_marks(upper, axis)
    else:
        axis.set_title(
            style.label(
                "線形メモリ容量と上限 N (右軸は MC / N)",
                "Linear memory capacity and the bound N (right axis: MC / N)",
            ),
            fontsize=10,
        )
    axis.set_xlabel(style.label("スペクトル半径 rho", "spectral radius rho"))
    axis.set_ylabel(
        style.label(
            "MC_total (レプリケート平均±s.d.)", "MC_total (mean +- s.d. over reps)"
        )
    )
    axis.legend(loc="lower right", fontsize=7, ncols=2)


def _plot_mc_profile_panel(
    axis: Axes,
    rows: Sequence[CapacityRow],
    profile: Sequence[CapacityProfileRow],
    leak_rate: float,
    style: StyleContext,
) -> None:
    """右パネル: 代表リーク率での遅延プロファイルを rho 別に重ねる。

    横軸を対数にすると「rho を上げるとプロファイルが右へ伸びる」が形として
    読める。各 rho の容量重心 (``mc_effective_delay``、受け入れ条件1 が測る量
    そのもの) を同色の縦点線で入れ、図と受け入れ条件の対応を明示する。
    """
    means = mc_profile_means(rows, profile, leak_rate)
    rhos = tuple(means)
    colors = sequential_colors(len(rhos))
    for index, rho in enumerate(rhos):
        cells = means[rho]
        delays = np.arange(1, cells.shape[0] + 1, dtype=np.float64)
        axis.plot(
            delays,
            cells,
            color=colors[index],
            linewidth=1.4,
            label=style.label(f"rho = {rho:g}", f"rho = {rho:g}"),
        )
        effective, _ = mean_std(
            [
                row.mc_effective_delay
                for row in rows
                if row.rho == rho and row.leak_rate == leak_rate
            ]
        )
        if math.isfinite(effective) and effective > 0.0:
            axis.axvline(effective, color=colors[index], linestyle=":", linewidth=1.0)
    axis.set_xscale("log")
    axis.set_xlabel(style.label("遅延 k [ステップ]", "delay k [steps]"))
    axis.set_ylabel(
        style.label(
            "遅延ごとの容量 (しきい値後)", "capacity per delay (after thresholding)"
        )
    )
    axis.set_title(
        style.label(
            f"遅延プロファイル (a = {leak_rate:g}、縦点線は容量重心)",
            f"Delay profile (a = {leak_rate:g}; dotted: centre of mass)",
        ),
        fontsize=10,
    )
    axis.legend(loc="upper right", fontsize=8, ncols=2)


def _mc_sweep_axes(
    figure: Figure, rows: Sequence[CapacityRow]
) -> tuple[Axes, Axes | None, Axes]:
    """3-A の軸を作る。左パネルは上限が遠いときだけ破断する (2-7)。

    Args:
        figure: 描画先。
        rows: 3-A の行 (上限 N と実測の隔たりを見る)。

    Returns:
        ``(実測パネル, 上限パネル または None, プロファイルパネル)``。
    """
    if not needs_break(
        float(max(row.n_units for row in rows)), max(row.mc_total for row in rows)
    ):
        # 縮小設定では上限と実測が近い。空白が無いのに軸を割ると、読者は
        # 無いはずの不連続を探すことになる。
        axes = figure.subplots(1, 2, squeeze=False)
        return axes[0][0], None, axes[0][1]
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 4.0))
    return (
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[:, 1]),
    )


def plot_mc_sweep(
    rows: Sequence[CapacityRow],
    profile: Sequence[CapacityProfileRow],
    path: Path,
    *,
    style: StyleContext,
) -> Path:
    """実験 3-A の図を書く (受け入れ条件1)。

    Args:
        rows: 3-A の行 (``capacity.csv`` と同じ)。
        profile: 3-A の長形式の行 (``capacity_profile.csv`` と同じ、D-38)。
        path: 出力先 PNG。
        style: ``setup_style()`` の戻り値 (ラベル言語の決定に使う)。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    require_rows(rows)
    leak_rate = representative_leak_rate(rows, lambda row: row.mc_total)
    with rc_context_for(style):
        figure = new_figure(
            12.0, 6.2
        )  # 1-6: Zenn の本文幅 700px で潰れないよう比を 2.0 前後に抑える
        total, bound, profile_axis = _mc_sweep_axes(figure, rows)
        # パネル記号は「読者が数えるパネル」に付ける。破断した上下は1枚なので、
        # 記号は左列の**いちばん上**に置く (破断の下に (a) があると、上段が
        # 別のパネルに見える)。
        label_panels([bound if bound is not None else total, profile_axis], style=style)
        _plot_mc_total_panel(total, bound, rows, style)
        _plot_mc_profile_panel(profile_axis, rows, profile, leak_rate, style)
        first = rows[0]
        figure.suptitle(
            style.label(
                "実験 3-A: 線形メモリ容量は rho = 1 付近で最大になるが、"
                "上限 N の2割にも届かない",
                "Experiment 3-A: linear memory capacity peaks near rho = 1"
                " yet stays below 20% of the bound N",
            )
        )
        conditions = sweep_conditions(first)
        add_provenance(figure, conditions, rows, style=style)
        return _save(figure, path)


__all__ = ["plot_mc_sweep"]

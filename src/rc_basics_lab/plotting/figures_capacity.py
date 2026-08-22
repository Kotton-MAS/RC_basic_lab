"""記事03の図5枚 (実験 3-A / 3-B / 3-B' / 3-C).

- ``plot_mc_sweep``: rho x リーク率 に対する線形メモリ容量と、遅延プロファイル
  の伸び (受け入れ条件1)。左は上限線 y=N つきの ``mc_total``、右は代表リーク率
  での遅延プロファイルを rho 別に重ねる。
- ``plot_ipc_profile``: (次数 x 遅延) の容量ヒートマップを rho 別のパネルに
  並べる (受け入れ条件4)。
- ``plot_memory_nonlinearity``: ``ipc_linear`` と ``ipc_nonlinear`` の積み上げで
  線形/非線形の配分の移動を見せる (受け入れ条件4)。
- ``plot_ipc_conservation``: x=N, y=``ipc_total`` に**傾き1の対角線 y=N** を
  重ねる (受け入れ条件2)。この対角線は図の主張そのものなので、線が実際に
  描かれていることを ``test_conservation_figure_draws_the_bound_line`` が固定する。
- ``plot_narma10_control``: 実験 3-C。x=手法 (線形 / 遅延線 / ESN)、y=NMSE。
  参照線 NMSE = 0.16 / 0.107 を**原典未特定と明記した注つき**で引く
  (要件書 未確定1。数字だけを孤立して引かない)。

``figures.py`` / ``figures_esp.py`` と同じ規律に従う: pyplot を使わず ``Figure``
+ ``FigureCanvasAgg`` を直接組み、描画設定は ``matplotlib.rc_context`` で描画中
だけ一時適用する (F-1-008)。ラベルは必ず ``style.label(ja, en)`` を通す (D-10)。

**ギリシャ文字は書かない**: ruff の RUF001/RUF002 が ASCII と紛らわしい文字を
弾くため、ソース中では ``rho`` / ``sigma_u`` と綴る (02 の図と同じ)。

**配列ではなく長形式の行を読む** (D-38): 遅延プロファイルとヒートマップは
``CapacityProfileRow`` (= ``capacity_profile.csv`` の行) から復元する。
長形式には**しきい値後の容量が厳密に正のセルだけ**が在るので、格子に戻すときは
**欠けているセルを 0 で埋める**。ただし打ち切りの外は「容量 0」ではなく
**未計算**なので、ヒートマップでは 0 と別の色にする (FIG-7 / D-88)。
診断はここでは一切走らせない。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.figure import Figure

from rc_basics_lab.experiment.capacity import (
    CapacityProfileRow,
    CapacityRow,
)
from rc_basics_lab.experiment.narma import (
    NARMA10_REFERENCE_NMSE,
    NARMA10_REFERENCE_NOTE,
    NARMA10_REFERENCE_NOTE_EN,
)
from rc_basics_lab.experiment.runner import ResultRow
from rc_basics_lab.plotting.capacity_grids import (
    BOUND_MARGIN,
    conservation_bound,
    even_degree_note,
    ipc_heatmap_means,
    mc_profile_means,
    mean_std,
    representative_leak_rate,
    sweep_conditions,
)
from rc_basics_lab.plotting.heatmap import (
    colormap_with_uncomputed,
    draw_truncation_edges,
    masked_beyond_truncation,
)
from rc_basics_lab.plotting.labels import (
    DAMBRE_2012,
    IPC_BOUND_SOURCE,
    MC_BOUND_SOURCE,
    SOURCE_UNIDENTIFIED,
    cited_bound,
    cited_measurement,
)
from rc_basics_lab.plotting.narma10_panel import (
    REFERENCE_CONDITIONS,
    REFERENCE_LABELS,
    narma10_headline,
    narma10_method_labels,
    narma10_subtitle,
)
from rc_basics_lab.plotting.style import (
    DELAY_LINE_METHOD,
    METHOD_COLORS,
    SEQUENTIAL_CMAP,
    StyleContext,
    add_provenance,
    method_color,
    rc_context_for,
    reference_line_kwargs,
    require_rows,
    save_png,
    sequential_colors,
)
from rc_basics_lab.plotting.style import new_figure as _new_figure
from rc_basics_lab.plotting.style import unique_sorted as _unique_sorted
from rc_basics_lab.types import FloatArray

_MIN_COLOR_MAX = 1.0e-12
"""ヒートマップの配色上限の下駄。

全セルが 0 の縮退ケースで ``Normalize(0, 0)`` を作るとゼロ除算になるため、
上限が正でないときだけ 1.0 に読み替える (下駄そのものは描画に出ない)。
"""

_HEATMAP_GAMMA = 0.5
"""ヒートマップの配色スケール (平方根)。

セルの値域が次数ごとに2桁近く違う (本番の 3-B では次数3 が最大 6.0、次数1 は
1.0 が上限)。線形の配色にすると次数1 の行が真っ暗になり、「rho を上げると
線形 (次数1) の取り分が増え非線形 (次数3) が減る」という受け入れ条件4 の主張の
片側が図から消える。スケールを変えたことは colorbar のラベルに明記する。
"""

_STACK_COLORS = (METHOD_COLORS[DELAY_LINE_METHOD], "#e08214")
"""積み上げ棒 (線形, 非線形) の色 (FIG-5: 単一の真実は ``style`` 側)。"""

"""参照線の凡例 (``NARMA10_REFERENCE_NMSE`` のキーに対応)。

**キーが増減したら図を描く前に落とす** (下記 ``_reference_lines``)。値だけを
実験層に置いて凡例を図の側に持つと、参照点を1本足したときに図から静かに
消える。出典 (ここでは ``SOURCE_UNIDENTIFIED``) は ``cited`` が必ず付ける
(FIG-3 / D-84)。
"""


def _save(figure: Figure, path: Path) -> Path:
    """PNG を書く (``style.save_png`` への薄い委譲)。

    ``tests/test_plotting_capacity.py::captured`` がこの名前を monkeypatch して
    保存直前の ``Figure`` を捕まえるため、ローカル関数として定義する
    (``save_png as _save`` の import エイリアスのままだと、mypy strict の
    implicit-reexport 検査がテストからの ``figures_capacity._save`` 参照を
    "does not explicitly export" として拒否する)。
    """
    return save_png(figure, path)


# --- 3-A: 線形メモリ容量の掃引 ---------------------------------------------


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


def _plot_mc_total_panel(
    axis: Axes, rows: Sequence[CapacityRow], style: StyleContext
) -> None:
    """左パネル: rho x リーク率 の ``mc_total`` と上限線 y=N。

    縦軸を対数にするのは、上限線 (本番 N=200) と実測 (10〜36) を1枚に載せる
    ためである。線形軸だと上限線を入れた瞬間に実測の差が潰れて、受け入れ条件1
    の「rho とともに伸びる」が読めなくなる。

    対数軸にしても「上限からどれだけ遠いか」は目分量になるので、右側に
    ``MC / N`` の第2軸を置く (FIG-4)。N が1つに定まるときだけ引く ——
    格子に複数の N が混ざっていたら ``MC / N`` は一意に決まらない。
    """
    rhos = _unique_sorted([row.rho for row in rows])
    leaks = _unique_sorted([row.leak_rate for row in rows])
    colors = sequential_colors(len(leaks))
    for index, leak in enumerate(leaks):
        stats = [
            mean_std(
                [
                    row.mc_total
                    for row in rows
                    if row.rho == rho and row.leak_rate == leak
                ]
            )
            for rho in rhos
        ]
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
    for index, n_units in enumerate(units):
        axis.axhline(
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
    axis.set_xlabel(style.label("スペクトル半径 rho", "spectral radius rho"))
    axis.set_ylabel(
        style.label(
            "MC_total (レプリケート平均±s.d.)", "MC_total (mean +- s.d. over reps)"
        )
    )
    axis.set_title(
        style.label(
            "線形メモリ容量と上限 N (右軸は MC / N)",
            "Linear memory capacity and the bound N (right axis: MC / N)",
        ),
        fontsize=10,
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
        figure = _new_figure(12.0, 4.8)
        axes = figure.subplots(1, 2, squeeze=False)
        _plot_mc_total_panel(axes[0][0], rows, style)
        _plot_mc_profile_panel(axes[0][1], rows, profile, leak_rate, style)
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


# --- 3-B: 次数 x 遅延 のヒートマップ ---------------------------------------


def _plot_heatmap_panel(
    axis: Axes,
    cells: FloatArray,
    rho: float,
    norm: Normalize,
    style: StyleContext,
    *,
    show_ylabel: bool,
    truncation: Mapping[int, int] | None,
) -> QuadMesh:
    """1つの rho ぶんの (次数 x 遅延) ヒートマップ (配色は呼び出し側と共通)。

    打ち切りの外は ``masked_beyond_truncation`` でマスクし、0 とは別のグレーで
    描く (FIG-7 / D-88)。遅延軸は対数にする —— 打ち切りが次数ごとに 60/20/10/6
    と一桁違うので、線形軸では低遅延側 (容量の大半が在る場所) が潰れる。
    """
    n_degrees, n_delays = cells.shape
    mesh = axis.pcolormesh(
        np.arange(n_delays + 1, dtype=np.float64) + 0.5,
        np.arange(n_degrees + 1, dtype=np.float64) + 0.5,
        masked_beyond_truncation(cells, truncation),
        cmap=colormap_with_uncomputed(SEQUENTIAL_CMAP),
        norm=norm,
        shading="flat",
    )
    draw_truncation_edges(axis, truncation, n_delays)
    axis.set_xscale("log")
    axis.set_yticks(np.arange(1, n_degrees + 1, dtype=np.float64))
    axis.set_xlabel(style.label("遅延 k [ステップ・対数]", "delay k [steps, log]"))
    if show_ylabel:
        axis.set_ylabel(style.label("次数 d", "degree d"))
    axis.set_title(
        style.label(f"rho = {rho:g}", f"rho = {rho:g}"),
        fontsize=10,
    )
    axis.grid(visible=False)
    return mesh


def plot_ipc_profile(
    rows: Sequence[CapacityRow],
    profile: Sequence[CapacityProfileRow],
    path: Path,
    *,
    style: StyleContext,
    max_delay_by_degree: Mapping[int, int] | None = None,
) -> Path:
    """実験 3-B の (次数 x 遅延) ヒートマップを rho 別に並べる (受け入れ条件4)。

    パネルは代表リーク率 1本 x rho 4点。配色は全パネルで共通の上限を使う
    (パネルごとに正規化すると「rho を上げると非線形が減る」という主張が
    色の付け替えで消える)。

    Args:
        rows: 3-B の行。
        profile: 3-B の長形式の行 (D-38)。
        path: 出力先 PNG。
        style: ``setup_style()`` の戻り値。
        max_delay_by_degree: 次数ごとの遅延の打ち切り (``cfg`` 由来)。
            与えると打ち切りの外を「未計算」のグレーに落とす (FIG-7 / D-88)。
            **省略すると全セルが計算済みとして描かれる** —— 打ち切りが
            分からないときに未計算の領域を捏造しないため。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    require_rows(rows)
    leak_rate = representative_leak_rate(rows, lambda row: row.ipc_total)
    means = ipc_heatmap_means(rows, profile, leak_rate)
    rhos = tuple(means)
    ceiling = max((float(cells.max()) for cells in means.values()), default=0.0)
    norm = PowerNorm(gamma=_HEATMAP_GAMMA, vmin=0.0, vmax=max(ceiling, _MIN_COLOR_MAX))

    with rc_context_for(style):
        figure = _new_figure(3.4 * len(rhos) + 1.2, 3.6)
        axes = figure.subplots(1, len(rhos), squeeze=False)
        meshes = [
            _plot_heatmap_panel(
                axes[0][index],
                means[rho],
                rho,
                norm,
                style,
                show_ylabel=index == 0,
                truncation=max_delay_by_degree,
            )
            for index, rho in enumerate(rhos)
        ]
        figure.colorbar(
            meshes[-1],
            ax=list(axes[0]),
            label=style.label(
                "容量 (平方根スケール。灰色 = 未計算)",
                "capacity (sqrt colour scale; grey = not computed)",
            ),
        )
        first = rows[0]
        figure.suptitle(
            style.label(
                "実験 3-B: rho を上げると次数3 の容量が失われ、残るのは次数1 になる",
                "Experiment 3-B: raising rho destroys the degree-3 capacity"
                " and leaves the degree-1 part",
            )
        )
        figure.supxlabel(style.label(*even_degree_note(profile)), fontsize=8)
        conditions = sweep_conditions(first, leak_rate)
        add_provenance(figure, conditions, rows, style=style)
        return _save(figure, path)


# --- 3-B: 線形 / 非線形 の配分 ----------------------------------------------


def _plot_stack_panel(
    axis: Axes,
    rows: Sequence[CapacityRow],
    leak_rate: float,
    style: StyleContext,
    *,
    show_ylabel: bool,
) -> float:
    """1つのリーク率ぶんの積み上げ棒 (線形 + 非線形)。

    Returns:
        このパネルで一番高い点 (総容量 + s.d.)。縦軸の余白を全パネル共通で
        決めるために使う。
    """
    rhos = _unique_sorted([row.rho for row in rows])
    positions = np.arange(len(rhos), dtype=np.float64)
    selected = [
        [row for row in rows if row.rho == rho and row.leak_rate == leak_rate]
        for rho in rhos
    ]
    linear = [mean_std([row.ipc_linear for row in group])[0] for group in selected]
    nonlinear = [
        mean_std([row.ipc_nonlinear for row in group])[0] for group in selected
    ]
    total_std = [mean_std([row.ipc_total for row in group])[1] for group in selected]
    axis.bar(
        positions,
        linear,
        width=0.6,
        color=_STACK_COLORS[0],
        label=style.label("線形 (次数1)", "linear (degree 1)"),
    )
    axis.bar(
        positions,
        nonlinear,
        width=0.6,
        bottom=linear,
        color=_STACK_COLORS[1],
        label=style.label("非線形 (次数2以上)", "nonlinear (degree >= 2)"),
    )
    axis.errorbar(
        positions,
        [a + b for a, b in zip(linear, nonlinear, strict=True)],
        yerr=total_std,
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    for position, low, high, std in zip(
        positions, linear, nonlinear, total_std, strict=True
    ):
        total = low + high
        if total <= 0.0:
            continue
        # 注記は誤差棒の**上**に置く (棒の高さに置くと s.d. と重なって読めない)
        axis.annotate(
            f"{high / total:.0%}",
            (position, total + std),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{rho:g}" for rho in rhos])
    axis.set_xlabel(style.label("スペクトル半径 rho", "spectral radius rho"))
    if show_ylabel:
        axis.set_ylabel(
            style.label(
                "容量 (レプリケート平均、上の数字は非線形の割合)",
                "capacity (mean over reps; label: nonlinear share)",
            )
        )
    axis.set_title(
        style.label(f"a = {leak_rate:g}", f"a = {leak_rate:g}"),
        fontsize=10,
    )
    if show_ylabel:
        # 凡例は左端の1枚だけに出す。全パネルに出すと積み上げと注記に重なる。
        axis.legend(loc="upper left", fontsize=8)
    return max(
        (a + b + c for a, b, c in zip(linear, nonlinear, total_std, strict=True)),
        default=0.0,
    )


def plot_memory_nonlinearity(
    rows: Sequence[CapacityRow], path: Path, *, style: StyleContext
) -> Path:
    """線形容量と非線形容量の配分の移動を積み上げで描く (受け入れ条件4)。

    パネルはリーク率ごと、横軸は rho。棒の上の数値は非線形の割合で、
    「rho を上げると総容量のうち非線形が減る」という主張を数値でも読ませる
    (積み上げの高さだけだと、総容量の減少と配分の移動が分離できない)。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    require_rows(rows)
    leaks = _unique_sorted([row.leak_rate for row in rows])
    with rc_context_for(style):
        figure = _new_figure(4.0 * len(leaks) + 1.0, 4.4)
        axes = figure.subplots(1, len(leaks), squeeze=False, sharey=True)
        ceiling = max(
            _plot_stack_panel(axes[0][index], rows, leak, style, show_ylabel=index == 0)
            for index, leak in enumerate(leaks)
        )
        # 凡例と注記のぶんの余白 (縦軸は sharey なので1枚に設定すれば足りる)
        axes[0][0].set_ylim(0.0, ceiling * 1.35 if ceiling > 0.0 else 1.0)
        first = rows[0]
        figure.suptitle(
            style.label(
                "実験 3-B: rho を上げると非線形容量の割合は下がる",
                "Experiment 3-B: raising rho lowers the nonlinear share"
                " of the capacity",
            )
        )
        conditions = sweep_conditions(first)
        add_provenance(figure, conditions, rows, style=style)
        return _save(figure, path)


# --- 3-B': 保存則 IPC_total <= N --------------------------------------------


def _draw_conservation_bound(
    axis: Axes, units: Sequence[int], style: StyleContext
) -> None:
    """上限線 y=N (傾き1の対角線) を描く (受け入れ条件2、**図の主張そのもの**)。

    ``conservation_bound`` が返す座標をそのまま描く。テストは同じ関数から
    座標を取り、この線が軸に実在することを確かめる。
    """
    x, y = conservation_bound(units)
    axis.plot(
        x,
        y,
        **reference_line_kwargs(),
        label=cited_bound(
            style.label(
                "上限 IPC_total <= N (傾き1)", "bound IPC_total <= N (slope 1)"
            ),
            IPC_BOUND_SOURCE,
        ),
    )


def plot_ipc_conservation(
    rows: Sequence[CapacityRow], path: Path, *, style: StyleContext
) -> Path:
    """実験 3-B' の保存則 ``IPC_total <= N`` を描く (受け入れ条件2)。

    横軸が N、縦軸が ``ipc_total``、線は ``state_noise`` 別。**傾き1の対角線**
    が上限で、ノイズを入れると点が対角線から下へ離れる (ノイズがリザバーの
    自由度を潰し、線形読み出しで取り出せる容量が N に届かなくなる)。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    require_rows(rows)
    units = sorted({row.n_units for row in rows})
    noises = _unique_sorted([row.state_noise for row in rows])
    # 状態ノイズは連続量なので viridis 系にそろえる (FIG-5)。以前は plasma で、
    # 同じ「連続量の掃引」が記事の中で2種類の配色になっていた。
    colors = sequential_colors(len(noises))

    with rc_context_for(style):
        figure = _new_figure(7.2, 5.0)
        axis = figure.subplots(1, 1)
        for index, noise in enumerate(noises):
            stats = [
                mean_std(
                    [
                        row.ipc_total
                        for row in rows
                        if row.n_units == n_units and row.state_noise == noise
                    ]
                )
                for n_units in units
            ]
            axis.errorbar(
                [float(n_units) for n_units in units],
                [mean for mean, _ in stats],
                yerr=[std for _, std in stats],
                fmt="o-",
                capsize=4,
                color=colors[index],
                label=style.label(
                    f"状態ノイズ = {noise:g}", f"state noise = {noise:g}"
                ),
            )
        _draw_conservation_bound(axis, units, style)
        axis.set_xlabel(style.label("リザバーのユニット数 N", "reservoir size N"))
        axis.set_ylabel(
            style.label(
                "IPC_total (レプリケート平均±s.d.)",
                "IPC_total (mean +- s.d. over reps)",
            )
        )
        first = rows[0]
        figure.suptitle(
            style.label(
                "実験 3-B': 状態ノイズを入れると IPC_total は上限 N から下へ離れる\n"
                f"{DAMBRE_2012} の保存則の再実演",
                "Experiment 3-B': state noise pushes IPC_total away from"
                f" the bound N\nRe-enacting the capacity bound of {DAMBRE_2012}",
            )
        )
        axis.legend(loc="upper left", fontsize=8)
        conditions = f"rho = {first.rho:g}, a = {first.leak_rate:g}"
        add_provenance(figure, conditions, rows, style=style)
        return _save(figure, path)


# --- 3-C: 公平な対照での NARMA10 -------------------------------------------


def _reference_lines(axis: Axes, style: StyleContext) -> None:
    """参照線 NMSE = 0.16 / 0.107 を**注記つき**で引く (原典未特定)。

    値は ``experiment/narma.py`` の ``NARMA10_REFERENCE_NMSE`` が単一の真実で、
    ここは描くだけである (``meta.json`` も同じ定数を書く)。凡例の対応表に
    無いキーが在れば描く前に落とす —— 参照点を足したのに図に出ない、を
    黙って通さない。
    """
    missing = set(NARMA10_REFERENCE_NMSE) - set(REFERENCE_LABELS)
    if missing:
        raise ValueError(f"参照線のラベルがありません: {sorted(missing)}")
    for index, (key, value) in enumerate(NARMA10_REFERENCE_NMSE.items()):
        japanese, english = REFERENCE_LABELS[key]
        axis.axhline(
            value,
            **reference_line_kwargs(index),
            label=cited_measurement(
                style.label(japanese.format(value=value), english.format(value=value)),
                style.label(*SOURCE_UNIDENTIFIED),
                style.label(*REFERENCE_CONDITIONS[key]),
            ),
        )


def plot_narma10_control(
    rows: Sequence[ResultRow], path: Path, *, style: StyleContext
) -> Path:
    """実験 3-C: 探索予算をそろえた3手法の NARMA10 成績 (受け入れ条件5)。

    横軸が手法 (線形 / 遅延線(選ばれた k) / ESN)、縦軸がテスト NMSE の
    レプリケート平均±s.d.。参照線 (0.16 / 0.107) は**原典未特定**である旨を
    図の注に書いて引く —— 数字だけを引くと、後から出典が違っていたときに
    図の側から辿れない。

    誤差指標を NMSE にするのは、参照値が NMSE で流通しているからである
    (D-02 の主指標 NRMSE は ``narma10.csv`` の ``nrmse`` 列に併記されている)。

    Raises:
        ValueError: ``rows`` が空、または参照線のラベルが欠けている場合。
    """
    require_rows(rows)
    labels = narma10_method_labels(rows, style)
    positions = np.arange(len(labels), dtype=np.float64)
    stats = [
        mean_std([row.nmse for row in rows if row.method == method])
        for method, _ in labels
    ]
    means = [mean for mean, _ in stats]
    stds = [std for _, std in stats]

    with rc_context_for(style):
        figure = _new_figure(7.2, 5.0)
        axis = figure.subplots(1, 1)
        axis.errorbar(
            positions,
            means,
            # 対数軸なので下側の誤差棒が 0 以下に落ちないよう抑える
            yerr=np.vstack(
                [
                    np.minimum(stds, np.asarray(means, dtype=np.float64) * 0.999),
                    np.asarray(stds, dtype=np.float64),
                ]
            ),
            fmt="o",
            capsize=5,
            color="#666666",
        )
        # 点の色は手法の固定色 (FIG-5)。記事をまたいで ESN は緑、遅延線は青。
        axis.scatter(
            positions, means, c=[method_color(method) for method, _ in labels], zorder=3
        )
        _reference_lines(axis, style)
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
        axis.set_xticklabels([text for _, text in labels])
        axis.set_xlim(-0.5, len(labels) - 0.5)
        n_replicates = len({row.replicate for row in rows})
        axis.set_ylabel(
            style.label(
                f"NMSE (テスト区間・{n_replicates}レプリケートの平均±標準偏差)",
                f"NMSE (test split, mean +- s.d. of {n_replicates} replicates)",
            )
        )
        axis.legend(loc="best", fontsize=8)
        figure.suptitle(
            f"{style.label('実験 3-C', 'Experiment 3-C')}:"
            f" {narma10_headline(rows, style)}\n{narma10_subtitle(style)}"
        )
        figure.supxlabel(
            style.label(NARMA10_REFERENCE_NOTE, NARMA10_REFERENCE_NOTE_EN),
            fontsize=8,
        )
        conditions = f"n_train = {rows[0].n_train}, task = {rows[0].task}"
        add_provenance(figure, conditions, rows, style=style)
        return _save(figure, path)


__all__ = [
    "BOUND_MARGIN",
    "conservation_bound",
    "ipc_heatmap_means",
    "mc_profile_means",
    "narma10_method_labels",
    "plot_ipc_conservation",
    "plot_ipc_profile",
    "plot_mc_sweep",
    "plot_memory_nonlinearity",
    "plot_narma10_control",
    "representative_leak_rate",
]

"""記事03の図5枚 (実験 3-A / 3-B / 3-B' / 3-C).

- ``plot_mc_sweep``: rho x リーク率 の線形メモリ容量と遅延プロファイルの伸び
  (受け入れ条件1)。左は上限線 y=N つきの ``mc_total``、右は代表リーク率での
  遅延プロファイル。
- ``plot_ipc_profile``: (次数 x 遅延) の容量ヒートマップを rho 別のパネルに
  並べる (受け入れ条件4)。
- ``plot_memory_nonlinearity``: ``ipc_linear`` と ``ipc_nonlinear`` の積み上げで
  線形/非線形の配分の移動を見せる (受け入れ条件4)。
- ``plot_ipc_conservation``: x=N, y=``ipc_total`` に**傾き1の対角線 y=N** を
  重ねる (受け入れ条件2)。線が在ることは
  ``test_conservation_figure_draws_the_bound_line`` が固定する。
- ``plot_narma10_control``: 実験 3-C。x=手法、y=NMSE。参照線 0.16 / 0.107 は
  **原典未特定と明記した注つき**で引く (数字だけを孤立して引かない)。

図の外枠 (rcParams の一時適用 / 保存) は ``plotting/style.py`` が持つ。ラベルは
必ず ``style.label(ja, en)`` を通す (D-10)。**ギリシャ文字は書かない**
(RUF001/RUF002。ソース中では ``rho`` / ``sigma_u`` と綴る)。

**配列ではなく長形式の行を読む** (D-38): 遅延プロファイルとヒートマップは
``CapacityProfileRow`` (= ``capacity_profile.csv`` の行) から復元する。
長形式には**しきい値後の容量が厳密に正のセルだけ**が在るので、格子に戻すときは
**欠けているセルを 0 で埋める**。ただし打ち切りの外は「容量 0」ではなく
**未計算**なので、ヒートマップでは 0 と別の色にする (FIG-7 / D-88)。
診断はここでは一切走らせない。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from rc_basics_lab.experiment.capacity_rows import (
    CapacityRow,
)
from rc_basics_lab.experiment.narma import (
    NARMA10_REFERENCE_NMSE,
)
from rc_basics_lab.experiment.runner import ResultRow
from rc_basics_lab.experiment.topology_ladder import TopologyLadderRow
from rc_basics_lab.plotting.capacity_grids import (
    BOUND_MARGIN,
    conservation_bound,
    ipc_heatmap_means,
    mc_profile_means,
    mean_std,
    representative_leak_rate,
    sweep_conditions,
)
from rc_basics_lab.plotting.figures_ladder import (
    LADDER_ARTICLE_AXES,
    draw_ladder_panel,
    ladder_headline,
)
from rc_basics_lab.plotting.labels import (
    DAMBRE_2012,
    IPC_BOUND_SOURCE,
    cited_bound,
    cited_measurement,
)
from rc_basics_lab.plotting.layout import label_panels
from rc_basics_lab.plotting.narma10_panel import (
    REFERENCE_CONDITIONS,
    REFERENCE_LABELS,
    REFERENCE_SOURCES,
    narma10_headline,
    narma10_method_labels,
)
from rc_basics_lab.plotting.style import (
    DELAY_LINE_METHOD,
    METHOD_COLORS,
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


# --- 3-A (``plot_mc_sweep``) は figures_mc_sweep.py にある (D-77) ----------


# --- 3-B: 次数 x 遅延 のヒートマップ ---------------------------------------


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
        figure = _new_figure(
            4.0 * len(leaks) + 1.0, 6.4
        )  # 1-6: Zenn の本文幅 700px で潰れないよう比を 2.0 前後に抑える
        axes = figure.subplots(1, len(leaks), squeeze=False, sharey=True)
        label_panels(list(axes[0]), style=style)
        ceiling = max(
            _plot_stack_panel(axes[0][index], rows, leak, style, show_ylabel=index == 0)
            for index, leak in enumerate(leaks)
        )
        # 凡例と注記のぶんの余白 (縦軸は sharey なので1枚に設定すれば足りる)
        axes[0][0].set_ylim(0.0, ceiling * 1.35 if ceiling > 0.0 else 1.0)
        # 1 行なので全パネルが最下段にあたる。軸ラベルを各パネルに置くと
        # 同じ文字列が 3 回並ぶので、figure に1つだけ出す (1-7)。
        figure.supxlabel(style.label("スペクトル半径 rho", "spectral radius rho"))
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
    """上限線 (比 = 1 の水平線) を描く (受け入れ条件2、**図の主張そのもの**)。

    ``conservation_bound`` が返す座標をそのまま描く。テストは同じ関数から
    座標を取り、この線が軸に実在することを確かめる。
    """
    x, y = conservation_bound(units)
    axis.plot(
        x,
        y,
        **reference_line_kwargs(),
        label=cited_bound(
            style.label("上限 IPC_total / N <= 1", "bound IPC_total / N <= 1"),
            IPC_BOUND_SOURCE,
        ),
    )


def plot_ipc_conservation(
    rows: Sequence[CapacityRow],
    ladder_rows: Sequence[TopologyLadderRow],
    path: Path,
    *,
    style: StyleContext,
) -> Path:
    """保存則 (3-B') と対照の梯子 (3-T) を1枚に並べる (受け入れ条件2 / D-146)。

    左は保存則そのもの: 横軸が N、縦軸が ``ipc_total / N``、線は
    ``state_noise`` 別。**水平線 1** が上限で、ノイズを入れると点が下へ離れる。

    右2枚は**その容量がトポロジで動くか**である。同じ図に置くのは、
    「容量には上限がある」と「その中でトポロジは効くのか」が読者にとって
    連続した1つの問いだからで、別々の図にすると『上限の話』と『構造の話』が
    無関係に見える。

    梯子は総容量ではなく**対応のある差**で描く (D-145)。N や rho で絶対値が
    2桁動くので、総容量では水準間の差が潰れて読めない。

    Args:
        rows: ``capacity.csv`` の行 (左のパネル)。
        ladder_rows: ``capacity_topology.csv`` の行 (右2枚)。
        path: 出力先の PNG。
        style: 配色・言語・commit。

    Returns:
        書き出した PNG のパス。

    Raises:
        ValueError: いずれかが空、または梯子に N / rho の掃引が無い場合。
    """
    require_rows(rows)
    units = sorted({row.n_units for row in rows})
    noises = _unique_sorted([row.state_noise for row in rows])
    # 状態ノイズは連続量なので viridis 系にそろえる (FIG-5)。以前は plasma で、
    # 同じ「連続量の掃引」が記事の中で2種類の配色になっていた。
    colors = sequential_colors(len(noises))

    require_rows(ladder_rows)
    with rc_context_for(style):
        figure = _new_figure(15.0, 5.0)
        grid = figure.add_gridspec(1, 3)
        axis = figure.add_subplot(grid[0, 0])
        for index, noise in enumerate(noises):
            # **比 (ipc_total / N) を縦軸にする** (2-7)。絶対値だと上限線が斜めになり、
            # 「上限からどれだけ離れたか」が目分量になる。比なら水平線 1 から縦に読む。
            stats = [
                mean_std(
                    [
                        row.ipc_total / n_units
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
        ladder_axes = [figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[0, 2])]
        # 梯子の4軸のうち **N と rho** を出す。ノイズと T は「順位が動かない」
        # ことの確認であって、動く軸ではない (CSV に全件ある)。
        for panel, axis_name in zip(ladder_axes, LADDER_ARTICLE_AXES, strict=True):
            draw_ladder_panel(panel, ladder_rows, axis_name, "mc_total", style)
        ladder_axes[0].legend(loc="best", fontsize=7)
        label_panels([axis, *ladder_axes], style=style)
        axis.set_xlabel(style.label("リザバーのユニット数 N", "reservoir size N"))
        axis.set_ylabel(
            style.label(
                "IPC_total / N (レプリケート平均±s.d.)",
                "IPC_total / N (mean +- s.d. over reps)",
            )
        )
        first = rows[0]
        figure.suptitle(
            style.label(
                "実験 3-B' / 3-T: 容量には上限があり、"
                f"その中で {ladder_headline(ladder_rows, style)}\n"
                f"{DAMBRE_2012} の保存則の再実演と、交差する交絡を剥がした対照",
                "Experiments 3-B' / 3-T: capacity is bounded, and within that"
                f" bound {ladder_headline(ladder_rows, style)}\n"
                f"Re-enacting the capacity bound of {DAMBRE_2012} with"
                " confound-stripping controls",
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
                REFERENCE_SOURCES[key],
                style.label(*REFERENCE_CONDITIONS[key]),
            ),
        )


def draw_narma10_control_panel(
    axis: Axes, rows: Sequence[ResultRow], style: StyleContext
) -> None:
    """実験 3-C (探索予算をそろえた3手法の NARMA10 成績) を1つの軸に描く。

    単独の figure をやめてパネルにしたのは FIG-12 / C-6 による (NARMA10 の 3 枚は
    同じ主張を支えている)。

    Args:
        axis: 描画先。
        rows: ``narma10.csv`` と同じ行。
        style: 配色・言語。

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
    axis.set_xticklabels([text for _, text in labels], fontsize=8)
    axis.set_xlim(-0.5, len(labels) - 0.5)
    n_replicates = len({row.replicate for row in rows})
    axis.set_ylabel(
        style.label(
            f"NMSE ({n_replicates}レプリケートの平均±s.d.)",
            f"NMSE (mean +- s.d. of {n_replicates} replicates)",
        )
    )
    axis.set_title(narma10_headline(rows, style), fontsize=9)
    axis.legend(loc="best", fontsize=7)


__all__ = [
    "BOUND_MARGIN",
    "conservation_bound",
    "draw_narma10_control_panel",
    "ipc_heatmap_means",
    "mc_profile_means",
    "narma10_method_labels",
    "plot_ipc_conservation",
    "plot_memory_nonlinearity",
    "representative_leak_rate",
]

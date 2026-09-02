"""実験 3-T の図 —— 対照の梯子が掃引のどこでも同じ結論を出すこと (D-145).

梯子が答えるのは「BA の優位/劣位は次数分布で説明できるか」である
(D-138)。その答えが**掃引の1点で選んだ結果ではない**ことを示すには、
軸ごとに差を並べるしかない。

描くのは総容量ではなく**対応のある差**である:

- ``BA - ER``: 先行が主張する優位が出ているか
- ``BA - 次数保存ランダム化``: 出ているとして、それは次数分布で説明できるか

総容量を水準ごとに並べると、N や rho で絶対値が2桁動くぶん水準間の差が
潰れて読めない (実測: N=25 で MC 約10、N=100 で約19)。差にすれば軸をまたい
で同じ縦軸で読める。

**0 の線が結論そのもの**である。上にあれば BA が良く、下にあれば悪い。
2本目 (次数保存との差) が 0 の線に張り付いていれば、BA の効果は次数分布に
帰着する。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from rc_basics_lab.experiment.topology_ladder import TopologyLadderRow
from rc_basics_lab.metrics_significance import sign_test_p_value
from rc_basics_lab.plotting.layout import (
    hide_minor_tick_labels,
    label_panels,
    wrapped_note,
)
from rc_basics_lab.plotting.style import (
    REFERENCE_COLOR,
    REFERENCE_DASHES,
    StyleContext,
    add_footnote,
    new_figure,
    rc_context_for,
    save_png,
)
from rc_basics_lab.types import FloatArray

BASELINE_LEVEL = "erdos_renyi"
"""比較の基準にする水準 (交絡を1つも動かしていない側)。"""

NULL_MODEL_LEVEL = "degree_preserving"
"""本命の帰無モデル (BA の次数列だけを残した水準、D-135)。"""

TARGET_LEVEL = "barabasi_albert"
"""主張の対象 (先行が優れると報告している水準)。"""

AXIS_LABELS: dict[str, tuple[str, str]] = {
    "n_units": ("ユニット数 N", "units N"),
    "rho": ("スペクトル半径 rho", "spectral radius rho"),
    "state_noise": ("状態ノイズ", "state noise"),
    "n_steps": ("系列長 T [ステップ]", "series length T [steps]"),
}
"""掃引軸の見出し。**ここに無い軸は描く前に落とす** (FIG-5 と同じ規律)。"""


def _paired_differences(
    rows: Sequence[TopologyLadderRow], other: str, column: str
) -> FloatArray:
    """同じ (グラフ, 重み) で対にした ``TARGET_LEVEL - other`` の並び。

    対にしないとグラフの実現値の分散が差に混ざる (D-134)。
    """
    target = {
        (row.graph, row.replicate): float(getattr(row, column))
        for row in rows
        if row.level == TARGET_LEVEL
    }
    values = [
        target[(row.graph, row.replicate)] - float(getattr(row, column))
        for row in rows
        if row.level == other and (row.graph, row.replicate) in target
    ]
    return np.asarray(values, dtype=np.float64)


def _difference_series(
    rows: Sequence[TopologyLadderRow], axis_name: str, other: str, column: str
) -> tuple[list[float], list[float], list[float]]:
    """1つの軸に沿った (軸の値, 差の平均, 差の s.d.) を昇順で返す。"""
    at_axis = [row for row in rows if row.sweep_axis == axis_name]
    points = sorted({float(getattr(row, axis_name)) for row in at_axis})
    xs: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    for value in points:
        group = [row for row in at_axis if float(getattr(row, axis_name)) == value]
        diffs = _paired_differences(group, other, column)
        if diffs.size == 0:
            continue
        xs.append(value)
        means.append(float(np.mean(diffs)))
        stds.append(float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0)
    return xs, means, stds


def ladder_headline(rows: Sequence[TopologyLadderRow], style: StyleContext) -> str:
    """タイトルの結論文を**行から導く** (固定文にしない、D-90 と同じ規律)。

    掃引を広げて結論が変わったときに図が静かに嘘をつくのを防ぐ。
    """
    wins = 0
    total = 0
    for axis_name in dict.fromkeys(row.sweep_axis for row in rows if row.sweep_axis):
        _, means, _ = _difference_series(rows, axis_name, BASELINE_LEVEL, "mc_total")
        total += len(means)
        wins += sum(1 for value in means if value > 0.0)
    if total == 0:
        raise ValueError("掃引の点が1つもありません")
    if wins == 0:
        return style.label(
            f"BA は掃引した {total} 点のどこでも ER を上回らない",
            f"BA never beats ER at any of the {total} swept points",
        )
    return style.label(
        f"BA が ER を上回るのは {total} 点中 {wins} 点",
        f"BA beats ER at {wins} of {total} swept points",
    )


def draw_ladder_panel(
    axis: Axes,
    rows: Sequence[TopologyLadderRow],
    axis_name: str,
    column: str,
    style: StyleContext,
) -> None:
    """1つの掃引軸ぶんの対応のある差を描く。

    Args:
        axis: 描画先。
        rows: 梯子の行 (``capacity_topology.csv`` と同じ)。
        axis_name: 掃引軸の名前 (``AXIS_LABELS`` にあること)。
        column: 縦軸にする列 (``mc_total`` など)。
        style: 配色・言語。

    Raises:
        ValueError: 未知の軸、またはその軸の行が無い場合。
    """
    rows = [row for row in rows if row.sweep_axis == axis_name]
    if axis_name not in AXIS_LABELS:
        raise ValueError(
            f"軸の見出しが決まっていません: {axis_name!r} "
            f"(AXIS_LABELS に足してください: {sorted(AXIS_LABELS)})"
        )
    for other, marker, label in (
        (
            BASELINE_LEVEL,
            "o-",
            style.label("BA - ER (全部の交絡)", "BA - ER (all confounds)"),
        ),
        (
            NULL_MODEL_LEVEL,
            "s--",
            style.label(
                "BA - 次数保存ランダム化 (相関構造だけ)",
                "BA - degree-preserving null (correlation only)",
            ),
        ),
    ):
        xs, means, stds = _difference_series(rows, axis_name, other, column)
        if not xs:
            raise ValueError(f"{axis_name} の行がありません")
        axis.errorbar(
            xs,
            means,
            yerr=np.asarray(stds, dtype=np.float64),
            fmt=marker,
            capsize=4,
            label=label,
        )
    axis.axhline(
        0.0,
        color=REFERENCE_COLOR,
        dashes=REFERENCE_DASHES[0],
        label=style.label("差 0 (優劣なし)", "no difference"),
    )
    if axis_name in ("n_units", "n_steps"):
        axis.set_xscale("log")
        # **測った点だけに目盛りを置く** (FIG-19)。対数軸の既定は 10^2 の
        # ような桁の位置にしか主目盛りを置かないので、25 / 50 / 100 を
        # 振っても軸には何も出ない (実測でそうなった)。
        measured = sorted({float(getattr(row, axis_name)) for row in rows})
        axis.set_xticks(measured)
        axis.set_xticklabels([f"{value:g}" for value in measured])
        hide_minor_tick_labels(axis, which="x")
    axis.set_xlabel(style.label(*AXIS_LABELS[axis_name]))
    axis.set_ylabel(style.label("対応のある差 [ビット]", "paired difference [bits]"))


def plot_ladder(
    rows: Sequence[TopologyLadderRow],
    path: Path,
    *,
    style: StyleContext,
) -> Path:
    """3-T の梯子を掃引軸ごとに1枚へ並べる (D-145)。

    Args:
        rows: ``capacity_topology.csv`` と同じ行。
        path: 出力先の PNG。
        style: 配色・言語・commit。

    Returns:
        書き出した PNG のパス。

    Raises:
        ValueError: 行が空、または掃引軸が見出し表に無い場合。
    """
    if not rows:
        raise ValueError("rows が空です")
    # **並びは行の出現順**にする (= config の sweeps の宣言順)。名前の
    # アルファベット順にすると T が先頭に来て、問いの順 (N -> ノイズ -> rho
    # -> T) と図の順が食い違う。
    axes_present = list(dict.fromkeys(row.sweep_axis for row in rows if row.sweep_axis))
    if not axes_present:
        raise ValueError("掃引の行がありません (sweep_axis が空の行だけです)")
    with rc_context_for(style):
        figure = new_figure(14.0, 8.0)
        grid = figure.add_gridspec(2, 2)
        drawn: list[Axes] = []
        for index, axis_name in enumerate(axes_present[:4]):
            panel = figure.add_subplot(grid[index // 2, index % 2])
            draw_ladder_panel(panel, rows, axis_name, "mc_total", style)
            drawn.append(panel)
        drawn[0].legend(loc="best")
        label_panels(drawn, style=style)
        figure.suptitle(f"3-T: {ladder_headline(rows, style)}")
        figure.supxlabel(
            wrapped_note(
                style.label(
                    "縦軸は同じグラフ・同じ重み行列で対にした差 (D-134)。"
                    "密度は全水準で実測がそろえてある (D-140)。"
                    "誤差棒は対ごとの差の標準偏差。",
                    "The y axis is a paired difference over the same graph and"
                    " weight matrix (D-134). Density is matched across levels"
                    " by measurement (D-140). Error bars are the s.d. of the"
                    " paired differences.",
                )
            )
        )
        add_footnote(figure, _ladder_conditions(rows, style), style=style)
        return save_png(figure, path)


def _ladder_conditions(rows: Sequence[TopologyLadderRow], style: StyleContext) -> str:
    """footnote の再現条件 (FIG-6)。"""
    pairs = len({(row.graph, row.replicate) for row in rows})
    levels = len({row.level for row in rows})
    return style.label(
        f"3-T, 水準 {levels}, 対 {pairs} (グラフ x 重み)",
        f"3-T, {levels} levels, {pairs} pairs (graph x weight)",
    )


def paired_sign_test(
    rows: Sequence[TopologyLadderRow], other: str, column: str
) -> tuple[float, float]:
    """``(差の平均, 符号検定の p)`` を返す (図の外でも使う)。

    Raises:
        ValueError: 対が1組も作れない場合。
    """
    diffs = _paired_differences(rows, other, column)
    if diffs.size == 0:
        raise ValueError("対が作れません")
    wins = int(np.sum(diffs > 0.0))
    return float(np.mean(diffs)), sign_test_p_value(int(diffs.size), wins)


__all__ = [
    "AXIS_LABELS",
    "BASELINE_LEVEL",
    "NULL_MODEL_LEVEL",
    "TARGET_LEVEL",
    "draw_ladder_panel",
    "ladder_headline",
    "paired_sign_test",
    "plot_ladder",
]

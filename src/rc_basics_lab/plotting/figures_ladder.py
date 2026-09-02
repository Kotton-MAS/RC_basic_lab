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

import numpy as np
from matplotlib.axes import Axes

from rc_basics_lab.experiment.topology_ladder import TopologyLadderRow
from rc_basics_lab.metrics_significance import sign_test_p_value
from rc_basics_lab.plotting.labels import cited_bound
from rc_basics_lab.plotting.layout import (
    hide_minor_tick_labels,
)
from rc_basics_lab.plotting.style import (
    REFERENCE_COLOR,
    REFERENCE_DASHES,
    StyleContext,
)
from rc_basics_lab.types import FloatArray

BASELINE_LEVEL = "erdos_renyi"
"""比較の基準にする水準 (交絡を1つも動かしていない側)。"""

NULL_MODEL_LEVEL = "degree_preserving"
"""本命の帰無モデル (BA の次数列だけを残した水準、D-135)。"""

TARGET_LEVEL = "barabasi_albert"
"""主張の対象 (先行が優れると報告している水準)。"""

DEGREE_PRESERVING_SOURCE = "Maslov & Sneppen 2002"
"""次数保存ランダム化 (double edge swap) の出典。

ネットワーク科学では標準の帰無モデルだが、**RC のトポロジ論文ではまず
置かれない** (D-135)。0 の線が「差が無いはず」を意味する根拠がここにある。
"""

LADDER_ARTICLE_AXES: tuple[str, ...] = ("n_units", "rho")
"""記事の図に出す掃引軸 (D-146)。

4軸すべてを出すと保存則のパネルと合わせて5枚になり、1記事あたりの図の
上限 (FIG-12 / D-112) を守れない。**N と rho を出す** —— ノイズと T は
「順位が動かない」ことの確認であって、動く軸ではないからである
(全件は ``capacity_topology.csv`` にある)。
"""

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
    # 0 の線は**この図の主張そのもの** (上なら BA が良い)。出典は帰無モデルの
    # 側にある —— 差が 0 であることを期待するのは次数保存ランダム化が
    # 帰無モデルだからで、その根拠は D-135 が引く標準的な構成法である。
    axis.axhline(
        0.0,
        color=REFERENCE_COLOR,
        dashes=REFERENCE_DASHES[0],
        label=cited_bound(
            style.label("差 0 (優劣なし)", "no difference"),
            DEGREE_PRESERVING_SOURCE,
        ),
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
    "LADDER_ARTICLE_AXES",
    "NULL_MODEL_LEVEL",
    "TARGET_LEVEL",
    "draw_ladder_panel",
    "ladder_headline",
    "paired_sign_test",
]

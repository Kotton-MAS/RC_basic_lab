"""連載の第1図 —— ESN の模式図 (FIG-11 / D-107).

## なぜ matplotlib で描くのか

手描き SVG は見た目で勝るが、**再生成できないため `artifact_manifest` と
D-74 の規律の外に出る**。コードと乖離しても誰も気づけない状態を、
記事の第1図で作らない。

## 何を書き、何を書かないか (D-107)

主役は**「固定」と「学習」の二色分け**の1点だけである。
リーク率・washout・リッジの正則化は**図に入れない** —— 入れると密度が
上がりすぎて、第1図の役割 (何が動いているかを見せる) を果たさなくなる。
それらは記事本文へ逃がす。

図の中の記号は ``docs/design.md`` の式と同じものを使う
(``W_in`` / ``W`` / ``W_out`` / ``x`` / ``u`` / ``y``)。
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from rc_basics_lab.plotting.style import (
    FIGURE_SIZES,
    StyleContext,
    new_figure,
    rc_context_for,
    save_png,
)

FIXED_COLOR = "#8c8c8c"
"""**学習しない**部分の色 (D-107)。無彩色にするのは、手法4色 (D-85) とも
参照線の黒 (D-86) とも衝突しないためである。"""

TRAINED_COLOR = "#b2182b"
"""**学習する**部分の色 (D-107)。図の中でここだけが有彩色になる。

ESN の要点は「``W_out`` だけを学習する」ことなので、**色が付いている箇所が
1つだけ**という状態そのものが主張になる。"""

_RESERVOIR_NODES: tuple[tuple[float, float], ...] = (
    (0.44, 0.72),
    (0.52, 0.80),
    (0.60, 0.70),
    (0.47, 0.55),
    (0.58, 0.52),
    (0.51, 0.64),
)
"""リザバー内のノードの位置。**個数と配置に意味は無い** ——
「たくさんあって互いに繋がっている」以上のことを言わせない。"""


def plot_architecture(path: Path, *, style: StyleContext) -> Path:
    """ESN の模式図を描く (FIG-11 追加図1 / D-107)。

    Args:
        path: 出力先の PNG。
        style: 言語 (ja / en) と commit。**データを取らない図**なので
            行は受け取らない。

    Returns:
        書き出した PNG のパス。
    """
    width, height = FIGURE_SIZES["single"]
    with rc_context_for(style):
        figure = new_figure(width, height)
        axis = figure.subplots(1, 1)
        axis.set_xlim(0.0, 1.0)
        # 内容は y = 0.24〜0.96 にしか無い。1.0 まで取ると下半分が空白になる。
        axis.set_ylim(0.24, 0.96)
        axis.axis("off")

        # --- リザバー (固定) ---
        axis.add_patch(
            FancyBboxPatch(
                (0.36, 0.42),
                0.34,
                0.46,
                boxstyle="round,pad=0.02",
                linewidth=1.6,
                edgecolor=FIXED_COLOR,
                facecolor="none",
            )
        )
        # Circle だと軸の縦横比で楕円に潰れる。マーカーは常に真円になる。
        axis.scatter(
            [x for x, _ in _RESERVOIR_NODES],
            [y for _, y in _RESERVOIR_NODES],
            s=170,
            color=FIXED_COLOR,
            zorder=3,
        )
        for start in _RESERVOIR_NODES:
            for end in _RESERVOIR_NODES:
                if start >= end:
                    continue
                axis.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    color=FIXED_COLOR,
                    linewidth=0.6,
                    alpha=0.45,
                    zorder=2,
                )
        axis.text(
            0.53,
            0.92,
            style.label("リザバー (固定)", "reservoir (fixed)"),
            ha="center",
            va="bottom",
            color=FIXED_COLOR,
            fontsize=11,
        )
        axis.text(0.53, 0.46, "W", ha="center", va="center", color=FIXED_COLOR)

        # --- 入力 -> リザバー (固定) ---
        _arrow(axis, (0.14, 0.65), (0.35, 0.65), FIXED_COLOR)
        axis.text(0.10, 0.65, "u", ha="center", va="center", fontsize=13)
        axis.text(0.245, 0.68, "W_in", ha="center", va="bottom", color=FIXED_COLOR)

        # --- リザバー -> 出力 (学習) ---
        _arrow(axis, (0.71, 0.65), (0.90, 0.65), TRAINED_COLOR)
        axis.text(0.945, 0.65, "y", ha="center", va="center", fontsize=13)
        axis.text(
            0.805,
            0.68,
            "W_out",
            ha="center",
            va="bottom",
            color=TRAINED_COLOR,
            fontweight="bold",
        )
        axis.text(
            0.805,
            0.60,
            style.label("ここだけ学習する", "the only trained part"),
            ha="center",
            va="top",
            color=TRAINED_COLOR,
            fontsize=10,
        )

        axis.text(
            0.53,
            0.28,
            style.label(
                "状態 x は u と W だけで決まる —— 学習は x から y への線形写像だけ",
                "the state x depends only on u and W;"
                " training fits only the linear map from x to y",
            ),
            ha="center",
            va="center",
            fontsize=10,
        )
        figure.suptitle(
            style.label(
                "リザバー計算: 固定した力学系に、学習した線形読み出しを載せる",
                "Reservoir computing: a trained linear readout"
                " on top of a fixed dynamical system",
            )
        )
        return save_png(figure, path)


def _arrow(
    axis: object, start: tuple[float, float], end: tuple[float, float], color: str
) -> None:
    """矢印を1本引く。"""
    axis.add_patch(  # type: ignore[attr-defined]
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.8,
            color=color,
        )
    )


__all__ = ["FIXED_COLOR", "TRAINED_COLOR", "plot_architecture"]

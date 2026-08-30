"""図のレイアウト補助 —— 注記の折り返し・凡例の外出し・パネル記号.

``style.py`` から分けてあるのは行数上限 (D-77) のためである。**上限のほうを
緩めない**。``style`` は「色・線種・footnote など図の見た目の単一の真実」を持つ
場所で、こちらは「軸をどう並べ、どこに何を置くか」なので、まとまりとして切れる。

ここが解いている問題は3つ (FIG-16 / FIG-17):

1. **多パネル図に記号が無い**。本文が「図3の右上のパネル」と位置で指すしか
   なかった (``label_panels``)
2. **凡例がデータや軸を潰す**。系列が多い図や引用つきの長いラベルがある図では、
   軸の中に置くと読めなくなる (``legend_below``)
3. **長い注記が図そのものを横に伸ばす**。保存は ``bbox_inches="tight"`` なので、
   軸より横に長い ``supxlabel`` があると tight bbox がその幅まで広がる
   (``wrapped_note``)
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter

from rc_basics_lab.plotting.style import StyleContext

NOTE_WRAP_WIDTH = 78
"""図の下に置く注記を折り返す幅 (全角を2文字と数えた概算)。

**折り返さないと図そのものが横に伸びる。** 保存は ``bbox_inches="tight"`` なので、
軸より横に長い ``supxlabel`` があると tight bbox がその幅まで広がり、軸が中央に
取り残される (実測: ``fig_valid_time`` は軸 1000px に対し canvas 2883px)。
"""


_NO_LINE_START = frozenset("\u3002\u3001\uff09\u300d\u300f\u3011\u3015\u30fb\u2026)")
"""行頭に置かない文字 (``wrapped_note``)。

ruff の RUF001 が全角の閉じ括弧を ASCII と紛らわしいと判定するので、
コードポイントで書く (句点・読点・各種閉じ括弧・中黒・三点リーダ)。
"""


LEGEND_BELOW_FONTSIZE = 8
"""図の外に出す凡例の文字サイズ。"""


LEGEND_BELOW_ANCHOR: tuple[float, float] = (0.5, -0.08)
"""図の外に出す凡例の位置 (figure 座標。負 = 枠の下)。"""


PANEL_LABEL_FONTSIZE = 11
"""パネル記号 (a) (b) (c) の文字サイズ (本文より 1pt 大きい)。"""


PANEL_LABEL_POSITION: tuple[float, float] = (-0.12, 1.05)
"""パネル記号を置く位置 (axes 座標。軸の左上の外側)。"""


def wrapped_note(text: str, width: int = NOTE_WRAP_WIDTH) -> str:
    """長い注記を ``width`` 桁で折り返す (全角は2桁と数える)。

    ``textwrap`` は CJK の文字幅を見ないので、全角を2桁として自前で折る。
    句読点と閉じ括弧は行頭に置かない (``_NO_LINE_START``) —— 折り返しただけの
    行頭に「。」が来ると、図の注記としては目に付きすぎる。

    Args:
        text: 折り返す注記。改行は入っていない前提。
        width: 1行の桁数 (半角換算)。

    Returns:
        改行を挟んだ文字列。
    """
    lines: list[str] = []
    current = ""
    used = 0
    for char in text:
        cost = 2 if unicodedata.east_asian_width(char) in "WFA" else 1
        # 行頭禁則の文字は、幅を1文字ぶん超えても前の行に残す
        if used + cost > width and current and char not in _NO_LINE_START:
            lines.append(current)
            current, used = "", 0
        current += char
        used += cost
    if current:
        lines.append(current)
    return "\n".join(lines)


def legend_below(
    figure: Figure,
    axes: Sequence[Axes],
    *,
    style: StyleContext,
    ncol: int = 3,
) -> None:
    """複数の軸の凡例を**図の外・下**に1つへ統合する (FIG-17)。

    軸の中に置くと、系列が多い図や引用つきの長いラベルがある図では凡例が
    データを隠すか、軸そのものを押し潰す (実測: ``fig_threshold_tradeoff`` は
    右パネルの高さが左の半分になっていた)。同じ凡例が複数パネルに繰り返し出る
    問題も同時に消える。

    ラベルが同じ項目は**最初の1つだけ**を残す。パネルごとに同じ系列を描いた図
    (6 パネルに同じ閾値線がある等) で、凡例が系列数 x パネル数に膨らむのを防ぐ。

    Args:
        figure: 凡例を置く図。
        axes: 凡例の項目を集める軸。
        style: 描画コンテキスト (署名をそろえるために受け取る)。
        ncol: 凡例の列数。
    """
    del style
    handles: list[Artist] = []
    labels: list[str] = []
    for axis in axes:
        for handle, text in zip(*axis.get_legend_handles_labels(), strict=True):
            if text not in labels:
                handles.append(handle)
                labels.append(text)
    if not handles:
        return
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        fontsize=LEGEND_BELOW_FONTSIZE,
        frameon=False,
        bbox_to_anchor=LEGEND_BELOW_ANCHOR,
    )


def panel_label(axis: Axes, letter: str, *, style: StyleContext) -> None:
    """パネルの左上に記号 ``(a)`` を振る (FIG-16)。

    多パネル図で本文が「図3の右上」と位置で指すしかない状態を解消する。
    論文の図はほぼ例外なく記号で指すので、位置の言い回しに依存しない。

    ``style`` は受け取るが**ラベル言語では分岐しない** —— 記号は ja/en で
    同じであり、本文とキャプションが同じ記号で指せることのほうが重要である。

    Args:
        axis: 記号を振る軸。
        letter: ``"a"`` のような1文字 (``(a)`` の形で描く)。
        style: 描画コンテキスト (署名をそろえるために受け取る)。
    """
    del style
    axis.text(
        *PANEL_LABEL_POSITION,
        f"({letter})",
        transform=axis.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def label_panels(axes: Sequence[Axes], *, style: StyleContext) -> None:
    """並んだ軸に ``(a) (b) (c) ...`` を順に振る (FIG-16)。

    **1枚しか軸が無い図には振らない** —— 記号は「どのパネルか」を指すための
    ものなので、1枚のときは指す対象が無く、記号だけが浮く。
    """
    if len(axes) < 2:
        return
    for index, axis in enumerate(axes):
        panel_label(axis, chr(ord("a") + index), style=style)


def hide_minor_tick_labels(axis: Axes, *, which: str = "x") -> None:
    """対数軸の**副目盛りのラベル**を消す (FIG-19)。

    測っていない点のラベルが軸に並ぶのを防ぐ。実測は 25/50/100/200 の4点だけ
    なのに ``25 / 3x10^1 / 4x10^1 / 50 / 6x10^1 / 100 / 200`` と表示されていた
    (``fig_size_vs_performance``)。主目盛りを ``set_xticks`` で明示している図
    では、副目盛りのラベルは**必ず余計**である。

    目盛り線そのものは残す (対数のスケール感が読めるため)。消すのはラベルだけ。

    Args:
        axis: 対象の軸。
        which: ``"x"`` / ``"y"`` / ``"both"``。
    """
    targets = (
        (axis.xaxis, axis.yaxis)
        if which == "both"
        else ((axis.xaxis,) if which == "x" else (axis.yaxis,))
    )
    for target in targets:
        target.set_minor_formatter(NullFormatter())


__all__ = [
    "LEGEND_BELOW_ANCHOR",
    "LEGEND_BELOW_FONTSIZE",
    "NOTE_WRAP_WIDTH",
    "PANEL_LABEL_FONTSIZE",
    "PANEL_LABEL_POSITION",
    "hide_minor_tick_labels",
    "label_panels",
    "legend_below",
    "panel_label",
    "wrapped_note",
]

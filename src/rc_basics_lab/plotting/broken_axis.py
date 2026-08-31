"""縦軸を破断して上限線と実測を1枚に載せる (2-7 / FIG-4).

上限が実測から桁で離れている図 (3-A の ``MC <= N``) では、上限線を同じ軸に
描くと**実測の帯より広い空白**が生まれる。対数軸にしても縮まらない ——
空白の高さは比 (200 / 36 = 5.5 倍) で決まり、目盛の取り方では変わらない。

論文の作法どおり軸を破断する。上段は上限線だけ、下段は実測だけを持ち、
境目に破断記号を置く。「上限からどれだけ遠いか」は下段に添える正規化軸
(``MC / N``) で読む —— **破断した軸の見た目の距離は量ではない**ので、
数値を読める軸を必ず残す。

破断が要らない (上限と実測が近い) ときは分割しない。空白が無いのに軸を
割ると、読者は無いはずの不連続を探すことになる。
"""

from __future__ import annotations

from matplotlib.axes import Axes

BREAK_RATIO = 3.0
"""破断する閾値 (上限 / 実測の最大値)。

これ未満なら1本の軸に収まる。3.0 は「上段の空白が実測の帯と同じ高さになる」
あたりで、ここを下回る図で破断すると読みにくくなるだけである。
"""

_MARK_MARKER = (2, 0, 45)
"""破断記号。matplotlib の (頂点数, 種類, 回転角) 表記で、45 度に傾けた線分。

軸の外へ描くので ``clip_on=False`` にする (境目のスパインを消した位置に
記号だけを残す)。
"""

_MARK_SIZE = 8.0
"""破断記号の大きさ [pt]。"""


def draw_break_marks(upper: Axes, lower: Axes) -> None:
    """上下のパネルの境目に破断記号を描く。

    上段の下スパインと下段の上スパインを消し、その位置に斜線を並べる。
    上段の x 目盛りも消す (同じ横軸を2回書かない、1-7)。

    Args:
        upper: 上段 (上限線側) の軸。
        lower: 下段 (実測側) の軸。
    """
    upper.spines["bottom"].set_visible(False)
    lower.spines["top"].set_visible(False)
    upper.tick_params(bottom=False, labelbottom=False)
    for axis, height in ((upper, 0.0), (lower, 1.0)):
        axis.plot(
            [0.0, 1.0],
            [height, height],
            transform=axis.transAxes,
            marker=_MARK_MARKER,
            markersize=_MARK_SIZE,
            linestyle="none",
            color="0.4",
            markeredgecolor="0.4",
            markeredgewidth=1.0,
            clip_on=False,
        )


def needs_break(bound: float, data_max: float) -> bool:
    """上限と実測が ``BREAK_RATIO`` 倍以上離れているか。

    Args:
        bound: 上限線の値。
        data_max: 実測 (誤差棒の上端を含む) の最大値。

    Returns:
        破断したほうが読めるなら ``True``。
    """
    if data_max <= 0.0:
        return False
    return bound / data_max >= BREAK_RATIO


__all__ = ["BREAK_RATIO", "draw_break_marks", "needs_break"]

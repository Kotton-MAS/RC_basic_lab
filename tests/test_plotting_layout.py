"""作図の版面 (凡例・footnote・パネル記号) の検査.

**空虚になる形は「図が書ければ通る」**である。重なりは PNG を開くまで
分からないので、位置を数値で比べる形にしておく。
"""

from __future__ import annotations

from rc_basics_lab.plotting import layout
from rc_basics_lab.plotting.style import setup_style

CONTEXT = setup_style(commit="0" * 40)
"""検査用の描画コンテキスト (commit は固定)。"""


def test_the_gid_prefix_is_the_same_on_both_sides() -> None:
    """凡例の下端を伝える ``gid`` の接頭辞が2箇所で一致する (C-1)。

    ``layout`` が ``style`` を import しているので逆向きは引けず、定数を
    書き写している。**写した以上、一致は機械で見る。**
    """
    from rc_basics_lab.plotting import style as style_module

    assert layout.LEGEND_GID_PREFIX == style_module.LEGEND_GID_PREFIX


def test_the_footnote_sits_below_a_multi_row_legend() -> None:
    """**図の外の凡例と footnote が重ならない** (C-1、3サイクル指摘された)。

    凡例は上端を固定して下へ伸ばし、footnote はその下端よりさらに下に置く。
    行数が増えても枠の中に入らないことを、行数を変えて測る。
    """
    from matplotlib.figure import Figure

    from rc_basics_lab.plotting.style import FOOTNOTE_OFFSET, _footnote_y

    def bottom_of(n_labels: int, ncol: int) -> tuple[float, float]:
        figure = Figure(figsize=(8.0, 5.0))
        axis = figure.subplots(1, 1)
        for index in range(n_labels):
            axis.plot([0.0, 1.0], [0.0, 1.0], label=f"series {index}")
        layout.legend_below(figure, [axis], style=CONTEXT, ncol=ncol)
        (legend,) = figure.legends
        gid = legend.get_gid()
        assert gid is not None and gid.startswith(layout.LEGEND_GID_PREFIX)
        return float(gid[len(layout.LEGEND_GID_PREFIX) :]), _footnote_y(figure)

    one_row_bottom, one_row_footnote = bottom_of(3, 3)
    three_row_bottom, three_row_footnote = bottom_of(9, 3)

    # 凡例は枠の外にとどまる (上端が負なので、下へ伸ばす限り中へ入らない)
    assert layout.LEGEND_BELOW_ANCHOR[1] < 0.0
    # 行数が増えたら下端も footnote も下がる
    assert three_row_bottom < one_row_bottom
    assert three_row_footnote < one_row_footnote
    # footnote は必ず凡例より下
    assert one_row_footnote < one_row_bottom
    assert three_row_footnote < three_row_bottom
    # 凡例が無ければ既定のまま
    assert _footnote_y(Figure()) == FOOTNOTE_OFFSET


def test_the_outside_artists_clear_the_content_after_settling() -> None:
    """**描画後の実寸**で凡例と footnote が中身に食い込まない (D-152)。

    ``legend_below`` の上端 (-0.02) と ``add_footnote`` の行数からの見積りは
    どちらも**描画前の推定**なので、``supxlabel`` の注記が折り返された図や
    軸ラベルの長い図では足りなかった。``settle_outside_artists`` が一度
    描画してから実寸で積み直す。

    ここが空虚になる形は「推定のままの座標を見て通る」なので、
    **``get_window_extent`` で実際に描かれた矩形を測る**。
    """
    from matplotlib.figure import Figure

    from rc_basics_lab.plotting.style import (
        FOOTNOTE_GID,
        add_footnote,
        settle_outside_artists,
    )

    figure = Figure(figsize=(8.0, 5.0))
    axis = figure.subplots(1, 1)
    figure.set_layout_engine("constrained")
    for index in range(9):
        axis.plot([0.0, 1.0], [0.0, 1.0], label=f"series {index}")
    axis.set_xlabel("横軸のラベル")
    # **2行に折り返る長い注記**を置く (これが無いと推定でも足りてしまう)
    figure.supxlabel(layout.wrapped_note("注: " + "折り返す注記。" * 12))
    layout.legend_below(figure, [axis], style=CONTEXT, ncol=3)
    add_footnote(figure, "N = 200", style=CONTEXT)
    settle_outside_artists(figure)

    figure.canvas.draw()
    to_figure = figure.transFigure.inverted()

    def bottom_top(artist: object) -> tuple[float, float]:
        extent = artist.get_window_extent(None)  # type: ignore[attr-defined]
        low = float(to_figure.transform((0.0, extent.y0))[1])
        high = float(to_figure.transform((0.0, extent.y1))[1])
        return low, high

    (legend,) = figure.legends
    (note,) = [text for text in figure.texts if text.get_gid() == FOOTNOTE_GID]
    legend_bottom, legend_top = bottom_top(legend)
    note_bottom, note_top = bottom_top(note)

    # 中身 (凡例と footnote を除いた tight bbox) の下端
    for artist in (legend, note):
        artist.set_visible(False)
    figure.canvas.draw()
    content_bottom = float(to_figure.transform((0.0, figure.get_tightbbox(None).y0))[1])
    for artist in (legend, note):
        artist.set_visible(True)

    assert legend_top <= content_bottom, (
        f"凡例が中身に食い込んでいます (凡例の上端 {legend_top:.4f} > "
        f"中身の下端 {content_bottom:.4f})"
    )
    assert note_top <= legend_bottom, (
        f"footnote が凡例に食い込んでいます (footnote の上端 {note_top:.4f} > "
        f"凡例の下端 {legend_bottom:.4f})"
    )
    assert note_bottom < note_top <= legend_bottom < legend_top <= content_bottom


def test_settling_is_a_no_op_without_outside_artists() -> None:
    """図の外に何も無ければ ``settle_outside_artists`` は何もしない。

    全図が ``save_png`` を通るので、凡例も footnote も無い図で描画を1回
    余分に走らせないことを確かめる (速度ではなく、**何も動かさない**ことの確認)。
    """
    from matplotlib.figure import Figure

    from rc_basics_lab.plotting.style import settle_outside_artists

    figure = Figure(figsize=(4.0, 3.0))
    axis = figure.subplots(1, 1)
    axis.plot([0.0, 1.0], [0.0, 1.0])
    settle_outside_artists(figure)
    assert not figure.legends
    assert not figure.texts

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

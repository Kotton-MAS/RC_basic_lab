"""``plotting/broken_axis.py`` の検査 (2-7 / FIG-4).

破断の判定と記号の描画を**軸の状態から**確かめる。PNG の画素からは
「スパインが消えているか」も「記号がいくつ在るか」も読めない。
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from rc_basics_lab.plotting.broken_axis import (
    BREAK_RATIO,
    draw_break_marks,
    needs_break,
)


@pytest.mark.parametrize(
    ("bound", "data_max", "expected"),
    [
        (200.0, 36.0, True),  # 本番の 3-A (5.5 倍)
        (200.0, 200.0 / BREAK_RATIO, True),  # ちょうど閾値 (境界は破断する側)
        (200.0, 200.0 / BREAK_RATIO + 1.0, False),  # 閾値のすぐ内側
        (20.0, 20.0, False),  # 上限と実測が同じ
        (200.0, 0.0, False),  # 実測が 0 (縮退。割り算に落とさない)
    ],
)
def test_the_break_threshold_is_the_ratio_of_bound_to_data(
    bound: float, data_max: float, expected: bool
) -> None:
    """破断するかは**上限 / 実測**だけで決まる (2-7)。

    絶対値ではなく比で決めるのは、空白の高さが比で決まるためである
    (対数軸でも正規化でも縮まらない)。
    """
    assert needs_break(bound, data_max) is expected


def test_the_break_marks_open_the_facing_spines() -> None:
    """破断記号を描くと**向かい合うスパインが消える** (2-7)。

    スパインが残ったままだと、記号だけが乗った閉じた枠になり「軸が切れて
    いる」ことが読み取れない。上段の x 目盛りラベルも消える (1-7)。
    """
    figure = Figure()
    upper, lower = figure.subplots(2, 1)
    draw_break_marks(upper, lower)
    assert not upper.spines["bottom"].get_visible()
    assert not lower.spines["top"].get_visible()
    assert not any(label.get_visible() for label in upper.get_xticklabels())
    # 記号は線を描かない (``linestyle="none"``)。ここが実線になると
    # 参照線を数える検査 (test_figure_policy) が破断記号を上限線と誤認する。
    for axis in (upper, lower):
        marks = [line for line in axis.get_lines() if line.get_linestyle() == "None"]
        assert len(marks) == 1, f"破断記号が {len(marks)} 本です"


def test_the_break_marks_sit_on_the_boundary_in_axes_coordinates() -> None:
    """記号は上段の下端 (y=0) と下段の上端 (y=1) に置かれる。

    データ座標で置くと、軸の範囲を変えた瞬間に記号が枠の外へ飛ぶ。
    """
    figure = Figure()
    upper, lower = figure.subplots(2, 1)
    draw_break_marks(upper, lower)
    for axis, expected in ((upper, 0.0), (lower, 1.0)):
        mark = next(line for line in axis.get_lines() if line.get_linestyle() == "None")
        assert list(mark.get_ydata()) == [expected, expected]
        assert mark.get_transform() is axis.transAxes
        assert not mark.get_clip_on(), "記号は軸の外へ出すのでクリップしない"

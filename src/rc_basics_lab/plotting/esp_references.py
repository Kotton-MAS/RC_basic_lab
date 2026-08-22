"""実験 02 の図が引く文献の参照値 (D-104).

``figures_esp.py`` は D-77 の凍結対象なので、定数はここへ出す。
**描画を含まない** —— 値と、その値が成り立つ条件の文字列だけである。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from rc_basics_lab.plotting.heatmap import cell_edges
from rc_basics_lab.plotting.labels import YILDIZ_2012, cited_measurement
from rc_basics_lab.plotting.style import (
    REFERENCE_COLOR,
    REFERENCE_DASHES,
    StyleContext,
)
from rc_basics_lab.types import FloatArray

GALLICCHIO = (
    "Gallicchio (2019) Chasing the Echo State Property の再実演",
    "Re-enacting Gallicchio (2019), Chasing the Echo State Property",
)
"""2-C の副題 (仕様 §4 T3: 先行研究の再実演であることを図に明記する)。"""

ZERO_INPUT_ESP_BOUNDARY = 1.0
"""無入力で ESP が失われる rho の境界 (D-104)。

一次資料は Yildiz, Jaeger, Kiebel (2012) *Re-visiting the echo state property*,
Neural Networks **35**:1-9。本文にこうある:

    if the matrix W~ = dt*W + (1 - a*dt)*I ... has a spectral radius
    rho(W~) > 1 then the echo state property (for zero input) is not
    satisfied since the origin becomes unstable.

つまり**無入力に限れば** rho < 1 は必要条件であり、境界は 1 である。
同論文は同時に「rho < 1 は十分条件ではない」を解析的な反例で示している ——
この図が問うているのはその**逆側**、「入力があれば rho > 1 でも ESP は
成立しうる」のほうである。

実測: 無入力パネルの境界は 1.0 で一致する。駆動下では sigma_u を上げると
境界が単調に上がり、sigma_u = 2.0 では rho = 1.9 でも成立する。
"""

ZERO_INPUT_ESP_CONDITIONS: tuple[str, str] = (
    "無入力に限る・原点の不安定化による必要条件",
    "zero input only; necessary condition from instability of the origin",
)
"""参照線の**動作点** (D-97 / D-104)。

「無入力に限る」を条件に書くのが要点である。駆動下のパネルへこの線を
そのまま延ばすと、この図が否定しようとしている通説そのものになる。
"""


def zero_input_boundary_note(style: StyleContext) -> str:
    """無入力の境界線の出典を**注として**返す (FIG-14 / D-104)。

    凡例にしない。無入力パネルは幅が狭く、出典つきのラベル (FIG-3 で長くなる)
    を凡例に置くと**軸の 4.19 倍の幅**になり、はみ出して軸ラベルまで潰す
    (実測)。FIG-14 の規約「凡例が収まらないなら注へ移す」に従う。
    """
    return cited_measurement(
        style.label(
            f"左の破線は無入力の境界 rho = {ZERO_INPUT_ESP_BOUNDARY:g}",
            f"the dashed line on the left is the zero-input boundary"
            f" rho = {ZERO_INPUT_ESP_BOUNDARY:g}",
        ),
        YILDIZ_2012,
        style.label(*ZERO_INPUT_ESP_CONDITIONS),
    )


def draw_zero_input_boundary(axis: Axes, style: StyleContext) -> None:
    """無入力パネルに文献の境界線を引く (D-104)。

    **無入力パネルにだけ**引く。駆動下のパネルへ延ばすと、この図が
    否定しようとしている通説 (rho < 1 が ESP の条件) そのものになる。

    凡例は付けない —— 出典は ``zero_input_boundary_note`` が図の注へ出す。
    """
    del style
    axis.axhline(
        ZERO_INPUT_ESP_BOUNDARY,
        color=REFERENCE_COLOR,
        dashes=REFERENCE_DASHES[0],
        linewidth=1.2,
    )


def plot_no_input_panel(
    axis: Axes,
    rates: FloatArray,
    rhos: Sequence[float],
    norm: Normalize,
    style: StyleContext,
) -> None:
    """無入力 (sigma_u = 0) の列。駆動下の領域と同じ配色で別枠に出す。"""
    axis.pcolormesh(
        np.array([0.0, 1.0]),
        cell_edges(rhos),
        rates.reshape(-1, 1),
        cmap="RdYlBu",
        norm=norm,
        shading="flat",
    )
    axis.set_xticks([0.5])
    axis.set_xticklabels([style.label("無入力", "no input")])
    axis.set_ylabel(style.label("スペクトル半径 rho", "spectral radius rho"))
    draw_zero_input_boundary(axis, style)
    axis.set_title(
        style.label("sigma_u = 0", "sigma_u = 0"),
        fontsize=10,
    )


__all__ = [
    "GALLICCHIO",
    "ZERO_INPUT_ESP_BOUNDARY",
    "ZERO_INPUT_ESP_CONDITIONS",
    "draw_zero_input_boundary",
    "plot_no_input_panel",
    "zero_input_boundary_note",
]

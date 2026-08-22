"""予測波形の図 —— 「NRMSE 0.0007」がどう見えるか (FIG-11 / D-107).

## なぜ要るのか

`fig_comparison` は「NRMSE 0.0007」と言うだけで、**それがどう見えるかを
見せていない**。読者は数値の良し悪しを判断できない。

## どこを切り出すか (D-107)

**テスト区間の先頭から固定オフセット**で切り出し、レプリケートは 0 に固定する。
「よく当たっている区間」を選べるようにすると、同じデータから好きな結論の図が
作れてしまう —— 仕様 §5 の禁止する構造そのものである。

そのため区間は**モジュール定数**であり、呼び出し側から渡せない。
引数にすると「この図だけ別の区間」が書けてしまい、決定が実質無くなる
(D-90 で alpha 格子を手法ごとに変えられなくしたのと同じ形)。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from rc_basics_lab.plotting.style import (
    FIGURE_SIZES,
    METHOD_COLORS,
    StyleContext,
    add_provenance,
    method_color,
    new_figure,
    rc_context_for,
    save_png,
)
from rc_basics_lab.types import FloatArray

WAVEFORM_OFFSET = 0
"""テスト区間の先頭から数えた切り出しの開始位置 [ステップ] (D-107)。

**0 に固定する。** 「どこから切るか」を動かせると、同じデータから
好きな結論の図が作れる。先頭からなら「選んでいない」ことが読者にも分かる。
"""

WAVEFORM_STEPS = 300
"""切り出す長さの既定値 [ステップ] (D-107)。

Mackey-Glass (Delta t = 1.0) で振動が数周見える長さ。
**長さも定数**にするのは、短くすれば当たって見え、長くすれば外れて見える
ためである。
"""

WAVEFORM_STEPS_BY_TASK: dict[str, int] = {
    # 1 ステップごとに +-1 が反転するので、300 だと線が塗り潰れる (実測)。
    # 80 なら個々の反転が分離し、「線形と遅延線が 0 に張り付く」が読める。
    "delay_parity": 80,
}
"""課題ごとの切り出し長 (D-107)。載っていない課題は ``WAVEFORM_STEPS``。

**課題ごとに変えても「選べない」性質は保たれる** —— これは定数表であり、
実行のたびに同じ窓が出る。呼び出し側が長さを指定する経路は作らない。
"""


def waveform_steps_for(task: str) -> int:
    """課題の切り出し長を返す (D-107)。

    Args:
        task: 課題名。

    Returns:
        ``WAVEFORM_STEPS_BY_TASK`` の値、無ければ ``WAVEFORM_STEPS``。
    """
    return WAVEFORM_STEPS_BY_TASK.get(task, WAVEFORM_STEPS)


WAVEFORM_REPLICATE = 0
"""波形に使うレプリケート (D-107)。**0 に固定する。**

レプリケートを選べると、うまくいった1本を出せてしまう。
ばらつきは `fig_comparison` の誤差棒が受け持つ。
"""


@dataclass(frozen=True, slots=True)
class _OneReplicate:
    """footnote に渡す「レプリケート {WAVEFORM_REPLICATE} を1本」の器。"""

    replicate: int = WAVEFORM_REPLICATE


TRUTH_COLOR = "#333333"
"""真値の色。手法4色 (D-85) とも参照線の黒 (D-86) とも別の無彩色にする。"""


def draw_prediction_waveform(
    axis: Axes,
    truth: FloatArray,
    predictions: dict[str, FloatArray],
    style: StyleContext,
) -> int:
    """真値と各手法の予測を **1 つの軸に**重ねる (FIG-11 / D-107)。

    単独の figure にする経路 (``plot_prediction_waveform``) と、他の図の
    パネルにする経路 (01 の ``fig_comparison``, FIG-12) の**両方がこれを呼ぶ**。
    片方に書き写すと、線の太さや真値の色が2箇所で独立にずれていく。

    Args:
        axis: 描画先。
        truth: テスト区間の真値 (切り出し済み)。
        predictions: 手法名 -> 予測 (同じ長さ)。色は ``METHOD_COLORS`` から引く。
        style: 配色・言語。

    Returns:
        描いた区間の長さ [ステップ]。脚注の条件に使う。

    Raises:
        ValueError: ``predictions`` が空、または長さが揃っていない場合。
    """
    if not predictions:
        raise ValueError("predictions が空です")
    length = int(np.asarray(truth).reshape(-1).size)
    for name, values in predictions.items():
        if int(np.asarray(values).reshape(-1).size) != length:
            raise ValueError(f"{name} の長さが真値と違います")
    steps = np.arange(length, dtype=np.float64)
    axis.plot(
        steps,
        np.asarray(truth).reshape(-1),
        color=TRUTH_COLOR,
        linewidth=2.0,
        label=style.label("真値", "ground truth"),
        zorder=2,
    )
    for name, values in predictions.items():
        axis.plot(
            steps,
            np.asarray(values).reshape(-1),
            color=method_color(name) if name in METHOD_COLORS else None,
            linewidth=1.2,
            alpha=0.9,
            label=name,
            zorder=3,
        )
    axis.set_xlabel(
        style.label(
            f"テスト区間の先頭からのステップ (オフセット {WAVEFORM_OFFSET})",
            f"steps from the start of the test split (offset {WAVEFORM_OFFSET})",
        )
    )
    axis.set_ylabel(style.label("値 (標準化前)", "value (before standardisation)"))
    axis.legend(loc="upper right", fontsize=8, ncols=2)
    return length


def plot_prediction_waveform(
    truth: FloatArray,
    predictions: dict[str, FloatArray],
    path: Path,
    *,
    task_label: tuple[str, str],
    style: StyleContext,
) -> Path:
    """真値と各手法の予測を時間軸で重ねる (FIG-11 追加図2/3 / D-107)。

    Args:
        truth: テスト区間の真値 (切り出し済み)。
        predictions: 手法名 -> 予測 (同じ長さ)。色は ``METHOD_COLORS`` から引く。
        path: 出力先の PNG。
        task_label: 課題名の (日本語, 英語)。
        style: 配色・言語・commit。

    Returns:
        書き出した PNG のパス。

    Raises:
        ValueError: ``predictions`` が空、または長さが揃っていない場合。
    """
    width, height = FIGURE_SIZES["wide"]
    with rc_context_for(style):
        figure = new_figure(width, height)
        axis = figure.subplots(1, 1)
        length = draw_prediction_waveform(axis, truth, predictions, style)
        figure.suptitle(
            style.label(
                f"{task_label[0]}: 予測がどう見えるか",
                f"{task_label[1]}: what the predictions look like",
            )
        )
        figure.supxlabel(
            style.label(
                "注: 区間もレプリケートも固定である (D-107)。"
                "「よく当たっている区間」を選べる図にしない。",
                "Note: the window and the replicate are fixed (D-107)."
                " The figure must not let anyone pick a favourable window.",
            ),
            fontsize=8,
        )
        conditions = f"steps = {WAVEFORM_OFFSET}..{WAVEFORM_OFFSET + length}"
        # この図は**レプリケート1本**しか使わない (D-107)。空の行を渡すと
        # replicates_field が「replicates が空です」で落ちるので、
        # 使った1本をそのまま渡す。
        add_provenance(figure, conditions, (_OneReplicate(),), style=style)
        return save_png(figure, path)


def slice_window(values: FloatArray, start: int, task: str = "") -> FloatArray:
    """テスト区間から**固定の窓**を切り出す (D-107)。

    Args:
        values: 系列全体。
        start: テスト区間の先頭 index。
        task: 課題名。長さの決定にだけ使う (``WAVEFORM_STEPS_BY_TASK``)。
            **窓を選ぶ引数ではない** —— 同じ課題なら常に同じ窓になる。

    Returns:
        ``WAVEFORM_STEPS`` 行の切り出し (足りなければあるだけ)。
    """
    begin = start + WAVEFORM_OFFSET
    flat: FloatArray = np.asarray(values, dtype=np.float64).reshape(-1)
    return flat[begin : begin + waveform_steps_for(task)]


def selection_is_fixed() -> Sequence[tuple[str, int]]:
    """波形の選び方が定数であることを外から確かめるための一覧 (D-107)。"""
    return (
        ("offset", WAVEFORM_OFFSET),
        ("steps", WAVEFORM_STEPS),
        ("replicate", WAVEFORM_REPLICATE),
        # 課題ごとの長さも定数である。ここに載せておかないと、
        # 「既定は固定だが課題別は自由」という抜け道がガードの外に出る。
        *sorted(WAVEFORM_STEPS_BY_TASK.items()),
    )


__all__ = [
    "TRUTH_COLOR",
    "WAVEFORM_OFFSET",
    "WAVEFORM_REPLICATE",
    "WAVEFORM_STEPS",
    "WAVEFORM_STEPS_BY_TASK",
    "draw_prediction_waveform",
    "plot_prediction_waveform",
    "selection_is_fixed",
    "slice_window",
    "waveform_steps_for",
]

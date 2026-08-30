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
from matplotlib.figure import Figure

from rc_basics_lab.plotting.freerun_grids import label_of
from rc_basics_lab.plotting.labels import METHOD_LABELS
from rc_basics_lab.plotting.layout import wrapped_note
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
    # 1 ステップごとに +-1 が反転する。80 では 20 周期以上が入って真値と
    # ESN が重なり判別しづらかったので 40 に縮めた (C-4)。
    "delay_parity": 40,
    # NARMA10 は高周波で、4 本重ねると 300 ステップではスパゲッティになる。
    "narma10": 100,
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
class FixedReplicate:
    """footnote に渡す「レプリケート {WAVEFORM_REPLICATE} を1本」の器。

    **波形系の図で共有する** (D-92)。図ごとに同じ器を書き直すと、
    固定するレプリケートが2箇所で独立にずれる。
    """

    replicate: int = WAVEFORM_REPLICATE


FLAT_RATIO = 0.9
"""「平均に張り付いている」と判定する残差比 (FIG-1 / C-1)。

残差の RMS が真値の標準偏差の 9 割以上なら、平均を答えているのと
実質変わらない。**閾値を定数にする**のは、図ごとに変えると同じ結果に
違う結論文が付くためである。
"""

TRUTH_COLOR = "#333333"
"""真値の色。手法4色 (D-85) とも参照線の黒 (D-86) とも別の無彩色にする。"""


@dataclass(frozen=True, slots=True)
class WaveformAxes:
    """``draw_prediction_waveform`` が作った2段の軸 (FIG-11 / C-3)。

    Attributes:
        top: 真値と予測の段。見出しはここに付ける。
        bottom: 残差の段。
        length: 描いた区間の長さ [ステップ]。脚注の条件に使う。
    """

    top: Axes
    bottom: Axes
    length: int


def waveform_headline(
    truth: FloatArray, predictions: dict[str, FloatArray], style: StyleContext
) -> str:
    """波形図の**結論文**を残差から導く (FIG-1 / C-1)。

    「予測がどう見えるか」は疑問形で、図が何を示したかを言っていない。
    **数えるのは「真値の変動に対して残差がどれだけ残ったか」**で、
    1 に近い手法は平均を答えているのと変わらない。

    Args:
        truth: テスト区間の真値。
        predictions: 手法名 -> 予測。
        style: 言語。

    Returns:
        見出しの1文。手法名は対応表を通す (C-2)。

    Raises:
        ValueError: ``predictions`` が空の場合。
    """
    if not predictions:
        raise ValueError("predictions が空です")
    true_values: FloatArray = np.asarray(truth, dtype=np.float64).reshape(-1)
    spread = float(np.std(true_values))
    ratios = {
        name: float(
            np.sqrt(
                np.mean(
                    (true_values - np.asarray(values, dtype=np.float64).reshape(-1))
                    ** 2
                )
            )
        )
        / spread
        if spread > 0.0
        else float("inf")
        for name, values in predictions.items()
    }
    flat = sorted(name for name, ratio in ratios.items() if ratio >= FLAT_RATIO)
    tracking = sorted(name for name in ratios if name not in flat)
    if not flat:
        # **順位を書かない。** この窓は数十〜数百ステップしかなく、
        # 手法間の順位は窓で反転しうる (実測: NARMA10 の 100 ステップ窓では
        # ESN が最小だが、テスト区間全体の NMSE は遅延線のほうが小さい)。
        # 順位を主張するのは、全区間で測っているスカラーのパネルの仕事である。
        return style.label(
            f"どの手法も真値を追う (この区間の残差は"
            f" {min(ratios.values()):.0%}〜{max(ratios.values()):.0%})",
            f"every method tracks the truth (residuals in this window:"
            f" {min(ratios.values()):.0%}-{max(ratios.values()):.0%})",
        )
    flat_labels = " / ".join(label_of(METHOD_LABELS, name, style) for name in flat)
    rest_labels = " / ".join(label_of(METHOD_LABELS, name, style) for name in tracking)
    if not tracking:
        return style.label(
            f"{flat_labels} はいずれも平均に張り付く",
            f"{flat_labels} all stay at the mean",
        )
    return style.label(
        f"{flat_labels} は平均に張り付き、{rest_labels} は真値を追う",
        f"{flat_labels} stay at the mean; {rest_labels} track the truth",
    )


def draw_prediction_waveform(
    figure: Figure,
    axis: Axes,
    truth: FloatArray,
    predictions: dict[str, FloatArray],
    style: StyleContext,
) -> WaveformAxes:
    """真値と予測を上段に、**残差を下段に**重ねる (FIG-11 / C-3 / D-107)。

    単独の figure にする経路 (``plot_prediction_waveform``) と、他の図の
    パネルにする経路 (01 の ``fig_comparison``) の**両方がこれを呼ぶ**。
    片方に書き写すと、線の太さや真値の色が2箇所で独立にずれていく。

    **残差の段が要る理由**: 上段だけだと「全部合っている」しか伝わらない。
    Mackey-Glass では 4 本が完全に重なって 1 本にしか見えず、同じ図の
    スカラーパネルが主張する 200 倍の差がどこに出ているのか読めなかった。
    残差の y 軸は**手法間で共通**にする (手法ごとにスケールを変えると差が消える)。
    課題間では揃えない —— 3 桁違うので、揃えると片方の残差が完全に潰れる。

    Args:
        figure: ``axis`` が属する figure (段を2つに割るのに要る)。
        axis: 描画先。**この軸は2段に置き換えられる**。
        truth: テスト区間の真値 (切り出し済み)。
        predictions: 手法名 -> 予測 (同じ長さ)。色は ``METHOD_COLORS`` から引く。
        style: 配色・言語。

    Returns:
        作った2段の軸と区間の長さ。

    Raises:
        ValueError: ``predictions`` が空、または長さが揃っていない場合。
    """
    if not predictions:
        raise ValueError("predictions が空です")
    true_values: FloatArray = np.asarray(truth, dtype=np.float64).reshape(-1)
    length = int(true_values.size)
    for name, values in predictions.items():
        if int(np.asarray(values).reshape(-1).size) != length:
            raise ValueError(f"{name} の長さが真値と違います")

    spec = axis.get_subplotspec()
    if spec is None:
        raise ValueError(
            "subplot として作られた軸を渡してください (残差の段を割るため)"
        )
    axis.remove()
    grid = spec.subgridspec(2, 1, height_ratios=(2.2, 1.0), hspace=0.10)
    top = figure.add_subplot(grid[0])
    bottom = figure.add_subplot(grid[1], sharex=top)

    steps = np.arange(length, dtype=np.float64)
    top.plot(
        steps,
        true_values,
        color=TRUTH_COLOR,
        linewidth=2.0,
        label=style.label("真値", "ground truth"),
        zorder=2,
    )
    for name, values in predictions.items():
        prediction: FloatArray = np.asarray(values, dtype=np.float64).reshape(-1)
        # **手法名は対応表を通す** (C-2 / FIG-5)。生のキーを凡例に出すと、
        # 同じ図の x 軸ラベルと凡例で同じものが2通りに呼ばれる。
        # style.label を通さないと CJK フォントが無い環境で豆腐になる。
        label = label_of(METHOD_LABELS, name, style)
        color = method_color(name) if name in METHOD_COLORS else None
        top.plot(
            steps,
            prediction,
            color=color,
            linewidth=1.2,
            alpha=0.9,
            label=label,
            zorder=3,
        )
        bottom.plot(
            steps,
            true_values - prediction,
            color=color,
            linewidth=1.0,
            alpha=0.9,
        )
    bottom.axhline(0.0, color=TRUTH_COLOR, linewidth=0.8, zorder=1)
    top.tick_params(labelbottom=False)
    top.set_ylabel(style.label("値 (標準化前)", "value (before standardisation)"))
    bottom.set_ylabel(style.label("残差", "residual"))
    bottom.set_xlabel(
        style.label(
            f"テスト区間の先頭からのステップ (オフセット {WAVEFORM_OFFSET})",
            f"steps from the start of the test split (offset {WAVEFORM_OFFSET})",
        )
    )
    top.legend(loc="upper right", fontsize=8, ncols=2)
    return WaveformAxes(top=top, bottom=bottom, length=length)


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
        drawn = draw_prediction_waveform(figure, axis, truth, predictions, style)
        length = drawn.length
        figure.suptitle(
            f"{style.label(*task_label)}: "
            f"{waveform_headline(truth, predictions, style)}"
        )
        figure.supxlabel(
            wrapped_note(
                style.label(
                    "注: 区間もレプリケートも固定である (D-107)。"
                    "「よく当たっている区間」を選べる図にしない。",
                    "Note: the window and the replicate are fixed (D-107)."
                    " The figure must not let anyone pick a favourable window.",
                )
            ),
            fontsize=8,
        )
        conditions = f"steps = {WAVEFORM_OFFSET}..{WAVEFORM_OFFSET + length}"
        # この図は**レプリケート1本**しか使わない (D-107)。空の行を渡すと
        # replicates_field が「replicates が空です」で落ちるので、
        # 使った1本をそのまま渡す。
        add_provenance(figure, conditions, (FixedReplicate(),), style=style)
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
    "FLAT_RATIO",
    "TRUTH_COLOR",
    "WAVEFORM_OFFSET",
    "WAVEFORM_REPLICATE",
    "WAVEFORM_STEPS",
    "WAVEFORM_STEPS_BY_TASK",
    "FixedReplicate",
    "WaveformAxes",
    "draw_prediction_waveform",
    "plot_prediction_waveform",
    "selection_is_fixed",
    "slice_window",
    "waveform_headline",
    "waveform_steps_for",
]

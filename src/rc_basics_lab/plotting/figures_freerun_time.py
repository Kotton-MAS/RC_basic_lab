"""自走が外れていく時系列 (FIG-11 追加図4 / D-107).

4-B には位相図 (``fig_freerun_attractor``) と有効予測時間の分布
(``fig_valid_time``) があるのに、**時間軸の図が無かった**。
「いつ外れるか」は時間軸でしか見えない。

有効予測時間の点に縦線を引くので、``fig_valid_time`` が数字で言っている
ことと同じものを、この図が形で見せる。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rc_basics_lab.plotting.layout import wrapped_note
from rc_basics_lab.plotting.style import (
    FIGURE_SIZES,
    REFERENCE_COLOR,
    REFERENCE_DASHES,
    StyleContext,
    add_provenance,
    method_color,
    new_figure,
    rc_context_for,
    save_png,
)
from rc_basics_lab.plotting.waveforms import TRUTH_COLOR, FixedReplicate
from rc_basics_lab.types import FloatArray


def plot_freerun_timeline(
    truth: FloatArray,
    predicted: FloatArray,
    path: Path,
    *,
    method: str,
    task_label: tuple[str, str],
    valid_steps: float,
    lyapunov_time: float,
    style: StyleContext,
) -> Path:
    """自走と真値を時間軸で重ね、有効予測時間に縦線を引く (D-107)。

    Args:
        truth: 自走区間の真値 ``(T,)`` か ``(T, 1)``。
        predicted: 同じ長さの自走出力。
        path: 出力先の PNG。
        method: 手法名 (色を ``METHOD_COLORS`` から引く)。
        task_label: 課題名の (日本語, 英語)。
        valid_steps: 有効予測時間 [ステップ]。縦線を引く位置。
        lyapunov_time: Lyapunov 時間 [ステップ]。横軸の副目盛りに使う。
        style: 配色・言語・commit。

    Returns:
        書き出した PNG のパス。

    Raises:
        ValueError: 長さが違う、または空の場合。
    """
    true_values: FloatArray = np.asarray(truth, dtype=np.float64).reshape(-1)
    run_values: FloatArray = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if true_values.size == 0:
        raise ValueError("truth が空です")
    if true_values.size != run_values.size:
        raise ValueError(
            f"長さが違います: truth {true_values.size} / predicted {run_values.size}"
        )

    width, height = FIGURE_SIZES["wide"]
    steps = np.arange(true_values.size, dtype=np.float64)
    with rc_context_for(style):
        figure = new_figure(width, height)
        axis = figure.subplots(1, 1)
        axis.plot(
            steps,
            true_values,
            color=TRUTH_COLOR,
            linewidth=2.0,
            label=style.label("真値", "ground truth"),
            zorder=2,
        )
        axis.plot(
            steps,
            run_values,
            color=method_color(method),
            linewidth=1.3,
            label=style.label(f"{method} の自走", f"{method}, free run"),
            zorder=3,
        )
        if np.isfinite(valid_steps) and 0.0 <= valid_steps <= float(steps[-1]):
            axis.axvline(
                valid_steps,
                color=REFERENCE_COLOR,
                dashes=REFERENCE_DASHES[0],
                linewidth=1.4,
                label=style.label(
                    f"有効予測時間 {valid_steps:.0f} ステップ"
                    f" (= {valid_steps / lyapunov_time:.2f} Lyapunov 時間)",
                    f"valid time {valid_steps:.0f} steps"
                    f" (= {valid_steps / lyapunov_time:.2f} Lyapunov times)",
                ),
            )
        axis.set_xlabel(
            style.label("自走開始からのステップ", "steps since the free run started")
        )
        axis.set_ylabel(style.label("値 (標準化後)", "value (standardised)"))
        axis.legend(loc="upper left", fontsize=8)
        # 見出しは**行から導く** (FIG-1 / C-1)。「どう外れていくか」は疑問形で、
        # 図が何を示したかを言っていない。この図が実際に示しているのは
        # 「位相はずれるが振幅は保たれる」ことである。
        lyapunov_times = valid_steps / lyapunov_time
        amplitude = _amplitude_ratio(true_values, run_values)
        figure.suptitle(
            style.label(
                f"実験 4-B: {task_label[0]} の自走は約 {lyapunov_times:.1f}"
                " Lyapunov 時間で位相がずれるが、振幅は"
                f"真値の {amplitude:.0%} を保つ",
                f"Experiment 4-B: the free run of {task_label[1]} loses phase"
                f" after about {lyapunov_times:.1f} Lyapunov times, but keeps"
                f" {amplitude:.0%} of the amplitude",
            )
        )
        figure.supxlabel(
            wrapped_note(
                style.label(
                    "注: 縦線は有効予測時間 (D-43 の閾値をはじめて超えた点)。"
                    "レプリケートは固定である (D-107)。",
                    "Note: the vertical line marks the valid time (first crossing of"
                    " the D-43 threshold). The replicate is fixed (D-107).",
                )
            ),
            fontsize=8,
        )
        conditions = f"task = {task_label[1]}, method = {method}"
        add_provenance(figure, conditions, (FixedReplicate(),), style=style)
        return save_png(figure, path)


def _amplitude_ratio(truth: FloatArray, predicted: FloatArray) -> float:
    """自走の振幅が真値の何倍か (FIG-1 / C-1)。

    位相がずれても振幅とアトラクタの形が保たれる、というのが 4-B の主張で、
    その「振幅」の側を数える。**標準偏差の比**で測るのは、位相ずれの影響を
    受けずに振れ幅だけを見るためである。

    Args:
        truth: 自走区間の真値。
        predicted: 同じ長さの自走出力。

    Returns:
        ``std(predicted) / std(truth)``。真値が定数なら ``nan``。
    """
    spread = float(np.std(truth))
    if spread <= 0.0:
        return float("nan")
    return float(np.std(predicted)) / spread


__all__ = ["plot_freerun_timeline"]

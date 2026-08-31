"""リザバー状態の波形 (FIG-11 追加図5).

上段が入力、下段が状態 8 本。**同じ入力に対してユニットごとに違う応答が
出る**ことを見せるための図で、01 の PCA 散布図 (``fig_state_space``) が
「少ない主成分で説明できる」と言っている、その元の信号にあたる。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rc_basics_lab.experiment.state_waveform import StateWaveform
from rc_basics_lab.plotting.layout import label_panels, wrapped_note
from rc_basics_lab.plotting.style import (
    FIGURE_SIZES,
    StyleContext,
    add_provenance,
    new_figure,
    rc_context_for,
    save_png,
)
from rc_basics_lab.plotting.waveforms import (
    TRUTH_COLOR,
    WAVEFORM_OFFSET,
    FixedReplicate,
)

STATE_COLORMAP = "viridis"
"""状態 8 本の色。**手法4色 (D-85) とは別系統**にする。

ユニットに意味の順序は無いが、連続的な色にすると『8 本ある』が
一目で読める (同色にすると重なって本数が数えられない)。
"""


def plot_state_waveform(
    waveform: StateWaveform, path: Path, *, style: StyleContext
) -> Path:
    """入力と状態 8 本を時間軸で重ねる (FIG-11 追加図5 / D-107)。

    Args:
        waveform: ``state_waveform`` の出力。
        path: 出力先の PNG。
        style: 配色・言語・commit。

    Returns:
        書き出した PNG のパス。

    Raises:
        ValueError: 状態が空の場合。
    """
    states = np.asarray(waveform.states, dtype=np.float64)
    if states.size == 0:
        raise ValueError("states が空です")
    signal = np.asarray(waveform.input_signal, dtype=np.float64).reshape(-1)
    steps = np.arange(states.shape[0], dtype=np.float64)

    width, height = FIGURE_SIZES["wide"]
    with rc_context_for(style):
        figure = new_figure(width, height)
        # 上段を薄くするのは、主役が下段の状態だからである。
        axes = figure.subplots(2, 1, sharex=True, height_ratios=(1.0, 2.4))
        label_panels(list(axes), style=style)
        axes[0].plot(steps, signal[: steps.size], color=TRUTH_COLOR, linewidth=1.6)
        axes[0].set_ylabel(style.label("入力 u[t]", "input u[t]"))
        colors = _unit_colors(states.shape[1])
        for column in range(states.shape[1]):
            axes[1].plot(
                steps,
                states[:, column],
                color=colors[column],
                linewidth=1.0,
                alpha=0.9,
            )
        axes[1].set_ylabel(style.label("状態 x_i[t]", "state x_i[t]"))
        axes[1].set_xlabel(
            style.label(
                f"テスト区間の先頭からのステップ (オフセット {WAVEFORM_OFFSET})",
                f"steps from the start of the test split (offset {WAVEFORM_OFFSET})",
            )
        )
        figure.suptitle(
            style.label(
                "実験 1: 同じ入力に対して、ユニットごとに違う応答が出る",
                "Experiment 1: the same input drives each unit differently",
            )
        )
        figure.supxlabel(
            wrapped_note(
                style.label(
                    f"注: 先頭の {states.shape[1]} ユニットを番号順に描いている "
                    "(D-107)。「よく散っているユニット」を選べる図にしない。",
                    f"Note: the first {states.shape[1]} units in index order"
                    " (D-107). The figure must not let anyone pick lively units.",
                )
            ),
            fontsize=8,
        )
        conditions = (
            f"task = {waveform.task}, units = {list(waveform.unit_indices)}, "
            f"steps = {WAVEFORM_OFFSET}..{WAVEFORM_OFFSET + states.shape[0]}"
        )
        add_provenance(figure, conditions, (FixedReplicate(),), style=style)
        return save_png(figure, path)


def _unit_colors(count: int) -> Sequence[tuple[float, float, float, float]]:
    """ユニット番号順の色を返す。

    Args:
        count: ユニット数。

    Returns:
        ``count`` 個の RGBA。
    """
    import matplotlib

    colormap = matplotlib.colormaps[STATE_COLORMAP]
    return [colormap(value) for value in np.linspace(0.1, 0.9, count)]


__all__ = ["STATE_COLORMAP", "plot_state_waveform"]

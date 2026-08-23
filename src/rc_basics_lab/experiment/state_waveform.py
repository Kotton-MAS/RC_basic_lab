"""リザバー状態の波形に載せる行を組む (FIG-11 追加図5 / D-107).

01 は状態空間を PCA 散布図でしか見せておらず、**状態そのもの**を見せて
いなかった。「高次元に散る」の実体は、同じ入力に対してユニットごとに
違う時定数・違う位相の応答が出ることであり、それは時間軸でしか見えない。

状態は**作り直さない** —— 学習に使った設計行列の状態列をそのまま切り出す。
作り直すと、図が示す状態と ``comparison.csv`` の係数を作った状態が別物になる。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rc_basics_lab.config import ExperimentConfig
from rc_basics_lab.experiment.runner import ESN_METHOD, ReplicatePlan
from rc_basics_lab.types import FloatArray

STATE_WAVEFORM_UNITS = 8
"""波形に描くユニット数 (D-107)。

**先頭から順に取る。** 「よく散っているユニット」を選べる図にしない。
8 本なのは、それ以上重ねると個々の線が追えなくなるため。
"""


@dataclass(frozen=True, slots=True)
class StateWaveform:
    """状態波形パネル1枚ぶんの入力 (FIG-11 追加図5)。

    Attributes:
        task: 課題名。見出しは作図側が引く。
        input_signal: 上段に描く入力 ``(T,)``。
        states: 下段に描く状態 ``(T, STATE_WAVEFORM_UNITS)``。
        unit_indices: 描いたユニットの番号 (脚注に出す)。
    """

    task: str
    input_signal: FloatArray
    states: FloatArray
    unit_indices: tuple[int, ...]


def state_waveform(
    config: ExperimentConfig, plan: ReplicatePlan, task: str
) -> StateWaveform:
    """ESN の状態列を固定の窓で切り出す (D-107)。

    Args:
        config: 01 の設定 (窓の決定には使わない。呼び出しの形をそろえるため)。
        plan: レプリケート0の ``ReplicatePlan``。
        task: 課題名 (窓の長さの決定に使う)。

    Returns:
        入力と状態を同じ窓で切り出した ``StateWaveform``。

    Raises:
        ValueError: ESN の設計行列に状態列が無い場合。
    """
    from rc_basics_lab.plotting.waveforms import slice_window

    del config
    design = plan.designs[ESN_METHOD][0]
    columns = [
        index for index, name in enumerate(design.feature_names) if name.startswith("x")
    ]
    if not columns:
        raise ValueError(f"ESN の設計行列に状態列がありません: {design.feature_names}")
    units = tuple(range(min(STATE_WAVEFORM_UNITS, len(columns))))
    start = plan.split.test.start
    states = np.column_stack(
        [slice_window(design.phi[:, columns[unit]], start, task) for unit in units]
    )
    return StateWaveform(
        task=task,
        input_signal=slice_window(plan.task.u, start, task),
        states=states,
        unit_indices=units,
    )


__all__ = ["STATE_WAVEFORM_UNITS", "StateWaveform", "state_waveform"]

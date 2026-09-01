"""課題の生成口 — **生成関数を直接呼ぶ場所を1つにする** (D-123).

``reservoir/registry.py`` と同じ役割である。実験層はここだけを呼ぶので、
課題を1つ足しても生成箇所を触らずに済む。
"""

from __future__ import annotations

import numpy as np

from rc_basics_lab.config import DelayParityConfig, MackeyGlassConfig
from rc_basics_lab.tasks.base import TaskData
from rc_basics_lab.tasks.delay_parity import generate_delay_parity
from rc_basics_lab.tasks.mackey_glass import generate_mackey_glass
from rc_basics_lab.tasks.protocol import TaskConfig


def build_task(config: TaskConfig, rng: np.random.Generator) -> TaskData:
    """設定から課題データを1本作る (**課題の分岐はここだけ**)。

    課題を足すときは ``TaskConfig`` の union に設定型を足し、ここへ ``case``
    を1つ書く —— mypy が網羅性を見るので、書き忘れは型検査で落ちる
    (実行時に「なぜか Mackey-Glass になっている」にはならない)。

    Args:
        config: 課題の生成パラメータ。
        rng: 課題生成用の Generator (``seeds.make_rng`` の task ストリーム)。

    Returns:
        ``TaskData``。

    Raises:
        ValueError: 設定値が範囲外の場合 (各生成関数が投げる)。
    """
    match config:
        case MackeyGlassConfig():
            return generate_mackey_glass(config, rng)
        case DelayParityConfig():
            return generate_delay_parity(config, rng)


__all__ = ["build_task"]

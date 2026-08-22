"""波形図に載せる (真値, 手法 -> 予測) を組む (FIG-11 / D-107).

01 と 03 の両方が同じものを要る。**片方にだけ書くと、もう片方が
自分で書き直すことになる** (D-92 の同名ラチェットが止める形) ので、
最初から1箇所に置く。

学習は 1 ステップ先の読み出しそのもので、**図のために作り直さない** ——
作り直すと ``comparison.csv`` / ``narma10.csv`` の行と別物の予測になる。
"""

from __future__ import annotations

from rc_basics_lab.config import ExperimentConfig
from rc_basics_lab.experiment.runner import ReplicatePlan, build_methods
from rc_basics_lab.readout.ridge import fit_ridge, predict
from rc_basics_lab.types import FloatArray


def waveform_predictions(
    config: ExperimentConfig, plan: ReplicatePlan
) -> tuple[FloatArray, dict[str, FloatArray]]:
    """テスト区間の固定窓について (真値, 手法 -> 予測) を返す (D-107)。

    Args:
        config: 01 の設定 (``narma.base`` を含む)。
        plan: レプリケート 0 の ``ReplicatePlan``。**D-107 が固定した
            レプリケートそのもの**を渡す。

    Returns:
        ``(真値, {手法名: 予測})``。どれも同じ長さに切り出してある。

    Raises:
        ValueError: 手法が1つも無い場合。
    """
    from rc_basics_lab.plotting.waveforms import slice_window

    methods = build_methods(config)
    if not methods:
        raise ValueError("手法が空です")
    split = plan.split
    start = split.test.start
    truth = slice_window(plan.task.y, start)
    predictions: dict[str, FloatArray] = {}
    for method in methods:
        design = plan.designs[method.name][0]
        coefficients = fit_ridge(
            design.phi[split.train.start : split.train.stop],
            plan.task.y[split.train.start : split.train.stop],
            config.ridge.alpha_grid[0],
            bias_column=design.bias_column,
        )
        predictions[method.name] = slice_window(
            predict(design.phi, coefficients), start
        )
    return truth, predictions


__all__ = ["waveform_predictions"]

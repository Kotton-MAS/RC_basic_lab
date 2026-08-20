"""運用閾値の決定と閾値掃引 (5-B) —— テストラベルは**引数に取らない** (D-56).

このモジュールの要点は署名にある:

- ``calibrate_threshold(calibration_scores, target_false_alarm_rate)`` は
  **ラベルを1つも受け取らない** (D-56)。
- テスト側最適化は ``best_test_f1`` という**別の関数**で、名前も戻り値も
  「参考値」であることを言っている。結果は ``f1_test_optimal`` という
  **別列**にのみ出る (D-56)。

``best_test_f1`` は掃引の格子ではなく **PR 曲線の全点**の最大を返す
(``f1_test_optimal >= f1_calibrated`` が定義から成り立つ理由。D-56)。

5-B の掃引は「警報予算 (誤報率) を振ると P / R / F1 がどう動くか」を出す。
閾値の切り方は ``metrics_detection.threshold_at_false_alarm_rate`` を
そのまま使う (順位で切る、D-56 の計算側) —— 掃引だけ別の切り方にすると、
図の上の1点と運用点が別の規則で決まることになる。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rc_basics_lab.metrics_detection import (
    BoolArray,
    point_scores,
    precision_recall_curve,
    threshold_at_false_alarm_rate,
)
from rc_basics_lab.types import FloatArray


def calibrate_threshold(
    calibration_scores: FloatArray, target_false_alarm_rate: float
) -> float:
    """較正区間のスコア分位点から運用閾値を決める (D-56)。

    **ラベルを引数に取らない**。``metrics_detection.
    threshold_at_false_alarm_rate`` への薄い委譲だが、実験層がこの名前を
    通ることで「閾値はここでしか決まらない」が読み取れる。

    Args:
        calibration_scores: ``(T_cal,)`` の較正区間スコア。
        target_false_alarm_rate: 目標誤報率 ``0 < rate <= 1``。

    Returns:
        ``scores >= threshold`` を警報とする閾値。
    """
    return threshold_at_false_alarm_rate(calibration_scores, target_false_alarm_rate)


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """ある閾値での運用点。

    Attributes:
        threshold: 使った閾値 (``scores >= threshold`` が警報)。
        precision: 適合率。
        recall: 再現率。
        f1: F1。
        false_alarm_rate: 陰性点のうち警報を出した割合 ``FP / (FP + TN)``。
        n_alarms: 警報を出した点数。
    """

    threshold: float
    precision: float
    recall: float
    f1: float
    false_alarm_rate: float
    n_alarms: int


def alarms_at(scores: FloatArray, threshold: float) -> BoolArray:
    """``scores >= threshold`` の警報マスク (**この不等号を1箇所に閉じる**)。

    ``>`` と ``>=`` が混ざると、同順位のスコアがあるときに警報数が
    掃引の点と運用点で食い違う。
    """
    mask: BoolArray = np.asarray(scores, dtype=np.float64) >= threshold
    return mask


def evaluate_at_threshold(
    labels: BoolArray, scores: FloatArray, threshold: float
) -> OperatingPoint:
    """固定した閾値での運用点を返す (閾値を**選ばない**)。

    Args:
        labels: ``(T,)`` の真の異常ラベル。
        scores: ``(T,)`` の異常スコア。
        threshold: ``calibrate_threshold`` が決めた閾値。
    """
    label_array = np.asarray(labels).astype(np.bool_)
    predictions = alarms_at(scores, threshold)
    scored = point_scores(label_array, predictions)
    n_negative = int(np.count_nonzero(~label_array))
    false_positive = int(np.count_nonzero(predictions & ~label_array))
    false_alarm_rate = (
        float(false_positive) / float(n_negative) if n_negative > 0 else 0.0
    )
    return OperatingPoint(
        threshold=float(threshold),
        precision=scored.precision,
        recall=scored.recall,
        f1=scored.f1,
        false_alarm_rate=false_alarm_rate,
        n_alarms=int(np.count_nonzero(predictions)),
    )


def best_test_f1(labels: BoolArray, scores: FloatArray) -> float:
    """テスト側で閾値を最適化したときの F1 (**参考値**、D-56)。

    PR 曲線の全点の F1 の最大を返す。曲線の点は「相異なるスコアで切った
    ときの (P, R)」の全体なので、``evaluate_at_threshold`` がどんな閾値で
    返した F1 も必ずこの最大以下になる。

    Args:
        labels: ``(T,)`` の真の異常ラベル。
        scores: ``(T,)`` の異常スコア。

    Raises:
        ValueError: 陽性が1つも無い場合 (``precision_recall_curve`` と同じ)。
    """
    curve = precision_recall_curve(labels, scores)
    denominator = curve.precision + curve.recall
    f1: FloatArray = np.divide(
        2.0 * curve.precision * curve.recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    return float(np.max(f1))


def sweep_thresholds(
    labels: BoolArray, scores: FloatArray, n_points: int
) -> tuple[OperatingPoint, ...]:
    """警報予算を等間隔に振った掃引 (5-B の行の素)。

    ``i / n_points`` (``i = 1..n_points``) を目標警報率として
    ``threshold_at_false_alarm_rate`` で切る。**運用点と同じ切り方**なので、
    図の曲線の上に運用点がそのまま乗る。

    Args:
        labels: ``(T,)`` の真の異常ラベル。
        scores: ``(T,)`` の異常スコア。
        n_points: 掃引の点数 (``threshold.sweep_points``)。

    Raises:
        ValueError: ``n_points`` が 1 未満の場合。
    """
    if n_points < 1:
        raise ValueError(f"sweep_points は 1 以上が必要です: {n_points}")
    return tuple(
        evaluate_at_threshold(
            labels,
            scores,
            threshold_at_false_alarm_rate(scores, float(index) / float(n_points)),
        )
        for index in range(1, n_points + 1)
    )


__all__ = [
    "OperatingPoint",
    "alarms_at",
    "best_test_f1",
    "calibrate_threshold",
    "evaluate_at_threshold",
    "sweep_thresholds",
]

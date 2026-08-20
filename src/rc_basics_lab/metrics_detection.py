"""異常検知の評価指標 (D-54 / D-55 / D-62).

`metrics.py` が予測誤差 (NRMSE 系) を担うのに対し、こちらは
**ラベル付き時系列に対する検知性能**を担う。既存の `metrics.py` には
1バイトも触れず、同じ「形状検証 → 計算」の並びだけを踏襲する。

このモジュールが守る決定:

- **D-54**: AUPRC は average precision (階段和) ``Σ (R_n - R_{n-1}) · P_n`` で
  計算する。台形則・線形補間を使わない。台形則は補間の分だけ楽観バイアスが
  乗るが、値は 0〜1 に収まり曲線も滑らかなので**図でも CSV でも壊れて見えない**。
- **D-55**: point-adjust の F1 は `PointAdjustReport` (一様乱数対照つき) でしか
  取得できない。`pa_f1` 単独を返す公開関数を置かない。PA-F1 は「高い値が出る」
  形でしか壊れないため、単独で報告された瞬間に読者も実装者も検出できない。
- **D-59**: ネットワーク・ファイル I/O を1行も持たない純関数層。import は
  標準ライブラリと numpy だけ。
- **D-62**: 自前実装。scikit-learn は dev グループのテストオラクルとしてのみ使う。

同順位 (同スコア) の扱いは **1つの閾値に畳む**。`sklearn.metrics.
average_precision_score` と同じ規則であり、`tests/test_metrics_detection.py::
test_matches_scikit_learn_average_precision_on_random_inputs` が
ランダム入力 1000 ケースで厳密一致を要求する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from rc_basics_lab.types import FloatArray

type BoolArray = npt.NDArray[np.bool_]
"""bool の numpy 配列。ラベル・マスク・予測に使う (形状は各 API に記す)。"""


def _as_labeled_scores(
    labels: BoolArray, scores: FloatArray
) -> tuple[BoolArray, FloatArray]:
    """``(labels, scores)`` を検証して ``(bool, float64)`` の1次元ペアに揃える。"""
    label_array = np.asarray(labels)
    score_array = np.asarray(scores, dtype=np.float64)
    if label_array.ndim != 1 or score_array.ndim != 1:
        raise ValueError(
            "labels と scores は1次元配列が必要です: "
            f"{label_array.shape}, {score_array.shape}"
        )
    if label_array.shape != score_array.shape:
        raise ValueError(
            "labels と scores の形状が一致しません: "
            f"{label_array.shape} != {score_array.shape}"
        )
    if label_array.size == 0:
        raise ValueError("空の系列に対して検知指標は定義されません")
    if not np.all(np.isfinite(score_array)):
        raise ValueError(
            "scores に非有限値が含まれます (nan/inf は順序が定義できません)"
        )
    return label_array.astype(np.bool_), score_array


def _as_mask_pair(first: BoolArray, second: BoolArray) -> tuple[BoolArray, BoolArray]:
    """同形状の1次元 bool 配列2本に揃える。"""
    left = np.asarray(first)
    right = np.asarray(second)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError(f"1次元配列が必要です: {left.shape}, {right.shape}")
    if left.shape != right.shape:
        raise ValueError(f"形状が一致しません: {left.shape} != {right.shape}")
    if left.size == 0:
        raise ValueError("空の系列に対して検知指標は定義されません")
    return left.astype(np.bool_), right.astype(np.bool_)


@dataclass(frozen=True, slots=True)
class PrecisionRecallCurve:
    """PR 曲線の点列。**図も CSV もこの1本から作る** (2実装に割れないように)。

    点は閾値の降順 = recall の非減少順に並ぶ。各点 ``n`` は
    「``scores >= threshold[n]`` を陽性と予測したとき」の値であり、
    同順位のスコアは**1つの点に畳まれている** (sklearn と同じ規則)。

    ``(recall=0, precision=1)`` のような合成端点は**含めない**。
    階段和 ``Σ (R_n - R_{n-1}) · P_n`` (``R_0 = 0``) がそのまま
    `average_precision` の定義になるようにするためで、作図側で端点が要る場合は
    呼び出し側で足す。

    Attributes:
        threshold: ``(N,)`` 降順の閾値 (入力スコアの相異なる値)。
        precision: ``(N,)`` 各閾値での適合率。分母 0 の点は 0.0。
        recall: ``(N,)`` 各閾値での再現率 (非減少)。
    """

    threshold: FloatArray
    precision: FloatArray
    recall: FloatArray

    def __post_init__(self) -> None:
        shapes = {self.threshold.shape, self.precision.shape, self.recall.shape}
        if len(shapes) != 1:
            raise ValueError(f"点列の形状が揃っていません: {shapes}")

    @property
    def n_points(self) -> int:
        """点の個数 (= 相異なるスコアの個数)。"""
        return int(self.threshold.shape[0])

    def average_precision(self) -> float:
        """階段和 ``Σ (R_n - R_{n-1}) · P_n`` (``R_0 = 0``)。D-54。"""
        previous_recall: FloatArray = np.concatenate(
            (np.zeros(1, dtype=np.float64), self.recall[:-1])
        )
        return float(np.sum((self.recall - previous_recall) * self.precision))


def precision_recall_curve(
    labels: BoolArray, scores: FloatArray
) -> PrecisionRecallCurve:
    """PR 曲線の点列を作る (同順位は1つの閾値に畳む)。

    Args:
        labels: ``(T,)`` の真の異常ラベル。
        scores: ``(T,)`` の異常スコア (大きいほど異常)。

    Raises:
        ValueError: 形状不正、空、非有限スコア、または陽性が1つも無い場合
            (陽性ゼロでは recall が定義できず、黙って 0 を返すと下流の集計が
            静かに壊れるため即座に失敗させる。`metrics.nrmse` と同じ規律)。
    """
    label_array, score_array = _as_labeled_scores(labels, scores)
    n_positive = int(np.count_nonzero(label_array))
    if n_positive == 0:
        raise ValueError("labels に陽性が1つも無いため recall を定義できません")

    order = np.argsort(score_array, kind="stable")[::-1]
    sorted_scores = score_array[order]
    sorted_labels = label_array[order].astype(np.float64)

    distinct = np.flatnonzero(np.diff(sorted_scores))
    threshold_index = np.concatenate(
        (distinct, np.array([sorted_scores.size - 1], dtype=distinct.dtype))
    )
    true_positive: FloatArray = np.cumsum(sorted_labels)[threshold_index]
    predicted_positive: FloatArray = (1.0 + threshold_index - true_positive).astype(
        np.float64
    )

    precision: FloatArray = np.zeros_like(true_positive)
    np.divide(
        true_positive,
        predicted_positive,
        out=precision,
        where=predicted_positive != 0.0,
    )
    recall: FloatArray = true_positive / float(n_positive)
    return PrecisionRecallCurve(
        threshold=sorted_scores[threshold_index],
        precision=precision,
        recall=recall,
    )


def average_precision(labels: BoolArray, scores: FloatArray) -> float:
    """AUPRC を average precision (階段和) で計算する (D-54)。

    ``Σ_n (R_n - R_{n-1}) · P_n``。**台形則・線形補間を使わない**。
    `sklearn.metrics.average_precision_score` と同じ値になる
    (同順位は1つの閾値に畳む)。

    Args:
        labels: ``(T,)`` の真の異常ラベル。
        scores: ``(T,)`` の異常スコア (大きいほど異常)。
    """
    return precision_recall_curve(labels, scores).average_precision()


@dataclass(frozen=True, slots=True)
class PointScores:
    """点単位の適合率・再現率・F1。

    分母が 0 になる点 (陽性を1つも予測しなかった等) は 0.0 とする。
    閾値掃引では必ず通る境界であり、例外にすると掃引そのものが書けない。

    Attributes:
        precision: 適合率 ``TP / (TP + FP)``。
        recall: 再現率 ``TP / (TP + FN)``。
        f1: 調和平均 ``2PR / (P + R)``。
    """

    precision: float
    recall: float
    f1: float


def point_scores(labels: BoolArray, predictions: BoolArray) -> PointScores:
    """点単位の適合率・再現率・F1 を計算する。

    Args:
        labels: ``(T,)`` の真の異常ラベル。
        predictions: ``(T,)`` の予測 (True = 異常と判定)。
    """
    label_array, prediction_array = _as_mask_pair(labels, predictions)
    true_positive = float(np.count_nonzero(label_array & prediction_array))
    n_predicted = float(np.count_nonzero(prediction_array))
    n_actual = float(np.count_nonzero(label_array))
    precision = true_positive / n_predicted if n_predicted > 0.0 else 0.0
    recall = true_positive / n_actual if n_actual > 0.0 else 0.0
    denominator = precision + recall
    f1 = 2.0 * precision * recall / denominator if denominator > 0.0 else 0.0
    return PointScores(precision=precision, recall=recall, f1=f1)


def _segment_bounds(
    labels: BoolArray,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """連続する True の区間の ``(start, stop)`` を返す (stop は排他)。"""
    padded = np.concatenate(
        (np.zeros(1, dtype=np.bool_), labels, np.zeros(1, dtype=np.bool_))
    )
    changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return (changes[0::2].astype(np.int64), changes[1::2].astype(np.int64))


def _point_adjusted(
    labels: BoolArray, predictions: BoolArray, k_percent: float
) -> BoolArray:
    """PA%K を適用した予測マスクを返す (Kim et al. AAAI 2022)。

    真の異常区間のうち、検知率が ``k_percent`` % を**超えた**区間だけを
    区間まるごと検知済みに書き換える。``k_percent = 0`` で従来の point-adjust
    (1点でも当たれば全部当たり扱い)、``k_percent = 100`` で素の点単位評価に
    一致する (検知率が 1 を超えることは無いので、どの区間も書き換わらない)。
    """
    adjusted = predictions.copy()
    starts, stops = _segment_bounds(labels)
    ratio_threshold = k_percent / 100.0
    for start, stop in zip(starts, stops, strict=True):
        segment = predictions[start:stop]
        detected_ratio = float(np.count_nonzero(segment)) / float(stop - start)
        if detected_ratio > ratio_threshold:
            adjusted[start:stop] = True
    return adjusted


def _top_alarm_mask(scores: FloatArray, n_alarms: int) -> BoolArray:
    """スコア上位 ``n_alarms`` 点だけを True にするマスク。

    同順位があっても**予測陽性数が厳密に ``n_alarms``** になる。対照との
    「警報予算をそろえる」比較でずれが出ないようにするため、分位点ではなく
    順位で切る。
    """
    mask = np.zeros(scores.shape, dtype=np.bool_)
    if n_alarms <= 0:
        return mask
    order = np.argsort(scores, kind="stable")[::-1]
    mask[order[:n_alarms]] = True
    return mask


@dataclass(frozen=True, slots=True)
class PointAdjustReport:
    """point-adjust の F1 と、**同じ警報予算での一様乱数対照** (D-55)。

    `pa_f1` 単独を返す公開関数はこのモジュールに存在しない。PA-F1 は
    「高い値が出る」形でしか壊れないため、対照と同時にしか取れない型に
    しておかないと、単独で報告された瞬間に誰も検出できない
    (Kim et al. AAAI 2022: 一様乱数の F1_PA が5データセット中4つで SOTA 超え)。

    Attributes:
        pa_f1: 評価対象スコアの PA%K F1。
        pa_f1_random: 一様乱数スコアを**同じ警報数**で切ったときの PA%K F1。
        k: PA%K の K [%]。``0`` が従来の point-adjust、``100`` が素の F1。
    """

    pa_f1: float
    pa_f1_random: float
    k: float


def point_adjust_report(
    labels: BoolArray,
    predictions: BoolArray,
    control_scores: FloatArray,
    *,
    k: float = 0.0,
) -> PointAdjustReport:
    """PA%K の F1 を、一様乱数対照とセットでのみ返す (D-55)。

    対照は ``control_scores`` の上位から ``predictions`` と**同じ個数**だけ
    警報を出したものとする。警報予算をそろえないと「対照の方が警報を多く
    出しているから高い」という逃げ道が残る。

    Args:
        labels: ``(T,)`` の真の異常ラベル。
        predictions: ``(T,)`` の評価対象の予測 (True = 異常と判定)。
        control_scores: ``(T,)`` の一様乱数スコア。呼び出し側が
            `numpy.random.Generator` から引く (このモジュールは乱数源を持たない)。
        k: PA%K の K [%]。``0 <= k <= 100``。

    Raises:
        ValueError: 形状不正、空、または ``k`` が範囲外の場合。
    """
    label_array, prediction_array = _as_mask_pair(labels, predictions)
    control_array = np.asarray(control_scores, dtype=np.float64)
    if control_array.shape != label_array.shape:
        raise ValueError(
            "control_scores の形状が labels と一致しません: "
            f"{control_array.shape} != {label_array.shape}"
        )
    if not 0.0 <= k <= 100.0:
        raise ValueError(f"k は 0 以上 100 以下の百分率です: {k}")

    n_alarms = int(np.count_nonzero(prediction_array))
    control_prediction = _top_alarm_mask(control_array, n_alarms)
    return PointAdjustReport(
        pa_f1=point_scores(
            label_array, _point_adjusted(label_array, prediction_array, k)
        ).f1,
        pa_f1_random=point_scores(
            label_array, _point_adjusted(label_array, control_prediction, k)
        ).f1,
        k=float(k),
    )


def threshold_at_false_alarm_rate(
    calibration_scores: FloatArray, target_false_alarm_rate: float
) -> float:
    """較正区間のスコア分位点から運用閾値を決める (D-56 の計算側)。

    ``scores >= threshold`` を警報とするとき、較正区間での警報数が
    ``floor(rate * n)`` (最低1) になる閾値を返す。分位点の補間を使わず
    順位で切るのは、補間した閾値が「較正区間のどのスコアとも一致しない値」に
    なり、警報数が目標からずれるため。

    **テスト区間のラベルを引数に取らない** ことが D-56 の本体であり、
    この署名がその型側の強制になる。

    Args:
        calibration_scores: ``(T,)`` の較正区間スコア。
        target_false_alarm_rate: 目標誤報率 ``0 < rate <= 1``。

    Raises:
        ValueError: 形状不正、空、非有限、または誤報率が範囲外の場合。
    """
    score_array = np.asarray(calibration_scores, dtype=np.float64)
    if score_array.ndim != 1:
        raise ValueError(
            f"calibration_scores は1次元配列が必要です: {score_array.shape}"
        )
    if score_array.size == 0:
        raise ValueError("空の較正区間からは閾値を決められません")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("calibration_scores に非有限値が含まれます")
    if not 0.0 < target_false_alarm_rate <= 1.0:
        raise ValueError(
            "target_false_alarm_rate は 0 より大きく 1 以下である必要があります: "
            f"{target_false_alarm_rate}"
        )
    n_alarms = max(1, int(np.floor(target_false_alarm_rate * score_array.size)))
    descending: FloatArray = np.sort(score_array)[::-1]
    return float(descending[n_alarms - 1])


@dataclass(frozen=True, slots=True)
class MaskedEvaluation:
    """``is_ignored`` を落とした後の評価対象。

    Attributes:
        labels: ``(T',)`` 残った点の真ラベル。
        scores: ``(T',)`` 残った点のスコア。
    """

    labels: BoolArray
    scores: FloatArray


def apply_ignore_mask(
    labels: BoolArray, scores: FloatArray, is_ignored: BoolArray
) -> MaskedEvaluation:
    """評価対象から ``is_ignored`` の点を落とす。

    **点単位の指標 (AP / P / R / F1) 専用**である。点を落とすと異常区間の
    連続性が壊れるため、point-adjust 系はマスク前の系列で計算すること。

    Args:
        labels: ``(T,)`` の真の異常ラベル。
        scores: ``(T,)`` の異常スコア。
        is_ignored: ``(T,)`` True の点を評価から除く。

    Raises:
        ValueError: 形状不正、空、または全点が除外される場合。
    """
    label_array, score_array = _as_labeled_scores(labels, scores)
    ignore_array = np.asarray(is_ignored)
    if ignore_array.shape != label_array.shape:
        raise ValueError(
            "is_ignored の形状が labels と一致しません: "
            f"{ignore_array.shape} != {label_array.shape}"
        )
    keep = ~ignore_array.astype(np.bool_)
    if not np.any(keep):
        raise ValueError("is_ignored が全点を除外しているため評価できません")
    return MaskedEvaluation(labels=label_array[keep], scores=score_array[keep])


__all__ = [
    "BoolArray",
    "MaskedEvaluation",
    "PointAdjustReport",
    "PointScores",
    "PrecisionRecallCurve",
    "apply_ignore_mask",
    "average_precision",
    "point_adjust_report",
    "point_scores",
    "precision_recall_curve",
    "threshold_at_false_alarm_rate",
]

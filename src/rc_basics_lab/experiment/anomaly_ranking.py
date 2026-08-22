"""順位と「対照と区別できるか」の印 (5-C / 5-D が共有する純関数層、D-78).

``anomaly_sweep.py`` から分けたのは D-63 の上限に当たったためで
(1本に書くと 686 行)、分ける単位としても素直である —— ここには**行から数を
作る計算しかなく**、系列も設定も源も知らない。掃引の配線 (格子の組み立てと
実行) は ``anomaly_sweep.py`` にある。

このモジュールが持つ規律は2つ:

1. **集計から対照を落とせない** (D-61)。``aggregate_methods`` は
   ``ANOMALY_METHODS`` と過不足なく一致する行しか受け取らない ——
   一様乱数と入力ノルムを外した順位表を作る経路が無い
2. **順位は6系統すべてに付ける** (D-78)。「対照を超えた系統だけで順位を
   計算する」除外方式は採らず、代わりに各系統へ
   「一様乱数対照と区別できるか」の印を付ける。除外方式にすると、
   除外の閾値をどこに置くかが新しい任意性になる

印の根拠 (``n_pairs`` / ``n_better_than_control`` / ``control_sign_p``) は
``MethodAggregate`` が全部持つので、成果物の行だけで印の再判定ができる。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from rc_basics_lab.experiment.anomaly_rows import AnomalyRow
from rc_basics_lab.experiment.anomaly_score import ANOMALY_METHODS
from rc_basics_lab.metrics_significance import sign_test_p_value

CONTROL_SIGN_TEST_ALPHA = 0.05
"""「一様乱数対照と区別できる」と印を付ける有意水準 (片側符号検定、D-78)。

**設定の葉にしない**。印の根拠を行がすべて持ち歩くので、別の水準で読み直したい
読者は CSV の3列から自分で判定できる —— 葉にすると「水準を緩めて印を増やした
図」が作れてしまう。
"""


@dataclass(frozen=True, slots=True)
class MethodAggregate:
    """1条件 x 1系統の集計 (成果物の行ではなく計算の途中結果)。

    Attributes:
        method: 系統名。
        auprc_mean: ``auprc`` の平均。
        auprc_sd: ``auprc`` の標準偏差 (``ddof=1``。対が1つなら 0)。
        auprc_random_mean: ``auprc_random`` の平均 (D-61)。
        n_pairs: (系列, レプリケート) の対の数。
        n_better_than_control: ``auprc > auprc_random`` だった対の数。
        control_sign_p: 片側符号検定の p 値。
        distinguishable: 一様乱数対照と区別できるか (D-78)。
    """

    method: str
    auprc_mean: float
    auprc_sd: float
    auprc_random_mean: float
    n_pairs: int
    n_better_than_control: int
    control_sign_p: float
    distinguishable: bool


def kendall_tau(first: Sequence[float], second: Sequence[float]) -> float:
    """2つの並びの Kendall tau-b (同値を含む順位相関)。

    Args:
        first: 基準側の値 (大きいほど上位)。
        second: 比較側の値。``first`` と同じ順 (同じ系統) で並べる。

    Returns:
        ``(C - D) / sqrt((n0 - n1)(n0 - n2))``。どちらかの並びが全同値で
        分母が 0 になる場合は ``nan`` (順序が定義できない)。

    Raises:
        ValueError: 長さが違う、または2要素未満の場合。
    """
    if len(first) != len(second):
        raise ValueError(f"長さが違います: {len(first)} != {len(second)}")
    if len(first) < 2:
        raise ValueError(f"2要素以上が必要です: {len(first)}")
    concordant = discordant = tied_first = tied_second = 0
    for left in range(len(first)):
        for right in range(left + 1, len(first)):
            delta_first = first[left] - first[right]
            delta_second = second[left] - second[right]
            if delta_first == 0.0:
                tied_first += 1
            if delta_second == 0.0:
                tied_second += 1
            if delta_first == 0.0 or delta_second == 0.0:
                continue
            if delta_first * delta_second > 0.0:
                concordant += 1
            else:
                discordant += 1
    n0 = len(first) * (len(first) - 1) // 2
    denominator = math.sqrt(float((n0 - tied_first) * (n0 - tied_second)))
    if denominator == 0.0:
        return math.nan
    return float(concordant - discordant) / denominator


def _aggregate_one(method: str, rows: Sequence[AnomalyRow]) -> MethodAggregate:
    values = np.asarray([row.auprc for row in rows], dtype=np.float64)
    control = np.asarray([row.auprc_random for row in rows], dtype=np.float64)
    n_pairs = int(values.size)
    n_better = int(np.count_nonzero(values > control))
    p_value = sign_test_p_value(n_pairs, n_better)
    return MethodAggregate(
        method=method,
        auprc_mean=float(np.mean(values)),
        auprc_sd=float(np.std(values, ddof=1)) if n_pairs > 1 else 0.0,
        auprc_random_mean=float(np.mean(control)),
        n_pairs=n_pairs,
        n_better_than_control=n_better,
        control_sign_p=p_value,
        distinguishable=p_value <= CONTROL_SIGN_TEST_ALPHA,
    )


def aggregate_methods(rows: Sequence[AnomalyRow]) -> Mapping[str, MethodAggregate]:
    """5-A の行を系統ごとに畳む (鍵の順は ``ANOMALY_METHODS``)。

    対照 (``random_control`` / ``input_norm_control``) も畳む —— 集計で
    落とせるようにすると、対照の無い順位表が作れてしまう (D-61)。

    Raises:
        ValueError: 行が空、または行に現れる系統が ``ANOMALY_METHODS`` と
            一致しない場合。
    """
    if not rows:
        raise ValueError("集計する行がありません")
    grouped: dict[str, list[AnomalyRow]] = {}
    for row in rows:
        grouped.setdefault(row.method, []).append(row)
    if set(grouped) != set(ANOMALY_METHODS):
        raise ValueError(
            "系統の集合が ANOMALY_METHODS と一致しません: "
            f"{sorted(grouped)} != {sorted(ANOMALY_METHODS)}"
        )
    return {
        method: _aggregate_one(method, grouped[method]) for method in ANOMALY_METHODS
    }


def method_ranks(aggregates: Mapping[str, MethodAggregate]) -> dict[str, int]:
    """``auprc_mean`` の降順の順位 (1 が最良、同値は同順位)。

    同値に同じ順位を与える (競技順位) —— 並び順で先に来たほうを上位にすると、
    ``ANOMALY_METHODS`` の宣言順が順位に漏れる。
    """
    means = {method: item.auprc_mean for method, item in aggregates.items()}
    return {
        method: 1 + sum(1 for other in means.values() if other > mean)
        for method, mean in means.items()
    }


def discordant_counts(
    reference: Mapping[str, MethodAggregate], current: Mapping[str, MethodAggregate]
) -> tuple[int, int]:
    """基準と順序が逆転した系統対の数と、そのうち両方に印がある対の数 (D-78)。

    「両方に印がある」は**両条件で**区別できることを要求する —— 片方の条件で
    しか対照から離れていない系統の順位変化は、雑音と区別がつかない。

    Args:
        reference: 基準条件の集計。
        current: 比較する条件の集計。

    Returns:
        ``(逆転した対の数, そのうち4つの印がすべて立っている対の数)``。

    Raises:
        ValueError: どちらかが ``ANOMALY_METHODS`` を網羅していない場合。
    """
    for name, aggregates in (("reference", reference), ("current", current)):
        if set(aggregates) != set(ANOMALY_METHODS):
            raise ValueError(f"{name} が ANOMALY_METHODS を網羅していません")
    methods = list(ANOMALY_METHODS)
    discordant = 0
    marked = 0
    for left in range(len(methods)):
        for right in range(left + 1, len(methods)):
            first, second = methods[left], methods[right]
            delta_reference = reference[first].auprc_mean - reference[second].auprc_mean
            delta_current = current[first].auprc_mean - current[second].auprc_mean
            if delta_reference == 0.0 or delta_current == 0.0:
                continue
            if delta_reference * delta_current > 0.0:
                continue
            discordant += 1
            if all(
                item.distinguishable
                for item in (
                    reference[first],
                    reference[second],
                    current[first],
                    current[second],
                )
            ):
                marked += 1
    return discordant, marked


__all__ = [
    "CONTROL_SIGN_TEST_ALPHA",
    "MethodAggregate",
    "aggregate_methods",
    "discordant_counts",
    "kendall_tau",
    "method_ranks",
    "sign_test_p_value",
]

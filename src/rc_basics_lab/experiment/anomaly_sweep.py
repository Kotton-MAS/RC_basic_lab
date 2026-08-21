"""実験 5-C (プロトコル感度) / 5-D (N と性能) の掃引 (D-57 / D-78 / D-79).

掃引は**新しい実験ではない**。格子点ごとに設定を差し替えて 5-A をそのまま
回し、出てきた行を集計するだけである:

    ``run_anomaly_headline(apply_protocol_condition(config, 条件), sources)``

条件の適用は ``dataclasses.replace`` 1本で、前処理を作る経路は 5-A と同じ
``plan_anomaly_replicate`` -> ``AnomalyPreprocessor.from_training_prefix``
しかない (D-57)。このモジュールは ``AnomalyPreprocessor`` を名前でも知らない
——「掃引の側にもう1つ前処理の実装が生える」経路が構造上書けない。

**基準の条件は 5-A と同じ条件**である (D-79)。5-C は ``config.preprocess`` と
一致する格子点、5-D は ``config.reservoir.n_units`` の行が基準で、どちらも
格子に含まれていなければ ``ValueError`` にする。含まれていないと
「5-A と同じ条件の行」が成果物に無いまま順位や劣化点だけが出てしまい、
2つの実験が同じ前処理を通っていることを CSV だけでは確かめられない。

5-C の測り方 (ユーザー確定、D-78): **全系統の順位を記録したうえで、各系統に
「一様乱数対照と区別できるか」の印を付ける**。対照を超えた系統だけで順位を
計算する除外方式は採らない —— 除外の閾値をどこに置くかが新しい任意性になる。
印が無いと、対照と区別できない系統どうしの雑音の入れ替わりが
「プロトコルに敏感」と読まれる。

5-D の前提 (D-78): **全系列が同じ学習量 (``n_train``) で回っていること**を
入口で検査する。系列ごとに ``train_end`` が違う源では、学習量不足による劣化と
N 不足による劣化が混ざって測定そのものが壊れる。
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import product

from rc_basics_lab.config import Anomaly05Config
from rc_basics_lab.experiment.anomaly import run_anomaly_headline
from rc_basics_lab.experiment.anomaly_ranking import (
    MethodAggregate,
    aggregate_methods,
    kendall_tau,
)
from rc_basics_lab.experiment.anomaly_rows import (
    AnomalyRow,
    ProtocolSweepRow,
    SizeSweepRow,
)
from rc_basics_lab.experiment.anomaly_score import ANOMALY_METHODS, ESN_RESIDUAL
from rc_basics_lab.tasks.anomaly import SeriesSource

logger = logging.getLogger(__name__)

DEGRADATION_FRACTION = 0.9
"""5-D の劣化点の定義 (基準 N の AUPRC の何割を割ったら「落ちた」とするか)。

**設定の葉にしない** —— 報告する量の名前 ``n_units_at_90pct`` がこの値
そのものなので、葉にすると列名が嘘になる (``SPLIT_OFFSET_DIVISOR`` と同じ
判断)。定義を変えたいときは列名ごと変えること。
"""


@dataclass(frozen=True, slots=True)
class ProtocolCondition:
    """5-C の格子点 (``preprocess`` の3葉と1対1)。

    Attributes:
        normalize: ``preprocess.normalize``。
        input_window: ``preprocess.input_window``。
        score_smoothing: ``preprocess.score_smoothing``。
    """

    normalize: str
    input_window: int
    score_smoothing: int


@dataclass(frozen=True, slots=True)
class ProtocolSweepSummary:
    """5-C の要約 (``meta.json`` に載る量。**行から計算する**)。

    Attributes:
        n_conditions: 格子点の数。
        n_conditions_with_rank_change: 基準条件から順位が動いた格子点の数。
        n_rank_changed_rows: 順位が動いた (格子点, 系統) の数。
        n_discordant_pairs: 基準と順序が逆転した系統対の総数。
        n_discordant_pairs_distinguishable: そのうち**両方の系統が両条件で
            対照と区別できる**対の数 (D-78)。
        min_kendall_tau: 格子点ごとの Kendall tau-b の最小値。
    """

    n_conditions: int
    n_conditions_with_rank_change: int
    n_rank_changed_rows: int
    n_discordant_pairs: int
    n_discordant_pairs_distinguishable: int
    min_kendall_tau: float


@dataclass(frozen=True, slots=True)
class SizeSweepSummary:
    """5-D の要約 (``meta.json`` に載る量。**行から計算する**)。

    Attributes:
        method: 劣化点を測った系統 (N に依存するのは ESN 系統だけ)。
        reference_n_units: 基準 N (5-A と同じ条件)。
        reference_auprc: 基準 N での ``auprc_mean``。
        degradation_fraction: 劣化の定義 (``DEGRADATION_FRACTION``)。
        n_units_at_90pct: 基準の ``degradation_fraction`` 倍を初めて割る N。
            **格子内に見つからなければ格子の下端**を返す (``nan`` にしない)。
        saturated: 格子内に劣化点が無く、下端を報告しているか。
    """

    method: str
    reference_n_units: int
    reference_auprc: float
    degradation_fraction: float
    n_units_at_90pct: int
    saturated: bool


def protocol_conditions(config: Anomaly05Config) -> tuple[ProtocolCondition, ...]:
    """5-C の格子点を列挙する (3軸の全組合せ)。

    Raises:
        ValueError: いずれかの軸が空、または同じ値を2度並べている場合
            (同じ条件を2度回した行が成果物に出る)。
    """
    sweep = config.protocol_sweep
    axes: tuple[tuple[str, tuple[object, ...]], ...] = (
        ("normalize_grid", tuple(sweep.normalize_grid)),
        ("input_window_grid", tuple(sweep.input_window_grid)),
        ("score_smoothing_grid", tuple(sweep.score_smoothing_grid)),
    )
    for name, values in axes:
        if not values:
            raise ValueError(f"protocol_sweep.{name} が空です")
        if len(set(values)) != len(values):
            raise ValueError(f"protocol_sweep.{name} に重複があります: {values}")
    return tuple(
        ProtocolCondition(
            normalize=normalize,
            input_window=input_window,
            score_smoothing=score_smoothing,
        )
        for normalize, input_window, score_smoothing in product(
            sweep.normalize_grid, sweep.input_window_grid, sweep.score_smoothing_grid
        )
    )


def headline_condition(config: Anomaly05Config) -> ProtocolCondition:
    """``config.preprocess`` と一致する格子点 (5-A と同じ条件、D-79)。

    Raises:
        ValueError: その点が格子に含まれていない場合。含まれていないと
            「5-A と同じ条件の行」が 5-C の成果物に無いまま順位だけが出る。
    """
    preprocess = config.preprocess
    condition = ProtocolCondition(
        normalize=preprocess.normalize,
        input_window=preprocess.input_window,
        score_smoothing=preprocess.score_smoothing,
    )
    if condition not in protocol_conditions(config):
        raise ValueError(
            "protocol_sweep の格子が preprocess の既定条件を含んでいません "
            f"({condition})。5-A と同じ条件の行が 5-C に出ないため、"
            "両者が同じ前処理を通っていることを成果物で照合できません (D-79)"
        )
    return condition


def apply_protocol_condition(
    config: Anomaly05Config, condition: ProtocolCondition
) -> Anomaly05Config:
    """格子点を設定に適用する (**掃引が設定を作る唯一の口**)。

    基準の格子点では ``apply_protocol_condition(config, headline) == config``
    が成り立つ —— 5-C の基準行が 5-A と厳密一致することの構造的な根拠である。
    """
    preprocess = replace(
        config.preprocess,
        normalize=condition.normalize,
        input_window=condition.input_window,
        score_smoothing=condition.score_smoothing,
    )
    return replace(config, preprocess=preprocess)


def apply_size_condition(config: Anomaly05Config, n_units: int) -> Anomaly05Config:
    """N を設定に適用する (5-D 版の ``apply_protocol_condition``)。"""
    return replace(config, reservoir=replace(config.reservoir, n_units=n_units))


def _protocol_rows(
    config: Anomaly05Config,
    condition: ProtocolCondition,
    aggregates: Mapping[str, MethodAggregate],
    reference: Mapping[str, MethodAggregate],
    is_headline: bool,
) -> tuple[ProtocolSweepRow, ...]:
    ranks = method_ranks(aggregates)
    reference_ranks = method_ranks(reference)
    tau = kendall_tau(
        [reference[method].auprc_mean for method in ANOMALY_METHODS],
        [aggregates[method].auprc_mean for method in ANOMALY_METHODS],
    )
    discordant, marked = discordant_counts(reference, aggregates)
    return tuple(
        ProtocolSweepRow(
            dataset=config.dataset.source,
            normalize=condition.normalize,
            input_window=condition.input_window,
            score_smoothing=condition.score_smoothing,
            is_headline=is_headline,
            method=method,
            auprc_mean=aggregates[method].auprc_mean,
            auprc_sd=aggregates[method].auprc_sd,
            auprc_random_mean=aggregates[method].auprc_random_mean,
            n_pairs=aggregates[method].n_pairs,
            n_better_than_control=aggregates[method].n_better_than_control,
            control_sign_p=aggregates[method].control_sign_p,
            distinguishable=aggregates[method].distinguishable,
            rank=ranks[method],
            reference_rank=reference_ranks[method],
            rank_changed=ranks[method] != reference_ranks[method],
            reference_distinguishable=reference[method].distinguishable,
            kendall_tau=tau,
            n_discordant_pairs=discordant,
            n_discordant_pairs_distinguishable=marked,
        )
        for method in ANOMALY_METHODS
    )


def run_protocol_sweep(
    config: Anomaly05Config, sources: Mapping[str, SeriesSource]
) -> tuple[ProtocolSweepRow, ...]:
    """5-C を回す (格子点 x 6系統の行を返す)。

    格子点ごとに 5-A をそのまま回すので、前処理を作る経路は1本のままである
    (D-57)。基準は ``config.preprocess`` と一致する格子点 (D-79)。

    Args:
        config: 実験設定。
        sources: 系列名 -> 系列源 (``build_sources`` が作る)。

    Returns:
        格子の順 (``protocol_conditions``) x ``ANOMALY_METHODS`` の順の行。

    Raises:
        ValueError: 格子が空・重複・基準条件を含まない場合。
    """
    conditions = protocol_conditions(config)
    reference_condition = headline_condition(config)
    aggregated: list[tuple[ProtocolCondition, Mapping[str, MethodAggregate]]] = []
    for condition in conditions:
        started = time.perf_counter()
        results = run_anomaly_headline(
            apply_protocol_condition(config, condition), sources
        )
        aggregates = aggregate_methods(results.rows)
        aggregated.append((condition, aggregates))
        logger.info(
            "protocol normalize=%s input_window=%d score_smoothing=%d (%.2fs)",
            condition.normalize,
            condition.input_window,
            condition.score_smoothing,
            time.perf_counter() - started,
        )
    reference = next(
        item for condition, item in aggregated if condition == reference_condition
    )
    return tuple(
        row
        for condition, aggregates in aggregated
        for row in _protocol_rows(
            config,
            condition,
            aggregates,
            reference,
            is_headline=condition == reference_condition,
        )
    )


def _size_grid(config: Anomaly05Config) -> tuple[int, ...]:
    """5-D の N の格子 (基準 N を必ず含む、D-79)。

    Raises:
        ValueError: 格子が空・重複を含む・非正の N を含む、または基準 N
            (``reservoir.n_units``) を含まない場合。
    """
    grid = tuple(config.size_sweep.n_units_grid)
    if not grid:
        raise ValueError("size_sweep.n_units_grid が空です")
    if len(set(grid)) != len(grid):
        raise ValueError(f"size_sweep.n_units_grid に重複があります: {grid}")
    if any(n_units < 1 for n_units in grid):
        raise ValueError(f"n_units は 1 以上が必要です: {grid}")
    reference = config.reservoir.n_units
    if reference not in grid:
        raise ValueError(
            f"size_sweep.n_units_grid が基準 N (reservoir.n_units={reference}) を"
            f" 含んでいません: {grid}。5-D の基準行が 5-A と別条件になります (D-79)"
        )
    return grid


def _require_uniform_training(rows: Sequence[AnomalyRow], n_units: int) -> int:
    """全行の ``n_train`` が同一であることを要求する (D-78)。

    Raises:
        ValueError: 系列ごとに学習量が違う場合。5-D は「N を削ると性能が
            どこで落ちるか」を測るので、学習量の差が混ざると測定が壊れる。
    """
    amounts = sorted({row.n_train for row in rows})
    if len(amounts) != 1:
        raise ValueError(
            "5-D は全系列が同じ学習量で回っている必要があります "
            f"(n_units={n_units} で n_train={amounts})。系列ごとに train_end が"
            " 異なる源は 5-A / 5-C で使ってください (D-78)"
        )
    return amounts[0]


def _size_rows(
    config: Anomaly05Config,
    n_units: int,
    aggregates: Mapping[str, MethodAggregate],
    reference: Mapping[str, MethodAggregate],
    n_train: int,
) -> tuple[SizeSweepRow, ...]:
    reference_n_units = config.reservoir.n_units
    rows: list[SizeSweepRow] = []
    for method in ANOMALY_METHODS:
        item = aggregates[method]
        baseline = reference[method].auprc_mean
        ratio = item.auprc_mean / baseline if baseline > 0.0 else math.nan
        rows.append(
            SizeSweepRow(
                dataset=config.dataset.source,
                n_units=n_units,
                method=method,
                auprc_mean=item.auprc_mean,
                auprc_sd=item.auprc_sd,
                auprc_random_mean=item.auprc_random_mean,
                n_pairs=item.n_pairs,
                n_better_than_control=item.n_better_than_control,
                control_sign_p=item.control_sign_p,
                distinguishable=item.distinguishable,
                reference_n_units=reference_n_units,
                auprc_reference=baseline,
                auprc_ratio=ratio,
                below_reference_fraction=bool(ratio < DEGRADATION_FRACTION),
                n_train=n_train,
            )
        )
    return tuple(rows)


def run_size_sweep(
    config: Anomaly05Config, sources: Mapping[str, SeriesSource]
) -> tuple[SizeSweepRow, ...]:
    """5-D を回す (N x 6系統の行を返す)。

    N に依存しない系統も落とさない (D-61) —— 図の基準線になるうえ、
    「対照は N で動かない」ことが成果物で確かめられる。

    Args:
        config: 実験設定。
        sources: 系列名 -> 系列源。

    Returns:
        ``size_sweep.n_units_grid`` の順 x ``ANOMALY_METHODS`` の順の行。

    Raises:
        ValueError: 格子が不正 (``_size_grid``)、または系列ごとに学習量が
            違う場合 (``_require_uniform_training``、D-78)。
    """
    grid = _size_grid(config)
    aggregated: dict[int, Mapping[str, MethodAggregate]] = {}
    training: dict[int, int] = {}
    for n_units in grid:
        started = time.perf_counter()
        results = run_anomaly_headline(apply_size_condition(config, n_units), sources)
        training[n_units] = _require_uniform_training(results.rows, n_units)
        aggregated[n_units] = aggregate_methods(results.rows)
        logger.info(
            "size n_units=%d n_train=%d (%.2fs)",
            n_units,
            training[n_units],
            time.perf_counter() - started,
        )
    reference = aggregated[config.reservoir.n_units]
    return tuple(
        row
        for n_units in grid
        for row in _size_rows(
            config, n_units, aggregated[n_units], reference, training[n_units]
        )
    )


def summarize_protocol_sweep(
    rows: Sequence[ProtocolSweepRow],
) -> ProtocolSweepSummary:
    """5-C の行を ``meta.json`` 向けに畳む (仕様 §5 の条件4)。

    **行だけを読む** —— 図と meta.json が同じ数を見るようにするため、
    集計をもう一度掃引から作り直さない。

    Raises:
        ValueError: 行が空の場合。
    """
    if not rows:
        raise ValueError("5-C の行が空です")
    by_condition: dict[tuple[str, int, int], list[ProtocolSweepRow]] = {}
    for row in rows:
        key = (row.normalize, row.input_window, row.score_smoothing)
        by_condition.setdefault(key, []).append(row)
    taus = [
        group[0].kendall_tau
        for group in by_condition.values()
        if not math.isnan(group[0].kendall_tau)
    ]
    return ProtocolSweepSummary(
        n_conditions=len(by_condition),
        n_conditions_with_rank_change=sum(
            1
            for group in by_condition.values()
            if any(row.rank_changed for row in group)
        ),
        n_rank_changed_rows=sum(1 for row in rows if row.rank_changed),
        n_discordant_pairs=sum(
            group[0].n_discordant_pairs for group in by_condition.values()
        ),
        n_discordant_pairs_distinguishable=sum(
            group[0].n_discordant_pairs_distinguishable
            for group in by_condition.values()
        ),
        min_kendall_tau=min(taus) if taus else math.nan,
    )


def summarize_size_sweep(
    rows: Sequence[SizeSweepRow], method: str = ESN_RESIDUAL
) -> SizeSweepSummary:
    """5-D の行から劣化点を決める (仕様 §5 の条件5)。

    基準 N から**下へ**たどり、``DEGRADATION_FRACTION`` を割る最初の (= 最大の)
    N を返す。格子内に見つからなければ**格子の下端**を返し ``saturated=True``
    にする —— ``nan`` にすると「測れなかった」と「格子が足りなかった」が
    区別できないうえ、meta.json の数値としても扱いにくい。

    Args:
        rows: ``run_size_sweep`` の行。
        method: 劣化点を測る系統 (既定は N に依存する唯一の系統)。

    Raises:
        ValueError: 行が空、または指定した系統の行が無い場合。
    """
    selected = [row for row in rows if row.method == method]
    if not selected:
        raise ValueError(f"5-D に系統 {method} の行がありません")
    reference_n_units = selected[0].reference_n_units
    reference_rows = [row for row in selected if row.n_units == reference_n_units]
    if not reference_rows:
        raise ValueError(f"5-D に基準 N={reference_n_units} の行がありません")
    degraded = [
        row.n_units
        for row in selected
        if row.n_units < reference_n_units and row.below_reference_fraction
    ]
    saturated = not degraded
    return SizeSweepSummary(
        method=method,
        reference_n_units=reference_n_units,
        reference_auprc=reference_rows[0].auprc_mean,
        degradation_fraction=DEGRADATION_FRACTION,
        n_units_at_90pct=max(degraded)
        if degraded
        else min(row.n_units for row in selected),
        saturated=saturated,
    )


__all__ = [
    "CONTROL_SIGN_TEST_ALPHA",
    "DEGRADATION_FRACTION",
    "MethodAggregate",
    "ProtocolCondition",
    "ProtocolSweepSummary",
    "SizeSweepSummary",
    "aggregate_methods",
    "apply_protocol_condition",
    "apply_size_condition",
    "headline_condition",
    "kendall_tau",
    "protocol_conditions",
    "run_protocol_sweep",
    "run_size_sweep",
    "sign_test_p_value",
    "summarize_protocol_sweep",
    "summarize_size_sweep",
]

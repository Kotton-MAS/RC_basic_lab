"""実験 5-A / 5-B の配線 —— 1レプリケート = 1前処理 = 6系統の同一行 (D-05 / D-57).

ここが守るのは「比較が成立していること」であって、検知性能そのものではない:

1. **1レプリケートにつき ``AnomalyPreprocessor`` は1インスタンス**しか作らず、
   6系統すべてに同じものを値で配る (D-57)。手法ごとに再推定する経路が
   構造上書けない
2. **6系統が同一の行 index で評価される** (D-05)。基準行
   ``t0 = compute_t0(全系統の first_valid, washout)`` は1つだけで、そこから
   ``make_split`` が訓練 / 較正 / テストの連続3分割を切る
3. **一様乱数と入力ノルムの対照を成果物から外せない** (D-61)。系統の列挙は
   ``anomaly_score.ANOMALY_METHODS`` だけにあり、``Anomaly05Config`` には
   手法を選ぶ葉が1つも無い
4. **運用閾値はテストラベルを見ずに決まる** (D-56)。閾値を決める関数
   (``calibrate_threshold``) はラベルを引数に取らず、テスト側最適化の結果は
   ``f1_test_optimal`` という別列にしか出ない
5. **PA-F1 は一様乱数対照と対でしか列にならない** (D-55)。列名を作る関数
   ``pa_columns`` が2本を同時にしか返さない

系列源は ``Mapping[str, SeriesSource]`` の1辞書で渡す (D-71)。実験ループは
Protocol にしか触れないので、合成源・MGAB・UCR のどれであっても同じコードを
通る。**源の具象名で分岐してよいのは ``build_sources`` 1箇所だけ**である。
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields

import numpy as np

from rc_basics_lab.config import (
    Anomaly05Config,
    SplitConfig,
    anomaly_stream_seed,
)
from rc_basics_lab.datasets import mgab, ucr
from rc_basics_lab.experiment.anomaly_rows import (
    AnomalyRow,
    ThresholdSweepRow,
)
from rc_basics_lab.experiment.anomaly_score import (
    ANOMALY_METHODS,
    RANDOM_CONTROL,
    AnomalyScore,
    ScoreInputs,
    build_score,
    build_score_specs,
    score_first_valid,
    smooth_score,
    smoothing_shift,
)
from rc_basics_lab.experiment.anomaly_threshold import (
    alarms_at,
    best_test_f1,
    calibrate_threshold,
    evaluate_at_threshold,
    sweep_thresholds,
)
from rc_basics_lab.experiment.split import Split, compute_t0, make_split
from rc_basics_lab.metrics_detection import (
    BoolArray,
    PointAdjustReport,
    apply_ignore_mask,
    average_precision,
    point_adjust_report,
)
from rc_basics_lab.reservoir.esn import ESN
from rc_basics_lab.seeds import SeedStream, make_rng_for
from rc_basics_lab.tasks.anomaly import (
    AnomalyPreprocessor,
    AnomalySeries,
    SeriesSource,
    SyntheticSeriesSource,
)
from rc_basics_lab.types import FloatArray

logger = logging.getLogger(__name__)

SYNTHETIC_SOURCE = "synthetic"
MGAB_SOURCE = "mgab"
UCR_SOURCE = "ucr"

ANOMALY_SOURCES: tuple[str, ...] = (SYNTHETIC_SOURCE, MGAB_SOURCE, UCR_SOURCE)
"""``dataset.source`` に書ける値。

既定は合成 (D-60: pytest はネットワークに触れない)。
"""

SPLIT_OFFSET_DIVISOR = 100
"""分割オフセットの上限を系列長の何分の1にするか (``max_start_offset``)。

**設定の葉にしない**。``seeds.split`` が効く (= 分割境界が動く) ために正の値が
要るだけで、値そのものは結論を動かさない —— 葉にすると「YAML から設定できる
のに図も表も変わらない」フィールドが1つ増える (D-69 と同じ判断)。系列長に
比例させるのは、打ち切り長を変えたときにオフセットが系列に対して相対的に
同じ大きさであってほしいため。
"""


def build_sources(config: Anomaly05Config) -> Mapping[str, SeriesSource]:
    """``dataset.source`` から系列源の辞書を作る (**具象名で分岐する唯一の場所**)。

    実験ループはここが返した ``Mapping[str, SeriesSource]`` にしか触れない
    ので、源が合成でも MGAB でも UCR でも同じコードを通る (D-71)。分岐が
    2箇所目に生えると、``datasets -> tasks`` の一方向依存 (D-59) が実験層で
    崩れる。

    Args:
        config: 実験設定。``dataset.series`` の各要素が鍵になる。

    Raises:
        ValueError: ``dataset.source`` が未対応、または ``dataset.series`` が空。
    """
    names = config.dataset.series
    if not names:
        raise ValueError("dataset.series が空です")
    match config.dataset.source:
        case "synthetic":
            synthetic = SyntheticSeriesSource(cfg=config.synthetic)
            return {name: synthetic for name in names}
        case "mgab":
            return {name: mgab.MgabSeriesSource(series=name) for name in names}
        case "ucr":
            return {name: ucr.UcrSeriesSource(filename=name) for name in names}
        case _:
            raise ValueError(
                f"dataset.source は {ANOMALY_SOURCES} のいずれかです: "
                f"{config.dataset.source!r}"
            )


@dataclass(frozen=True, slots=True)
class AnomalyCondition:
    """1行を作る条件 = (系列, レプリケート)。

    Attributes:
        series: 系列名 (``dataset.series`` の要素、CSV の ``series`` 列)。
        series_index: 系列の並び順。
        replicate: レプリケート番号 (0 始まり)。
        n_replicates: 1系列あたりのレプリケート数 (``draw`` の畳み込みに使う)。
    """

    series: str
    series_index: int
    replicate: int
    n_replicates: int

    @property
    def draw(self) -> int:
        """``make_rng_for`` に渡す通し番号 ((系列, レプリケート) を1本に畳む)。

        ``seeds.py`` の ``make_rng_for`` は ``(stream, replicate)`` の2軸しか
        持たないので、系列軸をここで畳む。``n_replicates`` を変えると
        2本目以降の系列の乱数列も動くが、レプリケート数を変えた時点で
        平均±標準偏差の意味自体が変わるので、同一性を主張する相手ではない。
        """
        return self.series_index * self.n_replicates + self.replicate


@dataclass(frozen=True, slots=True)
class AnomalyPlan:
    """1レプリケートで**全系統が共有する前提** (D-05 / D-57)。

    Attributes:
        series: 打ち切り後の系列。
        preprocessor: **この1インスタンスだけ**が全系統・全区間に配られる。
        scores: 系統名 -> 平滑化済みスコア。鍵は ``ANOMALY_METHODS`` と一致。
        t0: 全系統共通の基準行。
        split: 訓練 / 較正 / テストの連続3分割 (``val`` が較正区間)。
    """

    series: AnomalySeries
    preprocessor: AnomalyPreprocessor
    scores: Mapping[str, AnomalyScore]
    t0: int
    split: Split


@dataclass(frozen=True, slots=True)
class AnomalyOutcome:
    """1レプリケートぶんの結果。

    Attributes:
        plan: 全系統が共有した前提 (受け入れ基準1 の照合先)。
        rows: 6系統ぶんの ``anomaly.csv`` の行。
        threshold_rows: 5-B の掃引の行。
    """

    plan: AnomalyPlan
    rows: tuple[AnomalyRow, ...]
    threshold_rows: tuple[ThresholdSweepRow, ...]


@dataclass(frozen=True, slots=True)
class AnomalyResults:
    """5-A / 5-B の全条件ぶんの結果。

    Attributes:
        rows: ``anomaly.csv`` の行。
        threshold_rows: ``anomaly_threshold.csv`` の行。
    """

    rows: tuple[AnomalyRow, ...]
    threshold_rows: tuple[ThresholdSweepRow, ...]


def truncate_series(series: AnomalySeries, max_length: int) -> AnomalySeries:
    """系列を先頭 ``max_length`` 点に切り詰める (予算調整の第一の軸)。

    Raises:
        ValueError: ``max_length`` が 1 未満、または切り詰めた結果として
            異常が1点も残らない場合 (AUPRC が定義できなくなる)。
    """
    if max_length < 1:
        raise ValueError(f"max_length は 1 以上が必要です: {max_length}")
    if series.n_steps <= max_length:
        return series
    labels: BoolArray = series.labels[:max_length]
    if not bool(np.any(labels)):
        raise ValueError(
            f"max_length={max_length} では異常が1点も残りません "
            f"(系列 {series.name}、元の長さ {series.n_steps})"
        )
    return AnomalySeries(
        values=series.values[:max_length],
        labels=labels,
        ignore=series.ignore[:max_length],
        train_end=min(series.train_end, max_length - 1),
        name=series.name,
        params=series.params,
    )


def split_config_for(config: Anomaly05Config, n_steps: int) -> SplitConfig:
    """3分割の設定を作る (``val`` = 較正区間)。

    ``experiment/split.py`` をそのまま使うため、較正区間を ``val`` に載せる。
    ``max_start_offset`` は系列長から導く (``SPLIT_OFFSET_DIVISOR``)。
    """
    dataset = config.dataset
    return SplitConfig(
        train_ratio=dataset.train_ratio,
        val_ratio=dataset.calibration_ratio,
        test_ratio=1.0 - dataset.train_ratio - dataset.calibration_ratio,
        washout=config.reservoir.washout,
        max_start_offset=max(1, n_steps // SPLIT_OFFSET_DIVISOR),
    )


def _validate_training_window(
    config: Anomaly05Config, series: AnomalySeries, split: Split
) -> None:
    """学習区間と前処理の推定区間が「正常が保証された前半」に収まるか (D-57)。

    ここを検査しないと、異常を含む区間から前処理係数や読み出し重みを推定する
    実験が**そのまま緑で通る** —— しかも結果は「よく当たっている」形で出る。

    Raises:
        ValueError: 学習区間が ``train_end`` を越える、または
            ``standardize_steps`` が学習区間の外にある場合。
    """
    steps = config.preprocess.standardize_steps
    if split.train.stop > series.train_end:
        raise ValueError(
            "学習区間が正常保証区間を越えています "
            f"(train.stop={split.train.stop} > train_end={series.train_end})。"
            " dataset.train_ratio を下げるか max_length を伸ばしてください"
        )
    if steps > series.train_end or steps > split.train.stop:
        raise ValueError(
            "preprocess.standardize_steps が学習区間の外にあります "
            f"(steps={steps}, train_end={series.train_end}, "
            f"train.stop={split.train.stop})"
        )


def preprocessor_id(preprocessor: AnomalyPreprocessor) -> str:
    """前処理係数の指紋 (CSV の ``preprocessor_id`` 列)。

    係数そのもの (center / scale) と来歴 (normalize / n_steps) を1本の
    ダイジェストに畳む。手法ごとに係数を作り直す実装に変異させると、
    同じ (系列, レプリケート) の行でこの値が割れる。
    """
    digest = hashlib.sha256()
    digest.update(np.asarray(preprocessor.center, dtype=np.float64).tobytes())
    digest.update(np.asarray(preprocessor.scale, dtype=np.float64).tobytes())
    digest.update(f"{preprocessor.normalize}:{preprocessor.n_steps}".encode())
    return digest.hexdigest()[:12]


def _visible(values: FloatArray, ignore: BoolArray, enabled: bool) -> FloatArray:
    """``is_ignored`` の点を落とす (**ラベルを持たない側**の経路)。

    較正区間には ``apply_ignore_mask`` を使わない —— あちらはラベルを引数に
    取るので、閾値決定の経路にラベルが1本でも通ることになる (D-56 が型で
    禁じたい形そのもの)。落とす規則は同じ ``is_ignored`` である。
    """
    if not enabled:
        return values
    kept: FloatArray = values[~ignore]
    return kept


def _slice(array: FloatArray, selection: range) -> FloatArray:
    block: FloatArray = array[selection.start : selection.stop]
    return block


def _mask_slice(array: BoolArray, selection: range) -> BoolArray:
    block: BoolArray = array[selection.start : selection.stop]
    return block


def _point_adjust_reports(
    config: Anomaly05Config,
    labels: BoolArray,
    predictions: BoolArray,
    control: FloatArray,
) -> tuple[PointAdjustReport, ...]:
    """PA%K の報告 (D-55)。**マスクをかける前**の系列で計算する。

    点を落とすと異常区間の連続性が壊れるため、``metrics_detection`` の
    docstring どおり PA 系はマスク前で測る。
    """
    if not config.evaluation.report_point_adjust:
        return ()
    return tuple(
        point_adjust_report(labels, predictions, control, k=k)
        for k in config.evaluation.pa_k_grid
    )


def _evaluate(
    config: Anomaly05Config,
    plan: AnomalyPlan,
    condition: AnomalyCondition,
    method: str,
) -> tuple[AnomalyRow, tuple[ThresholdSweepRow, ...]]:
    """1系統を評価して ``anomaly.csv`` の1行と 5-B の掃引行を返す。"""
    started = time.perf_counter()
    series, split = plan.series, plan.split
    ignore_transition = config.evaluation.ignore_transition
    score = plan.scores[method].values
    control = plan.scores[RANDOM_CONTROL].values

    test_labels = _mask_slice(series.labels, split.test)
    test_ignore = _mask_slice(series.ignore, split.test)
    test_scores = _slice(score, split.test)
    predictions_raw: BoolArray

    calibration = _visible(
        _slice(score, split.val),
        _mask_slice(series.ignore, split.val),
        ignore_transition,
    )
    threshold = calibrate_threshold(
        calibration, config.threshold.target_false_alarm_rate
    )
    predictions_raw = alarms_at(test_scores, threshold)

    if ignore_transition:
        masked = apply_ignore_mask(test_labels, test_scores, test_ignore)
        control_masked = apply_ignore_mask(
            test_labels, _slice(control, split.test), test_ignore
        )
        labels, scores, control_scores = (
            masked.labels,
            masked.scores,
            control_masked.scores,
        )
    else:
        labels, scores = test_labels, test_scores
        control_scores = _slice(control, split.test)

    operating = evaluate_at_threshold(labels, scores, threshold)
    n_train, n_calibration, n_test = split.sizes
    row = AnomalyRow(
        dataset=config.dataset.source,
        series=condition.series,
        method=method,
        replicate=condition.replicate,
        seed_reservoir=config.seeds.reservoir,
        seed_task=config.seeds.task,
        seed_split=config.seeds.split,
        seed_control=config.seeds.control,
        normalize=plan.preprocessor.normalize,
        preprocessor_id=preprocessor_id(plan.preprocessor),
        selected_alpha=plan.scores[method].selected_alpha,
        auprc=average_precision(labels, scores),
        auprc_random=average_precision(labels, control_scores),
        anomaly_rate=float(np.count_nonzero(labels)) / float(labels.size),
        threshold=operating.threshold,
        f1_calibrated=operating.f1,
        precision_calibrated=operating.precision,
        recall_calibrated=operating.recall,
        far_test=operating.false_alarm_rate,
        n_evaluated=int(labels.size),
        n_train=n_train,
        n_calibration=n_calibration,
        n_test=n_test,
        t0=plan.t0,
        split_offset=split.offset,
        wall_time_s=0.0,
        f1_test_optimal=(
            best_test_f1(labels, scores)
            if config.threshold.report_test_optimal
            else math.nan
        ),
        point_adjust=_point_adjust_reports(
            config, test_labels, predictions_raw, _slice(control, split.test)
        ),
    )
    sweep = tuple(
        ThresholdSweepRow(
            dataset=row.dataset,
            series=row.series,
            method=row.method,
            replicate=row.replicate,
            target_false_alarm_rate=float(index + 1)
            / float(config.threshold.sweep_points),
            threshold=point.threshold,
            precision=point.precision,
            recall=point.recall,
            f1=point.f1,
            false_alarm_rate=point.false_alarm_rate,
            n_alarms=point.n_alarms,
            calibrated_threshold=threshold,
        )
        for index, point in enumerate(
            sweep_thresholds(labels, scores, config.threshold.sweep_points)
        )
    )
    elapsed = time.perf_counter() - started
    logger.info(
        "series=%s method=%s replicate=%d auprc=%.4f (random %.4f) "
        "f1=%.3f far=%.4f (%.2fs)",
        row.series,
        row.method,
        row.replicate,
        row.auprc,
        row.auprc_random,
        row.f1_calibrated,
        row.far_test,
        elapsed,
    )
    return (
        AnomalyRow(
            **{
                item.name: getattr(row, item.name)
                for item in fields(AnomalyRow)
                if item.name != "wall_time_s"
            },
            wall_time_s=elapsed,
        ),
        sweep,
    )


def plan_replicate(
    config: Anomaly05Config, source: SeriesSource, condition: AnomalyCondition
) -> AnomalyPlan:
    """1レプリケートぶんの系列・前処理・分割・6系統のスコアを作る。

    **前処理は1インスタンスだけ**作り、6系統すべてに値で配る (D-57)。
    ``t0`` はスコアを作る**前**に全系統の ``first_valid`` から決まる (D-05)
    —— 学習区間の位置が ``t0`` で決まり、学習はスコア構成の中で起きるため。
    """
    seeds = config.seeds
    task_rng = make_rng_for(
        anomaly_stream_seed(seeds, SeedStream.TASK), SeedStream.TASK, condition.draw
    )
    series = truncate_series(source(task_rng), config.dataset.max_length)

    specs = build_score_specs(config)
    shift = smoothing_shift(config.preprocess.score_smoothing)
    t0 = compute_t0(
        (score_first_valid(spec) + shift for spec in specs.values()),
        config.reservoir.washout,
    )
    split = make_split(
        split_config_for(config, series.n_steps),
        series.n_steps,
        t0,
        make_rng_for(
            anomaly_stream_seed(seeds, SeedStream.SPLIT),
            SeedStream.SPLIT,
            condition.draw,
        ),
    )
    _validate_training_window(config, series, split)

    preprocessor = AnomalyPreprocessor.from_training_prefix(
        series.values,
        config.preprocess.standardize_steps,
        config.preprocess.normalize,
    )
    scaled = preprocessor.apply(series.values)
    reservoir_rng = make_rng_for(
        anomaly_stream_seed(seeds, SeedStream.RESERVOIR),
        SeedStream.RESERVOIR,
        condition.draw,
    )
    states = ESN(config.reservoir.to_esn(), reservoir_rng, n_inputs=1).run(
        scaled, rng=reservoir_rng
    )
    control_rng = make_rng_for(
        anomaly_stream_seed(seeds, SeedStream.PROBE), SeedStream.PROBE, condition.draw
    )
    inputs = ScoreInputs(
        values=scaled,
        states=states,
        control_scores=control_rng.random(series.n_steps),
        train=split.train,
        calibration=split.val,
        alphas=tuple(config.ridge.alpha_grid),
    )
    scores = {
        name: smooth_score(build_score(spec, inputs), config.preprocess.score_smoothing)
        for name, spec in specs.items()
    }
    return AnomalyPlan(
        series=series,
        preprocessor=preprocessor,
        scores=scores,
        t0=t0,
        split=split,
    )


def run_anomaly_replicate(
    config: Anomaly05Config, source: SeriesSource, condition: AnomalyCondition
) -> AnomalyOutcome:
    """1 (系列, レプリケート) を回して6系統ぶんの行を返す。

    行の並びは ``ANOMALY_METHODS`` の順で、**設定に関わらず6行**出る (D-61)。
    """
    plan = plan_replicate(config, source, condition)
    evaluated = [
        _evaluate(config, plan, condition, method) for method in ANOMALY_METHODS
    ]
    return AnomalyOutcome(
        plan=plan,
        rows=tuple(row for row, _ in evaluated),
        threshold_rows=tuple(point for _, sweep in evaluated for point in sweep),
    )


def run_anomaly_headline(
    config: Anomaly05Config, sources: Mapping[str, SeriesSource]
) -> AnomalyResults:
    """5-A / 5-B を回す (系列 x レプリケート x 6系統)。

    源が使えるか (``is_available``) だけで絞る —— 具象名を見ない (D-71)。
    キャッシュの無い実データ源は静かに落ちるので、1本も残らなければ例外に
    する (「0行の CSV が出て気づかない」を避ける)。

    Args:
        config: 実験設定。
        sources: 系列名 -> 系列源 (``build_sources`` が作る)。

    Raises:
        ValueError: ``n_replicates`` が 1 未満、または使える源が1本も無い場合。
    """
    if config.reservoir.n_replicates < 1:
        raise ValueError(
            f"n_replicates は 1 以上が必要です: {config.reservoir.n_replicates}"
        )
    available = [
        (name, source) for name, source in sources.items() if source.is_available()
    ]
    if not available:
        raise ValueError(
            "使える系列源が1本もありません "
            f"(要求={sorted(sources)}、キャッシュを取得しましたか)"
        )
    rows: list[AnomalyRow] = []
    threshold_rows: list[ThresholdSweepRow] = []
    for series_index, (name, source) in enumerate(available):
        for replicate in range(config.reservoir.n_replicates):
            outcome = run_anomaly_replicate(
                config,
                source,
                AnomalyCondition(
                    series=name,
                    series_index=series_index,
                    replicate=replicate,
                    n_replicates=config.reservoir.n_replicates,
                ),
            )
            rows.extend(outcome.rows)
            threshold_rows.extend(outcome.threshold_rows)
    return AnomalyResults(rows=tuple(rows), threshold_rows=tuple(threshold_rows))


__all__ = [
    "ANOMALY_SOURCES",
    "MGAB_SOURCE",
    "SPLIT_OFFSET_DIVISOR",
    "SYNTHETIC_SOURCE",
    "UCR_SOURCE",
    "AnomalyCondition",
    "AnomalyOutcome",
    "AnomalyPlan",
    "AnomalyResults",
    "build_sources",
    "plan_replicate",
    "preprocessor_id",
    "run_anomaly_headline",
    "run_anomaly_replicate",
    "split_config_for",
    "truncate_series",
]

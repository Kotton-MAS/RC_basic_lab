"""実験 5-A / 5-B の検査 (D-05 / D-54 / D-55 / D-56 / D-57 / D-61 / D-63).

仕様 (``docs/plans/rc-basics-05.md`` §4 T3) の受け入れ基準7項目に1テストずつ
対応する。ここが測るのは検知性能ではなく**比較が成立していること**である:

1. ``test_all_methods_share_identical_rows_and_preprocessor`` (D-05 / D-57)
2. ``test_random_and_input_norm_controls_are_always_present`` (D-61)
3. ``test_operating_threshold_is_calibrated_without_test_labels`` (D-56)
4. ``test_point_adjust_is_never_reported_without_the_random_control`` (D-55)
5. ``test_test_optimal_f1_is_a_separate_column_never_below_the_calibrated_one``
6. ``test_the_headline_auprc_never_passes_through_point_adjust`` (D-54 / D-55)
7. ``test_anomaly_modules_stay_under_the_line_budget`` (D-63)

**ネットワークに触れない** (D-60)。既定のデータ源は合成で、実データ源は
``build_sources`` が型を組み立てるところまでしか触らない (``is_available`` は
キャッシュを見るだけでネットワークを開かない)。

``src/`` を書き換える変異試験はしない —— 差し替えは ``monkeypatch`` と、
系列を作り替えたスタブ源 (``_FixedSeriesSource``) で行う。
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from test_config_wiring_anomaly import ANOMALY_CASES, REDUCED
from wiring import apply_case

from rc_basics_lab.config import (
    Anomaly05Config,
    AnomalyDatasetConfig,
    AnomalyEvaluationConfig,
    AnomalyPreprocessConfig,
    AnomalyReservoirConfig,
    AnomalyRidgeConfig,
    AnomalyThresholdConfig,
    SyntheticAnomalyConfig,
)
from rc_basics_lab.experiment.anomaly import (
    AnomalyCondition,
    AnomalyPlan,
    plan_anomaly_replicate,
    preprocessor_id,
    run_anomaly_headline,
    run_anomaly_replicate,
    truncate_series,
)
from rc_basics_lab.experiment.anomaly_rows import (
    ANOMALY_SCALAR_COLUMNS,
    PA_F1_PREFIX,
    PA_F1_RANDOM_PREFIX,
    AnomalyRow,
    anomaly_csv_columns,
    anomaly_row_as_dict,
    pa_columns,
)
from rc_basics_lab.experiment.anomaly_score import (
    ANOMALY_METHODS,
    CONTROL_METHODS,
    ESN_RESIDUAL,
    INPUT_NORM_CONTROL,
    MOVING_STATISTICS,
    RANDOM_CONTROL,
    MovingStatisticsSpec,
    ScoreInputs,
    build_score,
    build_score_specs,
    score_first_valid,
    smooth_score,
)
from rc_basics_lab.experiment.anomaly_sources import ANOMALY_SOURCES, build_sources
from rc_basics_lab.experiment.anomaly_threshold import (
    alarms_at,
    best_test_f1,
    calibrate_threshold,
    evaluate_at_threshold,
    sweep_thresholds,
)
from rc_basics_lab.metrics_detection import point_scores
from rc_basics_lab.tasks.anomaly import AnomalyPreprocessor, AnomalySeries
from rc_basics_lab.types import BoolArray, FloatArray

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "src" / "rc_basics_lab" / "experiment"

MAX_LINES_PER_MODULE = 600
"""1ファイルの上限 (D-63)。**上限そのものを緩めない**。

実測の反面教師: ``experiment/freerun.py`` 1620 行 / ``capacity.py`` 1204 /
``attractor.py`` 715 / ``stability.py`` 631。行数は乱暴な代理指標だが、
決定論的に落ちるという一点で散文より強い。
"""

CONDITION = AnomalyCondition(series="s1", series_index=0, replicate=0, n_replicates=1)
"""``REDUCED`` の1条件 (系列1本 x レプリケート0)。"""


@dataclass(frozen=True, slots=True)
class _FixedSeriesSource:
    """決め打ちの系列を返すスタブ源 (``SeriesSource`` を満たす、D-71)。

    ラベルを差し替えた系列を実験へ流し込むために使う。``src/`` を書き換える
    代わりにここで系列を作り替えるのが、この層の変異試験のやり方である。
    """

    series: AnomalySeries

    def is_available(self) -> bool:
        """常に使える (合成源と同じ)。"""
        return True

    def __call__(self, rng: np.random.Generator) -> AnomalySeries:
        """``rng`` を使わずに決め打ちの系列を返す。"""
        del rng
        return self.series


def _plan() -> AnomalyPlan:
    return plan_anomaly_replicate(REDUCED, build_sources(REDUCED)["s1"], CONDITION)


def _rows_from(series: AnomalySeries) -> tuple[AnomalyRow, ...]:
    return run_anomaly_replicate(
        REDUCED, _FixedSeriesSource(series=series), CONDITION
    ).rows


def _with_labels(series: AnomalySeries, labels: BoolArray) -> AnomalySeries:
    return dataclasses.replace(series, labels=labels)


def _flipped(series: AnomalySeries, start: int, stop: int) -> AnomalySeries:
    labels: BoolArray = np.array(series.labels, dtype=np.bool_, copy=True)
    labels[start:stop] = ~labels[start:stop]
    return _with_labels(series, labels)


def _row_payload(row: AnomalyRow) -> dict[str, object]:
    payload = anomaly_row_as_dict(row)
    payload.pop("wall_time_s")
    return payload


# --- 受け入れ基準1: 同一前処理・同一行 --------------------------------------


def test_all_methods_share_identical_rows_and_preprocessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """6系統が同一の ``AnomalyPreprocessor`` と同一の行 index で評価される。

    3方向から測る (D-05 / D-57):

    (a) ``AnomalyPreprocessor.from_training_prefix`` の呼び出しが1レプリケート
        につき**ちょうど1回**である —— 手法ごとに係数を作り直す実装なら 6 回に
        なる。値の一致だけを見る検査では「同じ区間から作り直した」経路を
        検出できない
    (b) 6行の ``preprocessor_id`` が ``plan.preprocessor`` の指紋と一致する
    (c) 6行の ``t0`` / 分割サイズ / 評価点数が完全に一致し、全系統の
        ``first_valid`` が ``t0`` 以下である (= どの系統も NaN 行を評価しない)
    """
    calls: list[tuple[int, str]] = []
    original = AnomalyPreprocessor.from_training_prefix

    def counting(
        series: FloatArray, n_steps: int, normalize: str = "zscore"
    ) -> AnomalyPreprocessor:
        calls.append((n_steps, normalize))
        return original(series, n_steps, normalize)

    monkeypatch.setattr(
        AnomalyPreprocessor, "from_training_prefix", staticmethod(counting)
    )
    outcome = run_anomaly_replicate(REDUCED, build_sources(REDUCED)["s1"], CONDITION)

    assert len(calls) == 1, (
        f"1レプリケートで前処理係数が {len(calls)} 回作られています "
        "(D-57: 作れる場所は from_training_prefix 1本で、値を全系統へ配る)"
    )
    assert calls[0] == (
        REDUCED.preprocess.standardize_steps,
        REDUCED.preprocess.normalize,
    )

    expected_id = preprocessor_id(outcome.plan.preprocessor)
    assert [row.method for row in outcome.rows] == list(ANOMALY_METHODS)
    assert {row.preprocessor_id for row in outcome.rows} == {expected_id}
    assert len({row.normalize for row in outcome.rows}) == 1

    shared = {
        (row.t0, row.split_offset, row.n_train, row.n_calibration, row.n_test)
        for row in outcome.rows
    }
    assert len(shared) == 1, f"系統ごとに行が割れています (D-05): {shared}"
    assert len({row.n_evaluated for row in outcome.rows}) == 1
    assert all(
        score.first_valid <= outcome.plan.t0 for score in outcome.plan.scores.values()
    ), "t0 より後ろに first_valid を持つ系統があります (NaN 行を評価します)"


def test_the_preprocessor_is_fitted_strictly_inside_the_training_window() -> None:
    """前処理の推定区間が学習区間の内側に収まる (D-57 の異常検知版)。

    テスト区間から推定した尺度にはその区間の**異常が入っている**ため、
    異常が「正常なばらつき」として吸収される。
    """
    plan = _plan()
    assert plan.preprocessor.n_steps <= plan.split.train.stop
    assert plan.preprocessor.n_steps <= plan.series.train_end
    assert plan.split.train.stop <= plan.series.train_end


def test_the_training_window_must_not_reach_into_the_anomalous_part() -> None:
    """学習区間が正常保証区間を越える設定は例外になる (静かに通さない)。"""
    greedy = dataclasses.replace(
        REDUCED, dataset=dataclasses.replace(REDUCED.dataset, train_ratio=0.6)
    )
    with pytest.raises(ValueError, match="正常保証区間"):
        plan_anomaly_replicate(greedy, build_sources(greedy)["s1"], CONDITION)


# --- 受け入れ基準2: 対照を外せない ------------------------------------------


@pytest.mark.parametrize(
    "wiring_case", ANOMALY_CASES, ids=[item.field for item in ANOMALY_CASES]
)
def test_random_and_input_norm_controls_are_always_present(
    wiring_case: object,
) -> None:
    """どの設定でも一様乱数と入力ノルムの対照が残る (D-61)。

    ``Anomaly05Config`` の**全葉**について値を差し替え、系統の集合が
    ``ANOMALY_METHODS`` から1つも減らないことを測る。系統を列挙するのは
    ``build_score_specs`` だけなので、「設定で対照を外す」経路を作るには
    そこに分岐を書くしかなく、書けばここが赤くなる。

    ``synthetic.*`` は合成源の設定で系統の集合に触れないため、
    ``ANOMALY_CASES`` (委譲済み) の一覧をそのまま使う。
    """
    assert set(CONTROL_METHODS) <= set(ANOMALY_METHODS)
    changed = apply_case(REDUCED, wiring_case)  # type: ignore[arg-type]
    specs = build_score_specs(changed)
    assert tuple(specs) == ANOMALY_METHODS, (
        f"{getattr(wiring_case, 'field', '?')} が系統の集合を変えています (D-61)"
    )
    for control in CONTROL_METHODS:
        assert control in specs


def test_the_controls_reach_the_rows_of_every_condition() -> None:
    """成果物の行にも対照が必ず現れる (D-61 の出力側)。"""
    results = run_anomaly_headline(REDUCED, build_sources(REDUCED))
    assert {row.method for row in results.rows} == set(ANOMALY_METHODS)
    for control in CONTROL_METHODS:
        assert any(row.method == control for row in results.rows)
    assert len(results.rows) == len(ANOMALY_METHODS) * len(REDUCED.dataset.series)


def test_the_config_has_no_leaf_that_selects_methods() -> None:
    """設定に「手法を選ぶ葉」が無い (対照を外す入口が存在しない、D-61)。

    ``build_score_specs`` が読むのは ``preprocess.input_window`` だけである。
    手法名を値に持つ葉を足すと、そこが対照を外す入口になる。
    """
    for row in dataclasses.asdict(REDUCED).values():
        assert row not in ANOMALY_METHODS
    parameters = inspect.signature(build_score_specs).parameters
    assert list(parameters) == ["config"]


# --- 受け入れ基準3: 閾値は較正区間だけで決まる ------------------------------


def test_operating_threshold_is_calibrated_without_test_labels() -> None:
    """運用閾値がテスト区間のラベルを1ビットも見ていない (D-56)。

    2方向から測る:

    (a) **較正区間のラベルを全反転**させても行が1バイトも変わらない ——
        較正区間でラベルを使って閾値を選ぶ実装 (F1 最大化など) に変異させると
        閾値が動くので落ちる
    (b) **テスト区間のラベルを全反転**させると ``f1_calibrated`` は変わるが
        ``threshold`` は変わらない —— テスト側で閾値を選ぶ実装なら閾値も動く
    """
    plan = _plan()
    series = plan.series
    baseline = _rows_from(series)

    calibration_start = max(plan.split.val.start, series.train_end)
    calibration_flipped = _rows_from(
        _flipped(series, calibration_start, plan.split.val.stop)
    )
    assert calibration_start < plan.split.val.stop, "較正区間の反転範囲が空です"
    for row, reference in zip(calibration_flipped, baseline, strict=True):
        assert row.threshold == reference.threshold, (
            "較正区間のラベルを反転すると運用閾値が動きます (D-56)"
        )
        assert _row_payload(row) == _row_payload(reference), (
            "較正区間のラベルが結果行に漏れています (D-56)"
        )

    test_flipped = _rows_from(
        _flipped(series, plan.split.test.start, plan.split.test.stop)
    )
    for row, reference in zip(test_flipped, baseline, strict=True):
        assert row.threshold == reference.threshold, (
            "テスト区間のラベルを反転すると運用閾値が動きます (D-56)"
        )
    assert [row.f1_calibrated for row in test_flipped] != [
        row.f1_calibrated for row in baseline
    ], "テスト区間のラベルを反転しても F1 が変わりません (評価が効いていません)"


def test_the_threshold_function_cannot_take_labels_at_all() -> None:
    """``calibrate_threshold`` の署名がラベルを受け付けない (D-56 の型側)。

    D-56 の本体は「引数から外せば混入が型検査で書けなくなる」ことなので、
    署名そのものを固定する。
    """
    parameters = inspect.signature(calibrate_threshold).parameters
    assert list(parameters) == ["calibration_scores", "target_false_alarm_rate"]
    assert not [name for name in parameters if "label" in name]


def test_the_calibrated_threshold_hits_the_requested_alarm_budget() -> None:
    """較正区間での警報数が目標誤報率どおりになる (順位で切る、D-56)。"""
    scores: FloatArray = np.linspace(0.0, 1.0, 1000)
    threshold = calibrate_threshold(scores, 0.05)
    assert int(np.count_nonzero(scores >= threshold)) == 50


# --- 受け入れ基準4: PA は乱数対照と対でしか出ない ---------------------------


@pytest.mark.parametrize("pa_k_grid", [(0.0,), (0.0, 20.0), (0.0, 50.0, 100.0)])
def test_point_adjust_is_never_reported_without_the_random_control(
    pa_k_grid: tuple[float, ...],
) -> None:
    """PA-F1 の列が ``pa_f1_random`` 列と**同時にしか**現れない (D-55)。

    列名の集合を数える。``pa_f1`` 側と ``pa_f1_random`` 側が全単射で対応し、
    報告しない設定では**どちらも1本も出ない**。片方だけ出す列を作るには
    ``pa_columns`` (2本を同時にしか返さない) を書き換えるしかない。
    """
    config = dataclasses.replace(
        REDUCED,
        evaluation=dataclasses.replace(
            REDUCED.evaluation, report_point_adjust=True, pa_k_grid=pa_k_grid
        ),
    )
    columns = anomaly_csv_columns(config)
    scored = {name for name in columns if name.startswith(PA_F1_PREFIX)}
    controls = {name for name in columns if name.startswith(PA_F1_RANDOM_PREFIX)}
    assert scored and controls
    assert {name[len(PA_F1_PREFIX) :] for name in scored} == {
        name[len(PA_F1_RANDOM_PREFIX) :] for name in controls
    }
    assert not [
        name for name in columns if "pa_f1" in name and name not in scored | controls
    ], "対にならない PA 列があります (D-55)"

    rows, _ = _run(config)
    for row in rows:
        payload = anomaly_row_as_dict(row)
        assert set(payload) == set(columns)
        assert [report.k for report in row.point_adjust] == list(pa_k_grid)
        for report in row.point_adjust:
            scored_column, control_column = pa_columns(report.k)
            assert payload[scored_column] == report.pa_f1
            assert payload[control_column] == report.pa_f1_random


def test_point_adjust_columns_disappear_together() -> None:
    """``report_point_adjust=False`` で PA 列が**両方とも**消える (D-55)。"""
    config = dataclasses.replace(
        REDUCED,
        evaluation=dataclasses.replace(REDUCED.evaluation, report_point_adjust=False),
    )
    columns = anomaly_csv_columns(config)
    assert not [name for name in columns if "pa_f1" in name]
    rows, _ = _run(config)
    for row in rows:
        assert row.point_adjust == ()
        assert not [name for name in anomaly_row_as_dict(row) if "pa_f1" in name]


def test_the_random_control_row_reports_itself_as_its_own_control() -> None:
    """乱数対照の行では ``pa_f1`` と ``pa_f1_random`` が一致する (自己整合)。

    対照に渡している乱数列が「対照系統のスコアそのもの」であることの実測。
    別々の乱数を引いていると、ここが一致しない。
    """
    rows, _ = _run(REDUCED)
    control_row = next(row for row in rows if row.method == RANDOM_CONTROL)
    for report in control_row.point_adjust:
        assert report.pa_f1 == pytest.approx(report.pa_f1_random)
    assert control_row.auprc == pytest.approx(control_row.auprc_random)


# --- 受け入れ基準5: f1_test_optimal は別列で、較正値を下回らない -------------


def test_test_optimal_f1_is_a_separate_column_never_below_the_calibrated_one() -> None:
    """``f1_test_optimal`` が別列で、全行で ``f1_calibrated`` 以上 (D-56)。

    ``best_test_f1`` は PR 曲線の**全点**の最大なので、この不等号は定義から
    成り立つ (格子で近似すると格子の粗さ次第で破れる行が出る)。
    """
    rows, _ = _run(REDUCED)
    columns = anomaly_csv_columns(REDUCED)
    assert "f1_test_optimal" in columns
    assert "f1_calibrated" in ANOMALY_SCALAR_COLUMNS
    assert columns.index("f1_calibrated") != columns.index("f1_test_optimal")
    for row in rows:
        assert row.f1_test_optimal >= row.f1_calibrated - 1e-12, (
            f"{row.method}: f1_test_optimal({row.f1_test_optimal}) < "
            f"f1_calibrated({row.f1_calibrated})"
        )


def test_the_test_optimal_column_can_be_switched_off() -> None:
    """``report_test_optimal=False`` で列が消える (``f1_calibrated`` は残る)。"""
    config = dataclasses.replace(
        REDUCED,
        threshold=dataclasses.replace(REDUCED.threshold, report_test_optimal=False),
    )
    columns = anomaly_csv_columns(config)
    assert "f1_test_optimal" not in columns
    assert "f1_calibrated" in columns
    rows, _ = _run(config)
    for row in rows:
        assert math.isnan(row.f1_test_optimal)
        assert "f1_test_optimal" not in anomaly_row_as_dict(row)


def test_best_test_f1_is_at_least_the_f1_at_any_threshold() -> None:
    """``best_test_f1`` が総当たりの最大と一致する (参考値の定義の実測)。"""
    rng = np.random.default_rng(20250820)
    labels: BoolArray = rng.random(400) < 0.15
    scores: FloatArray = rng.random(400)
    brute = max(
        point_scores(labels, alarms_at(scores, float(threshold))).f1
        for threshold in np.unique(scores)
    )
    assert best_test_f1(labels, scores) == pytest.approx(brute)


# --- 受け入れ基準6: 主指標は point-adjust を通らない ------------------------


PA_CONTRAST = Anomaly05Config(
    dataset=AnomalyDatasetConfig(
        series=("s1",), max_length=6000, train_ratio=0.25, calibration_ratio=0.15
    ),
    synthetic=SyntheticAnomalyConfig(
        length=6000, n_anomalies=3, segment_length=400, ignore_margin=20
    ),
    preprocess=AnomalyPreprocessConfig(
        standardize_steps=400, input_window=8, score_smoothing=1
    ),
    reservoir=AnomalyReservoirConfig(n_units=20, washout=20, n_replicates=1),
    ridge=AnomalyRidgeConfig(alpha_grid=(1e-2,)),
    threshold=AnomalyThresholdConfig(target_false_alarm_rate=0.02, sweep_points=3),
    evaluation=AnomalyEvaluationConfig(pa_k_grid=(0.0,)),
)
"""PA と AUPRC の差が最大になる形の設定 (異常区間が長く、警報予算が広い)。

区間長 400 点 x 3 本 / 警報率 2% —— T1 の
``test_point_adjust_random_control_is_weaker_on_short_segments`` が実測した
とおり **PA の乱数対照の強さは異常区間の長さで決まる**ので、Kim et al. の
SWaT (平均区間長 約 1,540 点) と同じ「長い区間」の形にしないと
``pa_f1 > 0.9`` は原理的に出ない。既定設定 (区間長 200 / 警報率 1%) での
実測は ``pa_f1_random`` = 0.69〜0.83 で、これはこれで
``test_the_default_configuration_keeps_the_two_metrics_apart`` が別途固定する。
"""


def test_the_headline_auprc_never_passes_through_point_adjust() -> None:
    """乱数スコアで ``auprc ≈ 異常率`` かつ ``pa_f1 > 0.9`` (D-54 / D-55)。

    主指標 ``auprc`` が point-adjust を一切通していないことの実測である。
    もし通していれば、一様乱数の ``auprc`` も 0.9 付近まで跳ね上がる ——
    Kim et al. AAAI 2022 が「一様乱数の F1_PA が5データセット中4つで SOTA を
    上回る」と示したのがまさにこの現象で、PA-F1 は**高い値が出る形でしか
    壊れない**ため、単独で報告された瞬間に読者も実装者も検出できない。
    """
    rows, _ = _run(PA_CONTRAST)
    control = next(row for row in rows if row.method == RANDOM_CONTROL)
    assert control.auprc == pytest.approx(control.anomaly_rate, abs=0.03), (
        f"一様乱数の AUPRC {control.auprc:.4f} が異常率 "
        f"{control.anomaly_rate:.4f} から離れています"
    )
    (report,) = control.point_adjust
    assert report.k == 0.0
    assert report.pa_f1 > 0.9, (
        f"一様乱数の PA-F1 が {report.pa_f1:.4f} しかありません "
        "(この条件で PA の甘さが出ないと基準6の対照になりません)"
    )
    assert control.auprc < 0.5 * report.pa_f1, (
        "主指標が point-adjust の値に近づいています (auprc が PA を通った疑い)"
    )


def test_the_default_configuration_keeps_the_two_metrics_apart() -> None:
    """既定設定でも「AUPRC は分離し、PA-F1 は分離しない」(記事の主張の実測)。

    ESN と一様乱数対照を既定の縮小設定で比べ、**AUPRC の比が PA-F1 の比より
    大きい**ことを固定する。PA-F1 で並べると差が潰れることが D-55 の根拠
    そのものなので、数値として1本残しておく。
    """
    rows, _ = _run(REDUCED)
    esn = next(row for row in rows if row.method == ESN_RESIDUAL)
    control = next(row for row in rows if row.method == RANDOM_CONTROL)
    auprc_ratio = esn.auprc / control.auprc
    pa_ratio = esn.point_adjust[0].pa_f1 / max(control.point_adjust[0].pa_f1, 1e-12)
    assert auprc_ratio > pa_ratio, (
        f"AUPRC 比 {auprc_ratio:.3f} が PA-F1 比 {pa_ratio:.3f} を上回りません"
    )


# --- 受け入れ基準7: 行数の上限 ----------------------------------------------


def _anomaly_modules() -> list[Path]:
    return sorted(EXPERIMENT_DIR.glob("anomaly*.py"))


def test_anomaly_modules_stay_under_the_line_budget() -> None:
    """05 の実験層のどのモジュールも 600 行以下 (D-63)。

    「後で割る」は必ず割られないので、着手時点から複数モジュールに分けた
    うえで機械が上限を要求する。**上限そのものを緩めない** —— 超えたら
    もう1段割る。
    """
    modules = _anomaly_modules()
    assert len(modules) >= 3, (
        f"05 の実験層が分かれていません (検査が空振りします): {modules}"
    )
    too_long = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in modules
        if len(path.read_text(encoding="utf-8").splitlines()) > MAX_LINES_PER_MODULE
    }
    assert not too_long, (
        f"600 行を超えたモジュールがあります (D-63。割ってください): {too_long}"
    )


# --- スコア構成器の中身 -----------------------------------------------------


def _run(config: Anomaly05Config) -> tuple[tuple[AnomalyRow, ...], Sequence[object]]:
    results = run_anomaly_headline(config, build_sources(config))
    return results.rows, results.threshold_rows


def _score_inputs(plan: AnomalyPlan) -> ScoreInputs:
    rng = np.random.default_rng(7)
    return ScoreInputs(
        values=plan.preprocessor.apply(plan.series.values),
        states=np.zeros((plan.series.n_steps, 3), dtype=np.float64),
        control_scores=rng.random(plan.series.n_steps),
        train=plan.split.train,
        calibration=plan.split.val,
        alphas=REDUCED.ridge.alpha_grid,
    )


@pytest.mark.parametrize("method", ANOMALY_METHODS)
def test_score_first_valid_predicts_the_built_score(method: str) -> None:
    """``score_first_valid`` の予測が実際の ``first_valid`` と一致する (D-05)。

    実験層は系列を流す前に ``t0`` を決めるので、予測と実経路がずれると
    「全系統が同一の行で評価される」が黙って崩れる。
    """
    plan = _plan()
    spec = build_score_specs(REDUCED)[method]
    score = build_score(spec, _score_inputs(plan))
    assert score.first_valid == score_first_valid(spec)
    assert bool(np.all(np.isfinite(score.values[score.first_valid :])))
    assert bool(np.all(np.isnan(score.values[: score.first_valid])))


def test_moving_statistics_matches_a_naive_trailing_window() -> None:
    """移動統計が素朴な後方窓と一致する (累積和の実装の照合)。"""
    rng = np.random.default_rng(3)
    values: FloatArray = rng.normal(size=(200, 1))
    inputs = ScoreInputs(
        values=values,
        states=np.zeros((200, 2), dtype=np.float64),
        control_scores=rng.random(200),
        train=range(0, 100),
        calibration=range(100, 150),
        alphas=(1.0,),
    )
    window = 12
    score = build_score(MovingStatisticsSpec(window=window), inputs)
    series = values[:, 0]
    for index in (window, window + 5, 199):
        chunk = series[index - window : index]
        expected = abs(series[index] - chunk.mean()) / chunk.std()
        assert score.values[index] == pytest.approx(expected, rel=1e-9)


def test_the_input_norm_control_is_the_absolute_value_of_the_scaled_series() -> None:
    """入力ノルム対照が前処理後の系列の絶対値そのもの (D-61)。"""
    plan = _plan()
    inputs = _score_inputs(plan)
    score = build_score(build_score_specs(REDUCED)[INPUT_NORM_CONTROL], inputs)
    assert np.allclose(score.values, np.abs(inputs.values[:, 0]))


def test_the_random_control_uses_the_scores_handed_to_it() -> None:
    """一様乱数対照が ``control_scores`` をそのまま使う (乱数源を持たない)。"""
    plan = _plan()
    inputs = _score_inputs(plan)
    score = build_score(build_score_specs(REDUCED)[RANDOM_CONTROL], inputs)
    assert np.array_equal(score.values, inputs.control_scores)
    assert score.values is not inputs.control_scores


def test_smoothing_shifts_every_method_by_the_same_amount() -> None:
    """平滑化の窓が全系統に同じだけ効く (片方だけ平滑化された比較を作れない)。"""
    plan = _plan()
    inputs = _score_inputs(plan)
    window = 5
    for method in ANOMALY_METHODS:
        spec = build_score_specs(REDUCED)[method]
        raw = build_score(spec, inputs)
        smoothed = smooth_score(raw, window)
        assert smoothed.first_valid == raw.first_valid + window - 1
        start = smoothed.first_valid
        assert smoothed.values[start] == pytest.approx(
            float(np.mean(raw.values[start - window + 1 : start + 1]))
        )


def test_smoothing_of_one_is_the_identity() -> None:
    """``score_smoothing=1`` は何もしない (窓1が「平滑化なし」)。"""
    plan = _plan()
    spec = build_score_specs(REDUCED)[MOVING_STATISTICS]
    score = build_score(spec, _score_inputs(plan))
    assert smooth_score(score, 1) is score


# --- 閾値掃引 (5-B) ---------------------------------------------------------


def test_the_sweep_uses_the_same_comparison_as_the_operating_point() -> None:
    """掃引の各点が運用点と同じ規則 (``>=`` と順位切り) で計算される。

    図の曲線の上に運用点が乗ることの根拠。掃引だけ別の切り方にすると、
    「運用点だけ曲線から外れている」図が静かに出る。
    """
    rng = np.random.default_rng(11)
    labels: BoolArray = rng.random(500) < 0.1
    scores: FloatArray = rng.random(500)
    points = sweep_thresholds(labels, scores, 10)
    assert len(points) == 10
    for point in points:
        recomputed = evaluate_at_threshold(labels, scores, point.threshold)
        assert recomputed == point


def test_the_threshold_sweep_row_count_follows_the_configuration() -> None:
    """5-B の行数が ``sweep_points x 系統数 x 条件数`` になる。"""
    _, sweep = _run(REDUCED)
    assert len(sweep) == (
        REDUCED.threshold.sweep_points
        * len(ANOMALY_METHODS)
        * len(REDUCED.dataset.series)
        * REDUCED.reservoir.n_replicates
    )


# --- 系列源と打ち切り -------------------------------------------------------


def test_build_sources_returns_one_source_per_series_name() -> None:
    """``dataset.series`` の各要素が鍵になる (行数が系列数で決まる)。"""
    sources = build_sources(REDUCED)
    assert set(sources) == set(REDUCED.dataset.series)
    assert all(source.is_available() for source in sources.values())


@pytest.mark.parametrize("source", ANOMALY_SOURCES)
def test_every_declared_source_can_be_built_without_network(source: str) -> None:
    """3源すべてが**ネットワークに触れずに**組み立てられる (D-60 / D-71)。

    実データ源は ``is_available`` がキャッシュを見るだけなので、キャッシュが
    無い環境でも構築と可用性判定は通る (系列の読み取りはしない)。
    """
    config = dataclasses.replace(
        REDUCED, dataset=dataclasses.replace(REDUCED.dataset, source=source)
    )
    sources = build_sources(config)
    assert set(sources) == set(config.dataset.series)
    for item in sources.values():
        assert isinstance(item.is_available(), bool)


def test_build_sources_rejects_an_unknown_source() -> None:
    """未対応の ``dataset.source`` は例外 (黙って合成に倒さない)。"""
    config = dataclasses.replace(
        REDUCED, dataset=dataclasses.replace(REDUCED.dataset, source="nope")
    )
    with pytest.raises(ValueError, match="dataset.source"):
        build_sources(config)


def test_run_anomaly_headline_raises_when_no_source_is_available() -> None:
    """使える源が1本も無ければ例外 (0行の CSV を静かに出さない)。"""
    with pytest.raises(ValueError, match="使える系列源"):
        run_anomaly_headline(REDUCED, {})


def test_truncate_series_keeps_the_invariants() -> None:
    """打ち切りが ``AnomalySeries`` の不変条件を壊さない。"""
    plan = _plan()
    series = plan.series
    truncated = truncate_series(series, series.n_steps - 200)
    assert truncated.n_steps == series.n_steps - 200
    assert truncated.train_end <= series.train_end
    assert not bool(np.any(truncated.labels[: truncated.train_end]))
    assert truncate_series(series, series.n_steps + 10) is series


def test_truncate_series_rejects_a_length_that_removes_every_anomaly() -> None:
    """異常が残らない打ち切りは例外 (AUPRC が定義できなくなる)。"""
    plan = _plan()
    with pytest.raises(ValueError, match="異常が1点も残りません"):
        truncate_series(plan.series, plan.series.train_end)


# --- CSV の形 ---------------------------------------------------------------


def test_the_row_dict_matches_the_declared_columns() -> None:
    """行の dict が宣言した列と過不足なく一致する (列順の単一の真実)。"""
    rows, _ = _run(REDUCED)
    columns = anomaly_csv_columns(REDUCED)
    assert len(set(columns)) == len(columns)
    for row in rows:
        assert set(anomaly_row_as_dict(row)) == set(columns)


def test_the_scalar_columns_follow_the_row_declaration_order() -> None:
    """``ANOMALY_SCALAR_COLUMNS`` が ``AnomalyRow`` の宣言順そのもの。"""
    declared = [
        item.name
        for item in dataclasses.fields(AnomalyRow)
        if item.name not in {"f1_test_optimal", "point_adjust"}
    ]
    assert list(ANOMALY_SCALAR_COLUMNS) == declared

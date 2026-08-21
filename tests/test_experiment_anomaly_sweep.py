"""実験 5-C / 5-D の検査 (D-57 / D-61 / D-78 / D-79 / D-80).

仕様 (``docs/plans/rc-basics-05.md`` §4 T4) の受け入れ基準5項目に対応する:

1. ``test_sweep_reproduces_the_headline_condition_exactly`` (基準1)
2. ``test_rank_change_indicators_live_in_the_rows`` (基準2)
3. ``test_the_size_sweep_reports_the_degradation_point_and_its_saturation`` (基準3)
4. ``test_changing_the_n_units_grid_moves_the_rows_and_the_degradation_point`` (基準4)
5. ``test_every_sweep_condition_goes_through_one_preprocessor_path`` (基準5、D-57)

加えて D-78 (除外ではなく印) / D-79 (基準条件は 5-A と同じ条件) /
D-80 (5-D は学習量が揃っていることを要求する) の guard_test を持つ。

**ネットワークに触れない** (D-60)。系列源は合成か、テスト内で作った
``SeriesSource`` のスタブだけである。符号検定と Kendall tau-b の照合には
scipy を使う (実行時依存にすでに入っている。D-62 が dev に留めているのは
scikit-learn であって scipy ではない)。
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from rc_basics_lab.config import (
    Anomaly05Config,
    AnomalyDatasetConfig,
    AnomalyPreprocessConfig,
    AnomalyProtocolSweepConfig,
    AnomalyReservoirConfig,
    AnomalyRidgeConfig,
    AnomalySizeSweepConfig,
    AnomalyThresholdConfig,
    SyntheticAnomalyConfig,
)
from rc_basics_lab.experiment.anomaly import run_anomaly_headline
from rc_basics_lab.experiment.anomaly_ranking import (
    CONTROL_SIGN_TEST_ALPHA,
    aggregate_methods,
    discordant_counts,
    kendall_tau,
    method_ranks,
    sign_test_p_value,
)
from rc_basics_lab.experiment.anomaly_rows import (
    ANOMALY_PROTOCOL_CSV_COLUMNS,
    ANOMALY_SIZE_CSV_COLUMNS,
    AnomalyRow,
    ProtocolSweepRow,
)
from rc_basics_lab.experiment.anomaly_score import (
    ANOMALY_METHODS,
    ESN_RESIDUAL,
    RANDOM_CONTROL,
)
from rc_basics_lab.experiment.anomaly_sources import build_sources
from rc_basics_lab.experiment.anomaly_sweep import (
    DEGRADATION_FRACTION,
    ProtocolCondition,
    apply_protocol_condition,
    apply_size_condition,
    headline_condition,
    protocol_conditions,
    run_protocol_sweep,
    run_size_sweep,
    summarize_protocol_sweep,
    summarize_size_sweep,
)
from rc_basics_lab.tasks.anomaly import AnomalySeries, SeriesSource

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "rc_basics_lab"

SWEEP = Anomaly05Config(
    dataset=AnomalyDatasetConfig(
        series=("s1",), max_length=2000, train_ratio=0.25, calibration_ratio=0.15
    ),
    synthetic=SyntheticAnomalyConfig(
        length=2000, n_anomalies=3, segment_length=40, ignore_margin=10
    ),
    preprocess=AnomalyPreprocessConfig(
        standardize_steps=200, input_window=8, score_smoothing=4
    ),
    reservoir=AnomalyReservoirConfig(n_units=30, washout=20, n_replicates=1),
    ridge=AnomalyRidgeConfig(alpha_grid=(1e-4, 1e-2, 1.0)),
    threshold=AnomalyThresholdConfig(sweep_points=5),
    protocol_sweep=AnomalyProtocolSweepConfig(
        normalize_grid=("zscore", "minmax"),
        input_window_grid=(4, 8),
        score_smoothing_grid=(2, 4),
    ),
    size_sweep=AnomalySizeSweepConfig(n_units_grid=(20, 25, 30)),
)
"""秒オーダーで回せる縮小設定 (**構造は本番と同じ**)。実測 5-C 0.16 秒 / 5-D 0.06 秒。

格子は ``preprocess`` の既定値 (``zscore`` / 8 / 4) と ``reservoir.n_units``
(30) を必ず含む —— 含まない格子は D-79 が ``ValueError`` にする。
``n_units`` の下限が 20 なのは、``density=0.1`` では N=10 のスペクトル半径が
0 になり ``ESN`` 側が (正しく) 例外にするためである。
"""

REPLICATED = replace(SWEEP, reservoir=replace(SWEEP.reservoir, n_replicates=5))
"""印 (``distinguishable``) を測るための設定。

片側符号検定は対が5つ以上ないと 0.05 に届かない (5対すべてで上回っても
p = 0.03125、4対なら 0.0625)。``n_replicates=1`` の ``SWEEP`` ではどの系統にも
印が付かない —— これは検定の性質であって配線の失敗ではないので、印そのものを
測るテストだけレプリケートを増やす。
"""


def _sources(config: Anomaly05Config) -> Mapping[str, SeriesSource]:
    return build_sources(config)


def _headline_rows(config: Anomaly05Config) -> tuple[AnomalyRow, ...]:
    return run_anomaly_headline(config, _sources(config)).rows


def _without_wall_time(row: AnomalyRow) -> AnomalyRow:
    return replace(row, wall_time_s=0.0)


def _condition_of(row: ProtocolSweepRow) -> tuple[str, int, int]:
    return (row.normalize, row.input_window, row.score_smoothing)


# --- 受け入れ基準1: 基準の格子点が 5-A と厳密一致する -----------------------


def test_sweep_reproduces_the_headline_condition_exactly() -> None:
    """``preprocess`` と一致する格子点の行が 5-A の行と**厳密一致**する (基準1)。

    前処理が2実装に割れる経路を塞ぐのが目的なので、値の一致だけでなく
    **設定そのものの一致** (``apply_protocol_condition`` が恒等になること) も
    測る。値だけを見ると「たまたま同じ数が出る別実装」が通ってしまう。
    """
    headline = headline_condition(SWEEP)
    assert apply_protocol_condition(SWEEP, headline) == SWEEP, (
        "基準の格子点が元の設定と一致しません (掃引が別条件を回しています)"
    )

    baseline = [_without_wall_time(row) for row in _headline_rows(SWEEP)]
    through_sweep = [
        _without_wall_time(row)
        for row in _headline_rows(apply_protocol_condition(SWEEP, headline))
    ]
    assert through_sweep == baseline

    rows = run_protocol_sweep(SWEEP, _sources(SWEEP))
    marked = [row for row in rows if row.is_headline]
    assert len(marked) == len(ANOMALY_METHODS)
    for row in marked:
        values = [item.auprc for item in baseline if item.method == row.method]
        assert row.auprc_mean == float(np.mean(values)), (
            f"{row.method}: 5-C の集計が 5-A の行から作られていません"
        )


def test_the_sweeps_reject_a_grid_without_the_headline_condition() -> None:
    """格子が 5-A と同じ条件を含まなければ ``ValueError`` (D-79)。

    含まないまま回すと「5-A と同じ条件の行」が成果物に無いまま順位や劣化点
    だけが出る —— 2つの実験が同じ前処理を通っていることを CSV で照合できない。
    """
    detached = replace(
        SWEEP,
        protocol_sweep=replace(SWEEP.protocol_sweep, normalize_grid=("minmax",)),
    )
    with pytest.raises(ValueError, match="既定条件"):
        headline_condition(detached)
    with pytest.raises(ValueError, match="既定条件"):
        run_protocol_sweep(detached, _sources(detached))

    detached_size = replace(
        SWEEP, size_sweep=AnomalySizeSweepConfig(n_units_grid=(20, 25))
    )
    with pytest.raises(ValueError, match="基準 N"):
        run_size_sweep(detached_size, _sources(detached_size))


@pytest.mark.parametrize(
    "axis", ["normalize_grid", "input_window_grid", "score_smoothing_grid"]
)
def test_the_protocol_grid_rejects_empty_and_duplicated_axes(axis: str) -> None:
    """空の軸と重複のある軸を拒む (同じ条件の行が2度出るのを防ぐ)。"""
    empty = replace(SWEEP, protocol_sweep=replace(SWEEP.protocol_sweep, **{axis: ()}))
    with pytest.raises(ValueError, match="空です"):
        protocol_conditions(empty)
    doubled = getattr(SWEEP.protocol_sweep, axis) * 2
    duplicated = replace(
        SWEEP, protocol_sweep=replace(SWEEP.protocol_sweep, **{axis: doubled})
    )
    with pytest.raises(ValueError, match="重複"):
        protocol_conditions(duplicated)


def test_the_default_grids_contain_the_headline_condition() -> None:
    """既定の格子が既定の条件を含む (書き写しが崩れていないこと)。

    ``config/anomaly05_sweep.py`` は循環 import を避けるために既定値を
    リテラルで書き写している。ここが崩れると既定設定で
    ``make figures-05`` が ``ValueError`` で止まる。
    """
    default = Anomaly05Config()
    assert headline_condition(default) == ProtocolCondition(
        normalize=default.preprocess.normalize,
        input_window=default.preprocess.input_window,
        score_smoothing=default.preprocess.score_smoothing,
    )
    assert default.reservoir.n_units in default.size_sweep.n_units_grid


# --- 受け入れ基準2: 順位入替の指標が行として存在する ------------------------


def test_rank_change_indicators_live_in_the_rows() -> None:
    """``rank_changed`` / ``kendall_tau`` が CSV の列として出る (基準2)。

    図は行だけを読むので、条件レベルの量 (tau・不一致対の数) もその格子点の
    全行が持ち歩く。基準の格子点では順位が動かず tau は 1.0 になる。
    """
    assert "rank_changed" in ANOMALY_PROTOCOL_CSV_COLUMNS
    assert "kendall_tau" in ANOMALY_PROTOCOL_CSV_COLUMNS
    rows = run_protocol_sweep(SWEEP, _sources(SWEEP))
    conditions = protocol_conditions(SWEEP)
    assert len(rows) == len(conditions) * len(ANOMALY_METHODS)

    for row in rows:
        if row.is_headline:
            assert not row.rank_changed
            assert row.rank == row.reference_rank
            assert row.kendall_tau == 1.0
            assert row.n_discordant_pairs == 0

    summary = summarize_protocol_sweep(rows)
    assert summary.n_conditions == len(conditions)
    assert summary.n_rank_changed_rows == sum(1 for row in rows if row.rank_changed)
    changed = {_condition_of(row) for row in rows if row.rank_changed}
    assert summary.n_conditions_with_rank_change == len(changed)
    assert summary.min_kendall_tau == min(row.kendall_tau for row in rows)


def test_the_kendall_tau_of_the_rows_matches_scipy() -> None:
    """行の ``kendall_tau`` が ``scipy.stats.kendalltau`` (tau-b) と一致する。

    自前実装の順位相関は「少し違う値」としてしか壊れないので、独立した
    オラクルを置く (D-62 が AUPRC に置いたのと同じ規律)。
    """
    rows = run_protocol_sweep(SWEEP, _sources(SWEEP))
    reference = {row.method: row.auprc_mean for row in rows if row.is_headline}
    by_condition: dict[tuple[str, int, int], dict[str, float]] = {}
    for row in rows:
        by_condition.setdefault(_condition_of(row), {})[row.method] = row.auprc_mean
    for condition, means in by_condition.items():
        first = [reference[method] for method in ANOMALY_METHODS]
        second = [means[method] for method in ANOMALY_METHODS]
        expected = float(stats.kendalltau(first, second, variant="b").statistic)
        found = next(row.kendall_tau for row in rows if _condition_of(row) == condition)
        assert found == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_kendall_tau_is_one_for_identical_orders_and_minus_one_when_reversed() -> None:
    """端の値を固定する (同順で 1.0、逆順で -1.0、全同値で ``nan``)。"""
    values = [0.4, 0.3, 0.2, 0.1]
    assert kendall_tau(values, values) == 1.0
    assert kendall_tau(values, list(reversed(values))) == -1.0
    assert math.isnan(kendall_tau([1.0, 1.0, 1.0], [3.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="長さ"):
        kendall_tau([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="2要素"):
        kendall_tau([1.0], [1.0])


# --- D-78: 除外ではなく印 ---------------------------------------------------


def test_every_ranked_method_carries_a_control_mark_and_its_evidence() -> None:
    """6系統すべてに順位が付き、各行が印とその根拠を持つ (D-78)。

    **対照を超えた系統だけで順位を計算する除外方式を採らない** —— 除外の閾値が
    新しい任意性になるためで、代わりに「一様乱数対照と区別できるか」を行の印
    (``distinguishable``) にする。印の根拠 (``n_pairs`` /
    ``n_better_than_control`` / ``control_sign_p``) を同じ行が持つので、
    別の水準で読み直す読者は CSV だけで再判定できる。

    一様乱数対照そのものは**自分と自分を比べる**ので、どの条件でも印が付かない
    (``auprc`` と ``auprc_random`` が同じ配列から出るため上回る対が0になる)。
    """
    rows = run_protocol_sweep(REPLICATED, _sources(REPLICATED))
    for name in (
        "distinguishable",
        "n_pairs",
        "n_better_than_control",
        "control_sign_p",
        "reference_distinguishable",
    ):
        assert name in ANOMALY_PROTOCOL_CSV_COLUMNS

    by_condition: dict[tuple[str, int, int], list[ProtocolSweepRow]] = {}
    for row in rows:
        by_condition.setdefault(_condition_of(row), []).append(row)
    for condition, group in by_condition.items():
        assert {row.method for row in group} == set(ANOMALY_METHODS), (
            f"{condition}: 順位から系統が落ちています (除外方式になっています)"
        )
        assert sorted(row.rank for row in group)[0] == 1
        for row in group:
            assert row.n_pairs == REPLICATED.reservoir.n_replicates * len(
                REPLICATED.dataset.series
            )
            assert row.control_sign_p == sign_test_p_value(
                row.n_pairs, row.n_better_than_control
            )
            assert row.distinguishable == (
                row.control_sign_p <= CONTROL_SIGN_TEST_ALPHA
            )

    control_rows = [row for row in rows if row.method == RANDOM_CONTROL]
    assert control_rows
    assert not any(row.distinguishable for row in control_rows), (
        "一様乱数対照が自分自身と区別できることになっています"
    )
    esn_rows = [row for row in rows if row.method == ESN_RESIDUAL]
    assert any(row.distinguishable for row in esn_rows), (
        "ESN 残差にどの条件でも印が付きません (印が空振りしています)"
    )


def test_the_distinguishable_mark_never_downgrades_the_ranking() -> None:
    """印が0件でも順位は6系統ぶん出る (雑音でも順位そのものは記録する、D-78)。

    ``n_replicates=1`` では符号検定が 0.05 に届かないのでどの系統にも印が
    付かない。それでも順位・``kendall_tau``・不一致対の数は出る ——
    「区別できないから順位を出さない」にすると、除外方式に戻ってしまう。
    """
    rows = run_protocol_sweep(SWEEP, _sources(SWEEP))
    assert not any(row.distinguishable for row in rows)
    assert {row.method for row in rows} == set(ANOMALY_METHODS)
    assert all(1 <= row.rank <= len(ANOMALY_METHODS) for row in rows)
    assert all(row.n_discordant_pairs_distinguishable == 0 for row in rows), (
        "印が1つも無いのに『区別できる系統どうしの入れ替わり』が数えられています"
    )


def test_the_sign_test_matches_the_binomial_tail() -> None:
    """符号検定の p 値が ``scipy.stats.binomtest`` と一致する (自前実装の照合)。"""
    for n_pairs in range(1, 21):
        for n_better in range(n_pairs + 1):
            expected = float(
                stats.binomtest(n_better, n_pairs, 0.5, alternative="greater").pvalue
            )
            assert sign_test_p_value(n_pairs, n_better) == pytest.approx(
                expected, rel=1e-12
            )
    assert sign_test_p_value(5, 5) == 0.03125
    assert sign_test_p_value(4, 4) == 0.0625
    assert sign_test_p_value(0, 0) == 1.0
    with pytest.raises(ValueError, match="超えています"):
        sign_test_p_value(3, 4)


def test_the_aggregation_cannot_drop_the_controls() -> None:
    """集計は ``ANOMALY_METHODS`` を網羅した行しか受け取らない (D-61)。"""
    rows = _headline_rows(SWEEP)
    without_control = [row for row in rows if row.method != RANDOM_CONTROL]
    with pytest.raises(ValueError, match="ANOMALY_METHODS"):
        aggregate_methods(without_control)
    with pytest.raises(ValueError, match="行がありません"):
        aggregate_methods([])
    aggregates = aggregate_methods(rows)
    assert tuple(aggregates) == ANOMALY_METHODS
    with pytest.raises(ValueError, match="網羅"):
        discordant_counts(aggregates, {ESN_RESIDUAL: aggregates[ESN_RESIDUAL]})


def test_ties_share_a_rank_instead_of_following_the_declaration_order() -> None:
    """同値は同順位になる (宣言順が順位に漏れない)。"""
    aggregates = aggregate_methods(_headline_rows(SWEEP))
    flattened = {
        method: replace(item, auprc_mean=0.5) for method, item in aggregates.items()
    }
    assert set(method_ranks(flattened).values()) == {1}
    assert discordant_counts(flattened, flattened) == (0, 0)


# --- 受け入れ基準3・4: 5-D の劣化点 -----------------------------------------


def test_the_size_sweep_reports_the_degradation_point_and_its_saturation() -> None:
    """劣化点が行から決まり、端が選ばれたら ``saturated`` で分かる (基準3)。

    ``nan`` にしないのが要点である —— 「測れなかった」と「格子が足りなかった」
    が区別できなくなるうえ、meta.json の数値としても扱いにくい。
    """
    assert "below_reference_fraction" in ANOMALY_SIZE_CSV_COLUMNS
    rows = run_size_sweep(SWEEP, _sources(SWEEP))
    assert len(rows) == len(SWEEP.size_sweep.n_units_grid) * len(ANOMALY_METHODS)
    summary = summarize_size_sweep(rows)
    assert summary.method == ESN_RESIDUAL
    assert summary.reference_n_units == SWEEP.reservoir.n_units
    assert summary.degradation_fraction == DEGRADATION_FRACTION
    assert summary.n_units_at_90pct in SWEEP.size_sweep.n_units_grid
    assert not math.isnan(float(summary.n_units_at_90pct))

    degraded = [
        row.n_units
        for row in rows
        if row.method == ESN_RESIDUAL
        and row.below_reference_fraction
        and row.n_units < summary.reference_n_units
    ]
    assert summary.saturated == (not degraded)
    assert summary.n_units_at_90pct == (
        max(degraded)
        if degraded
        else min(row.n_units for row in rows if row.method == ESN_RESIDUAL)
    )

    saturated_config = replace(
        SWEEP, size_sweep=AnomalySizeSweepConfig(n_units_grid=(30, 60))
    )
    saturated = summarize_size_sweep(
        run_size_sweep(saturated_config, _sources(saturated_config))
    )
    assert saturated.saturated
    assert saturated.n_units_at_90pct == 30


def test_changing_the_n_units_grid_moves_the_rows_and_the_degradation_point() -> None:
    """``size_sweep.n_units_grid`` が行数と劣化点を変える (基準4)。

    仕様 §5 が「落としてはいけない4つ」に挙げた葉のひとつ (効いていないと
    5-D が同じ N を回している)。
    """
    base = run_size_sweep(SWEEP, _sources(SWEEP))
    changed_config = replace(
        SWEEP, size_sweep=AnomalySizeSweepConfig(n_units_grid=(30, 60))
    )
    changed = run_size_sweep(changed_config, _sources(changed_config))
    assert len(changed) != len(base)
    assert (
        summarize_size_sweep(changed).n_units_at_90pct
        != summarize_size_sweep(base).n_units_at_90pct
    )


def test_the_size_sweep_keeps_the_controls_and_the_reference_row() -> None:
    """N に依存しない系統も落とさず、基準 N の行が 5-A と一致する (D-61)。"""
    rows = run_size_sweep(SWEEP, _sources(SWEEP))
    assert {row.method for row in rows} == set(ANOMALY_METHODS)
    control = [row for row in rows if row.method == RANDOM_CONTROL]
    assert len({row.auprc_mean for row in control}) == 1, (
        "一様乱数対照が N で動いています (N がスコア構成の外へ漏れています)"
    )
    reference_rows = [row for row in rows if row.n_units == SWEEP.reservoir.n_units]
    baseline = aggregate_methods(_headline_rows(SWEEP))
    for row in reference_rows:
        assert row.auprc_mean == baseline[row.method].auprc_mean
        assert row.auprc_ratio == 1.0
        assert not row.below_reference_fraction


def test_apply_size_condition_only_moves_n_units() -> None:
    """N の差し替えが ``reservoir.n_units`` 以外を触らない。"""
    changed = apply_size_condition(SWEEP, 60)
    assert changed.reservoir.n_units == 60
    assert replace(changed, reservoir=SWEEP.reservoir) == SWEEP


# --- D-80: 5-D は学習量が揃っていることを要求する ---------------------------


@dataclass(frozen=True, slots=True)
class _FixedSeriesSource:
    """長さだけを指定して系列を作るスタブ源 (``SeriesSource`` を満たす)。"""

    n_steps: int

    def is_available(self) -> bool:
        return True

    def __call__(self, rng: np.random.Generator) -> AnomalySeries:
        values = np.sin(np.arange(self.n_steps, dtype=np.float64) / 7.0).reshape(-1, 1)
        labels = np.zeros(self.n_steps, dtype=np.bool_)
        labels[int(self.n_steps * 0.8) : int(self.n_steps * 0.8) + 20] = True
        return AnomalySeries(
            values=values,
            labels=labels,
            ignore=np.zeros(self.n_steps, dtype=np.bool_),
            train_end=int(self.n_steps * 0.7),
            name=f"fixed{self.n_steps}",
            params={},
        )


def test_the_size_sweep_requires_a_uniform_training_amount() -> None:
    """系列ごとに学習量が違うと 5-D は ``ValueError`` (D-80)。

    UCR は系列ごとに ``train_end`` が違い、系列長に対する比は実測 0.137〜0.405
    とばらつく。そのまま 5-D を回すと**学習量不足による劣化と N 不足による
    劣化が混ざり**、「N を削るとどこで落ちるか」という問いの答えが出ない。
    5-A / 5-C は系列ごとの差をそのまま行に出せばよいので、この要求は 5-D だけ
    に置く。
    """
    config = replace(
        SWEEP, dataset=replace(SWEEP.dataset, series=("a", "b"), max_length=3000)
    )
    uneven: Mapping[str, SeriesSource] = {
        "a": _FixedSeriesSource(n_steps=2000),
        "b": _FixedSeriesSource(n_steps=3000),
    }
    with pytest.raises(ValueError, match="同じ学習量"):
        run_size_sweep(config, uneven)

    even: Mapping[str, SeriesSource] = {
        "a": _FixedSeriesSource(n_steps=2000),
        "b": _FixedSeriesSource(n_steps=2000),
    }
    rows = run_size_sweep(config, even)
    assert len({row.n_train for row in rows}) == 1
    assert rows[0].n_train > 0

    protocol_rows = run_protocol_sweep(config, uneven)
    assert protocol_rows, "5-C は系列長が揃っていなくても回ること"


# --- 受け入れ基準5: 前処理の経路は1本 ---------------------------------------


def _constructs_preprocessor(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, ast.Name) and node.id == "AnomalyPreprocessor"
        for node in ast.walk(tree)
    )


def test_every_sweep_condition_goes_through_one_preprocessor_path() -> None:
    """掃引の全条件が単一の ``AnomalyPreprocessor`` 生成経路を通る (基準5、D-57)。

    2つの向きから測る:

    1. **構造**: 掃引の2モジュールは ``AnomalyPreprocessor`` を名前でも
       知らない (自前で係数を作る経路が書かれていない)
    2. **値**: 前処理に効かない軸 (``input_window`` / ``score_smoothing``) を
       振っても ``preprocessor_id`` が動かず、効く軸 (``normalize``) を振ると
       動く
    """
    for name in ("anomaly_sweep.py", "anomaly_ranking.py"):
        assert not _constructs_preprocessor(SRC / "experiment" / name), (
            f"{name} が AnomalyPreprocessor を触っています (D-57)"
        )

    fingerprints: dict[ProtocolCondition, set[str]] = {}
    for condition in protocol_conditions(SWEEP):
        rows = _headline_rows(apply_protocol_condition(SWEEP, condition))
        assert len({row.preprocessor_id for row in rows}) == 1, (
            f"{condition}: 1レプリケート内で前処理が割れています (D-57)"
        )
        fingerprints[condition] = {row.preprocessor_id for row in rows}

    by_normalize: dict[str, set[str]] = {}
    for condition, found in fingerprints.items():
        by_normalize.setdefault(condition.normalize, set()).update(found)
    for normalize, found in by_normalize.items():
        assert len(found) == 1, (
            f"normalize={normalize} の格子点で前処理係数が割れています: {found}"
        )
    assert len({next(iter(found)) for found in by_normalize.values()}) == len(
        by_normalize
    ), "normalize を変えても前処理係数が変わりません (前処理が効いていません)"

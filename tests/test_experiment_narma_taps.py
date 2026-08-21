"""実験 3-C' (タップ数掃引) の配線テスト (D-95).

3-C' の主張は「**タップ数が訓練長に近づくと正則化なしだけが壊れる**」で、
これは先行 (Goudarzi et al. 2014) の対照設計への批判そのものである。
したがって次の4つが結論を決める:

1. 各 k が**独立に**分割を持つ (k が伸びると ``t0`` も伸びて訓練区間が縮む)。
   ここを共有すると k/n_train が動かず、掃引の意味が消える
2. 2つの水準が**同一の特徴・同一の分割**で、alpha だけが違う (D-90)
3. ``k >= n_train`` を**解く前に**落とす (scipy の LinAlgError まで進めない)
4. 3-C 本体の行を1行も変えない (掃引は別の CSV へ出る)
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rc_basics_lab.config import (
    Capacity03Config,
    CapacityDriveConfig,
    ESNConfig,
    ExperimentConfig,
    IpcConfig,
    MemoryCapacityConfig,
    Narma10Config,
    RidgeConfig,
    SplitConfig,
    load_config_as,
)
from rc_basics_lab.experiment.narma import run_narma10
from rc_basics_lab.experiment.narma_taps import (
    CSV_COLUMNS,
    EXPERIMENT_NARMA10_TAPS,
    SWEPT_METHODS,
    run_narma10_tap_sweep,
    summarize_tap_sweep,
)
from rc_basics_lab.experiment.runner import DELAY_LINE, DELAY_LINE_OLS
from rc_basics_lab.seeds import SeedConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG_03 = ROOT / "experiments" / "03_capacity" / "config.yaml"

TINY_SWEEP = (4, 40, 120)


def tiny_config(sweep: tuple[int, ...] = TINY_SWEEP) -> Capacity03Config:
    """秒未満で 3-C' を回せる縮小設定 (**構造は本番と同じ**)。"""
    return Capacity03Config(
        name="narma-taps-tiny",
        drive=CapacityDriveConfig(distribution="uniform", washout=40),
        mc=MemoryCapacityConfig(max_delay=20, n_surrogates=5),
        ipc=IpcConfig(
            max_delay_by_degree=(8, 4), n_surrogates=5, n_surrogate_targets=2
        ),
        narma=Narma10Config(
            length=700,
            n_lags_sweep=sweep,
            n_replicates_sweep=2,
            base=ExperimentConfig(
                name="narma-taps-base",
                n_replicates=3,
                seeds=SeedConfig(reservoir=0, task=1, split=2),
                split=SplitConfig(washout=50, max_start_offset=20),
                ridge=RidgeConfig(alpha_grid=(1e-4, 1.0), n_lags_grid=(2, 6)),
                esn_mackey_glass=ESNConfig(
                    n_units=12, leak_rate=1.0, input_scale=1.0, density=0.5
                ),
            ),
        ),
    )


def test_each_tap_count_gets_its_own_split() -> None:
    """k ごとに ``t0`` と訓練区間が変わる (共有していない)。

    掃引の横軸は ``k / n_train`` なので、分割を共有すると分母が動かず、
    先行の動作点 (k/n ≒ 0.9) まで**届かない**。ここが崩れると 3-C' は
    3-C の焼き直しになる。
    """
    rows = run_narma10_tap_sweep(tiny_config())
    by_k = {
        row.n_lags: (row.t0, row.n_train, row.taps_per_train)
        for row in rows
        if row.method == DELAY_LINE and row.replicate == 0
    }
    assert sorted(by_k) == list(TINY_SWEEP)
    t0s = [by_k[k][0] for k in TINY_SWEEP]
    n_trains = [by_k[k][1] for k in TINY_SWEEP]
    ratios = [by_k[k][2] for k in TINY_SWEEP]
    assert t0s == sorted(t0s) and t0s[0] < t0s[-1], t0s
    assert n_trains[0] > n_trains[-1], n_trains
    assert ratios == sorted(ratios) and ratios[0] < ratios[-1], ratios
    for k in TINY_SWEEP:
        assert by_k[k][2] == pytest.approx(k / by_k[k][1])


def test_the_two_levels_differ_only_in_alpha() -> None:
    """リッジと OLS が同じ k・同じ分割で、alpha だけが違う (D-90)。"""
    rows = run_narma10_tap_sweep(tiny_config())
    assert {row.method for row in rows} == set(SWEPT_METHODS)
    assert {row.experiment for row in rows} == {EXPERIMENT_NARMA10_TAPS}
    for k in TINY_SWEEP:
        at_k = [row for row in rows if row.n_lags == k]
        ridge = [row for row in at_k if row.method == DELAY_LINE]
        ols = [row for row in at_k if row.method == DELAY_LINE_OLS]
        assert len(ridge) == len(ols) == 2, at_k
        # 同一の分割 (D-05)
        assert {(row.t0, row.n_train) for row in at_k} == {
            (ridge[0].t0, ridge[0].n_train)
        }
        # alpha だけが違う
        assert {row.alpha for row in ols} == {0.0}
        assert all(row.alpha > 0.0 for row in ridge), ridge


def test_a_tap_count_that_exceeds_the_training_split_is_rejected_before_solving() -> (
    None
):
    """``k >= n_train`` を**解く前に**落とす (scipy まで進めない)。

    scipy の ``LinAlgError`` は「どの k が悪いのか」も「なぜ悪いのか」も
    言わないので、掃引の設計を直せない。実測ではこれが本番設定の k=2600
    (n_train=2600) で起きた。
    """
    config = tiny_config(sweep=(4, 600))
    with pytest.raises(ValueError, match="訓練区間に対して大きすぎます"):
        run_narma10_tap_sweep(config)


@pytest.mark.parametrize(
    ("sweep", "match"),
    [
        ((40, 4), "昇順"),
        ((4, 4), "昇順"),
        ((0, 4), "1 以上"),
        ((4, 900), "系列長以上"),
    ],
)
def test_an_invalid_sweep_grid_is_rejected(sweep: tuple[int, ...], match: str) -> None:
    """格子そのものの誤りを回す前に落とす。"""
    with pytest.raises(ValueError, match=match):
        run_narma10_tap_sweep(tiny_config(sweep=sweep))


def test_an_empty_sweep_produces_no_rows() -> None:
    """掃引を空にすると何も回らない (3-C' を止めるノブ)。"""
    assert run_narma10_tap_sweep(tiny_config(sweep=())) == ()


def test_the_sweep_does_not_touch_the_main_narma10_rows() -> None:
    """3-C' を回しても 3-C 本体の行が1つも変わらない (別 CSV / 別軸)。"""
    config = tiny_config()
    before = run_narma10(replace(config, narma=replace(config.narma, n_lags_sweep=())))
    after = run_narma10(config)
    assert [
        (row.method, row.replicate, row.nmse, row.n_lags) for row in before.rows
    ] == [(row.method, row.replicate, row.nmse, row.n_lags) for row in after.rows]


def test_the_summary_records_where_regularisation_starts_to_matter() -> None:
    """``meta.json`` の要約だけで「どこから効くか」が読める (D-95)。"""
    rows = run_narma10_tap_sweep(tiny_config())
    summary = summarize_tap_sweep(rows)
    assert summary["n_lags_sweep"] == list(TINY_SWEEP)
    ratios = summary["ols_over_ridge"]
    assert isinstance(ratios, dict)
    assert set(ratios) == {str(k) for k in TINY_SWEEP}
    worst = summary["worst_n_lags"]
    assert worst in TINY_SWEEP
    assert summary["worst_ols_over_ridge"] == pytest.approx(
        max(float(value) for value in ratios.values())
    )
    with pytest.raises(ValueError, match="rows が空"):
        summarize_tap_sweep([])


def test_the_csv_columns_follow_the_row_declaration() -> None:
    """列順は ``TapSweepRow`` の宣言順が単一の真実。"""
    assert CSV_COLUMNS[0] == "experiment"
    assert "taps_per_train" in CSV_COLUMNS
    assert CSV_COLUMNS[-1] == "wall_time_s"


def test_the_production_sweep_reaches_the_prior_operating_point() -> None:
    """本番の格子が先行の動作点 (k/n ≒ 0.9) まで届いている (D-95)。

    ここが届いていないと、3-C' は「先行の対照設計を検証した」と言えない。
    **本番 YAML を直接読む** —— 縮小設定では届いたかどうかを測れない。
    """
    production = load_config_as(CONFIG_03, Capacity03Config)
    sweep = production.narma.n_lags_sweep
    assert sweep, "本番の n_lags_sweep が空です"
    assert sweep == tuple(sorted(set(sweep)))
    # 系列長 8000 / 訓練はおよそ半分なので、最大の k はその 0.8 倍以上に届く
    assert sweep[-1] >= 2000, sweep
    assert sweep[0] <= 30, sweep

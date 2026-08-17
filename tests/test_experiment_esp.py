"""実験 2-A / 2-B / 2-C の受け入れ条件1〜4 と D-17 guard.

**縮小した格子で結論そのものを検査する**。本番 (rho16 x sigma7 x 3rep,
N=200, T=3000) は 77 秒かかるのでテストには載せられないが、「rho<1 無入力で
指数減衰」「強入力なら rho>1 でも ESP」「λ>0 なら決して収束しない」
「リーク率に対し時定数が単調」はどれも N と T を落としても成立する性質であり、
縮小版で落ちるなら本番でも落ちる。本番格子での実測値は
``docs/plans/rc-basics-02.md`` の T3 節に記録してある。

このファイルは ``rc_basics_lab.reservoir`` を import してよい (実験層のテスト
なので)。診断層が ESN に依存しないことは
``tests/test_diagnostics_esp.py`` が別途 ESN 非依存で実演している。
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pytest

from rc_basics_lab.config import (
    DriveConfig,
    Esp02Config,
    EspConfig,
    EspDecayConfig,
    EspMapConfig,
    EspSeedConfig,
    ReservoirSweepConfig,
    TimescaleConfig,
    TimescaleSweepConfig,
)
from rc_basics_lab.experiment.esp import (
    BIAS_SCALE,
    ESP_CSV_COLUMNS,
    ESP_DISTANCE_WASHOUT,
    EXPERIMENT_DECAY,
    EXPERIMENT_ESP_MAP,
    EXPERIMENT_TIMESCALE,
    UNIFORM,
    ConditionOutcome,
    EspResults,
    EspRow,
    build_esn_config,
    esn_propagator,
    evaluate_condition,
    make_drive,
    make_initial_states,
    run_esp_experiment,
    summarize_verdict_agreement,
)
from rc_basics_lab.reservoir.esn import ESN
from rc_basics_lab.seeds import SeedStream, make_rng_for
from rc_basics_lab.types import FloatArray

DECAY_TOLERANCE = 0.20
"""``decay_rate_per_step`` と ``log rho`` の許容相対誤差 (受け入れ条件1)。"""

CONVERGING_RHOS = (0.5, 0.8, 0.95)
DIVERGING_RHOS = (1.2, 1.5)

STRONG_SIGMAS = (1.0, 2.0)
"""受け入れ条件2 で「ESP が戻る」ことを要求する入力強度。"""

SUPERCRITICAL_RHO = 1.5
"""受け入れ条件2 の rho (通説の境界より十分上)。"""


def small_config() -> Esp02Config:
    """秒未満で3実験を回せる縮小設定 (構造は本番と同じ)。

    本番から落としているのは ``n_units`` / ``n_steps`` / 格子の点数だけで、
    判定基準 (``esp`` / ``lyapunov``) と乱数の配線は本番と同じものを通す。
    """
    return Esp02Config(
        name="esp-test",
        seeds=EspSeedConfig(reservoir=0, drive=1, probe=3),
        drive=DriveConfig(distribution=UNIFORM, n_steps=900, washout=100, n_pairs=8),
        reservoir=ReservoirSweepConfig(
            input_scale=1.0, n_units=60, density=0.2, n_replicates=3
        ),
        decay=EspDecayConfig(
            rho_grid=CONVERGING_RHOS + DIVERGING_RHOS, sigma_u=0.0, leak_rate=1.0
        ),
        timescale_sweep=TimescaleSweepConfig(
            leak_rate_grid=(0.1, 0.3, 0.5, 1.0), rho=0.9, sigma_u=0.5
        ),
        esp_map=EspMapConfig(
            rho_grid=(0.6, 0.9, 1.2, 1.5, 1.8),
            sigma_grid=(0.0, 0.1, 0.5) + STRONG_SIGMAS,
            leak_rate=1.0,
        ),
        esp=EspConfig(window=100, fit_skip=10),
        timescale=TimescaleConfig(max_lag=60),
    )


@lru_cache(maxsize=1)
def results() -> EspResults:
    """縮小設定の3実験を1度だけ回す (全テストが同じ結果を見る)。"""
    return run_esp_experiment(small_config())


def _rows(outcomes: tuple[ConditionOutcome, ...]) -> tuple[EspRow, ...]:
    return tuple(outcome.row for outcome in outcomes)


# --- D-17: 入力強度は標準偏差であって振幅ではない ---------------------------


def test_input_strength_is_standard_deviation_not_amplitude() -> None:
    """``make_drive(sigma)`` の実測標準偏差が ``sigma`` になる (D-17 guard)。

    一様分布 ``U[-a, a]`` の標準偏差は ``a / sqrt(3)`` なので、指定量を振幅と
    取り違えると実際の入力強度が **1/sqrt(3) = 0.577 倍**になる。2-C の横軸が
    それだけずれると「強入力なら rho>1 でも ESP」の境界位置が変わり、記事の
    主張と数値の対応が取れなくなる。振幅と標準偏差の両方を実測して、
    取り違えた実装では通らないようにする。
    """
    sigma = 0.5
    rng = np.random.default_rng(20240202)
    drive = make_drive(sigma, 200_000, rng)

    assert drive.shape == (200_000, 1)
    measured_std = float(np.std(drive))
    measured_amplitude = float(np.max(np.abs(drive)))
    assert measured_std == pytest.approx(sigma, rel=0.02)
    assert measured_amplitude == pytest.approx(math.sqrt(3.0) * sigma, rel=0.01)
    # 振幅として解釈する実装 (a = sigma) なら標準偏差は sigma/sqrt(3) になる
    assert measured_std != pytest.approx(sigma / math.sqrt(3.0), rel=0.1)


def test_row_reports_sigma_amplitude_and_measured_std() -> None:
    """CSV の3列 (指定 sigma_u / 振幅 / 実測標準偏差) が整合する (D-17)。"""
    row = evaluate_condition(
        small_config(),
        experiment=EXPERIMENT_ESP_MAP,
        rho=0.9,
        leak_rate=1.0,
        sigma_u=0.5,
        replicate=0,
    ).row
    assert row.sigma_u == pytest.approx(0.5)
    assert row.input_amplitude == pytest.approx(math.sqrt(3.0) * 0.5)
    assert row.input_drive_std == pytest.approx(0.5, rel=0.05)


def test_zero_sigma_gives_an_exactly_zero_drive() -> None:
    """``sigma_u = 0`` は厳密なゼロ系列 (「ほぼ無入力」にしない)。"""
    drive = make_drive(0.0, 100, np.random.default_rng(0))
    assert not np.any(drive)


def test_unknown_distribution_raises() -> None:
    """未対応の分布は黙って一様として扱わない (``drive.distribution`` の配線)。"""
    with pytest.raises(ValueError, match="uniform"):
        make_drive(0.5, 100, np.random.default_rng(0), distribution="gaussian")


def test_bias_scale_is_zero_so_that_zero_sigma_is_truly_no_input() -> None:
    """02 の ESN は ``bias_scale = 0``。定数バイアスは常時入力そのもの。

    ``ESNConfig`` の既定 0.1 のままだと ``[1; u]`` の先頭成分が常に効き、
    ``sigma_u = 0`` が「無入力」でなくなる。実測ではその状態だと無入力・
    rho=1.2 でも2軌道が収束してしまい、受け入れ条件1 が成立しない。
    ここでは (1) 設定値が 0 であること、(2) 無入力・rho<1 で状態がゼロへ
    落ちること (= ゼロが不動点であること)、(3) ``ESNConfig`` の既定
    ``bias_scale=0.1`` では落ちないこと、の3つを見る。(3) が無いと
    「バイアスを戻しても何も落ちない」状態になる。
    """
    config = small_config()
    assert BIAS_SCALE == 0.0
    esn_config = build_esn_config(config, 0.5, 1.0)
    assert esn_config.bias_scale == 0.0

    zero_drive: FloatArray = np.zeros((400, 1), dtype=np.float64)
    start: FloatArray = np.full(config.reservoir.n_units, 0.3, dtype=np.float64)

    unbiased = ESN(esn_config, make_rng_for(0, SeedStream.RESERVOIR, 0), n_inputs=1)
    tail = float(np.linalg.norm(unbiased.run(zero_drive, x0=start)[-1]))
    assert tail < 1.0e-50, f"無入力なのに状態がゼロへ落ちていません: {tail!r}"

    biased = ESN(
        dataclasses.replace(esn_config, bias_scale=0.1),
        make_rng_for(0, SeedStream.RESERVOIR, 0),
        n_inputs=1,
    )
    biased_tail = float(np.linalg.norm(biased.run(zero_drive, x0=start)[-1]))
    assert biased_tail > 1.0e-3, (
        "bias_scale>0 でも状態がゼロへ落ちています "
        "(この検査が空振りしています): "
        f"{biased_tail!r}"
    )


# --- 伝播器の時間規約 (D-18) ------------------------------------------------


def test_propagator_uses_the_next_input() -> None:
    """``esn_propagator`` が ``u[t+1]`` を使う (``u[t]`` ではない)。

    ``X[t]`` は ``u[t]`` を処理した**後**の状態なので、1ステップ進めるのに
    使うのは ``u[t+1]``。ずれると λ が"それらしい値"で出るため、ここで
    ``propagator(X[t], t) == X[t+1]`` をバイト単位で固定する。
    """
    config = small_config()
    esn = ESN(
        build_esn_config(config, 0.9, 1.0),
        make_rng_for(0, SeedStream.RESERVOIR, 0),
        n_inputs=1,
    )
    drive = make_drive(0.5, 200, make_rng_for(1, SeedStream.TASK, 0))
    states = esn.run(
        drive, x0=make_initial_states(esn.n_units, 1, np.random.default_rng(0))[0]
    )
    propagate = esn_propagator(esn, drive)
    for t in (0, 37, 198):
        assert np.array_equal(propagate(states[t], t), states[t + 1])
    # u[t] を使う実装なら一致しない (検査が空振りしていないことの確認)
    assert not np.array_equal(esn.step(states[10], drive[10]), states[11])


def test_initial_states_are_all_nonzero_and_independent() -> None:
    """初期状態は ``n_pairs + 1`` 本すべて ``U[-1,1]^N`` から引く (D-16)。

    片方をゼロ状態にすると、無入力ではゼロが不動点なので「2軌道の分離」では
    なく「単一軌道の原点への収束」を測ることになり ESP と別の量に化ける。
    """
    states = make_initial_states(50, 4, np.random.default_rng(0))
    assert len(states) == 5
    for state in states:
        assert state.shape == (50,)
        assert np.any(state), "ゼロ状態が混ざっています"
    for index in range(1, len(states)):
        assert not np.array_equal(states[0], states[index])


# --- 受け入れ条件1: 無入力の減衰がスペクトル半径と一致する -------------------


def test_no_input_decay_matches_spectral_radius() -> None:
    """無入力で rho<1 は指数減衰・rho>1 は非減衰 (受け入れ条件1)。

    ``decay_rate_per_step`` は距離 ``log d`` の傾きなので、無入力なら
    ゼロ不動点まわりの線形化 (Jacobian = W) が支配し ``log rho`` に一致する。
    **全レプリケートで**成立することを要求する (特定の draw だけで成り立つ
    主張にしない)。
    """
    rows = _rows(results().decay)
    assert rows, "2-A の結果が空です"
    assert {row.experiment for row in rows} == {EXPERIMENT_DECAY}
    assert all(row.sigma_u == 0.0 for row in rows)

    for row in rows:
        expected = math.log(row.rho)
        if row.rho in CONVERGING_RHOS:
            assert row.converged == 1, (
                f"rho={row.rho} rep={row.replicate}: 無入力・rho<1 なのに非収束"
            )
            assert math.isfinite(row.decay_rate_per_step)
            assert row.decay_rate_per_step == pytest.approx(
                expected, rel=DECAY_TOLERANCE
            ), f"rho={row.rho} rep={row.replicate}: 減衰率が log rho と合いません"
        else:
            assert row.rho in DIVERGING_RHOS
            assert row.converged == 0, (
                f"rho={row.rho} rep={row.replicate}: 無入力・rho>1 なのに収束"
            )


def test_decay_fit_starts_before_the_distance_underflows() -> None:
    """減衰率の当てはめ開始は ``drive.washout`` ではなく 0 でなければならない。

    無入力で rho<1 の距離は ``EspConfig.floor`` (1e-14) を**早々に**割る。
    当てはめ点は床より上の点だけなので、``drive.washout`` の後から当てはめると
    点が2点未満になり ``decay_rate_per_step`` が ``nan`` になる (実測: 本番の
    rho=0.5 は t≈46 で床を割る一方 washout は 200)。ここでは
    「測れる区間が washout より手前で終わること」と「それでも実際に測れて
    いること」を同時に固定し、``ESP_DISTANCE_WASHOUT`` を washout に戻す
    変更が入ったら落ちるようにする。
    """
    config = small_config()
    assert ESP_DISTANCE_WASHOUT == 0
    outcomes = [
        outcome for outcome in results().decay if outcome.row.rho == CONVERGING_RHOS[0]
    ]
    assert outcomes
    for outcome in outcomes:
        below: FloatArray = np.nonzero(outcome.distance <= config.esp.floor)[0]
        assert below.size, "距離が床まで落ちていません (前提が変わっています)"
        last_measurable = int(below[0])
        assert last_measurable < config.drive.washout, (
            "測れる区間が washout より後まで残っています "
            f"(last_measurable={last_measurable}, washout={config.drive.washout})"
        )
        assert last_measurable > config.esp.fit_skip + 1, (
            "fit_skip が測れる区間を食い潰しています"
        )
        assert math.isfinite(outcome.row.decay_rate_per_step)


# --- 受け入れ条件2: 強入力なら rho>1 でも ESP が戻る -----------------------


def test_strong_input_restores_esp_above_unit_spectral_radius() -> None:
    """rho>1 でも入力を強くすると ESP が成立する (受け入れ条件2。記事の目玉)。

    **図の目視ではなくデータで固定する**。同じ rho・同じ重みで、無入力なら
    非収束、sigma_u>=1.0 なら収束することを全レプリケートで要求する。
    """
    rows = _rows(results().esp_map)
    assert rows
    supercritical = [row for row in rows if row.rho == SUPERCRITICAL_RHO]
    assert supercritical, f"rho={SUPERCRITICAL_RHO} の条件がありません"

    no_input = [row for row in supercritical if row.sigma_u == 0.0]
    assert no_input
    assert all(row.converged == 0 for row in no_input), (
        f"rho={SUPERCRITICAL_RHO} が無入力で収束しています (2-C の対照が壊れます)"
    )
    for sigma in STRONG_SIGMAS:
        strong = [row for row in supercritical if row.sigma_u == sigma]
        assert strong, f"sigma_u={sigma} の条件がありません"
        assert all(row.converged == 1 for row in strong), (
            f"rho={SUPERCRITICAL_RHO}, sigma_u={sigma} で ESP が戻っていません"
        )


# --- 受け入れ条件3: λ の符号と ESP 判定の整合 (非対称) ----------------------


def test_lyapunov_sign_agrees_with_verdict_away_from_boundary() -> None:
    """λ の符号と ESP 判定の整合。**要求は意図的に非対称である**。

    条件付き Lyapunov 指数は参照軌道まわりの**局所**量なので、多安定性
    (複数の吸引子が共存する状態) を原理的に検出できない。tanh は奇関数なので
    ``x*`` が不動点なら ``-x*`` も不動点であり、どちらも局所安定 (λ<0) で
    ありながら初期状態によって行き先が割れる (= ESP は不成立)。したがって

    - ``λ > 0`` (局所的に伸びる) なのに収束する = **偽の ESP** は起きては
      ならない。これは全条件で要求する
    - ``λ < 0`` なのに非収束は起きうる。実測 (本番格子 336 条件) では 25 件
      あり、すべて ``sigma_u <= 0.2`` かつ ``rho in [1.1, 1.6]`` に限局していた

    **この非対称性を「揃っていないから」と対称化しないこと**。対称化すると、
    多安定性という実在の現象を実装バグとして潰すことになる。駆動が十分ある
    領域 (``sigma_u >= STRONG_DRIVE_SIGMA``) では両者は完全に一致するので、
    そちらは 0 件を要求して検査が空振りしないようにしてある。
    """
    rows = results().rows
    agreement = summarize_verdict_agreement(rows)

    assert agreement.n_compared > 0, "境界近傍を除いた比較対象が空です"
    assert agreement.n_false_esp == 0, (
        "λ>0 なのに ESP 成立と判定された条件があります "
        "(伝播器の入力インデックスずれか、判定閾値の緩みを疑うこと)"
    )
    assert agreement.n_compared_strong_drive > 0, "強駆動の比較対象が空です"
    assert agreement.n_disagreement_strong_drive == 0, (
        "駆動が十分ある領域で λ の符号と ESP 判定が食い違っています: "
        f"sigma_u>={agreement.strong_drive_sigma}"
    )


def test_verdict_agreement_summary_records_the_disagreement_breakdown() -> None:
    """不一致の内訳 (件数 + sigma_u / rho の分布) が要約に残る (記事の一次資料)。

    件数だけだと「どこで局所と大域が食い違うか」が復元できない。多安定性は
    弱駆動・臨界超えに限局するという観測そのものが記事の材料なので、分布まで
    ``meta.json`` に落とす。
    """
    summary = summarize_verdict_agreement(results().rows).to_summary()
    assert summary["n_false_esp"] == 0
    assert summary["n_compared"] > 0
    assert isinstance(summary["disagreement_by_sigma"], list)
    assert isinstance(summary["disagreement_by_rho"], list)
    assert summary["n_near_boundary"] + summary["n_compared"] == summary["n_rows"]


def test_disagreement_is_only_the_local_stable_but_not_global_direction() -> None:
    """不一致は「λ<0 なのに非収束」の一方向だけである (向きの固定)。"""
    agreement = summarize_verdict_agreement(results().rows)
    n_disagreements = agreement.n_false_esp + agreement.n_local_but_not_global
    assert agreement.disagreement_rate == pytest.approx(
        n_disagreements / agreement.n_compared
    )
    assert agreement.n_false_esp == 0


# --- 受け入れ条件4: リーク率と実効時定数の単調性 ---------------------------


def test_timescale_is_monotone_in_leak_rate() -> None:
    """リーク率を上げると実効時定数が単調に縮む (受け入れ条件4)。

    漏れ積分 ``x[t+1] = (1-a) x[t] + a f(...)`` は1次のローパスなので、
    ``a`` を上げるほど過去の重みが落ちる。理論線 ``-1/log(1-a)`` は線形域の
    値で実測より小さく出るが、**単調性は一致する**ことを要求する。
    """
    rows = _rows(results().timescale)
    assert rows
    assert {row.experiment for row in rows} == {EXPERIMENT_TIMESCALE}
    leak_rates = sorted({row.leak_rate for row in rows})
    assert len(leak_rates) > 1

    means = [
        float(np.mean([row.tau_censored for row in rows if row.leak_rate == leak_rate]))
        for leak_rate in leak_rates
    ]
    assert all(math.isfinite(value) for value in means)
    assert means == sorted(means, reverse=True), (
        f"リーク率に対し時定数が単調非増加ではありません: {means}"
    )
    assert means[0] > means[-1], "掃引の両端で時定数が変わっていません"


# --- 行と CSV 列 -----------------------------------------------------------


def test_csv_columns_follow_the_row_declaration_order() -> None:
    """CSV の列順は ``EspRow`` の宣言順が単一の真実 (01 の慣習と同じ)。"""
    assert ESP_CSV_COLUMNS[0] == "experiment"
    assert ESP_CSV_COLUMNS[-1] == "wall_time_s"
    assert "sigma_u" in ESP_CSV_COLUMNS
    assert "input_amplitude" in ESP_CSV_COLUMNS
    assert "input_drive_std" in ESP_CSV_COLUMNS


def test_all_three_experiments_are_labelled() -> None:
    """3実験ぶんの行がそろい、``experiment`` 列で区別できる。"""
    labels = {row.experiment for row in results().rows}
    assert labels == {EXPERIMENT_DECAY, EXPERIMENT_TIMESCALE, EXPERIMENT_ESP_MAP}


def test_rows_carry_the_reservoir_and_drive_conditions() -> None:
    """条件を後から復元できる列がすべて埋まっている。"""
    config = small_config()
    for row in results().rows:
        assert row.n_units == config.reservoir.n_units
        assert row.density == config.reservoir.density
        assert row.input_scale == config.reservoir.input_scale
        assert row.n_steps == config.drive.n_steps
        assert row.washout == config.drive.washout
        assert row.window == config.esp.window
        assert row.n_pairs == config.drive.n_pairs
        assert row.seed_reservoir == config.seeds.reservoir
        assert row.seed_drive == config.seeds.drive
        assert row.seed_probe == config.seeds.probe
        assert row.wall_time_s > 0.0

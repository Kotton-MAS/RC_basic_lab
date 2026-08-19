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

import dataclasses
import inspect
import math
from functools import lru_cache

import numpy as np
import pytest

import rc_basics_lab.experiment.esp as esp_module
from rc_basics_lab.config import (
    DriveConfig,
    ESNConfig,
    Esp02Config,
    EspConfig,
    EspDecayConfig,
    EspMapConfig,
    EspSeedConfig,
    ReservoirSweepConfig,
    TimescaleConfig,
    TimescaleSweepConfig,
)
from rc_basics_lab.diagnostics.base import DiagnosticContext
from rc_basics_lab.diagnostics.esp import DEFAULT_LYAPUNOV, conditional_lyapunov
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
    run_decay_sweep,
    run_esp_experiment,
    simulate_condition,
    simulate_reference_trajectory,
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
            sigma_grid=(0.0, 0.1, 0.5, *STRONG_SIGMAS),
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


def test_make_drive_rejects_a_negative_sigma() -> None:
    """``sigma`` が負なら ``ValueError`` (docstring の Raises)。"""
    with pytest.raises(ValueError, match="sigma_u"):
        make_drive(-0.1, 100, np.random.default_rng(0))


def test_make_drive_rejects_fewer_than_one_step() -> None:
    """``n_steps`` が1未満なら ``ValueError`` (docstring の Raises)。"""
    with pytest.raises(ValueError, match="n_steps"):
        make_drive(0.5, 0, np.random.default_rng(0))


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
    esn_config = build_esn_config(config.reservoir, 0.5, 1.0)
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
        build_esn_config(config.reservoir, 0.9, 1.0),
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


def test_perturbation_growth_stays_far_below_the_runaway_limit() -> None:
    """1区間あたりの成長率が ``max_growth`` に対して桁で余裕がある (D-18)。

    ``delta=1e-8`` と ``renorm_interval=1`` を選んだ理由は「摂動が線形域を
    外れない」ことであり、``max_growth=1000`` はその前提が崩れたときの
    暴走検出でしかない。両者が接近していたら再正規化間隔が長すぎる
    (= 推定値が線形化の範囲を出ている) というサインになるので、余裕そのものを
    固定する。``docs/design.md`` §9 に本番格子での実測 (最大 1.5372 /
    max_growth の 1/650) を記録してある。
    """
    config = small_config()
    limit = config.lyapunov.max_growth
    observed = []
    for rho in (0.9, 1.8):
        esn = ESN(
            build_esn_config(config.reservoir, rho, 1.0),
            make_rng_for(0, SeedStream.RESERVOIR, 0),
            n_inputs=1,
        )
        drive = make_drive(0.5, 400, make_rng_for(1, SeedStream.TASK, 0))
        states = esn.run(
            drive, x0=make_initial_states(esn.n_units, 1, np.random.default_rng(0))[0]
        )
        result = conditional_lyapunov(
            states,
            ctx=DiagnosticContext(washout=100, propagator=esn_propagator(esn, drive)),
            cfg=config.lyapunov,
        )
        observed.append(result.scalars["max_observed_growth"])
    assert all(math.isfinite(growth) for growth in observed)
    # 1ステップで1桁も成長しない = 摂動は線形域の内側にある
    assert max(observed) < 10.0, observed
    assert max(observed) < limit / 100.0, (observed, limit)


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


def test_make_initial_states_rejects_fewer_than_one_unit() -> None:
    """``n_units`` が1未満なら ``ValueError`` (docstring の Raises)。"""
    with pytest.raises(ValueError, match="n_units"):
        make_initial_states(0, 4, np.random.default_rng(0))


def test_make_initial_states_rejects_fewer_than_one_pair() -> None:
    """``n_pairs`` が1未満なら ``ValueError`` (docstring の Raises)。"""
    with pytest.raises(ValueError, match="n_pairs"):
        make_initial_states(50, 0, np.random.default_rng(0))


def test_sweep_rejects_fewer_than_one_replicate() -> None:
    """``reservoir.n_replicates`` が1未満なら ``_sweep`` が ``ValueError``。

    ``_sweep`` は ``run_decay_sweep`` / ``run_timescale_sweep`` /
    ``run_esp_map`` の共通ループなので、代表として ``run_decay_sweep`` から
    到達させる。
    """
    config = dataclasses.replace(
        small_config(),
        reservoir=dataclasses.replace(small_config().reservoir, n_replicates=0),
    )
    with pytest.raises(ValueError, match="n_replicates"):
        run_decay_sweep(config)


# --- F-1-005: 03 (MC/IPC) が写経せずに再利用できる継ぎ目 --------------------


def test_simulate_reference_trajectory_does_not_require_esp02_config() -> None:
    """``simulate_reference_trajectory`` は ``Esp02Config`` を要求しない。

    ``inspect.signature`` でパラメータの型注釈を見て、``ReservoirSweepConfig`` /
    ``DriveConfig`` / 基底シード (int) だけで呼べることを固定する。03 が
    ``Esp02Config`` 全体を写経せずにこの関数を再利用できることの根拠。
    """
    import inspect

    # ``from __future__ import annotations`` により注釈は文字列になるため、
    # 型オブジェクトの同一性ではなく名前で確認する。
    signature = inspect.signature(simulate_reference_trajectory)
    annotations = {
        name: parameter.annotation for name, parameter in signature.parameters.items()
    }
    assert annotations["reservoir"] == ReservoirSweepConfig.__name__
    assert annotations["drive_config"] == DriveConfig.__name__
    assert "Esp02Config" not in str(signature)


def test_simulate_reference_trajectory_matches_simulate_condition_bit_for_bit() -> None:
    """参照軌道の切り出しが ``simulate_condition`` の数値を1バイトも変えない。

    F-1-005 は ``simulate_condition`` を ``simulate_reference_trajectory`` +
    比較軌道の薄い層に書き換える修正だが、既存の成果物 (``results/``) は
    不変でなければならない。ここでは両者の駆動入力・参照軌道の状態系列が
    完全一致することを直接固定する。
    """
    config = small_config()
    trajectories = simulate_condition(
        config, rho=0.9, leak_rate=1.0, sigma_u=0.5, replicate=0
    )
    probe_rng = make_rng_for(config.seeds.probe, SeedStream.PROBE, 0)
    initial_states = make_initial_states(
        config.reservoir.n_units, config.drive.n_pairs, probe_rng
    )
    reference = simulate_reference_trajectory(
        config.reservoir,
        config.drive,
        reservoir_seed=config.seeds.reservoir,
        drive_seed=config.seeds.drive,
        rho=0.9,
        leak_rate=1.0,
        sigma_u=0.5,
        replicate=0,
        x0=initial_states[0],
    )
    assert np.array_equal(reference.drive, trajectories.drive)
    assert np.array_equal(reference.states, trajectories.states)


def test_simulate_condition_always_passes_rng_to_every_esn_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``simulate_condition`` の比較軌道ループも常に ``rng`` を渡す (D-36)。

    F-3b1-1-001: D-36 の rule は『``ESN.run`` には常に ``rng`` を渡す』だが、
    以前は参照軌道 (``simulate_reference_trajectory`` 経由) にしか適用されず、
    比較軌道ループ (``reference.esn.run(reference.drive, x0=x0)``) は ``rng``
    無しで呼ばれていた。既存の
    ``test_reference_states_match_esp_simulate_condition`` (D-36 guard) は
    ``state_noise=0`` でのバイト不変だけを見ており、この配線漏れを検出しない。
    ``ESN.run`` を monkeypatch して**すべての**呼び出し (参照軌道1回 + 比較軌道
    ``n_pairs`` 回) が ``rng is not None`` で呼ばれることを直接固定する。
    """
    config = small_config()
    original_run = ESN.run
    seen_rngs: list[np.random.Generator | None] = []

    def recording_run(
        self: ESN,
        u: FloatArray,
        x0: FloatArray | None = None,
        rng: np.random.Generator | None = None,
    ) -> FloatArray:
        seen_rngs.append(rng)
        return original_run(self, u, x0=x0, rng=rng)

    monkeypatch.setattr(ESN, "run", recording_run)
    simulate_condition(config, rho=0.9, leak_rate=1.0, sigma_u=0.5, replicate=0)

    assert len(seen_rngs) == 1 + config.drive.n_pairs
    assert all(rng is not None for rng in seen_rngs)


def test_simulate_reference_trajectory_defaults_to_a_zero_initial_state() -> None:
    """``x0`` を省略すると ``ESN.run`` の既定 (零ベクトル) が使われる。

    MC/IPC は参照軌道1本だけを要るので、ESP のように ``SeedStream.PROBE`` から
    ランダムな初期状態対を引く必要が無い。
    """
    config = small_config()
    reference = simulate_reference_trajectory(
        config.reservoir,
        config.drive,
        reservoir_seed=config.seeds.reservoir,
        drive_seed=config.seeds.drive,
        rho=0.9,
        leak_rate=1.0,
        sigma_u=0.5,
        replicate=0,
    )
    esn = ESN(
        build_esn_config(config.reservoir, 0.9, 1.0),
        make_rng_for(config.seeds.reservoir, SeedStream.RESERVOIR, 0),
        n_inputs=1,
    )
    expected = esn.run(reference.drive)
    assert np.array_equal(reference.states, expected)


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
        below = np.nonzero(outcome.distance <= config.esp.floor)[0]
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
    - ``λ < 0`` なのに非収束は起きうる。実測 (本番格子 332 条件、境界近傍を
      除いた比較対象) では 27 件あり、すべて ``sigma_u <= 0.2`` かつ
      ``rho in [1.1, 1.6]`` に限局していた

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
    agreement = summarize_verdict_agreement(results().rows)
    summary = agreement.to_summary()
    assert summary["n_false_esp"] == 0
    assert summary["n_compared"] == agreement.n_compared > 0
    assert isinstance(summary["disagreement_by_sigma"], list)
    assert isinstance(summary["disagreement_by_rho"], list)
    assert agreement.n_near_boundary + agreement.n_compared == agreement.n_rows


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


# --- D-48: 伝播器は決定的でなければならない ---------------------------------

NOISE_SIGMA = 1.0e-4
"""D-48 / D-47 の拒否テストで使う状態ノイズ。

04 の格子 {0, 1e-4, 1e-3, 1e-2} の下端。**最も効きの弱い値でも拒否される**
ことを固定するため、格子の中で一番小さい非ゼロを選んである。
"""


def _noisy_esn(state_noise: float = NOISE_SIGMA) -> ESN:
    """``state_noise>0`` の ESN。重みは ``small_config`` の条件と同じ乱数列から。"""
    return ESN(
        build_esn_config(small_config().reservoir, 0.9, 0.3, state_noise=state_noise),
        make_rng_for(0, SeedStream.RESERVOIR, 0),
        n_inputs=1,
    )


def test_propagator_refuses_a_noisy_esn(monkeypatch: pytest.MonkeyPatch) -> None:
    """``esn_propagator`` は ``state_noise>0`` の ESN を受理しない (D-48)。

    ``ESN.step`` が**1回も呼ばれない**ことまで測る —— 検査を消して
    ``esn.step(x, u, rng)`` に差し替える変異 (最も安い直し方) は、
    「``ValueError`` にならない」だけでなく「``step`` が呼ばれる」形で入るため。

    メッセージには4点 (何を / なぜ / やってはいけない直し方2つ / 正しい経路) が
    そろっている必要がある。ここが薄いと、次の実装者はメッセージではなく
    「一番安く緑にする手」を読んで ``rng`` を渡す。
    """
    esn = _noisy_esn()
    drive = make_drive(0.5, 50, make_rng_for(1, SeedStream.TASK, 0))

    calls: list[object] = []
    original_step = ESN.step

    def recording_step(
        self: ESN,
        x: FloatArray,
        u: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> FloatArray:
        calls.append(rng)
        return original_step(self, x, u, rng)

    monkeypatch.setattr(ESN, "step", recording_step)

    with pytest.raises(ValueError) as excinfo:
        esn_propagator(esn, drive)

    message = str(excinfo.value)
    assert not calls, "拒否する前に ESN.step が呼ばれています (D-48)"
    assert "D-48" in message
    assert "摂動" in message and "ノイズ実現値" in message  # なぜ
    assert "rng を渡して黙らせる" in message  # やってはいけない直し方 1
    assert "ノイズ無しの複製" in message  # やってはいけない直し方 2
    assert "state_noise=0" in message  # 正しい経路
    assert repr(NOISE_SIGMA) in message  # 実際の値


def test_propagator_accepts_a_noise_free_esn_and_is_deterministic() -> None:
    """``state_noise=0`` なら通り、同じ ``(x, t)`` の2回の呼び出しがビット一致 (D-48)。

    拒否テストだけだと「全部の ESN を拒否する」実装でも緑になる。決定的である
    ことと、正常系が通ることを同じ場所で固定する。
    """
    esn = ESN(
        build_esn_config(small_config().reservoir, 0.9, 0.3),
        make_rng_for(0, SeedStream.RESERVOIR, 0),
        n_inputs=1,
    )
    drive = make_drive(0.5, 50, make_rng_for(1, SeedStream.TASK, 0))
    propagate = esn_propagator(esn, drive)
    x = esn.run(drive)[10]
    assert np.array_equal(propagate(x, 10), propagate(x, 10))


def test_noise_free_clone_fails_the_propagator_check() -> None:
    """却下案A (ノイズ無しの複製で伝播する) が成立しないことの実測記録 (D-48)。

    ADR 0001 §2.3: ノイズ有りの参照軌道 ``X`` にノイズ無しの複製を当てると
    ``propagator(X[t], t) != X[t+1]`` になり、差は ``a·[tanh(pre) -
    tanh(pre + σξ)]`` のオーダー。``propagator_tol=1e-10`` を桁で超えるので
    ``check_propagator=True`` では**必ず** ``ValueError`` になり、しかも
    メッセージは「参照軌道と別の入力で伝播している疑い」という**誤った診断**で、
    次の実装者を存在しないバグの捜索へ送り込む。

    このテストが無いと、案A を却下した理由 (「安全に見えて成立しない」) が
    次のサイクルで失われ、同じ検討が最初からやり直しになる。
    """
    reservoir = small_config().reservoir
    drive = make_drive(0.5, 300, make_rng_for(1, SeedStream.TASK, 0))
    noisy = _noisy_esn()
    states = noisy.run(drive, rng=make_rng_for(0, SeedStream.RESERVOIR, 0))

    # ノイズ無しの複製 (重みは同じ乱数列から作るので W / W_in はビット一致)。
    clone = ESN(
        build_esn_config(reservoir, 0.9, 0.3),
        make_rng_for(0, SeedStream.RESERVOIR, 0),
        n_inputs=1,
    )
    assert np.array_equal(clone.W, noisy.W)
    assert clone.config.state_noise == 0.0

    # 1. 不一致量そのものが propagator_tol を桁で超える。
    tol = DEFAULT_LYAPUNOV.propagator_tol
    mismatches = [
        float(
            np.linalg.norm(clone.step(states[t], drive[t + 1]) - states[t + 1])
            / math.sqrt(clone.n_units)
        )
        for t in (100, 150, 200, 250)
    ]
    worst = max(mismatches)
    assert worst > tol * 1.0e3, (
        "ノイズ無し複製の不一致量が propagator_tol を桁で超えていません "
        f"(worst={worst!r}, tol={tol!r})。この前提が崩れたなら ADR 0001 §2.5 の "
        "見直し条件 (ノイズが tanh の外側へ移った) に該当します"
    )

    # 2. D-18 の検査が**誤った診断メッセージ**で落ちる (案A の本当の害)。
    with pytest.raises(ValueError) as excinfo:
        conditional_lyapunov(
            states,
            ctx=DiagnosticContext(washout=100, propagator=esn_propagator(clone, drive)),
        )
    message = str(excinfo.value)
    assert "参照軌道と別の入力で伝播している疑い" in message, (
        "案A の失敗が D-18 の誤診として出ることを固定できていません: " + message
    )


# --- D-47: 比較軌道の経路は state_noise>0 を受理しない -----------------------


def test_simulate_condition_rejects_state_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``simulate_condition(state_noise=...)`` は 0 以外を拒否する (D-47。主)。

    ``ESN.run`` が**1回も呼ばれない**ことまで測る。引数で受けて拒否する形に
    してあるので、guard は monkeypatch 依存の間接的なものにならない
    (ADR 0001 §3.3-1: 次の実装者の手が触れる場所に停止標識を置く)。
    """
    calls: list[object] = []
    original_run = ESN.run

    def recording_run(
        self: ESN,
        u: FloatArray,
        x0: FloatArray | None = None,
        rng: np.random.Generator | None = None,
    ) -> FloatArray:
        calls.append(rng)
        return original_run(self, u, x0=x0, rng=rng)

    monkeypatch.setattr(ESN, "run", recording_run)

    with pytest.raises(ValueError) as excinfo:
        simulate_condition(
            small_config(),
            rho=0.9,
            leak_rate=0.3,
            sigma_u=0.5,
            replicate=0,
            state_noise=NOISE_SIGMA,
        )

    message = str(excinfo.value)
    assert not calls, "拒否する前に ESN.run が呼ばれています (D-47)"
    assert "D-47" in message
    assert "評価順" in message  # なぜ (2つめ)
    assert "4本目" in message  # なぜ (1つめ)
    assert "5本目の乱数ストリームを新設" in message  # やってはいけない直し方
    assert "simulate_reference_trajectory" in message  # 正しい経路


def test_simulate_condition_rejects_a_noisy_esn_from_any_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """引数を通らない経路でノイズが入っても比較軌道ループ前に落ちる (D-47。副)。

    **二重化が空虚でないことの証明**: 入口の引数検査だけを消す変異はこのテストで、
    ESN 側の検査だけを消す変異は ``test_simulate_condition_rejects_state_noise``
    で落ちる。どちらか片方しか無いと、片方を消しても1件も落ちない。

    ここでは ``build_esn_config`` を差し替えて「設定にノイズのフィールドが
    生えた」状況を作る —— 04 の 4-C が ``state_noise`` を掃引軸にするので、
    この経路は仮想的な話ではない。
    """
    original = esp_module.build_esn_config

    def noisy_build(
        reservoir: ReservoirSweepConfig,
        rho: float,
        leak_rate: float,
        *,
        state_noise: float = 0.0,
    ) -> ESNConfig:
        return original(reservoir, rho, leak_rate, state_noise=NOISE_SIGMA)

    monkeypatch.setattr(esp_module, "build_esn_config", noisy_build)

    with pytest.raises(ValueError) as excinfo:
        simulate_condition(
            small_config(), rho=0.9, leak_rate=0.3, sigma_u=0.5, replicate=0
        )

    message = str(excinfo.value)
    assert "D-47" in message
    assert "比較軌道" in message
    assert "5本目の乱数ストリームを新設" in message
    assert "simulate_reference_trajectory" in message


def test_simulate_condition_defaults_to_zero_state_noise() -> None:
    """既定値のままなら 02 の経路は1つも書き換わらない (D-47)。

    ``experiment/threshold.py`` を含む既存の呼び出しは ``state_noise`` を
    渡さない。既定が 0 でなくなると 02・03 の成果物が黙って変わる。
    """
    signature = inspect.signature(simulate_condition)
    parameter = signature.parameters["state_noise"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == 0.0
    trajectories = simulate_condition(
        small_config(), rho=0.9, leak_rate=0.3, sigma_u=0.5, replicate=0
    )
    assert trajectories.esn.config.state_noise == 0.0

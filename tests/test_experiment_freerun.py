"""実験 4-A / 4-B と自走の入口の検査 (D-31 / D-34 / D-36 / D-41 / D-43〜D-46).

守るのは6系統。

1. **01 の経路をそのまま通していること** (D-31)。4-A は ``run_task`` を呼ぶ
   だけで、公平性 (D-04 / D-05 / D-08) を書き写していない
2. **自走が教師強制の係数をそのまま使うこと** (D-44)。同一性 **と**
   「学習し直すと値が変わる」の両方を測る (片方だけだと空虚になる)
3. **確保軸3** が自走の確保より前に効くこと (D-34)
4. **D-48 と D-36 の境界**。伝播器 (02) は決定的、自走 (04) は
   ``state_noise > 0`` なら rng を渡す
5. **教師強制と自走で同じ特徴・同じ係数を使うこと** (仕様 §5 禁止する構造2)。
   3手法とも、閉ループの設計行列が教師強制の行と一致する
6. **成果物 (``results/04_chaotic_freerun/``) に対する受け入れ条件1・2・3・5**。
   図ではなく行から判定する
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from wiring import experiment_config

from rc_basics_lab.config import (
    Chaos04Config,
    ESNConfig,
    ExperimentConfig,
    FreeRunConfig,
    LorenzConfig,
    MackeyGlassConfig,
    MackeyGlassStandardizeConfig,
    MackeyGlassTask,
    RidgeConfig,
    SplitConfig,
    require_task,
)
from rc_basics_lab.experiment import freerun as freerun_module
from rc_basics_lab.experiment import freerun_rows as freerun_rows_module
from rc_basics_lab.experiment import freerun_tasks as freerun_tasks_module
from rc_basics_lab.experiment.attractor import shuffled_surrogate
from rc_basics_lab.experiment.freerun import (
    CHAOS_ESN_SECTION,
    FREE_RUN_SPEC,
    FREERUN_CSV,
    FREERUN_CSV_COLUMNS,
    FREERUN_METHODS,
    ONESTEP_ARTIFACTS,
    ONESTEP_CSV,
    PROFILE_MAX_POINTS,
    PROFILE_REPLICATE,
    closed_loop_setup,
    delay_line_state_updater,
    esn_state_updater,
    estimate_lorenz_lyapunov,
    fit_teacher_forced,
    passthrough_state_updater,
    run_and_report_onestep,
    run_free_run,
    run_freerun_experiment,
    run_onestep,
    sign_test_p_value,
)
from rc_basics_lab.experiment.freerun_tasks import (
    chaos_reservoir_config,
    chaos_task_entries,
    lorenz_task_entry,
    validate_free_run_bounds,
    validate_standardization_window,
)
from rc_basics_lab.experiment.runner import (
    CSV_COLUMNS,
    DELAY_LINE,
    ESN_METHOD,
    LINEAR,
    ResultRow,
    build_methods,
    build_tasks,
    plan_replicate,
)
from rc_basics_lab.experiment.split import make_split
from rc_basics_lab.experiment.valid_time import (
    LITERATURE_VPT_THRESHOLD,
    VALID_TIME_THRESHOLD_GRID,
)
from rc_basics_lab.readout.design import ReservoirSpec, build_design_matrix
from rc_basics_lab.readout.ridge import fit_ridge
from rc_basics_lab.reservoir.esn import ESN
from rc_basics_lab.reservoir.registry import build_reservoir
from rc_basics_lab.reservoir.topology import ErdosRenyiConfig
from rc_basics_lab.seeds import SeedStream, make_rng
from rc_basics_lab.tasks.chaotic import TASK_NAME_LORENZ, integrate_lorenz
from rc_basics_lab.tasks.mackey_glass import TASK_NAME as TASK_NAME_MACKEY_GLASS
from rc_basics_lab.types import FloatArray


def small_config() -> Chaos04Config:
    """秒未満で 4-A と自走を回せる縮小設定 (**構造は本番と同じ**)。

    個々のテストは ``dataclasses.replace`` で必要な軸だけを動かす。
    """
    return Chaos04Config(
        name="chaos-freerun-test",
        base=experiment_config(
            name="chaos-freerun-test-base",
            n_replicates=2,
            split=SplitConfig(washout=30, max_start_offset=10),
            ridge=RidgeConfig(alpha_grid=(1.0e-6, 1.0e-3), n_lags_grid=(2, 4)),
            mackey_glass=MackeyGlassConfig(length=600, integration_burn_in=100),
            esn_mackey_glass=ESNConfig(
                n_units=20,
                leak_rate=0.5,
                input_scale=0.5,
                topology=ErdosRenyiConfig(density=0.5),
            ),
        ),
        lorenz=LorenzConfig(length=600, integration_burn_in=100, standardize_steps=150),
        mackey_glass=MackeyGlassStandardizeConfig(standardize_steps=150),
        freerun=FreeRunConfig(warmup_steps=10, free_run_steps=40, stats_steps=80),
    )


# --- 1. 01 の経路をそのまま通す (D-31) ---------------------------------------


def mg_params(config: Chaos04Config) -> MackeyGlassConfig:
    """04 の土台にある MG 課題の生成パラメータ (D-123)。"""
    return require_task(config.base, MackeyGlassTask, "04 のテスト").params


def test_onestep_reuses_the_01_result_row() -> None:
    """4-A の行が 01 の ``ResultRow`` そのものである (D-31)。"""
    rows = run_onestep(small_config())
    assert rows and all(isinstance(row, ResultRow) for row in rows)
    assert {row.task for row in rows} == {TASK_NAME_LORENZ, TASK_NAME_MACKEY_GLASS}
    assert {row.method for row in rows} == {LINEAR, DELAY_LINE, ESN_METHOD}


def test_onestep_shares_the_split_across_methods_within_a_replicate() -> None:
    """同一レプリケート内で ``(t0, n_train, n_val, n_test)`` が3手法で一致 (D-05)。

    01 の ``plan_replicate`` を通っていることの実測。4-A 側で分割を作り直す
    実装に変えるとここで落ちる。
    """
    rows = run_onestep(small_config())
    grouped: dict[tuple[str, int], set[tuple[int, int, int, int]]] = {}
    for row in rows:
        key = (row.task, row.replicate)
        grouped.setdefault(key, set()).add((row.t0, row.n_train, row.n_val, row.n_test))
    for key, shapes in grouped.items():
        assert len(shapes) == 1, f"{key} で手法ごとに分割が違います: {shapes}"
    assert len(grouped) == 2 * 2  # 課題2 x レプリケート2


def test_chaos_tasks_are_not_added_to_build_tasks() -> None:
    """``build_tasks`` / ``ExperimentConfig`` に 04 の課題を足していない (D-13 / D-31)。

    足すと 01 の ``comparison.csv`` に行が増え、01 の成果物が変わる。
    """
    names = {entry.name for entry in build_tasks(ExperimentConfig())}
    assert TASK_NAME_LORENZ not in names
    field_names = {item.name for item in dataclasses.fields(ExperimentConfig)}
    assert not [name for name in field_names if "lorenz" in name or "freerun" in name]


def test_chaos_esn_section_matches_the_declared_choice() -> None:
    """宣言した ESN セクション名と、実際に読む属性が一致する。"""
    config = small_config()
    assert CHAOS_ESN_SECTION == "mackey_glass"
    declared = require_task(config.base, MackeyGlassTask, "テスト").reservoir
    assert chaos_reservoir_config(config.base) is declared
    for entry in chaos_task_entries(config):
        assert entry.reservoir is chaos_reservoir_config(config.base)


def test_free_run_spec_matches_the_one_step_esn_candidate() -> None:
    """自走の特徴仕様が 4-A の ESN 手法の候補と同一の値である。

    ずれると「教師強制と自走で別の特徴を使う」(仕様 §5 禁止する構造2) になる。
    """
    methods = {method.name: method for method in build_methods(ExperimentConfig())}
    assert methods[ESN_METHOD].candidates == (FREE_RUN_SPEC,)
    assert ReservoirSpec() == FREE_RUN_SPEC


# --- 2. 自走は教師強制の係数をそのまま使う (D-44) -----------------------------


def test_free_run_uses_the_teacher_forced_coefficients() -> None:
    """自走が教師強制で学習した係数**そのもの**を使う (D-44)。

    2段で測る。片方だけでは空虚になる:

    1. **同一性**: ``FreeRunResult.coefficients`` が
       ``TeacherForcedReadout.coefficients`` と同じオブジェクト
       (自走のたびに学習し直す実装 = 仕様 §5 禁止する構造1 はここで落ちる)
    2. **再学習を挟むと値が変わる**: 別の alpha で学習し直した係数を渡して
       自走させると予測列が変わる。1. だけだと「係数を無視して定数を吐く
       実装」でも緑になる
    """
    config = small_config()
    entry = lorenz_task_entry(config)
    outcome = run_free_run(config, entry, 0)

    assert outcome.result.coefficients is outcome.readout.coefficients

    # 2. 同じ設計行列・同じ訓練行で alpha だけ変えて学習し直す。
    readout = outcome.readout
    split = readout.plan.split
    refit = fit_ridge(
        readout.design.phi[split.train.start : split.train.stop],
        readout.plan.task.y[split.train.start : split.train.stop],
        readout.alpha * 1.0e6,
        bias_column=readout.design.bias_column,
    )
    assert not np.array_equal(refit, readout.coefficients), (
        "alpha を変えても係数が動かないので、この検査は空虚です"
    )

    from rc_basics_lab.readout.autoregressive import free_run

    reservoir = build_reservoir(
        chaos_reservoir_config(config.base),
        make_rng(config.base.seeds, SeedStream.RESERVOIR, 0),
        n_inputs=readout.plan.task.n_inputs,
    )
    switch = outcome.switch_index
    relearned = free_run(
        esn_state_updater(reservoir),
        FREE_RUN_SPEC,
        refit,
        readout.plan.states[switch],
        outcome.result.inputs[0],
        config.freerun.free_run_steps,
    )
    assert not np.allclose(
        relearned.predictions[: outcome.result.n_completed],
        outcome.result.predictions[: outcome.result.n_completed],
    ), "係数を差し替えても予測が変わりません (係数が使われていません)"


def test_free_run_readout_matches_the_one_step_selection() -> None:
    """自走の read-out が 4-A の ESN 手法とまったく同じ alpha を選ぶ。

    ``fit_teacher_forced`` は 01 の ``_evaluate`` (private) を呼ばず
    ``select_alpha`` -> ``fit_ridge`` を自分で並べている。経路が2本ある以上、
    「同じ選択になる」ことは主張ではなく実測で固定する。
    """
    config = small_config()
    entry = lorenz_task_entry(config)
    readout = fit_teacher_forced(config, entry, 0)
    rows = [
        row
        for row in run_onestep(config)
        if row.task == TASK_NAME_LORENZ
        and row.method == ESN_METHOD
        and row.replicate == 0
    ]
    assert len(rows) == 1
    assert rows[0].alpha == readout.alpha
    assert rows[0].t0 == readout.plan.t0


def test_free_run_feeds_its_own_output_back_as_the_next_input() -> None:
    """自走の入力列が1つ前の予測そのものである (閉ループの実体)。"""
    config = small_config()
    outcome = run_free_run(config, lorenz_task_entry(config), 0)
    completed = outcome.result.n_completed
    assert completed >= 2
    np.testing.assert_allclose(
        outcome.result.inputs[1:completed],
        outcome.result.predictions[: completed - 1],
        rtol=0.0,
        atol=0.0,
    )
    assert outcome.truth.shape == (
        config.freerun.free_run_steps,
        outcome.result.predictions.shape[1],
    )


def test_free_run_uses_step_not_run() -> None:
    """自走が ``ESN.step`` の逐次ループである (仕様 §5 禁止する構造8)。

    ``ESN.run`` は入力系列が既知の区間専用で、自走では ``u[t+1]`` が
    ``y_hat[t]`` に依存するので使えない。自走中に ``run`` が呼ばれたら落ちる。
    """
    reservoir = ESN(
        ESNConfig(n_units=5, topology=ErdosRenyiConfig(density=1.0)),
        np.random.default_rng(0),
        n_inputs=1,
    )
    calls = {"step": 0, "run": 0}
    original_step = ESN.step

    def counting_step(
        self: ESN,
        x: FloatArray,
        u: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> FloatArray:
        calls["step"] += 1
        return original_step(self, x, u, None)

    def forbidden_run(*args: object, **kwargs: object) -> FloatArray:
        calls["run"] += 1
        raise AssertionError("自走で ESN.run が呼ばれました (仕様 §5 禁止する構造8)")

    from rc_basics_lab.readout.autoregressive import free_run

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ESN, "step", counting_step)
        patch.setattr(ESN, "run", forbidden_run)
        result = free_run(
            esn_state_updater(reservoir),
            FREE_RUN_SPEC,
            np.zeros((1 + 1 + 5, 1)),
            reservoir.initial_state(),
            np.array([0.5]),
            7,
        )
    assert result.n_completed == 7
    assert calls == {"step": 7, "run": 0}


# --- 3. 確保軸3 (D-34) --------------------------------------------------------


def test_free_run_bounds_are_checked_before_allocation() -> None:
    """確保軸3 (``free_run_steps * n_units``) が確保より前に落ちる (D-34)。

    ``run_free_run`` は教師強制の状態行列を作る**前に**検査するので、巨大な
    ``free_run_steps`` でも即座に ``ValueError`` になる (実際に確保しに行けば
    メモリを食い潰してから落ちる)。
    """
    with pytest.raises(ValueError, match="n_units \\* n_steps が上限"):
        validate_free_run_bounds(10_000_000, 100)
    with pytest.raises(ValueError, match="n_units が上限"):
        validate_free_run_bounds(10, 10_000)
    with pytest.raises(ValueError, match="free_run_steps"):
        validate_free_run_bounds(0, 10)

    config = dataclasses.replace(
        small_config(), freerun=FreeRunConfig(warmup_steps=10, free_run_steps=10**8)
    )
    with pytest.raises(ValueError, match="上限"):
        run_free_run(config, lorenz_task_entry(config), 0)


def test_free_run_bound_reuses_the_existing_capacity_guard() -> None:
    """04 で新しい上限を作らず 03 の ``validate_state_matrix_bounds`` を使う。

    上限が2か所にあると片方だけ緩められる。実測は「呼ばれていること」で取る。
    """
    calls: list[tuple[int, int]] = []

    def spy(n_units: int, n_steps: int) -> None:
        calls.append((n_units, n_steps))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(freerun_tasks_module, "validate_state_matrix_bounds", spy)
        validate_free_run_bounds(40, 20)
    assert calls == [(20, 40)]


def test_standardization_window_must_stay_inside_training() -> None:
    """標準化の推定区間が訓練区間を越えたら ``ValueError`` (D-41)。"""
    split = make_split(
        SplitConfig(washout=10, max_start_offset=0), 100, 10, np.random.default_rng(0)
    )
    validate_standardization_window(split.train.stop, split)
    with pytest.raises(ValueError, match="標準化係数の推定区間"):
        validate_standardization_window(split.train.stop + 1, split)

    config = dataclasses.replace(
        small_config(),
        lorenz=LorenzConfig(length=600, integration_burn_in=100, standardize_steps=600),
    )
    with pytest.raises(ValueError, match="標準化係数の推定区間"):
        run_free_run(config, lorenz_task_entry(config), 0)


def test_run_free_run_rejects_a_window_that_runs_past_the_series() -> None:
    """自走に必要な行が系列の外へ出る設定は ``ValueError``。"""
    config = dataclasses.replace(
        small_config(), freerun=FreeRunConfig(warmup_steps=10, free_run_steps=100_000)
    )
    with pytest.raises(ValueError, match=r"上限|テスト区間の先"):
        run_free_run(config, lorenz_task_entry(config), 0)


# --- 3b. 確保軸10 (逐次実行の本数、reviewer-security 実測) -------------------


def test_onestep_rejects_the_sequential_run_count_before_any_replicate_runs() -> None:
    """4-A の確保軸10 (課題数 x ``base.n_replicates``) が**確保より前に**落ちる。

    ``small_config`` は合法な既存の軸 (課題長・alpha 格子・n_lags 格子・ESN
    ユニット数など) をすべて通過する設定である。``base.n_replicates`` だけを
    上限超に変えても、他の軸検査はどれも引っかからない (この検査が無ければ
    ``run_task`` に到達して ESN のシミュレーションを回し始める)。
    """
    huge = dataclasses.replace(
        small_config(),
        base=dataclasses.replace(small_config().base, n_replicates=10_000),
    )
    with pytest.raises(ValueError, match="逐次実行の本数が上限"):
        run_onestep(huge)


def test_freerun_rejects_the_sequential_run_count_before_any_replicate_runs() -> None:
    """4-B の確保軸10 (課題数 x ``base.n_replicates`` x 手法数) が**確保より前に**
    落ちる。

    4-B は手法ごとに独立な閉ループを回すため、同じ ``base.n_replicates`` でも
    4-A より3倍多く数える。``small_config`` の他の軸 (``stats_steps`` 等) は
    そのままなので、この検査だけが効いていることの実測になる。
    """
    config = small_config()
    lyapunov = estimate_lorenz_lyapunov(config)
    huge = dataclasses.replace(
        config, base=dataclasses.replace(config.base, n_replicates=10_000)
    )
    with pytest.raises(ValueError, match="逐次実行の本数が上限"):
        run_freerun_experiment(huge, lyapunov)


def test_sequential_run_count_reuses_the_existing_capacity_guard() -> None:
    """04 で新しい上限を作らず ``experiment/capacity.py`` の検査を使う。

    上限が複数箇所にあると片方だけ緩められる。実測は「呼ばれていること」で
    取る (``test_free_run_bound_reuses_the_existing_capacity_guard`` と同じ
    流儀)。
    """
    calls: list[int] = []

    def spy(n_runs: int) -> None:
        calls.append(n_runs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(freerun_module, "validate_sequential_run_count", spy)
        run_onestep(small_config())
    assert calls == [4]  # 2課題 x base.n_replicates=2


# --- 4. D-48 と D-36 の境界 ---------------------------------------------------


def test_esn_state_updater_passes_the_rng_when_state_noise_is_positive() -> None:
    """``state_noise > 0`` の自走は rng を ``ESN.step`` へ渡す (D-36)。

    02 の伝播器 (``esn_propagator``) が決定的でなければならない (D-48) のは、
    条件付き Lyapunov 指数が「同じ軌道のまわりの摂動の成長率」を測るから
    であって、「ESN は常に決定的に回す」という規則ではない。自走は軌道を
    **作る**呼び出しなので、学習時に入れたノイズと同じ分布が乗る必要がある。
    """
    noisy = ESNConfig(
        n_units=5, topology=ErdosRenyiConfig(density=1.0), state_noise=1.0e-2
    )
    reservoir = ESN(noisy, np.random.default_rng(0), n_inputs=1)
    seen: list[object] = []
    original_step = ESN.step

    def recording_step(
        self: ESN,
        x: FloatArray,
        u: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> FloatArray:
        seen.append(rng)
        return original_step(self, x, u, rng)

    rng = np.random.default_rng(7)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ESN, "step", recording_step)
        updater = esn_state_updater(reservoir, rng)
        updater(reservoir.initial_state(), np.array([0.1]))
    assert seen == [rng], "自走で rng が ESN.step へ渡っていません (D-36)"

    with pytest.raises(ValueError, match="state_noise > 0 の自走には rng"):
        esn_state_updater(reservoir, None)


def test_noise_free_updater_does_not_need_an_rng() -> None:
    """``state_noise == 0`` なら rng なしで構築でき、決定的に回る。"""
    reservoir = ESN(
        ESNConfig(n_units=5, topology=ErdosRenyiConfig(density=1.0)),
        np.random.default_rng(0),
        n_inputs=1,
    )
    updater = esn_state_updater(reservoir)
    first = updater(reservoir.initial_state(), np.array([0.3]))
    second = updater(reservoir.initial_state(), np.array([0.3]))
    assert np.array_equal(first, second)


# --- Lyapunov の配線と成果物 ---------------------------------------------------


def test_lyapunov_is_estimated_from_a_single_reference_trajectory() -> None:
    """真の軌道は条件に依存しないので1回だけ積分する (仕様 §5 禁止する構造3)。

    ``run_and_report_onestep`` を1回回したときの ``integrate_lorenz`` の
    呼び出し回数を数える。条件ループの中で積分し直す実装はここで落ちる。
    """
    config = small_config()
    calls: list[int] = []

    def counting(cfg: LorenzConfig, x0: FloatArray, n_samples: int) -> FloatArray:
        calls.append(1)
        # 差し替えるのは**配線層が import 時に束縛した参照**なので、定義元
        # (tasks.chaotic) から取った関数をそのまま呼ぶ (04a T2 実装メモ8)。
        return integrate_lorenz(cfg, x0, n_samples)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(freerun_module, "integrate_lorenz", counting)
        estimate_lorenz_lyapunov(config)
    assert calls == [1]


def test_estimate_lorenz_lyapunov_reports_the_sampling_interval() -> None:
    """``ctx.dt`` に ``rk4_step * sample_interval`` が渡っている (D-43 の分母)。"""
    config = small_config()
    result = estimate_lorenz_lyapunov(config)
    expected = config.lorenz.rk4_step * config.lorenz.sample_interval
    assert result.params["dt"] == str(expected)
    assert result.scalars["lyapunov_per_time"] == pytest.approx(
        result.scalars["lyapunov_per_step"] / expected
    )


def test_committed_onestep_csv_shares_the_split_across_methods() -> None:
    """**コミット済みの** ``onestep.csv`` でも D-05 が成り立っている。

    縮小設定での検査 (同ファイルの
    ``test_onestep_shares_the_split_across_methods_within_a_replicate``) は経路を
    守るが、成果物そのものは見ていない。本番設定を再生成せずに ``results/`` だけ
    差し替えると、公平性が崩れた行が入り込みうる。
    """
    path = (
        Path(__file__).resolve().parents[1] / "results/04_chaotic_freerun/onestep.csv"
    )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "onestep.csv が空です"
    assert list(rows[0]) == list(CSV_COLUMNS)
    grouped: dict[tuple[str, str], set[tuple[str, ...]]] = {}
    for row in rows:
        grouped.setdefault((row["task"], row["replicate"]), set()).add(
            (row["t0"], row["n_train"], row["n_val"], row["n_test"])
        )
    for key, shapes in grouped.items():
        assert len(shapes) == 1, f"{key} で手法ごとに分割が違います: {shapes}"
    assert {row["method"] for row in rows} == {LINEAR, DELAY_LINE, ESN_METHOD}
    assert {row["task"] for row in rows} == {TASK_NAME_LORENZ, TASK_NAME_MACKEY_GLASS}
    # シードは10本以上 (要件書 受け入れ条件2)。
    assert len({row["replicate"] for row in rows}) >= 10


def test_chaos_artifacts_go_to_their_own_directory() -> None:
    """成果物は **04 専用のディレクトリ**に出す (D-51)。

    要件書は ``results/`` 直下と書いていたが、01 が ``results/`` 直下を使って
    いるので ``meta.json`` が衝突して黙って上書きされる (02・03 で同じ理由から
    実験ごとのディレクトリへ分けた)。レジストリの既定出力先と実験スクリプトの
    ``DEFAULT_OUT`` の一致は ``tests/test_main.py`` が全実験について回すので、
    ここでは 04 が**他の実験と混ざらない**ことを固定する。

    **親ディレクトリは固定しない。** 既定は ``scratch/`` (手元の試行) で、
    ``results/`` へ書くのは ``--out`` を明示する ``make figures-04`` だけである。
    D-51 が守っているのは「衝突しないこと」であって、``results/`` という綴り
    ではない。
    """
    import main

    assert main.EXPERIMENTS["04"].out_dir.name == "04_chaotic_freerun"
    out_dirs = [spec.out_dir for spec in main.EXPERIMENTS.values()]
    assert len(set(out_dirs)) == len(out_dirs), "既定の出力先が重複しています"


def test_run_and_report_onestep_writes_the_declared_artifacts(tmp_path: Path) -> None:
    """成果物が ``ONESTEP_ARTIFACTS`` と一致し、列順が 01 の ``CSV_COLUMNS``。"""
    config = small_config()
    paths = run_and_report_onestep(config, tmp_path)
    assert tuple(path.name for path in paths) == ONESTEP_ARTIFACTS
    header = (tmp_path / ONESTEP_CSV).read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(CSV_COLUMNS)

    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["config"]["name"] == config.name
    assert meta["n_rows"] == len(run_onestep(config))
    assert meta["lorenz_dt"] == pytest.approx(
        config.lorenz.rk4_step * config.lorenz.sample_interval
    )
    assert "lyapunov_per_time" in meta["lyapunov"]
    assert set(meta["wall_time_breakdown"]) == {"lyapunov_s", "onestep_s"}


# --- 5. 閉ループの組み立て (3手法) -------------------------------------------


@pytest.mark.parametrize("method", [LINEAR, DELAY_LINE, ESN_METHOD])
def test_closed_loop_design_matches_the_teacher_forced_row(method: str) -> None:
    """閉ループの設計行列が教師強制の行と**一致する** (仕様 §5 禁止する構造2)。

    遅延線だけは学習時 (``DelayLineSpec``) と閉ループ時
    (``ReservoirSpec(include_input=False)`` + シフトレジスタ) で仕様の表現が
    違う。組む列が同じであることを主張ではなく実測で固定しないと、係数の並びが
    黙ってずれても「それらしい自走」が出てしまう。
    """
    config = small_config()
    entry = lorenz_task_entry(config)
    readout = fit_teacher_forced(config, entry, 0, method=method)
    plan = readout.plan
    switch = plan.split.test.start + config.freerun.warmup_steps - 1
    reservoir = (
        build_reservoir(
            entry.reservoir,
            make_rng(config.base.seeds, SeedStream.RESERVOIR, 0),
            n_inputs=plan.task.n_inputs,
        )
        if method == ESN_METHOD
        else None
    )
    loop = closed_loop_setup(readout, switch, esn=reservoir)
    design = build_design_matrix(
        loop.spec, plan.task.u[switch : switch + 1], loop.x0.reshape(1, -1)
    )
    np.testing.assert_allclose(design.phi[0], readout.design.phi[switch], rtol=0.0)
    assert design.phi.shape[1] == readout.coefficients.shape[0]


def test_delay_line_free_run_never_builds_a_reservoir() -> None:
    """遅延線の自走は ESN を1つも作らない (D-50 の2つ目の実例)。

    ``free_run`` が状態生成器を ``StateUpdater`` で受けているので、シフト
    レジスタでも同じ関数がそのまま動く。ESN の構築を「呼ばれたら落ちる」ものに
    差し替えて完走させる。

    教師強制の ``ReplicatePlan`` は**先に**作っておく —— 分割と ``t0`` を3手法で
    共有する (D-05) ために、plan は手法によらず ESN の状態行列を含むからである。
    ここが測るのは「閉ループの側がリザバーを作らないこと」で、そのために
    ``plan=`` で渡す経路 (3手法が1つの plan を共有する経路) をそのまま使う。
    """
    config = small_config()
    entry = lorenz_task_entry(config)
    plan = fit_teacher_forced(config, entry, 0, method=DELAY_LINE).plan

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("遅延線の自走が ESN を作りました (D-50)")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ESN, "__init__", forbidden)
        outcome = run_free_run(config, entry, 0, method=DELAY_LINE, plan=plan)
    assert outcome.method == DELAY_LINE
    assert outcome.result.n_completed >= 1


def test_linear_free_run_state_updater_is_the_identity() -> None:
    """線形ベースラインは状態を持たない (記憶のない手法の閉ループ)。"""
    updater = passthrough_state_updater()
    state = np.array([0.25], dtype=np.float64)
    assert updater(state, np.array([9.0])) is state


def test_delay_line_state_updater_shifts_the_register() -> None:
    """シフトレジスタは ``D_in`` ずつずれ、先頭に今の入力が入る。"""
    updater = delay_line_state_updater(2)
    state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    np.testing.assert_allclose(
        updater(state, np.array([9.0, 8.0])), [9.0, 8.0, 1.0, 2.0]
    )


# --- 6. 実験 4-B (自走 + 有効予測時間 + 長時間統計) ---------------------------


def test_stats_axis_is_checked_before_any_free_run() -> None:
    """確保軸4 (``stats_steps``) が 4-B の入口で、自走より前に効く (D-34)。"""
    config = dataclasses.replace(
        small_config(),
        freerun=dataclasses.replace(config_freerun(), stats_steps=10**9),
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("上限検査より先に自走しています")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(freerun_module, "run_free_run", forbidden)
        with pytest.raises(ValueError, match="stats_steps が上限"):
            run_freerun_experiment(config, estimate_lorenz_lyapunov(config))


def config_freerun() -> FreeRunConfig:
    """``small_config`` の自走設定 (``dataclasses.replace`` の土台)。"""
    return small_config().freerun


def test_freerun_experiment_covers_tasks_methods_and_replicates() -> None:
    """4-B は (課題 x 手法 x レプリケート) を1本ずつ回す。"""
    config = small_config()
    results = run_freerun_experiment(config, estimate_lorenz_lyapunov(config))
    assert len(results.rows) == 2 * len(FREERUN_METHODS) * config.base.n_replicates
    assert {row.method for row in results.rows} == set(FREERUN_METHODS)
    assert all(row.stats_steps == config.freerun.stats_steps for row in results.rows)
    # 対照も同じ長さの自走を回す (「対照は原理的に不利」を主張だけにしない)。
    assert {row.free_run_steps for row in results.rows} == {
        config.freerun.free_run_steps
    }


def test_freerun_experiment_shares_one_plan_across_methods() -> None:
    """1レプリケートにつき ``ReplicatePlan`` は1個 (3手法で共有する)。

    共有しないと「同じ分割・同じ状態行列で比べた」が構造ではなく偶然になる。
    """
    config = small_config()
    calls: list[object] = []

    def spy(base: ExperimentConfig, task_entry: object, replicate: int) -> object:
        plan = plan_replicate(base, task_entry, replicate)  # type: ignore[arg-type]
        calls.append(plan)
        return plan

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(freerun_module, "plan_replicate", spy)
        run_freerun_experiment(config, estimate_lorenz_lyapunov(config))
    assert len(calls) == 2 * config.base.n_replicates


def test_only_lorenz_rows_carry_the_lyapunov_normalization() -> None:
    """lambda_max を推定してある系だけが Lyapunov 列を持つ (D-42)。

    推定していない量を他の系の値で埋めない。Mackey-Glass の行は
    ``lyapunov_time`` も ``valid_time_lyapunov`` も ``nan`` で、
    ``dt`` は MG 自身のサンプリング間隔である。
    """
    config = small_config()
    results = run_freerun_experiment(config, estimate_lorenz_lyapunov(config))
    for row in results.rows:
        if row.task == TASK_NAME_LORENZ:
            assert math.isfinite(row.lyapunov_time)
            assert math.isfinite(row.valid_time_lyapunov)
            assert row.dt == pytest.approx(
                config.lorenz.rk4_step * config.lorenz.sample_interval
            )
        else:
            assert math.isnan(row.lyapunov_time)
            assert math.isnan(row.valid_time_lyapunov)
            assert row.dt == pytest.approx(
                mg_params(config).rk4_step * mg_params(config).sample_interval
            )


def test_profile_rows_are_thinned_to_the_declared_cap() -> None:
    """確保軸6: 図に載せる点数が ``PROFILE_MAX_POINTS`` を超えない。"""
    config = dataclasses.replace(
        small_config(), freerun=dataclasses.replace(config_freerun(), stats_steps=400)
    )
    results = run_freerun_experiment(config, estimate_lorenz_lyapunov(config))
    grouped: dict[tuple[str, str, str], int] = {}
    for row in results.profile_rows:
        key = (row.task, row.kind, row.source)
        grouped[key] = grouped.get(key, 0) + 1
    assert grouped, "profile 行が空です"
    assert max(grouped.values()) <= PROFILE_MAX_POINTS
    # 代表レプリケートは結果を見て選ばない (常に 0)。
    assert {row.replicate for row in results.profile_rows} == {PROFILE_REPLICATE}
    assert {row.method for row in results.profile_rows} == {ESN_METHOD}


def test_closer_than_surrogate_is_derived_from_the_two_distances() -> None:
    """``closer_than_surrogate`` が**2指標の比較そのもの**である (D-46)。

    成果物に対する受け入れ条件のテスト
    (``test_attractor_distance_separates_true_and_surrogate``) は**コミット済みの
    CSV** を読むので、判定を「常に True」に書き換える変異を検出できない
    (CSV は変異の前に生成されている)。計算を実際に回す側のガードをここに置く。
    対照の行は代替より近くならないので、常に True にする実装はここで落ちる。
    """
    config = small_config()
    results = run_freerun_experiment(config, estimate_lorenz_lyapunov(config))
    for row in results.rows:
        expected = (
            row.return_map_distance < row.return_map_distance_surrogate
            and row.spectrum_distance < row.spectrum_distance_surrogate
        )
        assert row.closer_than_surrogate is expected, row
    assert not all(row.closer_than_surrogate for row in results.rows), (
        "全行が代替より近いので、この検査は空虚です (対照が混ざっていない)"
    )


def test_pipeline_actually_calls_the_real_shuffled_surrogate() -> None:
    """呼び出し配線: freerun.py が**本物の** ``shuffled_surrogate`` を呼ぶ (D-46)。

    ``test_closer_than_surrogate_is_derived_from_the_two_distances`` は
    ``closer_than_surrogate`` が2つの距離から正しく導出されているかしか見ておらず、
    その2つの距離が「本物のシャッフル代替」に対して計算されたものかは見ていない。
    freerun.py の呼び出し側 (``shuffled_surrogate(...)``) だけを no-op (シャッフル
    せず先頭 n_samples をそのまま返す) に差し替える変異は、このテストが無いと
    検出できない (``attractor.py`` 側の直接テストは呼び出しバインディングの破損を
    対象にしない)。本物へスパイをかぶせ、(1) 実際に呼ばれていること (2) 返り値が
    no-op の結果と一致しないこと (= 実際にシャッフルされていること) の両方を測る。
    """
    config = small_config()
    calls: list[tuple[FloatArray, FloatArray]] = []

    def spy(series: FloatArray, rng: np.random.Generator, n_samples: int) -> FloatArray:
        result = shuffled_surrogate(series, rng, n_samples)
        noop = np.asarray(series, dtype=float)[: min(n_samples, len(series))]
        calls.append((noop, result))
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(freerun_module, "shuffled_surrogate", spy)
        run_freerun_experiment(config, estimate_lorenz_lyapunov(config))

    assert calls, "shuffled_surrogate が1度も呼ばれていません"
    for noop, surrogate in calls:
        assert not np.array_equal(surrogate, noop), (
            "shuffled_surrogate の返り値が no-op (先頭 n_samples をそのまま返す) "
            "と一致しています (シャッフルされていない可能性)"
        )


def test_sign_test_p_value_matches_the_closed_form() -> None:
    """符号検定の p 値 (D-46 の「有意に近い」の根拠)。"""
    assert sign_test_p_value(10, 10) == pytest.approx(1.0 / 1024.0)
    assert sign_test_p_value(10, 0) == pytest.approx(1.0)
    assert sign_test_p_value(1, 1) == pytest.approx(0.5)


# --- 7. 成果物に対する受け入れ条件 (図は見ない) -------------------------------


CHAOS_RESULTS = Path(__file__).resolve().parents[1] / "results/04_chaotic_freerun"


def committed_freerun_rows() -> list[dict[str, str]]:
    """コミット済みの ``freerun.csv`` (実験も図も走らせない)。"""
    with (CHAOS_RESULTS / FREERUN_CSV).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "freerun.csv が空です"
    assert list(rows[0]) == list(FREERUN_CSV_COLUMNS)
    return rows


def test_valid_time_rows_cover_at_least_ten_seeds() -> None:
    """**受け入れ条件2**: 有効予測時間が Lyapunov 正規化・10シード以上。

    生のステップ数だけで報告していないこと (仕様 §5 禁止する構造5) も同時に
    測る —— ``valid_time_lyapunov`` が有限で、``valid_time_steps`` と
    ``lyapunov_time`` から復元できることまで確かめる。
    """
    rows = [
        row
        for row in committed_freerun_rows()
        if row["task"] == TASK_NAME_LORENZ and row["method"] == ESN_METHOD
    ]
    assert len({row["replicate"] for row in rows}) >= 10
    for row in rows:
        steps = int(row["valid_time_steps"])
        expected = steps * float(row["dt"]) / float(row["lyapunov_time"])
        assert float(row["valid_time_lyapunov"]) == pytest.approx(expected)
        assert math.isfinite(float(row["valid_time_lyapunov"]))
        assert row["valid_time_censored"] in {"True", "False"}
    values = sorted(float(row["valid_time_lyapunov"]) for row in rows)
    assert values[0] > 1.0, f"有効予測時間が 1 Lyapunov 時間に届いていません: {values}"

    # 成果物は**変異の前に**生成されているので、CSV を読むだけでは
    # 「正規化をやめて生のステップ数を書く」変異を検出できない。計算を実際に
    # 回す側の検査をここに足す (仕様 §5 禁止する構造5)。
    config = small_config()
    computed = [
        item
        for item in run_freerun_experiment(
            config, estimate_lorenz_lyapunov(config)
        ).rows
        if item.task == TASK_NAME_LORENZ
    ]
    assert computed
    for item in computed:
        assert item.valid_time_lyapunov == pytest.approx(
            item.valid_time_steps * item.dt / item.lyapunov_time
        )
        assert item.valid_time == pytest.approx(item.valid_time_steps * item.dt)
        assert item.valid_time_lyapunov != item.valid_time_steps


def test_censored_valid_time_propagates_to_the_sensitivity_summary() -> None:
    """統合テスト: 打ち切りが実際に起きたとき、行の censored フラグと
    ``ValidTimeSensitivity.n_censored`` の両方に正しく伝播する (D-43)。

    既存のテストは (a) 純関数 ``valid_time_from_errors`` の単体テスト
    (test_experiment_attractor.py) と (b) コミット済み CSV を読むだけの型チェック
    (``test_valid_time_rows_cover_at_least_ten_seeds``) のみで、``run_freerun_
    experiment`` / ``ValidTimeSensitivity`` (``n_censored`` 集計) を通した統合
    テストで打ち切りが実際に発生するケースは無かった (grep で ``n_censored`` /
    ``ValidTimeSensitivity`` を参照するテストは0件だった)。``valid_time_threshold``
    を極端に大きくし、感度表の格子 (``VALID_TIME_THRESHOLD_GRID``) もその1点に
    差し替えて、誤差が自走長まで一度も閾値を超えないようにする (=打ち切りを
    意図的に発生させる)。censored フラグを常に False にする変異はここで落ちる。
    """
    huge_threshold = 1.0e6
    config = dataclasses.replace(
        small_config(),
        freerun=dataclasses.replace(
            config_freerun(), valid_time_threshold=huge_threshold
        ),
    )
    with pytest.MonkeyPatch.context() as patch:
        # 格子は「評価する側」(freerun) と「畳む側」(freerun_rows) の両方が
        # 読む。**両方を差し替える** —— 片方だけだと、評価が 1 点で要約が
        # 4 点になり IndexError で落ちる (分割したときに実際そうなった)。
        patch.setattr(freerun_module, "VALID_TIME_THRESHOLD_GRID", (huge_threshold,))
        patch.setattr(
            freerun_rows_module, "VALID_TIME_THRESHOLD_GRID", (huge_threshold,)
        )
        results = run_freerun_experiment(config, estimate_lorenz_lyapunov(config))

    assert results.rows, "行が空です"
    censored_rows = [row for row in results.rows if row.valid_time_censored]
    assert censored_rows, "意図的に閾値を極端にしたのに打ち切りが1件も発生しません"
    assert len(censored_rows) == len(results.rows), (
        "全行が打ち切られるはずの設定なのに一部しか打ち切られていません"
    )

    assert results.sensitivity, "感度表 (ValidTimeSensitivity) が空です"
    total_n_censored = 0
    for entry in results.sensitivity:
        rows_in_group = [
            row
            for row in results.rows
            if row.task == entry.task and row.method == entry.method
        ]
        assert rows_in_group, entry
        assert entry.n_censored == len(rows_in_group), (
            "打ち切り行数が ValidTimeSensitivity.n_censored に伝播していません: "
            f"{entry}"
        )
        total_n_censored += entry.n_censored
    assert total_n_censored == len(censored_rows)


def test_attractor_distance_separates_true_and_surrogate() -> None:
    """**受け入れ条件1 / 5**: 自走がアトラクタを再現する (D-46)。**図では測らない**。

    2指標 (リターンマップの点集合距離・パワースペクトルの全変動距離) の両方が
    **真の軌道のシャッフル代替より小さい**ことを、全シードについて要求する。
    10 本が全部同じ向きなら片側符号検定の p は 1/1024 で、「有意に近い」と
    言える。対照 (線形・遅延線) では成立しないことも同時に測る —— 成立して
    しまうなら指標が自走の質を見ていない。
    """
    rows = committed_freerun_rows()
    for task in {row["task"] for row in rows}:
        esn_rows = [
            row for row in rows if row["task"] == task and row["method"] == ESN_METHOD
        ]
        assert len(esn_rows) >= 10
        for row in esn_rows:
            assert float(row["return_map_distance"]) < float(
                row["return_map_distance_surrogate"]
            ), row
            assert float(row["spectrum_distance"]) < float(
                row["spectrum_distance_surrogate"]
            ), row
            assert row["closer_than_surrogate"] == "True"
        assert sign_test_p_value(len(esn_rows), len(esn_rows)) <= 0.01
        controls = [
            row for row in rows if row["task"] == task and row["method"] != ESN_METHOD
        ]
        assert not any(row["closer_than_surrogate"] == "True" for row in controls), (
            "対照でもアトラクタ再現が成立しています (指標が自走の質を見ていない)"
        )


def test_freerun_csv_reports_both_long_run_metrics() -> None:
    """**受け入れ条件5**: 長時間統計が**2本**の列として成果物に在る (D-46)。"""
    header = set(FREERUN_CSV_COLUMNS)
    assert {
        "return_map_distance",
        "return_map_distance_surrogate",
        "spectrum_distance",
        "spectrum_distance_surrogate",
        "closer_than_surrogate",
        "n_stats_samples",
    } <= header
    rows = committed_freerun_rows()
    assert all(int(row["n_stats_samples"]) > 0 for row in rows)


def test_onestep_gap_is_small_and_freerun_gap_is_large() -> None:
    """**受け入れ条件3**: 教師強制では差が小さく、自走では対照が成立しない。

    **両方向を1本で測る**。片方だけのテストにすると、「1ステップ先でも ESN が
    圧勝する」設定 (Delta t を大きく取った較正の落選値) でも緑になってしまう。

    - 教師強制 (``onestep.csv``): ESN と遅延線の NRMSE の比が 1 桁の内側
    - 自走 (``freerun.csv``): ESN の有効予測時間が対照の 10 倍以上
    """
    with (CHAOS_RESULTS / ONESTEP_CSV).open(encoding="utf-8", newline="") as handle:
        onestep = list(csv.DictReader(handle))
    freerun = committed_freerun_rows()

    def mean_of(rows: list[dict[str, str]], column: str) -> float:
        values = [float(row[column]) for row in rows]
        return sum(values) / len(values)

    for task in {row["task"] for row in freerun}:
        esn_nrmse = mean_of(
            [
                row
                for row in onestep
                if row["task"] == task and row["method"] == ESN_METHOD
            ],
            "nrmse",
        )
        delay_nrmse = mean_of(
            [
                row
                for row in onestep
                if row["task"] == task and row["method"] == DELAY_LINE
            ],
            "nrmse",
        )
        ratio = max(esn_nrmse, delay_nrmse) / min(esn_nrmse, delay_nrmse)
        assert ratio < 10.0, (
            f"{task}: 教師強制で ESN と遅延線の差が 1 桁を超えています "
            f"(ESN={esn_nrmse:.3g} / 遅延線={delay_nrmse:.3g})"
        )
        esn_time = mean_of(
            [
                row
                for row in freerun
                if row["task"] == task and row["method"] == ESN_METHOD
            ],
            "valid_time_steps",
        )
        control_time = max(
            mean_of(
                [
                    row
                    for row in freerun
                    if row["task"] == task and row["method"] == method
                ],
                "valid_time_steps",
            )
            for method in (LINEAR, DELAY_LINE)
        )
        assert esn_time > 10.0 * control_time, (
            f"{task}: 自走で ESN と対照の差が 10 倍に届いていません "
            f"(ESN={esn_time:.1f} / 対照={control_time:.1f} ステップ)"
        )


def test_committed_artifacts_match_the_declared_list() -> None:
    """**受け入れ条件6**: 1コマンドで出る成果物が宣言と一致する。"""
    from rc_basics_lab.experiment.freerun_pipeline import FREERUN_ARTIFACTS

    for name in FREERUN_ARTIFACTS:
        assert (CHAOS_RESULTS / name).exists(), name
    total = sum(path.stat().st_size for path in CHAOS_RESULTS.glob("*.csv"))
    assert total < 5 * 1024 * 1024, total


def test_the_literature_vpt_threshold_is_in_the_sensitivity_grid() -> None:
    """文献比較に使う閾値が感度格子に入っていること (D-101)。

    一次資料は Platt et al. (2022) *Neural Networks* 153:530 で、VPT は
    ``RMSE(t) = sqrt((1/D) sum_i [(u_i^f - u_i)/sigma_i]^2) > eps`` を初めて
    超える時刻、``eps`` は「arbitrarily to 0.3」と定義されている。
    **平方根が入っている**ので、こちらの NRMSE 比 (D-43) と同じ次元であり、
    換算は要らない —— 比較点は**格子に元から入っていた 0.3** である。

    最初にこの検査を書いたときは二次情報 (arXiv:2508.06730 の要約) を信じて
    「正規化二乗差 0.4」だと思い込み、``sqrt(0.4)`` を格子に足していた。
    **一次資料を読んだら定義が違った。** 数値主張は一次資料で裏を取る。

    実測 (本番、Lorenz / ESN / 10 レプリケート): 閾値 0.3 で中央値 4.74、
    採用値 0.4 で 4.83 [1/lambda_max]。
    """
    assert pytest.approx(0.3) == LITERATURE_VPT_THRESHOLD
    assert LITERATURE_VPT_THRESHOLD in VALID_TIME_THRESHOLD_GRID, (
        "文献比較に使う閾値が格子から外れています。"
        "外すと、文献と並べるときに使う列が消えます (D-101)。"
    )
    # 換算 (sqrt(0.4)) は不要。戻ってきたら気づけるようにしておく。
    assert LITERATURE_VPT_THRESHOLD < 0.4

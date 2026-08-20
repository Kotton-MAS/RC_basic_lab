"""実験 4-A と自走の入口の検査 (D-31 / D-34 / D-36 / D-41 / D-44).

守るのは4系統。

1. **01 の経路をそのまま通していること** (D-31)。4-A は ``run_task`` を呼ぶ
   だけで、公平性 (D-04 / D-05 / D-08) を書き写していない
2. **自走が教師強制の係数をそのまま使うこと** (D-44)。同一性 **と**
   「学習し直すと値が変わる」の両方を測る (片方だけだと空虚になる)
3. **確保軸3** が自走の確保より前に効くこと (D-34)
4. **D-48 と D-36 の境界**。伝播器 (02) は決定的、自走 (04) は
   ``state_noise > 0`` なら rng を渡す
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from rc_basics_lab.config import (
    Chaos04Config,
    ESNConfig,
    ExperimentConfig,
    FreeRunConfig,
    LorenzConfig,
    MackeyGlassConfig,
    MackeyGlassStandardizeConfig,
    RidgeConfig,
    SplitConfig,
)
from rc_basics_lab.experiment import freerun as freerun_module
from rc_basics_lab.experiment.freerun import (
    CHAOS_ESN_SECTION,
    FREE_RUN_SPEC,
    ONESTEP_ARTIFACTS,
    ONESTEP_CSV,
    chaos_esn_config,
    chaos_task_entries,
    esn_state_updater,
    estimate_lorenz_lyapunov,
    fit_teacher_forced,
    lorenz_task_entry,
    run_and_report_onestep,
    run_free_run,
    run_onestep,
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
)
from rc_basics_lab.experiment.split import make_split
from rc_basics_lab.readout.design import ReservoirSpec
from rc_basics_lab.readout.ridge import fit_ridge
from rc_basics_lab.reservoir.esn import ESN
from rc_basics_lab.seeds import SeedStream, make_rng
from rc_basics_lab.tasks.chaotic import TASK_NAME_LORENZ
from rc_basics_lab.tasks.mackey_glass import TASK_NAME as TASK_NAME_MACKEY_GLASS
from rc_basics_lab.types import FloatArray


def small_config(**overrides: object) -> Chaos04Config:
    """秒未満で 4-A と自走を回せる縮小設定 (**構造は本番と同じ**)。"""
    base = ExperimentConfig(
        name="chaos-freerun-test",
        n_replicates=2,
        split=SplitConfig(washout=30, max_start_offset=10),
        ridge=RidgeConfig(alpha_grid=(1.0e-6, 1.0e-3), n_lags_grid=(2, 4)),
        mackey_glass=MackeyGlassConfig(length=600, integration_burn_in=100),
        esn_mackey_glass=ESNConfig(
            n_units=20, leak_rate=0.5, input_scale=0.5, density=0.5
        ),
    )
    defaults: dict[str, object] = {
        "base": base,
        "lorenz": LorenzConfig(
            length=600, integration_burn_in=100, standardize_steps=150
        ),
        "mackey_glass": MackeyGlassStandardizeConfig(standardize_steps=150),
        "freerun": FreeRunConfig(warmup_steps=10, free_run_steps=40),
    }
    return Chaos04Config(**{**defaults, **overrides})  # type: ignore[arg-type]


# --- 1. 01 の経路をそのまま通す (D-31) ---------------------------------------


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
    assert CHAOS_ESN_SECTION == "esn_mackey_glass"
    assert chaos_esn_config(config.base) is getattr(config.base, CHAOS_ESN_SECTION)
    for entry in chaos_task_entries(config):
        assert entry.esn is chaos_esn_config(config.base)


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

    reservoir = ESN(
        chaos_esn_config(config.base),
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
        ESNConfig(n_units=5, density=1.0), np.random.default_rng(0), n_inputs=1
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

    config = small_config(freerun=FreeRunConfig(warmup_steps=10, free_run_steps=10**8))
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
        patch.setattr(freerun_module, "validate_state_matrix_bounds", spy)
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

    config = small_config(
        lorenz=LorenzConfig(length=600, integration_burn_in=100, standardize_steps=600)
    )
    with pytest.raises(ValueError, match="標準化係数の推定区間"):
        run_free_run(config, lorenz_task_entry(config), 0)


def test_run_free_run_rejects_a_window_that_runs_past_the_series() -> None:
    """自走に必要な行が系列の外へ出る設定は ``ValueError``。"""
    config = small_config(
        freerun=FreeRunConfig(warmup_steps=10, free_run_steps=100_000)
    )
    with pytest.raises(ValueError, match=r"上限|テスト区間の先"):
        run_free_run(config, lorenz_task_entry(config), 0)


# --- 4. D-48 と D-36 の境界 ---------------------------------------------------


def test_esn_state_updater_passes_the_rng_when_state_noise_is_positive() -> None:
    """``state_noise > 0`` の自走は rng を ``ESN.step`` へ渡す (D-36)。

    02 の伝播器 (``esn_propagator``) が決定的でなければならない (D-48) のは、
    条件付き Lyapunov 指数が「同じ軌道のまわりの摂動の成長率」を測るから
    であって、「ESN は常に決定的に回す」という規則ではない。自走は軌道を
    **作る**呼び出しなので、学習時に入れたノイズと同じ分布が乗る必要がある。
    """
    noisy = ESNConfig(n_units=5, density=1.0, state_noise=1.0e-2)
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
        ESNConfig(n_units=5, density=1.0), np.random.default_rng(0), n_inputs=1
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
    original = freerun_module.integrate_lorenz  # type: ignore[attr-defined]

    def counting(*args: object, **kwargs: object) -> FloatArray:
        calls.append(1)
        return original(*args, **kwargs)  # type: ignore[arg-type]

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

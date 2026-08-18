"""03 の配線層 (``experiment/capacity.py``) の検査.

ここが守るのは「測定装置 (``diagnostics/``) は正しいのに、渡し方を間違えて
結果が静かに壊れる」系の事故である。3a のレビューで実際に踏んだものだけを
並べてある。

- **D-35**: ``X`` を読み取り専用にしてから診断へ渡す。``CapacityProblem`` は
  ``X`` のビューを持ち ``gram`` はスナップショットなので、構築後に ``X`` を
  書き換えると例外も警告もなく desync する (3a の実測で容量が 1.25e8)。
  ``CapacityProblem`` が塞げるのは自分が持つビューだけで**元の ``X`` は
  塞げない**ため、呼び出し側で塞ぐしかない。
- **D-36**: ``state_noise`` は既定値つきキーワードで足し、``ESN.run`` には
  常に ``rng`` を渡す。``state_noise=0`` では乱数を1個も引かないので、02 の
  成果物はバイト単位で不変である —— それを実測で固定する。
- **D-37**: サロゲートの ``ctx.seed`` は1個を全条件で共有する (共通乱数法)。
- **§10-1**: ``import rc_basics_lab.diagnostics.ipc as m`` は**モジュールでは
  なく関数**を返す (``diagnostics/__init__.py`` が関数 ``ipc`` を再エクスポート
  しているため)。3a のレビュー中に実際に踏み、変異試験が偽の緑になった。
  monkeypatch は**呼び出し側のモジュール属性**
  (``rc_basics_lab.experiment.capacity.ipc``) を差し替えて行う。
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

from rc_basics_lab.config import (
    Capacity03Config,
    CapacityDriveConfig,
    CapacityReservoirConfig,
    DriveConfig,
    Esp02Config,
    EspSeedConfig,
    IpcConfig,
    MemoryCapacityConfig,
    ReservoirSweepConfig,
    esp_stream_seed,
)
from rc_basics_lab.diagnostics.base import DiagnosticContext, DiagnosticResult
from rc_basics_lab.experiment.capacity import (
    EXPERIMENT_CONSERVATION,
    EXPERIMENT_IPC_SWEEP,
    EXPERIMENT_MC_SWEEP,
    CapacityCondition,
    evaluate_capacity_condition,
    ipc_config_for,
)
from rc_basics_lab.experiment.esp import (
    make_initial_states,
    simulate_condition,
    simulate_reference_trajectory,
)
from rc_basics_lab.seeds import SeedStream, make_rng_for
from rc_basics_lab.types import FloatArray

CAPACITY_MODULE = "rc_basics_lab.experiment.capacity"
"""monkeypatch の対象。**呼び出し側のモジュール属性**を差し替える (§10-1)。"""


def base_config() -> Capacity03Config:
    """秒未満で1条件を回せる縮小設定 (構造は本番と同じ)。

    ``mc.max_delay`` / ``ipc.max_delay_by_degree`` を系列長に合わせて下げないと
    診断側が「系列が短すぎます」で ``ValueError`` になる。``drive.washout`` を
    最大遅延より**大きく**取ってあるのは、``t0 = max(washout, 最大遅延)``
    (D-24) の binding side を washout にして「washout を変えたら行が変わる」を
    実測できるようにするため。
    """
    return Capacity03Config(
        name="capacity-test",
        drive=CapacityDriveConfig(distribution="uniform", washout=40),
        reservoir=CapacityReservoirConfig(input_scale=1.0, density=0.3, n_replicates=1),
        mc=MemoryCapacityConfig(max_delay=20, n_surrogates=5),
        ipc=IpcConfig(
            max_delay_by_degree=(8, 4), n_surrogates=5, n_surrogate_targets=2
        ),
    )


def condition(
    experiment: str = EXPERIMENT_MC_SWEEP, **overrides: object
) -> CapacityCondition:
    """縮小条件1本。``overrides`` は ``dataclasses.replace`` に流す。"""
    base = CapacityCondition(
        experiment=experiment,
        rho=0.9,
        leak_rate=1.0,
        n_units=12,
        state_noise=0.0,
        sigma_u=0.3,
        n_steps=1200,
        replicate=0,
    )
    return dataclasses.replace(base, **overrides)


class _StateSpy:
    """診断へ渡された ``X`` / ``u`` / ``ctx`` を記録しつつ本物を呼ぶ。

    本物を呼ぶのは、行の組み立て (``DiagnosticResult`` のキー参照) まで含めて
    実経路を通したいため。偽の結果を返すスパイにすると「配線層が診断の返り値の
    どのキーを読むか」が固定されなくなる。
    """

    def __init__(self, wrapped: Callable[..., DiagnosticResult]) -> None:
        self._wrapped = wrapped
        self.states: list[FloatArray] = []
        self.inputs: list[FloatArray] = []
        self.contexts: list[DiagnosticContext] = []

    def __call__(
        self,
        X: FloatArray,
        u: FloatArray | None = None,
        y: FloatArray | None = None,
        *,
        ctx: DiagnosticContext | None = None,
        cfg: object = None,
    ) -> DiagnosticResult:
        self.states.append(X)
        assert u is not None
        self.inputs.append(u)
        assert ctx is not None
        self.contexts.append(ctx)
        return self._wrapped(X, u, ctx=ctx, cfg=cfg)


def _install_spies(monkeypatch: pytest.MonkeyPatch) -> tuple[_StateSpy, _StateSpy]:
    """MC / IPC の**呼び出し側属性**をスパイに差し替える (§10-1)。"""
    module = importlib.import_module(CAPACITY_MODULE)
    mc_spy = _StateSpy(module.memory_capacity)
    ipc_spy = _StateSpy(module.ipc)
    monkeypatch.setattr(f"{CAPACITY_MODULE}.memory_capacity", mc_spy)
    monkeypatch.setattr(f"{CAPACITY_MODULE}.ipc", ipc_spy)
    return mc_spy, ipc_spy


def test_diagnostics_ipc_module_and_function_are_both_reachable() -> None:
    """``rc_basics_lab.diagnostics.ipc`` はモジュールでも関数でもある (§10-1)。

    ``diagnostics/__init__.py`` が関数 ``ipc`` を再エクスポートしているため、
    ``import rc_basics_lab.diagnostics.ipc as m`` で束縛されるのは**関数**で
    あり、``m.ipc`` を monkeypatch しても何も差し替わらない (3a のレビューで
    実際に踏み、変異試験が偽の緑になった)。この隠蔽は公開 API を壊さないため
    3b では直さない (仕様 §3.3-7) ので、代わりに両方が到達可能であることを
    ここで固定する。壊れたら、この罠を前提に書かれた monkeypatch がすべて
    無言で効かなくなる。
    """
    module = importlib.import_module("rc_basics_lab.diagnostics.ipc")
    assert isinstance(module, ModuleType)
    assert module.__name__ == "rc_basics_lab.diagnostics.ipc"

    import rc_basics_lab.diagnostics as diagnostics_package

    assert callable(diagnostics_package.ipc)
    assert not isinstance(diagnostics_package.ipc, ModuleType)
    assert diagnostics_package.ipc is module.ipc

    # 罠そのもの: ``import ... as m`` は関数を束縛する
    import rc_basics_lab.diagnostics.ipc as bound

    assert bound is diagnostics_package.ipc
    assert bound is not module


def test_states_are_read_only_before_capacity_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """診断へ渡す ``X`` は読み取り専用である (D-35)。

    ``CapacityProblem`` は ``X`` の**ビュー**を持ち ``gram`` は構築時点の
    スナップショットなので、構築後に ``X`` を書き換えると両者が例外も警告も
    なく desync する。``CapacityProblem`` 自身は ``problem.x`` しか塞げず
    元の ``X`` は塞げないため、塞ぐ責任は呼び出し側 (配線層) にある。

    変異試験: ``experiment/capacity.py`` の
    ``states.flags.writeable = False`` を消すとこのテストは落ちる (実測済み)。
    """
    mc_spy, ipc_spy = _install_spies(monkeypatch)
    evaluate_capacity_condition(base_config(), condition())

    assert len(mc_spy.states) == 1
    assert len(ipc_spy.states) == 1
    for states in (*mc_spy.states, *ipc_spy.states):
        assert states.flags.writeable is False
        with pytest.raises(ValueError):
            states[0, 0] = 1.0


def test_mc_and_ipc_share_the_same_state_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """参照軌道の生成は1条件につき1回だけで、MC と IPC は同じ ``X`` を見る。

    仕様 §5 が禁止する構造「条件ごとに X を2回作る (MC 用と IPC 用)」の guard。
    2回作ると実行時間が倍になるだけでなく、MC が見た系列と IPC が見た系列が
    違う CSV になり、同じ行に並べた意味が消える。
    """
    module = importlib.import_module(CAPACITY_MODULE)
    calls: list[CapacityCondition] = []
    real = module.simulate_reference_trajectory

    def counting(*args: object, **kwargs: object) -> object:
        calls.append(condition())
        return real(*args, **kwargs)

    monkeypatch.setattr(f"{CAPACITY_MODULE}.simulate_reference_trajectory", counting)
    mc_spy, ipc_spy = _install_spies(monkeypatch)

    config = base_config()
    conditions = (
        condition(EXPERIMENT_MC_SWEEP),
        condition(EXPERIMENT_IPC_SWEEP, rho=0.8, leak_rate=0.6),
        condition(EXPERIMENT_CONSERVATION, state_noise=0.02),
    )
    for item in conditions:
        evaluate_capacity_condition(config, item)

    assert len(calls) == len(conditions), "参照軌道の生成回数が条件数と一致しません"
    for mc_states, ipc_states in zip(mc_spy.states, ipc_spy.states, strict=True):
        assert mc_states is ipc_states, "MC と IPC が別の状態行列を見ています"
    for mc_u, ipc_u in zip(mc_spy.inputs, ipc_spy.inputs, strict=True):
        assert mc_u is ipc_u, "MC と IPC が別の駆動入力を見ています"


def test_surrogate_seed_is_shared_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """サロゲートの ``ctx.seed`` は1個を全条件で共有し、行に記録される (D-37)。

    条件ごとにシードを振ると、条件間の容量差にしきい値の推定ノイズが独立に
    乗る (共通乱数法が壊れる)。``washout`` も同じ ``ctx`` から届き、``t0`` の
    違い (MC と IPC で最大遅延が違う) は各診断が自分で決める (D-24)。
    """
    mc_spy, ipc_spy = _install_spies(monkeypatch)
    config = base_config()
    conditions = (
        condition(EXPERIMENT_MC_SWEEP),
        condition(EXPERIMENT_IPC_SWEEP, rho=0.8, n_units=10),
        condition(EXPERIMENT_CONSERVATION, state_noise=0.02),
    )
    rows = [evaluate_capacity_condition(config, item).row for item in conditions]

    contexts = [*mc_spy.contexts, *ipc_spy.contexts]
    assert len(contexts) == 2 * len(conditions)
    assert {ctx.seed for ctx in contexts} == {config.seeds.surrogate}
    assert {ctx.washout for ctx in contexts} == {config.drive.washout}
    assert {row.seed_surrogate for row in rows} == {config.seeds.surrogate}
    # 同一条件では MC と IPC が同じ ctx オブジェクトを共有する (取り違え防止)
    for mc_ctx, ipc_ctx in zip(mc_spy.contexts, ipc_spy.contexts, strict=True):
        assert mc_ctx is ipc_ctx


def test_surrogate_seed_actually_moves_the_threshold() -> None:
    """``seeds.surrogate`` を変えるとしきい値が動く (D-37 の配線の実体)。

    共有していること (上のテスト) だけでは「``ctx.seed`` が診断に届いている」
    ことの証明にならない。届いていなければ ``ValueError`` になるはずだが、
    しきい値が実際に動くことまで測っておかないと、既定値へフォールバック
    する実装に差し替わっても気づけない。
    """
    config = base_config()
    other = dataclasses.replace(
        config, seeds=dataclasses.replace(config.seeds, surrogate=999)
    )
    baseline = evaluate_capacity_condition(config, condition()).row
    changed = evaluate_capacity_condition(other, condition()).row
    assert baseline.mc_threshold != changed.mc_threshold


def esp_base_config() -> Esp02Config:
    """02 側の縮小設定 (バイト一致比較の相手)。"""
    return Esp02Config(
        seeds=EspSeedConfig(reservoir=0, drive=1, probe=3),
        drive=DriveConfig(distribution="uniform", n_steps=300, washout=40, n_pairs=2),
        reservoir=ReservoirSweepConfig(
            input_scale=1.0, n_units=15, density=0.3, n_replicates=1
        ),
    )


def test_reference_states_match_esp_simulate_condition() -> None:
    """``state_noise=0`` の 03 経路は 02 の状態とバイト一致する (D-36)。

    ``build_esn_config`` に ``state_noise`` を足し、``esn.run`` に**常に**
    ``rng`` を渡すようにした変更が、02 の成果物を1バイトも動かしていないこと
    の実測。``ESN`` は ``state_noise=0`` のとき乱数を1個も引かないため、
    重み生成に使った Generator をそのまま渡しても状態は変わらない
    (01 の ``runner.plan_replicate`` と同じ前例)。

    ここで比べる相手を 02 の ``simulate_condition`` にしているのは、02 の
    本番経路そのものだから。初期状態 ``x0`` は ``PROBE`` ストリームから
    引かれるので、03 側も同じ ``x0`` を渡して条件をそろえる。
    """
    config = esp_base_config()
    trajectories = simulate_condition(
        config, rho=0.9, leak_rate=0.6, sigma_u=0.3, replicate=0
    )
    probe_rng = make_rng_for(
        esp_stream_seed(config.seeds, SeedStream.PROBE), SeedStream.PROBE, 0
    )
    x0 = make_initial_states(config.reservoir.n_units, config.drive.n_pairs, probe_rng)[
        0
    ]
    reference = simulate_reference_trajectory(
        config.reservoir,
        config.drive,
        reservoir_seed=esp_stream_seed(config.seeds, SeedStream.RESERVOIR),
        drive_seed=esp_stream_seed(config.seeds, SeedStream.TASK),
        rho=0.9,
        leak_rate=0.6,
        sigma_u=0.3,
        replicate=0,
        x0=x0,
        state_noise=0.0,
    )
    assert np.array_equal(reference.states, trajectories.states)
    assert np.array_equal(reference.drive, trajectories.drive)
    assert reference.esn.config.state_noise == 0.0


def test_state_noise_changes_states_and_requires_rng() -> None:
    """``state_noise>0`` で状態が変わり、``rng`` 未指定の経路が残っていない。

    ``ESN.run`` は ``state_noise > 0`` かつ ``rng is None`` で ``ValueError``
    を投げる。``simulate_reference_trajectory`` が ``rng`` を渡さない分岐を
    残していれば、受け入れ条件2 (ノイズ下の保存則) を測る経路がそもそも
    存在しないことになる —— この変更前は実際にそうだった (仕様 §2.4-4)。
    """
    config = esp_base_config()
    kwargs = {
        "reservoir_seed": esp_stream_seed(config.seeds, SeedStream.RESERVOIR),
        "drive_seed": esp_stream_seed(config.seeds, SeedStream.TASK),
        "rho": 0.9,
        "leak_rate": 0.6,
        "sigma_u": 0.3,
        "replicate": 0,
    }
    quiet = simulate_reference_trajectory(
        config.reservoir, config.drive, state_noise=0.0, **kwargs
    )
    noisy = simulate_reference_trajectory(
        config.reservoir, config.drive, state_noise=0.05, **kwargs
    )
    assert noisy.esn.config.state_noise == 0.05
    assert not np.array_equal(quiet.states, noisy.states)

    # rng を渡さない経路が残っていたら、この ESN は ValueError で落ちるはず。
    # 上の呼び出しが通っている = 配線層が常に rng を渡している証拠。
    with pytest.raises(ValueError, match="rng"):
        noisy.esn.run(noisy.drive)


def test_state_noise_reaches_the_capacity_condition() -> None:
    """``CapacityCondition.state_noise`` が状態生成まで届く (D-36 の配線)。"""
    config = base_config()
    baseline = evaluate_capacity_condition(config, condition()).row
    noisy = evaluate_capacity_condition(config, condition(state_noise=0.05)).row
    assert baseline.state_noise == 0.0
    assert noisy.state_noise == 0.05
    assert baseline.mc_total != noisy.mc_total


def test_conservation_overrides_the_ipc_delays_one_way() -> None:
    """``conservation.max_delay_by_degree`` の上書きは 3-B' の片方向だけ。

    3-A / 3-B は ``config.ipc`` を素で使う。逆向き (3-B' の深い打ち切りを
    ``config.ipc`` の既定にする) にすると 3-A / 3-B の掃引まで重くなる。
    """
    config = dataclasses.replace(
        base_config(),
        conservation=dataclasses.replace(
            base_config().conservation, max_delay_by_degree=(12, 6)
        ),
    )
    assert ipc_config_for(config, EXPERIMENT_MC_SWEEP) is config.ipc
    assert ipc_config_for(config, EXPERIMENT_IPC_SWEEP) is config.ipc
    overridden = ipc_config_for(config, EXPERIMENT_CONSERVATION)
    assert overridden.max_delay_by_degree == (12, 6)
    assert (
        dataclasses.replace(
            overridden, max_delay_by_degree=config.ipc.max_delay_by_degree
        )
        == config.ipc
    ), "打ち切り以外のフィールドまで書き換わっています"

    rows = {
        item.experiment: evaluate_capacity_condition(config, condition(item)).row
        for item in (EXPERIMENT_IPC_SWEEP, EXPERIMENT_CONSERVATION)
        for _ in (0,)
    }
    assert (
        rows[EXPERIMENT_CONSERVATION].n_targets > rows[EXPERIMENT_IPC_SWEEP].n_targets
    )


def test_capacity_outcome_carries_the_arrays_the_figures_need() -> None:
    """行だけでなく図が使う配列も返る (02 の ``ConditionOutcome`` と同型)。

    行だけ返す設計にすると、図を描くために全条件をもう一度回すことになる。
    """
    config = base_config()
    outcome = evaluate_capacity_condition(config, condition())
    assert outcome.mc_profile.shape == (config.mc.max_delay,)
    assert outcome.ipc_heatmap.shape == (
        len(config.ipc.max_delay_by_degree),
        max(config.ipc.max_delay_by_degree),
    )
    assert outcome.ipc_by_degree.shape == (len(config.ipc.max_delay_by_degree),)
    assert outcome.row.n_degrees == len(config.ipc.max_delay_by_degree)
    assert outcome.row.n_samples_mc == condition().n_steps - outcome.row.t0_mc

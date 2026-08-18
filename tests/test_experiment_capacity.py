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
import sys
from types import ModuleType
from typing import Protocol

import numpy as np
import pytest

import rc_basics_lab.diagnostics as diagnostics_package
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
from rc_basics_lab.diagnostics.ipc import ipc
from rc_basics_lab.diagnostics.memory_capacity import memory_capacity
from rc_basics_lab.experiment.capacity import (
    EXPERIMENT_CONSERVATION,
    EXPERIMENT_IPC_SWEEP,
    EXPERIMENT_MC_SWEEP,
    CapacityCondition,
    evaluate_capacity_condition,
    ipc_config_for,
)
from rc_basics_lab.experiment.esp import (
    ReferenceTrajectory,
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
    experiment: str = EXPERIMENT_MC_SWEEP,
    *,
    rho: float = 0.9,
    leak_rate: float = 1.0,
    n_units: int = 12,
    state_noise: float = 0.0,
) -> CapacityCondition:
    """縮小条件1本 (秒未満で回る大きさ)。"""
    return CapacityCondition(
        experiment=experiment,
        rho=rho,
        leak_rate=leak_rate,
        n_units=n_units,
        state_noise=state_noise,
        sigma_u=0.3,
        n_steps=1200,
        replicate=0,
    )


class _DiagnosticCall[C](Protocol):
    """D-01 の署名に ``cfg`` (D-15) を足した呼び出し規約。

    ``cfg`` の型は診断ごとに違う (``MemoryCapacityConfig`` /
    ``IpcConfig``) ので型変数にする。``object`` で受けると
    「MC に IPC の cfg を渡す」取り違えが型で落ちなくなる。
    """

    def __call__(
        self,
        X: FloatArray,
        u: FloatArray | None = None,
        y: FloatArray | None = None,
        *,
        ctx: DiagnosticContext | None = None,
        cfg: C,
    ) -> DiagnosticResult: ...


class _StateSpy[C]:
    """診断へ渡された ``X`` / ``u`` / ``ctx`` を記録しつつ本物を呼ぶ。

    本物を呼ぶのは、行の組み立て (``DiagnosticResult`` のキー参照) まで含めて
    実経路を通したいため。偽の結果を返すスパイにすると「配線層が診断の返り値の
    どのキーを読むか」が固定されなくなる。
    """

    def __init__(self, wrapped: _DiagnosticCall[C]) -> None:
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
        cfg: C,
    ) -> DiagnosticResult:
        self.states.append(X)
        assert u is not None
        self.inputs.append(u)
        assert ctx is not None
        self.contexts.append(ctx)
        return self._wrapped(X, u, ctx=ctx, cfg=cfg)


def _install_spies(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_StateSpy[MemoryCapacityConfig], _StateSpy[IpcConfig]]:
    """MC / IPC を**呼び出し側のモジュール属性**でスパイに差し替える (§10-1)。

    ``rc_basics_lab.diagnostics.ipc`` を差し替えても配線層が既に束縛した参照は
    変わらないうえ、その名前は**関数**を指す (§10-1 の罠)。差し替える先は
    常に呼び出し側 (``rc_basics_lab.experiment.capacity``) の属性である。
    """
    mc_spy: _StateSpy[MemoryCapacityConfig] = _StateSpy(memory_capacity)
    ipc_spy: _StateSpy[IpcConfig] = _StateSpy(ipc)
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
    assert sys.modules["rc_basics_lab.diagnostics.ipc"] is module

    # 同じドット付き名前が、パッケージの属性としては**関数**を指す。
    # ``import rc_basics_lab.diagnostics.ipc as m`` で束縛されるのはこちら。
    assert callable(diagnostics_package.ipc)
    assert not isinstance(diagnostics_package.ipc, ModuleType)
    assert diagnostics_package.ipc is ipc
    # module は ModuleType 型なので mypy が module.ipc を解決できず、
    # getattr が要る (module.ipc だと attr-defined で型検査が落ちる)。
    assert getattr(module, "ipc") is ipc  # noqa: B009


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

    def simulate(state_noise: float) -> ReferenceTrajectory:
        return simulate_reference_trajectory(
            config.reservoir,
            config.drive,
            reservoir_seed=esp_stream_seed(config.seeds, SeedStream.RESERVOIR),
            drive_seed=esp_stream_seed(config.seeds, SeedStream.TASK),
            rho=0.9,
            leak_rate=0.6,
            sigma_u=0.3,
            replicate=0,
            state_noise=state_noise,
        )

    quiet = simulate(0.0)
    noisy = simulate(0.05)
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
        name: evaluate_capacity_condition(config, condition(name)).row
        for name in (EXPERIMENT_IPC_SWEEP, EXPERIMENT_CONSERVATION)
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
    assert outcome.row.n_degrees == len(config.ipc.max_delay_by_degree)
    assert outcome.row.n_samples_mc == condition().n_steps - outcome.row.t0_mc


def test_oversized_n_units_is_rejected_before_any_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``n_units`` の上限超過は ``ESN`` を作る前に ``ValueError`` になる (F-3b1-1-017)。

    3a の D-34 (IPC の確保・組合せ計算量を4段の上限で縛る) と同じ threat model
    ―― 実験層の確保軸 (``n_units`` / ``n_steps``) には上限検査が無く、設定
    YAML の1行変更 (例: ``conservation.n_units_grid: [100000]``) だけで
    ``ESN`` の重み行列に数十GB の確保が発生しうる。``simulate_reference_trajectory``
    を monkeypatch して**呼ばれないこと**を直接固定する (確保より前に落ちる
    ことの実証。D-34 の規律「確保より前に落とす」と同じ)。
    """
    module = importlib.import_module(CAPACITY_MODULE)
    called = False

    def fail_if_called(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("simulate_reference_trajectory が呼ばれました")

    monkeypatch.setattr(
        f"{CAPACITY_MODULE}.simulate_reference_trajectory", fail_if_called
    )
    config = base_config()
    huge = condition(n_units=module._MAX_UNITS + 1)
    with pytest.raises(ValueError, match="n_units"):
        evaluate_capacity_condition(config, huge)
    assert not called, "上限検査より前に状態行列の確保が始まっています"


def test_oversized_state_matrix_is_rejected_before_any_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``n_units*n_steps`` の上限超過も確保の前に ``ValueError`` になる (F-3b1-1-017)。

    ``n_units`` 単体では上限内でも、``n_steps`` を巨大にすれば状態行列
    ``(n_steps, n_units)`` の確保量は同じだけ膨らむ。両方を独立した軸として
    検査しないと、片方の上限だけを見て安全と誤認する (D-34 の rationale が
    ``max_degrees`` 単体では防げない、と言っているのと同じ形)。
    """
    module = importlib.import_module(CAPACITY_MODULE)
    called = False

    def fail_if_called(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("simulate_reference_trajectory が呼ばれました")

    monkeypatch.setattr(
        f"{CAPACITY_MODULE}.simulate_reference_trajectory", fail_if_called
    )
    config = base_config()
    huge_steps = dataclasses.replace(
        condition(n_units=10),
        n_steps=module._MAX_STATE_ELEMENTS // 10 + 1,
    )
    with pytest.raises(ValueError, match="n_units \\* n_steps"):
        evaluate_capacity_condition(config, huge_steps)
    assert not called, "上限検査より前に状態行列の確保が始まっています"


class _PastTheGuard(Exception):
    """上限検査を通り抜けて軌道生成に到達したことを示す番兵 (境界値テスト用)。

    「``ValueError`` にならない」を ``evaluate_capacity_condition`` の完走で
    確かめようとすると、上限値ちょうど (N=5,000 / N*T=2e8) の確保が実際に
    走ってしまう (重み行列 200MB + 5000x5000 の固有値計算、状態行列 1.6GB)。
    軌道生成の入口でこの例外を投げて止めれば、確保を1バイトもせずに
    「検査を通過した」ことだけを固定できる。
    """


def _stop_at_trajectory(monkeypatch: pytest.MonkeyPatch) -> None:
    """軌道生成を ``_PastTheGuard`` に差し替える (確保させずに通過を測る)。"""

    def stop(*args: object, **kwargs: object) -> object:
        raise _PastTheGuard

    monkeypatch.setattr(f"{CAPACITY_MODULE}.simulate_reference_trajectory", stop)


def test_bounds_accept_the_exact_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """上限値**ちょうど**は通る —— 検査は ``>`` であって ``>=`` ではない。

    F-3b1-2-005: 上限ガードの既存テストは「上限 + 1 が落ちる」しか測って
    いなかったため、``>`` が ``>=`` に書き換えられても (= 本番設定の
    ``n_units`` が上限に等しい場合に実験が丸ごと落ちるようになっても)
    検出できなかった。境界のどちら側が許容側かを固定するのはここだけである。

    ``n_units = _MAX_UNITS`` かつ ``n_steps = _MAX_STATE_ELEMENTS //
    _MAX_UNITS`` を選ぶと、**2つの上限を同時にちょうどで踏む**
    (5,000 * 40,000 = 200,000,000)。到達を ``_PastTheGuard`` で観測するので
    確保は起きない (このサイズを実際に確保すると 1.6GB になる)。
    """
    module = importlib.import_module(CAPACITY_MODULE)
    max_units = module._MAX_UNITS
    max_elements = module._MAX_STATE_ELEMENTS
    exact_steps = max_elements // max_units
    assert max_units * exact_steps == max_elements, (
        "上限がちょうどで割り切れる前提が崩れています (テスト側の作り直しが必要)"
    )

    _stop_at_trajectory(monkeypatch)
    boundary = dataclasses.replace(condition(n_units=max_units), n_steps=exact_steps)
    # ValueError が出れば pytest.raises がそれを素通しして落ちる (= 変異検出)。
    with pytest.raises(_PastTheGuard):
        evaluate_capacity_condition(base_config(), boundary)


def test_exact_limit_condition_runs_to_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """上限値ちょうどの条件が**実経路で最後まで回る** (F-3b1-2-005)。

    上のテストは軌道生成の入口で止めるので「検査を通った」までしか測らない。
    ここでは上限そのものを条件に合わせて小さく差し替え (``n_units=12`` /
    ``n_units * n_steps = 14,400``)、確保も診断も実際に走らせて行が出ることを
    固定する。実サイズ (5,000 / 2e8) を回すのは非現実的なので、**測る対象は
    上限の値ではなく比較演算子の向き**である。

    ``>`` を ``>=`` に変えると、ここは行を返さず ``ValueError`` になる。
    """
    module = importlib.import_module(CAPACITY_MODULE)
    exact = condition(n_units=12)
    assert exact.n_units * exact.n_steps == 12 * 1200
    monkeypatch.setattr(module, "_MAX_UNITS", exact.n_units)
    monkeypatch.setattr(module, "_MAX_STATE_ELEMENTS", exact.n_units * exact.n_steps)

    row = evaluate_capacity_condition(base_config(), exact).row
    assert row.n_units == exact.n_units
    assert row.n_steps == exact.n_steps

    # 反対側 (上限 + 1) は両軸とも落ちる —— 境界の両側を1つのテストで押さえる。
    with pytest.raises(ValueError, match="n_units"):
        evaluate_capacity_condition(base_config(), condition(n_units=exact.n_units + 1))
    monkeypatch.setattr(module, "_MAX_UNITS", exact.n_units + 1)
    with pytest.raises(ValueError, match="n_units \\* n_steps"):
        evaluate_capacity_condition(
            base_config(),
            dataclasses.replace(exact, n_steps=exact.n_steps + 1),
        )


def tiny_01_config() -> ExperimentConfig:
    """01 側の最小構成 (3-C が使う ``run_task`` / ``plan_replicate`` の入口)。

    ``mackey_glass`` を使うのは入力が1系列 (``u.shape == (T, 1)``) で、
    3b-2 の T4 が足す NARMA10 と同じ形だからである。
    """
    return ExperimentConfig(
        n_replicates=1,
        seeds=SeedConfig(reservoir=0, task=1, split=2),
        split=SplitConfig(washout=50, max_start_offset=10),
        ridge=RidgeConfig(alpha_grid=(1e-2,), n_lags_grid=(1,)),
        mackey_glass=MackeyGlassConfig(length=1200),
        esn_mackey_glass=ESNConfig(n_units=20, leak_rate=1.0, input_scale=1.0),
    )


def test_externally_built_states_can_produce_a_capacity_row() -> None:
    """**外部で作った ``X``** から ``CapacityRow`` が作れる (F-3b1-1-004)。

    3b-2 の T4 (実験 3-C) の予行演習である。3-C の状態は 01 の
    ``plan_replicate`` が作るので ``CapacityCondition``
    (rho / leak_rate / sigma_u / n_steps を持つ) では表現できず、
    ``evaluate_capacity_condition`` には載せられない。それでも
    ``measure_capacity`` → ``capacity_row_from`` の2段を直接呼べば、
    ``CapacityRow`` の約35フィールドを1つも複製せずに同じ行が作れる ——
    それをここで実測する。壊れると T4 が行の組み立てを複製するしかなくなり、
    「CSV の列順の単一の真実 = 行 dataclass の宣言順」が実質的に破れる。
    """
    config_01 = tiny_01_config()
    entry = next(e for e in build_tasks(config_01) if e.name == "mackey_glass")
    plan = plan_replicate(config_01, entry, replicate=0)
    states = plan.states
    u = plan.task.u[:, 0]
    assert states.flags.writeable is True, "01 側の X は書き込み可能なまま届く"

    config = base_config()
    ctx = DiagnosticContext(washout=config.drive.washout, seed=config.seeds.surrogate)
    measurement = measure_capacity(
        states, u, ctx=ctx, cfg_holder := None or config.mc, ipc_cfg=config.ipc
    )
    row = capacity_row_from(
        measurement,
        experiment="3C_narma10",
        replicate=0,
        seed_reservoir=config_01.seeds.reservoir,
        seed_drive=config_01.seeds.task,
        seed_surrogate=config.seeds.surrogate,
        rho=float("nan"),
        leak_rate=entry.esn.leak_rate,
        input_scale=entry.esn.input_scale,
        sigma_u=float("nan"),
        n_units=entry.esn.n_units,
        density=entry.esn.density,
        state_noise=entry.esn.state_noise,
        n_steps=int(states.shape[0]),
        washout=config.drive.washout,
        wall_time_state_s=0.0,
        wall_time_s=0.0,
    )

    assert row.experiment == "3C_narma10"
    assert row.n_units == entry.esn.n_units
    assert row.n_steps == states.shape[0]
    assert row.n_degrees == len(config.ipc.max_delay_by_degree)
    assert row.mc_total > 0.0
    assert row.input_drive_std == pytest.approx(float(np.std(u)))
    # 列の集合は掃引経路の行と完全に一致する (行を複製していない証拠)
    assert dataclasses.asdict(row).keys() == dataclasses.asdict(
        evaluate_capacity_condition(config, condition()).row
    ).keys()
    # D-35 は外部生成の X にも効く (measure_capacity が塞ぐ)
    assert states.flags.writeable is False

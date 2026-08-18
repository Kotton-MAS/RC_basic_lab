"""情報処理容量 (IPC) のテスト (仕様 rc-basics-03 T2)。

状態系列はすべて**このパッケージの ``reservoir`` を通さず**テスト内で作る
(受け入れ条件6)。``diagnostics`` が移植可能であることの実体はここにある。

このファイルは D-24 / D-25 / D-26 / D-27 / D-28 の guard_test を持つ。
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import time
from collections.abc import Sequence

import numpy as np
import pytest

import rc_basics_lab.diagnostics._capacity as capacity_module
from rc_basics_lab.diagnostics._capacity import (
    HERMITE,
    LEGENDRE,
    NORMAL,
    UNIFORM,
    CapacityProblem,
    orthonormal_basis,
)
from rc_basics_lab.diagnostics.base import DiagnosticContext, DiagnosticResult
from rc_basics_lab.diagnostics.ipc import (
    THRESHOLD_CHI2,
    THRESHOLD_NONE,
    THRESHOLD_SURROGATE,
    IpcConfig,
    TargetSpec,
    _target_column,
    count_targets,
    enumerate_targets,
    ipc,
)
from rc_basics_lab.readout.ridge import fit_ridge_from_gram
from rc_basics_lab.types import FloatArray

CTX_SEED = 20240303
"""サロゲート閾値に使う ``ctx.seed`` (D-27: 乱数源はこれだけ)。"""

INPUT_SCALE = 0.5
"""テスト用リザバーの入力スケール。

MC 側 (0.1) より大きいのは、IPC では次数2以上の容量が実際に立っている状態を
見たいため。``tanh`` が完全な準線形領域にあると総容量が次数1にほぼ集中し、
「次数分解」という測定対象そのものが痩せる (実測: ``input_scale=0.1`` では
次数3 の容量が 3.97、0.5 では 6.82)。
"""

SMALL_CFG = IpcConfig(
    max_delay_by_degree=(20, 8, 4),
    max_variables=2,
    n_surrogates=30,
    chunk_size=64,
)
"""テストの既定より軽い設定 (目標 72 本)。1本 5 秒以内の予算を守るため。

打ち切りだけを浅くしてあり、しきい値法・基底・行合わせの経路は既定と同一。
"""


def _external_reservoir_states(
    rho: float,
    *,
    n_units: int,
    n_steps: int,
    seed: int,
    input_scale: float = INPUT_SCALE,
    state_noise: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """``rc_basics_lab.reservoir`` を使わずに作る状態系列と入力 (受け入れ条件6)。

    ``x[t] = tanh(W x[t-1] + w_in u[t]) (+ noise)`` という素の ESN。診断が
    外部由来の状態系列で動くことを示すのが目的なので、本体の ESN 実装は
    一切通さない。``state_noise`` は状態に加える加法ノイズで、容量を厳密に
    減らす (受け入れ条件2)。
    """
    rng = np.random.default_rng(seed)
    weights: FloatArray = rng.standard_normal((n_units, n_units))
    weights *= rho / max(abs(np.linalg.eigvals(weights)))
    w_in: FloatArray = rng.uniform(-1.0, 1.0, size=n_units) * input_scale
    inputs: FloatArray = rng.uniform(-1.0, 1.0, size=(n_steps, 1))
    states: FloatArray = np.empty((n_steps, n_units), dtype=np.float64)
    state: FloatArray = np.zeros(n_units, dtype=np.float64)
    for step in range(n_steps):
        state = np.tanh(weights @ state + w_in * inputs[step, 0])
        if state_noise > 0.0:
            state = state + rng.normal(0.0, state_noise, size=n_units)
        states[step] = state
    return states, inputs


@functools.cache
def _cached_states(
    rho: float, n_units: int, n_steps: int, seed: int
) -> tuple[FloatArray, FloatArray]:
    """状態生成はテスト間で使い回す (1本 5 秒以内の予算を守るため)。"""
    return _external_reservoir_states(rho, n_units=n_units, n_steps=n_steps, seed=seed)


def _scalars(result: DiagnosticResult) -> dict[str, float]:
    return {key: float(value) for key, value in result.scalars.items()}


def _dummy_problem(*, t0: int, n_samples: int) -> CapacityProblem:
    """``_target_column`` の窓計算 (``CapacityProblem.lagged``) だけを使う
    テストのための最小の ``CapacityProblem``。状態そのものの値は使わない
    (``t0`` / ``n_samples`` が一致していればよい)。
    """
    return CapacityProblem.from_states(np.zeros((t0 + n_samples, 1)), t0=t0)


def _independent_states(
    seed: int, *, n_steps: int, n_units: int
) -> tuple[FloatArray, FloatArray]:
    """入力と一切関係のない状態系列 (容量の帰無仮説そのもの)。"""
    rng = np.random.default_rng(seed)
    states: FloatArray = rng.standard_normal((n_steps, n_units))
    inputs: FloatArray = rng.uniform(-1.0, 1.0, size=(n_steps, 1))
    return states, inputs


# --------------------------------------------------------------------------
# 受け入れ基準1: 保存則 (IPC_total <= N) と飽和 (受け入れ条件2)
# --------------------------------------------------------------------------


def test_ipc_total_does_not_exceed_n_units() -> None:
    """N=25, T=50000 で ``ipc_total <= N*1.02`` かつ ``saturation_ratio >= 0.5``。

    保存則の検査だけだと「容量がほとんど立っていない」状態でも通ってしまう
    (0 <= N は常に真)。飽和比の下限を同時に課すことで、保存則が**空虚に**
    成立するだけの状態を弾く (仕様 §4 T2 受け入れ基準)。

    実行時間の予算はこのテストだけ 30 秒 (T=50000 のため)。実測は状態生成
    込みで約 1.5 秒。
    """
    n_units = 25
    started = time.perf_counter()
    states, inputs = _cached_states(0.9, n_units, 50_000, 7)
    result = ipc(
        states,
        inputs,
        ctx=DiagnosticContext(washout=200, seed=CTX_SEED),
        cfg=IpcConfig(),
    )
    elapsed = time.perf_counter() - started
    scalars = _scalars(result)

    assert scalars["n_targets"] == 601.0, (
        "既定設定の目標数が変わりました (打ち切り表または列挙規則の変更)"
    )
    assert scalars["ipc_total"] <= n_units * 1.02, (
        f"保存則が破れています: ipc_total={scalars['ipc_total']} > {n_units * 1.02}"
    )
    assert scalars["saturation_ratio"] >= 0.5, (
        "保存則が空虚に成立しているだけです: "
        f"saturation_ratio={scalars['saturation_ratio']}"
    )
    assert scalars["saturation_ratio"] == pytest.approx(scalars["ipc_total"] / n_units)
    assert scalars["ipc_linear"] + scalars["ipc_nonlinear"] == pytest.approx(
        scalars["ipc_total"]
    )
    assert scalars["ipc_total"] <= scalars["ipc_total_raw"] + 1.0e-12
    assert elapsed < 30.0, f"予算 (30 秒) を超えました: {elapsed:.1f}s"


def test_state_noise_strictly_reduces_total_capacity() -> None:
    """状態ノイズは総容量を厳密に下げる (受け入れ条件2)。

    差がレプリケート間のばらつきに埋もれていないこと (s.d. の3倍以上) まで
    要求する。ノイズを入れても容量が下がらないなら、容量が状態の情報量では
    なく回帰の自由度 (F/T のかさ上げ) を測っている疑いがある。
    """
    clean: list[float] = []
    noisy: list[float] = []
    ctx = DiagnosticContext(washout=100, seed=CTX_SEED)
    for replicate in range(5):
        states, inputs = _external_reservoir_states(
            0.9, n_units=15, n_steps=4000, seed=100 + replicate
        )
        clean.append(_scalars(ipc(states, inputs, ctx=ctx, cfg=SMALL_CFG))["ipc_total"])
        noisy_states, noisy_inputs = _external_reservoir_states(
            0.9, n_units=15, n_steps=4000, seed=100 + replicate, state_noise=0.1
        )
        noisy.append(
            _scalars(ipc(noisy_states, noisy_inputs, ctx=ctx, cfg=SMALL_CFG))[
                "ipc_total"
            ]
        )

    clean_array: FloatArray = np.asarray(clean, dtype=np.float64)
    noisy_array: FloatArray = np.asarray(noisy, dtype=np.float64)
    difference = float(np.mean(clean_array - noisy_array))
    spread = max(float(np.std(clean_array, ddof=1)), float(np.std(noisy_array, ddof=1)))
    assert np.all(clean_array > noisy_array), (
        f"ノイズ下で容量が下がらないレプリケートがあります: {clean} vs {noisy}"
    )
    assert difference >= 3.0 * spread, (
        "ノイズによる容量低下がレプリケート間のばらつきに埋もれています: "
        f"diff={difference:.4f}, 3*sd={3.0 * spread:.4f}"
    )


# --------------------------------------------------------------------------
# 受け入れ基準2: サロゲートしきい値 (D-27)
# --------------------------------------------------------------------------


def test_surrogate_threshold_zeroes_out_independent_targets() -> None:
    """入力と独立な状態では、しきい値後の総容量が生の 5% 未満になる。

    独立な乱数状態に対する「容量」は有限標本のかさ上げそのもの (各目標で
    平均 ``F / T_eff``) なので、しきい値が効いていればほぼ全部落ちる。
    レプリケートを4本平均するのは、生き残る目標数が二項分布で揺れるため
    (実測: 8 シードで 0.66%〜3.52%、平均 2.0%)。個々のシードでも 5% を
    下回ることまで固定する。
    """
    cfg = IpcConfig(n_surrogates=200)
    ratios: list[float] = []
    for replicate in range(4):
        states, inputs = _independent_states(2718 + replicate, n_steps=3000, n_units=15)
        scalars = _scalars(
            ipc(
                states,
                inputs,
                ctx=DiagnosticContext(washout=10, seed=CTX_SEED + replicate),
                cfg=cfg,
            )
        )
        assert scalars["ipc_total_raw"] > 0.0
        ratios.append(scalars["ipc_total"] / scalars["ipc_total_raw"])

    assert max(ratios) < 0.05, f"閾値後に容量が残りすぎています: {ratios}"
    assert float(np.mean(ratios)) < 0.05, f"平均でも 5% を超えています: {ratios}"


def test_surrogate_threshold_requires_ctx_seed_and_is_reproducible() -> None:
    """サロゲート閾値は ``ctx.seed`` を必須にし、同じ seed で再現する (D-27)。

    閾値が黙って非再現になると、しきい値法の比較 (受け入れ条件3) の記録が
    意味を失う。``chi2`` / ``none`` は乱数を使わないので seed を要求しない。
    """
    states, inputs = _cached_states(0.9, 15, 4000, 5)
    with pytest.raises(ValueError, match=r"ctx\.seed"):
        ipc(states, inputs, ctx=DiagnosticContext(), cfg=SMALL_CFG)

    first = _scalars(
        ipc(states, inputs, ctx=DiagnosticContext(seed=CTX_SEED), cfg=SMALL_CFG)
    )
    same = _scalars(
        ipc(states, inputs, ctx=DiagnosticContext(seed=CTX_SEED), cfg=SMALL_CFG)
    )
    other = _scalars(
        ipc(states, inputs, ctx=DiagnosticContext(seed=CTX_SEED + 1), cfg=SMALL_CFG)
    )
    thresholds = [f"ipc_threshold_degree{degree}" for degree in (1, 2, 3)]
    for key in thresholds:
        assert first[key] == same[key], f"{key} が同じ seed で再現しません"
        assert first[key] > 0.0, f"{key} が 0 です (閾値が効いていない疑い)"
    assert any(first[key] != other[key] for key in thresholds), (
        "seed を変えても閾値が1つも動きません (ctx.seed が使われていない疑い)"
    )
    assert first["ipc_total"] < first["ipc_total_raw"], (
        "サロゲート閾値が1本も落としていません"
    )

    # 乱数を使わない2法は seed 無しで通る。
    for mode in (THRESHOLD_CHI2, THRESHOLD_NONE):
        result = ipc(
            states,
            inputs,
            ctx=DiagnosticContext(),
            cfg=dataclasses.replace(SMALL_CFG, threshold_mode=mode),
        )
        assert result.params["threshold_mode"] == mode


def test_threshold_mode_changes_total_capacity() -> None:
    """3つのしきい値法で総容量が変わる (受け入れ条件3 の一次資料)。

    ``none`` が最大で、``surrogate`` / ``chi2`` はそれより小さく、かつ
    互いに異なる。3法が同じ値を返すなら「しきい値法の比較」という記録が
    そもそも成立しない。
    """
    states, inputs = _cached_states(0.9, 15, 4000, 5)
    ctx = DiagnosticContext(washout=100, seed=CTX_SEED)
    totals = {
        mode: _scalars(
            ipc(
                states,
                inputs,
                ctx=ctx,
                cfg=dataclasses.replace(SMALL_CFG, threshold_mode=mode),
            )
        )
        for mode in (THRESHOLD_SURROGATE, THRESHOLD_CHI2, THRESHOLD_NONE)
    }
    assert totals[THRESHOLD_NONE]["ipc_total"] == pytest.approx(
        totals[THRESHOLD_NONE]["ipc_total_raw"]
    )
    for mode in (THRESHOLD_SURROGATE, THRESHOLD_CHI2):
        assert totals[mode]["ipc_total"] < totals[THRESHOLD_NONE]["ipc_total"]
    assert (
        totals[THRESHOLD_SURROGATE]["ipc_total"] != totals[THRESHOLD_CHI2]["ipc_total"]
    ), "サロゲートとカイ二乗が同じ値です (どちらかが効いていない疑い)"
    # chi2 は次数に依存しないので全次数で同じ値、サロゲートは次数ごとに違う。
    chi2_thresholds = {
        totals[THRESHOLD_CHI2][f"ipc_threshold_degree{degree}"] for degree in (1, 2, 3)
    }
    assert len(chi2_thresholds) == 1
    surrogate_thresholds = {
        totals[THRESHOLD_SURROGATE][f"ipc_threshold_degree{degree}"]
        for degree in (1, 2, 3)
    }
    assert len(surrogate_thresholds) == 3, "次数ごとに閾値を推定していません (D-27)"


# --------------------------------------------------------------------------
# 受け入れ基準3: 性能構造 (D-26)
# --------------------------------------------------------------------------


def _expected_solve_count(cfg: IpcConfig) -> int:
    """``fit_ridge_from_gram`` の想定呼び出し回数 (目標チャンク + サロゲート)。"""
    n_targets = count_targets(cfg)
    total = math.ceil(n_targets / cfg.chunk_size)
    for degree, max_delay in enumerate(cfg.max_delay_by_degree, start=1):
        per_degree = sum(
            math.comb(max_delay, n_vars) * math.comb(degree - 1, n_vars - 1)
            for n_vars in range(1, min(cfg.max_variables, degree) + 1)
        )
        selected = min(cfg.n_surrogate_targets, per_degree)
        total += math.ceil(selected * cfg.n_surrogates / cfg.chunk_size)
    return total


def test_gram_solve_count_does_not_scale_with_target_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """solve の回数はチャンク数で決まり、目標数には比例しない (D-26)。

    目標を 25 本から 270 本 (10.8 倍) に増やしても solve は 3 回から 4 回に
    しか増えない。「(delay, degree) ごとに回帰し直す」構造に戻ると回数が
    目標数に比例し、実験そのものが回らなくなる (実測 3.1 時間)。
    """
    states, inputs = _cached_states(0.9, 15, 4000, 5)
    ctx = DiagnosticContext(washout=100, seed=CTX_SEED)
    original = fit_ridge_from_gram
    configs = (
        IpcConfig(
            max_delay_by_degree=(10, 5),
            max_variables=2,
            n_surrogates=20,
            chunk_size=256,
        ),
        IpcConfig(
            max_delay_by_degree=(60, 20),
            max_variables=2,
            n_surrogates=20,
            chunk_size=256,
        ),
    )
    observed: list[int] = []
    for cfg in configs:
        solves: list[str] = []

        def counting_solve(
            gram: FloatArray,
            rhs: FloatArray,
            alpha: float,
            *,
            bias_column: int | None,
            _sink: list[str] = solves,
        ) -> FloatArray:
            _sink.append("solve")
            return original(gram, rhs, alpha, bias_column=bias_column)

        monkeypatch.setattr(capacity_module, "fit_ridge_from_gram", counting_solve)
        result = ipc(states, inputs, ctx=ctx, cfg=cfg)
        assert result.scalars["n_targets"] == float(count_targets(cfg))
        assert len(solves) == _expected_solve_count(cfg), (
            f"solve の回数が想定と違います: {len(solves)}"
            f" != {_expected_solve_count(cfg)} (cfg={cfg})"
        )
        observed.append(len(solves))

    target_ratio = count_targets(configs[1]) / count_targets(configs[0])
    solve_ratio = observed[1] / observed[0]
    assert target_ratio > 10.0
    assert solve_ratio < 2.0, (
        "solve の回数が目標数に比例しています: "
        f"目標 {target_ratio:.1f} 倍に対し solve {solve_ratio:.1f} 倍"
    )


def test_chunk_size_does_not_change_results() -> None:
    """``chunk_size`` は性能パラメータで、結果を変えてはいけない (仕様 §10-2)。

    他の設定フィールドは「変えたら出力が変わる」ことを要求されるが、これだけは
    **逆向きの要求**である。チャンク分割にバグ (列の取り違え、サロゲート乱数の
    チャンク依存) があるとここが落ちる。
    """
    states, inputs = _cached_states(0.9, 15, 4000, 5)
    ctx = DiagnosticContext(washout=100, seed=CTX_SEED)
    reference = ipc(states, inputs, ctx=ctx, cfg=SMALL_CFG)

    for chunk_size in (1, 7, 64, 1000):
        other = ipc(
            states,
            inputs,
            ctx=ctx,
            cfg=dataclasses.replace(SMALL_CFG, chunk_size=chunk_size),
        )
        assert set(_scalars(other)) == set(_scalars(reference))
        for key, value in _scalars(reference).items():
            assert other.scalars[key] == pytest.approx(value, rel=1.0e-10), (
                f"chunk_size={chunk_size} で {key} が変わりました"
            )
        for key, array in reference.arrays.items():
            np.testing.assert_allclose(other.arrays[key], array, rtol=1.0e-10, atol=0.0)


# --------------------------------------------------------------------------
# 受け入れ基準4: 行集合の共有 (D-24)
# --------------------------------------------------------------------------


def test_all_targets_share_identical_rows() -> None:
    """行集合は ``t0 = max(washout, 全次数の最大遅延)`` で決まり、全目標で同一 (D-24)。

    2つの向きから固定する。

    1. ``washout`` が最大遅延より小さい間は結果が1ビットも変わらない
       (基準点が単一である)。目標ごとに使える行を変える実装 (深い遅延ほど
       標本が減る) では、washout を動かした瞬間に浅い遅延の容量だけが変わる。
    2. 目標の行 ``j`` が状態の行 ``t0 + j`` に対応し、遅延 ``k`` の目標が
       ``u[t0 + j - k]`` を見ている (行合わせそのもの)。入力を単調な系列
       (``u[t] = t``) にすると、目標の値から入力の index を逆算できる。
    """
    states, inputs = _cached_states(0.9, 15, 4000, 5)
    cfg = SMALL_CFG
    max_delay = max(cfg.max_delay_by_degree)
    results = [
        ipc(
            states,
            inputs,
            ctx=DiagnosticContext(washout=washout, seed=CTX_SEED),
            cfg=cfg,
        )
        for washout in (0, 7, max_delay)
    ]
    for result in results:
        assert result.params["t0"] == str(max_delay)
        assert result.params["n_samples"] == str(4000 - max_delay)
    for other in results[1:]:
        np.testing.assert_array_equal(
            other.arrays["ipc_heatmap"], results[0].arrays["ipc_heatmap"]
        )

    deeper = ipc(
        states,
        inputs,
        ctx=DiagnosticContext(washout=300, seed=CTX_SEED),
        cfg=cfg,
    )
    assert deeper.params["t0"] == "300", "washout が最大遅延を超えたら t0 は washout"

    # 行合わせの直接確認。u[t] = t なら psi_1 は単調なので index を逆算できる。
    n_steps = 500
    ramp: FloatArray = np.arange(n_steps, dtype=np.float64)
    psi_table = [orthonormal_basis(ramp, degree, UNIFORM) for degree in (1, 2)]
    t0 = 20
    n_samples = n_steps - t0
    problem = _dummy_problem(t0=t0, n_samples=n_samples)
    mean = float(np.mean(ramp))
    sigma = float(np.std(ramp))
    for delay in (1, 5, 20):
        spec: TargetSpec = ((delay, 1),)
        column = _target_column(problem, psi_table, spec)
        assert column.shape == (n_samples,)
        expected: FloatArray = (
            np.arange(t0 - delay, t0 - delay + n_samples, dtype=np.float64) - mean
        ) / sigma
        np.testing.assert_allclose(column, expected, rtol=0.0, atol=1.0e-12)


# --------------------------------------------------------------------------
# 受け入れ基準5: 基底の正規直交性と未対応な組 (D-28)
# --------------------------------------------------------------------------


def test_basis_is_orthonormal_and_mismatched_pair_raises() -> None:
    """IPC の目標は入力測度に対して正規直交で、未対応な組は ValueError (D-28)。

    検査するのは**積の目標そのもの**である。基底 1本ずつの直交性は T1 が
    固定しているが、IPC が実際に足し合わせるのは
    ``Π_i psi_{n_i}(u[t - k_i])`` であり、ここが直交していないと容量が
    目標間で二重計上され保存則が「N をわずかに超える」という穏やかな形で破れる。
    T=200000、許容差 0.02 (T1 の ``|G/T - I| < 0.02`` と同じ基準)。
    """
    n_steps = 200_000
    rng = np.random.default_rng(4242)
    series: FloatArray = rng.uniform(-1.0, 1.0, size=n_steps)
    cfg = IpcConfig(max_delay_by_degree=(4, 3, 2), max_variables=2)
    specs = enumerate_targets(cfg)
    psi_table = [
        orthonormal_basis(series, degree, UNIFORM, basis=LEGENDRE)
        for degree in (1, 2, 3)
    ]
    t0 = max(cfg.max_delay_by_degree)
    n_samples = n_steps - t0
    problem = _dummy_problem(t0=t0, n_samples=n_samples)
    columns: FloatArray = np.empty((n_samples, len(specs)), dtype=np.float64)
    for index, spec in enumerate(specs):
        columns[:, index] = _target_column(problem, psi_table, spec)
    gram: FloatArray = columns.T @ columns / n_samples
    deviation = float(np.max(np.abs(gram - np.eye(len(specs)))))
    assert deviation < 0.02, (
        f"目標が正規直交ではありません: max|G/T - I| = {deviation:.4f}"
    )

    states, inputs = _cached_states(0.9, 15, 4000, 5)
    ctx = DiagnosticContext(washout=100, seed=CTX_SEED)
    for distribution, basis in ((UNIFORM, HERMITE), (NORMAL, LEGENDRE)):
        with pytest.raises(ValueError, match="D-28"):
            ipc(
                states,
                inputs,
                ctx=ctx,
                cfg=dataclasses.replace(
                    SMALL_CFG, input_distribution=distribution, basis=basis
                ),
            )


# --------------------------------------------------------------------------
# 受け入れ基準6: alpha 単調性 (D-25)
# --------------------------------------------------------------------------


def test_capacity_is_monotone_decreasing_in_alpha() -> None:
    """容量は alpha に対して単調非増加で、大きな alpha では実際に減る (D-25)。

    正則化は「線形読み出しで到達可能な最大の説明率」という容量の定義を
    系統的に過小評価する。この向きが崩れていたら、容量測定に検証分割の
    alpha 選択 (D-04) を持ち込んではいけないという D-25 の前提が壊れている。
    """
    states, inputs = _cached_states(0.9, 15, 4000, 5)
    ctx = DiagnosticContext(washout=100, seed=CTX_SEED)
    totals = [
        _scalars(
            ipc(
                states,
                inputs,
                ctx=ctx,
                cfg=dataclasses.replace(
                    SMALL_CFG, alpha=alpha, threshold_mode=THRESHOLD_NONE
                ),
            )
        )["ipc_total_raw"]
        for alpha in (1.0e-9, 1.0e-6, 1.0e-3, 1.0, 100.0)
    ]
    for smaller, larger in itertools.pairwise(totals):
        assert larger <= smaller + 1.0e-9, f"alpha に対して容量が増えました: {totals}"
    assert totals[-1] < totals[0], f"alpha=100 で容量が減っていません: {totals}"


# --------------------------------------------------------------------------
# 目標の列挙と集約規則 (仕様 §4 T2-2 / T2-3)
# --------------------------------------------------------------------------


def _delays(spec: TargetSpec) -> tuple[int, ...]:
    return tuple(delay for delay, _ in spec)


@pytest.mark.parametrize(
    "cfg",
    [
        IpcConfig(),
        IpcConfig(max_delay_by_degree=(20, 8, 4), max_variables=2),
        IpcConfig(max_delay_by_degree=(5,), max_variables=3),
        IpcConfig(max_delay_by_degree=(3, 3, 3, 3), max_variables=1),
        IpcConfig(max_delay_by_degree=(2, 2, 2), max_variables=5),
    ],
)
def test_target_enumeration_matches_the_declared_rule(cfg: IpcConfig) -> None:
    """列挙された目標が仕様 §4 T2-2 の規則を満たし、閉形式の数と一致する。

    ``count_targets`` は ``max_targets`` の検査に使う (列挙してから数えると
    深い打ち切りでメモリと時間が尽きる) ので、実際の列挙と一致していなければ
    上限の検査そのものが嘘になる。
    """
    specs = enumerate_targets(cfg)
    assert len(specs) == count_targets(cfg)
    assert len(set(specs)) == len(specs), "重複した目標があります"
    for spec in specs:
        delays = _delays(spec)
        orders = tuple(order for _, order in spec)
        assert len(spec) <= cfg.max_variables
        assert all(order >= 1 for order in orders)
        assert list(delays) == sorted(set(delays)), (
            f"遅延が相異なる昇順ではありません: {spec}"
        )
        assert all(delay >= 1 for delay in delays)
        degree = sum(orders)
        assert 1 <= degree <= len(cfg.max_delay_by_degree)
        assert max(delays) <= cfg.max_delay_by_degree[degree - 1], (
            f"打ち切りを超えた遅延があります: {spec}"
        )
    # 各次数がすべて出現する (次数を丸ごと落としていない)。
    degrees = {sum(order for _, order in spec) for spec in specs}
    assert degrees == set(range(1, len(cfg.max_delay_by_degree) + 1))


def test_default_config_enumerates_601_targets() -> None:
    """既定設定の目標数を実測値で固定する (打ち切り表の変更を検出する)。

    内訳: 次数1 が 60、次数2 が 210 (20 + C(20,2))、次数3 が 220
    (10 + 2*C(10,2) + C(10,3))、次数4 が 111 (6 + 3*C(6,2) + 3*C(6,3))。
    """
    cfg = IpcConfig()
    specs = enumerate_targets(cfg)
    per_degree = [
        sum(1 for spec in specs if sum(order for _, order in spec) == degree)
        for degree in (1, 2, 3, 4)
    ]
    assert per_degree == [60, 210, 220, 111]
    assert len(specs) == 601


def test_target_enumeration_raises_instead_of_truncating() -> None:
    """目標数が ``max_targets`` を超えたら黙って切り詰めず ``ValueError``。

    切り詰めると「打ち切り表を深くしたのに総容量が増えない」という、CSV を
    見ても原因の分からない壊れ方をする。
    """
    cfg = dataclasses.replace(IpcConfig(), max_targets=100)
    assert count_targets(cfg) == 601
    with pytest.raises(ValueError, match="max_targets"):
        enumerate_targets(cfg)

    states, inputs = _cached_states(0.9, 15, 4000, 5)
    with pytest.raises(ValueError, match="max_targets"):
        ipc(
            states,
            inputs,
            ctx=DiagnosticContext(washout=100, seed=CTX_SEED),
            cfg=cfg,
        )
    # 上限ちょうどは通る (境界で off-by-one していない)。
    exact = dataclasses.replace(IpcConfig(), max_targets=601)
    assert len(enumerate_targets(exact)) == 601


def test_max_targets_also_bounds_the_heatmap_cell_count() -> None:
    """``max_targets`` は目標数だけでなく ``ipc_heatmap`` の確保サイズも縛る
    (F-03-1-016)。

    目標数と heatmap 面積 (``n_degrees x max(max_delay_by_degree)``) は独立に
    増やせる: 次数を大量に・遅延を浅く取ると目標数は少ないまま heatmap セル数
    だけが大きくなる (CWE-789)。目標数の検査だけを通過してから巨大な
    ``np.zeros`` を確保する経路が残っていないことを確認する。
    """
    # count_targets はここでは少ないが (max_variables=1 なので次数ごとに
    # max_delay 本)、heatmap は 20 x 11000 = 220,000 セルになる。次数の本数
    # (20) は max_degrees の既定値ちょうど (F-03-2-013 の独立な上限には
    # 抵触しない設定で、heatmap_cells の検査だけを単独で踏む)。
    cfg = IpcConfig(
        max_delay_by_degree=(11_000,) + (1,) * 19,
        max_variables=1,
        max_targets=200_000,
    )
    assert count_targets(cfg) < cfg.max_targets
    heatmap_cells = len(cfg.max_delay_by_degree) * max(cfg.max_delay_by_degree)
    assert heatmap_cells > cfg.max_targets

    states, inputs = _cached_states(0.9, 5, 600, 5)
    with pytest.raises(ValueError, match="max_targets"):
        ipc(
            states,
            inputs,
            ctx=DiagnosticContext(washout=10, seed=CTX_SEED),
            cfg=cfg,
        )


def test_heatmap_aggregates_targets_at_their_deepest_delay() -> None:
    """各目標は ``(次数 d, max(k_i))`` のセルに集約される (仕様 §4 T2-3)。

    ヒートマップの行和が次数ごとの容量に一致し、全体の和が ``ipc_total`` に
    一致することで、集約が「取りこぼしも二重計上もない」ことを固定する。
    さらに、次数 ``d`` の打ち切りより深い列が 0 であることを確認する
    (深い遅延の列に容量が漏れていれば集約規則が壊れている)。
    """
    states, inputs = _cached_states(0.9, 15, 4000, 5)
    result = ipc(
        states,
        inputs,
        ctx=DiagnosticContext(washout=100, seed=CTX_SEED),
        cfg=SMALL_CFG,
    )
    heatmap = result.arrays["ipc_heatmap"]
    by_degree = result.arrays["ipc_by_degree"]
    assert heatmap.shape == (3, max(SMALL_CFG.max_delay_by_degree))
    np.testing.assert_allclose(heatmap.sum(axis=1), by_degree, rtol=1.0e-12)
    assert float(heatmap.sum()) == pytest.approx(result.scalars["ipc_total"])
    assert float(by_degree[0]) == pytest.approx(result.scalars["ipc_linear"])
    assert float(by_degree[1:].sum()) == pytest.approx(result.scalars["ipc_nonlinear"])
    for degree, max_delay in enumerate(SMALL_CFG.max_delay_by_degree):
        assert np.all(heatmap[degree, max_delay:] == 0.0), (
            f"次数 {degree + 1} の打ち切り ({max_delay}) より深い列に容量があります"
        )
    np.testing.assert_allclose(
        result.arrays["ipc_by_degree_raw"].sum(),
        result.scalars["ipc_total_raw"],
        rtol=1.0e-12,
    )


# --------------------------------------------------------------------------
# 設定の被覆と入力の検証
# --------------------------------------------------------------------------


def test_ipc_config_fields_change_output() -> None:
    """``chunk_size`` / ``max_targets`` 以外の全フィールドは出力を変える。

    「設定したのに効いていない」除け。除外した2つは専用テストが担当する。

    - ``chunk_size``: 逆向きの要求 (``test_chunk_size_does_not_change_results``)
    - ``max_targets``: 上限であり、超えたときに ``ValueError`` にすることが
      唯一の観測可能な効果 (``test_target_enumeration_raises_instead_of_truncating``)

    ``basis`` と ``input_distribution`` は**対で**意味を持つ (D-28: 片方だけ
    変えると未対応な組になり ``ValueError``) ので、対で入れ替えた設定を
    両フィールドの検査に使う。片方だけを変えたときに実際に落ちることは
    ``test_basis_is_orthonormal_and_mismatched_pair_raises`` が固定する。
    """
    states, inputs = _cached_states(0.9, 15, 4000, 5)
    ctx = DiagnosticContext(washout=100, seed=CTX_SEED)
    base_cfg = IpcConfig(
        max_delay_by_degree=(12, 6, 3),
        max_variables=2,
        n_surrogates=20,
        chunk_size=16,
    )
    reference = _scalars(ipc(states, inputs, ctx=ctx, cfg=base_cfg))
    swapped_basis = dataclasses.replace(
        base_cfg, input_distribution=NORMAL, basis=HERMITE
    )
    changed: dict[str, IpcConfig] = {
        "max_delay_by_degree": dataclasses.replace(
            base_cfg, max_delay_by_degree=(14, 6, 3)
        ),
        "max_variables": dataclasses.replace(base_cfg, max_variables=1),
        "basis": swapped_basis,
        "input_distribution": swapped_basis,
        "alpha": dataclasses.replace(base_cfg, alpha=10.0),
        "threshold_mode": dataclasses.replace(base_cfg, threshold_mode=THRESHOLD_NONE),
        "n_surrogates": dataclasses.replace(base_cfg, n_surrogates=60),
        "n_surrogate_targets": dataclasses.replace(base_cfg, n_surrogate_targets=1),
        "surrogate_quantile": dataclasses.replace(base_cfg, surrogate_quantile=0.5),
    }
    covered = set(changed) | {"chunk_size", "max_targets"}
    actual = {field.name for field in dataclasses.fields(IpcConfig)}
    assert covered == actual, (
        "IpcConfig のフィールドに対する検査が不足しています: "
        f"{sorted(actual - covered)}"
    )

    for name, cfg in changed.items():
        other = _scalars(ipc(states, inputs, ctx=ctx, cfg=cfg))
        assert other != reference, f"{name} を変えても出力が変わりません"


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (IpcConfig(max_delay_by_degree=()), "max_delay_by_degree"),
        (IpcConfig(max_delay_by_degree=(10, 0)), "max_delay_by_degree"),
        (IpcConfig(max_variables=0), "max_variables"),
        (IpcConfig(alpha=-1.0), "alpha"),
        (IpcConfig(threshold_mode="bonferroni"), "threshold_mode"),
        (IpcConfig(chunk_size=0), "chunk_size"),
        (IpcConfig(max_targets=0), "max_targets"),
        (IpcConfig(n_surrogates=0), "n_surrogates"),
        (IpcConfig(n_surrogate_targets=0), "n_surrogate_targets"),
        (IpcConfig(surrogate_quantile=1.5), "surrogate_quantile"),
    ],
)
def test_out_of_range_config_raises(cfg: IpcConfig, message: str) -> None:
    """設定の値域違反は既定へフォールバックせず ``ValueError`` (D-09)。"""
    states, inputs = _cached_states(0.9, 15, 4000, 5)
    with pytest.raises(ValueError, match=message):
        ipc(states, inputs, ctx=DiagnosticContext(seed=CTX_SEED), cfg=cfg)


def test_ipc_requires_single_channel_input() -> None:
    """``u`` が無い / 多変数だと ``ValueError`` (安く緑にする逃げ道を塞ぐ)。"""
    states, inputs = _cached_states(0.9, 15, 4000, 5)
    ctx = DiagnosticContext(seed=CTX_SEED)
    with pytest.raises(ValueError, match="入力系列 u"):
        ipc(states, None, ctx=ctx, cfg=SMALL_CFG)
    with pytest.raises(ValueError, match="1変数入力"):
        ipc(states, np.repeat(inputs, 2, axis=1), ctx=ctx, cfg=SMALL_CFG)


def test_series_shorter_than_max_delay_raises_instead_of_truncating() -> None:
    """最大遅延が系列長を超えたら黙って切り詰めず ``ValueError``。"""
    states, inputs = _cached_states(0.9, 15, 4000, 5)
    with pytest.raises(ValueError, match="系列が短すぎます"):
        ipc(
            states,
            inputs,
            ctx=DiagnosticContext(seed=CTX_SEED),
            cfg=dataclasses.replace(SMALL_CFG, max_delay_by_degree=(4000, 8, 4)),
        )


# --------------------------------------------------------------------------
# 受け入れ条件6: 外部生成の状態系列で完走する
# --------------------------------------------------------------------------


def test_diagnostic_accepts_arbitrary_external_state_series() -> None:
    """リザバーですらない任意の X (独立な乱数) でも完走する (受け入れ条件6)。

    ``rc_basics_lab.reservoir`` を一切通していない状態系列で、全スカラが
    有限であることまで確認する。
    """
    states, inputs = _independent_states(31415, n_steps=1500, n_units=12)
    result = ipc(
        states,
        inputs,
        ctx=DiagnosticContext(washout=10, seed=CTX_SEED),
        cfg=SMALL_CFG,
    )
    scalars = _scalars(result)
    assert result.name == "ipc"
    assert all(np.isfinite(value) for value in scalars.values())
    assert set(scalars) >= {
        "ipc_total",
        "ipc_total_raw",
        "ipc_linear",
        "ipc_nonlinear",
        "n_targets",
        "n_targets_kept",
        "saturation_ratio",
        "ipc_threshold_degree1",
        "ipc_threshold_degree2",
        "ipc_threshold_degree3",
    }
    assert 0.0 <= scalars["n_targets_kept"] <= scalars["n_targets"]
    row = result.to_row()
    assert row["diagnostic"] == "ipc"


def _sorted_by_degree(specs: Sequence[TargetSpec]) -> list[int]:
    return [sum(order for _, order in spec) for spec in specs]


def test_targets_are_enumerated_in_degree_order() -> None:
    """列挙順が次数昇順である (次数ごとのしきい値を区間で切り出す前提)。"""
    degrees = _sorted_by_degree(enumerate_targets(IpcConfig()))
    assert degrees == sorted(degrees)

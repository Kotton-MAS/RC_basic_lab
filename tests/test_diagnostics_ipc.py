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
    UNIFORM_LEGENDRE,
    CapacityProblem,
    RowAlignment,
    bounded_chunk_size,
    orthonormal_basis,
    surrogate_threshold,
)
from rc_basics_lab.diagnostics.base import DiagnosticContext, DiagnosticResult
from rc_basics_lab.diagnostics.ipc import (
    AXIS_HEATMAP_CELLS,
    AXIS_TARGET_COUNT,
    MAX_TARGETS_BOUNDED_AXES,
    THRESHOLD_CHI2,
    THRESHOLD_NONE,
    THRESHOLD_SURROGATE,
    IpcConfig,
    TargetSpec,
    _format_target_count,
    _picked_target_blocks,
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


def test_surrogate_threshold_rejects_bare_ndarray_base_blocks() -> None:
    """``base_blocks`` に ``ndarray`` を直接渡すと専用の ``TypeError``
    (F-03-4-004)。

    ``base_blocks`` の型は単一の ``FloatArray`` から ``Iterable[FloatArray]``
    (ブロックの列) へ広がったが、``np.ndarray`` はイテレートすると行を返す
    ため構造的に ``Iterable`` を満たしてしまい、mypy はこの誤用 (2次元配列を
    直接渡す旧来の呼び方) を検出できない (実測: mypy exit 0。実行すると
    ``ValueError: base_blocks の要素は (T, M) が必要です: (2,)`` という
    分かりにくいメッセージになる)。境界で ``TypeError`` を出すことを固定する。
    """
    states, _ = _cached_states(0.9, 15, 200, 3)
    problem = CapacityProblem.from_states(
        states, rows=RowAlignment(t0=20, n_samples=180)
    )
    bare: FloatArray = np.random.default_rng(1).standard_normal((problem.n_samples, 2))
    with pytest.raises(TypeError, match="ndarray"):
        surrogate_threshold(
            problem,
            bare,  # 単一の (T, M) 配列を直接渡す誤用 (正しくは [bare])
            1.0e-9,
            n_surrogates=5,
            quantile=0.9,
            chunk_size=8,
            rng=np.random.default_rng(0),
        )
    # 正しい呼び方 ([bare] のように包む) は通ることも合わせて固定する。
    threshold, capacities = surrogate_threshold(
        problem,
        [bare],
        1.0e-9,
        n_surrogates=5,
        quantile=0.9,
        chunk_size=8,
        rng=np.random.default_rng(0),
    )
    assert np.isfinite(threshold)
    assert capacities.shape == (2 * 5,)


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


def _expected_solve_count(cfg: IpcConfig, n_samples: int) -> int:
    """``fit_ridge_from_gram`` の想定呼び出し回数 (目標チャンク + サロゲート)。

    F-03-2-001: ``chunk_size`` は ``CapacityProblem.effective_chunk_size``
    (``bounded_chunk_size``) で下げられうるため、閉形式は ``cfg.chunk_size``
    (設定値) ではなく実効値 (``ceil(K / effective_chunk_size)``) を使う。
    従来の閉形式は「上限が発動しない規模」でしか検証しておらず、本番規模
    (T=1e6 級) で崩れていた (HIGH: 期待2回に対し実際4回)。

    F-03-3-023: ここは本番の ``bounded_chunk_size`` を直接呼んで実効値を得る
    ため、``bounded_chunk_size`` 自体にバグがあると期待値側も同じバグを
    踏んで一致し、``test_gram_solve_count_does_not_scale_with_target_count``
    単独では検知できない (実測: バイト予算計算を4倍過剰に厳しくする変異を
    注入し、このテストだけを単独実行すると 1 passed で素通りする)。
    ``bounded_chunk_size`` 自身の正しさ (バイト予算からの独立な計算) は
    ``tests/test_diagnostics_memory_capacity.py`` の
    ``test_bounded_chunk_size_truncates_when_over_budget`` /
    ``test_bounded_chunk_size_keeps_configured_when_under_budget`` /
    ``test_bounded_chunk_size_never_returns_less_than_one`` が別途固定する
    (``make ci`` のフルスイート実行では検知される)。
    """
    effective = bounded_chunk_size(cfg.chunk_size, n_samples)
    n_targets = count_targets(cfg)
    total = math.ceil(n_targets / effective)
    for degree, max_delay in enumerate(cfg.max_delay_by_degree, start=1):
        per_degree = sum(
            math.comb(max_delay, n_vars) * math.comb(degree - 1, n_vars - 1)
            for n_vars in range(1, min(cfg.max_variables, degree) + 1)
        )
        selected = min(cfg.n_surrogate_targets, per_degree)
        total += math.ceil(selected * cfg.n_surrogates / effective)
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
        # F-03-2-001: n_samples=3900 (washout=100, T=4000) での 128MiB 予算
        # は約4302列。chunk_size=8192 (> 4302) は実際にキャップされ、実効値
        # は configured (8192) ではなく budget (約4302) になる。閉形式
        # (_expected_solve_count) が configured ではなく実効値を使うことを
        # この config で検証する (HIGH: 従来の閉形式はキャップが発動しない
        # 規模でしか検証していなかった)。
        IpcConfig(
            max_delay_by_degree=(60, 20),
            max_variables=2,
            n_surrogates=20,
            chunk_size=8192,
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
        n_samples = int(result.params["n_samples"])
        assert result.scalars["n_targets"] == float(count_targets(cfg))
        expected = _expected_solve_count(cfg, n_samples)
        assert len(solves) == expected, (
            f"solve の回数が想定と違います: {len(solves)} != {expected} (cfg={cfg})"
        )
        observed.append(len(solves))

    target_ratio = count_targets(configs[1]) / count_targets(configs[0])
    solve_ratio = observed[1] / observed[0]
    assert target_ratio > 10.0
    assert solve_ratio < 2.0, (
        "solve の回数が目標数に比例しています: "
        f"目標 {target_ratio:.1f} 倍に対し solve {solve_ratio:.1f} 倍"
    )

    # configs[2] は configs[1] と目標構成が同一で chunk_size だけが違う。
    # キャップが実際に発動していること (実効値 < 設定値) を明示し、その状態
    # でも solve 回数の閉形式 (ceil(K / effective_chunk_size)) が一致する
    # ことを上の assert がすでに固定している (D-26: HIGH の修正)。
    capped_n_samples = int(
        ipc(states, inputs, ctx=ctx, cfg=configs[2]).params["n_samples"]
    )
    assert (
        bounded_chunk_size(configs[2].chunk_size, capped_n_samples)
        < configs[2].chunk_size
    ), "この設定ではキャップが発動しません (テストの前提が崩れています)"


def test_params_record_configured_and_effective_chunk_size_when_capped() -> None:
    """``params`` は設定値と実効値の両方を記録する (D-33 の rule (iii))。

    F-03-3-004: D-33 の guard_test は round2 まで rule (i) (``bounded_chunk_size``
    の純関数としての正しさ) しか固定しておらず、``params['chunk_size_effective']``
    を削除しても落ちるテストが0件だった (実測: 削除する変異を注入しても
    522 passed で変化なし。``grep -rn 'chunk_size_effective' tests/`` も
    0ヒットだった)。キャップが実際に発動する規模で ``ipc()`` を実行し、
    ``params`` に設定値と実効値の両方が正しく残ることを固定する。
    """
    states, inputs = _cached_states(0.9, 15, 4000, 5)
    ctx = DiagnosticContext(washout=100, seed=CTX_SEED)
    cfg = IpcConfig(
        max_delay_by_degree=(10, 5), max_variables=2, n_surrogates=5, chunk_size=8192
    )
    result = ipc(states, inputs, ctx=ctx, cfg=cfg)
    n_samples = int(result.params["n_samples"])
    expected_effective = bounded_chunk_size(cfg.chunk_size, n_samples)
    assert expected_effective < cfg.chunk_size, (
        "この規模ではキャップが発動していません (テストの前提が崩れています)"
    )
    assert result.params["chunk_size"] == str(cfg.chunk_size)
    assert result.params["chunk_size_effective"] == str(expected_effective)


def test_representative_blocks_do_not_follow_chunk_size() -> None:
    """代表目標ブロックの確保幅は ``cfg.chunk_size`` に**従わない** (D-33 の確保軸)。

    ``_picked_target_blocks`` が一度に実体化する列数は「一度に何列を確保して
    よいか」という**確保軸**であり、1回の solve に何列畳むかという性能軸とは
    別物である。旧実装は性能軸の実効値 (``solve_width(cfg.chunk_size)``) を
    そのまま確保幅に使っていたため、運用者が性能のために ``chunk_size=1`` に
    しただけで代表目標が4ブロックに割れた —— 運用者の性能ノブが確保上限を
    動かしていた。

    ここでブロック**数**を数えるのが要点である。値だけを見る検査だと、
    確保幅を性能軸へ戻す変異 (却下案B: 名前だけ分ける) で1件も落ちない
    (どちらの実装でも容量の値は同じ)。
    """
    n_steps = 500
    ramp: FloatArray = np.arange(n_steps, dtype=np.float64)
    psi_table = [orthonormal_basis(ramp, degree, UNIFORM_LEGENDRE) for degree in (1, 2)]
    rows = RowAlignment(t0=20, n_samples=n_steps - 20)
    specs: tuple[TargetSpec, ...] = tuple(((delay, 1),) for delay in range(1, 9))
    picked = (0, 2, 4, 6)  # n_surrogate_targets=4 相当の代表目標

    blocks = list(_picked_target_blocks(rows, psi_table, specs, picked))
    assert len(blocks) == 1, (
        "代表目標が1ブロックに収まっていません "
        f"(確保幅が chunk_size を読んでいる疑い): {[b.shape for b in blocks]}"
    )
    assert blocks[0].shape == (rows.n_samples, len(picked))

    # 確保幅は 128 MiB 予算だけで決まる: この規模なら len(picked) 以上に
    # 予算があるので常に1ブロックになる。
    assert rows.block_width(len(picked)) == len(picked)

    # 旧実装 (性能軸に従う) なら chunk_size=1 で 4 ブロックに割れていた。
    # その振る舞いを「確保軸ではこうならない」として明示的に固定する。
    assert rows.solve_width(1) == 1
    assert rows.block_width(len(picked)) != rows.solve_width(1)


def test_max_targets_bounded_axes_are_enumerated() -> None:
    """``max_targets`` が縛る軸の**列挙が正本**であり、各軸が独立に効く (D-34)。

    ``max_targets`` は目標数 (本) と ``ipc_heatmap`` のセル数 (セル) という
    **単位の違う2量**を同じ1つの数値で縛っている。これは意図的な選択だが、
    「何を縛っているか」がどこにも書かれていなかったことが指摘の実体だった
    (reviewer が到達できたのは ``ipc.py`` のコメントを読んだからで、決定の
    rule からは読めなかった)。列挙表 ``MAX_TARGETS_BOUNDED_AXES`` を正本に
    置き、ここで (a) 各軸が**独立に** ``ValueError`` へ到達できること、
    (b) 列挙表に載っている軸がすべて検査されていること、の両方を固定する。

    軸を足すときは列挙表とこのケース表の両方に足すこと。片方だけだと (b) の
    完全性検査が落ちる。
    """
    # 軸ごとに「その軸だけが上限を超える」設定を作る。
    # target_count: heatmap セル数 (4 x 60 = 240) は上限内、目標数 4075 が超える。
    target_count_cfg = IpcConfig(max_targets=1_000)
    assert (
        len(target_count_cfg.max_delay_by_degree)
        * max(target_count_cfg.max_delay_by_degree)
        <= target_count_cfg.max_targets
    ), "この設定では heatmap 軸が先に落ちます (ケース表の前提が崩れています)"
    assert count_targets(target_count_cfg) > target_count_cfg.max_targets

    # heatmap_cells: 目標数 (11_019) は上限内、セル数 (220_000) が超える。
    heatmap_cfg = IpcConfig(
        max_delay_by_degree=(11_000,) + (1,) * 19,
        max_variables=1,
        max_targets=200_000,
    )
    assert count_targets(heatmap_cfg) < heatmap_cfg.max_targets

    cases: dict[str, IpcConfig] = {
        AXIS_TARGET_COUNT: target_count_cfg,
        AXIS_HEATMAP_CELLS: heatmap_cfg,
    }
    assert set(cases) == set(MAX_TARGETS_BOUNDED_AXES), (
        "max_targets が縛る軸の列挙とケース表が食い違っています: "
        f"{sorted(set(cases) ^ set(MAX_TARGETS_BOUNDED_AXES))}"
    )

    states, inputs = _cached_states(0.9, 5, 600, 5)
    ctx = DiagnosticContext(washout=10, seed=CTX_SEED)
    for axis, cfg in cases.items():
        with pytest.raises(ValueError, match=axis):
            ipc(states, inputs, ctx=ctx, cfg=cfg)


def test_chunk_size_does_not_change_results() -> None:
    """``chunk_size`` は性能パラメータで、結果を変えてはいけない (仕様 §10-2)。

    他の設定フィールドは「変えたら出力が変わる」ことを要求されるが、これだけは
    **逆向きの要求**である。チャンク分割にバグ (列の取り違え、サロゲート乱数の
    チャンク依存) があるとここが落ちる。
    """
    states, inputs = _cached_states(0.9, 15, 4000, 5)
    ctx = DiagnosticContext(washout=100, seed=CTX_SEED)
    reference = ipc(states, inputs, ctx=ctx, cfg=SMALL_CFG)

    # F-03-2-018: 20_000 は bounded_chunk_size の 128MiB 予算を実際に超え、
    # 無条件切り詰め (キャップ) が発動する。「キャップが発動しない規模」しか
    # 通っていなかった既存テストに、発動する規模のケースを1件足す。
    n_samples = int(reference.params["n_samples"])
    capped_chunk_size = 20_000
    assert bounded_chunk_size(capped_chunk_size, n_samples) < capped_chunk_size, (
        "この chunk_size ではキャップが発動しません (テストの前提が崩れています)"
    )

    for chunk_size in (1, 7, 64, 1000, capped_chunk_size):
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
    psi_table = [orthonormal_basis(ramp, degree, UNIFORM_LEGENDRE) for degree in (1, 2)]
    t0 = 20
    n_samples = n_steps - t0
    # 状態行列を作らない (``RowAlignment`` が無いと書けない)。04a T3 以前は
    # ダミーの ``CapacityProblem`` (値を使わない ``np.zeros`` と、その特異な
    # Gram) を構築する必要があった。
    rows = RowAlignment(t0=t0, n_samples=n_samples)
    mean = float(np.mean(ramp))
    sigma = float(np.std(ramp))
    for delay in (1, 5, 20):
        spec: TargetSpec = ((delay, 1),)
        column = _target_column(rows, psi_table, spec)
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
        orthonormal_basis(series, degree, UNIFORM_LEGENDRE) for degree in (1, 2, 3)
    ]
    t0 = max(cfg.max_delay_by_degree)
    n_samples = n_steps - t0
    rows = RowAlignment(t0=t0, n_samples=n_samples)
    columns: FloatArray = np.empty((n_samples, len(specs)), dtype=np.float64)
    for index, spec in enumerate(specs):
        columns[:, index] = _target_column(rows, psi_table, spec)
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


def test_format_target_count_handles_huge_and_normal_totals() -> None:
    """``_format_target_count`` は巨大な多倍長整数でも例外を投げず、
    通常範囲では ``str(total)`` と一致する (F-03-3-021 / F-03-4-012 BLOCKER)。

    Python 3.11+ の int→str 変換には桁数上限 (既定4300桁) があり、素の
    ``str`` を直接呼ぶとこの上限自体が別の ``ValueError`` (`Exceeds the
    limit ... for integer string conversion`) に化けて、意図した『目標数が
    max_targets を超えました』というメッセージが運用者に届かない。この関数
    を丸ごと ``str`` に戻す変異を注入しても、この対比 (a) を固定しない限り
    検出できない (round4 レビューで実測: 43 passed のまま検出漏れ)。
    """
    huge = 10**5000
    # (a) 例外を投げない。
    formatted = _format_target_count(huge)
    # (b) 指数表記になる。
    assert formatted == "~1e5000 (桁数が大きいため概算の指数表記)", formatted
    # 対比: 素の str は桁数上限で ValueError になる (この対比が (a)(b) の
    # 存在理由そのもの)。
    with pytest.raises(ValueError, match="digits"):
        str(huge)

    # (c) 通常範囲では str(total) と一致する。
    for normal in (0, 1, 601, 123_456_789, 10**3000):
        assert _format_target_count(normal) == str(normal)


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


def test_max_degrees_bounds_the_psi_table_row_count() -> None:
    """``max_targets`` / ``heatmap_cells`` は次数の本数 (psi_table の行数) を
    弱くしか縛らない (CWE-789、F-03-2-013)。

    次数を1000本超・遅延を1本ずつにすると目標数もヒートマップ面積も
    小さいまま ``psi_table`` (次数 x 系列長) だけが線形に伸びる。実測:
    ``max_delay_by_degree=(1,)*1400``, T=200000 で psi_table 単独 peak RSS
    2.69GB (この設定は heatmap_cells=1400 が max_targets=200000 を超えず、
    round1 の検査を素通りしていた)。``max_degrees`` (既定20) が確保の前に
    独立して ``ValueError`` にする。

    F-03-3-018: round3 以降は ``count_targets`` 自身も
    ``_validate_combinatorial_bounds`` (``max_degrees`` を含む) を先頭で
    呼ぶため、``count_targets`` を直接呼んでも同じ理由で ``ValueError`` に
    なることも合わせて固定する (round2 時点は ``count_targets`` 単体では
    捕まらなかった設定だが、それは弱点でありここで塞いだ)。
    """
    cfg = IpcConfig(max_delay_by_degree=(1,) * 1400, max_variables=1)
    heatmap_cells = len(cfg.max_delay_by_degree) * max(cfg.max_delay_by_degree)
    assert heatmap_cells < cfg.max_targets, "heatmap_cells の検査でも捕まらない設定"

    with pytest.raises(ValueError, match="max_degrees"):
        count_targets(cfg)

    states, inputs = _cached_states(0.9, 5, 100, 5)
    with pytest.raises(ValueError, match="max_degrees"):
        ipc(
            states,
            inputs,
            ctx=DiagnosticContext(washout=10, seed=CTX_SEED),
            cfg=cfg,
        )


def test_max_variables_bounds_the_combinatorial_blowup_in_count_targets() -> None:
    """``count_targets`` の閉形式は ``max_variables`` が大きいと組合せ爆発する
    (CWE-400、F-03-2-014)。

    ``count_targets`` は防御そのもの (``max_targets`` の検査に到達する前に
    閉形式で先に数える) だが、その閉形式自体が
    ``math.comb(max_delay, n_vars)`` の多倍長整数計算で、``n_vars`` の上限を
    決める ``max_variables`` が大きいと防御の前段が防御対象と同じ失敗モードを
    持つ。実測: ``max_delay_by_degree=(1,)*D``, ``max_variables=D`` で
    D=4000 のとき ``count_targets`` が 373.73s かかり、その手前の
    ``_validate_config`` は 0.0001s で通過していた。``_validate_config`` が
    ``max_variables`` を独立に縛るため、危険な設定は ``count_targets`` に
    到達する前に ``ValueError`` になる。
    """
    cfg = IpcConfig(max_delay_by_degree=(4000,), max_variables=4000)
    states, inputs = _cached_states(0.9, 5, 100, 5)
    with pytest.raises(ValueError, match="max_variables"):
        ipc(
            states,
            inputs,
            ctx=DiagnosticContext(washout=10, seed=CTX_SEED),
            cfg=cfg,
        )


def test_max_delay_bit_length_bounds_combinatorial_blowup_in_count_targets() -> None:
    """``max_delay_by_degree`` の要素は値ではなく**桁数**でも組合せ爆発しうる
    (CWE-400、F-03-4-007、D-34 の4段目)。

    ``math.comb`` のコストは値ではなく桁数に対しておよそ ``digits^1.6`` で
    伸びる。既定の ``max_degrees<=32`` / ``max_variables<=20`` の下でも
    ``max_delay`` の桁数だけを伸ばせば ``count_targets`` を無防備に長時間
    走らせられていた (round4 レビュー実測: 10^60000 で 57.02秒)。
    ``_validate_combinatorial_bounds`` が桁数 (``bit_length``) を確保・計算
    より前に ``ValueError`` にすることを完了条件5として固定する。
    """
    huge_delay = 10**60000
    cfg = IpcConfig(
        max_delay_by_degree=(huge_delay,) * 32, max_variables=20, max_degrees=32
    )
    start = time.perf_counter()
    with pytest.raises(ValueError, match="bit"):
        count_targets(cfg)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, (
        f"count_targets が確保・計算の前に落ちていません ({elapsed:.3f}s)"
    )

    states, inputs = _cached_states(0.9, 5, 100, 5)
    with pytest.raises(ValueError, match="bit"):
        ipc(
            states,
            inputs,
            ctx=DiagnosticContext(washout=10, seed=CTX_SEED),
            cfg=cfg,
        )


def test_surrogate_base_matrix_never_exceeds_the_effective_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """代表目標行列 ``base`` は一括確保されず、chunk_size と同じ上限で分割
    される (CWE-789、F-03-2-015)。

    round1 の BLOCKER 修正はサロゲート列の生成 (``_iter_surrogate_chunks``)
    をチャンク化したが、その入力である代表目標行列 ``base`` は対象外だった。
    ``n_surrogate_targets`` に上限が無いため ``len(picked)`` は
    ``chunk_size`` と無関係に大きくなりうる (実測: K=400, T=1e6,
    ``n_surrogate_targets=K``, ``chunk_size=1`` で base 単独 peak RSS
    3.23GB)。ここでは ``np.empty`` に渡された列数の最大値を監視し、
    ``effective_chunk_size`` を超える確保が一度も起きないことを確認する。
    """
    states, inputs = _cached_states(0.9, 15, 4000, 5)
    ctx = DiagnosticContext(washout=100, seed=CTX_SEED)
    cfg = IpcConfig(
        max_delay_by_degree=(20, 8, 4),
        max_variables=2,
        n_surrogates=5,
        n_surrogate_targets=1_000_000,  # 上限が無いことをそのまま突く
        chunk_size=3,
    )
    n_samples = 4000 - 100  # t0=max(washout=100, max_delay=20)=100
    max_columns_seen = 0
    original_empty = np.empty

    def spying_empty(
        shape: tuple[int, ...] | int, dtype: type[np.float64] = np.float64
    ) -> FloatArray:
        nonlocal max_columns_seen
        if isinstance(shape, tuple) and len(shape) == 2 and shape[0] == n_samples:
            max_columns_seen = max(max_columns_seen, shape[1])
        return original_empty(shape, dtype=dtype)

    monkeypatch.setattr(np, "empty", spying_empty)
    ipc(states, inputs, ctx=ctx, cfg=cfg)
    assert max_columns_seen > 0, "監視対象の確保が一度も起きませんでした"
    assert max_columns_seen <= cfg.chunk_size, (
        f"chunk_size={cfg.chunk_size} を超える列数の確保がありました: "
        f"{max_columns_seen}"
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
    """``chunk_size`` / ``max_targets`` / ``max_degrees`` 以外の全フィールドは
    出力を変える。

    「設定したのに効いていない」除け。除外した3つは専用テストが担当する。

    - ``chunk_size``: 逆向きの要求 (``test_chunk_size_does_not_change_results``)
    - ``max_targets``: 上限であり、超えたときに ``ValueError`` にすることが
      唯一の観測可能な効果 (``test_target_enumeration_raises_instead_of_truncating``)
    - ``max_degrees``: 上限であり、超えたときに ``ValueError`` にすることが
      唯一の観測可能な効果 (F-03-2-013、
      ``test_max_degrees_bounds_the_psi_table_row_count``)

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
    covered = set(changed) | {"chunk_size", "max_targets", "max_degrees"}
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
        (IpcConfig(max_degrees=0), "max_degrees"),
        (
            IpcConfig(max_delay_by_degree=(1,) * 21, max_degrees=20),
            "max_degrees",
        ),
        (IpcConfig(max_variables=0), "max_variables"),
        (IpcConfig(max_variables=21), "max_variables"),
        (IpcConfig(max_degrees=33), "max_degrees"),
        (IpcConfig(alpha=-1.0), "alpha"),
        (IpcConfig(threshold_mode="bonferroni"), "threshold_mode"),
        (IpcConfig(chunk_size=0), "chunk_size"),
        (IpcConfig(max_targets=0), "max_targets"),
        (IpcConfig(n_surrogates=0), "n_surrogates"),
        (IpcConfig(n_surrogate_targets=0), "n_surrogate_targets"),
        (IpcConfig(surrogate_quantile=1.5), "surrogate_quantile"),
        # F-03-4-007 / D-34 の4段目: max_delay_by_degree の要素の桁数
        # (bit_length) にも独立した絶対上限がある (CWE-400)。
        (IpcConfig(max_delay_by_degree=(1 << 200,)), "bit"),
        # D-34 (04a T3 の改訂): 4段の絶対上限に加えて、共有予算 max_targets が
        # 縛る2軸 (単位が違う) も1つの node id で固定する。列挙は
        # MAX_TARGETS_BOUNDED_AXES が正本で、
        # test_max_targets_bounded_axes_are_enumerated が完全性を守る。
        (IpcConfig(max_targets=1_000), AXIS_TARGET_COUNT),
        (
            IpcConfig(
                max_delay_by_degree=(11_000,) + (1,) * 19,
                max_variables=1,
                max_targets=200_000,
            ),
            AXIS_HEATMAP_CELLS,
        ),
    ],
)
def test_out_of_range_config_raises(cfg: IpcConfig, message: str) -> None:
    """設定の値域違反は既定へフォールバックせず ``ValueError`` (D-09)。"""
    states, inputs = _cached_states(0.9, 15, 4000, 5)
    with pytest.raises(ValueError, match=message):
        ipc(states, inputs, ctx=DiagnosticContext(seed=CTX_SEED), cfg=cfg)


def test_max_variables_boundary_value_20_is_accepted() -> None:
    """``max_variables`` の上限ちょうど (20) は成功する (境界値、F-03-3-022)。

    上限違反 (21) と下限違反 (0) は ``test_out_of_range_config_raises`` が
    カバーするが、成功すべき境界値 20 のテストが無かった。off-by-one で
    20 を誤って拒否しても、境界値の正常系テストが無ければ検出できない
    (実測: `_MAX_VARIABLES_FOR_COUNT` の比較を `>` から `>=` に変異させても
    tests/test_diagnostics_ipc.py の既存テストは1件も落ちない)。
    """
    states, inputs = _cached_states(0.9, 15, 4000, 5)
    cfg = dataclasses.replace(SMALL_CFG, max_variables=20)
    result = ipc(states, inputs, ctx=DiagnosticContext(seed=CTX_SEED), cfg=cfg)
    assert np.isfinite(result.scalars["ipc_total"])


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

"""ESP 判定・条件付き Lyapunov 指数のテスト (受け入れ条件3・6 / D-15・D-16・D-18)。

このファイルは ``rc_basics_lab.reservoir`` を一切 import しない。ESN で作った
状態でも外部生成の状態でも同じ診断が動くこと (移植性) が 02 の受け入れ条件6 で
あり、テスト側も同じ制約で書くことでそれを実演する。
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

if TYPE_CHECKING:  # pragma: no cover - 型検査時のみ必要
    from _typeshed import DataclassInstance

from rc_basics_lab.diagnostics.base import (
    Diagnostic,
    DiagnosticContext,
    DiagnosticResult,
    StatePropagator,
)
from rc_basics_lab.diagnostics.esp import (
    DEFAULT_ESP,
    DEFAULT_LYAPUNOV,
    EspConfig,
    LyapunovConfig,
    conditional_lyapunov,
    esp_convergence,
)
from rc_basics_lab.diagnostics.timescale import (
    DEFAULT_TIMESCALE,
    TimescaleConfig,
    autocorrelation_time,
)
from rc_basics_lab.types import FloatArray

N_STEPS = 800
N_UNITS = 10


def _pair_with_decay(
    decay: float, n_steps: int = N_STEPS, n_units: int = N_UNITS, seed: int = 3
) -> tuple[FloatArray, FloatArray]:
    """RMS/ユニット距離が厳密に ``decay ** t`` になる2軌道を作る。

    参照軌道そのものは有界な乱数列でよい (ESP 判定は2軌道の**差**しか見ない)。
    距離を解析的に固定できるので、閾値・窓・当てはめ区間の効き方を誤差なしで
    検査できる。
    """
    rng = np.random.default_rng(seed)
    states: FloatArray = rng.uniform(-1.0, 1.0, size=(n_steps, n_units))
    direction: FloatArray = rng.standard_normal(n_units)
    direction = direction / float(np.linalg.norm(direction)) * np.sqrt(n_units)
    factors: FloatArray = decay ** np.arange(n_steps, dtype=np.float64)
    companion: FloatArray = states + factors[:, None] * direction
    return states, companion


def _driven_linear_system(
    rho: float, n_steps: int = 600, n_units: int = 8, seed: int = 7
) -> tuple[FloatArray, FloatArray, StatePropagator]:
    """駆動された線形系 ``x[t+1] = A x[t] + c[t]`` (``A = rho * Q``、``Q`` は直交)。

    ``c[t]`` を参照軌道から逆算するので、参照軌道は有界な任意の系列にできる
    (実 ESN で入力が状態を有界に保つのと同じ役割)。Jacobian が全時刻 ``A`` で
    特異値がすべて ``rho`` なので、条件付き Lyapunov 指数は厳密に ``log rho``、
    2軌道の距離は厳密に ``rho ** t`` になる。

    Returns:
        ``(参照軌道, 第2軌道, 伝播器)``。
    """
    rng = np.random.default_rng(seed)
    orthogonal, _ = np.linalg.qr(rng.standard_normal((n_units, n_units)))
    matrix: FloatArray = rho * orthogonal
    states: FloatArray = rng.uniform(-1.0, 1.0, size=(n_steps, n_units))
    offsets: FloatArray = states[1:] - states[:-1] @ matrix.T

    def propagator(x: FloatArray, t: int) -> FloatArray:
        propagated: FloatArray = matrix @ x + offsets[t]
        return propagated

    separation: FloatArray = rng.standard_normal(n_units)
    separation = separation / float(np.linalg.norm(separation)) * np.sqrt(n_units)
    companion: FloatArray = np.empty_like(states)
    for index in range(n_steps):
        companion[index] = states[index] + separation
        separation = matrix @ separation
    return states, companion, propagator


def _driven_tanh_system(
    n_steps: int = 400, n_units: int = 6, gain: float = 1.5, seed: int = 11
) -> tuple[FloatArray, StatePropagator, StatePropagator]:
    """駆動された非線形系 ``x[t+1] = tanh(W x[t] + c[t+1])``。

    ``X[t]`` は ``c[t]`` を処理した**後**の状態という ESN と同じ時間規約で作る。
    そのため正しい伝播器は ``c[t+1]`` を使う。1つずれた (``c[t]`` を使う) 伝播器を
    第3の戻り値として返し、D-18 の整合検査の被験体にする。

    Returns:
        ``(状態系列, 正しい伝播器, 1ステップずれた伝播器)``。
    """
    rng = np.random.default_rng(seed)
    weights: FloatArray = (
        rng.standard_normal((n_units, n_units)) / np.sqrt(n_units) * gain
    )
    drive: FloatArray = rng.uniform(-0.5, 0.5, size=(n_steps, n_units))

    def propagator(x: FloatArray, t: int) -> FloatArray:
        propagated: FloatArray = np.tanh(weights @ x + drive[t + 1])
        return propagated

    def off_by_one_propagator(x: FloatArray, t: int) -> FloatArray:
        propagated: FloatArray = np.tanh(weights @ x + drive[t])
        return propagated

    states: FloatArray = np.empty((n_steps, n_units), dtype=np.float64)
    states[0] = np.tanh(weights @ np.zeros(n_units) + drive[0])
    for index in range(n_steps - 1):
        states[index + 1] = propagator(states[index], index)
    return states, propagator, off_by_one_propagator


def _slightly_off_propagator(
    propagator: StatePropagator, offset: float
) -> StatePropagator:
    """RMS/ユニット距離で厳密に ``offset`` だけずれた伝播器 (許容誤差の検査用)。"""

    def shifted(x: FloatArray, t: int) -> FloatArray:
        propagated: FloatArray = propagator(x, t) + offset
        return propagated

    return shifted


# --- 受け入れ条件6: 外部生成系列で動く ------------------------------------


def test_works_on_externally_generated_states() -> None:
    """ESN を一切使わずに作った状態系列で3診断がそのまま動く (受け入れ条件6)。

    ``_driven_linear_system`` は numpy だけで閉じており ``reservoir`` に触れない。
    さらに ESP 判定と Lyapunov 指数の符号が一致すること (収束するなら λ<0)
    まで確認する。
    """
    for rho, expect_converged in ((0.9, True), (1.05, False)):
        states, companion, propagator = _driven_linear_system(rho)
        ctx = DiagnosticContext(
            companion_states=(companion,), propagator=propagator, washout=100
        )
        esp = esp_convergence(states, ctx=ctx)
        lyapunov = conditional_lyapunov(states, ctx=ctx)
        timescale = autocorrelation_time(states, ctx=ctx, cfg=TimescaleConfig(50))

        assert bool(esp.scalars["converged"]) is expect_converged
        assert esp.scalars["decay_rate_per_step"] == pytest.approx(
            math.log(rho), rel=0.05
        )
        assert (lyapunov.scalars["lyapunov_per_step"] < 0.0) is expect_converged
        assert np.isfinite(lyapunov.scalars["lyapunov_per_step"])
        assert np.isfinite(timescale.scalars["tau_censored"])


def test_results_convert_to_flat_rows() -> None:
    """3診断の ``to_row`` が params/scalars のキー衝突なしに通る。"""
    states, companion, propagator = _driven_linear_system(0.9)
    ctx = DiagnosticContext(companion_states=(companion,), propagator=propagator)
    for result in (
        esp_convergence(states, ctx=ctx),
        conditional_lyapunov(states, ctx=ctx),
        autocorrelation_time(states, ctx=ctx, cfg=TimescaleConfig(50)),
    ):
        row = result.to_row()
        assert row["diagnostic"] == result.name


# --- D-16: 距離の定義と判定規則 --------------------------------------------


def test_distance_is_rms_per_unit_and_independent_of_n_units() -> None:
    """RMS/ユニット距離なので、ユニット数を変えても距離が変わらない (D-16)。

    ``sqrt(N)`` 正規化を落とすと N=10 と N=40 で距離が2倍ずれ、閾値が N 依存に
    なる (= N の違う系へ移植できなくなる)。
    """
    values = []
    for n_units in (10, 40):
        states, companion = _pair_with_decay(0.97, n_units=n_units)
        result = esp_convergence(
            states, ctx=DiagnosticContext(companion_states=(companion,))
        )
        values.append(result.scalars["d_initial"])
    assert values[0] == pytest.approx(1.0, rel=1e-12)
    assert values[0] == pytest.approx(values[1], rel=1e-12)


def test_decay_rate_matches_prescribed_decay() -> None:
    """距離が厳密に ``0.97 ** t`` の対で ``decay_rate_per_step == log 0.97``。"""
    states, companion = _pair_with_decay(0.97)
    result = esp_convergence(
        states, ctx=DiagnosticContext(companion_states=(companion,))
    )
    assert result.scalars["decay_rate_per_step"] == pytest.approx(
        math.log(0.97), rel=1e-4
    )
    assert result.scalars["n_fit_points"] == pytest.approx(float(N_STEPS - 50))


def _esp_config_with_abs_tol(tol: float) -> EspConfig:
    """``abs_tol`` だけを振る cfg (``rel_tol`` は 0 にして交絡を消す)。"""
    return EspConfig(abs_tol=tol, rel_tol=0.0)


def _esp_config_with_rel_tol(tol: float) -> EspConfig:
    """``rel_tol`` だけを振る cfg (``abs_tol`` は 0 にして交絡を消す)。"""
    return EspConfig(abs_tol=0.0, rel_tol=tol)


def test_verdict_is_monotone_in_tolerance() -> None:
    """閾値を緩めると判定が True→False に戻らない (D-16 guard)。

    判定規則 ``d_tail <= max(abs_tol, rel_tol * d_initial)`` の単調性は、
    「閾値を動かしたら判定が反転した」という報告を解釈可能にする最低条件。
    中央値ではなく末尾の最小値を取る等の実装に差し替わると壊れる。
    両閾値で False と True の両方が現れることも確認し、検査が空振りしていない
    ことを固定する。
    """
    states, companion = _pair_with_decay(0.97)
    ctx = DiagnosticContext(companion_states=(companion,))
    tolerances = (1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6)

    for make_cfg, field_name in (
        (_esp_config_with_abs_tol, "abs_tol"),
        (_esp_config_with_rel_tol, "rel_tol"),
    ):
        verdicts = [
            bool(
                esp_convergence(states, ctx=ctx, cfg=make_cfg(tol)).scalars["converged"]
            )
            for tol in tolerances
        ]
        assert verdicts == sorted(verdicts), (
            f"{field_name} を緩めると判定が反転しました: {verdicts}"
        )
        assert set(verdicts) == {False, True}, (
            f"{field_name} の掃引が閾値をまたいでいません: {verdicts}"
        )


def test_worst_pair_decides_the_verdict() -> None:
    """複数ペアでは最悪値で判定する (収束ペアが混ざっても True にならない)。"""
    states, converging = _pair_with_decay(0.97)
    same_states, diverging = _pair_with_decay(1.0)  # 同じ seed = 同じ参照軌道
    assert np.array_equal(states, same_states)
    ctx = DiagnosticContext(companion_states=(converging, diverging))
    result = esp_convergence(states, ctx=ctx)
    assert result.scalars["n_pairs"] == pytest.approx(2.0)
    assert result.scalars["converged"] == pytest.approx(0.0)
    assert result.arrays["distance_all"].shape == (2, N_STEPS)
    # 返す曲線は最悪ペア (= 収束しない方) のもの
    assert result.arrays["distance"][-1] == pytest.approx(1.0, rel=1e-12)


def test_missing_companion_states_raise() -> None:
    """第2軌道が渡っていないのに判定だけ返る事故を殺す (D-16)。"""
    states, _ = _pair_with_decay(0.97)
    with pytest.raises(ValueError, match="companion_states"):
        esp_convergence(states)


def test_series_shorter_than_window_raises() -> None:
    """``T < washout + window`` は ValueError (末尾窓が取れない)。"""
    states, companion = _pair_with_decay(0.97, n_steps=250)
    ctx = DiagnosticContext(companion_states=(companion,), washout=100)
    with pytest.raises(ValueError, match="window"):
        esp_convergence(states, ctx=ctx)


def test_decay_rate_is_nan_when_no_point_is_above_floor() -> None:
    """当てはめ点が2点未満なら ``nan`` (「測れなかった」を 0 と区別する)。"""
    states, companion = _pair_with_decay(0.5)
    result = esp_convergence(
        states, ctx=DiagnosticContext(companion_states=(companion,))
    )
    assert result.scalars["n_fit_points"] < 2.0
    assert math.isnan(result.scalars["decay_rate_per_step"])


# --- D-18: 条件付き Lyapunov 指数 ------------------------------------------


@pytest.mark.parametrize("rho", [0.5, 0.9, 1.1], ids=["rho0.5", "rho0.9", "rho1.1"])
def test_matches_analytic_exponent_for_linear_map(rho: float) -> None:
    """``x -> A x + c[t]`` で λ が ``log rho`` に一致する (D-18 guard)。

    ``A`` の特異値がすべて ``rho`` なので、どの方向の摂動も1ステップで厳密に
    ``rho`` 倍される。摂動法・delta=1e-8・再正規化間隔1 という既定値が、解析解と
    直接照合できる精度で動いていることを固定する (相対誤差 1e-6 以内)。
    ``max_observed_growth`` の許容が緩いのは、これが平均ではなく 499 区間の
    最悪値であり、``X[t] + delta`` の丸め (相対 1e-8 程度) がそのまま出るため。
    """
    states, _, propagator = _driven_linear_system(rho)
    result = conditional_lyapunov(
        states, ctx=DiagnosticContext(propagator=propagator, washout=100)
    )
    assert result.scalars["lyapunov_per_step"] == pytest.approx(
        math.log(rho), rel=1.0e-6
    )
    assert result.scalars["n_intervals"] == pytest.approx(float(600 - 1 - 100))
    assert result.scalars["max_observed_growth"] == pytest.approx(rho, rel=1e-6)


def _decoupled_quadratic_system(
    n_steps: int, n_units: int, x0: float, a: float, k: float
) -> tuple[FloatArray, StatePropagator]:
    """状態が全ユニット・全時刻で ``x0`` に留まる決定的な不動点系。

    ``f(x) = a*x + k*x**2`` はユニットごとに完全に独立 (非結合) な写像で、
    ``x0 = (1 - a) / k`` を ``f`` の不動点に選んであるため、参照軌道は
    定数 ``x0`` のまま厳密に ``propagator`` と整合する。局所的な線形ゲインは
    解析的に ``f'(x0) = a + 2*k*x0 = 2 - a`` になる。

    このテストが ``_driven_linear_system`` (直交行列による厳密なアフィン系)
    ではなくこの系を使うのは、直交系では ``growth = separation_norm / scale
    / delta`` の計算式から ``scale`` の値が厳密に相殺してしまい、``n_units``
    をいくら振っても ``scale = sqrt(n_units)`` を落とす変異を検出できない
    ことを実測で確認したため (アフィン写像には曲率が無く、摂動の絶対量
    ``delta * scale`` がどんな値でも成長率は不変)。この系は2次の曲率項
    ``k*x**2`` を持つため、``delta`` が表す「ユニットあたりの RMS 摂動量」を
    誤って ``n_units`` 倍 (``sqrt(n_units)`` 倍ではなく) にする変異を入れると、
    実際の摂動振幅が N とともに余計に大きくなり、曲率の効きが N に依存して
    しまう。``delta`` を十分小さく保つことで、正しい実装では曲率補正が
    ``O(delta**2)`` (N非依存) に留まることも実測で確認済み。
    """
    states: FloatArray = np.full((n_steps, n_units), x0, dtype=np.float64)

    def propagator(x: FloatArray, t: int) -> FloatArray:
        propagated: FloatArray = a * x + k * x**2
        return propagated

    return states, propagator


def test_lyapunov_per_step_is_independent_of_n_units() -> None:
    """``lyapunov_per_step`` はユニット数 N に依存しない (F-1-016, D-16 と対になる)。

    ``conditional_lyapunov`` 内の ``scale = float(np.sqrt(n_units))``
    (delta の RMS/ユニット解釈) を検査するテストが存在しなかった。
    ``_driven_linear_system`` を使う既存テスト
    (``test_matches_analytic_exponent_for_linear_map``) はアフィン系のため
    ``scale`` の値に関わらず厳密に解析解と一致してしまい、この class の
    バグ (``scale`` の sqrt を落とす等) を原理的に検出できない (実測:
    ``scale = float(n_units)`` という変異を入れても337件中1件も落ちなかった)。

    ``_decoupled_quadratic_system`` (曲率を持つ非結合系) を使い、極端に異なる
    ``n_units`` (8 と 2000) で同じ ``rho`` を与えて ``lyapunov_per_step`` を
    比較する。正しい実装では両者の相対差は 0.1% 未満に収まる一方 (実測)、
    ``scale`` の sqrt を落とす変異を入れると相対差が約 21% まで広がることを
    実測で確認した (安全手順に従いサンドボックスで確認・復元済み)。
    ``test_distance_is_rms_per_unit_and_independent_of_n_units`` と対になる。
    """
    a, k, x0 = 0.5, 1.0, 0.5  # x0 = (1 - a) / k: f(x) = a*x + k*x**2 の不動点
    delta = 1.0e-3
    cfg = LyapunovConfig(
        delta=delta, renorm_interval=1, max_growth=1.0e6, propagator_tol=1.0e-9
    )
    values = []
    for n_units in (8, 2000):
        states, propagator = _decoupled_quadratic_system(20, n_units, x0, a, k)
        result = conditional_lyapunov(
            states,
            ctx=DiagnosticContext(propagator=propagator, washout=2, seed=123),
            cfg=cfg,
        )
        values.append(result.scalars["lyapunov_per_step"])
    assert values[0] == pytest.approx(math.log(2.0 - a), rel=1e-2)
    assert values[0] == pytest.approx(values[1], rel=1e-2)


def test_lyapunov_per_time_divides_by_dt() -> None:
    """``ctx.dt`` は時間正規化にだけ効く (ステップ単位の値は変わらない)。"""
    states, _, propagator = _driven_linear_system(0.9)
    result = conditional_lyapunov(
        states, ctx=DiagnosticContext(propagator=propagator, dt=0.5)
    )
    assert result.scalars["lyapunov_per_time"] == pytest.approx(
        2.0 * result.scalars["lyapunov_per_step"], rel=1e-12
    )


def test_missing_propagator_raises() -> None:
    """``ctx.propagator`` が無ければ ValueError (伝播器なしで指数は出せない)。"""
    states, _, _ = _driven_linear_system(0.9)
    with pytest.raises(ValueError, match="propagator"):
        conditional_lyapunov(states)


def test_inconsistent_propagator_raises() -> None:
    """入力インデックスが1つずれた伝播器を実行時に落とす (D-18 guard)。

    ``X[t]`` は ``u[t]`` を処理した後の状態なので、正しい伝播器は ``u[t+1]`` を
    使う。``u[t]`` を使うと λ は"それらしい値"で出てしまい、レビューでは
    気づけない。既定で有効な整合検査だけがこれを落とす。
    """
    states, propagator, off_by_one = _driven_tanh_system()
    ctx_ok = DiagnosticContext(propagator=propagator)
    ctx_bad = DiagnosticContext(propagator=off_by_one)

    with pytest.raises(ValueError, match="propagator"):
        conditional_lyapunov(states, ctx=ctx_bad)

    # 検査を切ると、同じ配線ミスが有限の λ として通ってしまう。
    # (この系では max_growth もたまたま引っかかるので、伝播器の整合検査だけを
    #  切り分けるために上限を上げている。入力が弱い条件では成長率が既定の
    #  max_growth に届かず、整合検査だけが唯一の防波堤になる。)
    permissive = LyapunovConfig(check_propagator=False, max_growth=1e12)
    silently_wrong = conditional_lyapunov(states, ctx=ctx_bad, cfg=permissive)
    correct = conditional_lyapunov(states, ctx=ctx_ok)
    assert np.isfinite(silently_wrong.scalars["lyapunov_per_step"])
    assert silently_wrong.scalars["lyapunov_per_step"] != pytest.approx(
        correct.scalars["lyapunov_per_step"], rel=1e-6
    )


def test_propagator_returning_wrong_shape_raises() -> None:
    """伝播器が状態と異なる形状の配列を返すと ``ValueError`` (F-1-017, D-18)。

    ``_check_propagator_consistency`` の ``predicted.shape != states[t].shape``
    分岐が未検査だった。実装ミスとして十分あり得るケース (例: 末尾を落として
    返す) を、``_rms_distance`` の減算より先に落とせることを確認する。
    """
    states, _, propagator = _driven_linear_system(0.9)

    def wrong_shape_propagator(x: FloatArray, t: int) -> FloatArray:
        return x[:-1]

    ctx = DiagnosticContext(propagator=wrong_shape_propagator)
    with pytest.raises(ValueError, match="形状"):
        conditional_lyapunov(states, ctx=ctx)


def test_propagator_tolerance_is_rms_per_unit() -> None:
    """整合検査の許容量は RMS/ユニット距離で解釈される。"""
    states, propagator, _ = _driven_tanh_system()
    ctx = DiagnosticContext(propagator=_slightly_off_propagator(propagator, 1.0e-6))
    with pytest.raises(ValueError, match="propagator"):
        conditional_lyapunov(states, ctx=ctx)
    result = conditional_lyapunov(
        states, ctx=ctx, cfg=LyapunovConfig(propagator_tol=1.0e-5)
    )
    assert np.isfinite(result.scalars["lyapunov_per_step"])


def test_unknown_lyapunov_method_raises() -> None:
    """未実装の method を黙って受理しない (解析 Jacobian 版の差し込み口, D-18)。"""
    states, _, propagator = _driven_linear_system(0.9)
    with pytest.raises(ValueError, match="perturbation"):
        conditional_lyapunov(
            states,
            ctx=DiagnosticContext(propagator=propagator),
            cfg=LyapunovConfig(method="jacobian"),
        )


def test_growth_beyond_max_growth_raises() -> None:
    """再正規化間隔が長すぎて線形域を外れたら ValueError (D-18)。

    rho=1.1 を 100 ステップ再正規化なしで進めると成長率は 1.1**100 ≈ 1.4e4 で、
    既定の ``max_growth=1e3`` を超える。黙って値を返すと、線形域を外れた
    (= 意味を失った) 推定値が CSV に載る。
    """
    states, _, propagator = _driven_linear_system(1.1)
    with pytest.raises(ValueError, match="max_growth"):
        conditional_lyapunov(
            states,
            ctx=DiagnosticContext(propagator=propagator),
            cfg=LyapunovConfig(renorm_interval=100),
        )


# --- D-15: 設定は既定値つきキーワード引数 cfg で渡す -----------------------

_CONFIG_TYPES: tuple[tuple[str, Diagnostic, type[object]], ...] = (
    ("esp_convergence", esp_convergence, EspConfig),
    ("conditional_lyapunov", conditional_lyapunov, LyapunovConfig),
    ("autocorrelation_time", autocorrelation_time, TimescaleConfig),
)
"""``(名前, 診断, 対応する設定クラス)``。

``Diagnostic`` として持てるのは、``cfg`` が**既定値つき**だから (D-01 の署名
契約を一切変えていないことがここでも型で示される)。
"""


def _field_names(cls: type[object]) -> set[str]:
    """dataclass のフィールド名集合 (``Any`` を書かずに ``fields`` を呼ぶ)。"""
    return {f.name for f in fields(cast("type[DataclassInstance]", cls))}


def test_esp_config_is_passed_as_defaulted_keyword() -> None:
    """診断の設定は ``cfg`` (既定値つき keyword-only) で渡る (D-15 guard)。

    D-01 は「X/u/y/ctx 以外に**必須**引数を作らない」であり、既定値つきの
    ``cfg`` はこれに適合する。逆に判定基準を ``DiagnosticContext`` へ足すと、
    05 まで進んだ時点で ctx が全診断の設定の union になる。境界が守られて
    いることを、ctx のフィールド名と各 cfg のフィールド名が交わらないことで
    機械的に固定する。

    F-1-002: この検査対象は ``_CONFIG_TYPES`` という静的タプルから引いており、
    新しい診断モジュールを追加してもここへ1行足すのを忘れると検査対象から
    静かに外れる (D-01 の guard が pkgutil 自動列挙なのと対照的)。この
    テストは「``cfg`` という名前の引数が、期待した具体的な設定クラス
    (``EspConfig`` 等) の既定値を持つこと」まで確認する、02 の3診断に固有の
    詳細な検査として残す。新しい診断へ自動的に効く、より一般的な境界検査
    (「追加引数はすべて keyword-only かつ既定値つきで、既定値が dataclass
    なら ctx とフィールド名が重ならない」) は
    ``tests/test_diagnostics_base.py::test_extra_diagnostic_parameters_are_keyword_only_and_do_not_overlap_ctx``
    が pkgutil 列挙側で担う。
    """
    ctx_field_names = _field_names(DiagnosticContext)
    for name, func, config_type in _CONFIG_TYPES:
        parameters = inspect.signature(func).parameters
        assert "cfg" in parameters, f"{name}: cfg 引数がありません"
        cfg_param = parameters["cfg"]
        assert cfg_param.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{name}: cfg が keyword-only ではありません"
        )
        assert isinstance(cfg_param.default, config_type), (
            f"{name}: cfg に {config_type.__name__} の既定値がありません"
        )
        overlap = ctx_field_names & _field_names(config_type)
        assert not overlap, (
            f"{config_type.__name__} の判定基準が DiagnosticContext にも "
            f"あります (D-15 の境界違反): {sorted(overlap)}"
        )


# --- 有効性: 全設定フィールドが出力を変える --------------------------------


@dataclass(frozen=True, slots=True)
class _ConfigCase:
    """「このフィールドを変えたら出力が変わる」1件ぶんの検査。

    出力は「返った scalars」か「送出された ValueError」のどちらかで、
    ``method`` / ``max_growth`` のように値域が実質1点のフィールドは後者の
    チャネルで効きを確認する (``config.py`` 側の CHANNEL_ERROR と同じ考え方)。
    """

    config_type: type[object]
    field_name: str
    base: object
    changed: object
    call: Callable[[object], DiagnosticResult]


@dataclass(frozen=True, slots=True)
class _Outcome:
    """診断1回ぶんの「出力」。scalars か ValueError のどちらか。"""

    error: str | None
    scalars: Mapping[str, float]

    def differs_from(self, other: _Outcome) -> bool:
        if (self.error is None) != (other.error is None):
            return True
        if self.error is not None:
            return self.error != other.error
        if self.scalars.keys() != other.scalars.keys():
            return True
        for key, value in self.scalars.items():
            counterpart = other.scalars[key]
            if math.isnan(value) and math.isnan(counterpart):
                continue
            # 1e-9 より小さい差は「効いた」と見なさない (丸めの揺れと区別する)
            if value != pytest.approx(counterpart, rel=1e-9, abs=1e-12):
                return True
        return False


def _outcome(case: _ConfigCase, cfg: object) -> _Outcome:
    try:
        result = case.call(cfg)
    except ValueError as exc:
        return _Outcome(error=str(exc), scalars={})
    return _Outcome(error=None, scalars=dict(result.scalars))


def _esp_call(
    states: FloatArray, companion: FloatArray
) -> Callable[[object], DiagnosticResult]:
    ctx = DiagnosticContext(companion_states=(companion,))

    def call(cfg: object) -> DiagnosticResult:
        assert isinstance(cfg, EspConfig)
        return esp_convergence(states, ctx=ctx, cfg=cfg)

    return call


def _lyapunov_call(
    states: FloatArray, propagator: StatePropagator
) -> Callable[[object], DiagnosticResult]:
    ctx = DiagnosticContext(propagator=propagator)

    def call(cfg: object) -> DiagnosticResult:
        assert isinstance(cfg, LyapunovConfig)
        return conditional_lyapunov(states, ctx=ctx, cfg=cfg)

    return call


def _timescale_call(states: FloatArray) -> Callable[[object], DiagnosticResult]:
    def call(cfg: object) -> DiagnosticResult:
        assert isinstance(cfg, TimescaleConfig)
        return autocorrelation_time(states, cfg=cfg)

    return call


def _build_config_cases() -> tuple[_ConfigCase, ...]:
    esp_states, esp_companion = _pair_with_decay(0.97)
    esp_call = _esp_call(esp_states, esp_companion)
    tanh_states, tanh_propagator, tanh_off_by_one = _driven_tanh_system()
    lyapunov_call = _lyapunov_call(tanh_states, tanh_propagator)
    return (
        _ConfigCase(
            EspConfig,
            "abs_tol",
            EspConfig(rel_tol=0.0),
            EspConfig(rel_tol=0.0, abs_tol=1e-14),
            esp_call,
        ),
        _ConfigCase(
            EspConfig,
            "rel_tol",
            EspConfig(abs_tol=0.0),
            EspConfig(abs_tol=0.0, rel_tol=1e-12),
            esp_call,
        ),
        _ConfigCase(EspConfig, "window", DEFAULT_ESP, EspConfig(window=500), esp_call),
        _ConfigCase(
            EspConfig, "fit_skip", DEFAULT_ESP, EspConfig(fit_skip=400), esp_call
        ),
        _ConfigCase(EspConfig, "floor", DEFAULT_ESP, EspConfig(floor=1e-6), esp_call),
        _ConfigCase(
            LyapunovConfig,
            "method",
            DEFAULT_LYAPUNOV,
            LyapunovConfig(method="jacobian"),
            lyapunov_call,
        ),
        _ConfigCase(
            LyapunovConfig,
            "delta",
            DEFAULT_LYAPUNOV,
            LyapunovConfig(delta=1e-2),
            lyapunov_call,
        ),
        _ConfigCase(
            LyapunovConfig,
            "renorm_interval",
            DEFAULT_LYAPUNOV,
            LyapunovConfig(renorm_interval=5),
            lyapunov_call,
        ),
        _ConfigCase(
            LyapunovConfig,
            "max_growth",
            DEFAULT_LYAPUNOV,
            LyapunovConfig(max_growth=1e-3),
            lyapunov_call,
        ),
        _ConfigCase(
            LyapunovConfig,
            "check_propagator",
            # 伝播器が1ステップずれている条件で、検査の有無だけを振る。
            # max_growth を上げてあるのは、この系ではそちらでも落ちてしまい
            # check_propagator 単体の効きが見えなくなるため。
            LyapunovConfig(max_growth=1e12),
            LyapunovConfig(max_growth=1e12, check_propagator=False),
            _lyapunov_call(tanh_states, tanh_off_by_one),
        ),
        _ConfigCase(
            LyapunovConfig,
            "propagator_tol",
            DEFAULT_LYAPUNOV,
            LyapunovConfig(propagator_tol=1e-5),
            _lyapunov_call(
                tanh_states, _slightly_off_propagator(tanh_propagator, 1e-6)
            ),
        ),
        _ConfigCase(
            TimescaleConfig,
            "max_lag",
            TimescaleConfig(max_lag=50),
            TimescaleConfig(max_lag=100),
            _timescale_call(esp_states),
        ),
    )


CONFIG_CASES = _build_config_cases()


def test_esp_config_fields_change_output() -> None:
    """3設定クラスの全フィールドが出力 (scalars か例外) を実際に変える。

    本リポジトリ最大の失敗モードは「設定したのに効いていない」であり、
    ``test_config_wiring.py`` は YAML→実験の経路でこれを殺している。診断の設定は
    ``ExperimentConfig`` の外を通る (D-15) ため経路が別で、同じ検査を診断単体の
    レベルでも掛けないと「YAML から診断設定が届いていない」が黙って通る。
    """
    for case in CONFIG_CASES:
        base = _outcome(case, case.base)
        changed = _outcome(case, case.changed)
        assert base.differs_from(changed), (
            f"{case.config_type.__name__}.{case.field_name} を変えても "
            f"出力が変わりません: {base} -> {changed}"
        )


def test_all_config_fields_have_a_case() -> None:
    """設定クラスにフィールドを足したら上のケース登録を強制する。

    ケース未登録のまま「全フィールドで出力が変わる」を名乗ると、検査の
    網羅性が静かに落ちる (``test_all_config_fields_are_covered`` と同じ役割)。
    """
    for _, _func, config_type in _CONFIG_TYPES:
        expected = _field_names(config_type)
        covered = {
            case.field_name for case in CONFIG_CASES if case.config_type is config_type
        }
        assert covered == expected, (
            f"{config_type.__name__}: ケース未登録のフィールドがあります: "
            f"{sorted(expected - covered)} / 余剰: {sorted(covered - expected)}"
        )


def test_default_configs_match_documented_values() -> None:
    """既定値 (D-16 / D-18 に明記した値) が動いていないことを固定する。"""
    assert (
        EspConfig(abs_tol=1e-6, rel_tol=1e-3, window=200, fit_skip=50, floor=1e-14)
        == DEFAULT_ESP
    )
    assert (
        LyapunovConfig(
            method="perturbation",
            delta=1e-8,
            renorm_interval=1,
            max_growth=1e3,
            check_propagator=True,
            propagator_tol=1e-10,
        )
        == DEFAULT_LYAPUNOV
    )
    assert TimescaleConfig(max_lag=200) == DEFAULT_TIMESCALE


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (EspConfig(abs_tol=-1.0), "abs_tol"),
        (EspConfig(rel_tol=-1.0), "rel_tol"),
        (EspConfig(window=0), "window"),
        (EspConfig(fit_skip=-1), "fit_skip"),
        (EspConfig(floor=0.0), "floor"),
    ],
    ids=["abs_tol", "rel_tol", "window", "fit_skip", "floor"],
)
def test_invalid_esp_config_raises(cfg: EspConfig, message: str) -> None:
    """値域検証は設定 dataclass ではなく使う側で行う (慣習どおり)。"""
    states, companion = _pair_with_decay(0.97)
    with pytest.raises(ValueError, match=message):
        esp_convergence(
            states, ctx=DiagnosticContext(companion_states=(companion,)), cfg=cfg
        )


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (LyapunovConfig(delta=0.0), "delta"),
        (LyapunovConfig(renorm_interval=0), "renorm_interval"),
        (LyapunovConfig(max_growth=0.0), "max_growth"),
        (LyapunovConfig(propagator_tol=-1.0), "propagator_tol"),
    ],
    ids=["delta", "renorm_interval", "max_growth", "propagator_tol"],
)
def test_invalid_lyapunov_config_raises(cfg: LyapunovConfig, message: str) -> None:
    states, _, propagator = _driven_linear_system(0.9)
    with pytest.raises(ValueError, match=message):
        conditional_lyapunov(
            states, ctx=DiagnosticContext(propagator=propagator), cfg=cfg
        )

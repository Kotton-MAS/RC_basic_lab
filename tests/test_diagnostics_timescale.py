"""実効時定数の診断のテスト (受け入れ条件4 の解析解側)。

実 ESN でのリーク率と時定数の単調性 (2-B) は T3 で確認するが、そこで使う
物差しが正しいことはここで解析解に対して固定する。AR(1) は ESN が線形域で
とる形そのもの (``x[t+1] = phi x[t] + noise``) であり、自己相関は ``phi ** k``、
1/e 交差は ``-1 / log phi`` になる。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rc_basics_lab.diagnostics.base import DiagnosticContext
from rc_basics_lab.diagnostics.timescale import (
    TimescaleConfig,
    autocorrelation_time,
)
from rc_basics_lab.types import FloatArray

PHI_VALUES = (0.8, 0.9, 0.95)
"""検査する AR(1) の係数。

0.5 以下を使わないのは、1/e 交差が lag 1〜2 に来て**線形補間の誤差**
(指数曲線を弦で近似する誤差) が 6% に達し、診断の精度ではなく補間の粗さを
測ることになるため。実験 2-B が扱うリーク率の範囲 (時定数が数ステップ以上) は
この範囲に入る。
"""


def _ar1(
    phi: float, n_steps: int = 20000, n_units: int = 16, seed: int = 0
) -> FloatArray:
    """独立な AR(1) 過程を ``n_units`` 本並べた状態系列を作る。

    ユニットを複数本にするのは、ユニット平均 ACF の標本誤差を ``sqrt(n_units)``
    で落とすため (1本だと交差点の推定が 1% 程度ばらつく)。
    """
    rng = np.random.default_rng(seed)
    noise: FloatArray = rng.standard_normal((n_steps, n_units))
    states: FloatArray = np.empty((n_steps, n_units), dtype=np.float64)
    state: FloatArray = np.zeros(n_units)
    for index in range(n_steps):
        state = phi * state + noise[index]
        states[index] = state
    return states


def test_matches_analytic_timescale_for_ar1() -> None:
    """AR(1) の ``tau_1e`` が ``-1 / log phi`` と 2% 以内で一致し、phi に単調増加。

    単調性まで固定するのは、2-B の主張が「リーク率を下げると時定数が伸びる」
    という**順序**の話であり、絶対値がずれても順序が保たれれば図の結論は
    生き残る一方、順序が壊れたら結論そのものが壊れるため。
    """
    taus = []
    for phi in PHI_VALUES:
        result = autocorrelation_time(
            _ar1(phi), ctx=DiagnosticContext(washout=200), cfg=TimescaleConfig(100)
        )
        expected = -1.0 / math.log(phi)
        assert result.scalars["tau_1e"] == pytest.approx(expected, rel=0.02), (
            f"phi={phi}: tau_1e={result.scalars['tau_1e']}, 解析解={expected}"
        )
        taus.append(result.scalars["tau_1e"])
    assert taus == sorted(taus), f"phi を上げると時定数が伸びるはず: {taus}"


def test_acf_matches_analytic_shape_for_ar1() -> None:
    """AR(1) の自己相関そのものが ``phi ** k`` に一致する (曲線の形を固定)。"""
    phi = 0.9
    result = autocorrelation_time(_ar1(phi), cfg=TimescaleConfig(20))
    acf = result.arrays["acf"]
    expected: FloatArray = phi ** np.arange(21, dtype=np.float64)
    assert acf[0] == pytest.approx(1.0, rel=1e-12)
    assert acf == pytest.approx(expected, abs=0.01)


def test_integrated_time_matches_analytic_sum_for_ar1() -> None:
    """``tau_integrated`` が ``1 / (1 - phi)`` に一致する (初期正値列の和)。"""
    phi = 0.9
    result = autocorrelation_time(_ar1(phi), cfg=TimescaleConfig(100))
    assert result.scalars["tau_integrated"] == pytest.approx(
        1.0 / (1.0 - phi), rel=0.05
    )


def test_tau_is_nan_and_censored_at_max_lag_without_crossing() -> None:
    """1/e を切らない系列では ``tau_1e`` が nan、``tau_censored`` が ``max_lag``。

    交差が無いときに ``max_lag`` を返してしまうと「測れた時定数」と区別が
    つかなくなるため、両方を出して呼び出し側に選ばせる。
    """
    states = _ar1(0.999, n_steps=5000)
    result = autocorrelation_time(states, cfg=TimescaleConfig(20))
    assert math.isnan(result.scalars["tau_1e"])
    assert result.scalars["tau_censored"] == pytest.approx(20.0)
    assert result.scalars["max_lag"] == pytest.approx(20.0)


def test_max_lag_changes_the_acf_length() -> None:
    """``max_lag`` は ``acf`` の長さと ``tau_integrated`` に効く。"""
    states = _ar1(0.9, n_steps=5000)
    short = autocorrelation_time(states, cfg=TimescaleConfig(10))
    long = autocorrelation_time(states, cfg=TimescaleConfig(60))
    assert short.arrays["acf"].shape == (11,)
    assert long.arrays["acf"].shape == (61,)
    assert short.scalars["tau_integrated"] < long.scalars["tau_integrated"]
    # 1/e 交差は max_lag に依存しない (10 でも 60 でも同じ時定数)
    assert short.scalars["tau_1e"] == pytest.approx(long.scalars["tau_1e"], rel=1e-12)


def test_washout_is_applied() -> None:
    """washout 前の過渡を捨てて計算する。"""
    states = _ar1(0.9, n_steps=3000)
    states[:500] = 0.0  # 過渡を模した定数区間
    with_washout = autocorrelation_time(
        states, ctx=DiagnosticContext(washout=500), cfg=TimescaleConfig(30)
    )
    without = autocorrelation_time(states, cfg=TimescaleConfig(30))
    assert with_washout.params["washout"] == "500"
    assert with_washout.scalars["tau_1e"] != pytest.approx(
        without.scalars["tau_1e"], rel=1e-9
    )


def test_dead_units_are_excluded_from_the_mean() -> None:
    """分散 0 のユニットは平均から除き、使った本数を params に出す。"""
    states = _ar1(0.9, n_steps=3000, n_units=4)
    states[:, 0] = 1.0
    result = autocorrelation_time(states, cfg=TimescaleConfig(30))
    assert result.params["n_units_used"] == "3"
    assert result.scalars["tau_1e"] == pytest.approx(-1.0 / math.log(0.9), rel=0.05)


def test_all_constant_states_raise() -> None:
    with pytest.raises(ValueError, match="定数"):
        autocorrelation_time(np.ones((300, 4)), cfg=TimescaleConfig(30))


def test_series_shorter_than_max_lag_raises() -> None:
    """``max_lag + 2`` に満たない系列は ValueError (ラグが取れない)。"""
    states = _ar1(0.9, n_steps=300, n_units=2)
    with pytest.raises(ValueError, match="max_lag"):
        autocorrelation_time(
            states, ctx=DiagnosticContext(washout=250), cfg=TimescaleConfig(100)
        )


def test_invalid_max_lag_raises() -> None:
    with pytest.raises(ValueError, match="max_lag"):
        autocorrelation_time(_ar1(0.9, n_steps=300, n_units=2), cfg=TimescaleConfig(0))

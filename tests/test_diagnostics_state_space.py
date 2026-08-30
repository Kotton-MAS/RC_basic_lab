"""PCA 診断のテスト (受け入れ条件4 の土台)。"""

from __future__ import annotations

import numpy as np
import pytest

from rc_basics_lab.diagnostics.base import DiagnosticContext
from rc_basics_lab.diagnostics.state_space import (
    DORMANT_VARIANCE_RATIO,
    state_pca,
    unit_activity,
)
from rc_basics_lab.types import FloatArray


def _isotropic(n_steps: int = 5000, n_features: int = 10) -> FloatArray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n_steps, n_features))


def _low_rank(rank: int = 3, n_steps: int = 2000, n_features: int = 20) -> FloatArray:
    """厳密に ``rank`` 次元しか広がらず、各成分の分散が揃った合成データ。"""
    rng = np.random.default_rng(1)
    basis, _ = np.linalg.qr(rng.standard_normal((n_features, rank)))
    latent = rng.standard_normal((n_steps, rank))
    return latent @ basis.T


def test_participation_ratio_of_isotropic_gaussian() -> None:
    """等方ガウス (5000, 10) の participation_ratio は 10 に対し相対誤差 10% 以内。"""
    result = state_pca(_isotropic())
    assert result.scalars["participation_ratio"] == pytest.approx(10.0, rel=0.10)


def test_low_rank_input_has_smaller_effective_dimension() -> None:
    """rank 3 の合成データでは n_components_95 == 3。"""
    result = state_pca(_low_rank())
    assert result.scalars["n_components_95"] == pytest.approx(3.0)
    assert result.scalars["participation_ratio"] == pytest.approx(3.0, rel=0.15)


def test_low_rank_is_lower_dimensional_than_isotropic() -> None:
    """実効次元の比較が受け入れ条件4 の数値的な形になっている。"""
    low_rank = state_pca(_low_rank())
    isotropic = state_pca(_isotropic())
    assert low_rank.scalars["n_components_95"] < isotropic.scalars["n_components_95"]


def test_explained_variance_ratio_sums_to_one() -> None:
    result = state_pca(_isotropic())
    ratios = result.arrays["explained_variance_ratio"]
    assert float(np.sum(ratios)) == pytest.approx(1.0, rel=1e-12)
    assert result.arrays["cumulative_ratio"][-1] == pytest.approx(1.0, rel=1e-12)
    # 寄与率は降順
    assert np.all(np.diff(ratios) <= 1e-12)


def test_pc_scores_have_two_columns() -> None:
    states = _isotropic(n_steps=400)
    result = state_pca(states)
    assert result.arrays["pc_scores"].shape == (400, 2)


def test_washout_is_applied() -> None:
    states = _isotropic(n_steps=400)
    ctx = DiagnosticContext(washout=100)
    result = state_pca(states, ctx=ctx)
    assert result.scalars["n_samples"] == pytest.approx(300.0)
    assert result.arrays["pc_scores"].shape[0] == 300


def test_constant_states_raise() -> None:
    with pytest.raises(ValueError, match="定数"):
        state_pca(np.ones((100, 5)))


def test_scale_does_not_change_effective_dimension() -> None:
    """全体を定数倍しても実効次元は不変 (寄与率は比なので)。"""
    states = _low_rank()
    assert state_pca(states).scalars["participation_ratio"] == pytest.approx(
        state_pca(10.0 * states).scalars["participation_ratio"], rel=1e-10
    )


# --- unit_activity (T4) -------------------------------------------------------


def _states_with_dormant_units(
    n_dormant: int, n_active: int = 8, n_steps: int = 500
) -> FloatArray:
    """``n_dormant`` 本だけ分散が中央値の 1/10000 のユニットを混ぜた状態行列。"""
    rng = np.random.default_rng(0)
    active: FloatArray = rng.standard_normal((n_steps, n_active))
    dormant: FloatArray = 0.01 * rng.standard_normal((n_steps, n_dormant))
    return np.concatenate((active, dormant), axis=1)


def test_dormant_units_are_counted_by_the_variance_ratio() -> None:
    """分散が中央値の ``DORMANT_VARIANCE_RATIO`` 未満のユニットを数える。"""
    result = unit_activity(_states_with_dormant_units(n_dormant=3))
    assert result.scalars["n_units"] == pytest.approx(11.0)
    assert result.scalars["n_dormant"] == pytest.approx(3.0)
    assert result.scalars["dormant_fraction"] == pytest.approx(3.0 / 11.0)


def test_no_dormant_units_when_every_unit_moves_alike() -> None:
    """等方な状態では休眠ユニットは 0 本。"""
    rng = np.random.default_rng(1)
    result = unit_activity(rng.standard_normal((500, 12)))
    assert result.scalars["n_dormant"] == pytest.approx(0.0)
    assert result.scalars["dormant_fraction"] == pytest.approx(0.0)


def test_dormant_count_is_scale_invariant() -> None:
    """全体を定数倍しても休眠ユニット数は変わらない (**比**で判定しているため)。

    絶対値の閾値だと ``input_scale`` や ``leak_rate`` を変えるたびに意味が
    変わる。この不変性がその設計の実体である。
    """
    states = _states_with_dormant_units(n_dormant=2)
    assert unit_activity(states).scalars["n_dormant"] == pytest.approx(
        unit_activity(1000.0 * states).scalars["n_dormant"]
    )


def test_quantiles_are_monotone_and_bracket_the_median() -> None:
    """報告する分位点が単調で、中央値と整合する。"""
    result = unit_activity(_states_with_dormant_units(n_dormant=2))
    levels = (
        "variance_q05",
        "variance_q25",
        "variance_q50",
        "variance_q75",
        "variance_q95",
    )
    values = [result.scalars[name] for name in levels]
    assert values == sorted(values)
    assert result.scalars["variance_q50"] == pytest.approx(
        result.scalars["variance_median"]
    )
    assert result.scalars["variance_min"] <= values[0]
    assert values[-1] <= result.scalars["variance_max"]


def test_unit_variance_array_is_returned_per_column() -> None:
    """``unit_variance`` は列の順にユニットごとの分散を持つ。"""
    states = _states_with_dormant_units(n_dormant=3)
    variance = unit_activity(states).arrays["unit_variance"]
    assert variance.shape == (11,)
    np.testing.assert_allclose(variance, np.var(states, axis=0, ddof=1), rtol=1e-12)


def test_washout_is_applied_to_unit_activity() -> None:
    """``ctx.washout`` の行を落としてから分散を測る。"""
    states = _states_with_dormant_units(n_dormant=1, n_steps=400)
    full = unit_activity(states).arrays["unit_variance"]
    trimmed = unit_activity(states, ctx=DiagnosticContext(washout=100)).arrays[
        "unit_variance"
    ]
    np.testing.assert_allclose(
        trimmed, np.var(states[100:], axis=0, ddof=1), rtol=1e-12
    )
    assert not np.allclose(full, trimmed)


def test_constant_states_raise_for_unit_activity() -> None:
    """全ユニットが定数なら中央値が 0 になるので比を定義できない。"""
    with pytest.raises(ValueError, match="中央値"):
        unit_activity(np.ones((100, 5)))


def test_dormant_threshold_is_the_documented_ratio() -> None:
    """閾値がちょうど ``DORMANT_VARIANCE_RATIO`` であること (境界の実測)。

    分散が中央値のちょうど半分/2倍のユニットを置き、比の側だけが数えられる
    ことを見る。定数を変えたらここが落ちる。
    """
    rng = np.random.default_rng(2)
    base: FloatArray = rng.standard_normal((2000, 9))
    scaled = base.copy()
    # 分散比 = 係数^2。ちょうど閾値の半分と2倍に置く
    scaled[:, 0] *= (DORMANT_VARIANCE_RATIO * 0.5) ** 0.5
    scaled[:, 1] *= (DORMANT_VARIANCE_RATIO * 2.0) ** 0.5
    result = unit_activity(scaled)
    assert result.scalars["n_dormant"] == pytest.approx(1.0)

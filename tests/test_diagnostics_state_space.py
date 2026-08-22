"""PCA 診断のテスト (受け入れ条件4 の土台)。"""

from __future__ import annotations

import numpy as np
import pytest

from rc_basics_lab.diagnostics.base import DiagnosticContext
from rc_basics_lab.diagnostics.state_space import state_pca
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

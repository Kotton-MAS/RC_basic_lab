"""リッジ回帰と alpha 選択のテスト.

``test_bias_column_is_not_penalized`` は D-03 の guard_test。罰則行列を
``diag(1,1,...)`` に「簡略化」した瞬間に落ちる。
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from rc_basics_lab.config import DEFAULT_ALPHA_GRID, RidgeConfig
from rc_basics_lab.readout.design import PassthroughSpec, build_design_matrix
from rc_basics_lab.readout.ridge import (
    AlphaSelection,
    fit_ridge,
    fit_ridge_from_gram,
    penalty_matrix,
    predict,
    select_alpha,
)
from rc_basics_lab.types import FloatArray

CONSTANT_TARGET = 3.5


def _design(n_steps: int = 200, n_inputs: int = 3, seed: int = 0) -> FloatArray:
    """先頭がバイアス列の設計行列 (入力は非ゼロ平均にして交絡を作る)。"""
    rng = np.random.default_rng(seed)
    inputs = rng.standard_normal((n_steps, n_inputs)) + 2.0
    return build_design_matrix(PassthroughSpec(), inputs).phi


def _linear_target(phi: FloatArray, seed: int = 1) -> FloatArray:
    rng = np.random.default_rng(seed)
    true_weights = rng.standard_normal((phi.shape[1], 1))
    return phi @ true_weights + 0.1 * rng.standard_normal((phi.shape[0], 1))


def test_penalty_matrix_has_zero_for_bias() -> None:
    matrix = penalty_matrix(4, bias_column=0)
    assert np.array_equal(np.diag(matrix), np.array([0.0, 1.0, 1.0, 1.0]))
    assert np.array_equal(penalty_matrix(3, bias_column=None), np.eye(3))


def test_bias_column_is_not_penalized() -> None:
    """定数目標 y = c で、alpha を極端に大きくしてもバイアスは c のまま (D-03)。"""
    phi = _design()
    targets: FloatArray = np.full((phi.shape[0], 1), CONSTANT_TARGET)
    coefficients = fit_ridge(phi, targets, alpha=1e10, bias_column=0)
    assert coefficients[0, 0] == pytest.approx(CONSTANT_TARGET, rel=1e-8)
    assert float(np.max(np.abs(coefficients[1:]))) < 1e-8
    # 予測も定数目標を再現する (NRMSE=1 の基準線が壊れないことの実体)
    assert float(np.max(np.abs(predict(phi, coefficients) - targets))) < 1e-6
    # 対照: バイアスまで罰すると (bias_column=None) 目標の平均から外れる
    penalized_bias = fit_ridge(phi, targets, alpha=1e10, bias_column=None)
    assert abs(penalized_bias[0, 0]) < 0.1 * CONSTANT_TARGET


def test_bias_absorbs_target_mean_for_large_alpha() -> None:
    """一般の目標でも、alpha→大 で予測は目標の平均に収束する。"""
    phi = _design()
    targets = _linear_target(phi)
    coefficients = fit_ridge(phi, targets, alpha=1e12, bias_column=0)
    assert coefficients[0, 0] == pytest.approx(float(np.mean(targets)), rel=1e-6)


def test_alpha_changes_coefficient_norm() -> None:
    """alpha 単調増加に対し、罰せられる係数のノルムが単調減少する。"""
    phi = _design()
    targets = _linear_target(phi)
    norms = [
        float(np.linalg.norm(fit_ridge(phi, targets, alpha, bias_column=0)[1:]))
        for alpha in (1e-6, 1e-3, 1.0, 1e2, 1e4)
    ]
    assert all(later < earlier for earlier, later in pairwise(norms))
    assert norms[0] > 10.0 * norms[-1]


def test_closed_form_matches_naive_solution() -> None:
    """小さい系で lstsq ベースの素朴解と一致する。"""
    rng = np.random.default_rng(5)
    phi = _design(n_steps=30, n_inputs=4)
    targets: FloatArray = rng.standard_normal((30, 2))
    alpha = 0.37
    naive_matrix = np.vstack(
        [phi, np.sqrt(alpha) * penalty_matrix(phi.shape[1], bias_column=0)]
    )
    naive_targets = np.vstack([targets, np.zeros((phi.shape[1], 2))])
    naive, *_ = np.linalg.lstsq(naive_matrix, naive_targets, rcond=None)
    assert fit_ridge(phi, targets, alpha, bias_column=0) == pytest.approx(
        naive, rel=1e-8, abs=1e-10
    )


def test_alpha_zero_matches_ordinary_least_squares() -> None:
    phi = _design(n_steps=50, n_inputs=2)
    targets = _linear_target(phi, seed=3)
    ols, *_ = np.linalg.lstsq(phi, targets, rcond=None)
    assert fit_ridge(phi, targets, alpha=0.0, bias_column=0) == pytest.approx(
        ols, abs=1e-8
    )


def test_select_alpha_picks_validation_minimum() -> None:
    phi = _design(n_steps=300, seed=0)
    targets = _linear_target(phi)
    selection = select_alpha(
        phi[:200],
        targets[:200],
        phi[200:],
        targets[200:],
        DEFAULT_ALPHA_GRID,
        bias_column=0,
    )
    assert isinstance(selection, AlphaSelection)
    scores = [score for _, score in selection.curve]
    assert selection.val_nrmse == pytest.approx(min(scores))
    assert selection.alpha in {alpha for alpha, _ in selection.curve}


def test_select_alpha_reads_config_alpha_grid() -> None:
    """探索格子は関数の既定値ではなく ``config.ridge.alpha_grid`` から来る (D-04)。"""
    grid = RidgeConfig().alpha_grid
    phi = _design(n_steps=120)
    targets = _linear_target(phi)
    selection = select_alpha(
        phi[:80], targets[:80], phi[80:], targets[80:], grid, bias_column=0
    )
    assert tuple(alpha for alpha, _ in selection.curve) == tuple(sorted(grid))
    # 格子を変えれば探索範囲も変わる (格子が関数側に埋め込まれていない証拠)
    narrow = select_alpha(
        phi[:80], targets[:80], phi[80:], targets[80:], (1.0, 10.0), bias_column=0
    )
    assert tuple(alpha for alpha, _ in narrow.curve) == (1.0, 10.0)
    assert narrow.alpha in {1.0, 10.0}


def test_select_alpha_breaks_ties_toward_larger_alpha() -> None:
    """同点なら保守側 (大きい alpha) を選ぶ。"""
    # 罰せられる列が恒等的に 0 なら、その係数は alpha に依らず 0 になり、
    # 予測はバイアスだけで決まる → 全 alpha で検証 NRMSE が厳密に同点になる。
    rng = np.random.default_rng(8)
    phi: FloatArray = np.column_stack([np.ones(100), np.zeros(100), np.zeros(100)])
    targets: FloatArray = rng.standard_normal((100, 1))
    grid = (1e-3, 1.0, 1e3)
    selection = select_alpha(
        phi[:60], targets[:60], phi[60:], targets[60:], grid, bias_column=0
    )
    scores = [score for _, score in selection.curve]
    assert scores == [pytest.approx(scores[0], rel=1e-12)] * len(grid)
    assert selection.alpha == max(grid)


def test_select_alpha_rejects_empty_grid() -> None:
    phi = _design(n_steps=20)
    targets = _linear_target(phi)
    with pytest.raises(ValueError, match="alpha 格子"):
        select_alpha(phi[:10], targets[:10], phi[10:], targets[10:], (), bias_column=0)


def test_fit_ridge_rejects_negative_alpha() -> None:
    phi = _design(n_steps=20)
    with pytest.raises(ValueError, match="alpha"):
        fit_ridge(phi, _linear_target(phi), alpha=-1.0, bias_column=0)


def test_fit_ridge_rejects_non_finite_rows() -> None:
    """first_valid より手前を切り忘れた設計行列は静かに通さない。"""
    phi = _design(n_steps=20)
    targets = _linear_target(phi)
    poisoned = phi.copy()
    poisoned[0, 1] = np.nan
    with pytest.raises(ValueError, match="有限でない"):
        fit_ridge(poisoned, targets, alpha=1.0, bias_column=0)


def test_fit_ridge_rejects_one_dimensional_target() -> None:
    phi = _design(n_steps=20)
    with pytest.raises(ValueError, match="2次元"):
        fit_ridge(phi, np.zeros(20), alpha=1.0, bias_column=0)


def test_coefficient_shape_supports_multiple_outputs() -> None:
    rng = np.random.default_rng(2)
    phi = _design(n_steps=60, n_inputs=3)
    targets: FloatArray = rng.standard_normal((60, 2))
    assert fit_ridge(phi, targets, alpha=1.0, bias_column=0).shape == (4, 2)


def test_bias_column_is_keyword_required() -> None:
    """``bias_column`` を渡し忘れると TypeError で落ちる (F-1-002)。

    既定値 0 (= 先頭列は必ずバイアス) を持たせないことで、``bias=False`` の
    設計行列を渡し忘れたときに「静かに少し違う係数」ではなく型エラーになる。
    呼び出しは静的にも mypy が拾う欠落を意図的に再現しているため
    ``# type: ignore[call-arg]`` を付けている (mypy を黙らせて実行時の
    TypeError を実測するのがこのテストの目的そのもの)。
    """
    phi = _design(n_steps=20)
    targets = _linear_target(phi)
    with pytest.raises(TypeError, match="bias_column"):
        fit_ridge(phi, targets, alpha=1.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="bias_column"):
        select_alpha(  # type: ignore[call-arg]
            phi[:10], targets[:10], phi[10:], targets[10:], (1.0,)
        )
    with pytest.raises(TypeError, match="bias_column"):
        penalty_matrix(phi.shape[1])  # type: ignore[call-arg]


def test_fit_ridge_from_gram_matches_fit_ridge() -> None:
    """Gram 行列を先に計算する経路 (F-1-010) は ``fit_ridge`` と数学的に同一。"""
    phi = _design(n_steps=90, n_inputs=3)
    targets = _linear_target(phi)
    gram: FloatArray = phi.T @ phi
    rhs: FloatArray = phi.T @ targets
    for alpha in (0.0, 1e-3, 1.0, 1e3):
        direct = fit_ridge(phi, targets, alpha, bias_column=0)
        from_gram = fit_ridge_from_gram(gram, rhs, alpha, bias_column=0)
        assert from_gram == pytest.approx(direct, rel=1e-12, abs=1e-12)

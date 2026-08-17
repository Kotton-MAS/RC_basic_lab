"""誤差指標のテスト (D-02)。"""

from __future__ import annotations

import numpy as np
import pytest

from rc_basics_lab.metrics import nmse, nrmse, rmse, sign_accuracy
from rc_basics_lab.types import FloatArray


def _target() -> FloatArray:
    rng = np.random.default_rng(0)
    return rng.standard_normal(500)


def test_nrmse_of_mean_predictor_is_one() -> None:
    """目標の平均を返す予測子の NRMSE は厳密に 1.0 (D-02 の guard test)。"""
    y_true = _target()
    y_pred = np.full_like(y_true, float(np.mean(y_true)))
    assert nrmse(y_true, y_pred) == pytest.approx(1.0, rel=1e-12)


def test_nrmse_of_perfect_predictor_is_zero() -> None:
    y_true = _target()
    assert nrmse(y_true, y_true) == pytest.approx(0.0, abs=1e-15)


def test_nmse_is_nrmse_squared() -> None:
    y_true = _target()
    y_pred = y_true + 0.3 * np.roll(y_true, 1)
    assert nmse(y_true, y_pred) == pytest.approx(nrmse(y_true, y_pred) ** 2, rel=1e-12)


def test_rmse_matches_manual_computation() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(4.0 / 3.0), rel=1e-12)


def test_nrmse_is_scale_invariant() -> None:
    """目標を定数倍しても NRMSE は変わらない (正規化の定義確認)。"""
    y_true = _target()
    y_pred = 0.5 * y_true
    assert nrmse(10.0 * y_true, 10.0 * y_pred) == pytest.approx(
        nrmse(y_true, y_pred), rel=1e-12
    )


def test_sign_accuracy_counts_matching_signs() -> None:
    y_true = np.array([1.0, -1.0, 1.0, -1.0])
    y_pred = np.array([2.0, -0.5, -3.0, -0.1])
    assert sign_accuracy(y_true, y_pred) == pytest.approx(0.75)


def test_sign_of_zero_is_positive() -> None:
    """sign(0) は +1 に固定する (仕様 §3)。"""
    y_true = np.array([1.0, -1.0])
    y_pred = np.zeros(2)
    assert sign_accuracy(y_true, y_pred) == pytest.approx(0.5)


def test_metrics_support_multi_output() -> None:
    rng = np.random.default_rng(1)
    y_true = rng.standard_normal((200, 3))
    y_pred = np.full_like(y_true, float(np.mean(y_true)))
    assert nrmse(y_true, y_pred) == pytest.approx(1.0, rel=1e-12)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="形状"):
        rmse(np.zeros(3), np.zeros(4))


def test_empty_series_raises() -> None:
    with pytest.raises(ValueError, match="空"):
        rmse(np.zeros(0), np.zeros(0))


def test_constant_target_raises() -> None:
    """定数目標では NRMSE が定義できないため即座に失敗する。"""
    with pytest.raises(ValueError, match="std"):
        nrmse(np.ones(10), np.zeros(10))

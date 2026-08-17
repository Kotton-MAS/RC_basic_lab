"""リッジ回帰 (閉形式) と検証分割による alpha 選択.

**バイアス列は正則化しない** (D-03)。罰則行列は ``D = diag(0, 1, 1, ...)``。
バイアスを縮めると予測の平均が目標の平均からずれ、「NRMSE = 1 が平均予測と同等」
という基準線 (D-02) の意味が壊れる。

alpha の探索格子は関数側に既定値を持たせず、呼び出し側が
``config.ridge.alpha_grid`` を渡す。格子の既定値が2箇所に存在すると
「全手法が同一格子を読む」(D-04) が静かに破れるため。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve

from rc_basics_lab.metrics import nrmse
from rc_basics_lab.types import FloatArray


@dataclass(frozen=True, slots=True)
class AlphaSelection:
    """検証分割による alpha 選択の結果。

    Attributes:
        alpha: 選ばれた alpha。
        val_nrmse: その alpha での検証 NRMSE。
        curve: ``(alpha, val_nrmse)`` の列 (alpha 昇順)。図と CSV の材料。
    """

    alpha: float
    val_nrmse: float
    curve: tuple[tuple[float, float], ...]


def _check_pair(phi: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    features = np.asarray(phi, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"phi は (T, F) の2次元配列が必要です: {features.shape}")
    if targets.ndim != 2:
        raise ValueError(f"y は (T, D_out) の2次元配列が必要です: {targets.shape}")
    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            f"phi と y の行数が一致しません: {features.shape[0]} != {targets.shape[0]}"
        )
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"phi が空です: {features.shape}")
    # 設計行列の first_valid より手前は NaN。t0 の取り違えをここで大きな音で落とす。
    if not np.all(np.isfinite(features)):
        raise ValueError(
            "phi に有限でない値があります "
            "(first_valid より手前の行を切り落としましたか)"
        )
    if not np.all(np.isfinite(targets)):
        raise ValueError("y に有限でない値があります")
    return features, targets


def penalty_matrix(n_features: int, *, bias_column: int | None) -> FloatArray:
    """罰則行列 ``D = diag(0, 1, 1, ...)`` (D-03)。

    Args:
        n_features: 特徴数 F。
        bias_column: 正則化しない列の index。``None`` なら全列を正則化する
            (``bias=False`` の設計行列用)。既定値を持たない (キーワード必須)。
            渡し忘れると「先頭列が黙ってバイアス扱い」という誤りが型で
            落ちる (F-1-002)。``DesignMatrix.bias_column`` を渡すこと。
    """
    diagonal: FloatArray = np.ones(n_features, dtype=np.float64)
    if bias_column is not None:
        if not 0 <= bias_column < n_features:
            raise ValueError(
                f"bias_column が範囲外です: {bias_column} (F={n_features})"
            )
        diagonal[bias_column] = 0.0
    return np.diag(diagonal)


def fit_ridge_from_gram(
    gram: FloatArray,
    rhs: FloatArray,
    alpha: float,
    *,
    bias_column: int | None,
) -> FloatArray:
    """Gram 行列 ``Phi.T @ Phi`` と ``Phi.T @ y`` から閉形式解を返す。

    ``gram`` / ``rhs`` は alpha に依存しないため、alpha 格子を走査する
    呼び出し側 (``select_alpha``) はこれを1回だけ計算して使い回せる
    (F-1-010)。``fit_ridge`` はこの関数に委譲しており、数学的に同一の
    経路 (``solve(gram + alpha * D, rhs, assume_a="pos")``) を通る。

    Args:
        gram: ``Phi.T @ Phi`` ``(F, F)``。
        rhs: ``Phi.T @ y`` ``(F, D_out)``。
        alpha: 正則化係数 (0 以上)。
        bias_column: 正則化しない列の index。``DesignMatrix.bias_column`` を渡す。

    Returns:
        係数 ``(F, D_out)``。

    Raises:
        ValueError: 形状不整合、``alpha < 0``。
    """
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"gram は正方行列が必要です: {gram.shape}")
    if rhs.ndim != 2 or rhs.shape[0] != gram.shape[0]:
        raise ValueError(
            f"rhs の行数が gram と一致しません: {rhs.shape} vs {gram.shape}"
        )
    if alpha < 0.0:
        raise ValueError(f"alpha は 0 以上である必要があります: {alpha}")
    n_features = gram.shape[0]
    penalized: FloatArray = gram + alpha * penalty_matrix(
        n_features, bias_column=bias_column
    )
    coefficients: FloatArray = np.asarray(
        solve(penalized, rhs, assume_a="pos"), dtype=np.float64
    )
    return coefficients


def fit_ridge(
    phi: FloatArray,
    y: FloatArray,
    alpha: float,
    *,
    bias_column: int | None,
) -> FloatArray:
    """閉形式のリッジ解 ``inv(Phi.T @ Phi + alpha * D) @ Phi.T @ Y`` を返す。

    Args:
        phi: 設計行列 ``(T, F)``。
        y: 目標 ``(T, D_out)``。
        alpha: 正則化係数 (0 以上)。
        bias_column: 正則化しない列の index。既定値を持たない (キーワード必須)。
            渡し忘れると「先頭列が黙ってバイアス扱い」という誤りが型で
            落ちる (F-1-002)。``DesignMatrix.bias_column`` を渡すこと。

    Returns:
        係数 ``(F, D_out)``。

    Raises:
        ValueError: 形状不整合、非有限値、``alpha < 0``。
    """
    features, targets = _check_pair(phi, y)
    gram: FloatArray = features.T @ features
    rhs: FloatArray = features.T @ targets
    return fit_ridge_from_gram(gram, rhs, alpha, bias_column=bias_column)


def predict(phi: FloatArray, coefficients: FloatArray) -> FloatArray:
    """線形読み出しの予測 ``Φ W``。"""
    prediction: FloatArray = np.asarray(phi, dtype=np.float64) @ np.asarray(
        coefficients, dtype=np.float64
    )
    return prediction


def select_alpha(
    phi_tr: FloatArray,
    y_tr: FloatArray,
    phi_val: FloatArray,
    y_val: FloatArray,
    alphas: Sequence[float],
    *,
    bias_column: int | None,
) -> AlphaSelection:
    """検証 NRMSE が最小の alpha を選ぶ。同点なら**大きい** alpha (保守側)。

    Gram 行列 ``Phi_tr.T @ Phi_tr`` と ``Phi_tr.T @ y_tr`` は alpha に依存しないため、
    alpha 格子1本につき1回だけ計算し、格子の走査では ``fit_ridge_from_gram`` で
    solve のみを繰り返す (F-1-010)。``fit_ridge`` を alpha ごとに呼ぶのと
    数学的に同一の経路で、結果は変わらない。

    Args:
        phi_tr: 学習設計行列 ``(T_tr, F)``。
        y_tr: 学習目標 ``(T_tr, D_out)``。
        phi_val: 検証設計行列 ``(T_val, F)``。
        y_val: 検証目標 ``(T_val, D_out)``。
        alphas: 探索格子。``config.ridge.alpha_grid`` をそのまま渡す (D-04)。
        bias_column: ``fit_ridge`` に渡す無罰則列。既定値を持たない
            (キーワード必須)。``DesignMatrix.bias_column`` を渡すこと (F-1-002)。

    Raises:
        ValueError: ``alphas`` が空の場合。
    """
    if len(alphas) == 0:
        raise ValueError("alpha 格子が空です")
    features_tr, targets_tr = _check_pair(phi_tr, y_tr)
    gram: FloatArray = features_tr.T @ features_tr
    rhs: FloatArray = features_tr.T @ targets_tr
    curve: list[tuple[float, float]] = []
    best_alpha = float("nan")
    best_score = float("inf")
    for alpha in sorted(float(value) for value in alphas):
        coefficients = fit_ridge_from_gram(gram, rhs, alpha, bias_column=bias_column)
        score = nrmse(y_val, predict(phi_val, coefficients))
        curve.append((alpha, score))
        # 昇順に走査し「以下」で更新するため、同点では大きい alpha が残る。
        if score <= best_score:
            best_alpha = alpha
            best_score = score
    return AlphaSelection(alpha=best_alpha, val_nrmse=best_score, curve=tuple(curve))


__all__ = [
    "AlphaSelection",
    "fit_ridge",
    "fit_ridge_from_gram",
    "penalty_matrix",
    "predict",
    "select_alpha",
]

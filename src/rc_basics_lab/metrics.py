"""誤差指標 (D-02).

NRMSE = RMSE / std(y_true) (ddof=0) に統一する。NRMSE = 1 が「目標の平均を
返す予測子と同等」を意味するため、遅延パリティで線形手法が失敗したことを
解釈可能な形で示せる。
"""

from __future__ import annotations

import numpy as np

from rc_basics_lab.types import FloatArray


def _as_pair(y_true: FloatArray, y_pred: FloatArray) -> tuple[FloatArray, FloatArray]:
    """形状を検証し、float64 の同形状ペアに揃える。"""
    true_array = np.asarray(y_true, dtype=np.float64)
    pred_array = np.asarray(y_pred, dtype=np.float64)
    if true_array.shape != pred_array.shape:
        raise ValueError(
            "y_true と y_pred の形状が一致しません: "
            f"{true_array.shape} != {pred_array.shape}"
        )
    if true_array.size == 0:
        raise ValueError("空の系列に対して誤差指標は定義されません")
    return true_array, pred_array


def rmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    """二乗平均平方根誤差。"""
    true_array, pred_array = _as_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((true_array - pred_array) ** 2)))


def nrmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    """正規化 RMSE。``RMSE / std(y_true)`` (ddof=0)。

    Raises:
        ValueError: ``std(y_true) == 0`` のとき (定数目標では NRMSE が定義できず、
            0 除算で inf/nan を返すと下流の集計が静かに壊れるため即座に失敗させる)。
    """
    true_array, pred_array = _as_pair(y_true, y_pred)
    scale = float(np.std(true_array))
    if scale == 0.0:
        raise ValueError(
            "std(y_true) が 0 のため NRMSE を定義できません (目標が定数です)"
        )
    return rmse(true_array, pred_array) / scale


def nmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    """正規化平均二乗誤差。定義上 ``NRMSE ** 2`` と一致する。"""
    return nrmse(y_true, y_pred) ** 2


def _sign(values: FloatArray) -> FloatArray:
    """符号。``sign(0)`` は +1 に固定する (仕様 §3)。"""
    return np.where(values >= 0.0, 1.0, -1.0)


def sign_accuracy(y_true: FloatArray, y_pred: FloatArray) -> float:
    """符号一致率。``sign(0) = +1`` として計算する。"""
    true_array, pred_array = _as_pair(y_true, y_pred)
    return float(np.mean(_sign(true_array) == _sign(pred_array)))


__all__ = ["nmse", "nrmse", "rmse", "sign_accuracy"]

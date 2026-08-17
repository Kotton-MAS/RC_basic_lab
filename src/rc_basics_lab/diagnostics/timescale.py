"""状態の実効時定数を自己相関から測る診断 (実験 2-B).

リーク率 ``a`` の ESN は、線形域では ``x[t+1] = (1-a) x[t] + a f(...)`` という
1次のローパスなので、自己相関は ``(1-a)^k`` で減衰し時定数は
``-1 / log(1-a)`` になる。この診断はその時定数を実測して返す。理論線との
比較 (図) と、AR(1) 解析解との一致 (テスト) の両方がここを土台にする。

設定値は ``DiagnosticContext`` ではなく既定値つきキーワード引数 ``cfg`` で
渡す (D-15)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rc_basics_lab.diagnostics.base import (
    DiagnosticContext,
    DiagnosticResult,
    resolve_context,
    validate_diagnostic_input,
)
from rc_basics_lab.types import FloatArray

NAME = "autocorrelation_time"

_ONE_OVER_E = 1.0 / math.e
"""時定数の定義に使う交差水準。指数減衰なら交差点がそのまま時定数になる。"""


@dataclass(frozen=True, slots=True)
class TimescaleConfig:
    """自己相関時間の測定条件。純データ。値域検証は診断関数側で行う。

    Attributes:
        max_lag: 自己相関を計算する最大ラグ [ステップ]。
    """

    max_lag: int = 200


DEFAULT_TIMESCALE = TimescaleConfig()


def _mean_autocorrelation(centered: FloatArray, max_lag: int) -> tuple[FloatArray, int]:
    """ユニット平均の自己相関関数 ``(max_lag + 1,)`` と使用ユニット数を返す。

    分散 0 のユニット (飽和して定数になったユニットなど) は自己相関が定義
    できないため平均から除く。除いた事実は ``params`` の ``n_units_used`` に
    出す (黙って落とすと「全ユニットが死んでいるのに ACF が返る」が通る)。
    """
    variance: FloatArray = np.sum(centered**2, axis=0)
    alive = variance > 0.0
    n_used = int(np.count_nonzero(alive))
    if n_used == 0:
        raise ValueError("全ユニットが定数のため自己相関を定義できません")
    live: FloatArray = centered[:, alive]
    denominator: FloatArray = variance[alive]
    n_samples = live.shape[0]
    acf: FloatArray = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        products: FloatArray = live[: n_samples - lag] * live[lag:]
        acf[lag] = float(np.mean(np.sum(products, axis=0) / denominator))
    return acf, n_used


def _crossing_lag(acf: FloatArray, level: float) -> float:
    """``acf`` が初めて ``level`` を下回るラグを線形補間で返す。無ければ ``nan``。"""
    below = np.nonzero(acf <= level)[0]
    if below.size == 0:
        return float("nan")
    index = int(below[0])
    if index == 0:
        return 0.0
    upper = float(acf[index - 1])
    lower = float(acf[index])
    return (index - 1) + (upper - level) / (upper - lower)


def _integrated_time(acf: FloatArray) -> float:
    """初期正値列の和 (``acf`` が最初に非正になる直前までの和)。

    無限和を素直に打ち切ると、ラグが大きい領域の推定誤差 (期待値 0・分散有限)
    がそのまま積み上がって発散するため、標準的な initial positive sequence
    で切る。
    """
    non_positive = np.nonzero(acf <= 0.0)[0]
    end = int(non_positive[0]) if non_positive.size else acf.shape[0]
    return float(np.sum(acf[:end]))


def autocorrelation_time(
    X: FloatArray,
    u: FloatArray | None = None,
    y: FloatArray | None = None,
    *,
    ctx: DiagnosticContext | None = None,
    cfg: TimescaleConfig = DEFAULT_TIMESCALE,
) -> DiagnosticResult:
    """状態の自己相関が ``1/e`` を切るラグ (実効時定数) を返す。

    Args:
        X: 状態系列 ``(T, N)``。
        u: 未使用 (プロトコル適合のために受け取る)。
        y: 未使用 (同上)。
        ctx: ``washout`` のみ参照する。
        cfg: 測定条件 (D-15)。

    Returns:
        ``scalars``: ``tau_1e`` (交差が無ければ ``nan``) / ``tau_censored``
        (交差が無ければ ``max_lag``。単調性の比較に使う右打ち切り版) /
        ``tau_integrated`` / ``max_lag``。``arrays``: ``acf``。

    Raises:
        ValueError: ``max_lag`` が 1 未満、washout 後の長さが ``max_lag + 2``
            未満、または全ユニットが定数。

    Note:
        ラグの単位はステップであり ``ctx.dt`` では割らない。リーク率との比較で
        使う理論線 ``-1/log(1-a)`` がステップ単位の量であるため。
    """
    validate_diagnostic_input(X, u, y, ctx)
    if cfg.max_lag < 1:
        raise ValueError(f"max_lag は 1 以上である必要があります: {cfg.max_lag!r}")
    context = resolve_context(ctx)
    states = np.asarray(X, dtype=np.float64)[context.washout :]
    n_samples = states.shape[0]
    if n_samples < cfg.max_lag + 2:
        raise ValueError(
            "washout 後のサンプル数が max_lag + 2 に満たないため自己相関を"
            f"測れません: n_samples={n_samples}, max_lag={cfg.max_lag}"
        )

    centered: FloatArray = states - states.mean(axis=0, keepdims=True)
    acf, n_units_used = _mean_autocorrelation(centered, cfg.max_lag)
    tau_1e = _crossing_lag(acf, _ONE_OVER_E)
    tau_censored = float(cfg.max_lag) if math.isnan(tau_1e) else tau_1e

    return DiagnosticResult(
        name=NAME,
        scalars={
            "tau_1e": tau_1e,
            "tau_censored": tau_censored,
            "tau_integrated": _integrated_time(acf),
            "max_lag": float(cfg.max_lag),
        },
        arrays={"acf": acf},
        params={
            # max_lag は scalars 側に出す (to_row でキーが衝突するため
            # params には入れない)。
            "washout": str(context.washout),
            "n_units_used": str(n_units_used),
        },
    )


__all__ = [
    "DEFAULT_TIMESCALE",
    "NAME",
    "TimescaleConfig",
    "autocorrelation_time",
]

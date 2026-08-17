"""Mackey-Glass 系列の生成 (仕様 §3 未確定1 の決定値).

遅延微分方程式::

    dx/dt = beta * x(t - tau) / (1 + x(t - tau)^n) - gamma * x(t)

を RK4 (刻み ``rk4_step``) で積分し、``sample_interval`` ステップごとに
サブサンプルする。既定は tau=17, beta=0.2, gamma=0.1, n=10, h=0.1,
サンプリング間隔 10 (= Delta t 1.0)。tau=17 は Farmer (1982) 以来のカオス域の
標準値で、RC ベンチマークの事実上の既定である。

RK4 の中間段が必要とする ``x(t - tau + h/2)`` は履歴の線形補間で与える。
``tau / rk4_step`` は整数である必要があり、そうでなければ ``ValueError``
(黙って近い格子点に丸めると、設定した tau と実際に積分した tau がずれる)。
"""

from __future__ import annotations

import numpy as np

from rc_basics_lab.config import MackeyGlassConfig
from rc_basics_lab.tasks.base import TaskData
from rc_basics_lab.types import FloatArray

TASK_NAME = "mackey_glass"

INITIAL_VALUE = 1.2
"""初期履歴の中心値 (文献の慣例値)。"""

INITIAL_JITTER = 0.1
"""初期履歴に task ストリームで与える一様揺らぎの幅。

レプリケート間で軌道を変えるためのもの。カオス系なので、この幅の違いは
バーンイン後には初期値の記憶を残さない。
"""

_RATIO_TOLERANCE = 1e-9


def _validate(cfg: MackeyGlassConfig) -> None:
    if cfg.rk4_step <= 0.0:
        raise ValueError(f"rk4_step は正である必要があります: {cfg.rk4_step}")
    if cfg.tau <= 0.0:
        raise ValueError(f"tau は正である必要があります: {cfg.tau}")
    if cfg.sample_interval < 1:
        raise ValueError(
            f"sample_interval は 1 以上である必要があります: {cfg.sample_interval}"
        )
    if cfg.integration_burn_in < 0:
        raise ValueError(
            "integration_burn_in は 0 以上である必要があります: "
            f"{cfg.integration_burn_in}"
        )
    if cfg.length < 1:
        raise ValueError(f"length は 1 以上である必要があります: {cfg.length}")
    if cfg.horizon < 1:
        raise ValueError(f"horizon は 1 以上である必要があります: {cfg.horizon}")
    if cfg.exponent < 1:
        raise ValueError(f"exponent は 1 以上である必要があります: {cfg.exponent}")


def delay_steps(cfg: MackeyGlassConfig) -> int:
    """``tau / rk4_step``。整数にならない設定は ``ValueError``。"""
    ratio = cfg.tau / cfg.rk4_step
    steps = round(ratio)
    if steps < 1 or abs(ratio - steps) > _RATIO_TOLERANCE * max(1.0, ratio):
        raise ValueError(
            "tau / rk4_step が整数になりません "
            f"(tau={cfg.tau}, rk4_step={cfg.rk4_step}, 比={ratio})"
        )
    return steps


def _derivative(x: float, x_tau: float, cfg: MackeyGlassConfig) -> float:
    return cfg.beta * x_tau / (1.0 + x_tau**cfg.exponent) - cfg.gamma * x


def integrate(
    cfg: MackeyGlassConfig, rng: np.random.Generator, n_samples: int
) -> FloatArray:
    """RK4 で積分し、バーンイン後の ``n_samples`` 個のサブサンプルを返す。

    Args:
        cfg: 生成パラメータ。
        rng: task ストリームの Generator (初期履歴の揺らぎに使う)。
        n_samples: 返すサンプル数 (バーンインぶんは含まない)。

    Returns:
        ``(n_samples,)`` の系列。
    """
    _validate(cfg)
    if n_samples < 1:
        raise ValueError(f"n_samples は 1 以上である必要があります: {n_samples}")
    lag = delay_steps(cfg)
    total_samples = cfg.integration_burn_in + n_samples
    n_steps = total_samples * cfg.sample_interval
    history: FloatArray = np.empty(lag + 1 + n_steps, dtype=np.float64)
    history[: lag + 1] = INITIAL_VALUE + rng.uniform(
        -INITIAL_JITTER, INITIAL_JITTER, lag + 1
    )

    step = cfg.rk4_step
    for index in range(n_steps):
        here = lag + index
        x = float(history[here])
        # 遅延項: tau / h が整数なので x(t-tau) は履歴の格子点そのもの。
        # 中間段の x(t-tau+h/2) は隣接格子点の線形補間で与える。
        x_tau_0 = float(history[index])
        x_tau_1 = float(history[index + 1])
        x_tau_half = 0.5 * (x_tau_0 + x_tau_1)
        k1 = _derivative(x, x_tau_0, cfg)
        k2 = _derivative(x + 0.5 * step * k1, x_tau_half, cfg)
        k3 = _derivative(x + 0.5 * step * k2, x_tau_half, cfg)
        k4 = _derivative(x + step * k3, x_tau_1, cfg)
        history[here + 1] = x + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    sampled: FloatArray = history[lag + cfg.sample_interval :: cfg.sample_interval]
    return np.array(sampled[-n_samples:], dtype=np.float64)


def generate_mackey_glass(cfg: MackeyGlassConfig, rng: np.random.Generator) -> TaskData:
    """``horizon`` ステップ先予測の課題を作る。

    ``u[t] = x[t]``, ``y[t] = x[t + horizon]``。返す行数は ``cfg.length``。
    """
    _validate(cfg)
    series = integrate(cfg, rng, cfg.length + cfg.horizon)
    u: FloatArray = series[: cfg.length].reshape(-1, 1)
    y: FloatArray = series[cfg.horizon : cfg.horizon + cfg.length].reshape(-1, 1)
    params = {
        "tau": str(cfg.tau),
        "beta": str(cfg.beta),
        "gamma": str(cfg.gamma),
        "exponent": str(cfg.exponent),
        "rk4_step": str(cfg.rk4_step),
        "sample_interval": str(cfg.sample_interval),
        "horizon": str(cfg.horizon),
    }
    return TaskData(u=u, y=y, name=TASK_NAME, params=params)


__all__ = [
    "INITIAL_JITTER",
    "INITIAL_VALUE",
    "TASK_NAME",
    "delay_steps",
    "generate_mackey_glass",
    "integrate",
]

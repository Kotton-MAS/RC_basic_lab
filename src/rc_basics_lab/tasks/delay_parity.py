"""遅延パリティ課題 (D-07).

``u[t]`` は ±1 の i.i.d.、目標は該当ラグの積::

    y[t] = prod_{i=delay}^{delay + n_bits - 1} u[t - i]

既定 (n_bits=2, delay=1) では ``y[t] = u[t-1] * u[t-2]``。この符号化では、
任意の k について ``E[y[t] * u[t-k]] = 0`` かつ ``E[y[t]] = 0`` になる
(k が {1,2} の外なら独立3変数の積、k が {1,2} なら ``u^2 = 1`` により残る1変数の
期待値が 0)。つまり目標は ``{1, u[t], ..., u[t-k]}`` の張る線形空間に**厳密に
直交する**ため、線形回帰も遅延線も母集団最適解が恒等的に 0 になり、
NRMSE → 1.0 / 符号正解率 → 0.5 に落ちる。失敗は経験則ではなく解析的な帰結である
(``tests/test_tasks_parity.py::test_target_is_orthogonal_to_lagged_inputs``)。

先頭 ``delay + n_bits - 1`` 行ぶんの入力は**先読み分として内部で余分に生成**し、
返す前に切り落とす。目標に NaN を混ぜないためで、返る ``u`` / ``y`` は全行が有効。
"""

from __future__ import annotations

import numpy as np

from rc_basics_lab.config import DelayParityConfig
from rc_basics_lab.tasks.base import TaskData
from rc_basics_lab.types import FloatArray

TASK_NAME = "delay_parity"


def _validate(cfg: DelayParityConfig) -> None:
    if cfg.n_bits < 1:
        raise ValueError(f"n_bits は 1 以上である必要があります: {cfg.n_bits}")
    if cfg.delay < 0:
        raise ValueError(f"delay は 0 以上である必要があります: {cfg.delay}")
    if cfg.length < 1:
        raise ValueError(f"length は 1 以上である必要があります: {cfg.length}")


def lead_in(cfg: DelayParityConfig) -> int:
    """目標が定義されるまでに必要な先読みステップ数 ``delay + n_bits - 1``。"""
    return cfg.delay + cfg.n_bits - 1


def generate_delay_parity(cfg: DelayParityConfig, rng: np.random.Generator) -> TaskData:
    """遅延パリティの入出力系列を作る。返す行数は ``cfg.length``。"""
    _validate(cfg)
    lead = lead_in(cfg)
    total = cfg.length + lead
    bits: FloatArray = np.where(rng.integers(0, 2, total) == 1, 1.0, -1.0)
    target: FloatArray = np.ones(cfg.length, dtype=np.float64)
    for shift in range(cfg.delay, cfg.delay + cfg.n_bits):
        start = lead - shift
        target = target * bits[start : start + cfg.length]
    u: FloatArray = bits[lead:].reshape(-1, 1)
    y: FloatArray = target.reshape(-1, 1)
    params = {
        "n_bits": str(cfg.n_bits),
        "delay": str(cfg.delay),
    }
    return TaskData(u=u, y=y, name=TASK_NAME, params=params)


__all__ = ["TASK_NAME", "generate_delay_parity", "lead_in"]

"""容量の測定器を**答えが分かっている状態行列**で検証する (D-137).

## なぜ要るのか

既存の IPC / MC のテストは機構 (保存則・打ち切り・しきい値・分割) を見ているが、
**測った値が正しいか**は見ていない。「自前実装で測った」がこの装置の最大の弱点
なので、そこを塞ぐ。

## なぜ外部実装との突き合わせではないのか

方針書 (研究観点 §6-1) は ``kubota0130/ipc`` との突き合わせを挙げている。
実装が合うことは価値があるが、**両方が同じように間違っている可能性は消せない**。
解析解のある状態行列なら「合っているか」を直接測れるうえ、GitHub への依存
(CI がネットワークとリポジトリの生死に縛られる) も増えない。

外部実装との突き合わせは、ここが緑になったうえで**手で1回**やる価値がある
(手順は docs/guide/診断を足す.md)。

## 使う解析解

===============================  ==========================================
状態行列                         IPC の厳密解
===============================  ==========================================
``[u[t-1], ..., u[t-K]]``        次数1に K。非線形は 0
``[P2(u[t-1])]``                 次数2に 1。線形は 0
入力と独立な系列                 しきい値後に 0
===============================  ==========================================

1本目は「完全な遅延線」で、状態が目標そのものなので回帰は厳密に解ける。
2本目は Legendre の2次だけを持つ状態で、**次数の分解が効いているか**を見る
(総量だけ合っていても分解が壊れていることはある)。
"""

from __future__ import annotations

import numpy as np
import pytest

from rc_basics_lab.diagnostics._capacity import UNIFORM_LEGENDRE, orthonormal_basis
from rc_basics_lab.diagnostics.base import DiagnosticContext
from rc_basics_lab.diagnostics.ipc import IpcConfig, ipc
from rc_basics_lab.diagnostics.memory_capacity import (
    MemoryCapacityConfig,
    memory_capacity,
)
from rc_basics_lab.types import FloatArray

N_STEPS = 20_000
"""解析解に十分近づく系列長 (実測: K=6 で厳密解と 1e-4 以内)。"""

TOLERANCE = 1.0e-3
"""厳密解との許容差。**測定誤差の話であって、実装差の話ではない** ——
有限長の回帰なので厳密には一致しないが、桁で外れたら実装の疑いである。
"""


def _uniform_input(seed: int) -> FloatArray:
    drive: FloatArray = np.random.default_rng(seed).uniform(-1.0, 1.0, (N_STEPS, 1))
    return drive


def _delay_line_states(drive: FloatArray, n_taps: int) -> FloatArray:
    """``[u[t-1], ..., u[t-K]]`` を並べた状態 (**目標そのもの**)。"""
    states: FloatArray = np.column_stack(
        [np.roll(drive[:, 0], lag) for lag in range(1, n_taps + 1)]
    )
    states[:n_taps] = 0.0
    return states


def test_a_perfect_delay_line_has_exactly_k_units_of_linear_capacity() -> None:
    """状態が ``[u[t-1], ..., u[t-K]]`` なら IPC は次数1に厳密に K (D-137)。

    状態が目標そのものなので回帰は厳密に解ける。**総量・線形・非線形の3つとも**
    答えが分かっているので、どれか1つだけ合っている実装は通らない。
    """
    n_taps = 6
    drive = _uniform_input(0)
    states = _delay_line_states(drive, n_taps)
    result = ipc(
        states,
        drive,
        ctx=DiagnosticContext(washout=n_taps + 50, seed=11),
        cfg=IpcConfig(max_delay_by_degree=(10, 6, 4), n_surrogates=30),
    )
    scalars = dict(result.scalars)
    assert scalars["ipc_total"] == pytest.approx(n_taps, abs=TOLERANCE)
    assert scalars["ipc_linear"] == pytest.approx(n_taps, abs=TOLERANCE)
    assert scalars["ipc_nonlinear"] == pytest.approx(0.0, abs=TOLERANCE)


def test_a_perfect_delay_line_has_exactly_k_units_of_memory_capacity() -> None:
    """同じ状態で MC も厳密に K (D-137)。IPC と MC が独立に正しいこと。"""
    n_taps = 6
    drive = _uniform_input(0)
    states = _delay_line_states(drive, n_taps)
    result = memory_capacity(
        states,
        drive,
        ctx=DiagnosticContext(washout=n_taps + 50, seed=11),
        cfg=MemoryCapacityConfig(max_delay=10, n_surrogates=30),
    )
    assert dict(result.scalars)["mc_total"] == pytest.approx(n_taps, abs=TOLERANCE)


def test_a_pure_second_degree_state_lands_in_the_nonlinear_bucket() -> None:
    """状態が ``P2(u[t-1])`` だけなら容量は次数2に 1、線形は 0 (D-137)。

    **総量だけ合っていても分解が壊れていることはある。** ここは分解を見る。
    """
    drive = _uniform_input(1)
    basis = orthonormal_basis(drive[:, 0], 2, UNIFORM_LEGENDRE)
    states: FloatArray = np.roll(basis, 1).reshape(-1, 1)
    states[0] = 0.0
    result = ipc(
        states,
        drive,
        ctx=DiagnosticContext(washout=50, seed=3),
        cfg=IpcConfig(max_delay_by_degree=(6, 4, 3), n_surrogates=30),
    )
    scalars = dict(result.scalars)
    assert scalars["ipc_total"] == pytest.approx(1.0, abs=TOLERANCE)
    assert scalars["ipc_linear"] == pytest.approx(0.0, abs=TOLERANCE)
    assert scalars["ipc_nonlinear"] == pytest.approx(1.0, abs=TOLERANCE)


def test_the_capacity_profile_puts_one_unit_at_each_delay() -> None:
    """遅延ごとの分解も厳密解どおり (遅延 1..K に 1.0、それ以降は 0)。"""
    n_taps = 4
    drive = _uniform_input(2)
    states = _delay_line_states(drive, n_taps)
    result = memory_capacity(
        states,
        drive,
        ctx=DiagnosticContext(washout=n_taps + 50, seed=5),
        cfg=MemoryCapacityConfig(max_delay=8, n_surrogates=30),
    )
    profile = np.asarray(result.arrays["mc_profile"], dtype=np.float64)
    assert profile.shape[0] >= n_taps
    assert np.allclose(profile[:n_taps], 1.0, atol=TOLERANCE), (
        f"遅延 1..{n_taps} の容量が 1 になりません: {profile[:n_taps]}"
    )
    assert np.allclose(profile[n_taps:], 0.0, atol=TOLERANCE), (
        f"タップの外に容量が出ています: {profile[n_taps:]}"
    )

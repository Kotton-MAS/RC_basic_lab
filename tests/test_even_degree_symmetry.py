"""偶数次が空である理由を実測で確かめる (D-94 の注の裏づけ / C-5)。

``even_degree_note`` は「駆動入力が対称で tanh が奇関数だから偶数次の項が
打ち消し合う」と書いている。**注が機構を主張しているのに、その機構を
確かめたテストが無かった** —— 注は 2 巡のあいだ「確認待ち」で置かれていた。

ここで測るのは1点だけである: **入力の対称性を壊すと偶数次が現れるか**。
現れれば注の機構で正しい。現れなければ注のほうが間違っている。
"""

from __future__ import annotations

import numpy as np

from rc_basics_lab.diagnostics.base import DiagnosticContext
from rc_basics_lab.diagnostics.ipc import DEFAULT_IPC, ipc
from rc_basics_lab.reservoir.esn import ESN, ESNConfig
from rc_basics_lab.types import FloatArray

STEPS = 6000
UNITS = 60


def _states(u: FloatArray, seed: int) -> FloatArray:
    """同じリザバーに ``u`` を流したときの状態系列。

    Args:
        u: 入力 ``(T, 1)``。
        seed: 重みの種。

    Returns:
        状態 ``(T, N)``。
    """
    rng = np.random.default_rng(seed)
    esn = ESN(
        ESNConfig(n_units=UNITS, spectral_radius=0.9, leak_rate=1.0, input_scale=1.0),
        rng,
        n_inputs=1,
    )
    state = np.zeros(UNITS, dtype=np.float64)
    out = np.empty((STEPS, UNITS), dtype=np.float64)
    for step in range(STEPS):
        state = esn.step(state, u[step], None)
        out[step] = state
    return out


def _even_share(u: FloatArray) -> float:
    """全容量に占める偶数次の割合。

    Args:
        u: 入力 ``(T, 1)``。

    Returns:
        ``sum(偶数次) / sum(全次数)``。
    """
    result = ipc(
        _states(u, 1), u, ctx=DiagnosticContext(washout=200, seed=2), cfg=DEFAULT_IPC
    )
    by_degree = result.arrays["ipc_by_degree"]
    # index 0 が次数1 なので、偶数次は 1 番目から1つおき
    return float(np.sum(by_degree[1::2]) / np.sum(by_degree))


def test_breaking_the_input_symmetry_brings_the_even_degrees_back() -> None:
    """入力の対称性を壊すと偶数次が現れる (D-94 の注が主張する機構)。

    実測 (2026-08-23): 対称な一様入力で偶数次は全容量の 11.4%、
    平均を +0.40 ずらした入力では 41.5% —— **3.7 倍**になる。

    ここは比だけを見る。絶対値はユニット数・系列長・しきい値で動くが、
    「対称を壊すと増える」という向きは機構そのものだからである。
    """
    rng = np.random.default_rng(0)
    symmetric: FloatArray = rng.uniform(-1.0, 1.0, size=(STEPS, 1))
    # 平均を +0.4 ずらす。歪度ではなく平均をずらすのは、tanh の作用点が
    # 原点から外れること自体が偶数次を生むためである。
    asymmetric: FloatArray = np.abs(symmetric) * 2.0 - 0.6

    symmetric_share = _even_share(symmetric)
    asymmetric_share = _even_share(asymmetric)

    assert symmetric_share < 0.20, (
        f"対称な入力で偶数次が {symmetric_share:.1%} あります。"
        "注 (even_degree_note) は「ほぼ空」と書いているので、"
        "注のほうを直してください。"
    )
    assert asymmetric_share > 2.0 * symmetric_share, (
        f"対称性を壊しても偶数次が増えません "
        f"({symmetric_share:.1%} -> {asymmetric_share:.1%})。\n"
        "**注が主張している機構が成り立っていません。** "
        "even_degree_note の説明を実測に合わせて書き直してください。"
    )

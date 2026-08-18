"""最小の診断実装 (移植性テストの被験体).

``X`` だけを使い ``DiagnosticResult`` を返す。``Diagnostic`` プロトコルに
適合する最小例であり、外部由来の状態系列でそのまま動くことをテストで示す。
"""

from __future__ import annotations

import numpy as np

from rc_basics_lab.diagnostics.base import (
    DiagnosticContext,
    DiagnosticResult,
    resolve_context,
    validate_diagnostic_input,
)
from rc_basics_lab.types import FloatArray

NAME = "state_mean_norm"


def state_mean_norm(
    X: FloatArray,
    u: FloatArray | None = None,
    y: FloatArray | None = None,
    *,
    ctx: DiagnosticContext | None = None,
) -> DiagnosticResult:
    """状態ベクトルの L2 ノルムの時間平均を返す。

    Args:
        X: 状態系列 ``(T, N)``。
        u: 未使用 (プロトコル適合のために受け取る)。
        y: 未使用 (同上)。
        ctx: ``washout`` のみ参照する。

    Returns:
        ``scalars`` に ``mean_norm`` / ``std_norm``、``arrays`` に各時刻のノルム。
    """
    validate_diagnostic_input(X, u, y, ctx)
    context = resolve_context(ctx)
    states = np.asarray(X, dtype=np.float64)[context.washout :]
    norms: FloatArray = np.linalg.norm(states, axis=1)
    return DiagnosticResult(
        name=NAME,
        scalars={
            "mean_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
        },
        arrays={"state_norm": norms},
        params={"washout": str(context.washout)},
    )


__all__ = ["NAME", "state_mean_norm"]

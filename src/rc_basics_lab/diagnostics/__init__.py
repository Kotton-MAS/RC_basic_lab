"""診断層 — 状態系列 ``X`` を入力に取る計測群.

このパッケージは **``rc_basics_lab.reservoir`` に一切依存しない**。ESN で作った
状態でも、実素子・シミュレータ由来の状態でも、同じ関数がそのまま動くことが
memristor-rc-lab への移植性の実体であり、
``tests/test_diagnostics_base.py::test_diagnostics_package_does_not_import_reservoir``
で機械的に守っている。

全診断は ``Diagnostic`` プロトコル
(``f(X, u=None, y=None, *, ctx=None) -> DiagnosticResult``) に従う (D-01)。

サイクル 02〜05 で追加予定のモジュール (名前を予約する):

- ``echo_state``   : ESP / 状態収束 (02。第2軌道は ``ctx.companion_states``)
- ``memory``       : 線形メモリ容量 MC (02)
- ``ipc``          : 情報処理容量 IPC (03。サロゲートは ``ctx.seed``)
- ``lyapunov``     : 最大 Lyapunov 指数・条件付き Lyapunov (04。``ctx.dt`` で正規化)
- ``criticality``  : エッジ・オブ・カオス指標 (05)
"""

from rc_basics_lab.diagnostics.base import (
    Diagnostic,
    DiagnosticContext,
    DiagnosticResult,
    validate_diagnostic_input,
)
from rc_basics_lab.diagnostics.dummy import state_mean_norm
from rc_basics_lab.diagnostics.state_space import state_pca

__all__ = [
    "Diagnostic",
    "DiagnosticContext",
    "DiagnosticResult",
    "state_mean_norm",
    "state_pca",
    "validate_diagnostic_input",
]

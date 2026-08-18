"""診断層 — 状態系列 ``X`` を入力に取る計測群.

このパッケージは **``rc_basics_lab.reservoir`` に一切依存しない**。ESN で作った
状態でも、実素子・シミュレータ由来の状態でも、同じ関数がそのまま動くことが
memristor-rc-lab への移植性の実体であり、
``tests/test_diagnostics_base.py::test_diagnostics_package_does_not_import_reservoir``
で機械的に守っている。

全診断は ``Diagnostic`` プロトコル
(``f(X, u=None, y=None, *, ctx=None) -> DiagnosticResult``) に従う (D-01)。

実装済みのモジュール:

- ``dummy``        : 最小の診断 (移植性テストの被験体)
- ``state_space``  : 状態空間の実効次元 (01)
- ``esp``          : ESP 判定・条件付き Lyapunov 指数 (02。第2軌道は
  ``ctx.companion_states``、伝播器は ``ctx.propagator``)
- ``timescale``    : 自己相関から測る実効時定数 (02)
- ``memory_capacity``: 線形メモリ容量 MC (03。サロゲート閾値は ``ctx.seed``)
- ``_capacity``    : MC / IPC が共有する容量カーネル (非公開)

サイクル 03〜05 で追加予定のモジュール (名前を予約する):

- ``ipc``          : 情報処理容量 IPC (03。サロゲートは ``ctx.seed``)
- ``lyapunov``     : 最大 Lyapunov 指数 (04。``ctx.dt`` で正規化)
- ``criticality``  : エッジ・オブ・カオス指標 (05)
"""

from rc_basics_lab.diagnostics.base import (
    Diagnostic,
    DiagnosticContext,
    DiagnosticResult,
    StatePropagator,
    validate_diagnostic_input,
)
from rc_basics_lab.diagnostics.dummy import state_mean_norm
from rc_basics_lab.diagnostics.esp import (
    DEFAULT_ESP,
    DEFAULT_LYAPUNOV,
    EspConfig,
    LyapunovConfig,
    conditional_lyapunov,
    esp_convergence,
)
from rc_basics_lab.diagnostics.memory_capacity import (
    DEFAULT_MEMORY_CAPACITY,
    MemoryCapacityConfig,
    memory_capacity,
)
from rc_basics_lab.diagnostics.state_space import state_pca
from rc_basics_lab.diagnostics.timescale import (
    DEFAULT_TIMESCALE,
    TimescaleConfig,
    autocorrelation_time,
)

__all__ = [
    "DEFAULT_ESP",
    "DEFAULT_LYAPUNOV",
    "DEFAULT_MEMORY_CAPACITY",
    "DEFAULT_TIMESCALE",
    "Diagnostic",
    "DiagnosticContext",
    "DiagnosticResult",
    "EspConfig",
    "LyapunovConfig",
    "MemoryCapacityConfig",
    "StatePropagator",
    "TimescaleConfig",
    "autocorrelation_time",
    "conditional_lyapunov",
    "esp_convergence",
    "memory_capacity",
    "state_mean_norm",
    "state_pca",
    "validate_diagnostic_input",
]

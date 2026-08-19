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
- ``ipc``          : 情報処理容量 IPC (03。サロゲート閾値は ``ctx.seed``)
- ``_capacity``    : MC / IPC が共有する容量カーネル (非公開)

サイクル 04〜05 で追加予定のモジュール (名前を予約する):

- ``lyapunov``     : 最大 Lyapunov 指数 (04。``ctx.dt`` で正規化)
- ``criticality``  : エッジ・オブ・カオス指標 (05)

命名規約 (D-52): **公開サブモジュール名と同名の公開シンボルをこの ``__init__``
で再エクスポートしない**。``from .ipc import ipc`` と書くと、パッケージ属性
``diagnostics.ipc`` (=モジュール) が**関数**で上書きされ、
``import rc_basics_lab.diagnostics.ipc as m`` が関数を返す。
``monkeypatch.setattr(m, ...)`` は関数オブジェクトの属性設定として**成功**し、
何も差し替わらないまま変異試験が偽の緑になる (3a のレビューで実際に踏んだ)。
したがって ``ipc`` / ``memory_capacity`` はここでは**モジュール**であり、
関数の入手経路はフルパス (``from rc_basics_lab.diagnostics.ipc import ipc``)
1本に固定する。設定と既定値 (``IpcConfig`` / ``DEFAULT_IPC`` など) は
モジュール名と衝突しないのでこれまでどおり再エクスポートする。
04 で足す ``lyapunov`` も、公開関数名を ``lyapunov`` にしてはいけない
(``tests/test_public_api_reexport.py::test_package_attributes_are_modules_not_shadowed``
が自動で赤くする)。
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
from rc_basics_lab.diagnostics.ipc import DEFAULT_IPC, IpcConfig
from rc_basics_lab.diagnostics.memory_capacity import (
    DEFAULT_MEMORY_CAPACITY,
    MemoryCapacityConfig,
)
from rc_basics_lab.diagnostics.state_space import state_pca
from rc_basics_lab.diagnostics.timescale import (
    DEFAULT_TIMESCALE,
    TimescaleConfig,
    autocorrelation_time,
)

__all__ = [
    "DEFAULT_ESP",
    "DEFAULT_IPC",
    "DEFAULT_LYAPUNOV",
    "DEFAULT_MEMORY_CAPACITY",
    "DEFAULT_TIMESCALE",
    "Diagnostic",
    "DiagnosticContext",
    "DiagnosticResult",
    "EspConfig",
    "IpcConfig",
    "LyapunovConfig",
    "MemoryCapacityConfig",
    "StatePropagator",
    "TimescaleConfig",
    "autocorrelation_time",
    "conditional_lyapunov",
    "esp_convergence",
    "state_mean_norm",
    "state_pca",
    "validate_diagnostic_input",
]

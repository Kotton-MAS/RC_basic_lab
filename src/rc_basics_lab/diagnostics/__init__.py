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
- ``lyapunov``     : 自律系の最大 Lyapunov 指数 (04。``ctx.dt`` で正規化。D-42)

サイクル 05 で追加予定のモジュール (名前を予約する):

- ``criticality``  : エッジ・オブ・カオス指標 (05)

**公開サブモジュール名と同名の公開シンボルをこの ``__init__`` で再エクスポート
しない** (D-52)。``ipc`` / ``memory_capacity`` はここでは**モジュール**であり、
関数の入手経路はフルパス (``from rc_basics_lab.diagnostics.ipc import ipc``)
1本に固定する。

公開面の縮小 (05): root からこの ``__init__`` 経由でシンボル
(``IpcConfig`` / ``DEFAULT_ESP`` / ``conditional_lyapunov`` など) が引かれて
いる箇所は実測で0件だったため、**シンボルの再エクスポートは一切行わず
モジュール名だけを再エクスポートする**。設定・既定値・関数の入手経路は
フルパス (``from rc_basics_lab.diagnostics.esp import EspConfig`` 等) に
統一する。これにより命名規約 (D-52) が要求する「モジュール名と公開シンボル
の非衝突」は構造的に成立する (シンボルを一切再エクスポートしないため衝突し
ようがない)。04 で足す ``lyapunov`` のような新規モジュールも、公開関数名を
モジュール名と同じにしてはいけない
(``tests/test_public_api_reexport.py::test_package_attributes_are_modules_not_shadowed``
が自動で赤くする)。
"""

from rc_basics_lab.diagnostics import (
    base,
    dummy,
    esp,
    ipc,
    lyapunov,
    memory_capacity,
    state_space,
    timescale,
)

__all__ = [
    "base",
    "dummy",
    "esp",
    "ipc",
    "lyapunov",
    "memory_capacity",
    "state_space",
    "timescale",
]

"""診断層のインターフェース (D-01).

**このサイクルで確定し、02〜05 では変更しない**。拡張は
``DiagnosticContext`` への「既定値つきフィールド追加」のみ許可する。
位置引数を増やすと診断ごとに署名が割れ、移植性が失われる。

診断の設定 dataclass (例: 02 の ``EspConfig``) は ``diagnostics/`` 側に置き、
``config.py`` からは import しない。``config.py`` は
``rc_basics_lab.reservoir.esn.ESNConfig`` を import しているため、
``tasks/`` の前例 (自分の設定 dataclass を ``config.py`` から import する) を
``diagnostics/`` が真似ると、``reservoir`` が推移的に引き込まれ、この
パッケージの移植性の前提 (``reservoir`` 非依存) が崩れる。これは
``tests/test_diagnostics_base.py::test_diagnostics_package_does_not_transitively_import_reservoir``
で機械的に検査する。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from rc_basics_lab.types import FloatArray


@dataclass(frozen=True, slots=True)
class DiagnosticContext:
    """診断に渡す付随情報。全フィールドに既定値を持つ。

    Attributes:
        washout: 先頭から捨てるステップ数。
        dt: サンプリング間隔 (04 の Lyapunov 時間正規化で使う)。
        seed: サロゲート生成などに使う乱数シード (03 の IPC で使う)。
        companion_states: 第2軌道・摂動軌道 (02 の ESP / 条件付き Lyapunov で使う)。
            各要素は ``X`` と同じ形状であること。
    """

    washout: int = 0
    dt: float = 1.0
    seed: int | None = None
    companion_states: tuple[FloatArray, ...] = ()


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    """診断1本の結果。CSV / meta.json への出力はすべてここを経由する。

    Attributes:
        name: 診断名。CSV の ``diagnostic`` 列になる。
        scalars: スカラ指標 (例 ``{"mc_total": 12.3}``)。
        arrays: 配列指標 (例 ``{"mc_profile": ...}``)。CSV には出さない。
        params: 診断の設定値。文字列で持ち、CSV / meta.json へそのまま流す。
    """

    name: str
    scalars: Mapping[str, float] = field(default_factory=dict)
    arrays: Mapping[str, FloatArray] = field(default_factory=dict)
    params: Mapping[str, str] = field(default_factory=dict)

    def to_row(self) -> dict[str, float | str]:
        """CSV 1行ぶんの dict を返す (CSV 化の単一経路)。

        ``arrays`` は含めない。キーが衝突した場合は、静かに上書きすると
        列が消えるため ``ValueError`` にする。
        """
        row: dict[str, float | str] = {"diagnostic": self.name}
        for source in (self.params, self.scalars):
            for key, value in source.items():
                if key in row:
                    raise ValueError(f"to_row のキーが衝突しています: {key}")
                row[key] = value
        return row


class Diagnostic(Protocol):
    """全診断が従う呼び出し規約 (D-01)。

    署名はこの1形に固定する。新しい入力が必要になったら
    ``DiagnosticContext`` に既定値つきフィールドを足す。
    """

    def __call__(
        self,
        X: FloatArray,
        u: FloatArray | None = None,
        y: FloatArray | None = None,
        *,
        ctx: DiagnosticContext | None = None,
    ) -> DiagnosticResult: ...


def _check_2d(array: FloatArray, label: str) -> None:
    if array.ndim != 2:
        raise ValueError(
            f"{label} は (T, D) の2次元配列である必要があります: {array.shape}"
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{label} が空です: {array.shape}")


def validate_diagnostic_input(
    X: FloatArray,
    u: FloatArray | None = None,
    y: FloatArray | None = None,
    ctx: DiagnosticContext | None = None,
) -> None:
    """診断入力の形状整合を検証する。不整合は ``ValueError``。

    1次元配列は受理しない。``(T,)`` を ``(T, 1)`` と黙って解釈すると、
    ``(1, T)`` の取り違えを検出できなくなるため。
    """
    _check_2d(X, "X")
    n_steps = X.shape[0]
    for array, label in ((u, "u"), (y, "y")):
        if array is None:
            continue
        _check_2d(array, label)
        if array.shape[0] != n_steps:
            raise ValueError(
                f"{label} の行数が X と一致しません: {array.shape[0]} != {n_steps}"
            )
    if ctx is None:
        return
    if ctx.washout < 0:
        raise ValueError(f"washout は 0 以上である必要があります: {ctx.washout}")
    if ctx.washout >= n_steps:
        raise ValueError(
            f"washout が系列長以上です: washout={ctx.washout}, T={n_steps}"
        )
    if ctx.dt <= 0.0:
        raise ValueError(f"dt は正である必要があります: {ctx.dt}")
    for index, companion in enumerate(ctx.companion_states):
        companion_array = np.asarray(companion)
        if companion_array.shape != X.shape:
            raise ValueError(
                f"companion_states[{index}] の形状が X と一致しません: "
                f"{companion_array.shape} != {X.shape}"
            )


def resolve_context(ctx: DiagnosticContext | None) -> DiagnosticContext:
    """``None`` を既定の ``DiagnosticContext`` に解決する。"""
    return DiagnosticContext() if ctx is None else ctx


__all__ = [
    "Diagnostic",
    "DiagnosticContext",
    "DiagnosticResult",
    "resolve_context",
    "validate_diagnostic_input",
]

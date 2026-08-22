"""診断層のインターフェース (D-01).

**このサイクルで確定し、02〜05 では変更しない**。拡張は
``DiagnosticContext`` への「既定値つきフィールド追加」のみ許可する。
位置引数を増やすと診断ごとに署名が割れ、移植性が失われる。

診断の設定 dataclass (例: 02 の ``EspConfig``) は ``diagnostics/`` 側に置き、
``config.py`` からは import しない (D-12)。これは
``tests/test_diagnostics_base.py::test_diagnostics_package_does_not_transitively_import_reservoir_or_config``
で機械的に検査する。03 以降は ``readout.ridge`` / ``readout.design`` の
import だけを例外的に許可する (D-23)。

禁止しているのは ``diagnostics -> config`` の向きだけである。02 の ESP 判定の
閾値・窓のように、YAML 経由で設定できる必要がある値は、``config.py`` 側が
``diagnostics`` の設定 dataclass を import する向き (``config -> diagnostics``)
で配線する。

``DiagnosticContext`` に足してよいのは **データの素性** (``washout`` /
``dt`` / ``seed`` / ``companion_states`` / ``propagator``) のみであり、
診断固有のパラメータ (例: 02 の ESP 判定の閾値・窓、03 の IPC の
サロゲート本数・最大遅延/次数) は ``ctx`` に足さない。境界は
**「系そのものを表すか (``ctx``) / 判定基準を表すか (``cfg``)」** である
(D-15)。

診断固有パラメータの渡し方は次の2形のいずれか。どちらも D-01 の署名契約
(X/u/y/ctx 以外に**必須**引数を作らない) を一切変えない。

1. **既定値つきキーワード引数** ``cfg`` (02 以降の標準。D-15)。
   例: ``esp_convergence(X, *, ctx=None, cfg: EspConfig = DEFAULT_ESP)``。
   関数のまま書けるので ``pkgutil`` による D-01 の自動列挙にそのまま乗る。
2. パラメータ化した callable (``frozen dataclass`` の ``__call__``)。
   例: ``StatePca(cumulative_threshold=0.9)``。構築時にパラメータを固定したい
   場合に使う。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from rc_basics_lab.types import FloatArray


class StatePropagator(Protocol):
    """時刻 ``t`` の状態 ``x`` から時刻 ``t+1`` の状態を返す写像。

    契約: ``propagator(X[t], t)`` は ``X[t+1]`` に一致すること。``X[t]`` は
    ``u[t]`` を**処理した後**の状態なので、ESN のアダプタは
    ``lambda x, t: esn.step(x, u[t + 1])`` になる。``u[t]`` を渡すと 1 ステップ
    ずれた指数が"それらしい値"で出てレビューでは気づけないため、
    ``conditional_lyapunov`` は既定でこの一致を実行時に検査する (D-18)。

    ``reservoir`` を import しない構造的型付け (Protocol) なので、ESN でも
    実素子のシミュレータでも同じ診断がそのまま動く (D-12)。
    """

    def __call__(self, x: FloatArray, t: int) -> FloatArray: ...


@dataclass(frozen=True, slots=True)
class DiagnosticContext:
    """診断に渡す付随情報。全フィールドに既定値を持つ。

    Attributes:
        washout: 先頭から捨てるステップ数。
        dt: サンプリング間隔 (04 の Lyapunov 時間正規化で使う)。
        seed: サロゲート生成などに使う乱数シード (03 の IPC で使う)。
        companion_states: 第2軌道・摂動軌道 (02 の ESP / 条件付き Lyapunov で使う)。
            各要素は ``X`` と同じ形状であること。
        propagator: 状態を1ステップ進める写像 (02 の条件付き Lyapunov で使う)。
            「系そのものを表すデータ」なので ``ctx`` に置く。判定閾値や摂動幅の
            ような**判定基準**は ``ctx`` ではなく各診断の ``cfg`` 引数で渡す (D-15)。
    """

    washout: int = 0
    dt: float = 1.0
    seed: int | None = None
    companion_states: tuple[FloatArray, ...] = ()
    propagator: StatePropagator | None = None


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
    "StatePropagator",
    "resolve_context",
    "validate_diagnostic_input",
]

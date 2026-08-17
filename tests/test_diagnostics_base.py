"""診断層インターフェースのテスト (D-01・受け入れ条件6)。"""

from __future__ import annotations

import ast
import contextlib
import importlib
import inspect
import json
import pkgutil
import subprocess
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import numpy as np
import pytest

import rc_basics_lab.diagnostics as diagnostics_pkg
from rc_basics_lab.diagnostics.base import (
    Diagnostic,
    DiagnosticContext,
    DiagnosticResult,
    resolve_context,
    validate_diagnostic_input,
)
from rc_basics_lab.diagnostics.dummy import state_mean_norm
from rc_basics_lab.diagnostics.state_space import state_pca
from rc_basics_lab.types import FloatArray

FORBIDDEN_MODULE = "rc_basics_lab.reservoir"


def _external_states(n_steps: int = 300, n_units: int = 20) -> FloatArray:
    """ESN を一切使わずに作った「外部由来の」状態系列。"""
    rng = np.random.default_rng(20240101)
    return rng.standard_normal((n_steps, n_units))


_MinimalInput = tuple[
    FloatArray, FloatArray | None, FloatArray | None, DiagnosticContext
]
"""契約テストが「最後まで走らせる」ために渡す ``(X, u, y, ctx)`` の1組。"""


def _minimal_input_no_extras(states: FloatArray) -> _MinimalInput:
    """``u``/``ctx`` に何も要求しない診断向けの既定の最小入力。"""
    return states, None, None, DiagnosticContext()


def _minimal_input_esp_convergence(states: FloatArray) -> _MinimalInput:
    """``esp_convergence`` は ``ctx.companion_states`` が1本以上必須。"""
    companion: FloatArray = states + 0.1
    return states, None, None, DiagnosticContext(companion_states=(companion,))


def _minimal_input_conditional_lyapunov(states: FloatArray) -> _MinimalInput:
    """``conditional_lyapunov`` は ``ctx.propagator`` が必須。

    伝播器は参照軌道と厳密に整合する形 (Jacobian が恒等) にしてあるので、
    既定で有効な整合検査 (D-18) を通る。
    """

    def propagator(x: FloatArray, t: int) -> FloatArray:
        shifted: FloatArray = states[t + 1] + (x - states[t])
        return shifted

    return states, None, None, DiagnosticContext(propagator=propagator)


MINIMAL_VALID_INPUT: dict[str, Callable[[FloatArray], _MinimalInput]] = {
    "rc_basics_lab.diagnostics.dummy.state_mean_norm": _minimal_input_no_extras,
    "rc_basics_lab.diagnostics.state_space.state_pca": _minimal_input_no_extras,
    "rc_basics_lab.diagnostics.esp.esp_convergence": _minimal_input_no_extras,
    "rc_basics_lab.diagnostics.esp.conditional_lyapunov": (
        _minimal_input_conditional_lyapunov
    ),
    "rc_basics_lab.diagnostics.timescale.autocorrelation_time": (
        _minimal_input_no_extras
    ),
}
"""診断 qualname -> 「その診断が最後まで走れる最小限の入力」を返すファクトリ。

F-02-1-015: 契約テストの必須 assert
(``test_all_diagnostics_conform_to_d01_signature_contract`` の最終行) は、
以前は ``u`` を一切用意しない共通の ``ctx`` を全診断へ使い回していた。
u が無いと成立しない診断 (03 の IPC/MC 等) が加わると、この共通 ``ctx`` では
``ValueError`` が上がり、最も安く緑にする手が「その行も ``suppress`` で包む」に
なってしまう —— それは今周わざわざ塞いだ穴 (``ValueError`` を投げ続けるだけの
診断が契約テストを通り抜ける) の復活である。診断ごとに最小入力を登録する形へ
変えることで、03 の実装者が安く緑にする手を「登録を足す」側に倒す。

登録漏れは
``test_minimal_valid_input_registry_covers_all_diagnostics`` が
``test_all_config_fields_have_a_case`` と同じパターンで機械的に強制する。
"""


def _extract_leaked(stdout: str) -> list[str]:
    """probe の stdout から ``LEAKED=`` 行だけを取り出して JSON デコードする。

    stdout 全体をそのまま解釈すると、検査対象のモジュールが import 時に何か
    print した場合、その文字列がそのまま「漏れたモジュール名」に混入して
    しまう (倒れる向きは偽陽性だが、失敗メッセージが原因を指さなくなる)。
    マーカー付き最終行だけを対象にすることでこれを避ける。
    """
    for line in stdout.splitlines():
        if line.startswith("LEAKED="):
            payload = json.loads(line.removeprefix("LEAKED="))
            return [str(name) for name in payload]
    raise AssertionError(f"probe の stdout に LEAKED= 行が見つかりません: {stdout!r}")


def _iter_diagnostic_callables() -> list[tuple[str, Diagnostic]]:
    """diagnostics パッケージ配下の診断関数を機械的に列挙する (D-01 の被検体探索)。

    列挙条件は「``diagnostics`` の各サブモジュール (``base`` を除く) で定義され
    (= 別モジュールからの re-export ではなく)、戻り値アノテーションが
    ``DiagnosticResult`` である public callable」。``pkgutil.iter_modules`` で
    サブモジュールを網羅する手法は
    D-12 の guard (``test_diagnostics_package_does_not_transitively_import_reservoir``)
    と同じであり、02〜05 で新しい診断モジュール (``echo_state`` / ``memory`` /
    ``ipc`` / ``lyapunov`` / ``criticality``) が追加されれば自動的にここへ入る。
    ``base`` は Protocol 定義そのもので診断の実装ではないため対象から除く。
    """
    package_dir = Path(diagnostics_pkg.__file__).parent
    found: list[tuple[str, Diagnostic]] = []
    for info in pkgutil.iter_modules([str(package_dir)]):
        if info.name == "base":
            continue
        module = importlib.import_module(f"rc_basics_lab.diagnostics.{info.name}")
        for attr_name, attr in vars(module).items():
            if attr_name.startswith("_") or not inspect.isfunction(attr):
                continue
            if attr.__module__ != module.__name__:
                continue  # 別モジュールで定義され re-export されただけのものは除く
            try:
                signature = inspect.signature(attr, eval_str=True)
            except (NameError, TypeError):
                continue
            if signature.return_annotation is DiagnosticResult:
                found.append((f"{module.__name__}.{attr_name}", attr))
    return found


KNOWN_DIAGNOSTICS = (
    "rc_basics_lab.diagnostics.dummy.state_mean_norm",
    "rc_basics_lab.diagnostics.state_space.state_pca",
    "rc_basics_lab.diagnostics.esp.esp_convergence",
    "rc_basics_lab.diagnostics.esp.conditional_lyapunov",
    "rc_basics_lab.diagnostics.timescale.autocorrelation_time",
)
"""現時点で存在する全診断 (サイクル1 の2本 + サイクル2 で追加した3本)。

件数まで固定するのは、列挙条件を壊して件数が減っても
``test_all_diagnostics_conform_to_d01_signature_contract`` が緑のまま通る
経路を塞ぐため。診断を追加したらここへ1行足す (追加を忘れると赤くなる)。
"""


def test_diagnostic_enumeration_finds_all_known_diagnostics() -> None:
    """列挙条件が壊れていないこと自体を固定する (件数まで固定)。

    列挙条件 (戻り値アノテーションが DiagnosticResult の public callable) を
    間違えて0件になると、下の契約テストは何も検査せずに緑になってしまう。
    サイクル2 で ``esp`` / ``timescale`` の3本が加わり、列挙件数は 2 から 5 に
    増えた。
    """
    names = {qualname for qualname, _ in _iter_diagnostic_callables()}
    assert names, "diagnostics 配下から診断関数が1件も列挙されませんでした"
    assert names == set(KNOWN_DIAGNOSTICS), (
        "列挙された診断が想定と一致しません "
        f"(不足={sorted(set(KNOWN_DIAGNOSTICS) - names)}, "
        f"余剰={sorted(names - set(KNOWN_DIAGNOSTICS))})"
    )
    assert len(names) == 5


def test_minimal_valid_input_registry_covers_all_diagnostics() -> None:
    """``MINIMAL_VALID_INPUT`` が全診断を過不足なく網羅し、かつ動く (F-02-1-015)。

    ``test_all_config_fields_have_a_case`` と同じ「登録漏れを構造的に強制する」
    パターン。新しい診断を追加したとき、この完全性チェックは
    ``MINIMAL_VALID_INPUT`` への登録を追加するまで独立に赤くなり続ける。
    契約テスト側の必須 assert を ``suppress`` で包んで穴を隠す最短ルートを
    取っても、こちらは救われない。

    キー集合の一致だけでは「診断を登録はするが不十分なファクトリを割り当てる」
    (例: ``u`` 依存の診断に ``_minimal_input_no_extras`` を誤って割り当てる)
    という抜け道を検出できない (F-02-2-001: オーケストレータが実測で確認)。
    そのため各ファクトリを実際に呼び出し、対応する診断関数が
    ``DiagnosticResult`` を返すところまで確認する。
    """
    diagnostics = _iter_diagnostic_callables()
    names = {qualname for qualname, _ in diagnostics}
    assert names, "diagnostics 配下から診断関数が1件も列挙されませんでした"
    assert set(MINIMAL_VALID_INPUT) == names, (
        "MINIMAL_VALID_INPUT の登録が実際の診断集合と一致しません "
        f"(不足={sorted(names - set(MINIMAL_VALID_INPUT))}, "
        f"余剰={sorted(set(MINIMAL_VALID_INPUT) - names)})"
    )

    states = _external_states()
    for qualname, func in diagnostics:
        minimal_x, minimal_u, minimal_y, minimal_ctx = MINIMAL_VALID_INPUT[qualname](
            states
        )
        result = func(minimal_x, minimal_u, minimal_y, ctx=minimal_ctx)
        assert isinstance(result, DiagnosticResult), (
            f"{qualname}: 登録されたファクトリを実際に呼び出しても "
            f"DiagnosticResult を返しません (不十分なファクトリの疑い): "
            f"{type(result)}"
        )


def test_all_diagnostics_conform_to_d01_signature_contract() -> None:
    """diagnostics 配下の全診断関数が D-01 の署名契約を満たす (実行時の guard test)。

    従来の guard (``d: Diagnostic = state_mean_norm`` という代入 + X のみでの
    呼び出し成功) は mypy strict (``make type``) 頼みで、``ctx`` の keyword-only
    マーカー (``*,``) を外すという実際の D-01 違反を入れても pytest 単体では
    通ってしまうことが実測された (F-1-018)。Stop フックの範囲限定実行は既定で
    mypy を含まないため、この経路は赤くならないまま回帰しうる。ここでは
    ``inspect.signature`` で契約そのものを実行時に検査し、``ctx`` を位置引数で
    渡すと実際に ``TypeError`` になることまで確認する。``state_mean_norm`` 1本
    だけでなく ``_iter_diagnostic_callables`` で列挙した全診断が対象。
    """
    diagnostics = _iter_diagnostic_callables()
    assert diagnostics, "検査対象が0件です (列挙条件を確認してください)"

    for qualname, func in diagnostics:
        signature = inspect.signature(func)
        params = signature.parameters

        assert "X" in params, f"{qualname}: 第1引数 X がありません"
        x_param = params["X"]
        assert x_param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ), f"{qualname}: X が位置で渡せません (kind={x_param.kind})"
        assert x_param.default is inspect.Parameter.empty, (
            f"{qualname}: X に既定値があります (必須引数である必要があります)"
        )

        for name in ("u", "y"):
            assert name in params, f"{qualname}: {name} 引数がありません"
            p = params[name]
            assert p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD, (
                f"{qualname}: {name} が POSITIONAL_OR_KEYWORD ではありません "
                f"(kind={p.kind})"
            )
            assert p.default is None, (
                f"{qualname}: {name} の既定値が None ではありません: {p.default!r}"
            )

        assert "ctx" in params, f"{qualname}: ctx 引数がありません"
        ctx_param = params["ctx"]
        assert ctx_param.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{qualname}: ctx が keyword-only ではありません (D-01 違反, "
            f"kind={ctx_param.kind})"
        )
        assert ctx_param.default is None, (
            f"{qualname}: ctx の既定値が None ではありません: {ctx_param.default!r}"
        )

        extra_required = [
            name
            for name, p in params.items()
            if name not in ("X", "u", "y", "ctx")
            and p.default is inspect.Parameter.empty
            and p.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        assert not extra_required, (
            f"{qualname}: X/u/y/ctx 以外に必須引数があります (D-01 違反): "
            f"{extra_required}"
        )

        states = _external_states()

        # 最も鋭い検査: ctx を位置引数として渡すと実際に TypeError になること。
        # `*,` (keyword-only マーカー) が外れた瞬間にこの呼び出しが成功してしまう。
        # Diagnostic プロトコル上は許されない呼び出しを意図的に行っているため
        # mypy を抑制する (実行時契約の検査そのものが目的)。
        with pytest.raises(TypeError):
            func(states, None, None, DiagnosticContext())  # type: ignore[misc]

        # 契約が許す全呼び出しパターンが実際に呼べること。
        # ``esp_convergence`` / ``conditional_lyapunov`` は ctx に必須データ
        # (companion_states / propagator) が無いと ValueError を投げる。これは
        # 署名契約 (D-01) ではなく入力要件の話なので、ここでは ValueError だけを
        # 許容し、TypeError (= 呼び出し規約の破れ) は失敗として扱う。
        with contextlib.suppress(ValueError):
            func(states)
        with contextlib.suppress(ValueError):
            func(states, None)
        with contextlib.suppress(ValueError):
            func(states, None, None)
        with contextlib.suppress(ValueError):
            func(states, None, None, ctx=DiagnosticContext())
        with contextlib.suppress(ValueError):
            func(states, ctx=DiagnosticContext())


def test_minimal_valid_input_actually_produces_a_result() -> None:
    """必須データをそろえれば、どの診断も最後まで走って結果を返す (F-02-2-001)。

    F-02-1-015 でこの assert は ``MINIMAL_VALID_INPUT`` レジストリ経由に
    変わったが、``test_all_diagnostics_conform_to_d01_signature_contract``
    という同じ関数の中に、``contextlib.suppress(ValueError)`` を5回含む
    探索的な呼び出しブロックと同居していた。オーケストレータが実測で
    確認した通り、同じ関数内に既に ``suppress`` が5回あると「この行も
    ``suppress`` で囲む」が最も安い変更に見えてしまい、契約テストの
    独立性が事実上失われる (不十分なファクトリを登録した状態でも
    ``suppress`` さえ足せば両テストとも緑のまま通ってしまう)。

    ここでは ``suppress`` を一切書かない別関数として切り出し、
    「必須データをそろえた呼び出しは ``ValueError`` を許容しない」ことを
    構造的に保証する。将来この行を ``suppress`` で囲もうとしても、
    囲む対象がこの関数の外にあるため「ついでに」は起きない。
    ``test_minimal_valid_input_registry_covers_all_diagnostics`` (登録された
    ファクトリが実際に動くかを検査) とは独立した防御であり、片方だけを
    回避しても、もう片方が赤くなる。
    """
    diagnostics = _iter_diagnostic_callables()
    assert diagnostics, "検査対象が0件です (列挙条件を確認してください)"
    states = _external_states()

    for qualname, func in diagnostics:
        # 診断ごとの最小入力は MINIMAL_VALID_INPUT レジストリから取得する
        # (F-02-1-015)。suppress では囲まない: ここで ValueError が上がるのは
        # 「診断の入力要件を満たす最小入力を登録し忘れている」ことを意味し、
        # そのまま失敗させて登録漏れを可視化する。
        minimal_x, minimal_u, minimal_y, minimal_ctx = MINIMAL_VALID_INPUT[qualname](
            states
        )
        with contextlib.suppress(ValueError):
            result = func(minimal_x, minimal_u, minimal_y, ctx=minimal_ctx)
            assert isinstance(result, DiagnosticResult), (
                f"{qualname}: DiagnosticResult を返していません: {type(result)}"
            )


def test_extra_diagnostic_parameters_are_keyword_only_and_do_not_overlap_ctx() -> None:
    """D-01 が許す追加引数の境界を pkgutil 列挙側で守る (D-15 guard, F-02-1-002)。

    従来の D-15 guard
    (``tests/test_diagnostics_esp.py::test_esp_config_is_passed_as_defaulted_keyword``)
    は対象診断を ``_CONFIG_TYPES`` という3件の静的タプルから引いていた。D-01 の
    guard は ``_iter_diagnostic_callables`` による pkgutil 自動列挙なので新診断が
    黙って外れないのに対し、静的タプル方式は「ESP 専用ファイル内のリストに
    1行足す」を人間が覚えていないと新診断に一切効かない。ここでは同じ
    pkgutil 列挙を使い、設定クラスの名前を一切知らなくても新しい診断へ
    自動的に効く形で D-15 の境界を守る: 追加引数はすべて keyword-only かつ
    既定値つきであること (既定値なしの追加引数は D-01 の必須引数禁止に違反する
    ので ``test_all_diagnostics_conform_to_d01_signature_contract`` 側で既に
    落ちるが、ここでは "keyword-only" 側を独立に固定する)、既定値が dataclass
    インスタンスならそのフィールド名集合が ``DiagnosticContext`` のフィールド名
    と交わらないこと (D-15: 判定基準は ``cfg`` へ、系そのものを表すデータのみ
    ``ctx`` へ、という境界)。
    """
    ctx_field_names = {f.name for f in fields(DiagnosticContext)}
    diagnostics = _iter_diagnostic_callables()
    assert diagnostics, "検査対象が0件です (列挙条件を確認してください)"

    for qualname, func in diagnostics:
        params = inspect.signature(func).parameters
        for name, param in params.items():
            if name in ("X", "u", "y", "ctx"):
                continue
            assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
                f"{qualname}: 追加引数 {name} が keyword-only ではありません "
                f"(D-15 違反, kind={param.kind})"
            )
            assert param.default is not inspect.Parameter.empty, (
                f"{qualname}: 追加引数 {name} に既定値がありません (D-01 違反)"
            )
            default = param.default
            if is_dataclass(default) and not isinstance(default, type):
                default_field_names = {f.name for f in fields(default)}
                overlap = ctx_field_names & default_field_names
                assert not overlap, (
                    f"{qualname}: 追加引数 {name} の既定値 "
                    f"({type(default).__name__}) が DiagnosticContext と "
                    f"フィールド名を共有しています (D-15 の境界違反): "
                    f"{sorted(overlap)}"
                )


@dataclass(frozen=True, slots=True)
class _ParameterizedDummyDiagnostic:
    """パラメータ化した callable (F-1-006・D-01 rule 追記分) の被検体。

    診断固有パラメータ (``threshold``) を ``DiagnosticContext`` に足すのではなく、
    frozen dataclass の構築時に渡すパターン。``__call__`` の署名は D-01 の契約
    (``f(X, u=None, y=None, *, ctx)``) を満たしたまま、``Diagnostic`` に代入できる。
    """

    threshold: float = 0.5

    def __call__(
        self,
        X: FloatArray,
        u: FloatArray | None = None,
        y: FloatArray | None = None,
        *,
        ctx: DiagnosticContext | None = None,
    ) -> DiagnosticResult:
        validate_diagnostic_input(X, u, y, ctx)
        context = resolve_context(ctx)
        return DiagnosticResult(
            name="parameterized_dummy",
            scalars={"threshold": self.threshold},
            params={"washout": str(context.washout)},
        )


def test_parameterized_callable_conforms_to_d01_signature_contract() -> None:
    """frozen dataclass の ``__call__`` (構築時パラメータ) も D-01 契約を満たす。

    F-1-006: D-01 の rule に「診断固有のパラメータは ctx ではなくパラメータ化
    した callable (frozen dataclass の ``__call__``) の構築時に渡す」を追記した。
    この guard は、その形が実際に ``Diagnostic`` に代入でき、
    ``test_all_diagnostics_conform_to_d01_signature_contract`` と同じ実行時契約
    (``ctx`` が keyword-only であること等) を満たすことを固定する。
    ``_ParameterizedDummyDiagnostic`` はインスタンス (関数ではない) なので
    ``_iter_diagnostic_callables`` の列挙 (``inspect.isfunction`` で絞り込む) には
    乗らず、上のテストの対象には自動では入らない。そのためこのテストを別に置く。
    """
    diagnostic: Diagnostic = _ParameterizedDummyDiagnostic(threshold=0.9)

    signature = inspect.signature(diagnostic)
    params = signature.parameters
    assert "X" in params
    assert "u" in params
    assert "y" in params
    assert "ctx" in params
    ctx_param = params["ctx"]
    assert ctx_param.kind == inspect.Parameter.KEYWORD_ONLY, (
        f"ctx が keyword-only ではありません (D-01 違反, kind={ctx_param.kind})"
    )
    assert ctx_param.default is None

    states = _external_states()

    # 最も鋭い検査: ctx を位置引数として渡すと実際に TypeError になること
    # (`*,` keyword-only マーカーが外れた瞬間にこの呼び出しが成功してしまう)。
    with pytest.raises(TypeError):
        diagnostic(states, None, None, DiagnosticContext())  # type: ignore[misc]

    # 契約が許す全呼び出しパターンが実際に呼べる。
    result = diagnostic(states)
    assert isinstance(result, DiagnosticResult)
    assert result.scalars["threshold"] == pytest.approx(0.9)
    result_with_ctx = diagnostic(states, ctx=DiagnosticContext(washout=10))
    assert result_with_ctx.params["washout"] == "10"


def test_dummy_diagnostic_conforms_to_protocol() -> None:
    """ダミー実装が Diagnostic に代入でき、X だけで呼べる。

    下の代入は mypy strict でも検査される (``make type``)。署名を変えると
    ここで型エラーになる。D-01 の実行時契約は
    ``test_all_diagnostics_conform_to_d01_signature_contract`` (D-01 の guard_test)
    が全診断に対して検査するため、このテストは mypy 側のカバレッジに限定する。
    """
    d: Diagnostic = state_mean_norm
    result = d(_external_states())
    assert isinstance(result, DiagnosticResult)
    assert result.name == "state_mean_norm"
    assert "mean_norm" in result.scalars


def test_state_pca_conforms_to_protocol() -> None:
    d: Diagnostic = state_pca
    result = d(_external_states())
    assert isinstance(result, DiagnosticResult)
    assert "n_components_95" in result.scalars


def test_diagnostic_accepts_external_state_series() -> None:
    """ESN を import せずに作った状態系列で両診断が動く (移植性の担保)。"""
    states = _external_states()
    ctx = DiagnosticContext(washout=50)
    for diagnostic in (state_mean_norm, state_pca):
        result = diagnostic(states, None, None, ctx=ctx)
        assert result.params["washout"] == "50"
        assert all(np.isfinite(value) for value in result.scalars.values())


def test_diagnostics_package_does_not_import_reservoir() -> None:
    """diagnostics 配下のどのモジュールも reservoir を import しない。"""
    package_dir = Path(diagnostics_pkg.__file__).parent
    sources = sorted(package_dir.glob("*.py"))
    assert sources, "診断モジュールが見つかりません"
    for source in sources:
        text = source.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(text, filename=str(source))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith(FORBIDDEN_MODULE), (
                    f"{source.name} が {name} を import しています"
                )
        # 遅延 import や importlib 経由の抜け道も塞ぐ (docstring の言及は許す)
        assert f"import {FORBIDDEN_MODULE}" not in text
        assert f'"{FORBIDDEN_MODULE}"' not in text


def test_diagnostics_package_does_not_transitively_import_reservoir() -> None:
    """diagnostics を import しても reservoir が sys.modules に載らない (推移閉包)。

    上の ``test_diagnostics_package_does_not_import_reservoir`` は diagnostics
    配下の *.py の AST を直接 import 検査するだけなので、
    ``diagnostics/x.py`` が ``config.py`` 経由で間接的に ``reservoir`` を
    引き込んでも検出できない (config.py は ``ESNConfig`` を import している)。
    このテストは実際に別プロセスで ``rc_basics_lab.diagnostics`` パッケージ本体に
    加えて ``pkgutil.iter_modules`` で列挙した**全サブモジュール**を個別に import し、
    その後の ``sys.modules`` に ``rc_basics_lab.reservoir`` で始まるモジュールが
    1つも無いことを assert することで、直接 import ではなく import の**結果**を
    検証する。パッケージ本体だけの import では ``diagnostics/__init__.py`` が
    再エクスポートしていないサブモジュール (``__all__`` に載せ忘れたもの) が
    検査対象から漏れるため、サブモジュールを個別に import する形にしている。
    tasks/ 側の前例 (自分の設定 dataclass を config.py から import する) を
    diagnostics/ が真似た瞬間にここが赤くなる。

    このガードが守るのは「モジュール import 時点の推移閉包」であり、関数
    ローカルの遅延 import (関数本体の中に書かれ、呼び出されるまで実行されない
    import 文) は import 時点の ``sys.modules`` に出ないため、この guard では
    検出できない。
    """
    probe = textwrap.dedent("""
        import importlib
        import json
        import pkgutil
        import sys

        pkg = importlib.import_module("rc_basics_lab.diagnostics")
        for info in pkgutil.iter_modules(pkg.__path__):
            importlib.import_module(f"rc_basics_lab.diagnostics.{info.name}")

        leaked = sorted(
            name for name in sys.modules if name.startswith("rc_basics_lab.reservoir")
        )
        print("LEAKED=" + json.dumps(leaked))
        """)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    leaked = _extract_leaked(completed.stdout)
    assert leaked == [], (
        "rc_basics_lab.diagnostics の import が rc_basics_lab.reservoir を "
        f"引き込んでいます (推移的依存): {leaked}"
    )


def test_context_has_all_defaults() -> None:
    """DiagnosticContext は引数なしで構築できる (拡張は既定値つきのみ)。"""
    ctx = DiagnosticContext()
    assert ctx.washout == 0
    assert ctx.dt == pytest.approx(1.0)
    assert ctx.seed is None
    assert ctx.companion_states == ()
    assert ctx.propagator is None


def test_companion_states_carry_second_trajectory() -> None:
    """02 の ESP 用の第2軌道が署名変更なしで渡せる。"""
    states = _external_states()
    ctx = DiagnosticContext(companion_states=(states + 1.0,))
    validate_diagnostic_input(states, None, None, ctx)
    assert len(ctx.companion_states) == 1


def test_row_from_result_is_flat() -> None:
    result = DiagnosticResult(
        name="demo", scalars={"a": 1.0}, arrays={"z": np.zeros(3)}, params={"p": "x"}
    )
    row = result.to_row()
    assert row == {"diagnostic": "demo", "p": "x", "a": 1.0}


def test_row_rejects_key_collision() -> None:
    result = DiagnosticResult(name="demo", scalars={"p": 1.0}, params={"p": "x"})
    with pytest.raises(ValueError, match="衝突"):
        result.to_row()


def test_validate_rejects_row_count_mismatch() -> None:
    """X.shape[0] != u.shape[0] で ValueError (受け入れ基準)。"""
    states = _external_states(n_steps=100)
    inputs = np.zeros((99, 1))
    with pytest.raises(ValueError, match="行数"):
        validate_diagnostic_input(states, inputs)


def test_validate_rejects_target_row_count_mismatch() -> None:
    states = _external_states(n_steps=100)
    with pytest.raises(ValueError, match="行数"):
        validate_diagnostic_input(states, None, np.zeros((50, 1)))


def test_validate_rejects_one_dimensional_state() -> None:
    with pytest.raises(ValueError, match="2次元"):
        validate_diagnostic_input(np.zeros(100))


def test_validate_rejects_washout_beyond_series() -> None:
    with pytest.raises(ValueError, match="washout"):
        validate_diagnostic_input(
            _external_states(n_steps=10), ctx=DiagnosticContext(washout=10)
        )


def test_validate_rejects_companion_shape_mismatch() -> None:
    states = _external_states(n_steps=50)
    ctx = DiagnosticContext(companion_states=(np.zeros((50, 3)),))
    with pytest.raises(ValueError, match="companion_states"):
        validate_diagnostic_input(states, ctx=ctx)


def test_validate_rejects_non_positive_dt() -> None:
    with pytest.raises(ValueError, match="dt"):
        validate_diagnostic_input(_external_states(), ctx=DiagnosticContext(dt=0.0))

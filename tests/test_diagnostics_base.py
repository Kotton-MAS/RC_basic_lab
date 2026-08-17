"""診断層インターフェースのテスト (D-01・受け入れ条件6)。"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import rc_basics_lab.diagnostics as diagnostics_pkg
from rc_basics_lab.diagnostics.base import (
    Diagnostic,
    DiagnosticContext,
    DiagnosticResult,
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


def _iter_diagnostic_callables() -> list[tuple[str, Callable[..., DiagnosticResult]]]:
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
    found: list[tuple[str, Callable[..., DiagnosticResult]]] = []
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


def test_diagnostic_enumeration_finds_all_known_diagnostics() -> None:
    """列挙条件が0件に壊れていないこと自体を固定する。

    列挙条件 (戻り値アノテーションが DiagnosticResult の public callable) を
    間違えて0件になると、下の契約テストは何も検査せずに緑になってしまう。
    現時点で存在する2本 (dummy.state_mean_norm / state_space.state_pca) が
    確実に拾えていることを固定する。
    """
    names = {qualname for qualname, _ in _iter_diagnostic_callables()}
    assert names, "diagnostics 配下から診断関数が1件も列挙されませんでした"
    assert "rc_basics_lab.diagnostics.dummy.state_mean_norm" in names
    assert "rc_basics_lab.diagnostics.state_space.state_pca" in names


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
        with pytest.raises(TypeError):
            func(states, None, None, DiagnosticContext())

        # 契約が許す全呼び出しパターンが実際に呼べること。
        func(states)
        func(states, None)
        func(states, None, None)
        func(states, None, None, ctx=DiagnosticContext())
        func(states, ctx=DiagnosticContext())


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
    probe = (
        "import importlib, pkgutil, sys\n"
        "pkg = importlib.import_module('rc_basics_lab.diagnostics')\n"
        "for info in pkgutil.iter_modules(pkg.__path__):\n"
        "    importlib.import_module(f'rc_basics_lab.diagnostics.{info.name}')\n"
        "leaked = sorted(\n"
        "    k for k in sys.modules if k.startswith('rc_basics_lab.reservoir')\n"
        ")\n"
        "print(','.join(leaked))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    leaked = [name for name in completed.stdout.strip().split(",") if name]
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

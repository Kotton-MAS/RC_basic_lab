"""診断層インターフェースのテスト (D-01・受け入れ条件6)。"""

from __future__ import annotations

import ast
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


def test_dummy_diagnostic_conforms_to_protocol() -> None:
    """ダミー実装が Diagnostic に代入でき、X だけで呼べる (D-01 の guard test)。

    下の代入は mypy strict でも検査される (``make type``)。署名を変えると
    ここで型エラーになる。
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

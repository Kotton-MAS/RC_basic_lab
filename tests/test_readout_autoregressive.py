"""自由走行 (closed-loop) の検査 —— 移植性と係数の同一性 (D-44 / D-50).

要件書 受け入れ条件7 は「自走モジュールが**外部生成の状態系列生成器でも動く**」
である。Protocol を満たすだけでは動く証明にならないので、

1. **ESN を一切使わない**状態更新器 (解析的に閉じた線形写像) で自走を回し、
2. 予測列が**閉形式**と一致し、
3. その間 ``ESN`` のどのメソッドも呼ばれていない (呼べば即座に落ちる)

の3つをまとめて測る。3. が無いと「たまたま ESN を経由していただけ」を排除
できない。加えて ``readout/autoregressive.py`` が ``reservoir`` を推移的にも
引き込んでいないことを、AST とサブプロセスの両方で確かめる (D-50)。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ast_imports import imported_roots

import rc_basics_lab
from rc_basics_lab.readout.autoregressive import FreeRunResult, StateUpdater, free_run
from rc_basics_lab.readout.design import DelayLineSpec, ReservoirSpec
from rc_basics_lab.reservoir.esn import ESN
from rc_basics_lab.types import FloatArray

AUTOREGRESSIVE_SOURCE = (
    Path(rc_basics_lab.__file__).parent / "readout/autoregressive.py"
)
RESERVOIR_ROOT = "rc_basics_lab.reservoir"

# 解析的に閉じた線形の状態更新器のパラメータ。
#   x[k] = A * x[k-1] + B * u[k],  y[k] = C * x[k],  u[k+1] = y[k]
# を1次元で回すと x[k+1] = (A + B*C) * x[k] になるので、閉形式は等比数列である。
LINEAR_A = 0.5
LINEAR_B = 0.25
LINEAR_C = 0.8
LINEAR_X0 = 2.0
LINEAR_U0 = 1.0

LINEAR_SPEC = ReservoirSpec(include_input=False, bias=False)
"""特徴を ``[x]`` だけにする仕様 (閉形式を1行で書けるようにするため)。"""


def _linear_updater() -> tuple[StateUpdater, list[int]]:
    """ESN を使わない状態更新器と、呼ばれた回数を数えるカウンタ。"""
    calls: list[int] = []

    def update(x: FloatArray, u: FloatArray) -> FloatArray:
        calls.append(1)
        state: FloatArray = LINEAR_A * np.asarray(x, dtype=np.float64) + LINEAR_B * (
            np.asarray(u, dtype=np.float64)
        )
        return state

    return update, calls


def _closed_form(n_steps: int) -> FloatArray:
    """自走 ``n_steps`` ステップぶんの予測の閉形式。

    ``x[1] = A*x0 + B*u0``、``k >= 1`` で ``x[k+1] = (A + B*C) * x[k]`` なので
    ``y[k] = C * x[1] * (A + B*C)**(k-1)``。
    """
    ratio = LINEAR_A + LINEAR_B * LINEAR_C
    first_state = LINEAR_A * LINEAR_X0 + LINEAR_B * LINEAR_U0
    return np.array(
        [[LINEAR_C * first_state * ratio**index] for index in range(n_steps)],
        dtype=np.float64,
    )


def _forbid_esn(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ESN`` のどのメソッドを呼んでも即座に落ちるようにする。

    「外部生成器で動く」の実測を、Protocol への適合ではなく**ESN が実行経路に
    現れないこと**で取るための仕掛け。``free_run`` が内部でこっそり ESN を
    作る実装に変わればここで落ちる。
    """

    def _explode(*args: object, **kwargs: object) -> FloatArray:
        raise AssertionError("外部生成器での自走中に ESN が呼ばれました (D-50)")

    for name in ("__init__", "step", "run"):
        monkeypatch.setattr(ESN, name, _explode)


def test_free_run_works_with_an_external_state_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ESN を一切使わない状態更新器で自走が回り、閉形式と一致する (D-50)。

    要件書 受け入れ条件7 の実体。``ESN`` の ``__init__`` / ``step`` / ``run`` を
    「呼ばれたら落ちる」ものへ差し替えたうえで自走を完走させるので、
    「Protocol は満たすが内部で ESN を作っている」実装は緑にならない。
    """
    _forbid_esn(monkeypatch)
    updater, calls = _linear_updater()
    n_steps = 12
    result = free_run(
        updater,
        LINEAR_SPEC,
        np.array([[LINEAR_C]], dtype=np.float64),
        np.array([LINEAR_X0], dtype=np.float64),
        np.array([LINEAR_U0], dtype=np.float64),
        n_steps,
    )

    assert isinstance(result, FreeRunResult)
    assert not result.diverged
    assert result.n_completed == n_steps
    assert len(calls) == n_steps, "状態更新器が自走ステップ数だけ呼ばれていません"
    np.testing.assert_allclose(result.predictions, _closed_form(n_steps), rtol=1.0e-12)
    # 入力は「1つ前の予測」そのもの (出力を入力へ戻すことの実測)。
    np.testing.assert_allclose(result.inputs[0], [LINEAR_U0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.inputs[1:], result.predictions[:-1], rtol=1e-12)


def test_autoregressive_does_not_import_reservoir_at_module_level() -> None:
    """``readout/autoregressive.py`` に ``reservoir`` の import 文が無い (D-50)。

    AST を見るので、関数内 import へ逃がしても落ちる (D-53 の AST ガードと
    同じ流儀)。
    """
    imported = imported_roots(AUTOREGRESSIVE_SOURCE, include_function_bodies=True)
    offenders = sorted(
        name
        for name in imported
        if name == RESERVOIR_ROOT or name.startswith(f"{RESERVOIR_ROOT}.")
    )
    assert not offenders, (
        f"autoregressive が reservoir を import しています: {offenders}"
    )


def test_importing_autoregressive_does_not_pull_in_reservoir() -> None:
    """単独 import しても ``reservoir`` が ``sys.modules`` に現れない (D-50)。

    AST 検査だけだと**推移的な**引き込み (``readout.design`` 経由など) を
    見逃す。別プロセスで測るのは、同一プロセスでは他のテストが import 済みの
    ``rc_basics_lab.reservoir`` が残っており、検査が空虚になるため
    (``tests/test_layer_boundaries.py`` と同じ理由)。
    """
    script = (
        "import sys\n"
        "import rc_basics_lab.readout.autoregressive\n"
        "leaked = [name for name in sys.modules "
        f"if name == {RESERVOIR_ROOT!r} or name.startswith({RESERVOIR_ROOT + '.'!r})]\n"
        "assert not leaked, f'reservoir を推移的に引き込んでいます: {leaked}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_free_run_returns_the_coefficients_object_it_was_given() -> None:
    """``FreeRunResult.coefficients`` が渡された配列そのもの (D-44 の同一性)。

    自走の内部で学習し直す実装 (仕様 §5 禁止する構造1) は、新しい配列を返す
    ことになるのでここで落ちる。
    """
    updater, _ = _linear_updater()
    coefficients = np.array([[LINEAR_C]], dtype=np.float64)
    result = free_run(
        updater,
        LINEAR_SPEC,
        coefficients,
        np.array([LINEAR_X0]),
        np.array([LINEAR_U0]),
        5,
    )
    assert result.coefficients is coefficients


def test_free_run_rejects_a_spec_that_needs_lag_history() -> None:
    """ラグ履歴を要する仕様は閉ループに乗らないので ``ValueError``。"""
    updater, _ = _linear_updater()
    with pytest.raises(ValueError, match="first_valid"):
        free_run(
            updater,
            DelayLineSpec(n_lags=3),
            np.zeros((5, 1)),
            np.array([LINEAR_X0]),
            np.array([LINEAR_U0]),
            5,
        )


def test_free_run_rejects_mismatched_input_and_output_dimensions() -> None:
    """``D_out != D_in`` は ``ValueError`` (出力をそのまま入力へ戻せない)。"""
    updater, _ = _linear_updater()
    with pytest.raises(ValueError, match="出力次元が入力次元"):
        free_run(
            updater,
            LINEAR_SPEC,
            np.zeros((1, 2)),  # D_out=2
            np.array([LINEAR_X0]),
            np.array([LINEAR_U0]),  # D_in=1
            5,
        )


def test_free_run_rejects_a_spec_whose_feature_count_differs() -> None:
    """学習時と違う特徴数の係数は ``ValueError`` (静かに壊れた予測を出さない)。"""
    updater, _ = _linear_updater()
    with pytest.raises(ValueError, match="設計行列の特徴数"):
        free_run(
            updater,
            ReservoirSpec(),  # [1, u, x] で F=3
            np.array([[LINEAR_C]]),  # F=1
            np.array([LINEAR_X0]),
            np.array([LINEAR_U0]),
            3,
        )


def test_free_run_records_divergence_without_hiding_it() -> None:
    """発散したら ``diverged`` を立て、残りの行は 0 ではなく ``nan`` にする。

    発散は自走の結果の1つ (4-C の3態の1態) なので例外にしない。ただし
    0 埋めにすると「静かに真値へ近い予測」に化けるので、埋めない。
    """

    def exploding(x: FloatArray, u: FloatArray) -> FloatArray:
        with np.errstate(over="ignore"):
            blown: FloatArray = np.asarray(x, dtype=np.float64) * 1.0e300
        return blown

    result = free_run(
        exploding,
        LINEAR_SPEC,
        np.array([[1.0]]),
        np.array([1.0e300]),
        np.array([1.0]),
        10,
    )
    assert result.diverged
    assert 0 <= result.n_completed < 10
    assert np.all(np.isnan(result.predictions[result.n_completed :]))
    assert np.all(np.isnan(result.states[result.n_completed :]))


def test_free_run_rejects_a_zero_length_run() -> None:
    """``n_steps < 1`` は ``ValueError`` (空の自走結果を作らない)。"""
    updater, _ = _linear_updater()
    with pytest.raises(ValueError, match="n_steps"):
        free_run(
            updater,
            LINEAR_SPEC,
            np.array([[LINEAR_C]]),
            np.array([LINEAR_X0]),
            np.array([LINEAR_U0]),
            0,
        )


def test_free_run_rejects_an_updater_that_changes_the_state_shape() -> None:
    """状態の形状を変える更新器は ``ValueError`` (形状の食い違いを黙認しない)。"""

    def reshaping(x: FloatArray, u: FloatArray) -> FloatArray:
        widened: FloatArray = np.zeros(len(x) + 1, dtype=np.float64)
        return widened

    with pytest.raises(ValueError, match="updater の戻り値の形状"):
        free_run(
            reshaping,
            LINEAR_SPEC,
            np.array([[LINEAR_C]]),
            np.array([LINEAR_X0]),
            np.array([LINEAR_U0]),
            3,
        )

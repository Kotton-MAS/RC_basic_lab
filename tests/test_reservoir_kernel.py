"""リザバーが共有する更新式と検査 (``reservoir/_kernel.py``) の検査.

**この層の存在理由は「3 箇所に同じコードを置かない」ことである。** したがって
測るべきは (a) 更新式が仕様どおりであること と (b) **3 モデルが実際に同じ
実装を通っていること** の 2 つで、後者を落とすと共有した意味が消える。
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from rc_basics_lab.reservoir._kernel import (
    NOISE_WITHOUT_RNG,
    check_common_config,
    check_input_series,
    check_state,
    leaky_tanh_update,
)
from rc_basics_lab.types import FloatArray

RESERVOIR_DIR = Path("src/rc_basics_lab/reservoir")
MODELS = ("esn.py", "ring.py", "deep.py")


# --- 更新式 ------------------------------------------------------------------


def test_the_leak_rate_interpolates_between_old_and_new() -> None:
    """``a = 1`` なら過去を持ち越さず、``a`` が小さいほど過去が残る。"""
    state: FloatArray = np.full(3, 0.5, dtype=np.float64)
    drive: FloatArray = np.zeros(3, dtype=np.float64)
    recurrent: FloatArray = np.zeros((3, 3), dtype=np.float64)
    full = leaky_tanh_update(
        state, drive, recurrent, leak_rate=1.0, state_noise=0.0, rng=None
    )
    half = leaky_tanh_update(
        state, drive, recurrent, leak_rate=0.5, state_noise=0.0, rng=None
    )
    assert full == pytest.approx(np.tanh(drive))
    assert half == pytest.approx(0.5 * state + 0.5 * np.tanh(drive))


def test_the_state_stays_bounded_by_one() -> None:
    """状態は ``|x| <= 1`` に留まる (診断層が前提にしている有界性)。

    雑音を**活性化の内側**に足しているからで、外側に足すと壊れる。
    """
    rng = np.random.default_rng(0)
    state: FloatArray = np.zeros(16, dtype=np.float64)
    recurrent: FloatArray = rng.uniform(-1.0, 1.0, (16, 16))
    for _ in range(200):
        drive: FloatArray = rng.uniform(-5.0, 5.0, 16)
        state = leaky_tanh_update(
            state, drive, recurrent, leak_rate=0.5, state_noise=1.0, rng=rng
        )
    assert np.all(np.abs(state) <= 1.0), f"有界性が壊れています: {np.abs(state).max()}"


def test_noise_without_a_generator_is_rejected() -> None:
    """``state_noise > 0`` で ``rng`` が無ければ落ちる (D-36)。"""
    state: FloatArray = np.zeros(2, dtype=np.float64)
    with pytest.raises(ValueError, match="rng が必要"):
        leaky_tanh_update(
            state,
            state,
            np.zeros((2, 2), dtype=np.float64),
            leak_rate=1.0,
            state_noise=0.1,
            rng=None,
        )


def test_zero_noise_draws_nothing_from_the_generator() -> None:
    """``state_noise == 0`` では乱数を1個も引かない。

    引くと、雑音を使わない条件の結果がリザバーの乱数列の位置に依存して変わる
    (既存の成果物が動く)。
    """
    rng = np.random.default_rng(3)
    before = rng.bit_generator.state
    leaky_tanh_update(
        np.zeros(4, dtype=np.float64),
        np.zeros(4, dtype=np.float64),
        np.zeros((4, 4), dtype=np.float64),
        leak_rate=0.5,
        state_noise=0.0,
        rng=rng,
    )
    assert rng.bit_generator.state == before, "乱数を引いています"


# --- 検査 --------------------------------------------------------------------


def test_a_wrong_state_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="状態は"):
        check_state(np.zeros(3, dtype=np.float64), 4)


@pytest.mark.parametrize(
    ("array", "n_inputs", "match"),
    [
        (np.zeros(5), 1, "2次元"),
        (np.zeros((5, 2)), 1, "入力次元"),
        (np.zeros((0, 1)), 1, "空"),
    ],
)
def test_a_malformed_input_series_is_rejected(
    array: FloatArray, n_inputs: int, match: str
) -> None:
    """1次元を ``(T, 1)`` と黙って解釈しない (多次元の取り違えを通さない)。"""
    with pytest.raises(ValueError, match=match):
        check_input_series(array, n_inputs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_units": 0}, "n_units"),
        ({"leak_rate": 0.0}, "leak_rate"),
        ({"leak_rate": 1.5}, "leak_rate"),
        ({"spectral_radius": 0.0}, "spectral_radius"),
        ({"scales": {"input_scale": -1.0}}, "input_scale"),
        ({"state_noise": -0.1}, "state_noise"),
        ({"n_inputs": 0}, "n_inputs"),
    ],
)
def test_out_of_range_common_settings_are_rejected(
    kwargs: dict[str, object], match: str
) -> None:
    """共通の設定値の範囲検査。**3 モデルが同じ文言で落ちる**ことの実体。"""
    base: dict[str, object] = {
        "n_units": 8,
        "min_units": 1,
        "leak_rate": 0.5,
        "spectral_radius": 0.9,
        "scales": {"input_scale": 0.5},
        "state_noise": 0.0,
        "n_inputs": 1,
    }
    with pytest.raises(ValueError, match=match):
        check_common_config(**{**base, **kwargs})  # type: ignore[arg-type]


# --- 共有されていることの検査 ------------------------------------------------


@pytest.mark.parametrize("filename", MODELS)
def test_no_model_reimplements_the_update(filename: str) -> None:
    """モデル側に更新式を書き戻していない。

    ``(1.0 - leak) * x + leak * tanh(...)`` の形が再び現れたら、共有した意味が
    消えている (直した不具合が1つのモデルにだけ残る)。
    """
    text = (RESERVOIR_DIR / filename).read_text(encoding="utf-8")
    assert "1.0 - leak" not in text.replace("1.0 - leak_rate", ""), (
        f"{filename} が更新式を自前で持っています (_kernel を使ってください)"
    )


@pytest.mark.parametrize("filename", MODELS)
def test_every_model_uses_the_shared_kernel(filename: str) -> None:
    """全モデルが ``_kernel`` を import している (**空振り防止**)。

    上の検査は「書いていない」ことしか見ないので、そもそも使っていないモデルが
    あっても緑になる。使っていることを別に測る。
    """
    tree = ast.parse((RESERVOIR_DIR / filename).read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "rc_basics_lab.reservoir._kernel" in imported, (
        f"{filename} が _kernel を使っていません"
    )


@pytest.mark.parametrize("filename", MODELS)
def test_no_model_repeats_the_noise_message(filename: str) -> None:
    """雑音の文言をモデル側に写経していない (``NOISE_WITHOUT_RNG`` に一本化)。"""
    text = (RESERVOIR_DIR / filename).read_text(encoding="utf-8")
    assert "state_noise > 0 のときは rng が必要です" not in text, (
        f"{filename} が文言を写経しています ({NOISE_WITHOUT_RNG[:20]}…)"
    )

"""リング結合リザバー (SCR) の検査.

**文献の主張が実際に成り立つことまで測る。** 「動く」だけなら 0 を返す実装でも
緑になる。SCR (Rodan & Tino 2011) の核は次の2つで、どちらも構造から検証できる:

1. 閉路行列のスペクトル半径が**厳密に** ``|r|``
2. 入力重みの**大きさが全部同じ**
"""

from __future__ import annotations

import numpy as np
import pytest

from rc_basics_lab.reservoir.esn import spectral_radius
from rc_basics_lab.reservoir.protocol import Reservoir
from rc_basics_lab.reservoir.ring import RingConfig, RingReservoir, cycle_matrix


@pytest.mark.parametrize("n_units", [2, 5, 64])
@pytest.mark.parametrize("weight", [0.5, 0.9, 1.3])
def test_the_cycle_has_exactly_the_requested_spectral_radius(
    n_units: int, weight: float
) -> None:
    """閉路のスペクトル半径は厳密に ``|r|``。

    **ESN と違って測って割り直さない**ので、ここがずれると
    ``spectral_radius`` の設定が別の量を指すことになる。
    """
    assert spectral_radius(cycle_matrix(n_units, weight)) == pytest.approx(
        weight, rel=1e-12
    )


def test_the_cycle_is_a_single_unidirectional_loop() -> None:
    """非零は各行1つだけで、閉路を1周する。"""
    matrix = cycle_matrix(5, 0.9)
    assert np.count_nonzero(matrix) == 5
    for row in range(5):
        assert matrix[row, row - 1] != 0.0, f"{row} 行の結合先が違います"


def test_the_input_weights_share_one_magnitude() -> None:
    """入力重みの大きさが全部同じ (**SCR の主張の核**)。

    符号だけが ``rng`` で変わる。ここが崩れると「ランダム性をほとんど使わない」
    という対照の意味が消える。
    """
    config = RingConfig(n_units=32, input_scale=0.4, bias_scale=0.15)
    reservoir = RingReservoir(config, np.random.default_rng(3), n_inputs=2)
    assert reservoir.run(np.zeros((3, 2))).shape == (3, 32)
    weights = reservoir.W_in
    assert np.allclose(np.abs(weights[:, 0]), config.bias_scale)
    assert np.allclose(np.abs(weights[:, 1:]), config.input_scale)
    assert set(np.unique(np.sign(weights[:, 1:]))) == {-1.0, 1.0}, (
        "符号が振れていません (rng を使っていない可能性)"
    )


def test_the_same_seed_reproduces_the_weights() -> None:
    """同一シードで状態がバイト一致する (D-06)。"""
    config = RingConfig(n_units=16)
    signal = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    first = RingReservoir(config, np.random.default_rng(11)).run(signal)
    second = RingReservoir(config, np.random.default_rng(11)).run(signal)
    assert first.tobytes() == second.tobytes()


def test_a_different_seed_changes_the_states() -> None:
    """シードが違えば状態が変わる (符号が実際に効いている)。"""
    config = RingConfig(n_units=16)
    signal = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    first = RingReservoir(config, np.random.default_rng(11)).run(signal)
    other = RingReservoir(config, np.random.default_rng(12)).run(signal)
    assert not np.array_equal(first, other)


def test_it_satisfies_the_reservoir_protocol() -> None:
    """接合面を満たす。"""
    reservoir = RingReservoir(RingConfig(n_units=8), np.random.default_rng(0))
    assert isinstance(reservoir, Reservoir)
    assert (reservoir.n_units, reservoir.n_inputs) == (8, 1)


def test_step_and_run_agree() -> None:
    """``step`` を回した結果と ``run`` が一致する (自走と教師強制で同じ更新式)。"""
    config = RingConfig(n_units=12)
    signal = np.linspace(-0.5, 0.5, 7).reshape(-1, 1)
    reservoir = RingReservoir(config, np.random.default_rng(5))
    by_run = reservoir.run(signal)
    state = reservoir.initial_state()
    for index in range(signal.shape[0]):
        state = reservoir.step(state, signal[index])
        assert state == pytest.approx(by_run[index])


def test_the_state_noise_needs_a_generator() -> None:
    """``state_noise > 0`` で ``rng`` が無ければ落ちる (D-36)。"""
    reservoir = RingReservoir(
        RingConfig(n_units=8, state_noise=0.1), np.random.default_rng(0)
    )
    with pytest.raises(ValueError, match="rng が必要"):
        reservoir.run(np.zeros((3, 1)))


@pytest.mark.parametrize(
    "config",
    [
        RingConfig(n_units=1),
        RingConfig(leak_rate=0.0),
        RingConfig(leak_rate=1.5),
        RingConfig(spectral_radius=0.0),
        RingConfig(input_scale=-0.1),
        RingConfig(state_noise=-0.1),
    ],
)
def test_an_out_of_range_setting_is_rejected(config: RingConfig) -> None:
    """設定値の範囲検査 (``ESN`` と同じ規律)。"""
    with pytest.raises(ValueError):
        RingReservoir(config, np.random.default_rng(0))

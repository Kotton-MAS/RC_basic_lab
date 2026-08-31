"""深層 ESN の検査.

**文献の主張が実際に出ることまで測る。** Gallicchio & Micheli (2017) の核は
「層を積むと層ごとに違う時間スケールが自然に現れる (深い層ほど遅い)」で、
リーク率を層ごとに変えなくてもそうなる、という点にある。自己相関の減衰で測る。

構造の側では「連結後の次元が ``n_units`` である」ことが重要である。読み出し層も
診断層も ``n_units`` を状態の次元として読み、容量の上限 ``MC <= N`` もその N を
指すため、ここがずれると下流の主張が全部ずれる。
"""

from __future__ import annotations

import numpy as np
import pytest

from rc_basics_lab.reservoir.deep import DeepESN, DeepESNConfig
from rc_basics_lab.reservoir.protocol import Reservoir
from rc_basics_lab.types import FloatArray


def _drive(n_steps: int, seed: int = 0) -> FloatArray:
    drawn: FloatArray = np.random.default_rng(seed).uniform(-1.0, 1.0, (n_steps, 1))
    return drawn


def test_the_state_dimension_is_the_declared_total() -> None:
    """``run`` が返す列数が ``n_units`` と一致する (**連結後の総次元**)。"""
    config = DeepESNConfig(n_units=60, n_layers=3, density=0.3)
    reservoir = DeepESN(config, np.random.default_rng(0))
    assert reservoir.n_units == 60
    assert reservoir.run(_drive(12)).shape == (12, 60)


def test_the_layers_partition_the_state() -> None:
    """層の切り出しが重ならず、全部で状態を覆う。"""
    config = DeepESNConfig(n_units=60, n_layers=3, density=0.3)
    reservoir = DeepESN(config, np.random.default_rng(0))
    covered: list[int] = []
    for layer in range(reservoir.n_layers):
        covered += list(range(*reservoir.layer_slice(layer).indices(60)))
    assert sorted(covered) == list(range(60))


def test_an_out_of_range_layer_is_rejected() -> None:
    """層の番号が範囲外なら落ちる。"""
    reservoir = DeepESN(
        DeepESNConfig(n_units=20, n_layers=2, density=0.3), np.random.default_rng(0)
    )
    with pytest.raises(IndexError):
        reservoir.layer_slice(2)


def test_a_single_layer_matches_the_state_dimension_of_an_esn() -> None:
    """``n_layers=1`` は単層と同じ形になる (積む前の基準点)。"""
    reservoir = DeepESN(
        DeepESNConfig(n_units=40, n_layers=1, density=0.2), np.random.default_rng(0)
    )
    assert reservoir.run(_drive(8)).shape == (8, 40)


def test_deeper_layers_decay_more_slowly() -> None:
    """**深い層ほど時間スケールが遅い** (この構成の主張そのもの)。

    リーク率は全層で同じなので、差が出るなら「積んだこと」から出ている。
    ラグ1の自己相関で測る (大きいほど遅い)。
    """
    config = DeepESNConfig(
        n_units=180, n_layers=3, density=0.2, leak_rate=0.3, spectral_radius=0.9
    )
    reservoir = DeepESN(config, np.random.default_rng(7))
    states = reservoir.run(_drive(3000, seed=1))[500:]  # 過渡を捨てる

    lag1: list[float] = []
    for layer in range(reservoir.n_layers):
        block = states[:, reservoir.layer_slice(layer)]
        centered = block - block.mean(axis=0)
        variance = np.sum(centered[:-1] * centered[:-1], axis=0)
        covariance = np.sum(centered[:-1] * centered[1:], axis=0)
        lag1.append(float(np.mean(covariance / variance)))

    assert lag1 == sorted(lag1), f"深い層ほど遅くなっていません: {lag1}"
    assert lag1[-1] - lag1[0] > 0.01, f"層による差がほとんどありません: {lag1}"


def test_the_same_seed_reproduces_the_states() -> None:
    """同一シードで状態がバイト一致する (D-06)。"""
    config = DeepESNConfig(n_units=40, n_layers=2, density=0.3)
    signal = _drive(15)
    first = DeepESN(config, np.random.default_rng(4)).run(signal)
    second = DeepESN(config, np.random.default_rng(4)).run(signal)
    assert first.tobytes() == second.tobytes()


def test_adding_a_layer_changes_the_states() -> None:
    """層の数が効く (D-13: 効かない設定は飾りである)。"""
    signal = _drive(15)
    shallow = DeepESN(
        DeepESNConfig(n_units=40, n_layers=1, density=0.3), np.random.default_rng(4)
    ).run(signal)
    deep = DeepESN(
        DeepESNConfig(n_units=40, n_layers=2, density=0.3), np.random.default_rng(4)
    ).run(signal)
    assert not np.array_equal(shallow, deep)


def test_step_and_run_agree() -> None:
    """``step`` を回した結果と ``run`` が一致する。"""
    config = DeepESNConfig(n_units=30, n_layers=3, density=0.4)
    signal = _drive(9)
    reservoir = DeepESN(config, np.random.default_rng(2))
    by_run = reservoir.run(signal)
    state = reservoir.initial_state()
    for index in range(signal.shape[0]):
        state = reservoir.step(state, signal[index])
        assert state == pytest.approx(by_run[index])


def test_it_satisfies_the_reservoir_protocol() -> None:
    """接合面を満たす。"""
    reservoir = DeepESN(
        DeepESNConfig(n_units=20, n_layers=2, density=0.3), np.random.default_rng(0)
    )
    assert isinstance(reservoir, Reservoir)


def test_a_non_divisible_total_is_rejected() -> None:
    """``n_units`` が ``n_layers`` で割り切れなければ落ちる。

    暗黙に丸めると、設定した総次元と実際の状態の次元がずれる。
    """
    with pytest.raises(ValueError, match="割り切れる"):
        DeepESN(
            DeepESNConfig(n_units=50, n_layers=3, density=0.3),
            np.random.default_rng(0),
        )


def test_a_layer_too_sparse_to_recur_is_rejected() -> None:
    """``density * 層あたりのユニット数`` が 1 未満なら落ちる。

    この条件では再帰の無い W (冪零) が**シード次第で**生まれる。通ったり
    落ちたりする設定は最悪なので、条件そのものを先に落とす。
    """
    with pytest.raises(ValueError, match="density \\* "):
        DeepESN(
            DeepESNConfig(n_units=12, n_layers=3, density=0.1),
            np.random.default_rng(0),
        )


def test_the_state_noise_needs_a_generator() -> None:
    """``state_noise > 0`` で ``rng`` が無ければ落ちる (D-36)。"""
    reservoir = DeepESN(
        DeepESNConfig(n_units=20, n_layers=2, density=0.3, state_noise=0.1),
        np.random.default_rng(0),
    )
    with pytest.raises(ValueError, match="rng が必要"):
        reservoir.run(_drive(4))

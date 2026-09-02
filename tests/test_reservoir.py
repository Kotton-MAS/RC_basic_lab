"""ESN コアのテスト.

「値を変えたら出力が変わる」系 (``leak_rate`` / ``x0`` / ``state_noise``) は、
閾値を緩めて通すのではなく、実際に大きな差が出ることを確認する。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from rc_basics_lab.reservoir.esn import ESN, ESNConfig, spectral_radius
from rc_basics_lab.reservoir.topology import ErdosRenyiConfig
from rc_basics_lab.types import FloatArray


def _inputs(n_steps: int = 500, n_inputs: int = 1, seed: int = 0) -> FloatArray:
    return np.random.default_rng(seed).standard_normal((n_steps, n_inputs))


def _lag1_autocorrelation(states: FloatArray) -> float:
    """ユニット平均のラグ1自己相関。"""
    centered = states - states.mean(axis=0, keepdims=True)
    numerator = np.sum(centered[1:] * centered[:-1], axis=0)
    denominator = np.sum(centered * centered, axis=0)
    return float(np.mean(numerator / denominator))


def test_spectral_radius_matches_config() -> None:
    """生成した W の実測スペクトル半径が設定値に一致する (rtol 1e-10)。"""
    for target in (0.5, 0.9, 1.2):
        config = ESNConfig(n_units=120, spectral_radius=target)
        esn = ESN(config, np.random.default_rng(11))
        assert spectral_radius(esn.W) == pytest.approx(target, rel=1e-10)


def test_spectral_radius_is_deterministic() -> None:
    """同一シードで2回生成した重みがバイト一致する。"""
    config = ESNConfig(n_units=80)
    first = ESN(config, np.random.default_rng(5))
    second = ESN(config, np.random.default_rng(5))
    assert first.W.tobytes() == second.W.tobytes()
    assert first.W_in.tobytes() == second.W_in.tobytes()


def test_different_seed_changes_weights() -> None:
    config = ESNConfig(n_units=80)
    first = ESN(config, np.random.default_rng(5))
    other = ESN(config, np.random.default_rng(6))
    assert first.W.tobytes() != other.W.tobytes()


def test_weights_are_read_only() -> None:
    """重みは生成後に書き換えられない (学習は読み出し層だけで行う)。"""
    esn = ESN(ESNConfig(n_units=20), np.random.default_rng(0))
    with pytest.raises(ValueError, match="read-only"):
        esn.W[0, 0] = 1.0


def test_density_controls_sparsity() -> None:
    """density を下げると W の非零率が下がる。"""
    sparse = ESN(
        ESNConfig(n_units=200, topology=ErdosRenyiConfig(density=0.02)),
        np.random.default_rng(1),
    )
    dense = ESN(
        ESNConfig(n_units=200, topology=ErdosRenyiConfig(density=0.5)),
        np.random.default_rng(1),
    )
    assert float(np.mean(sparse.W != 0.0)) == pytest.approx(0.02, abs=0.01)
    assert float(np.mean(dense.W != 0.0)) == pytest.approx(0.5, abs=0.02)


def test_shapes_of_weights_and_states() -> None:
    esn = ESN(ESNConfig(n_units=30), np.random.default_rng(0), n_inputs=3)
    assert esn.W_in.shape == (30, 4)
    assert esn.W.shape == (30, 30)
    states = esn.run(_inputs(n_steps=50, n_inputs=3))
    assert states.shape == (50, 30)


def test_leak_rate_changes_state_autocorrelation() -> None:
    """leak 0.1 と 1.0 で状態のラグ1自己相関が明確に異なる。"""
    inputs = _inputs(n_steps=1000)
    slow = ESN(ESNConfig(n_units=100, leak_rate=0.1), np.random.default_rng(7))
    fast = ESN(ESNConfig(n_units=100, leak_rate=1.0), np.random.default_rng(7))
    slow_autocorrelation = _lag1_autocorrelation(slow.run(inputs))
    fast_autocorrelation = _lag1_autocorrelation(fast.run(inputs))
    # 実測: 0.93 対 0.07。閾値は実測値から十分離した位置に置く。
    assert slow_autocorrelation > 0.8
    assert fast_autocorrelation < 0.3
    assert slow_autocorrelation - fast_autocorrelation > 0.5


def test_x0_and_state_noise_are_wired() -> None:
    """02 (2初期状態) と 04 (ノイズ注入) 用の API が実際に効いている。"""
    inputs = _inputs(n_steps=200)
    esn = ESN(ESNConfig(n_units=60), np.random.default_rng(3))
    from_zero = esn.run(inputs)
    from_other = esn.run(inputs, x0=np.full(60, 0.5))
    # 初期は明確に異なる (実測 0.56)
    assert float(np.max(np.abs(from_zero[0] - from_other[0]))) > 0.1
    # rho=0.9 の減衰系なので十分先では一致する (ESP の予兆。02 で本格的に測る)
    assert float(np.max(np.abs(from_zero[-1] - from_other[-1]))) < 1e-6

    noisy_config = ESNConfig(n_units=60, state_noise=0.05)
    noisy = ESN(noisy_config, np.random.default_rng(3))
    first = noisy.run(inputs, rng=np.random.default_rng(100))
    second = noisy.run(inputs, rng=np.random.default_rng(101))
    assert not np.allclose(first, second)
    # 同じシードなら再現する
    repeated = noisy.run(inputs, rng=np.random.default_rng(100))
    assert np.array_equal(first, repeated)


def test_state_noise_without_rng_raises() -> None:
    """ノイズ設定が黙って無効化されない (設定したのに効いていない、を防ぐ)。"""
    # N=10 / density=0.1 は期待辺数が 10 本しかなく、乱数次第で零行列に
    # なる (D-134 で引き順を直したら実際に引いた)。構造の検査には
    # 大きさは要らないので、退化しない規模にする。
    esn = ESN(ESNConfig(n_units=30, state_noise=0.1), np.random.default_rng(0))
    with pytest.raises(ValueError, match="rng"):
        esn.run(_inputs(n_steps=10))


def test_run_equals_repeated_step() -> None:
    """``run`` と ``step`` の逐次呼び出しがビット一致する (04 の閉ループ用)。"""
    inputs = _inputs(n_steps=50)
    esn = ESN(ESNConfig(n_units=40), np.random.default_rng(2))
    states = esn.run(inputs)
    manual = esn.initial_state()
    for index in range(inputs.shape[0]):
        manual = esn.step(manual, inputs[index])
        assert np.array_equal(manual, states[index])


def test_run_equals_repeated_step_with_noise() -> None:
    """ノイズ有りでも ``run`` == 逐次 ``step`` (同一 rng)。"""
    inputs = _inputs(n_steps=30)
    esn = ESN(ESNConfig(n_units=25, state_noise=0.02), np.random.default_rng(4))
    states = esn.run(inputs, rng=np.random.default_rng(9))
    step_rng = np.random.default_rng(9)
    manual = esn.initial_state()
    for index in range(inputs.shape[0]):
        manual = esn.step(manual, inputs[index], step_rng)
    assert np.array_equal(manual, states[-1])


def test_states_are_bounded_by_tanh() -> None:
    esn = ESN(ESNConfig(n_units=50, leak_rate=1.0), np.random.default_rng(0))
    states = esn.run(_inputs(n_steps=300))
    assert float(np.max(np.abs(states))) < 1.0


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (ESNConfig(n_units=0), "n_units"),
        (ESNConfig(n_units=10, activation="relu"), "活性化関数"),
        (ESNConfig(n_units=10, topology=ErdosRenyiConfig(density=0.0)), "density"),
        (ESNConfig(n_units=10, leak_rate=1.5), "leak_rate"),
        (ESNConfig(n_units=10, state_noise=-0.1), "state_noise"),
    ],
    ids=["n_units", "activation", "density", "leak_rate", "state_noise"],
)
def test_invalid_config_raises(config: ESNConfig, match: str) -> None:
    """不正な ESNConfig はそれぞれ該当パラメータ名を含む ValueError を送出する。"""
    with pytest.raises(ValueError, match=match):
        ESN(config, np.random.default_rng(0))


@pytest.mark.parametrize(
    ("action", "match"),
    [
        (lambda esn: esn.run(np.zeros(20)), "2次元"),
        (lambda esn: esn.run(np.zeros((20, 2))), "入力次元"),
        (lambda esn: esn.step(np.zeros(29), np.zeros(1)), "状態"),
        (lambda esn: esn.step(np.zeros(30), np.zeros(2)), "入力"),
    ],
    ids=[
        "run_not_2d",
        "run_wrong_input_dim",
        "step_wrong_state_dim",
        "step_wrong_input_dim",
    ],
)
def test_shape_errors(action: Callable[[ESN], object], match: str) -> None:
    """形状不整合な呼び出しはそれぞれ該当箇所を含む ValueError を送出する。"""
    # N=10 / density=0.1 は期待辺数が 10 本しかなく、乱数次第で零行列になる
    # (D-134 で引き順を直したら実際に引いた)。形状の検査に大きさは要らない。
    esn = ESN(ESNConfig(n_units=30), np.random.default_rng(0))
    with pytest.raises(ValueError, match=match):
        action(esn)


def test_spectral_radius_requires_square_matrix() -> None:
    with pytest.raises(ValueError, match="正方行列"):
        spectral_radius(np.zeros((3, 4)))

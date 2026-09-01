"""結合行列を取り出す面の検査 (D-122).

``GraphReservoir`` は「持っているモデルだけが満たす追加の面」である。
ここで測るのは3つ:

1. 3モデルとも ``adjacency()`` が ``(N, N)`` を返し、**更新式が実際に使う行列**
   と一致すること (別物を返しても形は合うので、値まで見る)
2. ``require_graph`` が、面を持たないものを**黙って素通しにしない**こと
3. DeepESN の隣接行列が**層間の辺を含む**こと。落とすとグラフが層ごとに割れ、
   平均最短路長が「層の中だけの値」になる
"""

from __future__ import annotations

import numpy as np
import pytest

from rc_basics_lab.diagnostics.topology import small_world
from rc_basics_lab.reservoir.deep import DeepESN, DeepESNConfig
from rc_basics_lab.reservoir.esn import ESN, ESNConfig
from rc_basics_lab.reservoir.protocol import GraphReservoir, Reservoir
from rc_basics_lab.reservoir.registry import build_reservoir, require_graph
from rc_basics_lab.reservoir.ring import RingConfig, RingReservoir
from rc_basics_lab.types import FloatArray

CONFIGS = (
    ESNConfig(n_units=40),
    DeepESNConfig(n_units=80, n_layers=4),
    RingConfig(n_units=40),
)


def _build(config: object) -> Reservoir:
    assert not isinstance(config, str)
    return build_reservoir(config, np.random.default_rng(0), n_inputs=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: type(c).__name__)
def test_every_model_exposes_a_square_adjacency(config: object) -> None:
    reservoir = _build(config)
    matrix = require_graph(reservoir, used_by="テスト")
    assert matrix.shape == (reservoir.n_units, reservoir.n_units)
    assert matrix.dtype == np.float64


def test_the_esn_adjacency_is_the_matrix_the_update_uses() -> None:
    esn = ESN(ESNConfig(n_units=30), np.random.default_rng(1))
    assert np.array_equal(esn.adjacency(), esn.W)


def test_the_ring_adjacency_is_the_matrix_the_update_uses() -> None:
    ring = RingReservoir(RingConfig(n_units=30), np.random.default_rng(1))
    assert np.array_equal(ring.adjacency(), ring.W)


def test_the_deep_adjacency_keeps_each_layer_block_on_the_diagonal() -> None:
    deep = DeepESN(DeepESNConfig(n_units=80, n_layers=4), np.random.default_rng(2))
    matrix = deep.adjacency()
    for layer in range(4):
        span = deep.layer_slice(layer)
        assert np.array_equal(matrix[span, span], deep._recurrent[layer])


def test_the_deep_adjacency_carries_the_inter_layer_edges() -> None:
    """層間を落とすと層ごとに切れる —— それが起きていないこと。"""
    deep = DeepESN(DeepESNConfig(n_units=80, n_layers=4), np.random.default_rng(2))
    matrix = deep.adjacency()
    reachable = small_world(matrix).scalars["reachable_fraction"]

    block_diagonal: FloatArray = np.zeros_like(matrix)
    for layer in range(4):
        span = deep.layer_slice(layer)
        block_diagonal[span, span] = matrix[span, span]
    split = small_world(block_diagonal).scalars["reachable_fraction"]

    assert reachable > split, (
        "層間の辺が隣接行列に入っていません "
        f"(到達率 {reachable:.3f} が層内だけの {split:.3f} を上回りません)"
    )


def test_require_graph_rejects_a_model_without_a_matrix() -> None:
    class Stateless:
        """``Reservoir`` の5面は満たすが結合行列を持たないモデル。"""

        @property
        def config(self) -> ESNConfig:
            return ESNConfig()

        @property
        def n_units(self) -> int:
            return 3

        @property
        def n_inputs(self) -> int:
            return 1

        def step(
            self,
            x: FloatArray,
            u: FloatArray,
            rng: np.random.Generator | None = None,
        ) -> FloatArray:
            return x

        def run(
            self,
            u: FloatArray,
            x0: FloatArray | None = None,
            rng: np.random.Generator | None = None,
        ) -> FloatArray:
            return np.zeros((u.shape[0], 3))

    model = Stateless()
    assert isinstance(model, Reservoir)
    assert not isinstance(model, GraphReservoir)
    with pytest.raises(TypeError, match="adjacency を持ちません"):
        require_graph(model, used_by="トポロジ診断")


def test_require_graph_rejects_a_matrix_of_the_wrong_size() -> None:
    class Mismatched:
        @property
        def n_units(self) -> int:
            return 5

        def adjacency(self) -> FloatArray:
            return np.zeros((3, 3))

    with pytest.raises(ValueError, match="n_units と一致しません"):
        require_graph(Mismatched(), used_by="テスト")  # type: ignore[arg-type]

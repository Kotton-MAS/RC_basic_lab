"""結合構造 (``reservoir/topology.py``) の検査.

守るのは3つ。

1. **既定 (Erdos-Renyi) が分離前と同じ乱数の引き方であること**。ここが崩れると
   ``results/`` の成果物が動く (D-74 の合否判定)
2. **各トポロジが文献の性質を実際に示すこと**。「動く」だけなら全部同じ疎行列を
   返す実装でも緑になる
3. 設定値の検査が各トポロジに効くこと
"""

from __future__ import annotations

import numpy as np
import pytest

from rc_basics_lab.reservoir.topology import (
    BarabasiAlbertConfig,
    ErdosRenyiConfig,
    RingTopologyConfig,
    TopologyConfig,
    WattsStrogatzConfig,
    build_mask,
    nominal_density,
)
from rc_basics_lab.types import BoolArray

N = 120


def _mask(config: TopologyConfig, seed: int = 0, n_units: int = N) -> BoolArray:
    return build_mask(config, n_units, np.random.default_rng(seed))


# --- 1. 既定が分離前と同じ引き方であること (D-74) ----------------------------


def test_erdos_renyi_draws_exactly_what_the_old_code_drew() -> None:
    """``rng.random((n, n)) < density`` **そのもの**であること。

    分離前のコードはこの1行だった。ここが変わると乱数列がずれ、以後に引く
    重みまで全部ずれるので、``results/`` の成果物が動く。
    """
    density = 0.1
    expected = np.random.default_rng(7).random((N, N)) < density
    actual = _mask(ErdosRenyiConfig(density=density), seed=7)
    assert np.array_equal(actual, expected)


def test_erdos_renyi_consumes_one_draw_of_the_full_matrix() -> None:
    """引く乱数の**個数**も同じ (N*N 個ちょうど)。

    個数がずれると、この後に引く ``rng.uniform`` の値が全部ずれる。
    """
    rng_a = np.random.default_rng(3)
    build_mask(ErdosRenyiConfig(), N, rng_a)
    after_topology = rng_a.uniform(-1.0, 1.0, 5)

    rng_b = np.random.default_rng(3)
    rng_b.random((N, N))
    after_manual = rng_b.uniform(-1.0, 1.0, 5)

    assert np.array_equal(after_topology, after_manual)


def test_the_ring_draws_no_randomness() -> None:
    """閉路は構造が決まりきっているので乱数を1個も引かない。"""
    rng = np.random.default_rng(1)
    before = rng.bit_generator.state
    build_mask(RingTopologyConfig(), N, rng)
    assert rng.bit_generator.state == before


# --- 2. 文献の性質 -----------------------------------------------------------


def test_barabasi_albert_has_hubs() -> None:
    """スケールフリーは**次数の散らばりが大きい** (ハブができる)。

    同じくらいの密度の Watts-Strogatz と比べて、最大次数が明確に大きいこと。
    ここが同じなら「スケールフリーにした」と言えない。
    """
    scale_free = _mask(BarabasiAlbertConfig(m=2)).sum(axis=1)
    small_world = _mask(WattsStrogatzConfig(k=4, beta=0.1)).sum(axis=1)
    assert scale_free.max() > 2 * small_world.max(), (
        f"ハブができていません: BA max={scale_free.max()} WS max={small_world.max()}"
    )
    assert scale_free.std() > small_world.std()


def test_watts_strogatz_keeps_the_degree_nearly_uniform() -> None:
    """スモールワールドは次数がほぼ揃う (格子から張り替えるため)。"""
    degrees = _mask(WattsStrogatzConfig(k=4, beta=0.1)).sum(axis=1)
    assert degrees.std() < 1.5, f"次数が揃っていません: std={degrees.std()}"


def test_a_higher_rewiring_probability_scatters_the_degrees() -> None:
    """``beta`` が効く (D-13: 効かない設定は飾りである)。"""
    ordered = _mask(WattsStrogatzConfig(k=4, beta=0.0)).sum(axis=1)
    rewired = _mask(WattsStrogatzConfig(k=4, beta=1.0)).sum(axis=1)
    assert ordered.std() == 0.0, "beta=0 は格子なので次数が完全に揃うはず"
    assert rewired.std() > ordered.std()


def test_the_ring_is_a_single_cycle() -> None:
    """閉路は各行の非零が1つで、1周する。"""
    mask = _mask(RingTopologyConfig(), n_units=10)
    assert mask.sum() == 10
    for row in range(10):
        assert mask[row, row - 1]


@pytest.mark.parametrize(
    "config",
    [
        ErdosRenyiConfig(density=0.2),
        RingTopologyConfig(),
        BarabasiAlbertConfig(m=3),
        WattsStrogatzConfig(k=6, beta=0.2),
    ],
)
def test_every_topology_produces_at_least_one_edge(config: TopologyConfig) -> None:
    """どのトポロジも辺を作る (零行列を返すと W のスペクトル半径が 0 で落ちる)。"""
    assert _mask(config).sum() > 0


def test_the_same_seed_reproduces_the_mask() -> None:
    """同一シードで同じマスクになる (D-06)。"""
    for config in (ErdosRenyiConfig(), BarabasiAlbertConfig(), WattsStrogatzConfig()):
        assert np.array_equal(_mask(config, seed=5), _mask(config, seed=5))


def test_a_different_seed_changes_the_mask() -> None:
    """シードが違えば変わる (**乱数を実際に使っていることの確認**)。"""
    for config in (ErdosRenyiConfig(), BarabasiAlbertConfig(), WattsStrogatzConfig()):
        assert not np.array_equal(_mask(config, seed=5), _mask(config, seed=6))


# --- 3. 設定値の検査 ---------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (ErdosRenyiConfig(density=0.0), "density"),
        (ErdosRenyiConfig(density=1.5), "density"),
        (BarabasiAlbertConfig(m=0), "m"),
        (BarabasiAlbertConfig(m=N), "m"),
        (WattsStrogatzConfig(k=3), "k"),
        (WattsStrogatzConfig(k=0), "k"),
        (WattsStrogatzConfig(beta=-0.1), "beta"),
        (WattsStrogatzConfig(beta=1.1), "beta"),
    ],
)
def test_an_out_of_range_setting_is_rejected(
    config: TopologyConfig, match: str
) -> None:
    """トポロジ固有の値の範囲検査 (作る側と検査する側を分けない)。"""
    with pytest.raises(ValueError, match=match):
        _mask(config)


def test_too_few_units_is_rejected() -> None:
    """辺を引けない大きさは落とす。"""
    with pytest.raises(ValueError, match="n_units"):
        build_mask(ErdosRenyiConfig(), 1, np.random.default_rng(0))


def test_the_nominal_density_matches_erdos_renyi() -> None:
    """``nominal_density`` は ER では設定値そのもの (成果物の列に書く値)。"""
    assert nominal_density(ErdosRenyiConfig(density=0.25), N) == 0.25


@pytest.mark.parametrize(
    "config",
    [RingTopologyConfig(), BarabasiAlbertConfig(m=2), WattsStrogatzConfig(k=4)],
)
def test_the_nominal_density_is_close_to_the_realised_one(
    config: TopologyConfig,
) -> None:
    """見込みの密度が実際の密度と桁で合う (**でたらめな値を書かない**)。

    ``nominal_density`` は「そう設定した」を答えるが、実測と桁がずれていたら
    成果物の ``density`` 列が意味を失う。
    """
    realised = float(_mask(config).mean())
    nominal = nominal_density(config, N)
    assert 0.4 * nominal <= realised <= 2.5 * nominal, (
        f"{config}: 見込み {nominal:.4f} と実測 {realised:.4f} が離れすぎです"
    )

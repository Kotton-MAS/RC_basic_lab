"""結合構造 (``reservoir/topology.py``) の検査.

守るのは3つ。

1. **既定 (Erdos-Renyi) が分離前と同じ乱数の引き方であること**。ここが崩れると
   ``results/`` の成果物が動く (D-74 の合否判定)
2. **各トポロジが文献の性質を実際に示すこと**。「動く」だけなら全部同じ疎行列を
   返す実装でも緑になる
3. 設定値の検査が各トポロジに効くこと
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rc_basics_lab.reservoir import topology as topology_module
from rc_basics_lab.reservoir.topology import (
    BarabasiAlbertConfig,
    DegreePreservingConfig,
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


# --- docstring と実態の一致 (D-131) ---------------------------------------

SYMMETRY_TABLE: tuple[tuple[TopologyConfig, bool, bool], ...] = (
    (ErdosRenyiConfig(density=0.1), True, False),
    (BarabasiAlbertConfig(), False, True),
    (WattsStrogatzConfig(), False, True),
    (RingTopologyConfig(), False, False),
)
"""``(設定, 自己ループを持つか, 辺の有無が対称か)`` の実測 (N=200, seed=7)。

``reservoir/topology.py`` の docstring の表と**同じ内容**である。散文が実態と
ずれると、次に読む人が「自己結合はどれにもある」という前提で実験を設計する。
"""


@pytest.mark.parametrize(
    ("config", "has_self_loop", "symmetric"),
    SYMMETRY_TABLE,
    ids=lambda v: type(v).__name__ if hasattr(v, "__class__") else str(v),
)
def test_the_docstring_table_matches_the_measured_symmetry(
    config: TopologyConfig, has_self_loop: bool, symmetric: bool
) -> None:
    """自己ループと対称性が docstring の表どおりであること (D-131)。"""
    mask = _mask(config, seed=7, n_units=200)
    assert bool(np.trace(mask) > 0) is has_self_loop
    assert bool((mask == mask.T).all()) is symmetric


def test_the_docstring_records_every_topology_in_the_table() -> None:
    """docstring の表に全トポロジが載っていること (落とすと交絡が隠れる)。"""
    source = Path(topology_module.__file__).read_text(encoding="utf-8")
    header = source.split('"""')[1]
    for config, _, _ in SYMMETRY_TABLE:
        name = type(config).__name__
        assert name in header, f"docstring の表に {name} がありません (D-131)"


# --- ペアが組めること (D-134) ---------------------------------------------


def test_the_same_seed_gives_the_same_weights_under_every_topology() -> None:
    """同一シードなら**同じ重み行列を違うマスクで切り出す** (D-134)。

    値を先に引き、そのあとで結合の有無を切り出す順にしてある。逆だと
    ``build_mask`` が消費する乱数の個数がトポロジによって違うぶん重みの
    実現値までずれ、**トポロジの効果と重みの実現値の分散が分離できない**
    (このリポジトリの統計はペアが前提である)。
    """
    from rc_basics_lab.reservoir.esn import ESN, ESNConfig

    def build(topology: TopologyConfig) -> ESN:
        return ESN(ESNConfig(n_units=120, topology=topology), np.random.default_rng(0))

    reference = build(ErdosRenyiConfig(density=0.1))
    for topology in (
        BarabasiAlbertConfig(),
        WattsStrogatzConfig(),
        RingTopologyConfig(),
    ):
        other = build(topology)
        assert np.array_equal(reference.W_in, other.W_in), "W_in が一致しません"
        shared = (reference.W != 0) & (other.W != 0)
        assert shared.sum() > 0, f"{topology} と共通する辺がありません"
        left = reference.W[shared] / reference.config.spectral_radius
        right = other.W[shared] / other.config.spectral_radius
        assert np.array_equal(np.sign(left), np.sign(right)), (
            f"{type(topology).__name__} と共通辺の重みの符号が違います "
            "(値を先に引く順が壊れています)"
        )


def test_the_topology_stream_varies_the_graph_without_the_weights() -> None:
    """``topology_rng`` を分けると、グラフだけ / 重みだけを振れる (D-134)。"""
    from rc_basics_lab.reservoir.esn import ESN, ESNConfig
    from rc_basics_lab.seeds import SeedStream, make_rng_for

    def build(weight_replicate: int, graph_replicate: int) -> ESN:
        return ESN(
            ESNConfig(n_units=60),
            make_rng_for(0, SeedStream.RESERVOIR, weight_replicate),
            topology_rng=make_rng_for(0, SeedStream.TOPOLOGY, graph_replicate),
        )

    base = build(0, 0)
    other_graph = build(0, 1)
    other_weights = build(1, 0)
    assert not np.array_equal(base.W != 0, other_graph.W != 0), (
        "グラフのシードを変えてもマスクが同じです"
    )
    assert np.array_equal(base.W != 0, other_weights.W != 0), (
        "重みのシードを変えたらマスクまで変わりました"
    )


# --- 次数保存ランダム化 (D-135) -------------------------------------------


def test_degree_preserving_keeps_the_degree_sequence() -> None:
    """次数列を厳密に保ち、辺の集合は変える (D-135)。"""
    config = DegreePreservingConfig()
    base = _mask(config.base, seed=7, n_units=200)
    randomized = _mask(config, seed=7, n_units=200)
    assert np.array_equal(np.sort(base.sum(axis=1)), np.sort(randomized.sum(axis=1))), (
        "次数列が変わりました (帰無モデルとして成立していません)"
    )
    assert not np.array_equal(base, randomized), "辺が1本も動いていません"


def test_degree_preserving_breaks_the_correlation_structure() -> None:
    """次数を保ったまま**相関だけ**が壊れる (クラスタ係数が下がる)。"""
    config = DegreePreservingConfig()
    base = _mask(config.base, seed=7, n_units=200)
    randomized = _mask(config, seed=7, n_units=200)
    assert _clustering(randomized) < _clustering(base), (
        "クラスタ係数が下がっていません (張り替えが効いていません)"
    )


def _clustering(mask: BoolArray) -> float:
    undirected = (mask | mask.T).astype(np.float64)
    np.fill_diagonal(undirected, 0.0)
    triangles = float(np.trace(undirected @ undirected @ undirected))
    degrees = undirected.sum(axis=1)
    return triangles / float(np.sum(degrees * (degrees - 1.0)))


def test_degree_preserving_matches_networkx_double_edge_swap() -> None:
    """外部実装 (``networkx``) と**同じ性質**を持つことを突き合わせる。

    交換はランダムなので辺の集合そのものは一致しないが、**次数列が保たれる**
    という帰無モデルの定義は一致していなければならない (D-62 と同じ扱いで、
    ``networkx`` は dev のオラクル)。
    """
    import networkx as nx

    config = DegreePreservingConfig()
    base = _mask(config.base, seed=11, n_units=120)
    graph = nx.from_numpy_array(base | base.T)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    before = sorted(degree for _, degree in graph.degree())
    nx.double_edge_swap(graph, nswap=graph.number_of_edges(), max_tries=100_000, seed=1)
    assert sorted(degree for _, degree in graph.degree()) == before

    randomized = _mask(config, seed=11, n_units=120)
    ours = (randomized | randomized.T).astype(np.int64)
    np.fill_diagonal(ours, 0)
    assert sorted(ours.sum(axis=1).tolist()) == before, (
        "自前の張り替えが networkx と違う次数列を作っています"
    )

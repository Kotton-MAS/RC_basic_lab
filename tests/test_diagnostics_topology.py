"""結合行列を取る診断の検査 (D-122).

## オラクルの取り方

クラスタ係数・平均最短路長・次数は ``networkx`` が持っているので、**実装が
自前の numpy / scipy でも答えは外部の実装と突き合わせる**。``networkx`` は
dev グループにだけ入れる (実行時依存を増やさない。D-62 と同じ扱い)。

## 族の分離もここで測る

``X`` を取る族 (D-01) と ``W`` を取る族は第1引数の名前で分けてある。
分離が壊れると「``X`` を無視する診断」が D-01 契約テストを素通りするので、
両族が交わらないことと、この族の署名が揃っていることをここで固定する。
"""

from __future__ import annotations

import inspect

import networkx as nx
import numpy as np
import pytest
from test_diagnostics_base import iter_diagnostic_callables

from rc_basics_lab.diagnostics.base import DiagnosticResult
from rc_basics_lab.diagnostics.topology import (
    DEFAULT_DEGREE,
    DegreeConfig,
    SmallWorldConfig,
    SpectralConfig,
    degree_distribution,
    small_world,
    spectral_profile,
)
from rc_basics_lab.types import FloatArray

KNOWN_GRAPH_DIAGNOSTICS = (
    "rc_basics_lab.diagnostics.topology.degree_distribution",
    "rc_basics_lab.diagnostics.topology.spectral_profile",
    "rc_basics_lab.diagnostics.topology.small_world",
)
"""``W`` を取る診断の全件。**件数まで固定する** (D-01 側と同じ流儀)。

列挙条件を壊して件数が 0 になっても契約テストが緑のまま通る経路を塞ぐ。
"""


def _random_matrix(n_units: int, density: float, seed: int) -> FloatArray:
    rng = np.random.default_rng(seed)
    mask = rng.random((n_units, n_units)) < density
    values: FloatArray = rng.uniform(-1.0, 1.0, (n_units, n_units))
    matrix: FloatArray = np.where(mask, values, 0.0)
    return matrix


def _ring_matrix(n_units: int) -> FloatArray:
    matrix: FloatArray = np.zeros((n_units, n_units))
    for i in range(n_units):
        matrix[i, (i - 1) % n_units] = 0.5
    return matrix


# --- 族の分離 -------------------------------------------------------------


def test_the_graph_family_is_enumerated_exactly() -> None:
    found = {name for name, _ in iter_diagnostic_callables("W")}
    assert found == set(KNOWN_GRAPH_DIAGNOSTICS), (
        f"W を取る診断の集合が想定と一致しません: "
        f"不足={sorted(set(KNOWN_GRAPH_DIAGNOSTICS) - found)}, "
        f"余剰={sorted(found - set(KNOWN_GRAPH_DIAGNOSTICS))}"
    )


def test_the_two_families_never_overlap() -> None:
    state_family = {name for name, _ in iter_diagnostic_callables("X")}
    graph_family = {name for name, _ in iter_diagnostic_callables("W")}
    assert not (state_family & graph_family)
    assert not (state_family & set(KNOWN_GRAPH_DIAGNOSTICS)), (
        "トポロジ診断が X を取る族に紛れ込んでいます。"
        "第1引数を W にしてください (D-01 契約テストが意味を失います)。"
    )


@pytest.mark.parametrize("name", KNOWN_GRAPH_DIAGNOSTICS)
def test_graph_diagnostics_share_one_signature(name: str) -> None:
    """``f(W, *, ctx=None, cfg=...) -> DiagnosticResult`` に揃っていること。"""
    found = dict(iter_diagnostic_callables("W"))
    signature = inspect.signature(found[name], eval_str=True)
    parameters = list(signature.parameters.values())
    assert parameters[0].name == "W"
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for parameter in parameters[1:]:
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name}: {parameter.name} がキーワード専用ではありません"
        )
        assert parameter.default is not inspect.Parameter.empty, (
            f"{name}: {parameter.name} に既定値がありません"
        )
    assert {p.name for p in parameters[1:]} == {"ctx", "cfg"}


# --- 次数分布 -------------------------------------------------------------


def test_degrees_match_networkx() -> None:
    matrix = _random_matrix(60, 0.08, seed=3)
    result = degree_distribution(matrix)
    graph = nx.from_numpy_array(matrix != 0.0, create_using=nx.DiGraph)
    # W[i, j] は j -> i なので、行和 (実装の in_degree) は networkx の out_degree
    expected = np.array([d for _, d in sorted(graph.out_degree())], dtype=np.float64)
    assert np.array_equal(np.sort(result.arrays["in_degrees"]), np.sort(expected))
    assert result.scalars["n_edges"] == float(graph.number_of_edges())


def test_a_ring_has_exactly_one_incoming_edge_per_unit() -> None:
    result = degree_distribution(_ring_matrix(40))
    assert result.scalars["in_degree_max"] == 1.0
    assert result.scalars["in_degree_std"] == 0.0
    assert result.scalars["mean_degree"] == 1.0


def test_a_hub_shows_up_in_the_max_degree() -> None:
    matrix = _ring_matrix(40)
    matrix[0, :] = 0.5  # 1つの素子に全部から辺を張る
    result = degree_distribution(matrix)
    assert result.scalars["in_degree_max"] == 40.0
    assert result.scalars["in_degree_std"] > 1.0


def test_tail_min_degree_changes_the_exponent() -> None:
    """設定の葉が効くこと (D-13)。"""
    matrix = _random_matrix(80, 0.06, seed=11)
    low = degree_distribution(matrix, cfg=DegreeConfig(tail_min_degree=2))
    high = degree_distribution(matrix, cfg=DegreeConfig(tail_min_degree=5))
    assert low.scalars["tail_exponent"] != high.scalars["tail_exponent"]
    assert high.params["tail_min_degree"] == "5"


def test_tail_exponent_is_nan_when_it_cannot_be_fitted() -> None:
    result = degree_distribution(np.zeros((10, 10)), cfg=DEFAULT_DEGREE)
    assert np.isnan(result.scalars["tail_exponent"])


# --- スペクトル -----------------------------------------------------------


def test_spectral_radius_matches_numpy() -> None:
    matrix = _random_matrix(50, 0.1, seed=5)
    result = spectral_profile(matrix)
    expected = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    assert result.scalars["spectral_radius"] == pytest.approx(expected)


def test_spectral_gap_is_zero_for_a_ring() -> None:
    """巡回行列の固有値は同じ絶対値の円周上に並ぶので、ギャップは 0。"""
    result = spectral_profile(_ring_matrix(24))
    assert result.scalars["spectral_gap"] == pytest.approx(0.0, abs=1e-12)


def test_n_reported_changes_the_returned_array() -> None:
    """設定の葉が効くこと (D-13)。"""
    matrix = _random_matrix(30, 0.2, seed=7)
    assert spectral_profile(matrix, cfg=SpectralConfig(n_reported=3)).arrays[
        "eigenvalue_abs"
    ].shape == (3,)
    assert spectral_profile(matrix, cfg=SpectralConfig(n_reported=9)).arrays[
        "eigenvalue_abs"
    ].shape == (9,)


# --- スモールワールド -----------------------------------------------------


def test_clustering_matches_networkx() -> None:
    matrix = _random_matrix(60, 0.1, seed=13)
    result = small_world(matrix)
    mask = matrix != 0.0
    undirected = nx.from_numpy_array(mask | mask.T)
    undirected.remove_edges_from(nx.selfloop_edges(undirected))
    assert result.scalars["clustering"] == pytest.approx(
        nx.transitivity(undirected), rel=1e-9
    )


def test_path_length_matches_networkx() -> None:
    matrix = _random_matrix(50, 0.15, seed=17)
    result = small_world(matrix)
    graph = nx.from_numpy_array(matrix != 0.0, create_using=nx.DiGraph)
    lengths = [
        length
        for source, targets in nx.all_pairs_shortest_path_length(graph)
        for target, length in targets.items()
        if source != target
    ]
    # 実装は W[i, j] が j -> i なので、networkx 側は転置したグラフに当たる。
    # 平均は向きを反転しても変わらない (全順序対を取るため)。
    assert result.scalars["path_length"] == pytest.approx(float(np.mean(lengths)))


def test_a_ring_is_not_a_small_world() -> None:
    """リングはクラスタ 0・路が長い。sigma は 1 を大きく下回る。"""
    result = small_world(_ring_matrix(60))
    assert result.scalars["clustering"] == pytest.approx(0.0)
    assert result.scalars["path_length"] > 10.0
    assert result.scalars["reachable_fraction"] == pytest.approx(1.0)


def test_disconnected_units_lower_the_reachable_fraction() -> None:
    matrix: FloatArray = np.zeros((20, 20))
    matrix[:10, :10] = _ring_matrix(10)  # 片方だけ繋ぐ
    result = small_world(matrix)
    assert 0.0 < result.scalars["reachable_fraction"] < 1.0


def test_unweighted_changes_the_path_length() -> None:
    """設定の葉が効くこと (D-13)。"""
    matrix = _random_matrix(40, 0.12, seed=19)
    hops = small_world(matrix, cfg=SmallWorldConfig(unweighted=True))
    weighted = small_world(matrix, cfg=SmallWorldConfig(unweighted=False))
    assert hops.scalars["path_length"] != weighted.scalars["path_length"]
    assert weighted.params["unweighted"] == "False"


# --- 入力の検証 -----------------------------------------------------------


@pytest.mark.parametrize(
    "diagnostic", [degree_distribution, spectral_profile, small_world]
)
def test_a_non_square_matrix_is_rejected(diagnostic: object) -> None:
    assert callable(diagnostic)
    with pytest.raises(ValueError, match="正方行列"):
        diagnostic(np.zeros((3, 4)))


@pytest.mark.parametrize(
    "diagnostic", [degree_distribution, spectral_profile, small_world]
)
def test_an_empty_matrix_is_rejected(diagnostic: object) -> None:
    assert callable(diagnostic)
    with pytest.raises(ValueError, match="空"):
        diagnostic(np.zeros((0, 0)))


@pytest.mark.parametrize(
    "diagnostic", [degree_distribution, spectral_profile, small_world]
)
def test_every_graph_diagnostic_returns_a_result(diagnostic: object) -> None:
    assert callable(diagnostic)
    result = diagnostic(_random_matrix(20, 0.2, seed=23))
    assert isinstance(result, DiagnosticResult)
    assert result.name
    assert result.scalars

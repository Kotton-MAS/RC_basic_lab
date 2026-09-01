"""結合行列 ``W`` を入力に取る診断 (D-122).

## `X` を取る診断とは別の署名族である

既存の診断は状態系列 ``X`` を取る (D-01)。ここに置くのは**結合行列そのもの**を
取る診断で、``f(W, *, ctx=None, cfg=...) -> DiagnosticResult`` という別の形をする。

同じ ``Diagnostic`` Protocol に混ぜてはいけない。混ぜると「``X`` 引数を無視する
診断」が生まれ、``tests/test_diagnostics_base.py`` の D-01 契約テストが
**意味を失う** (無視してよい引数がある、という前例ができる)。2つの族は
**第1引数の名前**で機械的に分けてある —— 族の違いはまさにそこにあるので、
名前の一覧を別に持つより壊れにくい。

## 何を入力に取るか

``Reservoir`` ではなく ``FloatArray`` の隣接行列を取る。``diagnostics`` が
``reservoir`` を import しない規律 (D-12) をそのまま守れるうえ、外部素子の
結合行列にもそのまま当たる。モデル側から行列を取り出すのは
``reservoir.registry.require_graph`` の仕事である。

向きの規約は ``W[i, j] != 0`` が **j から i への辺**。``x_{t+1} = f(W x_t + ...)``
の形なので、行が受け手、列が送り手になる。したがって**行和が入次数**である。

## 実行時依存を増やしていない

次数分布・スペクトル・クラスタ係数・平均最短路長は numpy と
``scipy.sparse.csgraph`` で書ける。``networkx`` は**テストのオラクル**として
dev グループにだけ入れる (D-62 と同じ扱い)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import shortest_path

from rc_basics_lab.diagnostics.base import DiagnosticContext, DiagnosticResult
from rc_basics_lab.types import BoolArray, FloatArray


@dataclass(frozen=True, slots=True)
class DegreeConfig:
    """``degree_distribution`` の判定基準.

    Attributes:
        tail_min_degree: べき指数を当てはめる下限次数 k_min。これ未満の次数は
            当てはめから外す。べき則は裾の性質なので、小さい次数まで含めると
            指数が系統的に大きく出る。
    """

    tail_min_degree: int = 2


@dataclass(frozen=True, slots=True)
class SpectralConfig:
    """``spectral_profile`` の判定基準.

    Attributes:
        n_reported: 大きいほうから何個の固有値の絶対値を配列で返すか。
    """

    n_reported: int = 8


@dataclass(frozen=True, slots=True)
class SmallWorldConfig:
    """``small_world`` の判定基準.

    Attributes:
        unweighted: 最短路長を辺の有無だけで測るか。``False`` にすると
            ``|W|`` の逆数を辺の長さに使う (強い結合ほど近い)。既定を ``True``
            にするのは、辺の本数で測った路長のほうが乱数グラフの期待値
            ``ln(N) / ln(k)`` と同じ単位になり、``small_world_index`` の比が
            意味を持つため。**``False`` にすると sigma は単位が揃わなくなる**。
    """

    unweighted: bool = True


DEFAULT_DEGREE = DegreeConfig()
DEFAULT_SPECTRAL = SpectralConfig()
DEFAULT_SMALL_WORLD = SmallWorldConfig()


def _check_square(matrix: FloatArray) -> FloatArray:
    """``W`` が正方行列であることを確かめて float64 に正規化する。"""
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"W は正方行列である必要があります: {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("W が空です")
    return array


def _edge_mask(matrix: FloatArray) -> BoolArray:
    """辺の有無。``W[i, j] != 0`` が j -> i の辺 (自己ループを含む)。"""
    mask: BoolArray = matrix != 0.0
    return mask


def _tail_exponent(degrees: FloatArray, min_degree: int) -> float:
    """裾のべき指数を Hill 推定量で返す。当てはめられなければ ``nan``。

    ``alpha = 1 + n / sum(ln(k_i / (k_min - 0.5)))`` (Clauset ら 2009 の
    離散近似)。**この値だけでスケールフリーだと言ってはいけない** —— 有限の
    N では指数関数的な次数分布でも有限の alpha が出る。比較のための数である。
    """
    tail = degrees[degrees >= float(min_degree)]
    if tail.size < 2 or min_degree < 2:
        return float("nan")
    total = float(np.sum(np.log(tail / (float(min_degree) - 0.5))))
    if total <= 0.0:
        return float("nan")
    return 1.0 + float(tail.size) / total


def degree_distribution(
    W: FloatArray,
    *,
    ctx: DiagnosticContext | None = None,
    cfg: DegreeConfig = DEFAULT_DEGREE,
) -> DiagnosticResult:
    """次数分布の要約を返す。

    ハブの有無を見る指標である。同じ密度でも Erdos-Renyi と Barabasi-Albert
    では ``in_degree_max`` と ``in_degree_std`` が大きく違う。

    Args:
        W: 結合行列 ``(N, N)``。``W[i, j] != 0`` が j -> i の辺。
        ctx: 使わない (族の署名を揃えるために受け取る)。
        cfg: 判定基準。

    Returns:
        ``scalars`` に次数の要約、``arrays`` に ``in_degrees`` / ``out_degrees``。

    Raises:
        ValueError: ``W`` が正方でない、または空の場合。
    """
    matrix = _check_square(W)
    mask = _edge_mask(matrix)
    n_units = mask.shape[0]
    in_degrees: FloatArray = mask.sum(axis=1).astype(np.float64)
    out_degrees: FloatArray = mask.sum(axis=0).astype(np.float64)
    n_edges = float(mask.sum())
    return DiagnosticResult(
        name="degree_distribution",
        scalars={
            "n_units": float(n_units),
            "n_edges": n_edges,
            "density": n_edges / float(n_units * n_units),
            "mean_degree": float(np.mean(in_degrees)),
            "in_degree_max": float(np.max(in_degrees)),
            "in_degree_std": float(np.std(in_degrees)),
            "out_degree_max": float(np.max(out_degrees)),
            "out_degree_std": float(np.std(out_degrees)),
            "tail_exponent": _tail_exponent(in_degrees, cfg.tail_min_degree),
        },
        arrays={"in_degrees": in_degrees, "out_degrees": out_degrees},
        params={"tail_min_degree": str(cfg.tail_min_degree)},
    )


def spectral_profile(
    W: FloatArray,
    *,
    ctx: DiagnosticContext | None = None,
    cfg: SpectralConfig = DEFAULT_SPECTRAL,
) -> DiagnosticResult:
    """固有値の分布を返す。

    ``spectral_radius`` は既存の ``reservoir.esn.spectral_radius`` と同じ量だが、
    こちらは**行列だけを見る診断**として第2の族に属し、ギャップと分散も返す。
    ギャップが大きいほど1つのモードが支配的で、状態の実効次元が落ちる。

    Args:
        W: 結合行列 ``(N, N)``。
        ctx: 使わない (族の署名を揃えるために受け取る)。
        cfg: 判定基準。

    Returns:
        ``scalars`` にスペクトル半径・ギャップ・絶対値の分散、
        ``arrays`` に大きいほうから ``cfg.n_reported`` 個の絶対値。

    Raises:
        ValueError: ``W`` が正方でない、または空の場合。
    """
    matrix = _check_square(W)
    magnitudes = np.sort(np.abs(np.linalg.eigvals(matrix)))[::-1]
    largest = float(magnitudes[0])
    second = float(magnitudes[1]) if magnitudes.size > 1 else 0.0
    reported: FloatArray = np.asarray(magnitudes[: cfg.n_reported], dtype=np.float64)
    return DiagnosticResult(
        name="spectral_profile",
        scalars={
            "spectral_radius": largest,
            "spectral_gap": largest - second,
            "eigenvalue_abs_mean": float(np.mean(magnitudes)),
            "eigenvalue_abs_std": float(np.std(magnitudes)),
        },
        arrays={"eigenvalue_abs": reported},
        params={"n_reported": str(cfg.n_reported)},
    )


def _clustering_coefficient(mask: BoolArray) -> float:
    """無向化したグラフの大域クラスタ係数 (推移性) を返す。

    三角形の数 / 連結三つ組の数。自己ループは外す —— 自分自身への辺は
    三角形を作らないのに三つ組を水増しし、密度が上がるほど係数を押し下げる。
    """
    undirected = (mask | mask.T).astype(np.float64)
    np.fill_diagonal(undirected, 0.0)
    triangles = float(np.trace(undirected @ undirected @ undirected))
    degrees = undirected.sum(axis=1)
    triples = float(np.sum(degrees * (degrees - 1.0)))
    if triples == 0.0:
        return float("nan")
    return triangles / triples


def _path_length(matrix: FloatArray, *, unweighted: bool) -> tuple[float, float]:
    """平均最短路長と、到達できる順序対の割合を返す。

    有向のまま測る。到達できない対は平均から外し、代わりに割合を返す ——
    ``inf`` を混ぜると平均が ``inf`` になり、「疎で切れている」と
    「遠いが繋がっている」を区別できなくなる。

    ``unweighted=False`` のときは ``|W|`` の逆数を辺の長さに使う。**重みそのもの
    を長さにしてはいけない** —— ``W`` は符号を持ち、強い結合ほど値が大きいので、
    そのまま距離にすると「強く繋がっているほど遠い」ことになる。
    """
    mask = _edge_mask(matrix)
    if unweighted:
        graph: FloatArray = mask.astype(np.float64)
    else:
        graph = np.zeros_like(matrix)
        graph[mask] = 1.0 / np.abs(matrix[mask])
    distances = shortest_path(graph, directed=True, unweighted=unweighted)
    np.fill_diagonal(distances, np.inf)
    finite = np.isfinite(distances)
    n_pairs = matrix.shape[0] * (matrix.shape[0] - 1)
    if n_pairs == 0 or not finite.any():
        return float("nan"), 0.0
    return float(np.mean(distances[finite])), float(finite.sum()) / float(n_pairs)


def small_world(
    W: FloatArray,
    *,
    ctx: DiagnosticContext | None = None,
    cfg: SmallWorldConfig = DEFAULT_SMALL_WORLD,
) -> DiagnosticResult:
    """クラスタ係数と平均最短路長、および乱数グラフとの比を返す。

    「クラスタが高いのに路が短い」がスモールワールドの定義なので、両方を
    同じ乱数グラフの期待値で割った ``small_world_index`` を返す
    (Humphries & Gurney 2008 の sigma)。

    **比較対象は解析的な期待値である** (``C_rand = density``,
    ``L_rand = ln(N) / ln(mean_degree)``)。実際に乱数グラフを生成して測ると
    シードが要り、同じ ``W`` に対して結果が揺れる。診断は決定的であるべきなので
    近似を採る —— 桁の比較には足りるが、**1.0 のすぐ上下を論じてはいけない**。

    Args:
        W: 結合行列 ``(N, N)``。
        ctx: 使わない (族の署名を揃えるために受け取る)。
        cfg: 判定基準。

    Returns:
        ``scalars`` にクラスタ係数・平均最短路長・到達率・sigma。

    Raises:
        ValueError: ``W`` が正方でない、または空の場合。
    """
    matrix = _check_square(W)
    mask = _edge_mask(matrix)
    n_units = mask.shape[0]
    clustering = _clustering_coefficient(mask)
    path, reachable = _path_length(matrix, unweighted=cfg.unweighted)
    density = float(mask.sum()) / float(n_units * n_units)
    mean_degree = float(mask.sum()) / float(n_units)
    random_path = (
        np.log(n_units) / np.log(mean_degree) if mean_degree > 1.0 else float("nan")
    )
    sigma = float("nan")
    if density > 0.0 and np.isfinite(random_path) and path > 0.0:
        sigma = (clustering / density) / (path / float(random_path))
    return DiagnosticResult(
        name="small_world",
        scalars={
            "clustering": clustering,
            "path_length": path,
            "reachable_fraction": reachable,
            "random_path_length": float(random_path),
            "small_world_index": sigma,
        },
        params={"unweighted": str(cfg.unweighted)},
    )


__all__ = [
    "DEFAULT_DEGREE",
    "DEFAULT_SMALL_WORLD",
    "DEFAULT_SPECTRAL",
    "DegreeConfig",
    "SmallWorldConfig",
    "SpectralConfig",
    "degree_distribution",
    "small_world",
    "spectral_profile",
]

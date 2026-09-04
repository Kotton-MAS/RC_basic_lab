"""結合構造 (トポロジ) をモデル本体から分離する層 (拡張性方針 §2-1).

**「どこが繋がっているか」と「どのモデルか」は別の軸である。** 分離する前は
``mask = rng.random((n, n)) < density`` が ``esn.py`` と ``deep.py`` の2箇所に
埋まっており、``density`` は両方の設定フィールドだった。どちらも
Erdos-Renyi 固定なので、**この形のままスケールフリーを足すと「BA トポロジの
ESN」という新しいモデルを1つ足すことになる**。Watts-Strogatz を足せばもう1つで、
組み合わせが掛け算で増える側の設計だった。

軸を分ければ、モデルを1つも足さずに

.. code-block:: yaml

    esn_mackey_glass:
      kind: esn
      topology:
        kind: barabasi_albert
        m: 2

と書けるようになる。**更新式を固定してトポロジだけを振る実験**が設定で書けて、
``esn`` と ``deep_esn`` の両方が同時に対応する (両者が同じ関数を呼ぶため)。

## 返すのはマスクであって重みではない

``build_mask`` が返すのは ``bool`` の隣接行列で、重みの値は各モデルが引く。
この分け方にすると **Erdos-Renyi の既定が乱数の引き方ごと保存できる** ————
分離前は ``rng.random((n, n)) < density`` -> ``rng.uniform(...)`` の順だったので、
マスクだけをここへ移せば同じ順のままになる (D-74 の合否判定を通せる)。

## 自己結合と有向性 —— トポロジによって違う (D-131)

**「どれも有向で自己結合を残す」ではない。** N=200 / density=0.1 での実測
(200 シード):

=========================  =====================  ========
トポロジ                   自己ループ             対称
=========================  =====================  ========
``ErdosRenyiConfig``       平均 19.7 (10〜32)     非対称
``BarabasiAlbertConfig``   0 本 (**構成上**)      **対称**
``WattsStrogatzConfig``    0 本 (**構成上**)      **対称**
``RingTopologyConfig``     0 本 (**構成上**)      非対称
=========================  =====================  ========

**ER だけは乱数の実現値で変わる** (期待値 ``N * density`` = 20、s.d. 4.4)。
かつてここには「17 本」と1シードの値が書いてあったが、それは**分布の1点**で
あって性質ではない。他の3つの 0 本は構成上そうなるので、シードによらない。

ER は各要素を独立に引くので自己ループも非対称性も出る。BA / WS は無向グラフの
文献アルゴリズムなので、構成上どちらも作れない。リングは巡回なので自己ループを
持たず、向きがある。

**この差は密度以外の交絡である。** 同じ密度で ER と BA を比べると、次数分布の
ほかに (1) 相互結合率 (BA は ``W[i,j]`` と ``W[j,i]`` が必ず対) と
(2) 自己ループの有無 (スペクトル半径の正規化と実効的な漏れに効く) も同時に動く。
**「スケールフリーだから容量が高い」と言うには対照が要る** (D-131)。

非対称化はモデル側の重みの引き方 (``rng.uniform`` は各要素独立) が行うので、
BA / WS でも**重みの値**は非対称になる —— 対称なのは**辺の有無**だけである。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import ClassVar

import numpy as np

from rc_basics_lab.types import BoolArray

MIN_UNITS = 2
"""トポロジを作れる最小のユニット数 (辺を1本以上引けること)。"""


@dataclass(frozen=True, slots=True)
class ErdosRenyiConfig:
    """一様ランダム結合 (**既定**。分離前の ``density`` と同じ挙動)。

    Attributes:
        density: 各要素が非零になる確率。
    """

    KIND: ClassVar[str] = "erdos_renyi"

    density: float = 0.1


@dataclass(frozen=True, slots=True)
class RingTopologyConfig:
    """一方向の単一閉路 (SCR の結合、Rodan & Tino 2011)。

    密度は ``1/N`` で構造から決まるので設定を持たない。

    **``kind: "ring"`` は2箇所にある** (D-149)。ここは ESN の
    ``topology:`` に書く**結合構造**で、``reservoir.ring.RingConfig``
    は ``reservoir:`` の ``kind:`` に書く**モデルそのもの** (SCR) である。
    書く場所が違うので設定は取り違えようがないが、**読む人は取り違える**:

    .. code-block:: yaml

       reservoir: {kind: esn, topology: {kind: ring}}   # ESN の結合だけリング
       reservoir: {kind: ring, n_units: 200}            # モデルが SCR

    上は入力重みも読み出しも ESN のままで結合行列だけが閉路、下は Rodan &
    Tino の SCR (入力重みの符号だけが乱数) である。**同じ名前だが別の実験**
    なので、成果物を読むときは ``reservoir.kind`` まで見ること。
    """

    KIND: ClassVar[str] = "ring"


@dataclass(frozen=True, slots=True)
class BarabasiAlbertConfig:
    """優先的選択によるスケールフリー結合 (Barabasi & Albert 1999)。

    次数分布が冪則に従い、**次数の大きいハブが少数できる**。「スケールフリーな
    リザバーは記憶容量が高いか」を測るための構造。

    Attributes:
        m: 新しい点が張る辺の本数。``1 <= m < n_units``。
            密度はおよそ ``2m/N`` になる。
    """

    KIND: ClassVar[str] = "barabasi_albert"

    m: int = 2


@dataclass(frozen=True, slots=True)
class WattsStrogatzConfig:
    """スモールワールド結合 (Watts & Strogatz 1998)。

    輪状の近傍結合から始めて、各辺を確率 ``beta`` で張り替える。
    ``beta=0`` は格子、``beta=1`` はランダムに近づく。

    Attributes:
        k: 各点の近傍の本数 (偶数)。``2 <= k < n_units``。
        beta: 張り替え確率 ``[0, 1]``。
    """

    KIND: ClassVar[str] = "watts_strogatz"

    k: int = 4
    beta: float = 0.1


@dataclass(frozen=True, slots=True)
class DegreePreservingConfig:
    """**別のトポロジの次数列だけを保った帰無モデル** (D-135)。

    ``base`` のグラフを作ってから、次数列を保ったまま辺を張り替える
    (double edge swap / configuration model)。**ハブの「次数分布」は保たれ、
    生成過程が作った相関構造だけが壊れる**。

    これはネットワーク科学では標準の帰無モデルだが、RC のトポロジ研究では
    まず置かれていない。「スケールフリーだから容量が高い」という主張に対して:

    - 優位が**残る** -> 効いているのは次数分布 (先行の主張が支持される)
    - **消える** -> 効いていたのは次数分布ではなく生成過程が作る別の何か

    Attributes:
        base: 次数列を借りるトポロジ (既定は Barabasi-Albert)。
        swaps_per_edge: 辺1本あたり何回の交換を試みるか。**大きいほど元の
            相関が消える**。100 は文献の慣例 (交換の受理率が下がって
            平衡に達するのに十分な回数)。
    """

    KIND: ClassVar[str] = "degree_preserving"

    base: BarabasiAlbertConfig = field(default_factory=BarabasiAlbertConfig)
    swaps_per_edge: int = 100


@dataclass(frozen=True, slots=True)
class TopologyControlConfig:
    """**交絡を1つずつ剥がすための対照** (D-138)。

    同じ密度で Erdos-Renyi と Barabasi-Albert を比べると、次数分布のほかに
    相互結合率と自己ループの有無も同時に動く (D-131)。どれが効いているのかは、
    **1つだけ動かした水準**を並べないと分からない。

    ``base`` のマスクを作ってから:

    - ``symmetrize`` が真なら ``mask | mask.T`` (相互結合だけを入れる)
    - ``drop_self_loops`` が真なら対角を落とす (自己ループだけを抜く)

    どちらも偽なら ``base`` そのものである (恒等)。**次数分布は変える手段を
    持たない** —— それは ``DegreePreservingConfig`` の仕事で、剥がす交絡が
    違う (D-135)。

    Attributes:
        base: 土台のトポロジ。
        symmetrize: 辺を対称にするか (相互結合を入れる)。
        drop_self_loops: 対角を落とすか (自己ループを抜く)。
    """

    KIND: ClassVar[str] = "control"

    base: ErdosRenyiConfig = field(default_factory=ErdosRenyiConfig)
    symmetrize: bool = False
    drop_self_loops: bool = False


type TopologyConfig = (
    ErdosRenyiConfig
    | RingTopologyConfig
    | BarabasiAlbertConfig
    | WattsStrogatzConfig
    | DegreePreservingConfig
    | TopologyControlConfig
)
"""結合構造の設定。**先頭が既定** (``kind`` を省くと Erdos-Renyi)。

足すときは末尾へ。並びを変えると既存の設定の意味が変わる
(``ReservoirConfig`` と同じ流儀)。
"""


def nominal_density(config: TopologyConfig, n_units: int) -> float:
    """設定から見込まれる密度 (成果物の ``density`` 列に書く値)。

    **実測ではなく設定から決まる値**である。``capacity.csv`` などの主表は
    ``density`` 列を持っており (成果物の列は変えられない)、トポロジを分離した
    後もその列に何を書くかを決める必要がある。

    Erdos-Renyi はその ``density`` そのもの。ほかは辺の本数から見込みを出す。
    実際に生成された密度が要るなら ``build_mask(...).mean()`` を使うこと ——
    ここが答えるのは「そう設定した」であって「そうなった」ではない。

    Args:
        config: トポロジの設定。
        n_units: ユニット数 N。

    Returns:
        ``[0, 1]`` の密度。
    """
    match config:
        case ErdosRenyiConfig():
            return config.density
        case RingTopologyConfig():
            return 1.0 / float(n_units)
        case BarabasiAlbertConfig():
            # **辺の本数は決定的なので厳密に数える** (D-140)。近似
            # (2m/N) は N=25 / m=2 で 0.160 を返すが、実際に生成される
            # のは 0.1504 で 6% ずれる —— 梯子は BA の密度を基準に
            # 他をそろえるので、そのずれがそのまま水準間の密度差になる。
            #   初期の完全結合 (m+1 点): m(m+1) 成分
            #   以降の N-m-1 点: 1点あたり m 本を無向で張るので 2m 成分
            m = config.m
            entries = m * (m + 1) + 2 * m * (n_units - m - 1)
            return min(1.0, entries / float(n_units * n_units))
        case WattsStrogatzConfig():
            return min(1.0, float(config.k) / float(n_units))
        case DegreePreservingConfig():
            # 次数列を保つので、借りてきた BA と同じ密度になる (D-135)
            return nominal_density(config.base, n_units)
        case TopologyControlConfig():
            # 対称化は辺を増やし (2d - d^2)、自己ループの除去は減らす。
            # **土台のままにしてはいけない** (D-140) —— 実測すると N=50 /
            # d=0.08 で対称化は 0.152 になり、梯子が排除したはずの密度差が
            # 対照そのものに入る。
            quadratic, linear = _control_coefficients(config, n_units)
            base = config.base.density
            return quadratic * base * base + linear * base


def _control_coefficients(
    config: TopologyControlConfig, n_units: int
) -> tuple[float, float]:
    """control の密度を ``A * d0^2 + B * d0`` と書いたときの ``(A, B)`` (D-140).

    土台の ER の密度 ``d0`` に対し、``_control`` が作るマスクの各成分が
    True になる確率は

    - 非対角 (``(N^2 - N) / N^2`` の割合): 対称化するなら ``2 d0 - d0^2``
      (``mask | mask.T`` は独立な2つの和事象)、しないなら ``d0``
    - 対角 (``N / N^2`` の割合): 自己ループを抜くなら 0、抜かないなら ``d0``
      (``mask[i, i] | mask[i, i]`` は ``mask[i, i]`` のまま)

    である。
    """
    off_diagonal = (n_units - 1) / n_units
    diagonal = 1.0 / n_units
    quadratic = -off_diagonal if config.symmetrize else 0.0
    linear = 2.0 * off_diagonal if config.symmetrize else off_diagonal
    if not config.drop_self_loops:
        linear += diagonal
    return quadratic, linear


def _control_base_density(
    config: TopologyControlConfig, target: float, n_units: int
) -> float:
    """control が ``target`` の密度になる土台の ``d0`` を返す (D-140).

    Raises:
        ValueError: その ``target`` に届かない場合 (対称化には上限がある)。
    """
    quadratic, linear = _control_coefficients(config, n_units)
    if quadratic == 0.0:
        return target / linear
    # -a d0^2 + B d0 = target を解く (a > 0)。小さいほうの根が [0, 1] に入る。
    a = -quadratic
    discriminant = linear * linear - 4.0 * a * target
    if discriminant < 0.0:
        raise ValueError(
            f"対称化した対照は密度 {target} に届きません "
            f"(N={n_units} での上限は {linear * linear / (4.0 * a):.4f})"
        )
    return float((linear - math.sqrt(discriminant)) / (2.0 * a))


def rescaled_to_density(
    config: TopologyConfig, density: float, n_units: int
) -> TopologyConfig | None:
    """密度を ``density`` に合わせた複製を返す。**合わせられなければ None**。

    Erdos-Renyi は密度そのものを持つので任意の値に合わせられる。BA と
    Watts-Strogatz は**整数の枝数**で密度が決まるので、任意の値には合わせ
    られない (``None`` を返す)。リングも同様に ``1/N`` で固定である。

    合わせられない水準があること自体は異常ではない —— **そちらが密度を
    決める側**になる (3-T の梯子は BA の密度に他を合わせる。D-139)。

    Args:
        config: トポロジの設定。
        density: 合わせたい密度 (**生成されるマスクの密度**であって、土台の
            ER に設定する値ではない。control は変換のぶんを逆算する。D-140)。
        n_units: ユニット数 N (control の逆算に要る)。

    Returns:
        同じ kind の複製、または合わせられないなら ``None``。

    Raises:
        ValueError: control がその密度に届かない場合。
    """
    match config:
        case ErdosRenyiConfig():
            return replace(config, density=density)
        case TopologyControlConfig():
            base = _control_base_density(config, density, n_units)
            return replace(config, base=replace(config.base, density=base))
        case (
            RingTopologyConfig()
            | BarabasiAlbertConfig()
            | WattsStrogatzConfig()
            | DegreePreservingConfig()
        ):
            return None


def build_mask(
    config: TopologyConfig, n_units: int, rng: np.random.Generator
) -> BoolArray:
    """結合の有無を表す ``(N, N)`` の bool 行列を返す (**分岐はここだけ**)。

    Args:
        config: トポロジの設定。
        n_units: ユニット数 N。
        rng: 乱数生成器。

    Returns:
        ``mask[i, j]`` が True なら ``j -> i`` の結合がある。

    Raises:
        ValueError: ``n_units`` が小さすぎる / 設定値が範囲外の場合。
    """
    if n_units < MIN_UNITS:
        raise ValueError(f"n_units は {MIN_UNITS} 以上である必要があります: {n_units}")
    match config:
        case ErdosRenyiConfig():
            return _erdos_renyi(config, n_units, rng)
        case RingTopologyConfig():
            return _ring(n_units)
        case BarabasiAlbertConfig():
            return _barabasi_albert(config, n_units, rng)
        case WattsStrogatzConfig():
            return _watts_strogatz(config, n_units, rng)
        case DegreePreservingConfig():
            return _degree_preserving(config, n_units, rng)
        case TopologyControlConfig():
            return _control(config, n_units, rng)


def _erdos_renyi(
    config: ErdosRenyiConfig, n_units: int, rng: np.random.Generator
) -> BoolArray:
    """一様ランダム。**分離前と同じ乱数の引き方**を保つ (D-74)。"""
    if not 0.0 < config.density <= 1.0:
        raise ValueError(f"density は (0, 1] である必要があります: {config.density}")
    drawn: BoolArray = rng.random((n_units, n_units)) < config.density
    return drawn


def _ring(n_units: int) -> BoolArray:
    """一方向の閉路。乱数を1個も引かない (構造が決まりきっているため)。"""
    mask: BoolArray = np.zeros((n_units, n_units), dtype=np.bool_)
    rows = np.arange(n_units)
    mask[rows, rows - 1] = True
    return mask


def _barabasi_albert(
    config: BarabasiAlbertConfig, n_units: int, rng: np.random.Generator
) -> BoolArray:
    """優先的選択。次数に比例した確率で既存の点へ繋ぐ。

    ``m`` 個の完全結合から始め、1点ずつ ``m`` 本の辺を張る。張り先は
    **その時点の次数に比例**して選ぶ (これが冪則を生む)。同じ点を2回
    選ばないよう、1点ぶんの選択は非復元で行う。
    """
    if not 1 <= config.m < n_units:
        raise ValueError(
            f"m は 1 以上 n_units 未満である必要があります: {config.m} (N={n_units})"
        )
    mask: BoolArray = np.zeros((n_units, n_units), dtype=np.bool_)
    # 初期の完全結合 (m+1 ノード)。自己結合は作らない。
    seed_size = config.m + 1
    for i in range(seed_size):
        for j in range(seed_size):
            if i != j:
                mask[i, j] = True
    degree = np.full(n_units, 0.0, dtype=np.float64)
    degree[:seed_size] = float(seed_size - 1)
    for new_node in range(seed_size, n_units):
        weights = degree[:new_node].copy()
        total = weights.sum()
        probabilities = (
            weights / total
            if total > 0.0
            else np.full(new_node, 1.0 / new_node, dtype=np.float64)
        )
        targets = rng.choice(new_node, size=config.m, replace=False, p=probabilities)
        for target in targets:
            mask[new_node, target] = True
            mask[target, new_node] = True
            degree[target] += 1.0
        degree[new_node] = float(config.m)
    return mask


def _watts_strogatz(
    config: WattsStrogatzConfig, n_units: int, rng: np.random.Generator
) -> BoolArray:
    """輪状の近傍結合を確率 ``beta`` で張り替える。

    格子 (高いクラスタ係数・長い最短路) からランダム (低い係数・短い路) へ
    連続的に動かす軸で、途中に**両方が良い**スモールワールド領域がある。
    """
    if config.k < 2 or config.k % 2 != 0:
        raise ValueError(f"k は 2 以上の偶数である必要があります: {config.k}")
    if config.k >= n_units:
        raise ValueError(f"k は n_units 未満である必要があります: {config.k}")
    if not 0.0 <= config.beta <= 1.0:
        raise ValueError(f"beta は [0, 1] である必要があります: {config.beta}")
    mask: BoolArray = np.zeros((n_units, n_units), dtype=np.bool_)
    half = config.k // 2
    for node in range(n_units):
        for offset in range(1, half + 1):
            neighbour = (node + offset) % n_units
            mask[node, neighbour] = True
            mask[neighbour, node] = True
    # 張り替え: 近傍側の辺を、既存でない相手へ移す。
    for node in range(n_units):
        for offset in range(1, half + 1):
            if rng.random() >= config.beta:
                continue
            old = (node + offset) % n_units
            candidates = [
                other
                for other in range(n_units)
                if other != node and not mask[node, other]
            ]
            if not candidates:
                continue
            new = int(rng.choice(np.asarray(candidates)))
            mask[node, old] = mask[old, node] = False
            mask[node, new] = mask[new, node] = True
    return mask


def _degree_preserving(
    config: DegreePreservingConfig, n_units: int, rng: np.random.Generator
) -> BoolArray:
    """次数列を保ったまま辺を張り替える (D-135)。

    無向グラフの double edge swap: 辺 ``(a, b)`` と ``(c, d)`` を選び、
    ``(a, d)`` と ``(c, b)`` へ張り替える。**両端点の次数は変わらない**。
    既に辺がある / 自己ループになる交換は棄却する (次数が変わるため)。

    ``base`` は無向 (対称) のトポロジに限る —— 有向グラフの次数保存交換は
    入次数と出次数を別々に保つ必要があり、無向の交換とは別の手続きになる。
    ここで扱うのは「BA のハブを次数だけ残して壊す」用途なので無向で足りる。
    """
    mask = _barabasi_albert(config.base, n_units, rng)
    if config.swaps_per_edge < 0:
        raise ValueError(
            f"swaps_per_edge は 0 以上である必要があります: {config.swaps_per_edge}"
        )
    upper = np.triu(mask, k=1)
    edges = np.argwhere(upper)
    if edges.shape[0] < 2:
        return mask
    working = mask.copy()
    attempts = config.swaps_per_edge * edges.shape[0]
    for _ in range(attempts):
        first, second = rng.integers(0, edges.shape[0], size=2)
        if first == second:
            continue
        a, b = edges[first]
        c, d = edges[second]
        if len({int(a), int(b), int(c), int(d)}) < 4:
            continue
        if working[a, d] or working[c, b]:
            continue
        for i, j in ((a, b), (c, d)):
            working[i, j] = working[j, i] = False
        for i, j in ((a, d), (c, b)):
            working[i, j] = working[j, i] = True
        edges[first] = (a, d)
        edges[second] = (c, b)
    return working


def _control(
    config: TopologyControlConfig, n_units: int, rng: np.random.Generator
) -> BoolArray:
    """土台のマスクから交絡を1つだけ動かす (D-138)。"""
    mask = build_mask(config.base, n_units, rng)
    if config.symmetrize:
        mask = mask | mask.T
    if config.drop_self_loops:
        mask = mask.copy()
        np.fill_diagonal(mask, False)
    return mask


__all__ = [
    "MIN_UNITS",
    "BarabasiAlbertConfig",
    "DegreePreservingConfig",
    "ErdosRenyiConfig",
    "RingTopologyConfig",
    "TopologyConfig",
    "TopologyControlConfig",
    "WattsStrogatzConfig",
    "build_mask",
    "nominal_density",
    "rescaled_to_density",
]

"""リザバーの生成口 — **``ESN(...)`` を直接呼ぶ場所を1つにする**.

``build_reservoir`` が唯一の生成口である。実験層はここだけを呼ぶので、
モデルを1つ足しても生成箇所 (実測で5箇所あった) を触らずに済む。

**既定は ESN のまま**なので、既存の YAML も既存の成果物も変わらない
(``ReservoirConfig`` の先頭が ``ESNConfig`` である限り)。
乱数の引き方も変えていない —— 合否判定は成果物のバイト不変である (D-74)。
"""

from __future__ import annotations

import numpy as np

from rc_basics_lab.reservoir.deep import DeepESN, DeepESNConfig
from rc_basics_lab.reservoir.esn import ESN, ESNConfig
from rc_basics_lab.reservoir.protocol import (
    GraphReservoir,
    Reservoir,
    ReservoirConfig,
)
from rc_basics_lab.reservoir.ring import RingConfig, RingReservoir
from rc_basics_lab.reservoir.topology import nominal_density
from rc_basics_lab.types import FloatArray


def build_reservoir(
    config: ReservoirConfig,
    rng: np.random.Generator,
    *,
    n_inputs: int = 1,
) -> Reservoir:
    """設定からリザバーを1つ作る (**モデル分岐はここだけ**)。

    ``readout/design.py`` の ``_layout_of`` と同じ流儀で、分岐を1箇所に
    正規化する。モデルを足すときは ``ReservoirConfig`` の union に設定型を
    足し、ここへ ``case`` を1つ書く —— mypy が網羅性を見るので、書き忘れは
    型検査で落ちる (実行時に「なぜか ESN になっている」にはならない)。

    Args:
        config: 構造ハイパーパラメータ。
        rng: 重み生成用の Generator (``seeds.make_rng`` の reservoir ストリーム)。
        n_inputs: 入力次元 D_in。課題側が決める量なので YAML ではなくここで渡す。

    Returns:
        ``Reservoir`` を満たすインスタンス。

    Raises:
        ValueError: 設定値が範囲外の場合 (各モデルの ``__init__`` が投げる)。
    """
    match config:
        case ESNConfig():
            return ESN(config, rng, n_inputs=n_inputs)
        case DeepESNConfig():
            return DeepESN(config, rng, n_inputs=n_inputs)
        case RingConfig():
            return RingReservoir(config, rng, n_inputs=n_inputs)


def require_esn(config: ReservoirConfig, used_by: str) -> ESNConfig:
    """``ESNConfig`` に絞る。他のモデルなら**どこが対応していないか**を言って落とす。

    03-C と 04 は ``spectral_radius`` / ``leak_rate`` / ``state_noise`` を
    格子にして振る。これは ESN 固有の軸で、「そのモデルで何を掃引するのか」は
    配線ではなく実験設計の問題である。**黙って ESN として扱わない** ——
    設定した ``kind`` が効かない実験になる。

    Args:
        config: 設定から来たリザバー設定。
        used_by: 呼び出し元の説明 (エラーに出す)。

    Returns:
        ``ESNConfig``。

    Raises:
        TypeError: ``ESNConfig`` でない場合。
    """
    if isinstance(config, ESNConfig):
        return config
    raise TypeError(
        f"{used_by} は現在 kind: esn だけに対応しています "
        f"(渡されたのは {type(config).__name__})。"
        "掃引軸が ESN 固有 (spectral_radius / leak_rate / state_noise) なので、"
        "他のモデルで何を振るかを決めてから対応させてください"
    )


def reservoir_density(config: ReservoirConfig) -> float:
    """成果物の ``density`` 列に書く値を、モデルによらず返す (D-124)。

    **実測ではなく設定から見込まれる値**である (``topology.nominal_density``
    と同じ規約)。掃引を ESN 以外へ広げるとき、``config.topology`` を直接
    読んでいる箇所が ``RingConfig`` で落ちるのを防ぐ。

    リングは巡回結合なので、非零は1行に1つ = ``1 / N`` である。

    Args:
        config: リザバーの構造設定。

    Returns:
        密度 (0 以上 1 以下)。
    """
    match config:
        case ESNConfig():
            return nominal_density(config.topology, config.n_units)
        case DeepESNConfig():
            # 層内の密度である (層間の結合は含めない)。層ごとの N で決まる。
            return nominal_density(config.topology, config.n_units // config.n_layers)
        case RingConfig():
            return 1.0 / float(config.n_units)


def require_graph(reservoir: Reservoir, used_by: str) -> FloatArray:
    """結合行列を持つモデルに絞り、その隣接行列を返す (D-122)。

    トポロジ診断 (``diagnostics.topology``) は行列を入力に取るので、モデルから
    行列を取り出す経路が要る。**取り出せないモデルを黙って素通りさせない** ——
    そうすると「トポロジ診断の行だけが静かに消えた成果物」ができ、
    再生成しても誰も気づけない。

    解析側で ``isinstance`` して分岐するのではなく、**必要とする側が要求する**
    (``require_esn`` と同じ流儀)。

    Args:
        reservoir: 生成済みのリザバー。
        used_by: 呼び出し元の説明 (エラーに出す)。

    Returns:
        ``(N, N)`` の隣接行列。``W[i, j] != 0`` が j -> i の辺。

    Raises:
        TypeError: ``adjacency`` を持たないモデルの場合。
        ValueError: 返った行列が ``(n_units, n_units)`` でない場合。
    """
    if not isinstance(reservoir, GraphReservoir):
        raise TypeError(
            f"{used_by} は結合行列を持つモデルだけに対応しています "
            f"({type(reservoir).__name__} は adjacency を持ちません)。"
            "トポロジを測る対象が無いので、何を測るのかを決めてから"
            "対応させてください"
        )
    matrix = reservoir.adjacency()
    expected = (reservoir.n_units, reservoir.n_units)
    if matrix.shape != expected:
        raise ValueError(
            f"{type(reservoir).__name__}.adjacency() の形が n_units と"
            f"一致しません: {matrix.shape} != {expected}"
        )
    return matrix


__all__ = [
    "build_reservoir",
    "require_esn",
    "require_graph",
    "reservoir_density",
]

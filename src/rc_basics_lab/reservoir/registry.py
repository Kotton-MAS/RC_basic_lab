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
from rc_basics_lab.reservoir.protocol import Reservoir, ReservoirConfig
from rc_basics_lab.reservoir.ring import RingConfig, RingReservoir


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


__all__ = ["build_reservoir", "require_esn"]

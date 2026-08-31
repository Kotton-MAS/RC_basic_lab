"""リザバーの生成口 — **``ESN(...)`` を直接呼ぶ場所を1つにする**.

``build_reservoir`` が唯一の生成口である。実験層はここだけを呼ぶので、
モデルを1つ足しても生成箇所 (実測で5箇所あった) を触らずに済む。

**既定は ESN のまま**なので、既存の YAML も既存の成果物も変わらない。
乱数の引き方も変えていない —— 合否判定は成果物のバイト不変である (D-74)。
"""

from __future__ import annotations

import numpy as np

from rc_basics_lab.reservoir.esn import ESN, ESNConfig
from rc_basics_lab.reservoir.protocol import Reservoir, ReservoirConfig


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


__all__ = ["build_reservoir"]

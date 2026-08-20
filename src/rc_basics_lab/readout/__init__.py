"""読み出し層 — 設計行列の構築とリッジ回帰.

3ベースライン (線形 / 遅延線 / リザバー) の違いは ``FeatureSpec`` の差だけで表現し、
学習・評価のコードは1本にする (受け入れ条件1)。リッジは全手法・全タスクが
``config.ridge.alpha_grid`` という単一の格子を読む (D-04)。

``autoregressive`` は自由走行 (closed-loop) の実行系で、**``reservoir`` を
import しない** (D-50)。状態更新器を ``StateUpdater`` プロトコルで受けるので、
ESN 以外の生成器 (外部シミュレータ・実素子) でも同じ関数がそのまま動く。
"""

from rc_basics_lab.readout.autoregressive import (
    FreeRunResult,
    StateUpdater,
    free_run,
)
from rc_basics_lab.readout.design import (
    BIAS_NAME,
    DelayLineSpec,
    DesignMatrix,
    FeatureSpec,
    PassthroughSpec,
    ReservoirSpec,
    bias_column_index,
    build_design_matrix,
)
from rc_basics_lab.readout.ridge import (
    AlphaSelection,
    fit_ridge,
    fit_ridge_from_gram,
    penalty_matrix,
    predict,
    select_alpha,
)

__all__ = [
    "BIAS_NAME",
    "AlphaSelection",
    "DelayLineSpec",
    "DesignMatrix",
    "FeatureSpec",
    "FreeRunResult",
    "PassthroughSpec",
    "ReservoirSpec",
    "StateUpdater",
    "bias_column_index",
    "build_design_matrix",
    "fit_ridge",
    "fit_ridge_from_gram",
    "free_run",
    "penalty_matrix",
    "predict",
    "select_alpha",
]

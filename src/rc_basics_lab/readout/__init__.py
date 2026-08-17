"""読み出し層 — 設計行列の構築とリッジ回帰.

3ベースライン (線形 / 遅延線 / リザバー) の違いは ``FeatureSpec`` の差だけで表現し、
学習・評価のコードは1本にする (受け入れ条件1)。リッジは全手法・全タスクが
``config.ridge.alpha_grid`` という単一の格子を読む (D-04)。
"""

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
    "PassthroughSpec",
    "ReservoirSpec",
    "bias_column_index",
    "build_design_matrix",
    "fit_ridge",
    "penalty_matrix",
    "predict",
    "select_alpha",
]

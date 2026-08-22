"""読み出し層 — 設計行列の構築とリッジ回帰.

3ベースライン (線形 / 遅延線 / リザバー) の違いは ``FeatureSpec`` の差だけで表現し、
学習・評価のコードは1本にする (受け入れ条件1)。リッジは全手法・全タスクが
``config.ridge.alpha_grid`` という単一の格子を読む (D-04)。

``autoregressive`` は自由走行 (closed-loop) の実行系で、**``reservoir`` を
import しない** (D-50)。
"""

from rc_basics_lab.readout import autoregressive, design, ridge

__all__ = [
    "autoregressive",
    "design",
    "ridge",
]

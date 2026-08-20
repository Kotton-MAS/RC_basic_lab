"""課題層 — 入力系列と目標系列を作る側.

このパッケージは手法 (``readout`` の ``FeatureSpec``) も ``reservoir`` も知らない。
課題は ``TaskData`` を返すだけで、誰がそれを解くかに関与しない。

- ``mackey_glass``: カオス時系列の ``horizon`` ステップ先予測
- ``delay_parity``: 遅延パリティ (線形手法が解析的に解けない対照課題。D-07)
- ``narma``: NARMA10 (実験 3-C の課題。D-29 / D-30)
- ``chaotic``: Lorenz の生成と 04 の標準化 (D-41。MG は ``mackey_glass`` へ委譲)
- ``anomaly``: 異常検知の系列の器・共通前処理・合成源 (D-57 / D-59。実データの
  取得と読み取りは ``datasets/`` にあり、この層は I/O を持たない)
"""

from rc_basics_lab.tasks import (
    anomaly,
    base,
    chaotic,
    delay_parity,
    mackey_glass,
    narma,
)

__all__ = [
    "anomaly",
    "base",
    "chaotic",
    "delay_parity",
    "mackey_glass",
    "narma",
]

"""課題層 — 入力系列と目標系列を作る側.

このパッケージは手法 (``readout`` の ``FeatureSpec``) も ``reservoir`` も知らない。
課題は ``TaskData`` を返すだけで、誰がそれを解くかに関与しない。

- ``mackey_glass``: カオス時系列の ``horizon`` ステップ先予測
- ``delay_parity``: 遅延パリティ (線形手法が解析的に解けない対照課題。D-07)
"""

from rc_basics_lab.tasks.base import TaskData, TaskGenerator
from rc_basics_lab.tasks.delay_parity import generate_delay_parity
from rc_basics_lab.tasks.mackey_glass import generate_mackey_glass

__all__ = [
    "TaskData",
    "TaskGenerator",
    "generate_delay_parity",
    "generate_mackey_glass",
]

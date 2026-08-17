"""プロジェクト共通の型エイリアス.

``npt.NDArray[np.float64]`` を各所で書き下すと表記揺れが起きるため、
ここを単一定義とする (仕様 T1)。
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
"""float64 の numpy 配列。形状は各 API のドキュメントで指定する。"""

__all__ = ["FloatArray"]

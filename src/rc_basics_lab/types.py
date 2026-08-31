"""プロジェクト共通の型エイリアス.

``npt.NDArray[np.float64]`` を各所で書き下すと表記揺れが起きるため、
ここを単一定義とする (仕様 T1)。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
"""float64 の numpy 配列。形状は各 API のドキュメントで指定する。"""

type BoolArray = npt.NDArray[np.bool_]
"""bool の numpy 配列。ラベル・マスク・予測に使う (形状は各 API に記す)。

05 の課題層 (``tasks/anomaly.py`` の ``AnomalySeries``) と指標層
(``metrics_detection.py``) の両方が要るので、``FloatArray`` と同じくここが
**定義の唯一の場所**である。``metrics_detection`` は同名の別名を自前で持って
いたが (構造的別名なので型検査上は等価)、名前を追う側から見ると「どちらの
BoolArray か」を確かめる手間が残っていた。統合済み。
"""

__all__ = ["BoolArray", "FloatArray"]

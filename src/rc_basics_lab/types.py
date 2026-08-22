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
(``metrics_detection.py``) の両方が要るので、``FloatArray`` と同じくここを
定義の場所にする。``metrics_detection`` は T1 の時点で同名の別名を自前で
持っており (公開名 ``metrics_detection.BoolArray``)、**T2 ではそちらに触れて
いない** —— 構造的別名なので型検査上は等価で、統合は次サイクルで行う
(``docs/plans/rc-basics-05.md`` §4 T2 の「T2 実装時に決めたこと」)。
"""

__all__ = ["BoolArray", "FloatArray"]

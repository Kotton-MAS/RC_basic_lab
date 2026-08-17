"""課題データの共通形式.

課題は「入力系列 ``u`` と目標系列 ``y`` を作るだけ」の層であり、手法 (線形 /
遅延線 / ESN) を一切知らない。手法の差は ``readout.FeatureSpec`` だけで表現する
(受け入れ条件1) ため、課題側に手法ごとの分岐が入り込まないよう
``TaskData`` という単一の戻り値型に固定する。

``u`` / ``y`` はともに ``(T, D)`` の2次元配列。1次元を受理しないのは診断層
(``diagnostics.validate_diagnostic_input``) と同じ規約で、``(T,)`` を ``(T, 1)``
と黙って解釈すると ``(1, T)`` の取り違えを検出できなくなるためである。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from rc_basics_lab.types import FloatArray


@dataclass(frozen=True, slots=True)
class TaskData:
    """課題1本ぶんの入出力系列。

    Attributes:
        u: 入力系列 ``(T, D_in)``。
        y: 目標系列 ``(T, D_out)``。``u`` と同じ行数で、行 index が時刻に対応する。
        name: 課題名。CSV の ``task`` 列になる。
        params: 生成パラメータ (文字列)。meta.json / CSV へそのまま流す。

    形状の検証は ``__post_init__`` で行う。設定 dataclass 群 (``config``) は純粋な
    データ保持に留めるが、こちらは生成結果の器であり、配線ミスを生成直後に
    落とせる場所がここしかない。
    """

    u: FloatArray
    y: FloatArray
    name: str
    params: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for array, label in ((self.u, "u"), (self.y, "y")):
            if array.ndim != 2:
                raise ValueError(
                    f"{label} は (T, D) の2次元配列が必要です: {array.shape}"
                )
            if array.shape[0] == 0 or array.shape[1] == 0:
                raise ValueError(f"{label} が空です: {array.shape}")
        if self.u.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"u と y の行数が一致しません: {self.u.shape[0]} != {self.y.shape[0]}"
            )
        if not np.all(np.isfinite(self.u)) or not np.all(np.isfinite(self.y)):
            raise ValueError(f"課題 {self.name} の系列に有限でない値があります")

    @property
    def n_steps(self) -> int:
        """系列長 T。"""
        return int(self.u.shape[0])

    @property
    def n_inputs(self) -> int:
        """入力次元 D_in。"""
        return int(self.u.shape[1])


class TaskGenerator[ConfigT](Protocol):
    """課題生成関数の呼び出し規約。

    ``rng`` は ``seeds.make_rng`` の **task ストリーム**を渡す (D-06)。
    リザバー生成 / 分割の乱数は決してここに混ぜない。
    """

    def __call__(self, cfg: ConfigT, rng: np.random.Generator) -> TaskData: ...


__all__ = ["TaskData", "TaskGenerator"]

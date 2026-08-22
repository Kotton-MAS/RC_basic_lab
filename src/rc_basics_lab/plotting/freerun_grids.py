"""実験 04 の図が読む「行 -> 点列」の復元と小さなヘルパ (D-96).

``figures_freerun.py`` は 600 行上限 (D-77) に達したので、**描画を含まない**
部分をここへ出す。入力は ``FreeRunProfileRow`` などの行で、出力は点列と
ラベル文字列だけである。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from rc_basics_lab.experiment.freerun import FreeRunProfileRow
from rc_basics_lab.plotting.capacity_grids import mean_std
from rc_basics_lab.plotting.style import StyleContext
from rc_basics_lab.types import FloatArray


class _HasTask(Protocol):
    """``task`` 列を持つ行 (課題の並びを取り出すためだけの構造的型)。"""

    @property
    def task(self) -> str: ...


def label_of(table: dict[str, tuple[str, str]], key: str, style: StyleContext) -> str:
    """対応表からラベルを引く。**未知のキーは描く前に落とす**。"""
    if key not in table:
        raise ValueError(f"ラベルの対応表にありません: {key!r}")
    japanese, english = table[key]
    return style.label(japanese, english)


def tasks_of(rows: Sequence[_HasTask]) -> list[str]:
    """行に現れる課題名 (出現順を保つ)。"""
    seen: list[str] = []
    for row in rows:
        if row.task not in seen:
            seen.append(row.task)
    return seen


def profile_points(
    rows: Sequence[FreeRunProfileRow], task: str, kind: str, source: str
) -> FloatArray:
    """長形式の行から ``(x, y)`` の点列を復元する (**図の唯一の入力経路**)。

    ``index`` の昇順に並べ替えるので、CSV の行順が変わっても描画順は変わらない。
    """
    selected = sorted(
        (
            row
            for row in rows
            if row.task == task and row.kind == kind and row.source == source
        ),
        key=lambda row: row.index,
    )
    if not selected:
        return np.empty((0, 2), dtype=np.float64)
    points: FloatArray = np.asarray(
        [(row.x, row.y) for row in selected], dtype=np.float64
    )
    return points


#: ``mean_std`` は ``capacity_grids`` の1本に寄せた (D-92)。
#: 04 側は空入力で ValueError を投げる別実装を持っていたが、その挙動に
#: 依存するテストは1件も無く、同じ数式が2箇所で独立に育つほうが害が大きい。
__all__ = [
    "label_of",
    "mean_std",
    "profile_points",
    "tasks_of",
]

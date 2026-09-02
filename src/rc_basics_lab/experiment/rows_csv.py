"""dataclass の行列を CSV に書く共通ヘルパ (D-126).

``report.py`` から切り出した。あちらは 01 の ``ResultRow`` と
``ExperimentConfig`` を知っている**連載側**だが、この書き出しは
「どの実験か」を知らない —— 汎用側が使えるようにここへ置く
(``diagnostics_rows`` が実際に使う)。
"""

from __future__ import annotations

import csv
import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def write_rows_csv[RowT: "DataclassInstance"](
    rows: Sequence[RowT], path: Path, columns: Sequence[str]
) -> Path:
    """dataclass の行列を CSV に書く (05 削減候補 #4 の共通ヘルパー)。

    9 箇所 (``report`` / ``esp_pipeline`` / ``capacity_pipeline`` / ``freerun`` /
    ``stability``) が個別実装していた「``mkdir`` -> ``DictWriter`` -> ``asdict`` ->
    ``writerow``」を1本にまとめる。9箇所とも ``encoding="utf-8"`` /
    ``newline=""`` / ``mkdir(parents=True, exist_ok=True)`` が同一であることを
    実装前に実測で確認済み (異なる箇所があれば呼び出し側に残す)。

    Args:
        rows: 書き出す行 (dataclass インスタンス)。
        path: 出力先。
        columns: 列順 (``XXX_CSV_COLUMNS`` 定数。単一の真実は呼び出し側のまま)。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))
    return path


__all__ = ["write_rows_csv"]

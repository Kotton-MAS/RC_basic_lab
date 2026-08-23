"""CSV 書き出しの共通経路 (非公開モジュール).

成果物の CSV は11本あり、どれも「親ディレクトリを作る -> ``newline=""`` で開く
-> ``DictWriter`` にヘッダを書かせる -> 行を書く」という同じ手順である。手順を
11回書くと、``newline=""`` の付け忘れのような**成果物が壊れてもテストが緑のまま
になる**類の間違いが11回ぶん入りうる (``newline`` を省くと Windows で行区切りが
``\\r\\r\\n`` になる)。

**列順の単一の真実は呼び出し側が持つ** (通常は行 dataclass の宣言順から作った
``*_CSV_COLUMNS``)。ここは列順を決めず、渡された ``columns`` をそのまま
``fieldnames`` にする。

非公開モジュール (先頭が ``_``) なので ``experiment/__init__.py`` からは公開せず、
``tests/test_public_api_reexport.py`` の対象外である。
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


def write_rows(
    rows: Iterable[Mapping[str, object]], columns: Sequence[str], path: Path
) -> Path:
    """行 (列名 -> 値の Mapping) の並びを ``path`` へ書く。

    Args:
        rows: 1行ぶんの Mapping の並び。行 dataclass からは
            ``(dataclasses.asdict(row) for row in rows)`` で作る。
        columns: 列順 (**呼び出し側が持つ単一の真実**)。
        path: 出力先。親ディレクトリが無ければ作る。

    Returns:
        ``path`` (呼び出し側が成果物のパスとして返せるように)。

    Raises:
        ValueError: ``columns`` に無いキーが行に含まれる場合
            (``csv.DictWriter`` が送出する)。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    return path


__all__ = ["write_rows"]

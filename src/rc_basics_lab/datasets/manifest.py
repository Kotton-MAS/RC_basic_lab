"""マニフェスト CSV の読み取り (出典・ライセンス・SHA256 の単一の真実).

``datasets/manifests/{mgab,ucr}.csv`` は**リポジトリにコミットする**。
データ本体は入れない (D-58) ので、ここが「何を取ってきて何と照合するか」の
正本になる。

書式は次の2段で、どちらも人が読める:

- 先頭の ``# key: value`` 行 —— 出典 URL・ライセンス・引用要求・取得日。
  ``README.md`` の「データセットのライセンスと取得手順」と**同じ文字列**を
  持ち、``tests/test_datasets_anomaly.py::
  test_readme_license_matches_the_manifests`` が一致を機械検査する
- 続く普通の CSV —— 1行1ファイル

設定ローダ (``config/_common.py``) は ``dict`` を受理しないため、SHA256 表を
YAML に置けない (仕様 §3)。CSV を選んだのはそのためで、YAML 側には
``dataset.series: tuple[str, ...]`` だけが載る。
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
"""マニフェスト CSV の置き場所 (パッケージ同梱)。"""

_COMMENT_PREFIX = "#"


@dataclass(frozen=True, slots=True)
class Manifest:
    """マニフェスト1本ぶん。

    Attributes:
        header: 先頭のコメント行 (``# key: value``) を集めたもの。
        rows: データ行 (列名 -> 値)。
        path: 読み込んだ CSV のパス。
    """

    header: Mapping[str, str]
    rows: tuple[Mapping[str, str], ...]
    path: Path

    @property
    def license(self) -> str:
        """``# license:`` の値。README と一致することをテストが要求する。"""
        return self.header["license"]

    def row(self, key: str, value: str) -> Mapping[str, str]:
        """``key`` 列が ``value`` の行を1つ返す (無ければ ``KeyError``)。"""
        for row in self.rows:
            if row[key] == value:
                return row
        raise KeyError(f"{self.path.name} に {key}={value!r} の行がありません")


def read_manifest(path: Path) -> Manifest:
    """マニフェスト CSV を読む (コメント行とデータ行を分ける)。"""
    header: dict[str, str] = {}
    data_lines: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.startswith(_COMMENT_PREFIX):
                key, separator, value = line[1:].partition(":")
                if separator:
                    header[key.strip()] = value.strip()
                continue
            data_lines.append(line)
    rows = tuple(dict(row) for row in csv.DictReader(data_lines))
    if not rows:
        raise ValueError(f"マニフェストにデータ行がありません: {path}")
    return Manifest(header=header, rows=rows, path=path)


__all__ = ["MANIFEST_DIR", "Manifest", "read_manifest"]

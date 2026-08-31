"""生成物とその指紋が行単位でマージされないことを見る (D-119).

## なぜこれが要るのか

``tests/artifact_manifest.csv`` と ``tests/golden/manifest.json`` は行指向である。
2本のブランチが**別々の実験**を再生成すると、変更行が離れているので git は
衝突を出さずに両方の行を採用する。実測:

    A: 02 の指紋だけ書き換え  ->  Auto-merging, Merge made by the 'ort' strategy
    B: 04 の指紋だけ書き換え      できあがった指紋は A の 02 行と B の 04 行を持つ

この指紋は**どちらのツリーとも一致しない**。``make ci`` は赤くなるので黙って
壊れはしないが、気づくのがマージの後になり、統合してから再生成をやり直す
ことになる (全5実験で約 15 分)。``-merge`` を付ければ衝突として即座に出る。

## この検査が測らないこと

「マージがどうなるか」を実際に走らせて確かめてはいない。**環境が覆い隠す**からで、
``.gitattributes`` をデータとして読み、指紋に載っている経路がすべて ``-merge``
の対象かどうかだけを見る。指紋ファイルを1つ増やした人は、ここで気づく。
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GITATTRIBUTES = REPO_ROOT / ".gitattributes"
MANIFEST = REPO_ROOT / "tests" / "artifact_manifest.csv"

# 指紋そのもの。中身が行指向なので、これらが混ざるのが一番痛い
FINGERPRINTS = ("tests/artifact_manifest.csv", "tests/golden/manifest.json")


def _no_merge_patterns() -> list[str]:
    """``.gitattributes`` で ``-merge`` が付いている経路パターンを返す。"""
    patterns: list[str] = []
    for line in GITATTRIBUTES.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pattern, *attributes = stripped.split()
        if "-merge" in attributes:
            patterns.append(pattern)
    return patterns


def _is_covered(path: str, patterns: list[str]) -> bool:
    """``path`` が ``-merge`` のいずれかのパターンに入るか。

    ここで扱うのは完全一致と ``dir/**`` の2種類だけである。git のパターン構文を
    再実装すると、その実装のほうが壊れる。
    """
    for pattern in patterns:
        if pattern == path:
            return True
        if pattern.endswith("/**") and path.startswith(pattern[:-2]):
            return True
    return False


def test_the_two_fingerprint_files_never_line_merge() -> None:
    patterns = _no_merge_patterns()
    missing = [p for p in FINGERPRINTS if not _is_covered(p, patterns)]
    assert not missing, (
        f".gitattributes に -merge が無い指紋ファイルがあります: {missing}\n"
        "行指向なので、別々の実験を再生成した2本のブランチが黙って混ざります。"
    )


def test_every_artifact_in_the_manifest_never_line_merges() -> None:
    patterns = _no_merge_patterns()
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        recorded = [row["path"] for row in csv.DictReader(handle)]
    assert recorded, "指紋が空です"
    missing = sorted({p for p in recorded if not _is_covered(p, patterns)})
    assert not missing, (
        f".gitattributes に -merge が無い成果物があります: {missing[:5]}"
        f" (全 {len(missing)} 件)\n"
        "成果物は再生成で作るものなので、行を混ぜて作ってはいけません。"
    )

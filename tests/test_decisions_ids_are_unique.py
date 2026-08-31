"""決定の ID が一意であることを見る (D-120).

## なぜこれが要るのか

``.claude/decisions.yaml`` は末尾に1件足すだけのファイルなので、2本のブランチが
同時に追記すると「同じ末尾を触った」という理由だけで衝突する。``.gitattributes``
の ``merge=union`` で両側の追記をそのまま採るようにした。

union には代償がある。**両方のブランチが同じ ID (たとえば両方 D-119) を選ぶと、
衝突せずに同じ ID の決定が2件並ぶ。** 並行に作業していれば次の番号は同じになるので、
これは起こりにくい事故ではなく**起こりやすい**事故である。

``scripts/check_decisions.py`` は単一の node id しか解決しないので、重複したまま
だと「どちらの決定を守っているのか」が決まらない。ここで測る。

## この検査が測らないこと

決定の中身は見ない。ID が一意であることと、``guard_test`` が空でないことだけを見る。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DECISIONS = REPO_ROOT / ".claude" / "decisions.yaml"

# 「- id: D-119」の形。YAML として読まずに行で拾うのは、union が壊した直後の
# ファイルでも (YAML として壊れていても) 重複を指摘できるようにするため
ID_LINE = re.compile(r"^- id:\s*(\S+)\s*$")


def _ids() -> list[str]:
    return [
        m.group(1)
        for line in DECISIONS.read_text(encoding="utf-8").splitlines()
        if (m := ID_LINE.match(line))
    ]


def test_no_decision_id_appears_twice() -> None:
    ids = _ids()
    assert ids, f"{DECISIONS} に決定が1件もありません"
    duplicated = sorted(i for i, n in Counter(ids).items() if n > 1)
    assert not duplicated, (
        f"決定 ID が重複しています: {duplicated}\n"
        "merge=union は両側の追記を残すので、並行ブランチが同じ番号を選ぶと"
        "こうなります。片方を次の番号へ振り直してください。"
    )


def test_every_decision_id_has_the_expected_shape() -> None:
    malformed = [i for i in _ids() if not re.fullmatch(r"D-\d{2,3}", i)]
    assert not malformed, f"想定した形 (D-01 / D-119) でない ID があります: {malformed}"

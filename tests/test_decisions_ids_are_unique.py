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

**union の代償はもう1つある。** 両側の ``- id: D-119`` 行は同一なので union は
それを1本に畳み、続く ``rule`` / ``rationale`` / ``guard_test`` だけを重ねる。
できあがるのは**キーが2組ある1件の決定**で、PyYAML はこれをエラーにせず
**後に現れたほうを黙って採る**。実測:

    - id: D-201
      rule: "B の決定"      <- B 側
      ...
      rule: "A の決定"      <- A 側。safe_load はこちらを返す

つまり片方の決定が**衝突もエラーも出さずに消える**。ID の一意性だけでは
これを捕まえられない (``- id:`` 行は1本しかない) ので、キーの重複そのものを見る。

## この検査が測らないこと

決定の中身は見ない。ID が一意であること、キーが重複していないこと、
ID の形だけを見る。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import cast

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DECISIONS = REPO_ROOT / ".claude" / "decisions.yaml"

# 「- id: D-119」の形。YAML として読まずに行で拾うのは、union が壊した直後の
# ファイルでも (YAML として壊れていても) 重複を指摘できるようにするため
ID_LINE = re.compile(r"^- id:\s*(\S+)\s*$")


def _sources() -> list[Path]:
    """決定が書かれているファイルを全部返す (保管庫 + 1件1ファイル。D-121)。"""
    directory = DECISIONS.parent / "decisions"
    files = sorted(directory.glob("*.decisions.yaml")) if directory.is_dir() else []
    return [DECISIONS, *files]


def _ids() -> list[str]:
    return [
        m.group(1)
        for path in _sources()
        for line in path.read_text(encoding="utf-8").splitlines()
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


class _RejectDuplicateKeys(yaml.SafeLoader):
    """同じマッピングにキーが2度現れたら落ちるローダ。

    PyYAML の既定は後勝ちで黙って読む。merge=union が作る壊れ方が
    まさにこれなので、ここだけは厳しくする。
    """


def _construct_mapping(
    loader: _RejectDuplicateKeys, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    seen: set[object] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            line = key_node.start_mark.line + 1
            raise ValueError(f"キー {key!r} が同じ決定の中に2度あります (行 {line})")
        seen.add(key)
    return cast(
        "dict[object, object]",
        yaml.SafeLoader.construct_mapping(loader, node, deep=deep),
    )


_RejectDuplicateKeys.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def test_no_decision_has_the_same_key_twice() -> None:
    for path in _sources():
        _reject_duplicate_keys(path)


def _reject_duplicate_keys(path: Path) -> None:
    try:
        yaml.load(path.read_text(encoding="utf-8"), Loader=_RejectDuplicateKeys)
    except ValueError as error:
        pytest.fail(
            f"{path.name}: {error}\n"
            "merge=union が同じ ID の追記を1件に畳んだ形です。"
            "PyYAML は後勝ちで読むので、片方の決定が黙って消えています。"
        )

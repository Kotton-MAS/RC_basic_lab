"""``CLAUDE.md`` が実在する決定だけを参照していることを固定する (D-76)。

``CLAUDE.md`` は**次のサイクルの Claude が最初に読む唯一の文書**である。
ここに書かれた規約が実体を失っても、読み手 (人でもモデルでも) には分からない。

実際にこのリポジトリでは、テンプレートのプレースホルダ (``このプロジェクトは _____ を
行う``) が未置換のまま8サイクル回り、numpy の数値実験リポジトリに1つも該当しない
セキュリティ節が約40行常駐していた。そのあいだ「学習メモ」は0行だった。
つまり**この文書は誰にも検査されていなかった**。

そこで最低限、次の2つを機械で固定する:

1. 参照している決定 ID が ``.claude/decisions.yaml`` に実在すること
2. テンプレートの残骸 (プレースホルダ) が残っていないこと
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
DECISIONS = REPO_ROOT / ".claude" / "decisions.yaml"

DECISION_REF = re.compile(r"\bD-\d+\b")
DECISION_DEF = re.compile(r"^- id:\s*(D-\d+)\s*$", re.MULTILINE)


def test_every_decision_referenced_by_claude_md_exists() -> None:
    """``CLAUDE.md`` の ``D-xx`` がすべて実在すること。

    実在しない ID を引くと、読み手は「根拠があるらしい」と受け取ったまま
    根拠に辿り着けない。空虚な参照は無い参照より悪い。
    """
    referenced = set(DECISION_REF.findall(CLAUDE_MD.read_text(encoding="utf-8")))
    defined = set(DECISION_DEF.findall(DECISIONS.read_text(encoding="utf-8")))
    missing = sorted(referenced - defined, key=lambda name: int(name[2:]))
    assert not missing, (
        f"CLAUDE.md が実在しない決定を参照しています: {missing}\n"
        "決定を消したか、ID を書き間違えています。"
    )
    assert referenced, (
        "CLAUDE.md が決定を1つも参照していません。"
        "規約が decisions.yaml と切り離されている可能性があります。"
    )


def test_claude_md_has_no_template_placeholders_left() -> None:
    """テンプレートの残骸が残っていないこと。

    プレースホルダが未置換のまま8サイクル回った実績があるので、機械で止める。
    """
    text = CLAUDE_MD.read_text(encoding="utf-8")
    leftovers = [marker for marker in ("_____", "TODO:", "<!-- TODO") if marker in text]
    assert not leftovers, f"CLAUDE.md にテンプレートの残骸があります: {leftovers}"

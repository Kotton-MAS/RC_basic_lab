"""``docs/型と名前の対応表.md`` に抽象名が全部載っていることの検査.

**対応表が古いと、載っていない名前だけ調べ方が分からない**という最悪の形に
なる (表があるので読者は表を信じ、無いものは「無い」と受け取る)。

公開 Protocol と型エイリアスは数が少ない (実測: 10 と 8) ので、全数を見る。
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "rc_basics_lab"
GLOSSARY = ROOT / "docs" / "型と名前の対応表.md"


def _public_abstractions() -> tuple[set[str], set[str]]:
    """``(型エイリアス, 公開 Protocol)`` の名前。``_`` 始まりは除く。"""
    aliases: set[str] = set()
    protocols: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.TypeAlias):
                name = ast.unparse(node.name)
                if not name.startswith("_"):
                    aliases.add(name)
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                if "Protocol" in {ast.unparse(base) for base in node.bases}:
                    protocols.add(node.name)
    return aliases, protocols


def test_every_type_alias_is_in_the_glossary() -> None:
    """型エイリアスが全部載っている。"""
    aliases, _ = _public_abstractions()
    text = GLOSSARY.read_text(encoding="utf-8")
    missing = sorted(name for name in aliases if f"`{name}`" not in text)
    assert not missing, (
        f"対応表に無い型エイリアス: {missing}\n"
        "docs/型と名前の対応表.md に1行足してください。"
    )


def test_every_public_protocol_is_in_the_glossary() -> None:
    """公開 Protocol が全部載っている。

    Protocol は「満たすべき面」なので、足したのに表に無いと、実装しようと
    する人が面の存在に気づけない。
    """
    _, protocols = _public_abstractions()
    text = GLOSSARY.read_text(encoding="utf-8")
    missing = sorted(name for name in protocols if f"`{name}`" not in text)
    assert not missing, (
        f"対応表に無い Protocol: {missing}\n"
        "docs/型と名前の対応表.md に1行足してください。"
    )


def test_the_glossary_has_no_stale_entries() -> None:
    """表にあってコードに無い名前が残っていない。

    消したのに表に残ると、読者は存在しないものを探す。
    """
    aliases, protocols = _public_abstractions()
    known = aliases | protocols
    text = GLOSSARY.read_text(encoding="utf-8")
    # 表の1列目 (| `名前` |) だけを見る。本文中の言及は対象外。
    listed = {
        line.split("`")[1]
        for line in text.splitlines()
        if line.startswith("| `") and line.count("`") >= 2
    }
    stale = sorted(name for name in listed if name in _TRACKED and name not in known)
    assert not stale, f"コードに無い名前が表に残っています: {stale}"


_TRACKED = frozenset(
    {
        "FloatArray",
        "BoolArray",
        "FeatureSpec",
        "ReservoirConfig",
        "TargetSpec",
        "ScoreSpec",
        "Opener",
        "Reservoir",
        "Diagnostic",
        "StatePropagator",
        "StateUpdater",
        "TaskGenerator",
        "SeriesSource",
        "HttpResponse",
        "HasReplicate",
    }
)
"""表が追跡している名前。

「紛らわしい対」の節には ``run`` / ``step`` のようなメソッド名も出るので、
古い項目の検査はこの集合の中だけで行う (メソッド名まで見ると、表の書き方に
検査が縛られる)。
"""

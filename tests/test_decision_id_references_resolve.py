"""コード中の決定 ID (``D-NNN``) 参照が実在することを検証する回帰テスト.

``src/`` と ``tests/`` の docstring / コメントは、設計判断を ``(D-37)`` の
ような ID 参照だけ残して本文を ``.claude/decisions/`` へ逃がしている
(CLAUDE.md の「散文が3箇所にあると3箇所が独立にドリフトする」)。参照先が
実在しなければ、**コードには根拠があるように見えて根拠が無い**状態になる。

実際に起きた: ``D-136`` は ``tests/test_experiment_esp.py`` の2箇所から
実測値 (λ = -0.01395 = 境界の 1.4 倍) つきで参照されていたのに、
``.claude/decisions/`` にも ``decisions.yaml`` にも存在しなかった
(D-135 の次が D-138)。**8サイクル気づかれなかった。**

``tests/test_finding_id_references_resolve.py`` が F-ID について同じことを
していたが、D-ID 側は塞がれていなかった。D-120 が塞いだのは「union マージで
決定が消える」経路であって、「**書いたつもりで書いていない**」経路ではない。

参照先は2つある (D-121 で保管庫を分けた):

- ``.claude/decisions.yaml``: D-120 までの保管庫。**追記しない**
- ``.claude/decisions/D-NNN.decisions.yaml``: これから足す決定。1件1ファイル
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DECISION_ID = re.compile(r"\bD-(\d{2,3})\b")
"""コード中の決定 ID。``D-37`` / ``D-136`` の両方を拾う。"""

SEARCH_ROOTS = ("src", "tests")
"""参照を探す範囲。``docs/`` は散文なので対象外 (未来の決定を論じてよい)。"""

LEGACY_STORE = ROOT / ".claude" / "decisions.yaml"
"""D-120 までの保管庫 (D-121)。"""

DECISION_DIR = ROOT / ".claude" / "decisions"
"""D-121 以降の1件1ファイルの置き場。"""


def _recorded_ids() -> set[str]:
    """記録済みの決定 ID の全件。"""
    found: set[str] = set()
    if LEGACY_STORE.is_file():
        found |= {
            f"D-{number}"
            for number in DECISION_ID.findall(LEGACY_STORE.read_text(encoding="utf-8"))
        }
    for path in DECISION_DIR.glob("*decisions.yaml"):
        text = path.read_text(encoding="utf-8")
        found |= {
            f"D-{number}" for number in DECISION_ID.findall(text.split("rationale:")[0])
        }
    return found


def _referenced_ids() -> dict[str, list[str]]:
    """コードが参照している決定 ID -> 参照元 (リポジトリ相対パス:行)。"""
    found: dict[str, list[str]] = {}
    for root in SEARCH_ROOTS:
        for path in sorted((ROOT / root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if path.name == Path(__file__).name:
                continue  # このテスト自身の説明文は参照ではない
            for number, line in _matches(path):
                key = f"D-{number}"
                found.setdefault(key, []).append(line)
    return found


def _matches(path: Path) -> list[tuple[str, str]]:
    """1ファイルの (ID の数字, 参照元の表示) を返す。"""
    relative = path.relative_to(ROOT).as_posix()
    hits: list[tuple[str, str]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        hits.extend(
            (number, f"{relative}:{index}") for number in DECISION_ID.findall(line)
        )
    return hits


def test_every_decision_id_in_the_code_is_recorded() -> None:
    """コードが参照する決定 ID が**全部記録されている**。

    変異注入: ``.claude/decisions/D-136.decisions.yaml`` を消すと赤くなる。
    """
    recorded = _recorded_ids()
    assert recorded, "決定が1件も読めません (置き場のパスが変わっていませんか)"
    dangling = {
        key: sources
        for key, sources in sorted(_referenced_ids().items())
        if key not in recorded
    }
    assert not dangling, (
        "記録の無い決定 ID がコードから参照されています:\n"
        + "\n".join(
            f"  {key}: {', '.join(sources)}" for key, sources in dangling.items()
        )
        + "\n.claude/decisions/<ID>.decisions.yaml に guard_test つきで書いてください"
        " (D-121)。"
    )


def test_the_new_decisions_live_in_one_file_each() -> None:
    """D-121 以降の決定が**1件1ファイル**である。

    ファイル名と中身の ID が一致しないと、``check-decisions.sh`` が
    ``*decisions.yaml`` で拾う対象と参照解決の対象がずれる。
    """
    for path in sorted(DECISION_DIR.glob("*.yaml")):
        assert path.name.endswith(".decisions.yaml"), (
            f"{path.name} は *.decisions.yaml でないので"
            " check-decisions.sh の対象から静かに外れます (D-121)"
        )
        stem = path.name.split(".")[0]
        head = path.read_text(encoding="utf-8").split("rationale:")[0]
        assert f"id: {stem}" in head, (
            f"{path.name} の中身の id がファイル名 ({stem}) と違います"
        )

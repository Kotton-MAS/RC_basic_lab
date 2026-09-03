"""手引きが名指しするファイルが実在することの検査 (D-149).

``docs/guide/`` は「こう書けば動く」を示す文書なので、**名指ししたパスが
無ければ手引きそのものが嘘になる**。実際に起きた: ``条件を変えて試す.md`` が
``experiments/01_what_is_rc/presets/scalefree.yaml`` を2箇所で名指しし、
うち1つは ``uv run python main.py --preset <そのパス>`` という実行例だったが、
ファイルは存在しなかった。**読者がコピーして貼ると落ちる。**

``tests/test_finding_id_references_resolve.py`` (F-ID) と
``tests/test_decision_id_references_resolve.py`` (D-ID) と同じ形で、
**参照の解決**をもう1種類ぶん足したものである。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUIDE_DIR = ROOT / "docs" / "guide"

REPO_PATH = re.compile(
    r"(?<![\w/.-])((?:experiments|src|tests|scripts|results|docs)/[\w./-]+"
    r"\.(?:yaml|yml|py|md|csv|png|json|toml))"
)
"""手引きの中のリポジトリ相対パス。

拡張子を持つものだけを見る —— ディレクトリの言及 (``docs/guide/``) は
「そこにある」以上の約束をしていない。
"""

GLOB_MARKERS = ("*", "<", ">", "{", "}")
"""雛形を表す記号。``figures_*.py`` のような**書き方の説明**は対象外。"""


def _referenced_paths() -> dict[str, list[str]]:
    """手引きが名指しするパス -> 参照元 (相対パス:行)。"""
    found: dict[str, list[str]] = {}
    for path in sorted(GUIDE_DIR.glob("*.md")):
        relative = path.relative_to(ROOT).as_posix()
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for hit in REPO_PATH.findall(line):
                if any(marker in hit for marker in GLOB_MARKERS):
                    continue
                found.setdefault(hit, []).append(f"{relative}:{index}")
    return found


def test_every_path_named_by_a_guide_exists() -> None:
    """手引きが名指しするファイルが実在する。

    変異注入: ``experiments/01_what_is_rc/presets/scalefree.yaml`` を消すと
    赤くなる。
    """
    referenced = _referenced_paths()
    assert referenced, "手引きが1つもパスを名指ししていません (正規表現を疑う)"
    missing = {
        target: sources
        for target, sources in sorted(referenced.items())
        if not (ROOT / target).exists()
    }
    assert not missing, (
        "手引きが名指しするファイルがありません:\n"
        + "\n".join(
            f"  {target}: {', '.join(sources)}" for target, sources in missing.items()
        )
        + "\n読者がコピーして貼ると落ちます。作るか、参照を直してください。"
    )

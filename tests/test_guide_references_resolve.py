"""文書とコードが名指しするファイルが実在することの検査 (D-149 / D-150).

``docs/guide/`` は「こう書けば動く」を示す文書なので、**名指ししたパスが
無ければ手引きそのものが嘘になる**。実際に起きた: ``条件を変えて試す.md`` が
``experiments/01_what_is_rc/presets/scalefree.yaml`` を2箇所で名指しし、
うち1つは ``uv run python main.py --preset <そのパス>`` という実行例だったが、
ファイルは存在しなかった。**読者がコピーして貼ると落ちる。**

``tests/test_finding_id_references_resolve.py`` (F-ID) と
``tests/test_decision_id_references_resolve.py`` (D-ID) と同じ形で、
**参照の解決**をもう1種類ぶん足したものである。

**コード側の docstring も同じ問題を起こす。** 文書を畳んだり移したりしたとき、
``(docs/... を参照)`` と書いた docstring は静かに宙に浮く。実測で
``tests/test_anomaly_dataset_source.py`` が checkpoint-05b-t3.md を
``docs/plans/`` の下だと書いていたが、実体は ``docs/process/`` にあった
(役割別の再編で移した際の直し漏れ)。こちらも見る。

ただし**「かつてここにあった」という履歴の言及は正当**なので、
``HISTORICAL_MENTIONS`` に理由つきで並べて外す。外した以上、それが本当に
履歴であることは人が読んで確かめる —— 表に足す操作自体がレビューに出る。
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


HISTORICAL_MENTIONS: dict[str, str] = {
    "docs/削減候補-05.md": "docs/ の役割別再編で docs/process/ へ移した経緯の説明",
    "src/rc_basics_lab/config.py": (
        "config が単一モジュールだった頃の説明 (いまは package)"
    ),
}
"""**もう存在しないことを説明している**参照。実在を求めない (D-150)。

値は「なぜ存在しないのが正しいか」。空文字を許さないのは、外す判断に理由を
書かせるためである。
"""

CODE_ROOTS = ("src", "tests", "scripts")
"""コード側で参照を探す範囲。"""


def _paths_named_by_code() -> dict[str, list[str]]:
    """コードの docstring / コメントが名指しするパス -> 参照元。"""
    found: dict[str, list[str]] = {}
    for base in CODE_ROOTS:
        for path in sorted((ROOT / base).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            lines = path.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines, 1):
                for hit in REPO_PATH.findall(line):
                    if any(marker in hit for marker in GLOB_MARKERS):
                        continue
                    found.setdefault(hit, []).append(f"{relative}:{index}")
    return found


def test_every_historical_mention_has_a_reason() -> None:
    """例外表の全件に理由が書いてある (**外す判断を空欄にしない**)。"""
    blank = sorted(key for key, why in HISTORICAL_MENTIONS.items() if not why.strip())
    assert not blank, f"理由の無い例外があります: {blank}"
    stale = sorted(key for key in HISTORICAL_MENTIONS if (ROOT / key).exists())
    assert not stale, (
        f"実在するのに例外表に載っています (表から外してください): {stale}"
    )


def test_every_path_named_by_the_code_exists() -> None:
    """コードが名指しするファイルが実在する (D-150)。

    変異注入: ``tests/test_anomaly_dataset_source.py`` が引く
    checkpoint-05b-t3.md の置き場を ``docs/plans/`` に戻すと赤くなる。
    """
    missing = {
        target: sources
        for target, sources in sorted(_paths_named_by_code().items())
        if target not in HISTORICAL_MENTIONS and not (ROOT / target).exists()
    }
    assert not missing, (
        "コードが名指しするファイルがありません:\n"
        + "\n".join(
            f"  {target}: {', '.join(sources)}" for target, sources in missing.items()
        )
        + "\n移したなら参照を直し、消したなら HISTORICAL_MENTIONS へ理由つきで。"
    )


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

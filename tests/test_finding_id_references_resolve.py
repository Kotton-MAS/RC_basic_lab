"""コード中の finding ID 参照が壊れていないことを検証する回帰テスト.

``src/`` と ``tests/`` の docstring / コメントは、レビューの finding ID
(``F-<round>-<番号>`` 形式、例: ``F-1-002``) を「なぜこうなっているか」の
根拠として引用している。一時的な集約ファイル
(``.claude/tmp/findings/round-N/*.json`` 等) は gitignore 対象で削除されうる
ため、参照先は git 追跡下の恒久的な記録文書 ``docs/review-findings-01.md``
でなければ、その文書が消えた瞬間に参照が宙に浮く。

このテストは、ソース中に出現する全 finding ID を正規表現で抽出し、その全て
が記録文書に実在することを assert する。これにより「一時的な finding ID を
docstring に書いたが、記録文書への追記を忘れる」欠陥を機械的に検出する
(過去に一時 ID ``F-1-017`` が記録文書に存在しないまま docstring に残り、
round 2 の reviewer 2名が独立に指摘した実例がある)。

自己参照回避: このテストファイル自身はスキャン対象から除外する。含めると
このファイルに書かれた finding ID の例示 (docstring 中の ``F-1-002`` など)
が誤って「参照」として抽出され、自己参照による偽陽性/偽陰性を生みうる。
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORD_DOC = REPO_ROOT / "docs" / "review-findings-01.md"
SCAN_ROOTS = (REPO_ROOT / "src", REPO_ROOT / "tests")

# パターンは文字列連結で組み立てる。このテストファイル自身をスキャン対象
# から除外しているため必須ではないが、以下の理由でも都合がよい:
# 文字列リテラルとして "F-1-002" 等をこのファイルに書かないことで、
# 将来このファイルの内容を grep ベースの別ツールでスキャンする場合にも
# 自己参照による誤検出を避けられる。
_ID_PATTERN = re.compile("F-" + r"[0-9]+" + "-" + r"[0-9]{3}")

THIS_FILE = Path(__file__).resolve()


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        files.extend(sorted(root.rglob("*.py")))
    return [path for path in files if path.resolve() != THIS_FILE]


def _extract_ids(text: str) -> set[str]:
    return set(_ID_PATTERN.findall(text))


def _referenced_ids() -> set[str]:
    ids: set[str] = set()
    for path in _iter_python_files():
        ids |= _extract_ids(path.read_text(encoding="utf-8"))
    return ids


def _recorded_ids() -> set[str]:
    assert RECORD_DOC.exists(), f"記録文書が見つかりません: {RECORD_DOC}"
    return _extract_ids(RECORD_DOC.read_text(encoding="utf-8"))


def test_finding_id_references_resolve_to_record_doc() -> None:
    """``src/`` / ``tests/`` の finding ID 参照が全て記録文書に実在すること.

    解決できない ID が1つでもあれば、``docs/review-findings-01.md`` への
    追記漏れを意味するので落とす。一時的なレビュー集約 (``.claude/tmp/``
    配下) にしか存在しない ID を docstring に書くと、ここで検出される。
    """
    referenced = _referenced_ids()
    assert referenced, (
        "コード中に finding ID 参照が1件も見つかりませんでした。"
        "_ID_PATTERN またはスキャン対象ディレクトリが壊れていないか確認して"
        "ください (このテストは1件以上の参照が存在する前提で書かれています)"
    )

    recorded = _recorded_ids()
    unresolved = sorted(referenced - recorded)
    assert not unresolved, (
        "以下の finding ID がコードから参照されていますが、"
        f"{RECORD_DOC.relative_to(REPO_ROOT)} に見つかりません: {unresolved}\n"
        "一時的なレビュー findings (.claude/tmp/ 配下) を参照している場合は、"
        f"{RECORD_DOC.name} に該当エントリを追記してください。"
    )

"""コード中の finding ID 参照が壊れていないことを検証する回帰テスト.

``src/`` と ``tests/`` の docstring / コメントは、レビューの finding ID を
「なぜこうなっているか」の根拠として引用している。ID には2形式ある:

- レガシー形式 ``F-<round>-<番号>`` (例: ``F-1-002``)。サイクルという概念が
  導入される前 (サイクル1) に振られた ID で、``docs/review-findings-01.md``
  に記録されている。
- サイクル修飾形式 ``F-<cycle>-<round>-<番号>`` (例: ``F-02-1-004``)。
  サイクル2以降の ID で、``docs/review-findings-<cycle>.md`` に記録される。

サイクル修飾形式を導入したのは、レガシー形式がサイクルをまたいで衝突した
実例があるため: サイクル 2a の docstring が「今回の finding」のつもりで
書いた ``F-1-004`` は、``review-findings-01.md`` の ``F-1-004``
(``main.py`` の ``EXPERIMENTS`` docstring の件、全く無関係な内容) と
番号が衝突していた。旧版のこのテストは「ID が記録文書に**存在するか**」
しか見ておらず「**正しい記録文書を指しているか**」を見ていなかったため、
この衝突を検出できなかった (偽の緑)。

一時的な集約ファイル (``.claude/tmp/findings/round-N/*.json`` 等) は
gitignore 対象で削除されうるため、参照先は git 追跡下の恒久的な記録文書
でなければ、その文書が消えた瞬間に参照が宙に浮く。

このテストは、ソース中に出現する全 finding ID を正規表現で抽出し、ID の
接頭辞から一意に定まる記録文書にその ID が実在することを assert する。
これにより「一時的な finding ID を docstring に書いたが、記録文書への
追記を忘れる」欠陥に加え、「別サイクルの記録文書と衝突する ID を振って
しまう」欠陥も機械的に検出する
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
DOCS_DIR = REPO_ROOT / "docs"
LEGACY_RECORD_DOC = DOCS_DIR / "review-findings-01.md"
SCAN_ROOTS = (REPO_ROOT / "src", REPO_ROOT / "tests")

# パターンは文字列連結で組み立てる。このテストファイル自身をスキャン対象
# から除外しているため必須ではないが、以下の理由でも都合がよい:
# 文字列リテラルとして "F-1-002" 等をこのファイルに書かないことで、
# 将来このファイルの内容を grep ベースの別ツールでスキャンする場合にも
# 自己参照による誤検出を避けられる。
#
# サイクル修飾形式 (F-<cycle>-<round>-<seq>, 例: F-02-1-004) を先に試す。
# 先にレガシー形式 (F-<round>-<seq>) を試すと、"F-02-1-004" のような文字列
# から末尾3セグメントだけを切り出す部分一致は起きない (レガシー形式は
# "F-" の直後が丸ごと <round> であることを要求するため、"02-1" という
# round 部分は \d+ 単体にマッチせず、この順序でなくても事故は起きないが、
# 意図を明示するため長い (より具体的な) 形式を先に置く)。
#
# サイクル番号セグメントは元々 [0-9]+ (数字のみ) だったが、サイクル 3b-1 の
# 一時集約フォルダ (``.claude/tmp/findings/3b1-round-1/``) が振った ID
# ``F-3b1-1-NNN`` の "3b1" は数字だけでは表せない (F-3b1-1-001 の
# fixer/round1 で発見)。英数字混在のサイクルラベルを許すよう [0-9A-Za-z]+
# に広げた (数字のみだった既存サイクル ("01" 等) の解決には影響しない)。
_QUALIFIED_ID_PATTERN = re.compile(
    "F-" + r"[0-9A-Za-z]+" + "-" + r"[0-9]+" + "-" + r"[0-9]{3}"
)
_LEGACY_ID_PATTERN = re.compile("F-" + r"[0-9]+" + "-" + r"[0-9]{3}")
_ID_PATTERN = re.compile(
    f"{_QUALIFIED_ID_PATTERN.pattern}|{_LEGACY_ID_PATTERN.pattern}"
)

_CYCLE_LABEL_TO_DOC_SUFFIX = {
    # サイクル 3b-1 の一時集約フォルダ (``.claude/tmp/findings/3b1-round-1/``)
    # は ID のサイクルセグメントに "3b1" を使ったが、記録文書の命名規則は
    # 01/02/03 に続けて "03b" (docs/plans/rc-basics-03b.md と同じ) を使う。
    # 一時フォルダの命名 (レビュー実行時に機械的に決まる) と永続的な文書
    # 命名規則 (01/02/03 の続き) が異なる、という食い違いを吸収するための
    # 別名テーブル。両方を "3b1" に揃える (記録文書を review-findings-3b1.md
    # にリネームする) 選択肢もあったが、既存の 01/02/03 の連番規則から外れる
    # ため、ID 側はそのまま (fixer-input.json / triage.json との対応を保つ)
    # にし、文書名だけをここで正規化する方を選んだ。
    "3b1": "03b",
}

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


def _record_doc_for_id(finding_id: str) -> Path:
    """finding ID の接頭辞から、対応する記録文書のパスを返す.

    サイクル修飾形式 (``F-02-1-004``) はサイクル番号セグメント (``02``) を
    そのまま ``review-findings-<cycle>.md`` に対応付ける。``_CYCLE_LABEL_TO_DOC_SUFFIX``
    に載っているサイクルラベル (例: ``3b1``) は、一時集約フォルダの命名と
    記録文書の命名規則が食い違うため、対応する文書名へ変換してから対応付ける。
    レガシー形式 (``F-1-018`` のようにサイクル番号セグメントを持たない) は、
    サイクルという概念が導入される前の ID なのでサイクル1の記録文書
    (``review-findings-01.md``) に対応付ける。
    """
    if _QUALIFIED_ID_PATTERN.fullmatch(finding_id):
        cycle = finding_id.split("-")[1]
        suffix = _CYCLE_LABEL_TO_DOC_SUFFIX.get(cycle, cycle)
        return DOCS_DIR / f"review-findings-{suffix}.md"
    return LEGACY_RECORD_DOC


def _recorded_ids(doc: Path) -> set[str]:
    assert doc.exists(), f"記録文書が見つかりません: {doc}"
    return _extract_ids(doc.read_text(encoding="utf-8"))


def test_finding_id_references_resolve_to_record_doc() -> None:
    """``src/`` / ``tests/`` の finding ID 参照が、接頭辞に対応する記録文書に実在する.

    キー集合の一致 (旧実装) だけでは「別サイクルの記録文書に同じ番号の
    別内容が偶然存在する」衝突を見逃す。ID の接頭辞から対応する記録文書を
    一意に決め、その文書だけを見て解決を判定することで、衝突していれば
    (=期待した記録文書に実在しなければ) 確実に落ちる。
    """
    referenced = _referenced_ids()
    assert referenced, (
        "コード中に finding ID 参照が1件も見つかりませんでした。"
        "_ID_PATTERN またはスキャン対象ディレクトリが壊れていないか確認して"
        "ください (このテストは1件以上の参照が存在する前提で書かれています)"
    )

    doc_cache: dict[Path, set[str]] = {}
    unresolved: list[str] = []
    for finding_id in sorted(referenced):
        doc = _record_doc_for_id(finding_id)
        if doc not in doc_cache:
            doc_cache[doc] = _recorded_ids(doc)
        if finding_id not in doc_cache[doc]:
            unresolved.append(f"{finding_id} (期待される記録文書: {doc.name})")

    assert not unresolved, (
        "以下の finding ID がコードから参照されていますが、期待される記録文書に"
        f"見つかりません: {unresolved}\n"
        "一時的なレビュー findings (.claude/tmp/ 配下) を参照している場合は、"
        "接頭辞に対応する review-findings-XX.md に該当エントリを追記してください。"
    )

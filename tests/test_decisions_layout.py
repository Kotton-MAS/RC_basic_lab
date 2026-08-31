"""決定の置き場所を見る (D-121).

## なぜこれが要るのか

決定を1件足すたびに ``.claude/decisions.yaml`` の末尾が変わるので、並行する
2本のブランチが**必ず**衝突していた。D-120 で ``merge=union`` を入れて手元では
解決したが、**GitHub のマージはこれを見ない**。実測:

===========================  =========================
同じ base と head で         判定
===========================  =========================
手元 (.gitattributes あり)   衝突 0 件
GitHub のマージボタン        CONFLICTING
手元 (.gitattributes を退避) ``.claude/decisions.yaml`` が1件衝突
===========================  =========================

3行目が GitHub の判定と一致する。つまり union は「実際にマージする場所」では
効いていない。さらに git は**作業ツリー側**の ``.gitattributes`` を見るので、
それを含まないブランチの上でマージすると手元でも効かない (#23 で実際に起きた)。

1件1ファイルなら、並行ブランチは別々のファイルを作る。**マージの仕方に依存せず、
触る場所を分けることで衝突が消える。**

## なぜ保管庫を分割しないのか

119 件を動かすと #23 / #24 と新たに衝突を作る。**衝突を無くす変更が衝突を
作っては本末転倒である。** 保管庫は D-120 までを持つ器として据え置き、
既存の決定の改訂だけを許す。

## この検査が測らないこと

決定の中身は見ない (それはキットの ``check_decisions.py`` の仕事)。
置き場所と ID の連続性だけを見る。
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = REPO_ROOT / ".claude" / "decisions.yaml"
DECISIONS_DIR = REPO_ROOT / ".claude" / "decisions"

# 保管庫が持てる最後の決定。**この値を上げて通してはいけない** —— 上げた瞬間に
# 「末尾に追記する」経路が戻り、並行ブランチの衝突も戻る
LAST_ARCHIVED = 120

ID_LINE = re.compile(r"^- id:\s*(D-(\d+))\s*$")
FILE_NAME = re.compile(r"^(D-\d+)\.decisions\.yaml$")


def archived_ids() -> list[tuple[str, int]]:
    """保管庫に入っている決定の (ID, 番号) を出現順に返す。"""
    return [
        (m.group(1), int(m.group(2)))
        for line in ARCHIVE.read_text(encoding="utf-8").splitlines()
        if (m := ID_LINE.match(line))
    ]


def file_decisions() -> dict[str, Path]:
    """``.claude/decisions/`` に置かれた決定を ID -> パスで返す。"""
    if not DECISIONS_DIR.is_dir():
        return {}
    found: dict[str, Path] = {}
    for path in sorted(DECISIONS_DIR.iterdir()):
        if path.name.startswith(".") or path.is_dir():
            continue
        match = FILE_NAME.match(path.name)
        assert match, (
            f"{path.name} はキットの check-decisions.sh が拾える名前ではありません。\n"
            "D-NNN.decisions.yaml にしてください (*decisions.yaml で対象を選ぶため、"
            "D-NNN.yaml だと編集時のスキーマ検証が静かに外れます)。"
        )
        found[match.group(1)] = path
    return found


def test_new_decisions_are_not_appended_to_the_archive() -> None:
    too_new = sorted(i for i, n in archived_ids() if n > LAST_ARCHIVED)
    assert not too_new, (
        f"D-{LAST_ARCHIVED} より後の決定が保管庫に追記されています: {too_new}\n"
        f".claude/decisions/<ID>.decisions.yaml へ1件1ファイルで移してください。\n"
        "末尾に追記すると、並行するブランチが GitHub 上で必ず衝突します (D-121)。"
    )


def test_the_archive_still_holds_the_old_decisions() -> None:
    """保管庫を空にして「追記が無い」を満たす、という抜け道を塞ぐ。"""
    ids = archived_ids()
    assert len(ids) >= 100, (
        f"保管庫の決定が {len(ids)} 件しかありません。"
        "分割するなら D-121 を見直すのが先です。"
    )


def test_no_decision_id_is_defined_in_both_places() -> None:
    archived = {i for i, _ in archived_ids()}
    duplicated = sorted(archived & set(file_decisions()))
    assert not duplicated, (
        f"保管庫とファイルの両方にある決定: {duplicated}\n"
        "どちらが正本か決まりません。ファイル側に寄せて保管庫から消してください。"
    )

"""サイクル完了手続きのうち「飛ばしても何も起きない」ものを機械で見る (D-93).

## なぜこれが要るのか

`CLAUDE.md` の「サイクル完了時にやること」は4手順ある。3巡の実測で、
**実行されたのは結果が目に見えるもの、飛んだのは飛ばしても何も起きないもの**
という分かれ方をした。

| 手順 | 結果が目に見えるか | 実測 |
|---|---|---|
| 1. ``make ci`` | 赤になる | 実行された |
| 2. ``reviewer-deletion`` | 何も起きない | **飛んだ** (重複が4件増えた。D-92) |
| 3. タグを打つ | ログに出る | 実行された |
| 4. ``git worktree prune`` | 何も起きない | **飛んだ** (595MB のまま) |

文言の強さの問題ではない。手順2 は D-92 のラチェットが結果側で受け止める
ようにした。ここが受け持つのは手順4 と、成果物の生成版がそろっているか
(第3版 指摘 B-5) である。

## この検査が測らないこと

「手順を実行したか」ではなく「**実行していれば成り立つ状態か**」を測る。
実行の有無そのものを測ろうとすると、実行の記録を残す手順がまた1つ増え、
その手順が飛ぶ。状態で判定すれば、どの経路で直したかを問わずに済む。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"


def _git(*args: str) -> str:
    """リポジトリ内で git を走らせて標準出力を返す。"""
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"git を実行できません: {' '.join(args)}")
    return completed.stdout


def test_no_worktree_is_left_prunable() -> None:
    """使い終わった worktree が残っていないこと (完了手順4)。

    実測では ``.claude/worktrees/`` が 589MB -> 595MB と積み上がり、内訳の
    大半は gitignore 済みで再生成可能なもの (``.venv`` 297M + ``data/`` 205M)
    だった。コミットはされないのでレビューには出てこず、**ディスクだけを
    食い続ける**。だから成果物ではなく状態として見る。

    直し方は ``git worktree prune``。作業中の worktree は prunable にならない
    ので、この検査が作業を止めることはない。
    """
    porcelain = _git("worktree", "list", "--porcelain")
    prunable: list[str] = []
    current: str | None = None
    for line in porcelain.splitlines():
        if line.startswith("worktree "):
            current = line.removeprefix("worktree ").strip()
        elif line.startswith("prunable") and current is not None:
            prunable.append(current)
    assert not prunable, (
        f"使い終わった worktree が残っています: {prunable}\n"
        "`git worktree prune` を実行してください (CLAUDE.md の完了手順4)。"
    )


def _meta_commits() -> dict[str, str]:
    """``results/`` 配下の ``meta.json`` -> 生成時の commit。"""
    found: dict[str, str] = {}
    for path in sorted(RESULTS_ROOT.rglob("meta.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        commit = meta.get("commit")
        if isinstance(commit, str) and commit:
            found[path.relative_to(REPO_ROOT).as_posix()] = commit
    return found


def test_every_experiment_records_the_commit_it_was_generated_from() -> None:
    """全実験の ``meta.json`` が生成時の commit を持つこと (指摘 B-5 の前提)。

    commit が無い実験があると、下の「そろっているか」の検査がその実験を
    黙って見逃す。**測れない成果物を作らない**ほうを先に固定する。
    """
    expected = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted(RESULTS_ROOT.rglob("meta.json"))
    }
    missing = sorted(expected - set(_meta_commits()))
    assert not missing, f"commit を記録していない meta.json があります: {missing}"


def test_all_artifacts_were_generated_from_the_same_commit() -> None:
    """全成果物が同一の commit から生成されていること (指摘 B-5)。

    指紋検査 (D-74) はバイト不変を守るが、「**いつの版のコードで作られたか**」
    は見ていない。実測では 3-A / 3-C が ``2ee4aa4``、4-B が ``9565eca`` で、
    記事をまたいで並ぶ図が別バージョンのコードで生成されていた。

    導入時は 5 実験が 5 つの commit を指していたので ``strict=True`` の
    ``xfail`` で入れた。**そろえた瞬間に xpass して落ち**、そこでマーカーを
    外すことになる —— 「そろえたのに検査が無効のまま」を残さないための形で、
    実際にその通りに外れた (一斉再生成で全実験が ``3dc7322`` になった)。

    以降は普通の検査として働く。実験を1本だけ再生成すると赤くなるので、
    **最終稿の前に一斉再生成する**という運用がここで固定される。
    """
    commits = _meta_commits()
    unique = sorted(set(commits.values()))
    assert len(unique) == 1, (
        f"成果物の生成 commit がそろっていません ({len(unique)} 種類):\n"
        + "\n".join(f"  {path}: {commit[:7]}" for path, commit in commits.items())
        + "\n\n**一斉再生成は auto-commit を切って回してください**:\n"
        "    PDCA_KIT_AUTO_COMMIT=off make figures-01 figures-02 "
        "figures-03 figures-04 figures-05\n"
        "切らずに回すと、SubagentStop の auto-commit が途中で HEAD を動かし、"
        "実験ごとに違う commit が焼き込まれてこの検査が落ちます "
        "(実測: 4 回連続で踏んだ)。"
    )


# --- 完了手順2: reviewer-deletion の実行漏れ (D-98) ---------------------------

#: サイクル境界のタグ -> そのサイクルの「削れるか」レビューの証跡。
#: **タグ時点で存在していたこと**まで見る (後から足しても、そのサイクルを
#: レビュー無しで締めたという事実は変わらない)。
DELETION_REVIEW_DOCS: dict[str, str] = {
    "cycle-05a": "docs/削減候補-05.md",
}

#: **証跡が無いまま締めたサイクル** (2026-08-21 の実測)。
#: ``reviewer-deletion`` が手続き層に置かれていたため実行されず、
#: 同じサイクルで重複が4件増えた (D-92 の rationale)。
#: **この集合は増やせない。** 増えるということは、また手順2を飛ばして
#: タグを打ったということである。
KNOWN_WITHOUT_DELETION_REVIEW: frozenset[str] = frozenset({"cycle-05b"})

_HOW_TO_FIX_DELETION = (
    "サイクルを締める前に reviewer-deletion を1回走らせ、"
    "その結果を docs/削減候補-<サイクル>.md に残して "
    "DELETION_REVIEW_DOCS へ登録してください (findings が0件でもよい)。\n"
    "**KNOWN_WITHOUT_DELETION_REVIEW に追記して通すのは"
    "ラチェットを外す操作です。**"
)


def _cycle_tags() -> list[str]:
    """``cycle-*`` のタグ (名前順)。"""
    return sorted(
        tag
        for tag in _git("tag", "--list", "cycle-*").split()
        if tag.startswith("cycle-")
    )


def test_every_cycle_tag_is_accounted_for() -> None:
    """すべてのサイクルタグが、証跡ありか既知の穴かのどちらかであること。

    **新しいタグは必ずここを通る。** 登録も既知の穴への記載も無いタグが
    現れたら、それは「手順2を飛ばしてサイクルを締めた」ということである。
    """
    tags = _cycle_tags()
    if not tags:
        pytest.skip("cycle-* のタグがまだありません")
    accounted = set(DELETION_REVIEW_DOCS) | KNOWN_WITHOUT_DELETION_REVIEW
    unaccounted = sorted(set(tags) - accounted)
    assert not unaccounted, (
        f"証跡の登録が無いサイクルタグがあります: {unaccounted}\n{_HOW_TO_FIX_DELETION}"
    )
    stale = sorted(accounted - set(tags))
    assert not stale, (
        f"実在しないタグが登録されています: {stale}\n"
        "タグを消したか改名したなら、この表からも外してください。"
    )


@pytest.mark.parametrize("tag", sorted(DELETION_REVIEW_DOCS))
def test_the_deletion_review_existed_at_the_cycle_tag(tag: str) -> None:
    """証跡が**そのタグの時点で**存在していたこと (D-98)。

    後から足した文書でも「存在する」は満たせてしまう。しかしそれは
    「レビュー無しでサイクルを締めた」という事実を消さない。
    タグ時点で見るのは、締めた瞬間の状態を測るためである。
    """
    path = DELETION_REVIEW_DOCS[tag]
    # core.quotepath の既定 (true) だと日本語のパスが \345... にエスケープされ、
    # 文字列比較が常に外れる。**この検査は日本語のファイル名を扱う**ので
    # 明示的に切る (実測: 切らないと存在するファイルを「無い」と報告した)。
    listed = _git(
        "-c", "core.quotepath=false", "ls-tree", "--name-only", "-r", tag, path
    ).strip()
    assert listed == path, (
        f"{tag} の時点で {path} が存在しません。\n{_HOW_TO_FIX_DELETION}"
    )


def test_the_known_gap_list_has_no_stale_entries() -> None:
    """穴が埋まったサイクルが既知の穴に残っていないこと。"""
    filled = sorted(KNOWN_WITHOUT_DELETION_REVIEW & set(DELETION_REVIEW_DOCS))
    assert not filled, (
        f"証跡が登録されたのに既知の穴に残っています: {filled}\n"
        "KNOWN_WITHOUT_DELETION_REVIEW から外してください"
        " (ラチェットが1段締まります)。"
    )

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


@pytest.mark.xfail(
    reason=(
        "指摘 B-5: 実験ごとに再生成した時期が違うため、現状は 5 実験が 5 つの "
        "commit を指している。連載の最終稿を作る前に一斉再生成してそろえる。"
        "そろえたらこの xfail を外す (xpass すると落ちるので、外し忘れも気づける)"
    ),
    strict=True,
)
def test_all_artifacts_were_generated_from_the_same_commit() -> None:
    """全成果物が同一の commit から生成されていること (指摘 B-5)。

    指紋検査 (D-74) はバイト不変を守るが、「**いつの版のコードで作られたか**」
    は見ていない。実測では 3-A / 3-C が ``2ee4aa4``、4-B が ``9565eca`` で、
    記事をまたいで並ぶ図が別バージョンのコードで生成されていた。

    ``strict=True`` の ``xfail`` にしてあるので、**そろえた瞬間にこのテストが
    xpass して落ちる**。そこでマーカーを外すことになり、「そろえたのに検査が
    無効のまま」という状態が残らない。
    """
    commits = _meta_commits()
    unique = sorted(set(commits.values()))
    assert len(unique) == 1, (
        f"成果物の生成 commit がそろっていません ({len(unique)} 種類):\n"
        + "\n".join(f"  {path}: {commit[:7]}" for path, commit in commits.items())
        + "\n全実験を一斉再生成してください (make figures-01 .. figures-05)。"
    )

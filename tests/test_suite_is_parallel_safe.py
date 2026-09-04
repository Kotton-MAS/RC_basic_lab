"""テスト同士が状態を共有していないことの検査 (D-151).

``make test`` は ``-n auto`` (pytest-xdist) で走る。並列にできるのは
**テストが互いに状態を共有していない**からで、その前提が崩れると
「並列のときだけ落ちる」という一番読みにくい壊れ方をする。

前提は2つ:

1. ``results/`` へ書くテストが無い (成果物は ``make figures-0N`` だけが作る)
2. リポジトリ直下へ書くテストが無い (書くなら ``tmp_path``)

**実行して確かめる形にしない。** 並列で走っている最中に「いま並列か」を
問うガードは、直列で走らせた瞬間に空振りする (CLAUDE.md の学習メモ)。
**書き込み先をソースから読む**形にする。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

FORBIDDEN_WRITE = re.compile(
    r"""(?x)
    (?:write_text|write_bytes|mkdir|touch|unlink|savefig|to_csv)\s*\(
    |open\s*\(\s*[^)]*["']w
    """
)
"""書き込みを疑う呼び出し。"""

RESULTS_HINT = re.compile(r'RESULTS_DIR|["\']results/')
"""``results/`` を指していそうな式。"""

ALLOWED_RESULTS_WRITERS: frozenset[str] = frozenset()
"""``results/`` へ書いてよいテスト。**空である** (成果物は make が作る)。"""


def _lines(path: Path) -> list[tuple[int, str]]:
    return list(enumerate(path.read_text(encoding="utf-8").splitlines(), 1))


def test_no_test_writes_into_results() -> None:
    """``results/`` へ書くテストが無い (並列でも成果物が壊れない)。

    変異注入: どれかのテストに ``(ROOT / "results" / "x").write_text("")``
    を足すと赤くなる。
    """
    offenders: list[str] = []
    for path in sorted(TESTS.rglob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        for number, line in _lines(path):
            if FORBIDDEN_WRITE.search(line) and RESULTS_HINT.search(line):
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{number}")
    offenders = [
        item for item in offenders if item.split(":")[0] not in ALLOWED_RESULTS_WRITERS
    ]
    assert not offenders, (
        "results/ へ書いていそうなテストがあります (並列で成果物が壊れます):\n"
        + "\n".join(f"  {item}" for item in offenders)
        + "\n書き先を tmp_path にしてください。"
    )


def test_the_parallel_plugin_is_declared() -> None:
    """``-n auto`` が使える依存が dev グループに宣言されている。

    ``Makefile`` が ``-n auto`` を渡すので、宣言が消えると **make test が
    起動すらしなくなる** (テストが落ちるのではなく pytest が引数を知らない)。
    """
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "pytest-xdist" in pyproject, (
        "pytest-xdist が pyproject.toml にありません "
        "(Makefile の test が -n auto を渡します)"
    )
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "-n auto" in makefile, "Makefile の test が並列でなくなりました"

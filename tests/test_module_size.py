"""モジュールの大きさの上限 (``CLAUDE.md`` の「コード規約」).

``CLAUDE.md`` には「関数は単一責任、50行超は分割を検討」があったが、**ファイル
単位の上限が無かった**。非空 1,375 行の ``experiment/freerun.py`` はその抜け穴
から生えている。

上限を文章で書くだけでは守られないことは実測済み (``~/.claude/CLAUDE.md``
「再発している事象への対策をプロンプト層に置かない」) なので、ここで機械的に
強制する。``config/`` は既に ``tests/test_config_package_layout.py`` が非空
300 行で縛っており、こちらはそれ以外を含む ``src/`` 全体を 600 行で縛る。

**既に超えている6モジュールは現在値で記録し、増えることだけを禁じる**
(ラチェット)。いきなり一律 600 行で落とすと、リファクタリングと無関係な変更まで
赤くなって上限そのものが無効化される。記録値を下回ったモジュールは
``OVER_LIMIT`` から外すことを要求するので、リストが古びたまま残ることもない。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "src" / "rc_basics_lab"
CLAUDE_MD = ROOT / "CLAUDE.md"

MAX_NONEMPTY_LINES = 600
"""1モジュールあたりの非空行数の上限。

正本はこの定数1つで、``CLAUDE.md`` はその写しである
(``test_claude_md_records_the_same_line_budget`` が一致を固定する)。
"""

OVER_LIMIT: dict[str, int] = {
    "experiment/freerun.py": 1375,
    "experiment/capacity.py": 974,
    "diagnostics/ipc.py": 786,
    "experiment/esp.py": 782,
    "plotting/figures_capacity.py": 740,
    "plotting/figures_esp.py": 677,
}
"""上限を超えたまま残っているモジュールと、その**現在の**非空行数。

ここに載っている間は「記録値以下」だけを要求する (減らす方向にしか動かせない)。
600 行以下まで縮んだらこの表から**外す** —— 外し忘れは
``test_over_limit_modules_are_still_over_the_limit`` が赤くする。
"""


def _nonempty_line_count(path: Path) -> int:
    """非空行数。"""
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def _modules() -> list[Path]:
    """``src/rc_basics_lab/`` 配下の全モジュール。"""
    return sorted(PACKAGE_DIR.rglob("*.py"))


def _key(path: Path) -> str:
    """``OVER_LIMIT`` のキー (パッケージからの相対パス)。"""
    return path.relative_to(PACKAGE_DIR).as_posix()


@pytest.mark.parametrize("path", _modules(), ids=_key)
def test_module_is_within_the_line_budget(path: Path) -> None:
    """記録の無いモジュールは非空 600 行以下。

    新しく 600 行を超えるモジュールを作れないようにするのがこの検査の役割で、
    既存の超過分 (``OVER_LIMIT``) は別の検査が扱う。
    """
    key = _key(path)
    if key in OVER_LIMIT:
        pytest.skip(f"OVER_LIMIT に記録済み (現在値 {OVER_LIMIT[key]} 行)")
    nonempty = _nonempty_line_count(path)
    assert nonempty <= MAX_NONEMPTY_LINES, (
        f"{key} が非空 {nonempty} 行で上限 {MAX_NONEMPTY_LINES} 行を超えました。"
        " 分割してください (どうしても分割できない理由があるなら"
        " OVER_LIMIT へ追記し、.claude/decisions.yaml に理由を残すこと)"
    )


@pytest.mark.parametrize("key", sorted(OVER_LIMIT), ids=sorted(OVER_LIMIT))
def test_over_limit_module_does_not_grow(key: str) -> None:
    """超過中のモジュールは記録値より増えない (ラチェット)。"""
    nonempty = _nonempty_line_count(PACKAGE_DIR / key)
    assert nonempty <= OVER_LIMIT[key], (
        f"{key} が非空 {nonempty} 行に増えました (記録値 {OVER_LIMIT[key]} 行)。"
        " 超過中のモジュールは減らす方向にしか動かせません"
    )


@pytest.mark.parametrize("key", sorted(OVER_LIMIT), ids=sorted(OVER_LIMIT))
def test_over_limit_modules_are_still_over_the_limit(key: str) -> None:
    """記録値まで縮んだモジュールは ``OVER_LIMIT`` から外す。

    外し忘れると「600 行以下なのに免除されたまま」の枠が残り、次に太ったときに
    ``test_module_is_within_the_line_budget`` が働かない。
    """
    nonempty = _nonempty_line_count(PACKAGE_DIR / key)
    assert nonempty > MAX_NONEMPTY_LINES, (
        f"{key} が非空 {nonempty} 行まで縮みました。OVER_LIMIT から外してください"
    )


def test_over_limit_entries_point_at_existing_modules() -> None:
    """``OVER_LIMIT`` に実在しないモジュールが残っていない。"""
    missing = sorted(key for key in OVER_LIMIT if not (PACKAGE_DIR / key).is_file())
    assert not missing, f"OVER_LIMIT に実在しないモジュールがあります: {missing}"


def test_claude_md_records_the_same_line_budget() -> None:
    """``CLAUDE.md`` の「コード規約」に書いた上限がこの定数と一致する。

    上限の正本はこの定数で、``CLAUDE.md`` はその写しである。両方に数字を書く
    以上、片方だけ更新される事故は必ず起きるので機械で潰す
    (``tests/test_config_package_layout.py`` が design.md に対して行うのと同じ形)。
    """
    match = re.search(
        r"モジュールは非空 (\d+) 行以下", CLAUDE_MD.read_text(encoding="utf-8")
    )
    assert match, "CLAUDE.md の「コード規約」にモジュール行数の上限がありません"
    assert int(match[1]) == MAX_NONEMPTY_LINES

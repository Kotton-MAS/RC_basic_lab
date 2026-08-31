"""``src/`` 全体のモジュール行数をラチェットで縛る (D-77)。

## なぜラチェットなのか

`CLAUDE.md` の「コード規約」は「1モジュール 600 行を上限とする」と
**無条件に**書いている。一方で D-63 の射程は 05 の実験層
(``experiment/anomaly*.py``) だけであり、その guard_test も 05 の
モジュールしか見ていない。

**実測すると 600 行超は 9 本ある。** つまり次のサイクルの実装者は
「600 行上限が機械で守られている」と読んだうえで、
**守られていない場所に書き足す**ことになる。
これは「層どうしの矛盾」と「空虚なガード」の合わせ技であり、
**それらを直しているサイクルで新しく作られた**ものである。

単純に全体へ 600 行上限を課すと、既存の 9 本が即座に赤になって
「まず 9 本を割る」以外の作業ができなくなる。かといって
`CLAUDE.md` の文言を「05 以降の新規モジュールは」に戻すと、
既存 9 本は永久に据え置きになる。

そこで**ラチェット**にする:

- **新規モジュールは 600 行が上限**
- **既存の超過モジュールは現在値で凍結**する。**増えたら落ちる。減らすのは自由**

D-63 の rationale は「『後で割る』は必ず割られない」と書いている。
同じ論理で、許容リストを「後で減らす」でも減らない。
**現在値で凍結することが、悪化を止める唯一の機械的な形**である。
これで `CLAUDE.md` の「1モジュール 600 行を上限とする」が嘘でなくなる
(新規に対しては真、既存に対しては単調非増加)。

## 許容リストの直し方

**行数を増やしたいときにこの表を書き換えてはいけない。**
それはラチェットを外す操作である。
表を書き換えてよいのは**減らしたとき**だけで、その場合は実測値に更新する
(次に増やせる余地がその分なくなる = ラチェットが1段締まる)。
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

#: 新規モジュールの上限 (D-63 と同じ値)。
LINE_BUDGET = 600

#: 上限を超えたまま凍結している既存モジュール (2026-08-21 の実測値)。
#: **増やすために書き換えないこと。** 減らしたときだけ実測値へ更新する。
FROZEN: dict[str, int] = {
    "rc_basics_lab/experiment/freerun.py": 1516,
    "rc_basics_lab/experiment/capacity.py": 714,
    "rc_basics_lab/experiment/esp.py": 906,
    "rc_basics_lab/diagnostics/ipc.py": 910,
    "rc_basics_lab/diagnostics/_capacity.py": 829,
    "rc_basics_lab/experiment/attractor.py": 707,
    "rc_basics_lab/experiment/stability.py": 625,
}


def _modules() -> dict[str, int]:
    """``src/`` 配下の全モジュールの行数 (src 相対の POSIX パス)。"""
    found: dict[str, int] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        key = path.relative_to(SRC_ROOT).as_posix()
        found[key] = len(path.read_text(encoding="utf-8").splitlines())
    return found


def test_new_modules_stay_within_the_line_budget() -> None:
    """凍結リストに無いモジュールは 600 行以内であること。"""
    offenders = {
        name: count
        for name, count in _modules().items()
        if name not in FROZEN and count > LINE_BUDGET
    }
    assert not offenders, (
        f"{LINE_BUDGET} 行を超えたモジュールがあります: {offenders}\n"
        "割ってください。**上限のほうを緩めない** (D-63)。\n"
        "既存の超過分は FROZEN に凍結してありますが、"
        "新しく足すためにそこへ追記するのはラチェットを外す操作です。"
    )


@pytest.mark.parametrize("name", sorted(FROZEN))
def test_frozen_modules_never_grow(name: str) -> None:
    """凍結したモジュールが**増えていない**こと (減るのは自由)。"""
    found = _modules()
    assert name in found, (
        f"凍結リストの {name} が存在しません。"
        "消したか改名したなら FROZEN からも外してください。"
    )
    limit = FROZEN[name]
    actual = found[name]
    assert actual <= limit, (
        f"{name} が凍結時点より増えています: {limit} → {actual} 行。\n"
        "このファイルは既に上限を超えているので、**足すなら別モジュールへ**。\n"
        "FROZEN の数値を上げて通すのはラチェットを外す操作です。"
    )


def test_the_frozen_list_has_no_stale_entries() -> None:
    """凍結リストが実態とずれていないこと。

    減らしたのに凍結値が古いままだと、その差分だけ**また増やせてしまう**。
    ラチェットが静かに緩むので、ここで気づけるようにする。
    """
    found = _modules()
    slack = {
        name: (limit, found[name])
        for name, limit in FROZEN.items()
        if name in found and found[name] < limit
    }
    assert not slack, (
        f"凍結値より小さくなったモジュールがあります (凍結値, 実測): {slack}\n"
        "FROZEN を実測値に更新してください (ラチェットが1段締まります)。"
    )
    already_ok = {
        name: found[name]
        for name, limit in FROZEN.items()
        if name in found and found[name] <= LINE_BUDGET
    }
    assert not already_ok, (
        f"上限以下になったモジュールが凍結リストに残っています: {already_ok}\n"
        "FROZEN から外してください (通常の上限で守られるようになります)。"
    )

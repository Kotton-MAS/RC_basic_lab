"""``src/`` 全体の同名定義をラチェットで縛る (D-92).

## なぜラチェットなのか

`reviewer-deletion` の観点1 (同名の重複) は、第2版の指摘 A-3 に対して
**手続き層**で結着した —— `CLAUDE.md` の「サイクル完了時にやること」に
「`reviewer-deletion` を1回走らせる。**走らせないことが問題である**」と
明記された。

そのサイクルで走らなかった。そして走らなかった同じサイクルで、
その手順が拾うはずの重複が4件増えた:

| 重複 | 前 | 後 |
|---|---|---|
| ``SectionTiming`` | 2 | 3 |
| ``_log_timing`` | 1 | 2 |
| ``_meta_extra`` | 1 | 2 |
| ``sign_test_p_value`` | 1 | 2 |

4件目は**境界の振る舞いが違う p 値関数が2本**という実害を伴っていた
(対の数が0のとき ``nan`` と ``1.0``)。両方とも記事に載る数値を作る。

同じ巡で、**機械層**に落とした観点4 (肥大したモジュール、
``tests/test_module_line_budget.py``) は機能して実際に縮んだ。

読み取れるのは「文言の強さの問題ではない」ということである。
**手続き層はプロンプト層と同じ側に落ちる** —— やらなかったことに気づく
機械が無ければ、やらない回が出る。だから同じ形の機械にする:

- **新しい同名は増やせない**
- **既存の同名は現在の定義箇所で凍結**する。**増えたら落ちる。減らすのは自由**

## 凍結リストの直し方

**重複を増やしたいときにこの表を書き換えてはいけない。**
それはラチェットを外す操作である。
表を書き換えてよいのは**減らしたとき**だけで、その場合は実測値に更新する。

## 同名が正当な場合

``datasets/`` の5件 (``fetch`` / ``is_available`` / ``load_series`` /
``manifest`` / ``series_path``) は Protocol の実装なので**同名が正しい**。
``tasks/_validate`` のようにモジュール固有の私的ヘルパも同様である。
それらは凍結リストに載せたまま据え置けばよい —— この検査が止めたいのは
「同じ仕事をする関数が2箇所で独立に育つ」ことであって、同名そのものでは
ない。ただし**新しい同名は必ずここを通る**ので、正当かどうかを1度は
人が判断することになる。
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

#: 現在の同名定義 (2026-08-21 の実測値)。名前 -> 定義しているモジュール。
#: **増やすために書き換えないこと。** 減らしたときだけ実測値へ更新する。
FROZEN: dict[str, tuple[str, ...]] = {
    "SectionTiming": (
        "rc_basics_lab/experiment/anomaly_pipeline.py",
        "rc_basics_lab/experiment/capacity_pipeline.py",
        "rc_basics_lab/experiment/freerun_pipeline.py",
    ),
    "_derivative": (
        "rc_basics_lab/tasks/chaotic.py",
        "rc_basics_lab/tasks/mackey_glass.py",
    ),
    "_evaluate": (
        "rc_basics_lab/experiment/anomaly.py",
        "rc_basics_lab/experiment/runner.py",
    ),
    "_log_timing": (
        "rc_basics_lab/experiment/anomaly_pipeline.py",
        "rc_basics_lab/experiment/freerun_pipeline.py",
    ),
    "_meta_extra": (
        "rc_basics_lab/experiment/anomaly_pipeline.py",
        "rc_basics_lab/experiment/freerun_pipeline.py",
    ),
    "_methods_in": (
        "rc_basics_lab/plotting/figures_anomaly.py",
        "rc_basics_lab/plotting/figures_anomaly_sweep.py",
    ),
    "_rows": (
        "rc_basics_lab/experiment/anomaly_score.py",
        # freerun.py から freerun_fit.py へ**移した** (D-128)。件数は増えていない
        "rc_basics_lab/experiment/freerun_fit.py",
        "rc_basics_lab/experiment/runner.py",
    ),
    "_sweep": (
        "rc_basics_lab/experiment/capacity.py",
        "rc_basics_lab/experiment/esp.py",
    ),
    "_validate": (
        "rc_basics_lab/tasks/anomaly.py",
        "rc_basics_lab/tasks/chaotic.py",
        "rc_basics_lab/tasks/delay_parity.py",
        "rc_basics_lab/tasks/mackey_glass.py",
        "rc_basics_lab/tasks/narma.py",
    ),
    "_validate_config": (
        "rc_basics_lab/diagnostics/ipc.py",
        "rc_basics_lab/diagnostics/lyapunov.py",
        "rc_basics_lab/diagnostics/memory_capacity.py",
        "rc_basics_lab/reservoir/esn.py",
    ),
    "capacity_context": (
        "rc_basics_lab/experiment/capacity.py",
        "rc_basics_lab/experiment/stability.py",
    ),
    "fetch": (
        "rc_basics_lab/datasets/mgab.py",
        "rc_basics_lab/datasets/ucr.py",
    ),
    "is_available": (
        "rc_basics_lab/datasets/mgab.py",
        "rc_basics_lab/datasets/ucr.py",
    ),
    "load_series": (
        "rc_basics_lab/datasets/mgab.py",
        "rc_basics_lab/datasets/ucr.py",
    ),
    "manifest": (
        "rc_basics_lab/datasets/mgab.py",
        "rc_basics_lab/datasets/ucr.py",
    ),
    "series_path": (
        "rc_basics_lab/datasets/mgab.py",
        "rc_basics_lab/datasets/ucr.py",
    ),
}

_HOW_TO_FIX = (
    "同じ仕事をするなら**片方に寄せてください** "
    "(実例: sign_test_p_value は metrics_significance.py へ集約した。D-91)。\n"
    "**FROZEN に追記して通すのはラチェットを外す操作です。**\n"
    "モジュール固有の私的ヘルパで同名が正当な場合だけ、"
    "その理由をコミットメッセージに書いた上で追記してください。"
)


def _module_level_definitions() -> dict[str, tuple[str, ...]]:
    """``src/`` 配下のモジュール直下の def / class 名 -> 定義モジュール。

    モジュール直下だけを見るのは、メソッド名 (``run`` / ``__init__``) まで
    数えると同名が構造上必ず大量に出て、検査が意味を失うためである。
    """
    found: dict[str, list[str]] = defaultdict(list)
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        key = path.relative_to(SRC_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                found[node.name].append(key)
    return {name: tuple(sorted(paths)) for name, paths in found.items()}


def _duplicates() -> dict[str, tuple[str, ...]]:
    """2箇所以上で定義されている名前だけを返す。"""
    return {
        name: paths
        for name, paths in _module_level_definitions().items()
        if len(paths) > 1
    }


def test_no_new_duplicate_names_appear() -> None:
    """凍結リストに無い同名定義が現れていないこと。"""
    offenders = {
        name: paths for name, paths in _duplicates().items() if name not in FROZEN
    }
    assert not offenders, f"新しい同名定義が現れました: {offenders}\n{_HOW_TO_FIX}"


@pytest.mark.parametrize("name", sorted(FROZEN))
def test_frozen_duplicates_never_spread(name: str) -> None:
    """凍結した同名が**新しいモジュールへ広がっていない**こと (減るのは自由)。"""
    found = _duplicates().get(name, ())
    frozen = set(FROZEN[name])
    spread = sorted(set(found) - frozen)
    assert not spread, (
        f"{name} が凍結時点に無いモジュールでも定義されています: {spread}\n"
        f"{_HOW_TO_FIX}"
    )


def test_the_frozen_list_has_no_stale_entries() -> None:
    """凍結リストが実態とずれていないこと。

    重複を減らしたのに凍結値が古いままだと、その差分だけ**また増やせて
    しまう**。ラチェットが静かに緩むので、ここで気づけるようにする。
    """
    duplicates = _duplicates()
    definitions = _module_level_definitions()
    slack: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for name, frozen in FROZEN.items():
        actual = duplicates.get(name, definitions.get(name, ()))
        if set(actual) < set(frozen):
            slack[name] = (frozen, actual)
    assert not slack, (
        f"凍結時点より減った同名があります (凍結値, 実測): {slack}\n"
        "FROZEN を実測値に更新してください (ラチェットが1段締まります)。\n"
        "1箇所だけになった名前は FROZEN から外してください。"
    )

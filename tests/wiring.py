"""配線テストの共通機構 (01 の ``ExperimentConfig`` と 02 の ``Esp02Config`` で共有).

本リポジトリ最大の失敗モードは**「設定したのに効いていない」**であり、これは
出力が正常に見えるためレビューでは確率的にしか見つからない。防衛線は
「全設定フィールドについて、値を変えたら出力が変わることを実測する」テストで、
その機構 (ケースの記述・パスによる差し替え・葉フィールドの列挙) をここに置く。

実験ごとに設定 dataclass は分かれる (D-13) が、この機構だけは共有する。
実験ごとに写経すると、01 で効いている検出力が 02 で静かに落ちても誰も気づかない。

**何をここに置かないか**: 「出力」が何かは実験ごとに違う (01 は
``run_experiment`` の結果行、02 は診断の scalars や乱数列)。判定そのものは
各テストファイルに置き、ここはケースの記述と設定の差し替えだけを担う。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, cast, get_type_hints

if TYPE_CHECKING:  # pragma: no cover - 型検査時のみ必要
    from _typeshed import DataclassInstance

CHANNEL_ROWS = "rows"
"""結果行が変わる (ほとんどのパラメータ)。"""

CHANNEL_META = "meta"
"""結果行は変わらないが ``meta.json`` が変わる (純粋なメタ情報)。"""

CHANNEL_ERROR = "error"
"""値域が1点しかなく、別の値は即座に例外になる (``activation`` など)。"""


@dataclass(frozen=True, slots=True)
class WiringCase:
    """1パラメータぶんの配線テスト。

    Attributes:
        field: 検査対象のフィールド (ドット区切り)。coverage の単位。
        overrides: 実際に差し替える (パス, 値) の並び。制約のあるフィールド
            (分割比) だけ2つ以上になる。
        channel: 効き方 (``rows`` / ``meta`` / ``error`` / 実験固有のチャネル)。
        scope: 変化してよい課題名。``None`` なら全課題。
        changed_sizes: 変化していることを追加で要求する分割サイズの列名。
        note: 単独で動かせない等の理由。
    """

    field: str
    overrides: tuple[tuple[str, object], ...]
    channel: str = CHANNEL_ROWS
    scope: str | None = None
    changed_sizes: tuple[str, ...] = ()
    note: str = ""


def case(
    field: str,
    value: object,
    *,
    channel: str = CHANNEL_ROWS,
    scope: str | None = None,
    note: str = "",
) -> WiringCase:
    """単独で動かせるパラメータのケース。"""
    return WiringCase(
        field=field,
        overrides=((field, value),),
        channel=channel,
        scope=scope,
        note=note,
    )


def _replace_path(instance: object, path: str, value: object) -> object:
    """ドット区切りのパスで frozen dataclass を差し替えた複製を返す。"""
    head, _, rest = path.partition(".")
    new_value = _replace_path(getattr(instance, head), rest, value) if rest else value
    return dataclasses.replace(cast("DataclassInstance", instance), **{head: new_value})


def apply_case[T](config: T, wiring_case: WiringCase) -> T:
    """ケースの差し替えを適用した設定を返す。"""
    changed: object = config
    for path, value in wiring_case.overrides:
        changed = _replace_path(changed, path, value)
    return cast("T", changed)


def leaf_paths(cls: type, prefix: str = "") -> set[str]:
    """dataclass の葉フィールドをドット区切りのパスで列挙する。"""
    hints = get_type_hints(cls)
    paths: set[str] = set()
    for item in fields(cast("type[DataclassInstance]", cls)):
        annotation = hints[item.name]
        if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
            paths |= leaf_paths(annotation, f"{prefix}{item.name}.")
        else:
            paths.add(f"{prefix}{item.name}")
    return paths


def plain(value: object) -> object:
    """YAML に安全に書ける値へ落とす (tuple -> list)。"""
    if isinstance(value, tuple | list):
        return [plain(element) for element in value]
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    return value


def assert_yaml_has_all_leaves(written: object, cls: type) -> None:
    """YAML へ書き出した内容に ``cls`` の全葉フィールドが現れることを検査する。

    「dataclass には在るが YAML からは設定できない」パラメータを作らないための
    検査 (D-09 の未知キー検査と対になる)。
    """
    for leaf in leaf_paths(cls):
        node: object = written
        for part in leaf.split("."):
            assert isinstance(node, Mapping), leaf
            assert part in node, f"YAML に現れないフィールドです: {leaf}"
            node = node[part]


__all__ = [
    "CHANNEL_ERROR",
    "CHANNEL_META",
    "CHANNEL_ROWS",
    "WiringCase",
    "apply_case",
    "assert_yaml_has_all_leaves",
    "case",
    "leaf_paths",
    "plain",
]

"""配線テストの共通機構 (01〜04 の実験設定クラスで共有).

本リポジトリ最大の失敗モードは**「設定したのに効いていない」**であり、これは
出力が正常に見えるためレビューでは確率的にしか見つからない。防衛線は
「全設定フィールドについて、値を変えたら出力が変わることを実測する」テストで、
その機構 (ケースの記述・パスによる差し替え・葉フィールドの列挙) をここに置く。

実験ごとに設定 dataclass は分かれる (D-13) が、この機構だけは共有する。
実験ごとに写経すると、01 で効いている検出力が 02 で静かに落ちても誰も気づかない。

**何をここに置かないか**: 「出力」が何かは実験ごとに違う (01 は
``run_experiment`` の結果行、02 は診断の scalars や乱数列)。判定そのものは
各テストファイルに置き、ここはケースの記述と設定の差し替えだけを担う。

**02〜04 で追加され、3〜4ファイルに写経されていたもの**: 結果行の指紋
(``fingerprint``)・葉フィールド単位の差分 (``changed_leaves`` / ``leaf_value``)・
YAML 往復 (``round_trip``)・基底シードのケース (``seeds_case``)・セクション
固有のケース (``section_case``)・``rc_basics_lab.experiment`` 配下のモジュール
一覧 (``current_experiment_modules``)。これも同じ理由 (写経すると検出力の
劣化に誰も気づかない) でここへ集約する。**正規化 (揮発列・行クラス・課題名の
フィールド名) は実験ごとに違う**ため、それらは呼び出し側から引数で渡す。
"""

from __future__ import annotations

import dataclasses
import json
import pkgutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from types import UnionType
from typing import (
    TYPE_CHECKING,
    TypeAliasType,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml

import rc_basics_lab.experiment as _experiment_pkg
from rc_basics_lab.config import (
    DelayParityConfig,
    DelayParityTask,
    ExperimentConfig,
    MackeyGlassConfig,
    MackeyGlassTask,
    load_config_as,
)
from rc_basics_lab.config._dump import as_plain_mapping
from rc_basics_lab.reservoir.protocol import ReservoirConfig
from rc_basics_lab.seeds import SeedStream

if TYPE_CHECKING:  # pragma: no cover - 型検査時のみ必要
    from _typeshed import DataclassInstance

CHANNEL_ROWS = "rows"
"""結果行が変わる (ほとんどのパラメータ)。"""

CHANNEL_META = "meta"
"""結果行は変わらないが ``meta.json`` が変わる (純粋なメタ情報)。"""

CHANNEL_ERROR = "error"
"""値域が1点しかなく、別の値は即座に例外になる (``activation`` など)。"""

CHANNEL_SEEDS = "seeds"
"""基底シードを変えると、そのストリームの乱数列だけが変わる。"""


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
        task: 先送り (pending) がどのタスク段階で解消される予定かを表す
            構造化フィールド (例 ``"T3"``)。``note`` の自由文に埋めると
            判定に使えないため (F-02-2-004)、判定が要る場合はここへ書く。
            01 のケースは使わないため既定値 ``None``。
        note: 単独で動かせない等の理由。
    """

    field: str
    overrides: tuple[tuple[str, object], ...]
    channel: str = CHANNEL_ROWS
    scope: str | None = None
    changed_sizes: tuple[str, ...] = ()
    task: str | None = None
    note: str = ""


def case(
    field: str,
    value: object,
    *,
    channel: str = CHANNEL_ROWS,
    scope: str | None = None,
    task: str | None = None,
    note: str = "",
) -> WiringCase:
    """単独で動かせるパラメータのケース。"""
    return WiringCase(
        field=field,
        overrides=((field, value),),
        channel=channel,
        scope=scope,
        task=task,
        note=note,
    )


def seeds_case(field: str, value: int, stream: SeedStream) -> WiringCase:
    """基底シード1本ぶんのケース。``scope`` に変化してよいストリームを書く。"""
    return case(field, value, channel=CHANNEL_SEEDS, scope=stream.value)


def section_case(field: str, value: object, scope: str) -> WiringCase:
    """セクション/課題固有の葉のケース。**そのセクションの出力だけ**が変わることまで測る。"""
    return case(field, value, scope=scope)


def _replace_path(instance: object, path: str, value: object) -> object:
    """ドット区切りのパスで frozen dataclass を差し替えた複製を返す。

    ``tasks[0].params.length`` のような**添字**も歩く (D-123)。課題がリストに
    なったので、添字を扱えないと ``tasks`` 配下の葉が D-13 の検査から丸ごと
    外れる —— 「設定したのに効いていない」を捕まえる唯一の機構が、課題の設定
    についてだけ黙って無効になる。
    """
    head, _, rest = path.partition(".")
    name, index = _split_index(head)
    if index is None:
        new_value = (
            _replace_path(getattr(instance, name), rest, value) if rest else value
        )
        return dataclasses.replace(
            cast("DataclassInstance", instance), **{name: new_value}
        )
    items = list(cast("Sequence[object]", getattr(instance, name)))
    items[index] = _replace_path(items[index], rest, value) if rest else value
    return dataclasses.replace(
        cast("DataclassInstance", instance), **{name: tuple(items)}
    )


def _split_index(head: str) -> tuple[str, int | None]:
    """``tasks[0]`` を ``("tasks", 0)`` に、``split`` を ``("split", None)`` に。"""
    if head.endswith("]") and "[" in head:
        name, _, raw = head.partition("[")
        return name, int(raw[:-1])
    return head, None


def apply_case[T](config: T, wiring_case: WiringCase) -> T:
    """ケースの差し替えを適用した設定を返す。"""
    changed: object = config
    for path, value in wiring_case.overrides:
        changed = _replace_path(changed, path, value)
    return cast("T", changed)


def leaf_paths(cls: type, prefix: str = "") -> set[str]:
    """dataclass の葉フィールドをドット区切りのパスで列挙する。

    **型エイリアスと dataclass の union を開く。** ``type X = A`` (PEP 695) は
    ``get_type_hints`` が解決せず ``TypeAliasType`` のまま返すので、開かないと
    そのフィールドが葉に見えてしまう —— つまり ``A`` の中身が D-13 の検査から
    黙って外れる (``esn_mackey_glass`` を ``ReservoirConfig`` にしたときに実際
    そうなった)。

    union は**先頭の要素だけ**を歩く。この関数が答えるのは「既定の設定
    インスタンスにどんな葉があるか」で、``kind`` を省いた YAML が作るのは
    先頭の要素だからである。当初は全要素を合わせたが、2つ目・3つ目のモデルを
    足した時点で ``ESNConfig`` のインスタンスに ``inter_layer_scale`` を
    設定しようとして 100 件以上落ちた —— **存在しないフィールドの検査**に
    なっていた。他のモデルの設定値は、そのモデル自身のテスト
    (``test_reservoir_deep`` / ``test_reservoir_ring``) が「変えたら出力が
    変わる」ことを見ている。
    """
    hints = get_type_hints(cls)
    paths: set[str] = set()
    for item in fields(cast("type[DataclassInstance]", cls)):
        paths |= _leaf_paths_of(hints[item.name], f"{prefix}{item.name}")
    return paths


def _leaf_paths_of(annotation: object, path: str) -> set[str]:
    """1フィールドぶんの葉パス。"""
    if isinstance(annotation, TypeAliasType):
        return _leaf_paths_of(annotation.__value__, path)
    if get_origin(annotation) is UnionType:
        members = [
            arg
            for arg in get_args(annotation)
            if dataclasses.is_dataclass(arg) and isinstance(arg, type)
        ]
        if members and len(members) == len(get_args(annotation)):
            return leaf_paths(members[0], f"{path}.")
        return {path}
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return leaf_paths(annotation, f"{path}.")
    if get_origin(annotation) is tuple:
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return _leaf_paths_of_sequence(args[0], path)
    return {path}


def _leaf_paths_of_sequence(element: object, path: str) -> set[str]:
    """``tuple[X, ...]`` の葉パス (D-123)。

    要素が dataclass の union なら**全要素を1つずつ添字で**歩く。既定の課題列
    は union の先頭だけではなく複数の種類を並べるので、先頭だけを見ると
    2番目以降の課題の設定が D-13 の検査から外れる (単一フィールドのときに
    先頭だけを歩くのとは事情が違う —— あちらは1つのフィールドに入るのが
    1つの型だが、こちらは並びのそれぞれが別の型を取りうる)。

    要素が dataclass でも union でもない (格子のような値の並び) なら、
    リストそのものを1つの葉として扱う。
    """
    resolved = element.__value__ if isinstance(element, TypeAliasType) else element
    members: list[object] = []
    if get_origin(resolved) is UnionType:
        members = [
            arg
            for arg in get_args(resolved)
            if dataclasses.is_dataclass(arg) and isinstance(arg, type)
        ]
        if len(members) != len(get_args(resolved)):
            return {path}
    elif dataclasses.is_dataclass(resolved) and isinstance(resolved, type):
        members = [resolved]
    else:
        return {path}
    paths: set[str] = set()
    for index, member in enumerate(members):
        paths |= leaf_paths(cast("type", member), f"{path}[{index}].")
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
            name, index = _split_index(part)
            assert name in node, f"YAML に現れないフィールドです: {leaf}"
            node = node[name]
            if index is not None:
                node = cast("Sequence[object]", node)[index]


def leaf_value(config: object, leaf: str) -> object:
    """ドット区切りのパスでフィールド値を取り出す (``tasks[0].params.length`` 可)。"""
    node: object = config
    for part in leaf.split("."):
        name, index = _split_index(part)
        node = getattr(node, name)
        if index is not None:
            node = cast("Sequence[object]", node)[index]
    return node


def changed_leaves(base: object, changed: object) -> set[str]:
    """2つの設定で値が異なる葉フィールドのパス集合。

    比較する葉の一覧は ``type(base)`` から求める (``base`` と ``changed`` は
    同じ設定 dataclass のインスタンスであることが前提)。
    """
    return {
        leaf
        for leaf in leaf_paths(type(base))
        if leaf_value(base, leaf) != leaf_value(changed, leaf)
    }


def round_trip[T](config: T, tmp_path: Path, name: str, cls: type[T]) -> T:
    """設定を YAML へ書き出して読み直す (``load_config_as`` の経路そのもの)。"""
    path = tmp_path / f"{name}.yaml"
    # kind を書ける経路で落とす (D-123)。asdict は ClassVar を落とすので、
    # 課題リストの2番目が先頭の型として読み直される。
    dumped = cast("Mapping[str, object]", plain(as_plain_mapping(config)))
    path.write_text(yaml.safe_dump(dumped, allow_unicode=True), encoding="utf-8")
    return load_config_as(path, cls)


def fingerprint(
    rows: Sequence[object],
    row_cls: type,
    *,
    volatile_columns: frozenset[str],
    field: str | None = None,
    value: str | None = None,
) -> str:
    """結果行の指紋 (揮発列だけ除く)。

    ``field``/``value`` を渡すとその値に一致する行だけを見る。セクション固有の
    葉が他の行を動かしていないことの確認に使う (``field is None`` なら全行)。
    """
    selected = [row for row in rows if field is None or getattr(row, field) == value]
    return json.dumps(
        [
            {
                item.name: getattr(row, item.name)
                for item in fields(cast("type[DataclassInstance]", row_cls))
                if item.name not in volatile_columns
            }
            for row in selected
        ],
        sort_keys=True,
    )


def current_experiment_modules() -> frozenset[str]:
    """``rc_basics_lab.experiment`` 配下の公開モジュール名の実集合。"""
    return frozenset(
        info.name
        for info in pkgutil.iter_modules(_experiment_pkg.__path__)
        if not info.name.startswith("_")
    )


__all__ = [
    "CHANNEL_ERROR",
    "CHANNEL_META",
    "CHANNEL_ROWS",
    "CHANNEL_SEEDS",
    "WiringCase",
    "apply_case",
    "assert_yaml_has_all_leaves",
    "case",
    "changed_leaves",
    "current_experiment_modules",
    "fingerprint",
    "leaf_paths",
    "leaf_value",
    "plain",
    "round_trip",
    "section_case",
    "seeds_case",
]


def experiment_config(
    *,
    mackey_glass: MackeyGlassConfig | None = None,
    delay_parity: DelayParityConfig | None = None,
    esn_mackey_glass: ReservoirConfig | None = None,
    esn_delay_parity: ReservoirConfig | None = None,
    **rest: object,
) -> ExperimentConfig:
    """テスト用に 01 の設定を組む (課題2本の並びは本番と同じ。D-123)。

    課題がリストになった (D-123) ので、テストが毎回 ``MackeyGlassTask(...)`` を
    2つ書き下すと、**課題の並び順という本番と同じ前提**が 20 ファイルに写経
    される。ここに1つ置けば、3本目の課題を既定に入れるときに触るのは1か所で済む。

    省略した引数は各 ``TaskSpec`` の既定値になる (本番の既定と同じ)。

    Args:
        mackey_glass: MG の生成パラメータ。
        delay_parity: 遅延パリティの生成パラメータ。
        esn_mackey_glass: MG 課題で使うリザバー。
        esn_delay_parity: 遅延パリティ課題で使うリザバー。
        **rest: ``ExperimentConfig`` のそれ以外のフィールド。

    Returns:
        ``ExperimentConfig``。
    """
    mg = MackeyGlassTask(
        params=mackey_glass if mackey_glass is not None else MackeyGlassConfig(),
        **({"reservoir": esn_mackey_glass} if esn_mackey_glass is not None else {}),
    )
    dp = DelayParityTask(
        params=delay_parity if delay_parity is not None else DelayParityConfig(),
        **({"reservoir": esn_delay_parity} if esn_delay_parity is not None else {}),
    )
    return ExperimentConfig(tasks=(mg, dp), **rest)  # type: ignore[arg-type]

"""設定 YAML から dataclass を構築する共通ローダ (``config`` package の土台).

``config/`` は**実験サイクル単位**で割ってあり (``experiment01`` / ``esp02`` /
``capacity03``)、このモジュールはそのどれにも属さない読み込み規律だけを持つ:
未知キーで即失敗・暗黙の型変換をしない・dataclass のフィールド型から再帰構築
(D-09)。02 以降の実験がローダを写経すると D-09 の強度が実験ごとに割れる。

**``config`` package 内で他の設定モジュールを import しない** (D-49)。ここが
サイクル側を参照すると依存が双方向になり、01 の設定を読むためだけに 02・03 の
設定まで引き込まれる。01 向けの別名 ``load_config`` は
``ExperimentConfig`` を持つ ``experiment01`` 側に置いてある。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import UnionType
from typing import Protocol, TypeAliasType, cast, get_args, get_origin, get_type_hints

import yaml

from rc_basics_lab.overrides import KIND_KEY, apply_overrides, is_kinded_list


class _DataclassFactory[T_co](Protocol):
    """dataclass のコンストラクタ。``Any`` を書かずにキーワード構築を型付けする。"""

    def __call__(self, **kwargs: object) -> T_co: ...


class ConfigError(ValueError):
    """設定ファイルの内容が dataclass 群と噛み合わないときに送出される。"""


def _fail(location: str, message: str) -> ConfigError:
    return ConfigError(f"{location}: {message}")


def _coerce_scalar(value: object, target: type, location: str) -> object:
    """スカラ値を目標の型へ変換する。暗黙の切り捨てや bool→int は許さない。"""
    if target is bool:
        if isinstance(value, bool):
            return value
        raise _fail(location, f"真偽値が必要です: {value!r}")
    if target is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _fail(location, f"整数が必要です: {value!r}")
        return value
    if target is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _fail(location, f"数値が必要です: {value!r}")
        return float(value)
    if target is str:
        if not isinstance(value, str):
            raise _fail(location, f"文字列が必要です: {value!r}")
        return value
    raise _fail(location, f"未対応の設定型です: {target!r}")


def _coerce_tuple(value: object, annotation: object, location: str) -> object:
    """``tuple[X, ...]`` 型のフィールドを構築する。

    要素は**スカラとは限らない**。``tasks: tuple[TaskSpec, ...]`` のように
    判別子つき union のリストも受ける (D-123)。要素の変換を ``_coerce`` へ
    委ねるので、要素側で使える型はフィールド直下と同じである
    (スカラだけを許していると、設定をリストにした瞬間に読めなくなる)。
    """
    args = get_args(annotation)
    if len(args) != 2 or args[1] is not Ellipsis:
        raise _fail(location, f"未対応の tuple 型です: {annotation!r}")
    element_type = args[0]
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise _fail(location, f"リストが必要です: {value!r}")
    return tuple(
        _coerce(element, element_type, f"{location}[{index}]")
        for index, element in enumerate(value)
    )


def kind_of(cls: object) -> str | None:
    """union の要素が名乗る種別 (``KIND`` クラス変数)。無ければ ``None``。"""
    kind = getattr(cls, "KIND", None)
    return kind if isinstance(kind, str) else None


def _coerce_tagged_union(value: object, annotation: object, location: str) -> object:
    """``A | B`` (どちらも dataclass) を ``kind`` で選んで構築する。

    **ローダは具体的な型を1つも知らない。** 各 dataclass が ``KIND`` という
    ``ClassVar`` で自分の名前を名乗り、ローダはそれを突き合わせるだけである。
    リザバーのモデル名をローダに書くと、モデルを足すたびに ``config`` を
    触ることになる (``reservoir/registry.py`` に ``case`` を1つ足すだけ、
    という約束が崩れる)。

    ``kind`` を書かなければ**先頭の要素**を作る。既定を union の並び順で表す
    ので、既存の YAML はそのまま通る。

    Raises:
        ConfigError: マッピングでない / ``kind`` が未知 / 要素が ``KIND`` を
            名乗っていない場合。
    """
    members = [
        arg
        for arg in get_args(annotation)
        if dataclasses.is_dataclass(arg) and isinstance(arg, type)
    ]
    known = {name: cls for cls in members if (name := kind_of(cls)) is not None}
    if len(known) != len(members):
        raise _fail(location, f"KIND を名乗らない要素があります: {annotation!r}")
    if not isinstance(value, Mapping):
        raise _fail(location, f"マッピングが必要です: {value!r}")
    raw_kind = value.get(KIND_KEY)
    if raw_kind is None:
        return _build(members[0], value, location)
    if not isinstance(raw_kind, str) or raw_kind not in known:
        raise _fail(
            location,
            f"未知の {KIND_KEY} です: {raw_kind!r} (既知: {', '.join(sorted(known))})",
        )
    body = {key: item for key, item in value.items() if key != KIND_KEY}
    return _build(known[raw_kind], body, f"{location}.{raw_kind}")


def _coerce_optional(value: object, annotation: object, location: str) -> object:
    """``X | None`` 型のフィールドを構築する (``None`` は素通しする)。

    受理するのは ``None`` との2項 union だけで、``int | str`` のような
    「どちらでも良い」型は従来どおり ``ConfigError`` にする。YAML の値から
    どちらの型かを推測し始めると、``1`` を書いたのか ``"1"`` を書いたのかで
    挙動が変わる設定が生まれ、D-09 (未知キー・暗黙変換で落とす) の規律が
    崩れるため。

    ``None`` を許すのは「セクション側が名乗らなければ横断共有の値を継承する」
    という**片方向の上書き**を型で表すためで、既定値そのものをこちら側に
    書くことはしない (書くと二重定義になり、継承元を変えても効かなくなる)。
    """
    args = get_args(annotation)
    if len(args) != 2 or type(None) not in args:
        raise _fail(location, f"未対応の Union 型です: {annotation!r}")
    if value is None:
        return None
    (inner,) = (arg for arg in args if arg is not type(None))
    return _coerce(value, inner, location)


def _coerce(value: object, annotation: object, location: str) -> object:
    # ``type X = ...`` (PEP 695) は ``get_type_hints`` が解決せず TypeAliasType の
    # ままで返す。中身へ開かないと「未対応の設定型です: ReservoirConfig」で
    # 落ちる。**別名を挟んだだけで設定が読めなくなるのは、別名を付けた側の
    # 責任ではない**ので、ここで開く。
    if isinstance(annotation, TypeAliasType):
        return _coerce(value, annotation.__value__, location)
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return _build(annotation, value, location)
    origin = get_origin(annotation)
    if origin is tuple:
        return _coerce_tuple(value, annotation, location)
    if origin is UnionType:
        args = get_args(annotation)
        if type(None) not in args and all(
            dataclasses.is_dataclass(arg) for arg in args
        ):
            return _coerce_tagged_union(value, annotation, location)
        return _coerce_optional(value, annotation, location)
    if isinstance(annotation, type):
        return _coerce_scalar(value, annotation, location)
    raise _fail(location, f"未対応の設定型です: {annotation!r}")


def _build[T](cls: type[T], raw: object, location: str) -> T:
    """dataclass ``cls`` を ``raw`` (マッピング) から構築する。

    ``cls`` が ``KIND`` を名乗っていて ``raw`` に ``kind`` があれば、一致を
    確かめて取り除く。**union が1要素のうちから ``kind: esn`` を書けるように
    する**ためで、モデルが2つ目になった時点で YAML を書き換えずに済む
    (``_coerce_tagged_union`` が選ぶ側に回るだけ)。一致しなければ落とす ——
    書いたのに効かない ``kind`` を作らない。
    """
    if not isinstance(raw, Mapping):
        raise _fail(location, f"マッピングが必要です: {raw!r}")
    declared = kind_of(cls)
    if declared is not None and KIND_KEY in raw:
        written = raw[KIND_KEY]
        if written != declared:
            raise _fail(
                location,
                f"未知の {KIND_KEY} です: {written!r} (既知: {declared})",
            )
        raw = {key: item for key, item in raw.items() if key != KIND_KEY}
    known = {f.name for f in dataclasses.fields(cast("type", cls))}
    provided = {str(key) for key in raw}
    unknown = sorted(provided - known)
    if unknown:
        advice = ""
        if declared is not None:
            advice = (
                f"。kind: {declared} に無いキーです —— モデルを替えるときは"
                "セクションを丸ごと書き替えてください "
                "(プリセットなら kind を書けば自動で置き換わります)"
            )
        raise _fail(
            location,
            f"未知のキーです: {', '.join(unknown)}"
            f" (既知のキー: {', '.join(sorted(known))}){advice}",
        )
    hints = get_type_hints(cls)
    kwargs = {
        str(key): _coerce(value, hints[str(key)], f"{location}.{key}")
        for key, value in raw.items()
    }
    factory = cast("_DataclassFactory[T]", cls)
    return factory(**kwargs)


def _read_yaml(path: Path) -> dict[str, object]:
    """YAML を1つ読んでマッピングにする (空ファイルは空マッピング)。"""
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: YAML の解析に失敗しました: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path}: マッピングが必要です: {raw!r}")
    return {str(key): value for key, value in raw.items()}


def _changes_kind(base: object, patch: Mapping[str, object]) -> bool:
    """``patch`` が ``base`` と違う ``kind`` を名乗っているか。

    名乗っていれば**別の型の設定**なので、混ぜずに置き換える (下記)。
    """
    if KIND_KEY not in patch:
        return False
    if not isinstance(base, Mapping):
        return True
    return base.get(KIND_KEY) != patch[KIND_KEY]


def _unusedis_kinded_list(value: object) -> bool:
    """``kind`` を持つマッピングだけからなる、空でないリストか。"""
    if isinstance(value, str) or not isinstance(value, Sequence) or not value:
        return False
    return all(isinstance(item, Mapping) and KIND_KEY in item for item in value)


def _merge_kinded_list(
    base: Sequence[Mapping[str, object]], patch: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    """``kind`` で突き合わせて要素ごとに深く重ねる (D-123)。

    **並び順は本体のまま**である。プリセットの並びで上書きすると、プリセットに
    2件しか書かなかっただけで課題の順が変わり、``comparison.csv`` の行の順が
    プリセットごとに違うことになる。
    """
    overlay = {str(item[KIND_KEY]): item for item in patch}
    merged = [
        (
            _deep_merge(item, overlay.pop(str(item[KIND_KEY])))
            if str(item[KIND_KEY]) in overlay
            else dict(item)
        )
        for item in base
    ]
    merged.extend(dict(item) for item in overlay.values())
    return merged


def _deep_merge(
    base: Mapping[str, object], patch: Mapping[str, object]
) -> dict[str, object]:
    """``patch`` を ``base`` に深くかぶせる (プリセット用)。

    どちらもマッピングなら再帰し、そうでなければ ``patch`` で置き換える。
    **格子のリストは置き換える** —— ``alpha_grid`` などを部分的に混ぜると、
    プリセットを読んだだけでは実際に回る格子が分からなくなる。

    **``kind`` を持つ要素のリストだけは、``kind`` で突き合わせて重ねる**
    (``tasks``。D-123)。プリセットは「本体との差分だけを書く」規約なので、
    課題を1つ小さくするために課題定義を全部書き写すことになると、本体を
    直したときにプリセットだけ古いまま残る (実測でそういう複製が
    「効いていない設定」を生んだ)。プリセットに無い ``kind`` の要素は
    本体のまま残り、プリセットにしか無い ``kind`` は末尾に足される。

    **``kind`` が変わるセクションは丸ごと置き換える。** モデルを差し替えると
    設定の**型そのもの**が変わり、前のモデル固有のキー (ESN の ``density`` /
    ``activation`` など) は新しい型に存在しない。混ぜると必ず未知キーになり、
    「``kind`` を書いたのにモデルを替えられない」状態になる (実測でそうなった)。
    """
    merged = dict(base)
    for key, value in patch.items():
        current = merged.get(key)
        if is_kinded_list(current) and is_kinded_list(value):
            merged[key] = _merge_kinded_list(
                cast("Sequence[Mapping[str, object]]", current),
                cast("Sequence[Mapping[str, object]]", value),
            )
        elif isinstance(value, Mapping) and _changes_kind(current, value):
            merged[key] = dict(value)
        elif isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(
                cast("Mapping[str, object]", current),
                cast("Mapping[str, object]", value),
            )
        else:
            merged[key] = value
    return merged


def load_config_as[T](
    path: Path | str,
    cls: type[T],
    *,
    preset: Path | str | None = None,
    overrides: Sequence[str] = (),
) -> T:
    """YAML から任意の設定 dataclass ``cls`` を読み込む (D-13)。

    実験ごとに設定クラスは分かれるが、読み込み規律 (未知キーで即失敗・暗黙の
    型変換をしない・再帰構築) は1か所に置く。02 以降の実験がローダを写経すると
    D-09 の強度が実験ごとに割れるため。

    適用の順は **本体 YAML -> プリセット -> ``--set``** である。右のものが勝つ。
    どちらも**生のマッピングに適用してから** ``_build`` へ渡すので、未知キーの
    検査が上書きにもプリセットにも同じ強さで効く。

    Args:
        path: 本体の YAML ファイル。
        cls: 構築する設定 dataclass。
        preset: かぶせる YAML (``experiments/0N_*/presets/quick.yaml`` など)。
            差分だけを書く。``None`` なら何もかぶせない。
        overrides: ``--set`` の ``key.path=value`` の並び。

    Raises:
        ConfigError: ファイルが無い / 未知キーがある / 型が合わない場合。
        OverrideError: ``--set`` の書式か経路が不正な場合。
    """
    config_path = Path(path)
    raw = _read_yaml(config_path)
    if preset is not None:
        raw = _deep_merge(raw, _read_yaml(Path(preset)))
    if overrides:
        raw = apply_overrides(raw, overrides)
    return _build(cls, raw, str(config_path))

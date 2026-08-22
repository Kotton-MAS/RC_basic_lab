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
from typing import Protocol, cast, get_args, get_origin, get_type_hints

import yaml


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
    """``tuple[X, ...]`` 型のフィールドを構築する。"""
    args = get_args(annotation)
    if len(args) != 2 or args[1] is not Ellipsis:
        raise _fail(location, f"未対応の tuple 型です: {annotation!r}")
    element_type = args[0]
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise _fail(location, f"リストが必要です: {value!r}")
    return tuple(
        _coerce_scalar(element, element_type, f"{location}[{index}]")
        for index, element in enumerate(value)
    )


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
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return _build(annotation, value, location)
    origin = get_origin(annotation)
    if origin is tuple:
        return _coerce_tuple(value, annotation, location)
    if origin is UnionType:
        return _coerce_optional(value, annotation, location)
    if isinstance(annotation, type):
        return _coerce_scalar(value, annotation, location)
    raise _fail(location, f"未対応の設定型です: {annotation!r}")


def _build[T](cls: type[T], raw: object, location: str) -> T:
    """dataclass ``cls`` を ``raw`` (マッピング) から構築する。"""
    if not isinstance(raw, Mapping):
        raise _fail(location, f"マッピングが必要です: {raw!r}")
    known = {f.name for f in dataclasses.fields(cast("type", cls))}
    provided = {str(key) for key in raw}
    unknown = sorted(provided - known)
    if unknown:
        raise _fail(
            location,
            f"未知のキーです: {', '.join(unknown)}"
            f" (既知のキー: {', '.join(sorted(known))})",
        )
    hints = get_type_hints(cls)
    kwargs = {
        str(key): _coerce(value, hints[str(key)], f"{location}.{key}")
        for key, value in raw.items()
    }
    factory = cast("_DataclassFactory[T]", cls)
    return factory(**kwargs)


def load_config_as[T](path: Path | str, cls: type[T]) -> T:
    """YAML から任意の設定 dataclass ``cls`` を読み込む (D-13)。

    実験ごとに設定クラスは分かれるが、読み込み規律 (未知キーで即失敗・暗黙の
    型変換をしない・再帰構築) は1か所に置く。02 以降の実験がローダを写経すると
    D-09 の強度が実験ごとに割れるため。

    Args:
        path: YAML ファイルのパス。
        cls: 構築する設定 dataclass。

    Raises:
        ConfigError: ファイルが無い / 未知キーがある / 型が合わない場合。
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path}: YAML の解析に失敗しました: {exc}") from exc
    if raw is None:
        raw = {}
    return _build(cls, raw, str(config_path))

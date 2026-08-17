"""実験設定 (YAML) の読み込み.

frozen dataclass 群を単一の真実とし、YAML はそこへ値を流し込むだけにする。
**未知キーは即座に ``ConfigError``** とする (D-09)。本連載は十数個のパラメータを
YAML 化するため、キーのタイプミスが黙って無視されると「設定したのに効いていない」
実験結果が生まれる。

新しいセクションを足すときは dataclass にフィールドを追加するだけでよい
(ローダはフィールド型から再帰的に構築する)。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import UnionType
from typing import TypeVar, cast, get_args, get_origin, get_type_hints

import numpy as np
import yaml

from rc_basics_lab.seeds import SeedConfig

T = TypeVar("T")


class ConfigError(ValueError):
    """設定ファイルの内容が dataclass 群と噛み合わないときに送出される。"""


DEFAULT_ALPHA_GRID: tuple[float, ...] = tuple(
    float(value) for value in np.logspace(-8, 2, 11)
)
"""既定の ridge alpha 格子 (仕様 T3)。全手法・全タスクがこの単一格子を読む (D-04)。"""


@dataclass(frozen=True, slots=True)
class MackeyGlassConfig:
    """Mackey-Glass 系列の生成パラメータ (仕様 §3 未確定1 の決定値)。"""

    tau: float = 17.0
    beta: float = 0.2
    gamma: float = 0.1
    exponent: int = 10
    rk4_step: float = 0.1
    sample_interval: int = 10
    integration_burn_in: int = 1000
    length: int = 8000
    horizon: int = 1


@dataclass(frozen=True, slots=True)
class DelayParityConfig:
    """遅延パリティ課題の生成パラメータ (D-07)。"""

    n_bits: int = 2
    delay: int = 1
    length: int = 8000


@dataclass(frozen=True, slots=True)
class RidgeConfig:
    """リッジ回帰の設定。``alpha_grid`` は全手法が共有する単一キー (D-04)。"""

    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID
    n_lags_grid: tuple[int, ...] = (1, 2, 4, 8, 16)


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """時系列を連続区間で切る分割設定 (シャッフルしない)。"""

    train_ratio: float = 0.5
    val_ratio: float = 0.15
    test_ratio: float = 0.35
    washout: int = 200
    max_start_offset: int = 200


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """実験1本ぶんの設定。"""

    name: str = "01_what_is_rc"
    n_replicates: int = 5
    seeds: SeedConfig = field(default_factory=SeedConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    mackey_glass: MackeyGlassConfig = field(default_factory=MackeyGlassConfig)
    delay_parity: DelayParityConfig = field(default_factory=DelayParityConfig)


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


def _coerce(value: object, annotation: object, location: str) -> object:
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return _build(annotation, value, location)
    origin = get_origin(annotation)
    if origin is tuple:
        return _coerce_tuple(value, annotation, location)
    if origin is UnionType:
        raise _fail(location, f"未対応の Union 型です: {annotation!r}")
    if isinstance(annotation, type):
        return _coerce_scalar(value, annotation, location)
    raise _fail(location, f"未対応の設定型です: {annotation!r}")


def _build(cls: type[T], raw: object, location: str) -> T:
    """dataclass ``cls`` を ``raw`` (マッピング) から構築する。"""
    if not isinstance(raw, Mapping):
        raise _fail(location, f"マッピングが必要です: {raw!r}")
    known = {f.name for f in dataclasses.fields(cast("type", cls))}
    provided = {str(key) for key in raw}
    unknown = sorted(provided - known)
    if unknown:
        raise _fail(
            location,
            f"未知のキーです: {', '.join(unknown)} (既知のキー: {', '.join(sorted(known))})",
        )
    hints = get_type_hints(cls)
    kwargs = {
        str(key): _coerce(value, hints[str(key)], f"{location}.{key}")
        for key, value in raw.items()
    }
    constructor = cast("Callable[..., T]", cls)
    return constructor(**kwargs)


def load_config(path: Path | str) -> ExperimentConfig:
    """YAML から ``ExperimentConfig`` を読み込む。

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
    return _build(ExperimentConfig, raw, str(config_path))


__all__ = [
    "DEFAULT_ALPHA_GRID",
    "ConfigError",
    "DelayParityConfig",
    "ExperimentConfig",
    "MackeyGlassConfig",
    "RidgeConfig",
    "SplitConfig",
    "load_config",
]

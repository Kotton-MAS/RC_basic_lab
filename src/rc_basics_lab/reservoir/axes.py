"""掃引軸 — **モデルごとに違う軸を、同じ掃引コードで振る** (D-124).

## なぜ軸名を共通化しないのか

``leak_rate`` を持たないモデルに ``leak_rate`` の別名を生やすと、「同じ名前
だが意味が違う軸」ができて、図をまたいだ比較が静かに壊れる。**軸は各モデルの
言葉のままにし、集合が違うことを許す**。

掃引側は軸名を設定から受け取り、モデルがその軸を持たなければ
``require_esn`` と同じ調子で「そのモデルにその軸は無い。持っている軸はこれ」
と言って落とす。

## 何を軸とみなすか

**数値のフィールドだけ**である (``int`` / ``float``)。``activation`` のような
文字列や ``topology`` のような入れ子の設定は、格子で振る対象ではない
(結合構造は ``topology`` 自身の ``kind`` で選ぶ。D-122)。

モデルごとに軸の一覧を書き下さないのは、フィールドを1つ足したときに一覧の
更新を忘れる経路を作らないためである。**フィールドが軸である**。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

NUMERIC_TYPES: tuple[type, ...] = (int, float)
"""軸とみなすフィールドの型。``bool`` は ``int`` の派生だが、格子で振る量では
ないので ``numeric_axes`` が明示的に外す。"""


def numeric_axes(config: object) -> frozenset[str]:
    """設定が持つ数値フィールドの名前 (= 掃引できる軸)。

    Args:
        config: リザバーの設定 (frozen dataclass)。

    Returns:
        軸名の集合。

    Raises:
        TypeError: dataclass でない場合。
    """
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        raise TypeError(f"dataclass のインスタンスが必要です: {type(config).__name__}")
    return frozenset(
        item.name
        for item in dataclasses.fields(config)
        if isinstance(getattr(config, item.name), NUMERIC_TYPES)
        and not isinstance(getattr(config, item.name), bool)
    )


def require_axes(config: object, axes: Iterable[str], used_by: str) -> None:
    """モデルが要求された軸を全部持つことを確かめる。**無ければ落とす**。

    黙って素通しにすると、``kind`` を替えたときに「振ったはずの軸が効いて
    いない掃引」ができる。値が出てしまうぶん、落ちるより悪い。

    Args:
        config: リザバーの設定。
        axes: 掃引が振ろうとしている軸名。
        used_by: 呼び出し元の説明 (エラーに出す)。

    Raises:
        ValueError: 持っていない軸がある場合。
    """
    available = numeric_axes(config)
    missing = sorted(name for name in axes if name not in available)
    if missing:
        raise ValueError(
            f"{used_by} は軸 {missing} を振りますが、"
            f"{type(config).__name__} は持っていません "
            f"(持っている軸: {', '.join(sorted(available))})"
        )


def with_axis[T](config: T, name: str, value: float) -> T:
    """軸を1つ差し替えた複製を返す (D-124)。

    **``dataclasses.replace`` はキーワードが動的だと型が解けない** ——
    ``T`` が dataclass であることを型で示せないので ``type-var`` になる。
    軸名が実在することと値が数値であることは ``require_axes`` が実行時に
    確かめているので、**ignore を置くのはここ1箇所だけ**にする
    (呼び出し側に撒かない)。

    ``int`` の軸 (``n_units`` / ``n_layers``) に ``float`` を渡すと、
    ``dataclasses`` は黙って ``float`` のまま入れる。掃引の格子が
    ``tuple[int, ...]`` であることは設定側が保証する。

    Args:
        config: リザバーの設定 (frozen dataclass)。
        name: 軸名。
        value: 新しい値。``int`` の軸には整数を渡すこと (``dataclasses`` は
            変換しない)。

    Returns:
        同じ型の複製。

    Raises:
        ValueError: そのモデルに ``name`` という軸が無い場合。
    """
    require_axes(config, (name,), used_by=f"{type(config).__name__} の掃引")
    return dataclasses.replace(config, **{name: value})  # type: ignore[type-var]


def axis_value(config: object, name: str) -> float:
    """軸の現在値を返す (CSV の列に流すため)。

    Raises:
        ValueError: その軸が無い場合。
    """
    require_axes(config, (name,), used_by=f"軸 {name} の読み出し")
    return float(getattr(config, name))


__all__ = [
    "NUMERIC_TYPES",
    "axis_value",
    "numeric_axes",
    "require_axes",
    "with_axis",
]

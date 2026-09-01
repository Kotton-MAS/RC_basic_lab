"""設定 dataclass を「そのまま読み直せる」マッピングへ落とす (D-123).

``_common`` から分けてあるのは、``config/`` の1モジュール 300 行 (非空) という
上限があるためで、役割としても「読む側」と「書く側」で別である。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from rc_basics_lab.config._common import kind_of
from rc_basics_lab.overrides import KIND_KEY


def as_plain_mapping(config: object) -> dict[str, object]:
    """設定 dataclass を、**そのまま読み直せる**マッピングへ落とす (D-123)。

    ``dataclasses.asdict`` との違いは1点だけで、**リストの要素には ``kind`` を
    書く**。単一フィールドの union は「書かなければ先頭」という既定があるので
    位置から型が決まるが、リストの要素は位置では決まらない ——
    ``asdict`` の結果をそのまま読み直すと、2番目の課題が先頭の型
    (Mackey-Glass) として作られ、未知キーで落ちる。

    単一フィールドに ``kind`` を書かないのは既存の ``meta.json`` を動かさない
    ためである (``KIND_KEY`` の注を参照)。

    Args:
        config: frozen dataclass のインスタンス。

    Returns:
        YAML に書ける入れ子のマッピング。
    """
    plain = _plain(config, in_sequence=False)
    if not isinstance(plain, dict):
        raise TypeError(f"dataclass が必要です: {type(config).__name__}")
    return plain


def _plain(value: object, *, in_sequence: bool) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        body: dict[str, object] = {}
        kind = kind_of(type(value))
        if in_sequence and kind is not None:
            body[KIND_KEY] = kind
        for item in dataclasses.fields(value):
            body[item.name] = _plain(getattr(value, item.name), in_sequence=False)
        return body
    if isinstance(value, tuple | list):
        return [_plain(element, in_sequence=True) for element in value]
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item, in_sequence=False) for key, item in value.items()
        }
    return value


__all__ = ["as_plain_mapping"]

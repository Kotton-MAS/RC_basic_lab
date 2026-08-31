"""``--set key.path=value`` による設定の上書き.

YAML を複製せずに1つの値だけ振るための層。**``config`` パッケージの外に置く** ——
``config/_common.py`` は package 内の葉である必要があり (D-49)、そこから
``config`` 内の別モジュールを import できない。この層は特定の設定クラスを
知らない (扱うのはマッピングだけ) ので、外に出しても失うものが無い。

**読み込んだ生のマッピングに適用してから ``_build`` へ渡す**ので、既存の
読み込み規律 (未知キーで即失敗・暗黙の型変換をしない、D-09) がそのまま上書きにも
効く。上書き用の検査を別に書くと、YAML 側と CLI 側で厳しさが割れる。

値は **YAML として解釈する**。``n_units=200`` は int、``leak_rate=0.3`` は
float、``activation=tanh`` は str になる。独自の変換規則を作らないので、
「ファイルに書いたとき」と「CLI で振ったとき」で同じ文字列が同じ値になる。

**指数表記は ``1.0e-8`` と書く必要がある。** YAML 1.1 は ``1e-8`` を数値と
みなさず文字列にする (実測)。黙って文字列のまま通すと「数値が必要です:
'1e-8'」という、原因の分からないエラーが後段で出る。``parse_override`` が
その形を検出して**書き方まで示して落とす**。リポジトリの YAML も
``1.0e-10`` の形で書かれているので、規則は 1 つのままである。

**親のパスが無ければ失敗する。** ``esn_mackey_glas.n_units`` (親のタイプミス)
を黙って新しいキーとして作ると、「設定したのに効いていない」実験になる。
葉のタイプミス (``n_unit``) は ``_build`` の未知キー検査が捕まえる。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import cast

import yaml

SEPARATOR = "="
"""``key.path=value`` の区切り。値の側に ``=`` があっても最初の1つで割る。"""

PATH_SEPARATOR = "."
"""ネストしたキーの区切り。"""

_SCIENTIFIC = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)[eE][+-]?\d+$")
"""YAML 1.1 が数値として受けない指数表記 (``1e-8`` / ``1E-8`` / ``1e8``)。

``1.0e-8`` のように仮数に小数点があれば YAML は数値にする。落とす対象は
「利用者は数値のつもりだが YAML が文字列にした」形だけで、``tanh`` や
``zscore`` のような普通の文字列には当たらない。"""


class OverrideError(ValueError):
    """``--set`` の書式・経路が不正なときに送出する。"""


def parse_override(text: str) -> tuple[tuple[str, ...], object]:
    """``"a.b=1"`` を ``(("a", "b"), 1)`` にする。

    Args:
        text: ``key.path=value`` 形式の文字列。

    Returns:
        ``(キーの経路, YAML として解釈した値)``。

    Raises:
        OverrideError: ``=`` が無い / キーが空 / 値が YAML として壊れている場合。
    """
    if SEPARATOR not in text:
        raise OverrideError(f"--set は key.path=value の形で書いてください: {text!r}")
    raw_path, raw_value = text.split(SEPARATOR, 1)
    path = tuple(part.strip() for part in raw_path.split(PATH_SEPARATOR))
    if not path or any(not part for part in path):
        raise OverrideError(f"キーが空です: {text!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise OverrideError(f"値を解釈できません: {text!r} ({exc})") from exc
    if isinstance(value, str) and _SCIENTIFIC.match(value.strip()):
        raise OverrideError(
            f"{text}: 指数表記は仮数に小数点が要ります "
            f"(YAML 1.1 は {value!r} を文字列とみなします)。"
            f"{_with_decimal_point(value.strip())} と書いてください"
        )
    return path, value


def _with_decimal_point(text: str) -> str:
    """``1e-8`` を ``1.0e-8`` にする (エラーメッセージ用の書き換え例)。"""
    mantissa, marker, exponent = text.partition("e" if "e" in text else "E")
    if "." not in mantissa:
        mantissa = f"{mantissa}.0"
    return f"{mantissa}{marker}{exponent}"


def apply_overrides(
    raw: Mapping[str, object], overrides: Sequence[str]
) -> dict[str, object]:
    """生のマッピングに ``--set`` を順に適用した**新しい** dict を返す。

    元の ``raw`` は変更しない (同じ設定を2回読む経路があるため)。

    Args:
        raw: YAML を読んだ結果。
        overrides: ``key.path=value`` の並び。左から順に適用する。

    Returns:
        上書き後のマッピング。

    Raises:
        OverrideError: 書式が不正、または**親のパスが存在しない**場合。
    """
    result = _deep_copy(raw)
    for text in overrides:
        path, value = parse_override(text)
        _assign(result, path, value, text)
    return result


def _deep_copy(raw: Mapping[str, object]) -> dict[str, object]:
    """マッピングだけを再帰的に複製する (リストや値はそのまま共有する)。"""
    return {
        str(key): _deep_copy(cast("Mapping[str, object]", value))
        if isinstance(value, Mapping)
        else value
        for key, value in raw.items()
    }


def _assign(
    target: dict[str, object], path: tuple[str, ...], value: object, text: str
) -> None:
    """``path`` の位置に ``value`` を書く。親が無ければ ``OverrideError``。"""
    node = target
    for depth, part in enumerate(path[:-1]):
        child = node.get(part)
        if not isinstance(child, dict):
            location = PATH_SEPARATOR.join(path[: depth + 1])
            known = ", ".join(sorted(node)) or "(空)"
            raise OverrideError(
                f"{text}: {location} という設定はありません (既知: {known})"
            )
        node = child
    node[path[-1]] = value


__all__ = ["OverrideError", "apply_overrides", "parse_override"]

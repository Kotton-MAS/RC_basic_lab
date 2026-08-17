"""図のラベル文字列 — 日本語 / 英語の切り替え (D-10).

CJK フォントが見つからない環境では、**フォント設定だけを戻すのではなく
ラベル文字列ごと英語に切り替える**。フォントだけをフォールバックすると、
CI 上で豆腐文字 (□□□) の図が「正常に」生成されてしまい、生成の成否では
検出できない壊れ方をするため。
"""

from __future__ import annotations


def label(ja: str, en: str, *, cjk: bool) -> str:
    """日本語ラベルと英語ラベルを選ぶ。

    Args:
        ja: 日本語のラベル。
        en: 英語のラベル (CJK フォントが無いときに使う)。
        cjk: CJK フォントが利用可能か (``StyleContext.cjk_available``)。

    Returns:
        ``cjk`` が真なら ``ja``、偽なら ``en``。

    Raises:
        ValueError: どちらかが空文字の場合 (英語ラベルの書き忘れを落とす)。
    """
    if not ja or not en:
        raise ValueError(f"ja / en の両方が必要です: ja={ja!r}, en={en!r}")
    return ja if cjk else en


__all__ = ["label"]

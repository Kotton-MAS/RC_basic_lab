"""図のラベル文字列 — 日本語 / 英語の切り替え (D-10).

CJK フォントが見つからない環境では、**フォント設定だけを戻すのではなく
ラベル文字列ごと英語に切り替える**。フォントだけをフォールバックすると、
CI 上で豆腐文字 (□□□) の図が「正常に」生成されてしまい、生成の成否では
検出できない壊れ方をするため。

**参照線の出典もここが単一の真実である** (図の設計方針 FIG-3、D-84)。本文の
注は図が切り取られると消えるので、出典は凡例テキストの中に入れる。出典文字列
は ja / en で変えない (人名と年は翻訳しない) ため、``label`` の対ではなく
素の文字列として持つ。
"""

from __future__ import annotations

METHOD_LABELS: dict[str, tuple[str, str]] = {
    "linear": ("線形", "linear"),
    "delay_line": ("遅延線 (リッジ)", "delay line (ridge)"),
    "delay_line_ols": ("遅延線 (OLS)", "delay line (OLS)"),
    "esn": ("ESN", "ESN"),
}
"""手法名の表示ラベル (ja / en の対、D-10)。

キーは ``style.METHOD_COLORS`` と同じ集合であること
(``test_plotting_style.py::test_method_labels_and_colours_cover_the_same_methods``)。
遅延線に ``(リッジ)`` を付けるのは、先行 (Goudarzi et al. 2014) の遅延線が
**正則化なし OLS** で、そこが 3-C の論点そのものだからである。読者が図だけを
見たときに、どちらの遅延線かが分かる必要がある。
"""

JAEGER_2002 = "Jaeger 2002"
DAMBRE_2012 = "Dambre 2012"
GOUDARZI_2014 = "Goudarzi et al. 2014"
VISWANATH_1998 = "Viswanath 1998"
"""図の中に書く出典 (著者 年)。"""

MC_BOUND_SOURCE = f"{JAEGER_2002} / {DAMBRE_2012}"
"""線形メモリ容量の上限 ``MC <= N`` の出典。"""

IPC_BOUND_SOURCE = DAMBRE_2012
"""情報処理容量の保存則 ``IPC_total <= N`` の出典。"""

SOURCE_UNIDENTIFIED: tuple[str, str] = ("原典未特定", "source unidentified")
"""出典を特定できていない参照値に付す印 (survey 未解決1)。ja / en の対。

**空欄にしない。** 「出典が無い」も情報であり、黙って値だけ引くと、後から
出典が違っていたときに図の側から辿れない。
"""


def cited(text: str, source: str) -> str:
    """参照線の凡例テキストに出典を付す (FIG-3、D-84)。

    Args:
        text: 参照線が何かを述べる部分 (``上限 MC <= N = 200`` など)。
        source: 出典 (``Jaeger 2002 / Dambre 2012``) か
            ``SOURCE_UNIDENTIFIED``。

    Raises:
        ValueError: ``text`` か ``source`` が空の場合。値だけ足して出典を
            書き忘れる事故を**描画前に**落とす (``figures_capacity`` の
            「ラベルが欠けたら描く前に落とす」と同じ規律)。
    """
    if not text or not source:
        raise ValueError(f"参照線には出典が要ります: text={text!r}, source={source!r}")
    return f"{text} [{source}]"


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


__all__ = [
    "DAMBRE_2012",
    "GOUDARZI_2014",
    "IPC_BOUND_SOURCE",
    "JAEGER_2002",
    "MC_BOUND_SOURCE",
    "METHOD_LABELS",
    "SOURCE_UNIDENTIFIED",
    "VISWANATH_1998",
    "cited",
    "label",
]

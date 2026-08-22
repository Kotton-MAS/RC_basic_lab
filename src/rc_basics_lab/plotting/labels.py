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
VINCKIER_2015 = "Vinckier et al. 2015"
APPELTANT_2011 = "Appeltant et al. 2011"
GAUTHIER_2021 = "Gauthier et al. 2021"
"""図の中に書く出典 (著者 年)。"""

MC_BOUND_SOURCE = f"{JAEGER_2002} / {DAMBRE_2012}"
"""線形メモリ容量の上限 ``MC <= N`` の出典。"""

IPC_BOUND_SOURCE = DAMBRE_2012
"""情報処理容量の保存則 ``IPC_total <= N`` の出典。"""

SOURCE_UNIDENTIFIED: tuple[str, str] = ("原典未特定", "source unidentified")
"""出典を特定できていない参照値に付す印。ja / en の対。

**空欄にしない。** 「出典が無い」も情報であり、黙って値だけ引くと、後から
出典が違っていたときに図の側から辿れない。

NARMA10 の参照値 (0.16 / 0.107) は D-100 で原典を特定したので、この印は
**現在どの参照線にも使われていない**。次に出典不明の値を引くときのために
残してある (``cited_measurement`` の引数として渡せる形)。
"""


def _with_source(text: str, source: str, conditions: str) -> str:
    """``text [source]`` / ``text [source; conditions]`` を組む。"""
    if not text or not source:
        raise ValueError(f"参照線には出典が要ります: text={text!r}, source={source!r}")
    inside = source if not conditions else f"{source}; {conditions}"
    return f"{text} [{inside}]"


def cited_bound(text: str, source: str) -> str:
    """**理論上限・定義由来**の参照線に出典を付す (FIG-3 / D-84)。

    上限は動作点に依らないので条件を求めない (``MC <= N`` は N がいくつでも
    上限である)。D-84 の rationale が「定義由来の線 (1/e、``-1/log(1-a)``) には
    出典を求めない」と書いているのと同じ線引きで、**そちら側の入口**にあたる。

    Args:
        text: 参照線が何かを述べる部分 (``上限 MC <= N = 200`` など)。
        source: 出典 (``Jaeger 2002 / Dambre 2012`` など)。

    Raises:
        ValueError: ``text`` か ``source`` が空の場合。
    """
    return _with_source(text, source, "")


def cited_measurement(text: str, source: str, conditions: str) -> str:
    """**文献で測られた値**に出典と**動作点**を付す (D-97)。

    実測値は測った条件でしか意味を持たない。条件を書かずに値だけ引くと、
    読者は「その数字と自分の数字が比較可能か」を判断できないまま、
    比較可能だと受け取る。

    実例 (D-95): 3-C は先行 (Goudarzi et al. 2014) の「正則化なし」だけを
    再現して「先行の対照を足した」と書いたが、先行の動作点は
    ``k ≈ n_train`` でこちらは ``k/n_train <= 0.01`` だった。
    **手法は再現したが動作点は再現していない。** 条件を必須の引数にすると、
    この取り違えが引用を書く時点で1度は目に入る。

    Args:
        text: 参照線が何かを述べる部分 (``先行の動作点 k/n = 0.91`` など)。
        source: 出典。原典が特定できていなければ ``SOURCE_UNIDENTIFIED``。
        conditions: その値が測られた**動作点** (``1,810 タップ / 訓練 2,000 点``
            や ``N = 50 規模``)。**空にできない。**

    Raises:
        ValueError: ``text`` / ``source`` / ``conditions`` のいずれかが空の場合。
    """
    if not conditions:
        raise ValueError(
            f"文献の実測値には動作点が要ります: text={text!r}, source={source!r}。"
            "測った条件が分からない値は、読者が比較可能かを判断できません。"
            "理論上限や定義由来の線なら cited_bound を使ってください (D-97)。"
        )
    return _with_source(text, source, conditions)


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
    "APPELTANT_2011",
    "DAMBRE_2012",
    "GAUTHIER_2021",
    "GOUDARZI_2014",
    "IPC_BOUND_SOURCE",
    "JAEGER_2002",
    "MC_BOUND_SOURCE",
    "METHOD_LABELS",
    "SOURCE_UNIDENTIFIED",
    "VINCKIER_2015",
    "VISWANATH_1998",
    "cited_bound",
    "cited_measurement",
    "label",
]

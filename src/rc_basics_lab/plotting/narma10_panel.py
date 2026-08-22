"""実験 3-C (NARMA10) のパネルが使う「行 -> 文字列」の変換.

``figures_capacity.py`` は D-77 の凍結対象なので、3-C の第4水準 (D-90) を
足すぶんはここへ出す。**描画を含まない** —— 入力は ``ResultRow`` の列で、
出力は横軸ラベルとタイトルの結論文だけである。

結論文を関数にしているのは、FIG-1 が「結論をタイトルに書く」形式だからで、
文を固定すると結果が反転したときに図が静かに嘘をつく。
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from rc_basics_lab.experiment.runner import (
    DELAY_LINE,
    DELAY_LINE_OLS,
    ESN_METHOD,
    LINEAR,
    ResultRow,
)
from rc_basics_lab.plotting.labels import (
    APPELTANT_2011,
    GOUDARZI_2014,
    METHOD_LABELS,
    VINCKIER_2015,
)
from rc_basics_lab.plotting.style import StyleContext

REFERENCE_LABELS: dict[str, tuple[str, str]] = {
    "linear_ceiling": ("参照 NMSE = {value:g}", "reference NMSE = {value:g}"),
    "nonlinear_rc": ("参照 NMSE = {value:g}", "reference NMSE = {value:g}"),
}
"""3-C の参照線の凡例 (``NARMA10_REFERENCE_NMSE`` のキーに対応)。

**キーが増減したら図を描く前に落とす** (``figures_capacity._reference_lines``)。
値だけを実験層に置いて凡例を図の側に持つと、参照点を1本足したときに図から
静かに消える。
"""

REFERENCE_CONDITIONS: dict[str, tuple[str, str]] = {
    "linear_ceiling": ("線形シフトレジスタの上限", "linear shift-register bound"),
    "nonlinear_rc": ("N = 50・訓練 1000 点", "N = 50, 1000 training points"),
}
"""参照値が測られた**動作点** (D-97 / D-100)。``REFERENCE_LABELS`` と同じキー。

出典を特定したので (D-100)、動作点も原典の記述に合わせてある ——
0.107 は Vinckier et al. 2015 が N = 50 / 訓練 1000 点で測った実験値、
0.16 は同論文が Appeltant et al. 2011 に帰す線形シフトレジスタの上限である。
"""

REFERENCE_SOURCES: dict[str, str] = {
    "linear_ceiling": APPELTANT_2011,
    "nonlinear_rc": VINCKIER_2015,
}
"""参照値の出典 (D-100)。``REFERENCE_LABELS`` と同じキー。

**キーが欠けたら描画前に落ちる** (``figures_capacity._reference_lines`` が
``cited_measurement`` を通すため)。値だけ足して出典を書き忘れる経路を残さない。
"""
"""参照値が測られた**動作点** (D-97)。``REFERENCE_LABELS`` と同じキー。

値の説明とは別に持つのは、``cited_measurement`` が動作点を必須の引数として
要求するからである。ラベル本文に混ぜて書くと「書いたつもり」で空欄を通せて
しまい、条件を必須にした意味が消える。
"""


def narma10_method_labels(
    rows: Sequence[ResultRow], style: StyleContext
) -> tuple[tuple[str, str], ...]:
    """横軸の (手法名, 表示ラベル) を出現順で返す。

    遅延線のラベルには**検証分割で選ばれたタップ数 k** を入れる (D-08 が
    選ぶことを許した唯一の構造パラメータであり、対照の強さそのもの)。
    レプリケートごとに違う k が選ばれたら全部並べる —— 代表値を1つ選ぶと
    「ばらついている」という事実が図から消える。
    """
    seen = tuple(dict.fromkeys(row.method for row in rows))
    # 正則化の有無だけが違う2水準は**隣に置く** (D-90)。行の出現順に任せると
    # 間に ESN が挟まり、図の主題である「alpha だけを動かした対照」が読めない。
    order = (LINEAR, DELAY_LINE, DELAY_LINE_OLS, ESN_METHOD)
    methods = tuple(name for name in order if name in seen) + tuple(
        name for name in seen if name not in order
    )
    labels: list[tuple[str, str]] = []
    for method in methods:
        pair = METHOD_LABELS.get(method)
        text = method if pair is None else style.label(*pair)
        if method in {DELAY_LINE, DELAY_LINE_OLS}:
            lags = sorted({row.n_lags for row in rows if row.method == method})
            text = f"{text}\n(k = {', '.join(str(value) for value in lags)})"
        labels.append((method, text))
    return tuple(labels)


def narma10_headline(rows: Sequence[ResultRow], style: StyleContext) -> str:
    """タイトルの結論文を**行から導く** (D-90)。

    FIG-1 は「結論をタイトルに書く」形式なので、文を固定すると結果が反転した
    ときに図が静かに嘘をつく (このサイクルで実際に5枚見つかっている)。
    正則化の有無で ESN との順位が変わったかを毎回数え直して文を選ぶ。
    """
    means = {
        method: statistics.fmean([row.nmse for row in rows if row.method == method])
        for method in {row.method for row in rows}
    }
    esn = means.get(ESN_METHOD)
    ridge = means.get(DELAY_LINE)
    ols = means.get(DELAY_LINE_OLS)
    if esn is None or ridge is None or ols is None:
        return style.label(
            "探索予算をそろえた対照での NARMA10",
            "NARMA10 under a search-budget-matched control",
        )
    if (ridge < esn) == (ols < esn):
        return style.label(
            "遅延線が ESN を下回るのは正則化のおかげではない",
            "regularisation is not why the delay line beats the ESN",
        )
    return style.label(
        "正則化の有無で遅延線と ESN の順位が入れ替わる",
        "regularisation flips the ranking of the delay line and the ESN",
    )


def narma10_subtitle(style: StyleContext) -> str:
    """3-C の副題 (D-95)。

    **動作点を書く** —— 先行 (Goudarzi et al. 2014) の対照は
    「正則化なし かつ k ≈ n_train」であり、3-C の動作点は
    ``k/n_train <= 0.01`` でそこに届いていない。副題が「先行の対照を足した」
    とだけ書いていると、読者は先行の設計が検証されたと受け取る。
    タップ数の軸は 3-C' (``figures_narma_taps``) が受け持つ。
    """
    return style.label(
        f"{GOUDARZI_2014} の対照 (正則化なし) を第4水準に足した"
        " (同一分割・同一特徴。alpha だけが違う)\n"
        "この動作点は k/n_train <= 0.01 で先行の k/n ≈ 0.9 とは違う"
        " (タップ数の軸は 3-C')",
        f"Added the unregularised control of {GOUDARZI_2014} as a fourth level"
        " (identical splits and features; only alpha differs)\n"
        "This operating point has k/n_train <= 0.01, unlike the prior"
        " k/n ~ 0.9 (see 3-C' for the tap-count axis)",
    )


__all__ = [
    "REFERENCE_CONDITIONS",
    "REFERENCE_LABELS",
    "REFERENCE_SOURCES",
    "narma10_headline",
    "narma10_method_labels",
    "narma10_subtitle",
]

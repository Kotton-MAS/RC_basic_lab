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
from rc_basics_lab.plotting.labels import METHOD_LABELS
from rc_basics_lab.plotting.style import StyleContext


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


__all__ = ["narma10_headline", "narma10_method_labels"]

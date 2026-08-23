"""実験 04 の図が使う結論文の生成 (D-96).

``figures_freerun.py`` は D-77 の凍結対象なので、結論文をここへ出す。
**描画を含まない** —— 入力は行、出力はタイトルの1文である。

結論文を関数にしているのは、FIG-1 が「結論をタイトルに書く」形式で、
文を固定すると結果が変わったときに図が静かに嘘をつくためである
(このサイクルで誤ったタイトルが実際に5枚見つかっている)。
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from rc_basics_lab.experiment.freerun import FreeRunRow
from rc_basics_lab.plotting.labels import METHOD_LABELS
from rc_basics_lab.plotting.style import StyleContext

LITERATURE_VALID_TIME = 5.0
"""文献が報告する Lorenz63 の有効予測時間 [1/lambda_max] (D-102)。

一次資料は Gauthier, Bollt, Griffith, Barbosa (2021)
*Next generation reservoir computing*, Nat. Commun. **12**:5564 で、本文に

    The NG-RC forecasts well out to ~5 Lyapunov times.

とある。同じ段落が **「100s to 1000s of reservoir nodes を持つ最適化された
従来型 RC に匹敵する」**と書いており、こちらの ``N = 200`` はその範囲に入る。

同論文の Lorenz63 の Lyapunov 時間は **1.1 時間単位**で、こちらの数値推定
(``1 / lambda_max = 1 / 0.9161 = 1.0916``) と実質一致する。**縦軸の単位が
そろっている**ので、そのまま同じ図に引ける。
"""

LITERATURE_VALID_TIME_CONDITIONS: tuple[str, str] = (
    "従来型 RC 100〜1000 ノード相当・判定基準は定量的に未明示",
    "comparable to a traditional RC with 100s-1000s nodes;"
    " no quantitative threshold stated",
)
"""参照値の**動作点** (D-97 / D-102)。

判定基準を条件に含めるのは、原典が『forecasts well out to ~5』という
定性的な言い方をしており、**こちらの閾値 0.4 と同じ基準ではない**ためである。
読者がその差を図の上で判断できるようにする。
"""


def valid_time_headline(rows: Sequence[FreeRunRow], style: StyleContext) -> str:
    """4-B のタイトルの結論文を**行から導く** (D-96).

    文献値 (``LITERATURE_VALID_TIME``) を特定して参照線を引けるように
    なったので (D-102)、**水準についても述べる** —— ただし述べるのは
    「文献の水準に届いているか」であって「良いか悪いか」ではない。
    判定基準が原典と完全には同じでないため (``LITERATURE_VALID_TIME_CONDITIONS``)、
    比は目安として扱う。
    """
    by_method: dict[str, list[float]] = {}
    for row in rows:
        by_method.setdefault(row.method, []).append(row.valid_time_lyapunov)
    medians = {
        method: statistics.median(values)
        for method, values in by_method.items()
        if values
    }
    if len(medians) < 2:
        return style.label(
            "自走が真の軌道からずれるまでの時間 (Lyapunov 時間で正規化)",
            "how long the free run stays on the true trajectory (in Lyapunov times)",
        )
    best = max(medians, key=lambda method: (medians[method], method))
    worst = min(medians, key=lambda method: (medians[method], method))
    ratio = medians[best] / medians[worst] if medians[worst] > 0.0 else float("inf")
    best_label = style.label(*METHOD_LABELS[best])
    worst_label = style.label(*METHOD_LABELS[worst])
    share = medians[best] / LITERATURE_VALID_TIME
    return style.label(
        f"{best_label} の有効予測時間は {medians[best]:.2f} Lyapunov 時間 —— "
        f"{worst_label} の {ratio:.0f} 倍で、文献値のおよそ {share:.0%}",
        f"the valid time of {best_label} is {medians[best]:.2f} Lyapunov times:"
        f" {ratio:.0f}x that of {worst_label}, about {share:.0%} of the"
        " literature value",
    )


__all__ = [
    "LITERATURE_VALID_TIME",
    "LITERATURE_VALID_TIME_CONDITIONS",
    "valid_time_headline",
]

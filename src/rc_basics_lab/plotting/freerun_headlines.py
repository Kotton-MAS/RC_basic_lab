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


def valid_time_headline(rows: Sequence[FreeRunRow], style: StyleContext) -> str:
    """4-B のタイトルの結論文を**行から導く** (D-96).

    ここで**水準の良否を主張しない**。「数 Lyapunov 時間」という言い方は
    それが妥当な水準だと述べているが、図の中に文献の有効予測時間の参照線が
    無いので、読者はその判断を確かめられない。引くべき文献値と条件
    (N・入力次元・観測ノイズ) は未特定である
    (``docs/図の設計方針_RC基礎編.md`` の未解決1)。

    したがってこの図が言えるのは**手法間の差**であり、結論文もそこに限る。
    文献値が特定できたら参照線を引き、水準の主張へ戻せる。
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
    return style.label(
        f"有効予測時間の中央値は {best_label} が {medians[best]:.2f} Lyapunov 時間で、"
        f"{worst_label} の {ratio:.0f} 倍",
        f"the median valid time is {medians[best]:.2f} Lyapunov times for"
        f" {best_label}, {ratio:.0f}x that of {worst_label}",
    )


__all__ = ["valid_time_headline"]

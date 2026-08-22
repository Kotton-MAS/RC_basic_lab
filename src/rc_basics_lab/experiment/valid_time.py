"""有効予測時間の閾値の定義 (D-43 / D-101).

``attractor.py`` は 600 行上限 (D-77) を超えて凍結されているので、
閾値まわりの定義はここへ置く。**判定そのものは含まない** ——
入力は無く、出力は定数だけである。

文献との比較でつまずく点が1つあるので、その換算をここに閉じてある
(下の ``LITERATURE_VPT_THRESHOLD`` を参照)。
"""

from __future__ import annotations

LITERATURE_VPT_THRESHOLD = 0.632_455_532_033_675_9
"""文献の VPT 定義に対応する NRMSE 比の閾値 (D-101)。``sqrt(0.4)``。

RC のカオス予測で広く使われる有効予測時間 (VPT) は
**正規化二乗差が 0.4 を超えるまで**と定義される
(Platt et al. 2022, *Neural Networks* 153:530)。一方こちらの
有効予測時間は **NRMSE 比** が閾値を超えるまでである (D-43)。
**二乗の分だけ定義がずれる**ので、同じ「0.4」でも別の量を測っている。

比較したいときに換算し忘れる形を消すため、対応する点を格子に入れて
``valid_time_sensitivity`` に必ず出す。**文献と並べるときはこの列を使う。**
"""

VALID_TIME_THRESHOLD_GRID: tuple[float, ...] = (
    0.2,
    0.3,
    0.4,
    0.5,
    LITERATURE_VPT_THRESHOLD,
)
"""有効予測時間の閾値感度 (仕様 §8)。``docs/design.md`` §12 の感度表の一次資料。

本番の閾値 (``freerun.valid_time_threshold``) はこの格子とは**独立**に設定から
来る。格子は「閾値の取り方で結論が変わらないこと」を示すためだけに使う。

末尾の ``LITERATURE_VPT_THRESHOLD`` だけは役割が違い、**文献の定義に
対応する点**である (D-101)。ここが無いと、文献の VPT と自分の値を
定義の違いに気づかないまま並べることになる。
"""


__all__ = ["LITERATURE_VPT_THRESHOLD", "VALID_TIME_THRESHOLD_GRID"]

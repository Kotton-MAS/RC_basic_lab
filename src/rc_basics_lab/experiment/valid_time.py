"""有効予測時間の閾値の定義 (D-43 / D-101).

``attractor.py`` は 600 行上限 (D-77) を超えて凍結されているので、
閾値まわりの定義はここへ置く。**判定そのものは含まない** ——
入力は無く、出力は定数だけである。
"""

from __future__ import annotations

LITERATURE_VPT_THRESHOLD = 0.3
"""文献の VPT と比較するときに使う閾値 (D-101)。

RC のカオス予測で使われる有効予測時間 (VPT) の一次資料は
Platt et al. (2022) *Neural Networks* 153:530 で、定義はこうである:

    RMSE(t) = sqrt( (1/D) * sum_i [ (u_i^f(t) - u_i(t)) / sigma_i ]^2 ) > eps

``eps`` は「arbitrarily to 0.3」と書かれている (同論文が参照する先行研究は
0.5 を使う、とも書かれている)。**平方根が入っている**ので、こちらの
NRMSE 比 (D-43) と同じ次元の量であり、換算は要らない。

**この定数は「格子のどの点が文献比較用か」を名指しするためにある。**
値そのものは ``VALID_TIME_THRESHOLD_GRID`` に元から入っていた。

正規化の仕方だけは完全には一致しない: 一次資料は成分ごとに
``sigma_i`` で割ってから D 次元で二乗平均するのに対し、こちらは
真の系列の標準偏差1つで割る (D-43)。Lorenz は成分ごとに分散が違うので、
**同じ閾値でも厳密に同じ量ではない**。並べるときはこの差を注記すること。
"""

VALID_TIME_THRESHOLD_GRID: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5)
"""有効予測時間の閾値感度 (仕様 §8)。``docs/design.md`` §12 の感度表の一次資料。

本番の閾値 (``freerun.valid_time_threshold``) はこの格子とは**独立**に設定から
来る。格子は「閾値の取り方で結論が変わらないこと」を示すためだけに使う。

``LITERATURE_VPT_THRESHOLD`` (= 0.3) がこの中に入っていること自体が
文献比較の前提なので、``tests/test_experiment_freerun.py`` が固定する。
"""

__all__ = ["LITERATURE_VPT_THRESHOLD", "VALID_TIME_THRESHOLD_GRID"]

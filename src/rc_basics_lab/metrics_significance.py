"""有意性の指標 (符号検定) —— **純関数層** (D-59 / D-91).

``experiment/freerun.py`` (04 のアトラクタ再現) と
``experiment/anomaly_ranking.py`` (05 の対照比較) が、同じ閉形式

    P(X >= m),  X ~ Binomial(n, 1/2)  =  sum_(k >= m) C(n, k) / 2**n

を**独立に実装していた**。同じ数式でありながら境界の振る舞いが違い
(対の数が 0 のとき一方は ``nan``、他方は ``1.0``)、どちらも記事に載る
数値を作っていた。ここが単一の真実である (D-91)。

この層はネットワークもファイル I/O も持たず、``experiment`` からの一方向
依存だけを受ける。
"""

from __future__ import annotations

import math

__all__ = ["sign_test_p_value"]


def sign_test_p_value(n_pairs: int, n_successes: int) -> float:
    """片側符号検定の p 値 (帰無仮説「成功確率は 1/2」)。

    対応のある比較にするのが要点で、対ごとの符号を数えるほうが平均どうしの
    比較より系列間のばらつきに振り回されない (05 の UCR では s.d. が平均と
    同じ大きさになる)。10 対が全部同じ向きなら約 0.001 で、「有意に近い」
    (D-46) の根拠を数値で残せる。

    Args:
        n_pairs: 対の数。
        n_successes: そのうち評価対象が対照を上回った数。

    Returns:
        ``P(X >= n_successes)`` (``X ~ Binomial(n_pairs, 0.5)``)。
        **対が 0 なら 1.0** —— 対が無ければ ``X = 0`` かつ
        ``n_successes = 0`` なので、閉形式の値そのものが 1 である
        (``nan`` を返す実装もあったが、それは「データが無い」という別の
        情報を p 値の位置に混ぜる形だった。D-91)。

    Raises:
        ValueError: 数が負、または ``n_successes > n_pairs`` の場合。
    """
    if n_pairs < 0 or n_successes < 0:
        raise ValueError(f"対の数は 0 以上が必要です: {n_pairs}, {n_successes}")
    if n_successes > n_pairs:
        raise ValueError(
            f"n_successes が n_pairs を超えています: {n_successes} > {n_pairs}"
        )
    if n_pairs == 0:
        return 1.0
    tail = sum(math.comb(n_pairs, k) for k in range(n_successes, n_pairs + 1))
    return tail / float(2**n_pairs)

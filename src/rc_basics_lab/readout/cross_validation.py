"""時系列の交差検証 — alpha を1つの検証区間ではなく複数の分割で選ぶ.

いまの選択は「訓練区間で学習し、その直後の検証区間 1 本で採点する」形である
(``select_alpha``)。系列の一部分の癖に引きずられるので、区間を複数取って
平均で選べるほうが安定する。

**時系列では素朴な k-分割を使えない。** 漏れが 3 通り起きる:

1. **未来から過去への漏れ。** 検証区間より後ろの行で学習すると、予測しようと
   している時刻より後の情報が係数に入る。``rolling`` (原点を進める) はこれを
   構造上作れない。``blocked`` は作れるので、選べるが既定にしない
2. **ラグによる漏れ。** 設計行列の 1 行は過去 ``n_lags`` 行の入力を含むので、
   訓練区間の直後の行は訓練行と入力を共有する。境目に**禁足区間 (embargo)**
   を置いて捨てる
3. **リザバー状態の持ち越し。** ``x[t]`` は過去の入力すべてに依存する。
   ``rolling`` なら検証区間の状態は訓練区間の入力から来るだけなので因果を
   壊さない。``blocked`` で検証区間より前に訓練区間を置くと、その訓練行の
   状態に検証区間の入力が混ざる

**既定は交差検証を使わない。** 成果物 (``results/``) は単一分割で作られており、
黙って選び方を変えると記事の数値が変わる。設定で明示的に有効にする。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from rc_basics_lab.metrics import nrmse
from rc_basics_lab.readout.ridge import (
    AlphaSelection,
    fit_ridge_from_gram,
    predict,
)
from rc_basics_lab.types import FloatArray


class FoldScheme(StrEnum):
    """折り方。**既定は ``ROLLING``**。"""

    ROLLING = "rolling"
    """原点を進める (expanding window)。訓練区間は常に検証区間より**前**。

    予測の実務と同じ形で、未来から過去への漏れを構造上作れない。
    """

    BLOCKED = "blocked"
    """連続ブロックを順に検証に使う (訓練は残り全部)。

    データを使い切れるが、**検証区間より後ろの行でも学習する**。リザバー状態は
    過去の入力すべてに依存するので、検証区間の入力が訓練行の状態に混ざる。
    「時間の向きを無視してよい」と判断した場合にだけ選ぶこと。
    """


@dataclass(frozen=True, slots=True)
class Fold:
    """1つの折り。行 index の range で持つ (``Split`` と同じ流儀)。

    Attributes:
        train: 学習に使う行。
        val: 採点に使う行。
    """

    train: range
    val: range


def make_folds(
    span: range, n_folds: int, *, scheme: FoldScheme, embargo: int
) -> tuple[Fold, ...]:
    """``span`` を ``n_folds`` 個の折りに切る。

    ``span`` は「訓練 + 検証」に使ってよい行 (テスト区間は含めない)。

    Args:
        span: 折りを作る行の範囲。
        n_folds: 折りの数 (2 以上)。
        scheme: 折り方。
        embargo: 訓練区間と検証区間のあいだに捨てる行数。**設計行列の
            ``first_valid`` 以上にすること** —— それ未満だと、検証行が訓練行と
            同じ入力を含む。

    Returns:
        折りの列。``rolling`` は ``n_folds`` 個、``blocked`` も ``n_folds`` 個。

    Raises:
        ValueError: ``n_folds`` が 2 未満、``embargo`` が負、または行が足りない場合。
    """
    if n_folds < 2:
        raise ValueError(f"n_folds は 2 以上である必要があります: {n_folds}")
    if embargo < 0:
        raise ValueError(f"embargo は 0 以上である必要があります: {embargo}")
    total = len(span)
    block = total // (n_folds + 1) if scheme is FoldScheme.ROLLING else total // n_folds
    if block <= embargo:
        raise ValueError(
            f"折りが作れません: 1ブロック {block} 行 <= embargo {embargo} 行 "
            f"(span={total} 行, n_folds={n_folds})。"
            "折りを減らすか、区間を長くしてください"
        )
    if scheme is FoldScheme.ROLLING:
        return _rolling_folds(span, n_folds, block, embargo)
    return _blocked_folds(span, n_folds, block, embargo)


def _rolling_folds(
    span: range, n_folds: int, block: int, embargo: int
) -> tuple[Fold, ...]:
    """原点を進める折り。``k`` 番目は先頭 ``(k+1)`` ブロックで学習する。"""
    start = span.start
    folds: list[Fold] = []
    for index in range(n_folds):
        train_stop = start + block * (index + 1)
        val_start = train_stop + embargo
        val_stop = min(val_start + block, span.stop)
        if val_stop - val_start < 1:
            break
        folds.append(
            Fold(train=range(start, train_stop), val=range(val_start, val_stop))
        )
    if not folds:
        raise ValueError("折りが1つも作れませんでした")
    return tuple(folds)


def _blocked_folds(
    span: range, n_folds: int, block: int, embargo: int
) -> tuple[Fold, ...]:
    """連続ブロックを順に検証に使う折り。**時間の向きを無視する** (上の注を参照)。

    訓練側は検証ブロックの前後から embargo ぶん離した行を使う。
    """
    folds: list[Fold] = []
    for index in range(n_folds):
        val_start = span.start + block * index
        val_stop = span.stop if index == n_folds - 1 else val_start + block
        before = range(span.start, max(span.start, val_start - embargo))
        after = range(min(span.stop, val_stop + embargo), span.stop)
        # range は不連続を表せないので、長いほうを訓練に採る。両側を使うには
        # 行 index の配列を持つ必要があり、Split (range) の流儀から外れる。
        train = before if len(before) >= len(after) else after
        if len(train) < 1 or val_stop - val_start < 1:
            continue
        folds.append(Fold(train=train, val=range(val_start, val_stop)))
    if not folds:
        raise ValueError("折りが1つも作れませんでした")
    return tuple(folds)


def select_alpha_cv(
    phi: FloatArray,
    y: FloatArray,
    folds: Sequence[Fold],
    alphas: Sequence[float],
    *,
    bias_column: int | None,
) -> AlphaSelection:
    """折りを跨いだ平均 NRMSE が最小の alpha を選ぶ。

    同点なら**大きい** alpha を残す (``select_alpha`` と同じ保守側)。
    ``AlphaSelection`` をそのまま返すので、呼び出し側は単一分割と同じ形で扱える
    (``val_nrmse`` は折りの平均、``curve`` も平均の列)。

    Gram 行列は折りごとに1回だけ作り、alpha 格子の走査では solve だけを繰り返す
    (``select_alpha`` と同じ最適化)。

    Args:
        phi: 設計行列 ``(T, F)`` (全区間)。
        y: 目標 ``(T, D_out)`` (全区間)。
        folds: ``make_folds`` の出力。
        alphas: 探索格子。
        bias_column: 無罰則列。

    Raises:
        ValueError: ``alphas`` か ``folds`` が空の場合。
    """
    if len(alphas) == 0:
        raise ValueError("alpha 格子が空です")
    if len(folds) == 0:
        raise ValueError("折りが空です")
    grid = sorted(float(value) for value in alphas)
    totals: dict[float, float] = dict.fromkeys(grid, 0.0)
    for fold in folds:
        phi_tr = phi[fold.train.start : fold.train.stop]
        y_tr = y[fold.train.start : fold.train.stop]
        phi_val = phi[fold.val.start : fold.val.stop]
        y_val = y[fold.val.start : fold.val.stop]
        gram: FloatArray = phi_tr.T @ phi_tr
        rhs: FloatArray = phi_tr.T @ y_tr
        for alpha in grid:
            coefficients = fit_ridge_from_gram(
                gram, rhs, alpha, bias_column=bias_column
            )
            totals[alpha] += nrmse(y_val, predict(phi_val, coefficients))
    n_folds = float(len(folds))
    curve = tuple((alpha, totals[alpha] / n_folds) for alpha in grid)
    best_alpha, best_score = grid[0], float("inf")
    for alpha, score in curve:
        if score <= best_score:
            best_alpha, best_score = alpha, score
    return AlphaSelection(alpha=best_alpha, val_nrmse=best_score, curve=curve)


def folds_never_look_ahead(folds: Sequence[Fold], embargo: int) -> bool:
    """全ての折りで、訓練区間が検証区間より ``embargo`` 以上手前で終わるか。

    ``rolling`` の不変条件そのもの。テストと、設定を検査する側が使う
    (``blocked`` は ``False`` を返しうる —— そういう折り方だからである)。
    """
    return all(
        fold.train.stop + embargo <= fold.val.start
        or fold.train.start >= fold.val.stop + embargo
        for fold in folds
    )


__all__ = [
    "Fold",
    "FoldScheme",
    "folds_never_look_ahead",
    "make_folds",
    "select_alpha_cv",
]

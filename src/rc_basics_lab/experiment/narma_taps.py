"""実験 3-C' —— タップ数を振って正則化の効き目を測る (D-95).

## なぜこれが要るのか

3-C は「先行 (Goudarzi et al. 2014) の対照は正則化なし OLS だった。
公平な対照で再実験する」という位置づけで、D-90 で正則化なしの水準を足した。
結果は OLS 0.1534 / リッジ 0.1538 でほぼ同値だった。

ところが 3-C の動作点は ``n_lags_grid = [10..30]`` / ``n_train = 3800`` で、
**k / n_train は最大でも 0.008 である**。この領域では OLS が壊れる理由が
無いので、「OLS でも同じだった」は先行の対照設計への批判を**検証していない**。

先行の設定は 1,810〜2,000 タップ / 訓練 2,000 点、つまり **k / n ≈ 1** で
あり、彼らの対照の本質は「正則化なし」ではなく
「**正則化なし かつ k ≈ n_train**」だった。

ここが動かすのは**タップ数の軸**である。各 k は独立に
``plan_replicate`` を通るので ``t0`` も分割もその k のものになり、
k が伸びるほど訓練区間が短くなって k / n_train が自然に 1 へ近づく。

## 実測 (1 レプリケートの偵察)

| k | k/n_train | リッジ | OLS |
|---|---|---|---|
| 30 | 0.008 | 0.1562 | 0.1524 |
| 800 | 0.229 | 0.2133 | 0.2197 |
| 1500 | 0.476 | 0.2804 | 0.3016 |
| 2200 | 0.786 | 0.3609 | 0.6188 |
| 2600 | 1.000 | 0.4540 | **5.944** |
| 3000 | (k > n_train) | — | 正規方程式が特異 |

小さい k では両者が区別できず、k / n_train が 1 に近づいたところで
OLS だけが壊れる。**3-C の結論が正則化に依らないのは、動作点が
先行と違うからである** —— これが図で言えるようになる。

``k > n_train`` は正規方程式が構造的に特異になるので、掃引の前に
``ValueError`` で落とす (scipy の ``LinAlgError`` まで進ませない)。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace

from rc_basics_lab.config import Capacity03Config, ExperimentConfig
from rc_basics_lab.experiment.narma import narma10_extra_methods, narma_task_entry
from rc_basics_lab.experiment.runner import (
    DELAY_LINE,
    DELAY_LINE_OLS,
    ResultRow,
    TaskEntry,
    plan_replicate,
    run_task,
)

logger = logging.getLogger(__name__)

EXPERIMENT_NARMA10_TAPS = "3Cp_narma10_taps"
"""``narma10_taps.csv`` の ``experiment`` 列。"""

#: 掃引に残す手法。**遅延線の2水準だけ**を見る (D-95)。
#: 線形と ESN はタップ数の軸を持たないので、同じ図に乗せると
#: 「k を振ったのに動かない系列」が2本増えるだけになる。
SWEPT_METHODS: tuple[str, ...] = (DELAY_LINE, DELAY_LINE_OLS)


@dataclass(frozen=True, slots=True)
class TapSweepRow:
    """1 (k, 手法, レプリケート) の成績 (``narma10_taps.csv`` の1行)。

    Attributes:
        experiment: 実験名 (``EXPERIMENT_NARMA10_TAPS``)。
        n_lags: タップ数 k。
        method: ``delay_line`` (リッジ) か ``delay_line_ols`` (正則化なし)。
        replicate: レプリケート番号。
        alpha: 選ばれた alpha (OLS は常に 0)。
        n_train: その k での訓練区間の長さ。
        taps_per_train: ``n_lags / n_train``。**この図の横軸**。
        nmse / nrmse / rmse: テスト区間の誤差。
        t0: その k での基準点 (k が伸びるほど大きくなる)。
        wall_time_s: その1点の実測時間。
    """

    experiment: str
    n_lags: int
    method: str
    replicate: int
    alpha: float
    n_train: int
    taps_per_train: float
    nmse: float
    nrmse: float
    rmse: float
    t0: int
    wall_time_s: float


CSV_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(TapSweepRow))
"""``narma10_taps.csv`` の列順 (``TapSweepRow`` の宣言順が単一の真実)。"""


def _validate_sweep(n_lags_sweep: Sequence[int], length: int) -> None:
    """掃引の格子を回す前に検査する。

    Raises:
        ValueError: 昇順でない / 重複がある / 1 未満 / 系列長以上の場合。
    """
    if not n_lags_sweep:
        return
    if list(n_lags_sweep) != sorted(set(n_lags_sweep)):
        raise ValueError(
            f"n_lags_sweep は昇順・重複なしが必要です: {tuple(n_lags_sweep)}"
        )
    if n_lags_sweep[0] < 1:
        raise ValueError(f"タップ数は 1 以上が必要です: {n_lags_sweep[0]}")
    if n_lags_sweep[-1] >= length:
        raise ValueError(f"タップ数が系列長以上です: {n_lags_sweep[-1]} >= {length}")


def _base_for(
    base: ExperimentConfig, n_lags: int, n_replicates: int
) -> ExperimentConfig:
    """タップ数を1点に固定した土台を作る。

    ``n_lags_grid`` を1点にするので、その k での ``t0`` と分割が決まる。
    alpha 格子は触らない —— 動かす軸を1本に保つのがこの掃引の要点である。
    """
    return replace(
        base,
        n_replicates=n_replicates,
        ridge=replace(base.ridge, n_lags_grid=(n_lags,)),
    )


def run_narma10_tap_sweep(config: Capacity03Config) -> tuple[TapSweepRow, ...]:
    """タップ数を振って、リッジと正則化なしの成績を測る (D-95)。

    Args:
        config: 03 の設定 (``narma.n_lags_sweep`` を読む)。

    Returns:
        (k x 手法 x レプリケート) の行。掃引が空なら空タプル。

    Raises:
        ValueError: 格子が昇順でない / 重複がある / 範囲外の場合、および
            ``k > n_train`` で正規方程式が構造的に特異になる場合。
    """
    narma = config.narma
    _validate_sweep(narma.n_lags_sweep, narma.length)
    if not narma.n_lags_sweep:
        return ()
    base = narma.base
    n_replicates = (
        base.n_replicates
        if narma.n_replicates_sweep is None
        else narma.n_replicates_sweep
    )
    rows: list[TapSweepRow] = []
    for n_lags in narma.n_lags_sweep:
        started = time.perf_counter()
        cfg_k = replace(
            config,
            narma=replace(narma, base=_base_for(base, n_lags, n_replicates)),
        )
        entry = narma_task_entry(cfg_k)
        _reject_if_underdetermined(cfg_k.narma.base, entry, n_lags, n_replicates)
        produced = run_task(
            cfg_k.narma.base,
            entry,
            extra_methods=narma10_extra_methods(cfg_k.narma.base),
        )
        elapsed = time.perf_counter() - started
        rows.extend(_rows_for(produced, n_lags, elapsed))
    return tuple(rows)


def _reject_if_underdetermined(
    base: ExperimentConfig, entry: TaskEntry, n_lags: int, n_replicates: int
) -> None:
    """列数が行数以上になる k を**解く前に**落とす (D-95)。

    正則化なしの正規方程式は ``k + 1 > n_train`` (+1 はバイアス列) で
    特異になり、scipy が ``LinAlgError`` を送出する。それはどの k が
    悪いのかも、なぜ悪いのかも言わないので、掃引の設計を直せない。

    分割のオフセットはレプリケートごとに違う (``max_start_offset``) ので、
    **全レプリケートの最小 ``n_train``** で判定する。1本だけ通って他が
    落ちる、という形にしない。

    ``plan_replicate`` を先に回すぶん状態生成が二重になるが、実測で
    8 点 x 3 レプリケートの掃引が数秒なので、誤りを早く落とすほうを採る。

    Raises:
        ValueError: どれかのレプリケートで ``k + 1 > n_train`` になる場合。
    """
    sizes = [
        plan_replicate(base, entry, replicate).split.sizes[0]
        for replicate in range(n_replicates)
    ]
    smallest = min(sizes)
    if n_lags + 1 > smallest:
        raise ValueError(
            f"タップ数が訓練区間に対して大きすぎます: k = {n_lags}, "
            f"n_train = {smallest} (最小のレプリケート)。"
            "バイアス列を含めた列数が行数以上になり、正則化なしの正規方程式が"
            "構造的に特異になります。n_lags_sweep をこの手前で止めてください"
            " (実測: k = 2600 で n_train = 2600 になり scipy が"
            " singular matrix を送出)。"
        )


def _rows_for(
    produced: Sequence[ResultRow], n_lags: int, elapsed: float
) -> list[TapSweepRow]:
    """1つの k の結果を掃引の行に詰め替える。

    Raises:
        ValueError: ``k > n_train`` の場合 (正則化なしの正規方程式が
            構造的に特異になる。scipy の LinAlgError まで進ませない)。
    """
    kept = [row for row in produced if row.method in SWEPT_METHODS]
    if not kept:
        raise ValueError(f"k = {n_lags} で遅延線の行が1つも出ませんでした")
    n_train = kept[0].n_train
    rows: list[TapSweepRow] = []
    for row in kept:
        rows.append(
            TapSweepRow(
                experiment=EXPERIMENT_NARMA10_TAPS,
                n_lags=n_lags,
                method=row.method,
                replicate=row.replicate,
                alpha=row.alpha,
                n_train=row.n_train,
                taps_per_train=n_lags / row.n_train,
                nmse=row.nmse,
                nrmse=row.nrmse,
                rmse=row.rmse,
                t0=row.t0,
                wall_time_s=elapsed,
            )
        )
    logger.info(
        "k=%d n_train=%d k/n=%.3f nmse(ridge/ols)=%.4g/%.4g (%.2fs)",
        n_lags,
        n_train,
        n_lags / n_train,
        _mean_nmse(rows, DELAY_LINE),
        _mean_nmse(rows, DELAY_LINE_OLS),
        elapsed,
    )
    return rows


def _mean_nmse(rows: Sequence[TapSweepRow], method: str) -> float:
    """1つの k における手法ごとの NMSE 平均 (ログ用)。"""
    values = [row.nmse for row in rows if row.method == method]
    return sum(values) / len(values) if values else float("nan")


def summarize_tap_sweep(rows: Sequence[TapSweepRow]) -> dict[str, object]:
    """``meta.json`` に載せる要約 (D-95)。

    「正則化が効き始めるのはどこか」を成果物だけで読めるようにする。
    図を目で見ないと分からない状態にしない (D-90 と同じ規律)。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    if not rows:
        raise ValueError("rows が空です")
    by_k: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        by_k.setdefault(row.n_lags, {}).setdefault(row.method, []).append(row.nmse)
    means = {
        n_lags: {
            method: sum(values) / len(values) for method, values in methods.items()
        }
        for n_lags, methods in by_k.items()
    }
    ratios = {
        n_lags: methods[DELAY_LINE_OLS] / methods[DELAY_LINE]
        for n_lags, methods in means.items()
        if DELAY_LINE in methods
        and DELAY_LINE_OLS in methods
        and methods[DELAY_LINE] > 0.0
    }
    worst_k = max(ratios, key=lambda key: ratios[key]) if ratios else None
    taps_per_train = {
        row.n_lags: row.taps_per_train for row in rows if row.method == DELAY_LINE
    }
    return {
        "n_lags_sweep": sorted(means),
        "nmse_mean": {str(k): means[k] for k in sorted(means)},
        "taps_per_train": {str(k): taps_per_train[k] for k in sorted(taps_per_train)},
        "ols_over_ridge": {str(k): ratios[k] for k in sorted(ratios)},
        "worst_n_lags": worst_k,
        "worst_ols_over_ridge": None if worst_k is None else ratios[worst_k],
        "worst_taps_per_train": None if worst_k is None else taps_per_train[worst_k],
    }


__all__ = [
    "CSV_COLUMNS",
    "EXPERIMENT_NARMA10_TAPS",
    "SWEPT_METHODS",
    "TapSweepRow",
    "run_narma10_tap_sweep",
    "summarize_tap_sweep",
]

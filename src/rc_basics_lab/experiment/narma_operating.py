"""実験 3-C'' —— NARMA10 の勝敗が**手法側の動作点**で変わることの実測 (D-144).

## なぜこれが要るのか

先行 (Kubota & Nakajima, *Dynamical Anatomy of NARMA10 Benchmark Task*) は
NARMA10 が位相空間に不安定領域を持ち、**課題側の動作点** (入力範囲・初期値)
で結果が変わるので公平な比較を妨げる、と論じている。記事 03 §3.1 の実測
(「宣言した分布の内側でも発散する。定数入力 0.5 なら 33 ステップ」) は
独立に同じ現象へ到達している。

ここが足すのは**もう一方の側**である:

===============  ==============================  ==========================
                 先行                            こちら
===============  ==============================  ==========================
NARMA10 の問題   **課題側**の動作点で結果が変わる  **手法側**の動作点で
                                                 **勝敗**が変わる
根拠             位相空間の解析                   IPC 分解 (容量の大半が
                                                 要求されていない非線形へ)
===============  ==============================  ==========================

2つ合わせると「**NARMA10 は両側の動作点に敏感で、単独の数値に意味がない**」
という一段強い主張になる。

## 何を振るか

ESN の ``n_units`` と ``leak_rate`` だけである。3-C 本体は D-39 により
N=50 の1点を報告する (先行の参照値が N=50 規模のため) が、その1点で
「ESN が遅延線に負けた」と書くと、**動作点を1つ選んだ結果**を現象として
報告することになる。宣言した格子を全件出す。

**alpha 格子も分割も触らない** (D-04 / D-05)。動かす軸を2本に保たないと、
勝敗が動いた原因を動作点に帰せない。

## 容量の列は「その動作点のリザバー」を説明する

``ipc_*`` / ``mc_total`` は ESN が使ったリザバーの量なので、**手法によらず
同じ値が入る**。遅延線の行にも同じ値が乗るのは、1つのファイルで
「成績 <-> 容量」を join できるようにするためである (列を読むときは
『その点の ESN のリザバーはこうだった』と読む)。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, fields, replace

from rc_basics_lab.config import (
    Capacity03Config,
    ExperimentConfig,
    MackeyGlassTask,
    NarmaOperatingConfig,
    require_task,
)
from rc_basics_lab.experiment.narma import Narma10Results, run_narma10
from rc_basics_lab.reservoir.axes import with_axis

logger = logging.getLogger(__name__)

EXPERIMENT_NARMA10_OPERATING = "3C2_narma10_operating"
"""``narma10_operating.csv`` の ``experiment`` 列。"""

OPERATING_AXES: tuple[str, ...] = ("n_units", "leak_rate")
"""振る軸 (**モデルの言葉のまま**、D-124)。持たないモデルなら落ちる。"""


@dataclass(frozen=True, slots=True)
class OperatingPointRow:
    """1 (動作点, 手法, レプリケート) の行。**宣言順が CSV の列順**。

    Attributes:
        experiment: ``EXPERIMENT_NARMA10_OPERATING``。
        n_units: その動作点の N。
        leak_rate: その動作点のリーク率。
        method: ``linear`` / ``delay_line`` / ``esn`` / ``delay_line_ols``。
        replicate: レプリケート番号。
        alpha: 選ばれた alpha。
        n_lags: 選ばれたタップ数 (遅延線のみ。ESN は 0)。**遅延線が使える
            線形の本数**であり、ESN の ``ipc_linear`` と直接比べられる。
        nrmse / nmse / rmse: テスト区間の誤差。
        mc_total: その動作点の**リザバー**の線形メモリ容量。
        ipc_total: 同じくしきい値後の総容量。
        ipc_linear: 同じく次数1の容量。
        ipc_nonlinear: 同じく次数2以上の容量。
        nonlinear_share: ``ipc_nonlinear / ipc_total``。**NARMA10 が要求して
            いない側に容量のどれだけが行っているか**。
        wall_time_s: その動作点1点ぶんの実測時間 (行で共有する)。
    """

    experiment: str
    n_units: int
    leak_rate: float
    method: str
    replicate: int
    alpha: float
    n_lags: int
    nrmse: float
    nmse: float
    rmse: float
    mc_total: float
    ipc_total: float
    ipc_linear: float
    ipc_nonlinear: float
    nonlinear_share: float
    wall_time_s: float


OPERATING_CSV_COLUMNS: tuple[str, ...] = tuple(
    item.name for item in fields(OperatingPointRow)
)
"""``narma10_operating.csv`` の列順 (宣言順)。"""


def operating_points(
    section: NarmaOperatingConfig,
) -> tuple[tuple[int, float], ...]:
    """格子を (N, リーク率) の並びに展開する (D-144)。

    Args:
        section: 3-C'' の設定。

    Returns:
        動作点の並び。どちらかの格子が空なら空タプル (掃引しない)。

    Raises:
        ValueError: N が 1 未満、またはリーク率が ``(0, 1]`` の外の場合。
    """
    if not section.n_units_grid or not section.leak_rate_grid:
        return ()
    for n_units in section.n_units_grid:
        if n_units < 1:
            raise ValueError(f"n_units は 1 以上が必要です: {n_units}")
    for leak_rate in section.leak_rate_grid:
        if not 0.0 < leak_rate <= 1.0:
            raise ValueError(f"leak_rate は (0, 1] が必要です: {leak_rate}")
    return tuple(
        (n_units, leak_rate)
        for n_units in section.n_units_grid
        for leak_rate in section.leak_rate_grid
    )


def base_at(base: ExperimentConfig, n_units: int, leak_rate: float) -> ExperimentConfig:
    """動作点を差し替えた土台を返す (**ESN の1本だけ**を動かす)。

    ``narma_reservoir_config`` と同じセクション (``mackey_glass``) を書き換える。
    課題のパラメータも alpha 格子も分割も触らない —— 動かす軸を2本に保つのが
    この掃引の要点である。

    Args:
        base: 3-C の土台 (01 の ``ExperimentConfig``)。
        n_units: そのの動作点の N。
        leak_rate: その動作点のリーク率。

    Returns:
        ESN だけを差し替えた複製。

    Raises:
        ValueError: そのモデルが ``OPERATING_AXES`` を持たない場合。
    """
    task = require_task(base, MackeyGlassTask, "実験3-C'' (動作点の掃引)")
    reservoir = with_axis(task.reservoir, "n_units", n_units)
    reservoir = with_axis(reservoir, "leak_rate", leak_rate)
    replaced = replace(task, reservoir=reservoir)
    return replace(
        base,
        tasks=tuple(
            replaced if isinstance(item, MackeyGlassTask) else item
            for item in base.tasks
        ),
    )


def run_narma10_operating_sweep(
    config: Capacity03Config,
) -> tuple[OperatingPointRow, ...]:
    """動作点を振って NARMA10 の勝敗を測る (3-C''、D-144)。

    Args:
        config: 03 の設定 (``narma_operating`` と ``narma`` を読む)。

    Returns:
        (動作点 x 手法 x レプリケート) の行。格子が空なら空タプル。

    Raises:
        ValueError: 格子が範囲外、または 3-C の経路が投げた場合。
    """
    points = operating_points(config.narma_operating)
    if not points:
        return ()
    narma = config.narma
    rows: list[OperatingPointRow] = []
    for n_units, leak_rate in points:
        started = time.perf_counter()
        at_point = replace(
            config,
            narma=replace(narma, base=base_at(narma.base, n_units, leak_rate)),
        )
        results = run_narma10(at_point)
        elapsed = time.perf_counter() - started
        rows.extend(_rows_at_point(results, n_units, leak_rate, elapsed))
    logger.info(
        "3-C'': 動作点=%d x 手法x重み=%d = %d 行",
        len(points),
        len(rows) // len(points),
        len(rows),
    )
    return tuple(rows)


def _rows_at_point(
    results: Narma10Results, n_units: int, leak_rate: float, elapsed: float
) -> list[OperatingPointRow]:
    capacity = results.capacity.row
    total = capacity.ipc_total
    share = capacity.ipc_nonlinear / total if total > 0.0 else float("nan")
    return [
        OperatingPointRow(
            experiment=EXPERIMENT_NARMA10_OPERATING,
            n_units=n_units,
            leak_rate=leak_rate,
            method=row.method,
            replicate=row.replicate,
            alpha=row.alpha,
            n_lags=row.n_lags,
            nrmse=row.nrmse,
            nmse=row.nmse,
            rmse=row.rmse,
            mc_total=capacity.mc_total,
            ipc_total=total,
            ipc_linear=capacity.ipc_linear,
            ipc_nonlinear=capacity.ipc_nonlinear,
            nonlinear_share=share,
            wall_time_s=elapsed,
        )
        for row in results.rows
    ]


__all__ = [
    "EXPERIMENT_NARMA10_OPERATING",
    "OPERATING_AXES",
    "OPERATING_CSV_COLUMNS",
    "OperatingPointRow",
    "base_at",
    "operating_points",
    "run_narma10_operating_sweep",
]

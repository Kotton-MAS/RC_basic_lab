"""実験 01' —— 自走 84 ステップ先の誤差 (NRMSE84) を測る (D-105).

## なぜこれが要るのか

01 が測っているのは**1ステップ先**の誤差である。ところが Mackey-Glass で
広く引かれている文献値は **NRMSE84** —— 自走を 84 ステップ続けたときの、
**84 ステップ目だけ**の誤差 —— であり、予測長が違うので比較できない。

一次資料は Jaeger & Haas (2004) *Harnessing Nonlinearity*, Science
**304**:78-80 で、本文にこうある:

    For testing, a 84-step continuation d(3001),...,d(3084) of the original
    signal was computed for reference. The network output y(3084) was
    compared with the correct continuation d(3084).

そこで 01 にも同じ量を足す。**1ステップ先の行は1つも触らない** ——
自走は別の CSV (``horizon.csv``) へ出す。

## 実装の要点

自走そのものは ``readout.autoregressive.free_run`` (D-44) をそのまま使う。
04 の ``run_free_run`` は ``Chaos04Config`` に結びついているので呼ばない
—— 04 の設定を 01 に持ち込むことになるためである。再利用するのは
**読み出し層の primitive** で、そこが層として正しい単位になる。

学習した係数は 1ステップ先の読み出しそのものを使う。**別に学習し直さない**
のは、「1ステップ先を学習した同じモデルが 84 ステップ先でどうなるか」が
問いだからである (学習し直すと別のモデルの話になる)。
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np

from rc_basics_lab.config import ExperimentConfig
from rc_basics_lab.experiment.freerun import esn_state_updater
from rc_basics_lab.experiment.runner import (
    ESN_METHOD,
    ReplicatePlan,
    TaskEntry,
    build_methods,
    plan_replicate,
)
from rc_basics_lab.readout.autoregressive import StateUpdater, free_run
from rc_basics_lab.readout.ridge import fit_ridge, select_alpha
from rc_basics_lab.reservoir.esn import ESN
from rc_basics_lab.seeds import SeedStream, make_rng
from rc_basics_lab.types import FloatArray

logger = logging.getLogger(__name__)

HORIZON_STEPS = 84
"""文献と突き合わせる自走の長さ [ステップ] (D-105)。

Jaeger & Haas (2004) が「84-step continuation」で測っているので同じにする。
**この値は文献に合わせるためのものであって、こちらの都合で選んでいない。**
"""

TASK_NAME = "mackey_glass"
"""自走を測る課題。文献値が Mackey-Glass のものなので1本に絞る。"""


@dataclass(frozen=True, slots=True)
class HorizonRow:
    """1 (手法, レプリケート) の自走誤差 (``horizon.csv`` の1行)。

    Attributes:
        task: 課題名。
        method: 手法名。
        replicate: レプリケート番号。
        n_units: リザバー規模 (文献との比較で効く条件)。
        horizon: 自走したステップ数。
        nrmse_horizon: **``horizon`` ステップ目だけ**の NRMSE
            (真の系列の標準偏差で正規化。D-02 と同じ正規化)。
        log10_nrmse_horizon: その常用対数 (文献が log10 で報告するため)。
        nrmse_mean_to_horizon: 1..``horizon`` ステップの平均 NRMSE
            (84 ステップ目だけを見る文献値と違い、途中の壊れ方も見える)。
        diverged: 自走中に非有限値が出たか。
        wall_time_s: その1行の実測時間。
    """

    task: str
    method: str
    replicate: int
    n_units: int
    horizon: int
    nrmse_horizon: float
    log10_nrmse_horizon: float
    nrmse_mean_to_horizon: float
    diverged: bool
    wall_time_s: float


CSV_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(HorizonRow))
"""``horizon.csv`` の列順 (``HorizonRow`` の宣言順が単一の真実)。"""


def _fit_one_step(
    plan: ReplicatePlan, config: ExperimentConfig
) -> tuple[FloatArray, int]:
    """ESN の1ステップ先読み出しを学習して係数を返す。

    01 の ``_evaluate`` と同じ選び方 (検証分割で alpha を選ぶ) を通す。
    """
    design = plan.designs[ESN_METHOD][0]
    split = plan.split
    phi_train = design.phi[split.train.start : split.train.stop]
    phi_val = design.phi[split.val.start : split.val.stop]
    y_train = plan.task.y[split.train.start : split.train.stop]
    y_val = plan.task.y[split.val.start : split.val.stop]
    selection = select_alpha(
        phi_train,
        y_train,
        phi_val,
        y_val,
        config.ridge.alpha_grid,
        bias_column=design.bias_column,
    )
    coefficients = fit_ridge(
        phi_train, y_train, selection.alpha, bias_column=design.bias_column
    )
    return coefficients, split.test.start


def _nrmse_at_last(truth: FloatArray, predicted: FloatArray, sigma: float) -> float:
    """自走の**最終ステップだけ**の NRMSE (D-105)。

    文献 (Jaeger & Haas 2004) は 84 ステップ目の1点だけを見る。
    途中を平均すると別の量になるので、ここは1点で測る。
    """
    if sigma <= 0.0 or not math.isfinite(sigma):
        return math.nan
    return abs(float(predicted[-1]) - float(truth[-1])) / sigma


def run_horizon(config: ExperimentConfig, entry: TaskEntry) -> tuple[HorizonRow, ...]:
    """ESN を自走させ、``HORIZON_STEPS`` ステップ目の誤差を測る (D-105)。

    Args:
        config: 01 の設定。
        entry: Mackey-Glass の課題定義 (``build_tasks`` が組んだもの)。

    Returns:
        (手法 x レプリケート) の行。現在は ESN のみ ——
        文献値が ESN のものであり、線形と遅延線は自走の入口を持たない。

    Raises:
        ValueError: 自走に必要な長さがテスト区間に無い場合。
    """
    rows: list[HorizonRow] = []
    methods = {method.name: method for method in build_methods(config)}
    spec = methods[ESN_METHOD].candidates[0]
    for replicate in range(config.n_replicates):
        started = time.perf_counter()
        plan = plan_replicate(config, entry, replicate)
        coefficients, switch = _fit_one_step(plan, config)
        available = plan.task.n_steps - switch - 1
        if available < HORIZON_STEPS:
            raise ValueError(
                f"テスト区間が自走 {HORIZON_STEPS} ステップに足りません: "
                f"残り {available} ステップ。系列長を伸ばしてください。"
            )
        result = free_run(
            _updater(config, entry, plan, replicate),
            spec,
            coefficients,
            plan.states[switch - 1],
            plan.task.u[switch],
            HORIZON_STEPS,
        )
        truth = plan.task.y[switch : switch + HORIZON_STEPS]
        predicted = np.asarray(result.predictions, dtype=np.float64).reshape(-1)
        sigma = float(np.std(plan.task.y, ddof=0))
        elapsed = time.perf_counter() - started
        nrmse_h = _nrmse_at_last(truth.reshape(-1), predicted, sigma)
        mean_nrmse = (
            float(np.sqrt(np.mean((predicted - truth.reshape(-1)) ** 2)) / sigma)
            if sigma > 0.0
            else math.nan
        )
        rows.append(
            HorizonRow(
                task=entry.name,
                method=ESN_METHOD,
                replicate=replicate,
                n_units=entry.esn.n_units,
                horizon=HORIZON_STEPS,
                nrmse_horizon=nrmse_h,
                log10_nrmse_horizon=(
                    math.log10(nrmse_h) if nrmse_h > 0.0 else math.nan
                ),
                nrmse_mean_to_horizon=mean_nrmse,
                diverged=result.diverged,
                wall_time_s=elapsed,
            )
        )
        logger.info(
            "task=%s replicate=%d N=%d nrmse84=%.4g log10=%.2f (%.2fs)",
            entry.name,
            replicate,
            entry.esn.n_units,
            nrmse_h,
            rows[-1].log10_nrmse_horizon,
            elapsed,
        )
    return tuple(rows)


def _updater(
    config: ExperimentConfig,
    entry: TaskEntry,
    plan: ReplicatePlan,
    replicate: int,
) -> StateUpdater:
    """自走で使う状態更新器 (教師強制で使ったリザバーそのもの)。

    04 の ``esn_state_updater`` をそのまま使う —— ``ESN.run`` ではなく
    ``ESN.step`` を包んだもので、自走が1ステップずつ進む形に合っている
    (仕様 §5 禁止する構造8)。同名の更新器を 01 側に作り直さない (D-92)。
    """
    # **レプリケート番号を渡す。** plan_replicate は同じ番号で重みを作るので、
    # ここで 0 に固定すると『状態を作った ESN』と『自走する ESN』が食い違う。
    # 実測: 固定していた間、replicate 0 以外は NRMSE が 1e85 まで発散した。
    rng = make_rng(config.seeds, SeedStream.RESERVOIR, replicate)
    esn = ESN(entry.esn, rng, n_inputs=plan.task.n_inputs)
    return esn_state_updater(esn)


def summarize_horizon(rows: Sequence[HorizonRow]) -> dict[str, object]:
    """``meta.json`` に載せる要約 (D-105)。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    if not rows:
        raise ValueError("rows が空です")
    logs = [
        row.log10_nrmse_horizon
        for row in rows
        if math.isfinite(row.log10_nrmse_horizon)
    ]
    return {
        "horizon": HORIZON_STEPS,
        "n_units": rows[0].n_units,
        "n_rows": len(rows),
        "log10_nrmse_mean": float(np.mean(logs)) if logs else math.nan,
        "log10_nrmse_sd": float(np.std(logs, ddof=0)) if logs else math.nan,
        "n_diverged": sum(1 for row in rows if row.diverged),
    }


__all__ = [
    "CSV_COLUMNS",
    "HORIZON_STEPS",
    "TASK_NAME",
    "HorizonRow",
    "run_horizon",
    "summarize_horizon",
]

"""実験1-A ランナー — (課題 x 手法 x レプリケート) を回して長形式の行を作る.

**学習・評価パスに手法ごとの分岐は無い** (受け入れ条件1)。手法の定義は
``build_methods`` の1か所だけで、そこから先はすべて「候補 ``FeatureSpec`` の列」
として同じコードを通る。線形と ESN は候補が1つ、遅延線は ``ridge.n_lags_grid``
ぶんの候補を持つ、という長さの差しかない。

1レプリケート内では全手法が同一の行 index で学習・評価する (D-05)。基準点
``t0`` は ``compute_t0`` が全候補の ``first_valid`` と washout から1つだけ決める。

alpha 格子は ``config.ridge.alpha_grid`` という単一キーから全手法・全課題へ
そのまま渡る (D-04)。ESN の構造ハイパーパラメータは課題ごとの設定から
そのまま使い、検証分割では一切選ばない (D-08)。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields

import numpy as np

from rc_basics_lab.config import ESNConfig, ExperimentConfig
from rc_basics_lab.experiment.split import Split, compute_t0, make_split
from rc_basics_lab.metrics import nmse, nrmse, rmse, sign_accuracy
from rc_basics_lab.readout.design import (
    DelayLineSpec,
    DesignMatrix,
    FeatureSpec,
    PassthroughSpec,
    ReservoirSpec,
    bias_column_index,
    build_design_matrix,
)
from rc_basics_lab.readout.ridge import fit_ridge, predict, select_alpha
from rc_basics_lab.reservoir.esn import ESN
from rc_basics_lab.seeds import SeedStream, make_rng
from rc_basics_lab.tasks.base import TaskData
from rc_basics_lab.tasks.delay_parity import generate_delay_parity
from rc_basics_lab.tasks.mackey_glass import generate_mackey_glass
from rc_basics_lab.types import FloatArray

logger = logging.getLogger(__name__)

LINEAR = "linear"
DELAY_LINE = "delay_line"
ESN_METHOD = "esn"


@dataclass(frozen=True, slots=True)
class Method:
    """1手法の定義。

    Attributes:
        name: CSV の ``method`` 列。
        candidates: 検証分割で選ぶ候補の特徴仕様。**構造ハイパーパラメータは
            ここに入らない** (D-08: 選んでよいのは alpha と遅延線の n_lags だけ)。
    """

    name: str
    candidates: tuple[FeatureSpec, ...]


@dataclass(frozen=True, slots=True)
class TaskEntry:
    """1課題の定義 (生成関数と、その課題で使う ESN 設定)。"""

    name: str
    esn: ESNConfig
    generate: Callable[[np.random.Generator], TaskData]


@dataclass(frozen=True, slots=True)
class ReplicatePlan:
    """1レプリケートぶんの「全手法に共通の前提」。

    Attributes:
        task: 生成された課題データ。
        states: リザバー状態 ``(T, N)``。
        designs: 手法名 -> 候補ごとの設計行列。
        t0: 全手法共通の基準行 (D-05)。
        split: 全手法が共有する分割。
    """

    task: TaskData
    states: FloatArray
    designs: Mapping[str, tuple[DesignMatrix, ...]]
    t0: int
    split: Split


@dataclass(frozen=True, slots=True)
class ResultRow:
    """``comparison.csv`` の1行。"""

    task: str
    method: str
    replicate: int
    seed_reservoir: int
    seed_task: int
    seed_split: int
    alpha: float
    n_lags: int
    rmse: float
    nrmse: float
    nmse: float
    sign_accuracy: float
    n_train: int
    n_val: int
    n_test: int
    t0: int
    wall_time_s: float


CSV_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(ResultRow))
"""``comparison.csv`` の列順 (``ResultRow`` の宣言順が単一の真実)。"""


def build_methods(config: ExperimentConfig) -> tuple[Method, ...]:
    """比較する3手法を返す。**手法を列挙する唯一の場所**。

    線形と ESN の候補は1つ、遅延線は ``ridge.n_lags_grid`` ぶん。以降のコードは
    候補列の長さしか見ないため、学習・評価パスに手法名の分岐は現れない。
    """
    n_lags_grid = config.ridge.n_lags_grid
    if not n_lags_grid:
        raise ValueError("ridge.n_lags_grid が空です")
    return (
        Method(LINEAR, (PassthroughSpec(),)),
        Method(
            DELAY_LINE,
            tuple(DelayLineSpec(n_lags=n_lags) for n_lags in n_lags_grid),
        ),
        Method(ESN_METHOD, (ReservoirSpec(),)),
    )


def build_tasks(config: ExperimentConfig) -> tuple[TaskEntry, ...]:
    """比較する課題を返す。ESN 設定は config からそのまま渡る (D-08)。"""
    return (
        TaskEntry(
            name="mackey_glass",
            esn=config.esn_mackey_glass,
            generate=lambda rng: generate_mackey_glass(config.mackey_glass, rng),
        ),
        TaskEntry(
            name="delay_parity",
            esn=config.esn_delay_parity,
            generate=lambda rng: generate_delay_parity(config.delay_parity, rng),
        ),
    )


def _rows(array: FloatArray, selection: range) -> FloatArray:
    """連続区間の行を切り出す (分割は常に連続区間)。"""
    block: FloatArray = array[selection.start : selection.stop]
    return block


def plan_replicate(
    config: ExperimentConfig, task_entry: TaskEntry, replicate: int
) -> ReplicatePlan:
    """1レプリケートぶんの課題・状態・設計行列・分割を作る。

    全手法がここで作られた ``t0`` と ``split`` をそのまま使う (D-05)。
    """
    data = task_entry.generate(make_rng(config.seeds, SeedStream.TASK, replicate))
    reservoir_rng = make_rng(config.seeds, SeedStream.RESERVOIR, replicate)
    reservoir = ESN(task_entry.esn, reservoir_rng, n_inputs=data.n_inputs)
    # 重み生成に使った Generator をそのまま状態ノイズにも渡す (reservoir ストリームの
    # 続き)。state_noise=0 のときは1個も引かれないため既存の結果は不変で、
    # state_noise>0 を YAML で設定したときに ValueError で落ちる配線漏れが消える。
    states = reservoir.run(data.u, rng=reservoir_rng)
    designs = {
        method.name: tuple(
            build_design_matrix(spec, data.u, states) for spec in method.candidates
        )
        for method in build_methods(config)
    }
    t0 = compute_t0(
        (design.first_valid for group in designs.values() for design in group),
        config.split.washout,
    )
    split = make_split(
        config.split,
        data.n_steps,
        t0,
        make_rng(config.seeds, SeedStream.SPLIT, replicate),
    )
    return ReplicatePlan(task=data, states=states, designs=designs, t0=t0, split=split)


@dataclass(frozen=True, slots=True)
class _Selection:
    """検証分割で選ばれた (特徴仕様, alpha)。"""

    design: DesignMatrix
    alpha: float
    val_nrmse: float


def _select(
    plan: ReplicatePlan, candidates: Sequence[DesignMatrix], alphas: Sequence[float]
) -> _Selection:
    """検証 NRMSE が最小の (候補, alpha) を選ぶ。

    候補が1つの手法 (線形 / ESN) でも同じ経路を通る。同点なら先に評価した
    (= ``n_lags`` が小さい) 候補を残す。
    """
    if not candidates:
        raise ValueError("候補が空です")
    split = plan.split
    y_train = _rows(plan.task.y, split.train)
    y_val = _rows(plan.task.y, split.val)
    best: _Selection | None = None
    for design in candidates:
        selection = select_alpha(
            _rows(design.phi, split.train),
            y_train,
            _rows(design.phi, split.val),
            y_val,
            alphas,
            bias_column=bias_column_index(design.feature_names),
        )
        if best is None or selection.val_nrmse < best.val_nrmse:
            best = _Selection(
                design=design,
                alpha=selection.alpha,
                val_nrmse=selection.val_nrmse,
            )
    if best is None:  # pragma: no cover - 直上で空を弾いている
        raise ValueError("候補の選択に失敗しました")
    return best


def _evaluate(
    config: ExperimentConfig,
    task_name: str,
    method: Method,
    plan: ReplicatePlan,
    replicate: int,
) -> ResultRow:
    """1 (課題, 手法, レプリケート) を学習・評価して1行を返す。"""
    started = time.perf_counter()
    split = plan.split
    best = _select(plan, plan.designs[method.name], config.ridge.alpha_grid)
    bias_column = bias_column_index(best.design.feature_names)
    coefficients = fit_ridge(
        _rows(best.design.phi, split.train),
        _rows(plan.task.y, split.train),
        best.alpha,
        bias_column=bias_column,
    )
    y_test = _rows(plan.task.y, split.test)
    prediction = predict(_rows(best.design.phi, split.test), coefficients)
    n_train, n_val, n_test = split.sizes
    elapsed = time.perf_counter() - started
    row = ResultRow(
        task=task_name,
        method=method.name,
        replicate=replicate,
        seed_reservoir=config.seeds.reservoir,
        seed_task=config.seeds.task,
        seed_split=config.seeds.split,
        alpha=best.alpha,
        # 遅延線では first_valid == n_lags、他手法では 0。手法名で分岐せずに
        # 「特徴が何ステップ過去まで届くか」を記録する。
        n_lags=best.design.first_valid,
        rmse=rmse(y_test, prediction),
        nrmse=nrmse(y_test, prediction),
        nmse=nmse(y_test, prediction),
        sign_accuracy=sign_accuracy(y_test, prediction),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        t0=plan.t0,
        wall_time_s=elapsed,
    )
    logger.info(
        "task=%s method=%s replicate=%d alpha=%.3g n_lags=%d "
        "nrmse=%.4f sign_acc=%.3f (%.2fs)",
        row.task,
        row.method,
        row.replicate,
        row.alpha,
        row.n_lags,
        row.nrmse,
        row.sign_accuracy,
        row.wall_time_s,
    )
    return row


def run_task(config: ExperimentConfig, task_entry: TaskEntry) -> list[ResultRow]:
    """1課題について (手法 x レプリケート) を回す。"""
    if config.n_replicates < 1:
        raise ValueError(
            f"n_replicates は 1 以上である必要があります: {config.n_replicates}"
        )
    methods = build_methods(config)
    rows: list[ResultRow] = []
    for replicate in range(config.n_replicates):
        plan = plan_replicate(config, task_entry, replicate)
        logger.info(
            "task=%s replicate=%d t0=%d offset=%d sizes=%s",
            task_entry.name,
            replicate,
            plan.t0,
            plan.split.offset,
            plan.split.sizes,
        )
        rows.extend(
            _evaluate(config, task_entry.name, method, plan, replicate)
            for method in methods
        )
    return rows


def run_experiment(config: ExperimentConfig) -> list[ResultRow]:
    """全課題 x 全手法 x 全レプリケートを回す。"""
    rows: list[ResultRow] = []
    for task_entry in build_tasks(config):
        rows.extend(run_task(config, task_entry))
    return rows


__all__ = [
    "CSV_COLUMNS",
    "DELAY_LINE",
    "ESN_METHOD",
    "LINEAR",
    "Method",
    "ReplicatePlan",
    "ResultRow",
    "TaskEntry",
    "build_methods",
    "build_tasks",
    "plan_replicate",
    "run_experiment",
    "run_task",
]

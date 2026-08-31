"""実験1-A ランナー — (課題 x 手法 x レプリケート) を回して長形式の行を作る.

**学習・評価パスに手法ごとの分岐は無い** (受け入れ条件1)。手法の定義は
``build_methods`` の1か所だけで、そこから先はすべて「候補 ``FeatureSpec`` の列」
として同じコードを通る。線形と ESN は候補が1つ、遅延線は ``ridge.n_lags_grid``
ぶんの候補を持つ、という長さの差しかない。

1レプリケート内では全手法が同一の行 index で学習・評価する (D-05)。基準点
``t0`` は ``compute_t0`` が決める。

alpha 格子は ``config.ridge.alpha_grid`` から全手法・全課題へそのまま渡る
(D-04)。ESN の構造ハイパーパラメータは課題ごとの設定からそのまま使う (D-08)。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields

import numpy as np

from rc_basics_lab.config import CrossValidationConfig, ExperimentConfig
from rc_basics_lab.experiment.split import Split, compute_t0, make_split
from rc_basics_lab.metrics import nmse, nrmse, rmse, sign_accuracy
from rc_basics_lab.readout.cross_validation import (
    FoldScheme,
    make_folds,
    select_alpha_cv,
)
from rc_basics_lab.readout.design import (
    DelayLineSpec,
    DesignMatrix,
    FeatureSpec,
    PassthroughSpec,
    ReservoirSpec,
    build_design_matrix,
)
from rc_basics_lab.readout.ridge import (
    AlphaSelection,
    fit_ridge,
    predict,
    select_alpha,
)
from rc_basics_lab.reservoir.protocol import ReservoirConfig
from rc_basics_lab.reservoir.registry import build_reservoir
from rc_basics_lab.seeds import SeedStream, make_rng
from rc_basics_lab.tasks.base import TaskData
from rc_basics_lab.tasks.delay_parity import generate_delay_parity
from rc_basics_lab.tasks.mackey_glass import generate_mackey_glass
from rc_basics_lab.types import FloatArray

logger = logging.getLogger(__name__)

LINEAR = "linear"
DELAY_LINE = "delay_line"
DELAY_LINE_OLS = "delay_line_ols"
"""正則化なし (alpha = 0) の遅延線。3-C だけの対照水準 (D-90)。"""
ESN_METHOD = "esn"


@dataclass(frozen=True, slots=True)
class Method:
    """1手法の定義。

    Attributes:
        name: CSV の ``method`` 列。
        candidates: 検証分割で選ぶ候補の特徴仕様。**構造ハイパーパラメータは
            ここに入らない** (D-08: 選んでよいのは alpha と遅延線の n_lags だけ)。
        alphas: この手法だけが使う alpha 格子。``None`` なら
            ``config.ridge.alpha_grid`` (D-04 の共有格子) をそのまま使う。
            **既定は必ず None にすること** —— 手法ごとに格子を変えるのは
            D-04 が禁じている「探索予算の不平等」そのものであり、例外は
            ``.claude/decisions.yaml`` に記録した対照条件に限る (D-90)。
        design_key: 設計行列を借りてくる手法名。``None`` なら自分の名前。
            **同じ特徴行列を共有する対照** (正則化の有無だけが違う水準) を
            作るための口で、借りる側は候補の中身を1行も持たない。
    """

    name: str
    candidates: tuple[FeatureSpec, ...]
    alphas: tuple[float, ...] | None = None
    design_key: str | None = None

    @property
    def designs_key(self) -> str:
        """``ReplicatePlan.designs`` を引くキー。"""
        return self.design_key if self.design_key is not None else self.name


@dataclass(frozen=True, slots=True)
class TaskEntry:
    """1課題の定義 (生成関数と、その課題で使うリザバー設定)。

    ``esn`` の型は ``ReservoirConfig`` である。01 の本経路はどのモデルでも
    通る (``build_reservoir`` が生成する)。**ESN 固有の軸を振る経路**
    (03-C / 04) だけが ``require_esn`` で絞る —— 属性名が ``esn`` のままなのは
    既存の呼び出しを壊さないためで、意味は「この課題で使うリザバー」である。
    """

    name: str
    esn: ReservoirConfig
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
    reservoir = build_reservoir(task_entry.esn, reservoir_rng, n_inputs=data.n_inputs)
    # **状態ノイズ用の rng は常に渡す** (D-36)。重み生成に使った Generator を
    # そのまま渡す (reservoir ストリームの続き)。state_noise=0 では1個も引かれ
    # ないので既存の結果は不変で、rng を省く分岐を残すと state_noise>0 を
    # 設定した瞬間に ValueError で落ちる配線漏れが復活する。
    # **02 (esp.py) と 04 (freerun.py) も同じ形にしてある。ここが正本。**
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


def _score_candidate(
    plan: ReplicatePlan,
    design: DesignMatrix,
    alphas: Sequence[float],
    cv: CrossValidationConfig,
) -> AlphaSelection:
    """1候補の alpha を選ぶ。交差検証が有効ならそちらを通す。

    ``cv.n_folds == 0`` なら従来どおり単一の検証区間で採点する。**既定は
    そちら**で、成果物 (``results/``) は単一分割で作られている。

    交差検証の折りは**訓練 + 検証**の区間から切る。テスト区間は触らない ——
    折りに混ぜると、選択に使った行で最終評価することになる。

    禁足区間 (embargo) の既定は ``design.first_valid`` である。設計行列の1行は
    過去 ``first_valid`` 行の入力を含むので、それ未満だと検証行が訓練行と同じ
    入力を持つ (遅延線で顕著)。
    """
    split = plan.split
    if cv.n_folds <= 0:
        return select_alpha(
            _rows(design.phi, split.train),
            _rows(plan.task.y, split.train),
            _rows(design.phi, split.val),
            _rows(plan.task.y, split.val),
            alphas,
            bias_column=design.bias_column,
        )
    embargo = design.first_valid if cv.embargo is None else cv.embargo
    folds = make_folds(
        range(split.train.start, split.val.stop),
        cv.n_folds,
        scheme=FoldScheme(cv.scheme),
        embargo=embargo,
    )
    return select_alpha_cv(
        design.phi,
        plan.task.y,
        folds,
        alphas,
        bias_column=design.bias_column,
    )


def _select(
    plan: ReplicatePlan,
    candidates: Sequence[DesignMatrix],
    alphas: Sequence[float],
    cv: CrossValidationConfig,
) -> _Selection:
    """検証 NRMSE が最小の (候補, alpha) を選ぶ。

    候補が1つの手法 (線形 / ESN) でも同じ経路を通る。同点なら先に評価した
    (= ``n_lags`` が小さい) 候補を残す。
    """
    if not candidates:
        raise ValueError("候補が空です")
    best: _Selection | None = None
    for design in candidates:
        selection = _score_candidate(plan, design, alphas, cv)
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
    alphas = config.ridge.alpha_grid if method.alphas is None else method.alphas
    best = _select(plan, plan.designs[method.designs_key], alphas, config.ridge.cv)
    bias_column = best.design.bias_column
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


def _validate_extra_methods(extra: Sequence[Method], base: Sequence[Method]) -> None:
    """追加水準が**既存の3手法を書き換えていない**ことを実行前に落とす。

    静かに壊れる形が2つある。名前が既存手法とぶつかると CSV の同じ
    ``method`` 値に別条件の行が混ざり、図は平均を取って何も気づかない。
    ``design_key`` の借り先が無いと ``KeyError`` になるが、それはレプリケート
    ループの中まで進んでから落ちるので、どの水準が悪いのか分からない。
    """
    names = {method.name for method in base}
    for method in extra:
        if method.name in names:
            raise ValueError(f"追加水準の名前が既存手法と衝突しています: {method.name}")
        names.add(method.name)
        if method.design_key is not None and method.design_key not in {
            item.name for item in base
        }:
            raise ValueError(
                f"{method.name} の design_key が存在しません: {method.design_key}"
            )


def run_task(
    config: ExperimentConfig,
    task_entry: TaskEntry,
    *,
    plan0: ReplicatePlan | None = None,
    extra_methods: Sequence[Method] = (),
) -> list[ResultRow]:
    """1課題について (手法 x レプリケート) を回す。

    Args:
        plan0: レプリケート0の ``ReplicatePlan`` を渡すと、その分の
            ``plan_replicate`` 呼び直しを省く (F-1-009: ``collect_state_space``
            と計算を共有するための明示的な受け渡し)。省略時はこれまでどおり
            内部で作る。
        extra_methods: ``build_methods`` の3手法に**この課題だけ**足す水準。
            既定は空で、01 の ``comparison.csv`` は 1 行も変わらない
            (``build_methods`` に足すと 01 に行が増えて D-13 の分離が崩れる。
            D-31 が NARMA10 を ``build_tasks`` に足さないのと同じ理由)。
            設計行列は ``design_key`` で既存手法から借りるので、
            ``plan_replicate`` の ``t0`` も分割も変わらない。
    """
    if config.n_replicates < 1:
        raise ValueError(
            f"n_replicates は 1 以上である必要があります: {config.n_replicates}"
        )
    base = build_methods(config)
    _validate_extra_methods(extra_methods, base)
    methods = (*base, *extra_methods)
    rows: list[ResultRow] = []
    for replicate in range(config.n_replicates):
        plan = (
            plan0
            if replicate == 0 and plan0 is not None
            else plan_replicate(config, task_entry, replicate)
        )
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


def run_experiment(
    config: ExperimentConfig, *, plans0: Mapping[str, ReplicatePlan] | None = None
) -> list[ResultRow]:
    """全課題 x 全手法 x 全レプリケートを回す。

    Args:
        plans0: タスク名 -> レプリケート0の ``ReplicatePlan``。渡すとそのタスク
            のレプリケート0の計算を再利用する (F-1-009)。
    """
    rows: list[ResultRow] = []
    for task_entry in build_tasks(config):
        plan0 = None if plans0 is None else plans0.get(task_entry.name)
        rows.extend(run_task(config, task_entry, plan0=plan0))
    return rows


__all__ = [
    "CSV_COLUMNS",
    "DELAY_LINE",
    "DELAY_LINE_OLS",
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

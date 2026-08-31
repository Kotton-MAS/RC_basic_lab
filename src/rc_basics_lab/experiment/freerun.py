"""実験 4-A (教師強制の1ステップ先予測) と自走の入口の配線 (D-31 / D-44 / D-50).

4-A は **01 の ``run_task`` をそのまま通す** (D-31 と同じ形)。手法の列挙
(``build_methods``)・alpha 格子の共有 (D-04)・全手法が同一の行 index で学習評価
すること (D-05)・ESN の構造ハイパーパラメータを検証分割で選ばないこと (D-08)
は、すべて 01 の経路が担保する。ここが組み立てるのは ``TaskEntry`` (課題の生成
関数 + ESN 設定) だけで、**``build_tasks`` にも ``ExperimentConfig`` にも 04 の
課題を足さない** (足すと 01 の ``comparison.csv`` に行が増えて 01 の成果物が
変わる)。

自走の入口もここにある。ESN を ``StateUpdater`` に適合させるアダプタ
(``esn_state_updater``) と、確保軸の検査 (``validate_free_run_bounds``) は
実験層の責任である (D-50)。

**自走は教師強制で学習した係数をそのまま使う** (D-44)。``fit_teacher_forced``
が返した係数オブジェクトが ``FreeRunResult.coefficients`` に**同一オブジェクト
として**現れる。
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rc_basics_lab.config import Chaos04Config, ESNConfig
from rc_basics_lab.diagnostics.base import DiagnosticContext, DiagnosticResult
from rc_basics_lab.diagnostics.lyapunov import max_lyapunov
from rc_basics_lab.experiment.attractor import (
    AttractorDistance,
    RegimeVerdict,
    attractor_distance,
    classify_regime,
    first_autocorrelation_zero,
    lyapunov_normalized,
    normalized_error_curve,
    power_spectrum,
    return_map_points,
    shuffled_surrogate,
    valid_time_from_errors,
    validate_stats_bounds,
)
from rc_basics_lab.experiment.capacity_bounds import (
    validate_n_units_bound,
    validate_sequential_run_count,
    validate_state_matrix_bounds,
)
from rc_basics_lab.experiment.freerun_rows import (
    AttractorVerdict,
    FreeRunEvaluation,
    FreeRunProfileRow,
    FreeRunRow,
    ValidTimeSensitivity,
    summarize_attractor,
    summarize_valid_time,
)
from rc_basics_lab.experiment.freerun_tasks import (
    chaos_esn_config,
    chaos_task_entries,
    lorenz_task_entry,
    mackey_glass_task_entry,
    standardize_steps_for,
    task_length,
    task_sampling_interval,
    validate_free_run_bounds,
    validate_standardization_window,
)
from rc_basics_lab.experiment.report import (
    META_JSON,
    write_meta_for,
    write_rows_csv,
)
from rc_basics_lab.experiment.runner import (
    CSV_COLUMNS,
    DELAY_LINE,
    ESN_METHOD,
    LINEAR,
    ReplicatePlan,
    ResultRow,
    TaskEntry,
    build_methods,
    plan_replicate,
    run_task,
)
from rc_basics_lab.experiment.state_updaters import (
    delay_line_state_updater,
    esn_state_updater,
    passthrough_state_updater,
)
from rc_basics_lab.experiment.valid_time import VALID_TIME_THRESHOLD_GRID
from rc_basics_lab.metrics_significance import sign_test_p_value
from rc_basics_lab.readout.autoregressive import (
    FreeRunResult,
    StateUpdater,
    free_run,
)
from rc_basics_lab.readout.design import (
    DesignMatrix,
    FeatureSpec,
    PassthroughSpec,
    ReservoirSpec,
)
from rc_basics_lab.readout.ridge import fit_ridge, predict, select_alpha
from rc_basics_lab.reservoir.protocol import Reservoir
from rc_basics_lab.reservoir.registry import build_reservoir
from rc_basics_lab.seeds import SeedStream, make_rng
from rc_basics_lab.tasks.chaotic import (
    TASK_NAME_LORENZ,
    initial_state,
    integrate_lorenz,
    lorenz_sample_step,
    sampling_interval,
)
from rc_basics_lab.types import FloatArray

logger = logging.getLogger(__name__)

ONESTEP_CSV = "onestep.csv"
"""4-A の成果物名 (列順は 01 の ``ResultRow`` の宣言順が単一の真実)。"""

CHAOS_ESN_SECTION = "esn_mackey_glass"
"""04 が読む ESN セクション名 (3-C の ``NARMA10_ESN_SECTION`` と同じ流儀)。

``Chaos04Config.base`` は 01 の ``ExperimentConfig`` をまるごと内包しているので
ESN セクションが2本 (``esn_mackey_glass`` / ``esn_delay_parity``) 在るが、04 の
課題は連続値のカオス時系列2本だけなので読むのは一方である。**Mackey-Glass 側**
を選ぶ理由は、Lorenz も MG も連続値の入力を受けて連続値を出す回帰課題であり、
漏れ積分 (``leak_rate=0.3``) が効く点で同型だからである
(``esn_delay_parity`` は ±1 の2値入力・``leak_rate=1.0`` を前提とした設定)。

D-08 により ESN の構造ハイパーパラメータは検証分割で選ばれないので、
**宣言した1点をそのまま報告する**。実際に読む属性は ``chaos_esn_config``、
両者の一致は ``test_chaos_esn_section_matches_the_declared_choice`` が固定する。
"""

FREE_RUN_SPEC = ReservoirSpec()
"""自走に使う特徴仕様 ``[1, u[t], x[t]]``。

多項式読み出しは v0.1 では入れない (仕様 §3.2)。

01 の ``build_methods`` が ESN 手法に与える候補と**同一の値**であることを
``test_free_run_spec_matches_the_one_step_esn_candidate`` が固定する。ここが
ずれると「教師強制と自走で別の特徴を使う」(仕様 §5 禁止する構造2) になる。
"""


def run_onestep(config: Chaos04Config) -> list[ResultRow]:
    """実験 4-A: 教師強制の1ステップ先予測を3手法で回す (D-31)。

    01 の ``run_task`` を課題ごとに1回ずつ通すだけである。同一レプリケート内で
    ``(t0, n_train, n_val, n_test)`` が3手法で一致すること (D-05) も、alpha 格子
    の共有 (D-04) も、01 の経路が担保している。

    Returns:
        ``onestep.csv`` の行 (01 の ``ResultRow`` をそのまま使う)。

    Raises:
        ValueError: 確保軸を超える設定、または課題側の値域違反。
    """
    # 確保軸10 (逐次実行の本数)。plan_replicate は状態行列を1本だけ作り3手法で
    # 共有するので、4-A の本数は「課題数 x base.n_replicates」で決まる。
    validate_sequential_run_count(
        len(chaos_task_entries(config)) * config.base.n_replicates
    )
    rows: list[ResultRow] = []
    for entry in chaos_task_entries(config):
        # D-34 の規律を 04 の確保軸にも効かせる。plan_replicate が状態行列と
        # 重み行列を確保する**前に**落とす。
        validate_state_matrix_bounds(entry.esn.n_units, task_length(config, entry.name))
        rows.extend(run_task(config.base, entry))
    logger.info(
        "experiment=4A_onestep 行数=%d 課題=%s",
        len(rows),
        sorted({row.task for row in rows}),
    )
    return rows


@dataclass(frozen=True, slots=True)
class TeacherForcedReadout:
    """教師強制で学習した read-out (自走はこれをそのまま使う、D-44)。

    Attributes:
        plan: 01 の ``plan_replicate`` が作った課題・状態・設計行列・分割。
        design: 選ばれた候補の設計行列。
        alpha: 検証分割で選ばれた正則化係数 (D-04 の格子から)。
        coefficients: リッジ解 ``(F, D_out)``。**このオブジェクトを自走へ渡す**。
        val_nrmse: 選択時の検証 NRMSE。
        method: 手法名 (``LINEAR`` / ``DELAY_LINE`` / ``ESN_METHOD``)。
            対照 (線形・遅延線) も自走させるのは、受け入れ条件3 の後半
            「自走では対照が成立しない」を**数値で**示すためである。
            「原理的に不利」を主張だけで済ませると、読者は「回してみたら
            動いたかもしれない」を否定できない。
        spec: 学習に使った特徴仕様 (遅延線は選ばれた ``n_lags`` を持つ)。
            **閉ループで使う仕様とは表現が違うことがある** ——
            ``closed_loop_setup`` を参照。
    """

    plan: ReplicatePlan
    design: DesignMatrix
    alpha: float
    coefficients: FloatArray
    val_nrmse: float
    method: str = ESN_METHOD
    spec: FeatureSpec = FREE_RUN_SPEC


def _rows(array: FloatArray, selection: range) -> FloatArray:
    block: FloatArray = array[selection.start : selection.stop]
    return block


def method_candidates(config: Chaos04Config, method: str) -> tuple[FeatureSpec, ...]:
    """手法名 -> 候補の特徴仕様。

    **手法の列挙は 01 の ``build_methods`` が単一の真実**である。

    04 側で ``PassthroughSpec()`` / ``DelayLineSpec(...)`` を書き写さないのは、
    01 が候補を1本足したときに 04 の自走だけ古い候補を使い続ける事故を防ぐため
    である (``plan.designs[method]`` の並びとここが必ず一致する)。

    Raises:
        ValueError: 04 が自走させない手法名の場合。
    """
    for item in build_methods(config.base):
        if item.name == method:
            return item.candidates
    raise ValueError(f"01 の手法ではありません: {method!r}")


def fit_teacher_forced(
    config: Chaos04Config,
    task_entry: TaskEntry,
    replicate: int,
    *,
    method: str = ESN_METHOD,
    plan: ReplicatePlan | None = None,
) -> TeacherForcedReadout:
    """教師強制で読み出しを学習する (01 の ``_evaluate`` と同じ2段)。

    ``select_alpha`` (検証分割で候補と alpha を選ぶ) -> ``fit_ridge`` (訓練区間で
    係数を解く) の順序も、渡す行集合 (``plan.split``) も、同点のときに先に
    評価した候補を残す規律も 01 と同一である。**同じ (候補, alpha) が選ばれる
    こと**は ``test_free_run_readout_matches_the_one_step_selection`` が 4-A の
    行と突き合わせて実測する (経路が2本あることを主張だけで済ませない)。

    Args:
        config: 04 の設定。
        task_entry: 課題 (ESN 設定を含む)。
        replicate: レプリケート番号。
        method: 学習する手法 (既定は ESN)。対照も同じ経路で学習する。
        plan: すでに作ってある ``ReplicatePlan``。``None`` なら作る。
            **同じ (課題, レプリケート) で3手法を回すときは1個を共有する** ——
            01 の ``run_task(..., plan0=...)`` (3-C) と同じ形で、状態行列が
            1本しか存在しないことを構造で保証する。

    Raises:
        ValueError: 標準化の推定区間が訓練区間を越える場合 (D-41)、ESN 手法の
            候補が1本でない場合 (D-08)、または課題・分割側の値域違反。
    """
    base = config.base
    validate_n_units_bound(task_entry.esn.n_units)
    if plan is None:
        plan = plan_replicate(base, task_entry, replicate)
    validate_standardization_window(
        standardize_steps_for(config, task_entry.name), plan.split
    )
    specs = method_candidates(config, method)
    designs = plan.designs[method]
    if method == ESN_METHOD and len(designs) != 1:
        raise ValueError(f"ESN 手法の候補は1本のはずです (D-08): {len(designs)} 本")
    if len(designs) != len(specs):
        raise ValueError(
            f"候補数が一致しません: designs={len(designs)} specs={len(specs)}"
        )
    split = plan.split
    y_train = _rows(plan.task.y, split.train)
    y_val = _rows(plan.task.y, split.val)
    best_index = -1
    best_alpha = math.nan
    best_val_nrmse = math.inf
    for index, design in enumerate(designs):
        selection = select_alpha(
            _rows(design.phi, split.train),
            y_train,
            _rows(design.phi, split.val),
            y_val,
            base.ridge.alpha_grid,
            bias_column=design.bias_column,
        )
        if selection.val_nrmse < best_val_nrmse:
            best_index = index
            best_alpha = selection.alpha
            best_val_nrmse = selection.val_nrmse
    if best_index < 0:  # pragma: no cover - 候補が空なら build_methods が落ちる
        raise ValueError(f"候補の選択に失敗しました: {method!r}")
    design = designs[best_index]
    coefficients = fit_ridge(
        _rows(design.phi, split.train),
        y_train,
        best_alpha,
        bias_column=design.bias_column,
    )
    return TeacherForcedReadout(
        plan=plan,
        design=design,
        alpha=best_alpha,
        coefficients=coefficients,
        val_nrmse=best_val_nrmse,
        method=method,
        spec=specs[best_index],
    )


@dataclass(frozen=True, slots=True)
class ClosedLoop:
    """自走の初期条件一式 (手法ごとの違いを1か所に閉じる)。

    Attributes:
        updater: 状態を1ステップ進める写像 (D-50)。
        spec: **閉ループで使う**特徴仕様。遅延線だけは学習時
            (``DelayLineSpec``) と表現が違うが、``build_design_matrix`` が
            組む列は同一である (``test_closed_loop_design_matches_the_teacher_
            forced_row`` が実測)。
        x0: 切り替え点での状態。
        u0: 自走が最初に食う入力 (**モデル自身の予測**)。
    """

    updater: StateUpdater
    spec: FeatureSpec
    x0: FloatArray
    u0: FloatArray


def closed_loop_setup(
    readout: TeacherForcedReadout,
    switch_index: int,
    *,
    esn: Reservoir | None = None,
    noise_rng: np.random.Generator | None = None,
) -> ClosedLoop:
    """手法名から自走の初期条件を組む (**手法ごとの分岐はここだけ**)。

    ``u0`` は切り替え点の行に対する**モデル自身の予測**である。真値を与えると
    自走が1ステップぶん無料の情報を得る (T4 実装メモ 5)。3手法とも同じ規律で
    与えるので、対照が不利になる理由は「記憶が無いこと」だけになる。

    Args:
        readout: 教師強制で学習した読み出し。
        switch_index: 切り替え点の行 index。
        esn: ESN 手法のときに使うリザバー (他の手法では不要)。
        noise_rng: ``state_noise > 0`` のときのノイズ用 Generator (D-36)。

    Raises:
        ValueError: ESN 手法なのに ``esn`` が無い、または未知の手法名の場合。
    """
    plan = readout.plan
    u0: FloatArray = predict(
        readout.design.phi[switch_index : switch_index + 1], readout.coefficients
    )[0]
    match readout.method:
        case _ if readout.method == ESN_METHOD:
            if esn is None:
                raise ValueError("ESN 手法の自走には ESN が必要です")
            return ClosedLoop(
                updater=esn_state_updater(esn, noise_rng),
                spec=FREE_RUN_SPEC,
                x0=plan.states[switch_index],
                u0=u0,
            )
        case _ if readout.method == DELAY_LINE:
            n_inputs = plan.task.n_inputs
            n_lags = (readout.design.phi.shape[1] - 1) // n_inputs - 1
            window: FloatArray = plan.task.u[switch_index - n_lags : switch_index + 1][
                ::-1
            ]
            return ClosedLoop(
                updater=delay_line_state_updater(n_inputs),
                spec=ReservoirSpec(include_input=False),
                x0=window.reshape(-1),
                u0=u0,
            )
        case _ if readout.method == LINEAR:
            return ClosedLoop(
                updater=passthrough_state_updater(),
                spec=PassthroughSpec(),
                x0=np.zeros(1, dtype=np.float64),
                u0=u0,
            )
        case _:
            raise ValueError(f"04 が自走させない手法です: {readout.method!r}")


@dataclass(frozen=True, slots=True)
class FreeRunOutcome:
    """自走1本ぶんの結果 (4-B が統計を載せる土台)。

    Attributes:
        task: 課題名。
        method: 手法名 (対照も自走させる。受け入れ条件3 の後半)。
        replicate: レプリケート番号。
        switch_index: 教師強制から自走へ切り替えた行 index。
        readout: 教師強制で学習した read-out (D-44)。
        result: 自走の生の結果 (入力・状態・予測・打ち切り)。
        truth: 各自走ステップに対応する真の目標 ``(free_run_steps, D_out)``。
        wall_time_s: 実測 wall time [秒]。
    """

    task: str
    method: str
    replicate: int
    switch_index: int
    readout: TeacherForcedReadout
    result: FreeRunResult
    truth: FloatArray
    wall_time_s: float


def run_free_run(
    config: Chaos04Config,
    task_entry: TaskEntry,
    replicate: int,
    *,
    n_steps: int | None = None,
    method: str = ESN_METHOD,
    plan: ReplicatePlan | None = None,
) -> FreeRunOutcome:
    """教師強制で温めてから自走させる (自走の入口、D-44 / D-50)。

    手順と**順序そのものが設計判断**である。

    1. 確保軸3 (``free_run_steps * n_units``) を**確保より前に**検査する。
    2. 教師強制で read-out を学習する (``fit_teacher_forced``)。
    3. テスト区間の先頭で ``warmup_steps`` ぶん教師強制した状態を取り、
       その行の予測を自走の最初の入力にする。
    4. ``readout/autoregressive.free_run`` へ **2. の係数オブジェクトをそのまま**
       渡す (D-44)。自走側は学習の入口を1つも呼ばない。

    ウォームアップの状態を作り直さず ``plan.states`` の該当行を使うのは、
    「教師強制した ESN」と「自走を始める ESN」が同一のリザバー・同一の状態列で
    あることを構造で保証するためである (3-C が ``plan0`` を共有しているのと
    同じ形)。

    Args:
        config: 04 の設定。
        task_entry: ``chaos_task_entries`` が組んだ課題。**ESN 設定は
            ``task_entry.esn`` が単一の真実**である (4-C は条件ごとに
            ``spectral_radius`` / ``leak_rate`` / ``state_noise`` を差し替えた
            entry を渡す)。教師強制の状態を作る ``plan_replicate`` も同じ
            entry を見るので、「温めた ESN」と「自走する ESN」が食い違う経路が
            構造上ない。
        replicate: レプリケート番号。
        n_steps: 自走させるステップ数。``None`` なら ``freerun.free_run_steps``。
            4-B は ``freerun.stats_steps`` を渡して**1本の軌道**を長く回し、
            その先頭 ``free_run_steps`` を有効予測時間 (D-43)、全体を長時間統計
            (D-46) に使う (自走を2回回すと同じ軌道を2度計算することになる)。
            真の軌道と突き合わせる区間は常に ``free_run_steps`` ぶんである。
        method: 自走させる手法。既定は ESN。**対照 (線形・遅延線) も同じ
            経路で自走させる** —— 受け入れ条件3 の後半を数値で示すため。
        plan: すでに作ってある ``ReplicatePlan`` (同じ (課題, レプリケート) の
            3手法で1個を共有する)。``None`` なら作る。

    Returns:
        ``FreeRunOutcome``。

    Raises:
        ValueError: 確保軸を超える、自走に必要な行数がテスト区間に無い、または
            標準化の推定区間が訓練区間を越える場合。
    """
    started = time.perf_counter()
    freerun_cfg = config.freerun
    esn_cfg = task_entry.esn
    steps = freerun_cfg.free_run_steps if n_steps is None else n_steps
    validate_free_run_bounds(freerun_cfg.free_run_steps, esn_cfg.n_units)
    validate_free_run_bounds(steps, esn_cfg.n_units)
    if freerun_cfg.warmup_steps < 1:
        raise ValueError(
            f"warmup_steps は 1 以上である必要があります: {freerun_cfg.warmup_steps}"
        )

    readout = fit_teacher_forced(
        config, task_entry, replicate, method=method, plan=plan
    )
    replicate_plan = readout.plan
    switch_index = replicate_plan.split.test.start + freerun_cfg.warmup_steps - 1
    last_index = switch_index + freerun_cfg.free_run_steps
    if last_index >= replicate_plan.task.n_steps:
        raise ValueError(
            "自走に必要な行がテスト区間の先にありません "
            f"(T={replicate_plan.task.n_steps}, "
            f"test.start={replicate_plan.split.test.start}, "
            f"warmup_steps={freerun_cfg.warmup_steps}, "
            f"free_run_steps={freerun_cfg.free_run_steps})"
        )
    if switch_index < readout.design.first_valid:
        raise ValueError(
            "切り替え点が設計行列の有効行より手前です: "
            f"switch_index={switch_index} < first_valid={readout.design.first_valid}"
        )

    reservoir_rng = make_rng(config.base.seeds, SeedStream.RESERVOIR, replicate)
    reservoir = (
        build_reservoir(
            esn_cfg,
            reservoir_rng,
            n_inputs=replicate_plan.task.n_inputs,
        )
        if method == ESN_METHOD
        else None
    )
    # 状態ノイズ用の rng は**常に**渡す (D-36)。D-14 に4本目を足さない。
    noise_rng = reservoir_rng if esn_cfg.state_noise > 0.0 else None
    loop = closed_loop_setup(readout, switch_index, esn=reservoir, noise_rng=noise_rng)
    result = free_run(
        loop.updater,
        loop.spec,
        readout.coefficients,
        loop.x0,
        loop.u0,
        steps,
    )
    truth: FloatArray = replicate_plan.task.y[switch_index + 1 : last_index + 1]
    wall_time_s = time.perf_counter() - started
    logger.info(
        "experiment=4B_freerun task=%s method=%s replicate=%d switch=%d steps=%d "
        "diverged=%s completed=%d alpha=%.3g (%.2fs)",
        task_entry.name,
        method,
        replicate,
        switch_index,
        steps,
        result.diverged,
        result.n_completed,
        readout.alpha,
        wall_time_s,
    )
    return FreeRunOutcome(
        task=task_entry.name,
        method=method,
        replicate=replicate,
        switch_index=switch_index,
        readout=readout,
        result=result,
        truth=truth,
        wall_time_s=wall_time_s,
    )


def estimate_lorenz_lyapunov(config: Chaos04Config) -> DiagnosticResult:
    """Lorenz の最大 Lyapunov 指数を数値推定する (D-42)。

    **真の軌道は条件に依存しない量**なので、(rho, leak, noise) ごとに積分し直す
    のは禁止 (仕様 §5 禁止する構造3)。ここが1回だけ作り、Lyapunov 時間
    (``lyapunov_time``) を掃引へ配る。

    burn-in は ``lorenz.integration_burn_in`` が既に落としているので
    ``ctx.washout`` は 0 にする (二重に捨てると「どちらが効いているか」が
    設定から読めなくなる)。``ctx.dt`` は ``sampling_interval`` (=
    ``rk4_step * sample_interval``) で、これが Lyapunov 時間正規化の分母の
    単一の真実である。

    Returns:
        ``diagnostics/lyapunov.py`` の ``max_lyapunov`` の結果。
        ``cfg.reference_value`` を設定してあれば文献値との照合も含む。
    """
    lorenz_cfg = config.lorenz
    rng = make_rng(config.base.seeds, SeedStream.TASK, 0)
    trajectory = integrate_lorenz(lorenz_cfg, initial_state(rng), lorenz_cfg.length)
    ctx = DiagnosticContext(
        washout=0,
        dt=sampling_interval(lorenz_cfg),
        propagator=lambda x, t: lorenz_sample_step(lorenz_cfg, x),
    )
    result = max_lyapunov(trajectory, ctx=ctx, cfg=config.lyapunov)
    logger.info(
        "experiment=04 lambda_max=%.4f [1/時間] lyapunov_time=%.4f dt=%.4g",
        result.scalars["lyapunov_per_time"],
        result.scalars["lyapunov_time"],
        sampling_interval(lorenz_cfg),
    )
    return result


# --- 実験 4-B: 自走 + 有効予測時間 + 長時間統計 --------------------------------

EXPERIMENT_FREERUN = "4B_freerun"
"""``freerun.csv`` の ``experiment`` 列 (4-B)。"""

FREERUN_CSV = "freerun.csv"
"""4-B の成果物名 (列順は ``FreeRunRow`` の宣言順が単一の真実)。"""

FREERUN_PROFILE_CSV = "freerun_profile.csv"
"""図が読む長形式の配列 (位相図・リターンマップ・スペクトル)。

**図は成果物 CSV の行だけを読む** (§2.2-3) ので、位相図やスペクトルのように
「行」ではなく「配列」で表現される量も、書き出したのと同じ長形式の行として
図へ渡す。03 の ``capacity_profile.csv`` と同じ役割で、成果物と図が食い違う
経路 (CSV には無いものを図が描く) を構造で塞ぐ。
"""

KIND_PHASE = "phase"
"""``FreeRunProfileRow.kind``: 位相図 (Lorenz は (x, z)、1変数系は遅延座標)。"""

KIND_RETURN_MAP = "return_map"
"""``FreeRunProfileRow.kind``: リターンマップ ``(z_n, z_(n+1))`` (D-46 の1本目)。"""

KIND_SPECTRUM = "spectrum"
"""``FreeRunProfileRow.kind``: 正規化パワースペクトル (D-46 の2本目)。"""

SOURCE_TRUTH = "truth"
"""``FreeRunProfileRow.source``: 真の軌道。"""

SOURCE_FREERUN = "freerun"
"""``FreeRunProfileRow.source``: 自走の軌道。"""

PROFILE_MAX_POINTS = 4000
"""確保軸6: 位相図に載せる点数の上限 (間引きの上限)。

PNG のサイズと描画時間はここに比例する。**上書き不能な定数**で、
``freerun_profile_rows`` が ``stats_steps`` に関係なくこの本数まで間引く
(``stats_steps`` を伸ばすと図の点数が黙って増える、を塞ぐ)。
"""


FREERUN_CSV_COLUMNS: tuple[str, ...] = tuple(
    item.name for item in dataclasses.fields(FreeRunRow)
)
"""``freerun.csv`` の列順 (``FreeRunRow`` の宣言順が単一の真実)。"""


FREERUN_PROFILE_CSV_COLUMNS: tuple[str, ...] = tuple(
    item.name for item in dataclasses.fields(FreeRunProfileRow)
)
"""``freerun_profile.csv`` の列順 (``FreeRunProfileRow`` の宣言順)。"""


def evaluate_free_run(
    config: Chaos04Config,
    outcome: FreeRunOutcome,
    *,
    dt: float,
    lyapunov_per_time: float,
    lyapunov_time: float,
) -> FreeRunEvaluation:
    """自走1本を数値で評価する (D-43 / D-45 / D-46 をここで合流させる)。

    測るのは3つで、どれも ``experiment/attractor.py`` の純関数へ委譲する。

    1. **有効予測時間** (D-43): 誤差は NRMSE 比、閾値は
       ``freerun.valid_time_threshold``、報告は Lyapunov 時間で正規化した値。
       打ち切り (自走長まで超えなかった) はフラグで残す。
    2. **3態分類** (D-45): 振幅・標準偏差・自己相関から純関数が決める。
    3. **長時間統計** (D-46): リターンマップとパワースペクトルの距離を、
       自走と**真の軌道のシャッフル代替**の両方について測る。

    シャッフル用の乱数は **task ストリームの続き**である (D-14: 5本目の
    ストリームを新設しない)。代替は真の系列の並べ替えなので、素性としても
    課題側の乱数に属する。

    Args:
        config: 04 の設定。
        outcome: ``run_free_run`` の結果 (``n_steps=stats_steps`` で回したもの)。
        dt: サンプリング間隔 [時間]。
        lyapunov_per_time: lambda_max [1/時間] (D-42)。
        lyapunov_time: ``1 / lambda_max`` [時間]。

    Returns:
        ``FreeRunEvaluation``。
    """
    freerun_cfg = config.freerun
    result = outcome.result
    readout = outcome.readout
    esn_cfg = readout.plan.task
    trajectory: FloatArray = result.predictions[: result.n_completed]
    truth_series: FloatArray = readout.plan.task.y
    del esn_cfg

    horizon = min(freerun_cfg.free_run_steps, result.predictions.shape[0])
    errors = normalized_error_curve(
        outcome.truth[:horizon], result.predictions[:horizon]
    )
    valid = valid_time_from_errors(errors, freerun_cfg.valid_time_threshold)
    sensitivity = tuple(
        valid_time_from_errors(errors, threshold)
        for threshold in VALID_TIME_THRESHOLD_GRID
    )

    verdict: RegimeVerdict = classify_regime(
        trajectory, reference=truth_series, diverged=result.diverged
    )
    free_distance: AttractorDistance = attractor_distance(truth_series, trajectory, dt)
    if trajectory.shape[0] >= 1:
        surrogate = shuffled_surrogate(
            truth_series,
            make_rng(config.base.seeds, SeedStream.TASK, outcome.replicate),
            trajectory.shape[0],
        )
        surrogate_distance: AttractorDistance = attractor_distance(
            truth_series, surrogate, dt
        )
    else:
        surrogate_distance = AttractorDistance(math.nan, math.nan, 0, 0)

    esn = _esn_of(config, outcome)
    row = FreeRunRow(
        experiment=EXPERIMENT_FREERUN,
        task=outcome.task,
        method=outcome.method,
        replicate=outcome.replicate,
        seed_reservoir=config.base.seeds.reservoir,
        seed_task=config.base.seeds.task,
        seed_split=config.base.seeds.split,
        n_units=esn.n_units,
        rho=esn.spectral_radius,
        leak_rate=esn.leak_rate,
        state_noise=esn.state_noise,
        alpha=readout.alpha,
        val_nrmse=readout.val_nrmse,
        switch_index=outcome.switch_index,
        warmup_steps=freerun_cfg.warmup_steps,
        free_run_steps=freerun_cfg.free_run_steps,
        stats_steps=int(result.predictions.shape[0]),
        dt=dt,
        lyapunov_per_time=lyapunov_per_time,
        lyapunov_time=lyapunov_time,
        valid_time_threshold=valid.threshold,
        valid_time_steps=valid.steps,
        valid_time=valid.steps * dt,
        valid_time_lyapunov=lyapunov_normalized(valid.steps, dt, lyapunov_time),
        valid_time_censored=valid.censored,
        diverged=result.diverged,
        n_completed=result.n_completed,
        regime=verdict.regime,
        amplitude_ratio=verdict.amplitude_ratio,
        std_ratio=verdict.std_ratio,
        autocorr_peak=verdict.autocorr_peak,
        return_map_distance=free_distance.return_map,
        return_map_distance_surrogate=surrogate_distance.return_map,
        spectrum_distance=free_distance.spectrum,
        spectrum_distance_surrogate=surrogate_distance.spectrum,
        closer_than_surrogate=bool(
            free_distance.return_map < surrogate_distance.return_map
            and free_distance.spectrum < surrogate_distance.spectrum
        ),
        n_stats_samples=int(trajectory.shape[0]),
        n_return_map_points=free_distance.n_return_map_points,
        n_spectrum_bins=free_distance.n_spectrum_bins,
        wall_time_s=outcome.wall_time_s,
    )
    return FreeRunEvaluation(
        row=row,
        valid_time_by_threshold=tuple(
            lyapunov_normalized(item.steps, dt, lyapunov_time) for item in sensitivity
        ),
        censored_by_threshold=tuple(item.censored for item in sensitivity),
        trajectory=trajectory,
        truth_series=truth_series,
        truth_aligned=outcome.truth[: trajectory.shape[0]],
    )


def _esn_of(config: Chaos04Config, outcome: FreeRunOutcome) -> ESNConfig:
    """行に載せる ESN の条件 (対照の行にも同じ条件を書く)。

    対照 (線形・遅延線) はリザバーを使わないが、**同じ条件で回した対照である**
    ことを行から読めるようにするため同じ値を書く (3-C が ``capacity.csv`` の
    条件列を埋めるのと同じ流儀)。
    """
    del outcome
    return chaos_esn_config(config.base)


def _phase_points(series: FloatArray, lag: int) -> FloatArray:
    """位相図の2次元投影。多変数なら (第0成分, 最終成分)、1変数なら遅延座標。

    ``lag`` は**真の軌道から**決めた1個を自走側にも使う (別々に決めると同じ
    座標系で重ね描きできない)。
    """
    if series.shape[1] >= 2:
        projected: FloatArray = np.stack([series[:, 0], series[:, -1]], axis=1)
        return projected
    if series.shape[0] <= lag:
        return np.empty((0, 2), dtype=np.float64)
    embedded: FloatArray = np.stack(
        [series[lag:, 0], series[: series.shape[0] - lag, 0]], axis=1
    )
    return embedded


def _thinned(points: FloatArray) -> FloatArray:
    """確保軸6: 図に載せる点数を ``PROFILE_MAX_POINTS`` まで間引く。"""
    if points.shape[0] <= PROFILE_MAX_POINTS:
        return points
    stride = int(np.ceil(points.shape[0] / PROFILE_MAX_POINTS))
    thinned: FloatArray = points[::stride][:PROFILE_MAX_POINTS]
    return thinned


def _profile_block(
    row: FreeRunRow, kind: str, source: str, points: FloatArray
) -> list[FreeRunProfileRow]:
    return [
        FreeRunProfileRow(
            experiment=row.experiment,
            task=row.task,
            method=row.method,
            replicate=row.replicate,
            kind=kind,
            source=source,
            index=index,
            x=float(point[0]),
            y=float(point[1]),
        )
        for index, point in enumerate(points)
    ]


def freerun_profile_rows(
    evaluation: FreeRunEvaluation, dt: float
) -> tuple[FreeRunProfileRow, ...]:
    """図が読む長形式の行を組む (**診断も実験もここでは走らせない**)。

    位相図・リターンマップ・スペクトルの3種類を、真の軌道と自走の両方について
    出す。点数は ``PROFILE_MAX_POINTS`` (確保軸6) で間引く。

    Args:
        evaluation: ``evaluate_free_run`` の結果。
        dt: サンプリング間隔 [時間] (スペクトルの周波数軸)。

    Returns:
        長形式の行。
    """
    row = evaluation.row
    truth = evaluation.truth_series
    trajectory = evaluation.trajectory
    lag = first_autocorrelation_zero(truth)
    rows: list[FreeRunProfileRow] = []
    rows += _profile_block(
        row, KIND_PHASE, SOURCE_TRUTH, _thinned(_phase_points(truth, lag))
    )
    rows += _profile_block(
        row, KIND_RETURN_MAP, SOURCE_TRUTH, _thinned(return_map_points(truth))
    )
    if trajectory.shape[0] >= 3:
        rows += _profile_block(
            row, KIND_PHASE, SOURCE_FREERUN, _thinned(_phase_points(trajectory, lag))
        )
        rows += _profile_block(
            row,
            KIND_RETURN_MAP,
            SOURCE_FREERUN,
            _thinned(return_map_points(trajectory)),
        )
    n_common = min(truth.shape[0], trajectory.shape[0])
    if n_common >= 8:
        for source, series in (
            (SOURCE_TRUTH, truth[:n_common]),
            (SOURCE_FREERUN, trajectory[:n_common]),
        ):
            frequencies, power = power_spectrum(series, dt)
            rows += _profile_block(
                row,
                KIND_SPECTRUM,
                source,
                _thinned(np.stack([frequencies, power], axis=1)),
            )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class FreeRunResults:
    """実験 4-B の結果。

    Attributes:
        evaluations: 自走1本ごとの評価 (課題 x 手法 x レプリケート)。
        profile_rows: 代表レプリケートの長形式の行 (図の材料)。
        sensitivity: 閾値感度の要約 (``meta.json``)。
        attractor: アトラクタ再現の判定 (D-46。``meta.json``)。
        wall_time_s: 4-B 全体の実測 wall time [秒]。
    """

    evaluations: tuple[FreeRunEvaluation, ...]
    profile_rows: tuple[FreeRunProfileRow, ...]
    sensitivity: tuple[ValidTimeSensitivity, ...]
    attractor: tuple[AttractorVerdict, ...]
    wall_time_s: float

    @property
    def rows(self) -> tuple[FreeRunRow, ...]:
        """``freerun.csv`` と同じ行。"""
        return tuple(evaluation.row for evaluation in self.evaluations)


PROFILE_REPLICATE = 0
"""図に載せる代表レプリケート。**結果を見て選ばない** (常に 0)。

「一番きれいな回」を選べる形にすると、図が主張の証拠でなくなる。
"""

FREERUN_METHODS: tuple[str, ...] = (LINEAR, DELAY_LINE, ESN_METHOD)
"""4-B が自走させる手法 (01 の3手法すべて)。

対照も自走させるのは受け入れ条件3 の後半 (「自走では対照が成立しない」) を
**数値で**示すためである。並びは 01 の ``build_methods`` と同じ。
"""


def run_freerun_experiment(
    config: Chaos04Config, lyapunov: DiagnosticResult
) -> FreeRunResults:
    """実験 4-B を回す (自走 -> 有効予測時間 -> 長時間統計)。

    **確保軸4 (``stats_steps``) をここで、自走を1ステップも回す前に検査する。**

    1レプリケートにつき ``ReplicatePlan`` は1個だけ作り、3手法で共有する
    (01 の ``run_task(..., plan0=...)`` と同じ形)。共有しないと「同じ分割・
    同じ状態行列で比べた」が構造ではなく偶然になる。

    Args:
        config: 04 の設定。
        lyapunov: ``estimate_lorenz_lyapunov`` の結果 (**真の軌道は条件に
            依存しない量**なので掃引の中で推定し直さない、仕様 §5 禁止構造3)。

    Returns:
        ``FreeRunResults``。

    Raises:
        ValueError: 確保軸を超える設定、または課題・分割側の値域違反。
    """
    started = time.perf_counter()
    validate_stats_bounds(config.freerun.stats_steps)
    # 確保軸10 (逐次実行の本数)。4-B は手法ごとに独立な閉ループを回すので、
    # 4-A (状態行列を3手法で共有) とは違い手法数も掛かる。
    validate_sequential_run_count(
        len(chaos_task_entries(config))
        * config.base.n_replicates
        * len(FREERUN_METHODS)
    )

    evaluations: list[FreeRunEvaluation] = []
    profile: list[FreeRunProfileRow] = []
    for entry in chaos_task_entries(config):
        validate_state_matrix_bounds(entry.esn.n_units, task_length(config, entry.name))
        dt = task_sampling_interval(config, entry.name)
        # **lambda_max を推定してあるのは Lorenz だけ** (D-42)。MG の最大
        # Lyapunov 指数は 04 では推定していない —— Benettin 法には遅延系の履歴
        # (tau / h = 170 次元) を状態とする伝播器が要り、T4 が作ったのは Lorenz の
        # ものだけである。**推定していない量を他の系の値で埋めない**ので、MG 行の
        # Lyapunov 列と valid_time_lyapunov は nan になる (生の時間は出る)。
        if entry.name == TASK_NAME_LORENZ:
            lyapunov_per_time = lyapunov.scalars["lyapunov_per_time"]
            lyapunov_time = lyapunov.scalars["lyapunov_time"]
        else:
            lyapunov_per_time = math.nan
            lyapunov_time = math.nan
        for replicate in range(config.base.n_replicates):
            plan: ReplicatePlan | None = None
            for method in FREERUN_METHODS:
                outcome = run_free_run(
                    config,
                    entry,
                    replicate,
                    n_steps=config.freerun.stats_steps,
                    method=method,
                    plan=plan,
                )
                plan = outcome.readout.plan
                evaluation = evaluate_free_run(
                    config,
                    outcome,
                    dt=dt,
                    lyapunov_per_time=lyapunov_per_time,
                    lyapunov_time=lyapunov_time,
                )
                evaluations.append(evaluation)
                if replicate == PROFILE_REPLICATE and method == ESN_METHOD:
                    profile.extend(freerun_profile_rows(evaluation, dt))
    wall_time_s = time.perf_counter() - started
    logger.info(
        "experiment=%s 行数=%d 課題=%s 手法=%s "
        "有効予測時間の中央値 [1/lambda]=%s (%.2fs)",
        EXPERIMENT_FREERUN,
        len(evaluations),
        sorted({item.row.task for item in evaluations}),
        list(FREERUN_METHODS),
        {
            method: round(
                float(
                    np.median(
                        [
                            item.row.valid_time_lyapunov
                            for item in evaluations
                            if item.row.method == method
                        ]
                    )
                ),
                3,
            )
            for method in FREERUN_METHODS
        },
        wall_time_s,
    )
    return FreeRunResults(
        evaluations=tuple(evaluations),
        profile_rows=tuple(profile),
        sensitivity=summarize_valid_time(evaluations),
        attractor=summarize_attractor(evaluations),
        wall_time_s=wall_time_s,
    )


def write_freerun_csv(rows: Sequence[FreeRunRow], path: Path) -> Path:
    """4-B の結果を CSV に書く (列順は ``FreeRunRow`` の宣言順)。"""
    return write_rows_csv(rows, path, FREERUN_CSV_COLUMNS)


def write_freerun_profile_csv(rows: Sequence[FreeRunProfileRow], path: Path) -> Path:
    """図が読む長形式の行を CSV に書く (列順は宣言順)。"""
    return write_rows_csv(rows, path, FREERUN_PROFILE_CSV_COLUMNS)


ONESTEP_ARTIFACTS: tuple[str, ...] = (ONESTEP_CSV, META_JSON)
"""4-A が書く成果物の一覧 (**成果物を列挙する唯一の場所**)。

図5枚と ``freerun.csv`` / ``stability.csv`` は次サイクル (T5) が足す。ここに
名前を並べておくと、足したときに一覧の更新漏れがテストで落ちる。
"""


def write_onestep_csv(rows: Sequence[ResultRow], path: Path) -> Path:
    """4-A の結果を CSV に書く (列順は 01 の ``CSV_COLUMNS`` が単一の真実)。

    ``write_comparison_csv`` と同じ列順・同じ ``ResultRow`` を使う。01 の書き
    出し関数をそのまま呼ばないのは出力ファイル名が違うだけの差だが、列の定義は
    複製せず ``CSV_COLUMNS`` を参照する (D-05 の公平性の列が片方だけ欠ける事故を
    防ぐ)。
    """
    return write_rows_csv(rows, path, CSV_COLUMNS)


def run_and_report_onestep(config: Chaos04Config, out_dir: Path) -> tuple[Path, ...]:
    """4-A を回して ``onestep.csv`` と ``meta.json`` を書く。

    Lyapunov 指数の推定 (D-42) も1回だけ回して ``meta.json`` に載せる ——
    **真の軌道は条件に依存しない量**なので、掃引の中で積分し直さない
    (仕様 §5 禁止する構造3)。有効予測時間の正規化 (D-43) はこの値を読む。

    Args:
        config: 04 の設定。
        out_dir: 出力ディレクトリ (``results/04_chaotic_freerun``)。

    Returns:
        書いたファイルのパス (``ONESTEP_ARTIFACTS`` と同じ順)。
    """
    started = time.perf_counter()
    lyapunov_started = time.perf_counter()
    lyapunov = estimate_lorenz_lyapunov(config)
    wall_time_lyapunov_s = time.perf_counter() - lyapunov_started

    onestep_started = time.perf_counter()
    rows = run_onestep(config)
    wall_time_onestep_s = time.perf_counter() - onestep_started

    csv_path = write_onestep_csv(rows, out_dir / ONESTEP_CSV)
    wall_time_s = time.perf_counter() - started
    meta_path = write_meta_for(
        config,
        config.base.seeds,
        wall_time_s,
        len(rows),
        out_dir / META_JSON,
        extra={
            "lorenz_dt": sampling_interval(config.lorenz),
            "lyapunov": dict(lyapunov.scalars),
            "lyapunov_params": dict(lyapunov.params),
            "wall_time_breakdown": {
                "lyapunov_s": wall_time_lyapunov_s,
                "onestep_s": wall_time_onestep_s,
            },
        },
    )
    logger.info(
        "04 の成果物を書きました: %s (行数=%d, wall_time=%.1fs)",
        [str(path) for path in (csv_path, meta_path)],
        len(rows),
        wall_time_s,
    )
    return (csv_path, meta_path)


__all__ = [
    "CHAOS_ESN_SECTION",
    "EXPERIMENT_FREERUN",
    "FREERUN_CSV",
    "FREERUN_CSV_COLUMNS",
    "FREERUN_METHODS",
    "FREERUN_PROFILE_CSV",
    "FREERUN_PROFILE_CSV_COLUMNS",
    "FREE_RUN_SPEC",
    "KIND_PHASE",
    "KIND_RETURN_MAP",
    "KIND_SPECTRUM",
    "ONESTEP_ARTIFACTS",
    "ONESTEP_CSV",
    "PROFILE_MAX_POINTS",
    "PROFILE_REPLICATE",
    "SOURCE_FREERUN",
    "SOURCE_TRUTH",
    "AttractorVerdict",
    "ClosedLoop",
    "FreeRunEvaluation",
    "FreeRunOutcome",
    "FreeRunProfileRow",
    "FreeRunResults",
    "FreeRunRow",
    "TeacherForcedReadout",
    "ValidTimeSensitivity",
    "chaos_esn_config",
    "chaos_task_entries",
    "closed_loop_setup",
    "delay_line_state_updater",
    "esn_state_updater",
    "estimate_lorenz_lyapunov",
    "evaluate_free_run",
    "fit_teacher_forced",
    "freerun_profile_rows",
    "lorenz_task_entry",
    "mackey_glass_task_entry",
    "method_candidates",
    "passthrough_state_updater",
    "run_and_report_onestep",
    "run_free_run",
    "run_freerun_experiment",
    "run_onestep",
    "sign_test_p_value",
    "standardize_steps_for",
    "summarize_attractor",
    "summarize_valid_time",
    "task_length",
    "task_sampling_interval",
    "validate_free_run_bounds",
    "validate_standardization_window",
    "write_freerun_csv",
    "write_freerun_profile_csv",
    "write_onestep_csv",
]

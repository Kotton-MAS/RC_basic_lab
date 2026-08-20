"""実験 4-A (教師強制の1ステップ先予測) と自走の入口の配線 (D-31 / D-44 / D-50).

4-A は **01 の ``run_task`` をそのまま通す** (D-31 と同じ形)。手法の列挙
(``build_methods``)・alpha 格子の共有 (D-04)・全手法が同一の行 index で学習評価
すること (D-05)・ESN の構造ハイパーパラメータを検証分割で選ばないこと (D-08)
は、すべて 01 の経路が担保する。ここが組み立てるのは ``TaskEntry`` (課題の生成
関数 + ESN 設定) だけで、**``build_tasks`` にも ``ExperimentConfig`` にも 04 の
課題を足さない** (足すと 01 の ``comparison.csv`` に行が増えて 01 の成果物が
変わる)。

自走の入口もここにある。``readout/autoregressive.py`` は ``reservoir`` を
知らない (D-50) ので、ESN を ``StateUpdater`` に適合させるアダプタ
(``esn_state_updater``) と、確保軸の検査 (``validate_free_run_bounds``) は
実験層の責任である。

**自走は教師強制で学習した係数をそのまま使う** (D-44)。``fit_teacher_forced``
が返した係数オブジェクトが ``FreeRunResult.coefficients`` に**同一オブジェクト
として**現れるので、自走のたびに学習し直す実装 (仕様 §5 禁止する構造1) は
同一性の検査で落ちる。
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rc_basics_lab.config import Chaos04Config, ESNConfig, ExperimentConfig
from rc_basics_lab.diagnostics.base import DiagnosticContext, DiagnosticResult
from rc_basics_lab.diagnostics.lyapunov import max_lyapunov
from rc_basics_lab.experiment.capacity import (
    validate_n_units_bound,
    validate_state_matrix_bounds,
)
from rc_basics_lab.experiment.report import META_JSON, write_meta_for
from rc_basics_lab.experiment.runner import (
    CSV_COLUMNS,
    ESN_METHOD,
    ReplicatePlan,
    ResultRow,
    TaskEntry,
    plan_replicate,
    run_task,
)
from rc_basics_lab.experiment.split import Split
from rc_basics_lab.readout.autoregressive import (
    FreeRunResult,
    StateUpdater,
    free_run,
)
from rc_basics_lab.readout.design import (
    DesignMatrix,
    ReservoirSpec,
)
from rc_basics_lab.readout.ridge import fit_ridge, predict, select_alpha
from rc_basics_lab.reservoir.esn import ESN
from rc_basics_lab.seeds import SeedStream, make_rng
from rc_basics_lab.tasks.chaotic import (
    TASK_NAME_LORENZ,
    generate_lorenz,
    generate_standardized_mackey_glass,
    initial_state,
    integrate_lorenz,
    lorenz_sample_step,
    sampling_interval,
)
from rc_basics_lab.tasks.mackey_glass import TASK_NAME as TASK_NAME_MACKEY_GLASS
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


def chaos_esn_config(base: ExperimentConfig) -> ESNConfig:
    """04 が使う ESN 設定を返す (``CHAOS_ESN_SECTION`` の1本)。

    「どのセクションを読むか」をここ1か所に閉じる。呼び出し側が属性を直接
    書くと、宣言したセクションと実際に読むセクションが食い違っても何も落ちない。
    """
    return base.esn_mackey_glass


def lorenz_task_entry(config: Chaos04Config) -> TaskEntry:
    """Lorenz の ``TaskEntry`` を組む (**``build_tasks`` には足さない**、D-31)。"""
    return TaskEntry(
        name=TASK_NAME_LORENZ,
        esn=chaos_esn_config(config.base),
        generate=lambda rng: generate_lorenz(config.lorenz, rng),
    )


def mackey_glass_task_entry(config: Chaos04Config) -> TaskEntry:
    """04 の MG の ``TaskEntry`` を組む (生成は 01 の実装へ委譲、D-41)。

    生成パラメータの単一の真実は ``config.base.mackey_glass`` (01 の
    ``MackeyGlassConfig``) で、04 が足すのは標準化だけである。
    """
    return TaskEntry(
        name=TASK_NAME_MACKEY_GLASS,
        esn=chaos_esn_config(config.base),
        generate=lambda rng: generate_standardized_mackey_glass(
            config.base.mackey_glass,
            rng,
            standardize_steps=config.mackey_glass.standardize_steps,
        ),
    )


def chaos_task_entries(config: Chaos04Config) -> tuple[TaskEntry, ...]:
    """04 が回す課題 (Lorenz 主 + MG 従、仕様 §8)。**課題を列挙する唯一の場所**。"""
    return (lorenz_task_entry(config), mackey_glass_task_entry(config))


def task_length(config: Chaos04Config, task_name: str) -> int:
    """課題名 -> 系列長。確保軸の検査で「何行の状態を作るか」を知るために使う。

    ``TASK_LENGTH_FIELDS`` (01) と同じ役割だが、04 は Lorenz の長さを
    ``config.lorenz`` に、MG の長さを ``config.base.mackey_glass`` に持つので
    対応表がここに要る。未知の課題名は ``ValueError`` にする (課題を足して
    ここへの登録を忘れると確保軸の検査が黙って効かなくなる)。
    """
    match task_name:
        case _ if task_name == TASK_NAME_LORENZ:
            return config.lorenz.length
        case _ if task_name == TASK_NAME_MACKEY_GLASS:
            return config.base.mackey_glass.length
        case _:
            raise ValueError(f"04 の課題ではありません: {task_name!r}")


def validate_free_run_bounds(free_run_steps: int, n_units: int) -> None:
    """確保軸3 (``free_run_steps * n_units``) を**確保より前に**検査する (D-34)。

    自走は ``(free_run_steps, n_units)`` の状態行列を確保するので、容量実験の
    状態行列とまったく同じ軸である。**04 で新しい上限を作らず**、
    ``experiment/capacity.py`` の ``validate_state_matrix_bounds`` を再利用する
    (上限が2か所にあると片方だけ緩められる)。

    Raises:
        ValueError: ``free_run_steps`` が 1 未満、または確保軸が上限を超える場合。
    """
    if free_run_steps < 1:
        raise ValueError(
            f"free_run_steps は 1 以上である必要があります: {free_run_steps}"
        )
    validate_state_matrix_bounds(n_units, free_run_steps)


def validate_standardization_window(standardize_steps: int, split: Split) -> None:
    """標準化係数の推定区間が訓練区間の内側に収まることを検査する (D-41)。

    ``Standardizer.from_training_prefix`` は系列の**先頭**から係数を推定する。
    先頭 ``standardize_steps`` 行が検証区間・テスト区間に食い込むと、評価区間の
    統計量が係数へ混ざる —— 予測が当たっていない区間でも平均・分散が揃うため
    「当たっているように見える」壊れ方をし、図でも有効予測時間でも検出できない
    (仕様 §10-2)。分割が決まるのは課題生成の**後**なので、検査はここで行う。

    Raises:
        ValueError: 推定区間が訓練区間の終端を超える場合。
    """
    if standardize_steps > split.train.stop:
        raise ValueError(
            "標準化係数の推定区間が訓練区間を越えています (D-41): "
            f"standardize_steps={standardize_steps} > train.stop={split.train.stop} "
            "(評価区間の統計量が係数に混ざると『当たっているように見える』"
            "壊れ方をする)"
        )


def standardize_steps_for(config: Chaos04Config, task_name: str) -> int:
    """課題名 -> 標準化係数の推定に使う先頭サンプル数 (D-41)。"""
    match task_name:
        case _ if task_name == TASK_NAME_LORENZ:
            return config.lorenz.standardize_steps
        case _ if task_name == TASK_NAME_MACKEY_GLASS:
            return config.mackey_glass.standardize_steps
        case _:
            raise ValueError(f"04 の課題ではありません: {task_name!r}")


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
    """手法名 -> 候補の特徴仕様 (**手法の列挙は 01 の ``build_methods`` が単一の真実**)。

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


def esn_state_updater(esn: ESN, rng: np.random.Generator | None = None) -> StateUpdater:
    """ESN を ``StateUpdater`` (D-50) に適合させるアダプタ。

    **``ESN.run`` ではなく ``ESN.step`` を使う** (仕様 §5 禁止する構造8)。自走は
    ``u[t+1]`` が ``y_hat[t]`` に依存するので、入力系列が既知でないと動かない
    ``run`` では書けない。

    **``state_noise > 0`` なら ``rng`` を渡すのが正しい** —— 自走は伝播器では
    なく**軌道を作る**呼び出しであり、学習時の状態にノイズを入れた設定で自走中
    だけノイズを外すと、学習時と評価時で別の系を測ることになる (D-36)。02 の
    ``esn_propagator`` が決定的でなければならない (D-48) のは、条件付き
    Lyapunov 指数が「同じ軌道のまわりの摂動の成長率」を測るからであって、
    「ESN は常に決定的に回す」という規則ではない。

    Raises:
        ValueError: ``state_noise > 0`` なのに ``rng`` が ``None`` の場合。
    """
    if esn.config.state_noise > 0.0 and rng is None:
        raise ValueError(
            "state_noise > 0 の自走には rng が必要です (D-36)。"
            "黙ってノイズ無しで自走すると、学習時とは別の系を評価することに"
            "なる。決定性が要るのは 02 の伝播器 (D-48) であって自走ではない"
        )

    def update(x: FloatArray, u: FloatArray) -> FloatArray:
        return esn.step(x, u, rng)

    return update


@dataclass(frozen=True, slots=True)
class FreeRunOutcome:
    """自走1本ぶんの結果 (4-B が統計を載せる土台)。

    Attributes:
        task: 課題名。
        replicate: レプリケート番号。
        switch_index: 教師強制から自走へ切り替えた行 index。
        readout: 教師強制で学習した read-out (D-44)。
        result: 自走の生の結果 (入力・状態・予測・打ち切り)。
        truth: 各自走ステップに対応する真の目標 ``(free_run_steps, D_out)``。
        wall_time_s: 実測 wall time [秒]。
    """

    task: str
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

    readout = fit_teacher_forced(config, task_entry, replicate)
    plan = readout.plan
    switch_index = plan.split.test.start + freerun_cfg.warmup_steps - 1
    last_index = switch_index + freerun_cfg.free_run_steps
    if last_index >= plan.task.n_steps:
        raise ValueError(
            "自走に必要な行がテスト区間の先にありません "
            f"(T={plan.task.n_steps}, test.start={plan.split.test.start}, "
            f"warmup_steps={freerun_cfg.warmup_steps}, "
            f"free_run_steps={freerun_cfg.free_run_steps})"
        )

    reservoir = ESN(
        esn_cfg,
        make_rng(config.base.seeds, SeedStream.RESERVOIR, replicate),
        n_inputs=plan.task.n_inputs,
    )
    # 自走のノイズは reservoir ストリームの続き (D-14 に4本目を足さない)。
    noise_rng = (
        make_rng(config.base.seeds, SeedStream.RESERVOIR, replicate)
        if esn_cfg.state_noise > 0.0
        else None
    )
    updater = esn_state_updater(reservoir, noise_rng)

    x0: FloatArray = plan.states[switch_index]
    u0: FloatArray = predict(
        readout.design.phi[switch_index : switch_index + 1], readout.coefficients
    )[0]
    result = free_run(
        updater,
        FREE_RUN_SPEC,
        readout.coefficients,
        x0,
        u0,
        steps,
    )
    truth: FloatArray = plan.task.y[switch_index + 1 : last_index + 1]
    wall_time_s = time.perf_counter() - started
    logger.info(
        "experiment=4B_freerun task=%s replicate=%d switch=%d steps=%d "
        "diverged=%s completed=%d alpha=%.3g (%.2fs)",
        task_entry.name,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))
    return path


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
    "FREE_RUN_SPEC",
    "ONESTEP_ARTIFACTS",
    "ONESTEP_CSV",
    "FreeRunOutcome",
    "TeacherForcedReadout",
    "chaos_esn_config",
    "chaos_task_entries",
    "esn_state_updater",
    "estimate_lorenz_lyapunov",
    "fit_teacher_forced",
    "lorenz_task_entry",
    "mackey_glass_task_entry",
    "run_and_report_onestep",
    "run_free_run",
    "run_onestep",
    "standardize_steps_for",
    "task_length",
    "validate_free_run_bounds",
    "validate_standardization_window",
    "write_onestep_csv",
]

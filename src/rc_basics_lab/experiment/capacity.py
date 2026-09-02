"""実験 3-A / 3-B / 3-B' の配線層 —— ESN と容量診断をつなぐ唯一の場所.

``experiment/esp.py`` (02) と同じ分業で、``diagnostics/`` は行列を受け取って
``DiagnosticResult`` を返すだけ、どの行列をどの設定で渡すかは実験層が決める。
このモジュールは ``reservoir`` と ``diagnostics`` の**両方**を import してよい
場所であり (02 の ``esp.py`` と並ぶ2本目)、D-12 / D-23 が禁じているのは
``diagnostics -> config / reservoir`` の向きだけである。

**1条件につき状態行列 ``X`` は1回だけ作る** (D-26 / 仕様 §5 の禁止構造)。
MC と IPC は同じ ``X`` と同じ駆動入力 ``u`` を見る。条件ごとに ``X`` を2回
作ると ``ESN.run`` (T に線形、Python ループ) の実行時間がそのまま倍になり、
かつ「MC が見た系列と IPC が見た系列が違う」という比較不能な CSV になる。

**``X`` は診断へ渡す前に読み取り専用にする** (D-35)。呼び出し側であるここで
塞ぐのは、``CapacityProblem`` が自分の持つビューは読み取り専用にできても
**元の ``X`` は塞げない**ため。

なお MC と IPC は ``t0`` が異なる (MC は ``max(washout, mc.max_delay)``、
IPC は ``max(washout, max(ipc.max_delay_by_degree))``) ため、
``CapacityProblem`` は1条件あたり2個作られる。**これは正常であり**、
「1個にまとめる」最適化をすると D-24 の単一基準点がどちらかの診断で壊れる。

**サロゲートのシードは全条件で1個を共有する** (D-37、共通乱数法)。

**1条件の処理は3段に分かれている** (F-3b1-1-004)。``evaluate_capacity_condition``
は軌道生成と下の2つを繋ぐ薄い層であり、``X`` を**外から**渡す経路
(3-C は 01 の ``run_task`` が作った状態を測る) はこの2つを直接呼ぶ。

- ``measure_capacity(states, u, *, ctx, mc_cfg, ipc_cfg)`` —— read-only 化
  (D-35) と2診断の呼び出し (D-37 の ``ctx`` 共有) だけ
- ``capacity_row_from(measurement, *, experiment, ...)`` —— ``CapacityRow``
  (約35フィールド) の組み立て。**実験ごとに複製しない**
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass

import numpy as np

from rc_basics_lab.config import (
    Capacity03Config,
    DriveConfig,
    IpcConfig,
    MemoryCapacityConfig,
    ReservoirSweepConfig,
)
from rc_basics_lab.diagnostics.base import DiagnosticContext
from rc_basics_lab.diagnostics.ipc import ipc
from rc_basics_lab.diagnostics.memory_capacity import memory_capacity
from rc_basics_lab.experiment.capacity_bounds import (
    validate_n_units_bound,
    validate_sequential_run_count,
    validate_state_matrix_bounds,
    validate_total_step_count,
)
from rc_basics_lab.experiment.capacity_rows import (
    DIAGNOSTIC_IPC,
    DIAGNOSTIC_MC,
    CapacityCondition,
    CapacityMeasurement,
    CapacityOutcome,
    CapacityProfileRow,
    CapacityRow,
    CapacityRowTiming,
    capacity_outcome_from,
    capacity_row_from,
    identity_for,
    profile_rows,
)
from rc_basics_lab.experiment.esp import (
    ReferenceTrajectory,
    simulate_reference_trajectory,
)
from rc_basics_lab.types import FloatArray

logger = logging.getLogger(__name__)

EXPERIMENT_MC_SWEEP = "3A_mc_sweep"
"""実験 3-A: rho x リーク率 に対する線形メモリ容量 (受け入れ条件1)。"""

EXPERIMENT_IPC_SWEEP = "3B_ipc_sweep"
"""実験 3-B: rho x リーク率 に対する IPC の次数・遅延分解 (受け入れ条件4)。"""

EXPERIMENT_CONSERVATION = "3Bp_conservation"
"""実験 3-B': ノイズ下での保存則 IPC_total <= N (受け入れ条件2)。"""

EXPERIMENT_NARMA10 = "3C_narma10"
"""実験 3-C: 公平な対照下での NARMA10 (受け入れ条件5)。

**条件は ``CapacityCondition`` で表現できない** —— 状態は 01 の
``plan_replicate`` (課題の入力で駆動する ESN) が作るので、rho / leak_rate /
sigma_u / n_steps を掃引の軸として持たない。それでも同じ実験ラベルの空間に
載せるのは、3-C の ESN の容量 (MC / IPC) を ``capacity.csv`` に同じ列で
書き、成績 (``narma10.csv``) と条件キーで join できるようにするためである
(要件書「そのリザバーの IPC プロファイルと突き合わせ、NARMA10 の成績が容量の
どの成分と相関するかを見る」)。行は ``experiment/narma.py`` が
``measure_capacity`` -> ``capacity_row_from`` の2段で作る (F-3b1-1-004)。
"""

EXPERIMENT_LENGTH_SWEEP = "3L_length_sweep"
"""系列長 T に対する容量の飽和 (``make saturation-03``)。**本番には含めない**。

``figures-03`` の成果物 (``capacity.csv``) には入らず ``capacity_length.csv``
に別途書く (仕様 §8: 「本番 (figures-03) には含めない」)。それでも実験ラベルを
名乗るのは、``length_sweep.*`` の設定が**他の実験の行を動かしていない**ことを
scope 検査 (``tests/test_config_wiring_capacity.py``) で測るためである。
ラベルを共有すると、T 掃引の設定を変えたときに 3-A の行まで動いても
気づけない。
"""

EXPERIMENT_SYMMETRY = "3S_symmetry"
"""3-S: 駆動入力の対称性と IPC の偶数次 (``make symmetry-03``、D-116)。"""

CAPACITY_EXPERIMENTS: tuple[str, ...] = (
    EXPERIMENT_MC_SWEEP,
    EXPERIMENT_IPC_SWEEP,
    EXPERIMENT_CONSERVATION,
    EXPERIMENT_NARMA10,
    EXPERIMENT_LENGTH_SWEEP,
    EXPERIMENT_SYMMETRY,
)
"""``CapacityRow.experiment`` が取りうる値。

``capacity.csv`` に出るのは先頭4つ (掃引3本 + 3-C) で、``3L_length_sweep`` は
``capacity_length.csv`` (``make saturation-03``)、``3S_symmetry`` は
``capacity_symmetry.csv`` (``make symmetry-03``) 側にしか現れない。
"""

FIGURE_EXPERIMENTS: tuple[str, ...] = (
    EXPERIMENT_MC_SWEEP,
    EXPERIMENT_IPC_SWEEP,
    EXPERIMENT_CONSERVATION,
    EXPERIMENT_NARMA10,
)
"""``make figures-03`` (``capacity.csv``) が回す実験。予算 900 秒の対象。

3-C は掃引ではなく1条件だが、``capacity.csv`` に行が出る以上ここに入れる
(``meta.json`` の ``wall_time_breakdown`` の並びもこの定数が単一の真実)。
"""


def _validate_condition_bounds(condition: CapacityCondition) -> None:
    """状態行列・ESN の確保より前に、確保量に上書き不能な絶対上限をかける。

    D-34 の規律 (「確保より前に落とす」) を実験層の確保軸 (``n_units`` /
    ``n_steps``) にも適用する (F-3b1-1-017)。**``CapacityCondition`` を組み立てる
    経路すべてが ``simulate_condition_trajectory`` を通ることで守られる**
    (F-3b2-1-001/HIGH-1)。3-A / 3-B / 3-B' / length_sweep
    (``evaluate_capacity_condition`` 経由) としきい値法比較
    (``run_threshold_comparison``、``CapacityCondition`` は組むが3-A/3-B/3-B'/
    length_sweep のいずれでもない代表条件1本) の5経路が対象で、3-C
    (``run_narma10``) だけは ``CapacityCondition`` を持たない (状態は 01 の
    ``run_task`` が作る) ためここでは守れず、``tasks/narma.py`` の
    ``_validate`` (``length`` 軸) と ``validate_state_matrix_bounds``
    (``n_units`` 軸) の2本で塞ぐ。以前は ``evaluate_capacity_condition`` が
    直接この関数を呼んでおり、``run_threshold_comparison`` は
    ``CapacityCondition`` を組み立てながらここを1回も呼ばずに素通りしていた
    (3b-2 reviewer-security の実測)。
    """
    validate_state_matrix_bounds(condition.n_units, condition.n_steps)


def reservoir_config_for(
    config: Capacity03Config, condition: CapacityCondition
) -> ReservoirSweepConfig:
    """1条件ぶんのリザバー構造を組む (**値の組み立てはここ1か所**)。

    03 は 02 の ``ReservoirSweepConfig`` を**設定として**再利用しない (仕様
    §3.2) が、``simulate_reference_trajectory`` の引数型としては再利用する。
    横断共有の2つ (``input_scale`` / ``density``) は ``config.reservoir`` から、
    ``n_units`` は条件から取る (D-32)。``n_replicates`` は
    ``n_replicates_for(config, condition.experiment)`` の**実効値**を渡す
    (F-3b1-1-006)。``drive_config_for`` が『設定したのに効いていない
    フィールドを作らない』ために ``n_pairs`` を外しているのと同じ規律で、
    ``simulate_reference_trajectory`` は ``n_replicates`` を読まないため実害は
    無かったが、3-B' で ``conservation.n_replicates`` が横断共有値を上書き
    している場合に実効値と食い違った ``ReservoirSweepConfig`` が境界を越える
    (実効1に対し3が渡る) 状態だった。
    """
    return ReservoirSweepConfig(
        input_scale=config.reservoir.input_scale,
        n_units=condition.n_units,
        density=config.reservoir.density,
        n_replicates=n_replicates_for(config, condition.experiment),
    )


def drive_config_for(
    config: Capacity03Config, condition: CapacityCondition
) -> DriveConfig:
    """1条件ぶんの駆動条件を組む。

    ``n_pairs`` (比較軌道の本数) は ESP 判定専用で
    ``simulate_reference_trajectory`` は読まないため、02 の既定のまま触らない。
    03 の設定 (``CapacityDriveConfig``) に ``n_pairs`` を持たせないのは
    「設定したのに効いていない」フィールドを作らないためである。
    """
    return DriveConfig(
        distribution=config.drive.distribution,
        n_steps=condition.n_steps,
        washout=config.drive.washout,
    )


def ipc_config_for(config: Capacity03Config, experiment: str) -> IpcConfig:
    """実験ラベルに応じた IPC の測定条件を返す (**上書きは片方向**)。

    3-B' (``conservation``) だけが ``conservation.max_delay_by_degree`` で
    ``config.ipc`` を上書きする。保存則 IPC_total <= N は「打ち切りの外に
    残った容量」が見えないと N に届かないため、この実験だけ遅延を深く取る
    必要がある。逆向き (3-B' の値を ``config.ipc`` の既定にする) にすると
    3-A / 3-B の掃引まで重くなるので、上書きはここから外へ出さない。
    """
    if experiment == EXPERIMENT_CONSERVATION:
        return dataclasses.replace(
            config.ipc, max_delay_by_degree=config.conservation.max_delay_by_degree
        )
    return config.ipc


def simulate_condition_trajectory(
    config: Capacity03Config,
    condition: CapacityCondition,
    *,
    drive_offset: float = 0.0,
) -> ReferenceTrajectory:
    """``CapacityCondition`` から参照軌道 ``X`` を作る (**確保より前に上限検査**)。

    D-34/HIGH-1 (F-3b2-1-001)。

    ``evaluate_capacity_condition`` と ``run_threshold_comparison`` (しきい値法
    比較) は、``CapacityCondition`` を組んでから ``simulate_reference_trajectory``
    への9引数呼び出しをバイト一致で複製していた (F-3b2-1-001/M1)。かつ
    ``_validate_condition_bounds`` を呼ぶのは前者だけで、後者は
    ``CapacityCondition`` を組み立てながら上限検査を1回も呼ばずに素通り
    していた (F-3b2-1-001/HIGH-1、3b-2 reviewer-security の実測)。ここへ
    一本化することで、``CapacityCondition`` を組み立てる経路が増えても検査が
    自動的に付いてくる (3-C はここを通らないので別途 ``tasks/narma.py`` の
    ``_validate`` で塞ぐ)。

    Args:
        config: 03 の設定。
        condition: 回す1条件。
        drive_offset: 駆動入力の平均のずれ (D-116)。既定 0.0 = ゼロ対称で、
            3-S (``experiment/symmetry.py``) だけがここを振る。

    Returns:
        参照軌道 (``states`` / ``drive`` を持つ)。

    Raises:
        ValueError: ``n_units`` / ``n_units * n_steps`` が上限を超える
            (確保より前に検査する) / 駆動信号の分布が未対応の場合。
    """
    _validate_condition_bounds(condition)
    return simulate_reference_trajectory(
        reservoir_config_for(config, condition),
        drive_config_for(config, condition),
        reservoir_seed=config.seeds.reservoir,
        drive_seed=config.seeds.drive,
        rho=condition.rho,
        leak_rate=condition.leak_rate,
        sigma_u=condition.sigma_u,
        replicate=condition.replicate,
        state_noise=condition.state_noise,
        drive_offset=drive_offset,
    )


def capacity_context(config: Capacity03Config) -> DiagnosticContext:
    """全条件・両診断で共有する ``DiagnosticContext`` を1個作る (D-37 の実体)。

    ``washout`` は ``config.drive.washout``、``seed`` は
    ``config.seeds.surrogate`` (サロゲートのしきい値の共有シード、D-37)。
    以前は ``evaluate_capacity_condition`` / ``run_threshold_comparison`` /
    ``run_narma10`` の3か所が同じ2行 (``DiagnosticContext(washout=..., seed=...)``)
    を複製していた (F-3b2-1-001/M2)。3-C のように ``CapacityCondition`` を
    経由しない経路でも、この1関数を呼べば D-37 の共有規律から外れない。
    """
    return DiagnosticContext(washout=config.drive.washout, seed=config.seeds.surrogate)


def measure_capacity(
    states: FloatArray,
    u: FloatArray,
    *,
    ctx: DiagnosticContext,
    mc_cfg: MemoryCapacityConfig,
    ipc_cfg: IpcConfig,
) -> CapacityMeasurement:
    """**どこで作られた ``X`` でも**同じ規律で MC と IPC を測る (D-35 / D-37)。

    ``evaluate_capacity_condition`` から切り出した「測る」部分であり、
    **D-35 (read-only 化) と D-37 (``ctx`` の共有) の実体はここにある**。

    1. ``X`` を読み取り専用にしてから診断へ渡す (D-35)。呼び出し側であるここで
       塞ぐのは、``CapacityProblem`` が自分のビューしか塞げず**元の ``X`` は
       塞げない**ため。
    2. 同じ ``X`` と同じ ``u`` で ``memory_capacity`` と ``ipc`` を呼ぶ
       (D-26 / 仕様 §5 の禁止構造「条件ごとに X を2回作る」を避ける)。
    3. ``ctx`` は**呼び出し側が作った1個**をそのまま両診断へ渡す (D-37)。
       ``t0`` の違いは各診断が ``max(ctx.washout, 自分の最大遅延)`` として
       決める (D-24)。

    Args:
        states: 状態行列 ``(n_steps, n_units)``。**この関数が読み取り専用に
            する** (呼び出し側で塞いでおく必要はない)。
        u: 駆動入力 ``(n_steps,)``。``states`` と同じ走行のものであること。
        ctx: 両診断で共有する ``DiagnosticContext`` (D-37)。
        mc_cfg: 線形メモリ容量の測定条件。
        ipc_cfg: IPC の測定条件 (3-B' は ``ipc_config_for`` が上書き済み)。

    Returns:
        行の組み立てに必要な素材 (``capacity_row_from`` へ渡す)。

    Raises:
        ValueError: 系列が短すぎる / ``ctx.seed`` が要るのに無い / 設定が
            範囲外の場合 (診断層が投げる)。
    """
    # D-35: 診断へ渡す前にここで塞ぐ。CapacityProblem は自分が持つビューしか
    # 読み取り専用にできず、元の X への書き込みは黙って gram と desync する。
    states.flags.writeable = False

    mc_started = time.perf_counter()
    mc = memory_capacity(states, u, ctx=ctx, cfg=mc_cfg)
    wall_time_mc_s = time.perf_counter() - mc_started

    ipc_started = time.perf_counter()
    ipc_result = ipc(states, u, ctx=ctx, cfg=ipc_cfg)
    wall_time_ipc_s = time.perf_counter() - ipc_started

    n_degrees = int(ipc_result.arrays["ipc_by_degree"].shape[0])
    return CapacityMeasurement(
        mc=mc,
        ipc=ipc_result,
        ipc_thresholds=tuple(
            ipc_result.scalars[f"ipc_threshold_degree{degree}"]
            for degree in range(1, n_degrees + 1)
        ),
        input_drive_std=float(np.std(u)),
        wall_time_mc_s=wall_time_mc_s,
        wall_time_ipc_s=wall_time_ipc_s,
    )


def evaluate_capacity_condition(
    config: Capacity03Config, condition: CapacityCondition
) -> CapacityOutcome:
    """1条件を回して MC と IPC の**両方**を取る (**軌道生成 + 2段の薄い層**)。

    手順は4つで、順序そのものが設計判断である。

    1. ``simulate_condition_trajectory`` が上限検査 (``_validate_condition_bounds``、
       D-34/HIGH-1) をかけてから ``X`` を**1条件につき1回だけ**作る
       (仕様 §5 の禁止構造「条件ごとに X を2回作る」を避ける)。参照軌道の
       生成は 02 から切り出し済みの関数をそのまま呼び、03 側で書き直さない。
    2. ``capacity_context`` が ``ctx`` を1個だけ作る (D-37: サロゲートのシードは
       全条件で共通)。``washout`` も同じ値を使い、``t0`` の違いは各診断が
       ``max(washout, 自分の最大遅延)`` として決める (D-24)。
    3. ``measure_capacity`` が ``X`` を読み取り専用にして (D-35) 2診断を呼ぶ。
    4. ``capacity_row_from`` が行を組む。

    3 と 4 を別関数にしてあるのは、条件が ``CapacityCondition`` で表現できない
    実験 (3-C は 01 の ``run_task`` が作った状態を測る) が、行の組み立て
    (約35フィールド) を複製せずに合流できるようにするためである
    (F-3b1-1-004)。この関数自体はその2つと軌道生成を繋ぐ薄い層である。

    Args:
        config: 03 の設定。
        condition: 回す1条件。

    Returns:
        ``capacity.csv`` の1行と、図が使う3本の配列。

    Raises:
        ValueError: ``n_units`` / ``n_units * n_steps`` が上限を超える
            (F-3b1-1-017、確保より前に検査する) / 駆動信号の分布が未対応 /
            設定が範囲外 / 系列が短すぎる / ``ctx.seed`` が要るのに無い場合
            (後半は診断層・ESN 層が投げる)。
    """
    started = time.perf_counter()
    reference = simulate_condition_trajectory(config, condition)
    wall_time_state_s = time.perf_counter() - started

    ctx = capacity_context(config)
    measurement = measure_capacity(
        reference.states,
        reference.drive,
        ctx=ctx,
        mc_cfg=config.mc,
        ipc_cfg=ipc_config_for(config, condition.experiment),
    )
    row = capacity_row_from(
        measurement,
        identity_for(condition, config, seed_drive=config.seeds.drive),
        CapacityRowTiming(
            wall_time_state_s=wall_time_state_s,
            wall_time_s=time.perf_counter() - started,
        ),
    )
    logger.debug(
        "experiment=%s rep=%d rho=%.3f leak=%.2f N=%d noise=%.4f "
        "mc_total=%.3f ipc_total=%.3f (state %.3fs / mc %.3fs / ipc %.3fs)",
        row.experiment,
        row.replicate,
        row.rho,
        row.leak_rate,
        row.n_units,
        row.state_noise,
        row.mc_total,
        row.ipc_total,
        row.wall_time_state_s,
        row.wall_time_mc_s,
        row.wall_time_ipc_s,
    )
    return capacity_outcome_from(measurement, row)


def n_replicates_for(config: Capacity03Config, experiment: str) -> int:
    """実験ラベルに応じたレプリケート数を返す (**上書きは片方向**)。

    3-B' (``conservation``) だけが ``conservation.n_replicates`` で横断共有の
    ``reservoir.n_replicates`` を上書きできる。``None`` (既定) なら継承する。
    ``ipc_config_for`` と同じ形にしてあるのは、3-B' が予算 400 秒と3実験で
    最も重く、仕様 §7 リスク1 の縮退規則 (「合計見積りが 700 秒を超えた場合に
    許可される調整は ``conservation.n_replicates`` を 3 → 1 に落とすことだけ」)
    のノブがそこだけに効く必要があるためである。横断共有を 1 に落とすと
    3-A / 3-B の平均 +- s.d. まで消え、縮退の意味が変わる。

    Raises:
        ValueError: レプリケート数が 1 未満の場合。
    """
    n_replicates = config.reservoir.n_replicates
    if (
        experiment == EXPERIMENT_CONSERVATION
        and config.conservation.n_replicates is not None
    ):
        n_replicates = config.conservation.n_replicates
    if n_replicates < 1:
        raise ValueError(f"レプリケート数は 1 以上である必要があります: {n_replicates}")
    return n_replicates


def _sweep(
    config: Capacity03Config,
    experiment: str,
    axes: tuple[CapacityCondition, ...],
) -> tuple[CapacityOutcome, ...]:
    """レプリケート番号だけが違う条件を並べて回す共通ループ (02 の ``_sweep``)。

    ``axes`` は ``replicate=0`` の条件の並びで、ここがレプリケートぶん複製する。
    レプリケートを外側にするのは 02 と同じ並び順にするためで、CSV の行順が
    実験間で揃う。
    """
    started = time.perf_counter()
    outcomes = tuple(
        evaluate_capacity_condition(
            config, dataclasses.replace(condition, replicate=replicate)
        )
        for replicate in range(n_replicates_for(config, experiment))
        for condition in axes
    )
    logger.info(
        "experiment=%s 条件数=%d (軸=%d x レプリケート=%d) wall_time=%.2fs "
        "(状態生成 %.2fs / MC %.2fs / IPC %.2fs)",
        experiment,
        len(outcomes),
        len(axes),
        n_replicates_for(config, experiment),
        time.perf_counter() - started,
        sum(outcome.row.wall_time_state_s for outcome in outcomes),
        sum(outcome.row.wall_time_mc_s for outcome in outcomes),
        sum(outcome.row.wall_time_ipc_s for outcome in outcomes),
    )
    return outcomes


def run_mc_sweep(config: Capacity03Config) -> tuple[CapacityOutcome, ...]:
    """実験 3-A: rho x リーク率 に対する線形メモリ容量 (受け入れ条件1)。

    N は ``mc_sweep.n_units`` (既定 200) で、上限線 y=N を引ける規模にする
    (D-32)。IPC も同じ ``X`` から測る (仕様 §8: 全条件で MC と IPC の両方)。
    """
    section = config.mc_sweep
    return _sweep(
        config,
        EXPERIMENT_MC_SWEEP,
        tuple(
            CapacityCondition(
                experiment=EXPERIMENT_MC_SWEEP,
                rho=rho,
                leak_rate=leak_rate,
                n_units=section.n_units,
                state_noise=0.0,
                sigma_u=section.sigma_u,
                n_steps=section.n_steps,
                replicate=0,
            )
            for rho in section.rho_grid
            for leak_rate in section.leak_rate_grid
        ),
    )


def run_ipc_sweep(config: Capacity03Config) -> tuple[CapacityOutcome, ...]:
    """実験 3-B: rho x リーク率 に対する IPC の次数・遅延分解 (受け入れ条件4)。

    N は ``ipc_sweep.n_units`` (既定 50) で 3-A より小さい (D-32)。
    """
    section = config.ipc_sweep
    return _sweep(
        config,
        EXPERIMENT_IPC_SWEEP,
        tuple(
            CapacityCondition(
                experiment=EXPERIMENT_IPC_SWEEP,
                rho=rho,
                leak_rate=leak_rate,
                n_units=section.n_units,
                state_noise=0.0,
                sigma_u=section.sigma_u,
                n_steps=section.n_steps,
                replicate=0,
            )
            for rho in section.rho_grid
            for leak_rate in section.leak_rate_grid
        ),
    )


def run_conservation_sweep(config: Capacity03Config) -> tuple[CapacityOutcome, ...]:
    """実験 3-B': N x 状態ノイズ に対する保存則 IPC_total <= N (受け入れ条件2)。

    ``n_units`` が掃引軸そのものであり、IPC の遅延打ち切りだけが
    ``conservation.max_delay_by_degree`` で深くなる (``ipc_config_for``、片方向)。
    レプリケート数もこのセクションだけ上書きできる (``n_replicates_for``)。
    """
    section = config.conservation
    return _sweep(
        config,
        EXPERIMENT_CONSERVATION,
        tuple(
            CapacityCondition(
                experiment=EXPERIMENT_CONSERVATION,
                rho=section.rho,
                leak_rate=section.leak_rate,
                n_units=n_units,
                state_noise=state_noise,
                sigma_u=section.sigma_u,
                n_steps=section.n_steps,
                replicate=0,
            )
            for n_units in section.n_units_grid
            for state_noise in section.state_noise_grid
        ),
    )


def run_length_sweep(config: Capacity03Config) -> tuple[CapacityOutcome, ...]:
    """系列長 T の掃引 (``make saturation-03``)。**本番の figures-03 に含めない**。

    「容量が足りないのか T が足りないのか」を分けるための補助実験であり、
    T=1e6 まで回すので単独で 900 秒予算を食い潰す。成果物も
    ``capacity_length.csv`` に分ける (仕様 §8)。
    """
    section = config.length_sweep
    return _sweep(
        config,
        EXPERIMENT_LENGTH_SWEEP,
        tuple(
            CapacityCondition(
                experiment=EXPERIMENT_LENGTH_SWEEP,
                rho=section.rho,
                leak_rate=section.leak_rate,
                n_units=section.n_units,
                state_noise=0.0,
                sigma_u=section.sigma_u,
                n_steps=n_steps,
                replicate=0,
            )
            for n_steps in section.n_steps_grid
        ),
    )


@dataclass(frozen=True, slots=True)
class CapacityResults:
    """``figures-03`` が回す3実験ぶんの結果 (図はそれぞれ別の実験だけを見る)。

    02 の ``EspResults`` と同型。``length_sweep`` は本番に含めない (仕様 §8)
    ので**ここには入れない** —— 入れると ``make figures-03`` の予算 900 秒に
    T=1e6 の掃引が紛れ込む。
    """

    mc_sweep: tuple[CapacityOutcome, ...]
    ipc_sweep: tuple[CapacityOutcome, ...]
    conservation: tuple[CapacityOutcome, ...]

    @property
    def outcomes(self) -> tuple[CapacityOutcome, ...]:
        """3実験ぶんを宣言順につなげたもの。"""
        return self.mc_sweep + self.ipc_sweep + self.conservation

    @property
    def rows(self) -> tuple[CapacityRow, ...]:
        """``capacity.csv`` に書く行 (3実験ぶん)。"""
        return tuple(outcome.row for outcome in self.outcomes)

    @property
    def profile_rows(self) -> tuple[CapacityProfileRow, ...]:
        """``capacity_profile.csv`` に書く行 (正値セルのみ、D-38)。"""
        return tuple(row for outcome in self.outcomes for row in profile_rows(outcome))


def run_capacity_experiment(config: Capacity03Config) -> CapacityResults:
    """実験 3-A / 3-B / 3-B' をこの順に回す (``length_sweep`` は含めない)。"""
    return CapacityResults(
        mc_sweep=run_mc_sweep(config),
        ipc_sweep=run_ipc_sweep(config),
        conservation=run_conservation_sweep(config),
    )


__all__ = [
    "CAPACITY_EXPERIMENTS",
    "DIAGNOSTIC_IPC",
    "DIAGNOSTIC_MC",
    "EXPERIMENT_CONSERVATION",
    "EXPERIMENT_IPC_SWEEP",
    "EXPERIMENT_LENGTH_SWEEP",
    "EXPERIMENT_MC_SWEEP",
    "EXPERIMENT_NARMA10",
    "FIGURE_EXPERIMENTS",
    "CapacityOutcome",
    "CapacityProfileRow",
    "CapacityResults",
    "CapacityRow",
    "drive_config_for",
    "evaluate_capacity_condition",
    "ipc_config_for",
    "n_replicates_for",
    "profile_rows",
    "reservoir_config_for",
    "run_capacity_experiment",
    "run_conservation_sweep",
    "run_ipc_sweep",
    "run_length_sweep",
    "run_mc_sweep",
    "validate_n_units_bound",
    "validate_sequential_run_count",
    "validate_state_matrix_bounds",
    "validate_total_step_count",
]

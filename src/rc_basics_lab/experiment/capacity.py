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

**``X`` は診断へ渡す前に読み取り専用にする** (D-35)。``CapacityProblem`` は
``X`` の**ビュー**を持ち ``gram`` は構築時点の**スナップショット**なので、
構築後に ``X`` を書き換えると両者が例外も警告もなく desync する
(3a の実測では容量が 1.25e8 という桁違いの値になった)。``CapacityProblem``
は自分が持つビューを読み取り専用にするが**元の ``X`` は塞げない**ため、
呼び出し側であるここで塞ぐ。診断側でコピーすると T=1e6 で 1.6GB 増えて
4GB 予算を壊す (F-03-1-013 で潰したのと同じ失敗)。

なお MC と IPC は ``t0`` が異なる (MC は ``max(washout, mc.max_delay)``、
IPC は ``max(washout, max(ipc.max_delay_by_degree))``) ため、
``CapacityProblem`` は1条件あたり2個作られる。**これは正常であり**、
「1個にまとめる」最適化をすると D-24 の単一基準点がどちらかの診断で壊れる。

**サロゲートのシードは全条件で1個を共有する** (D-37、共通乱数法)。
条件ごとに振ると、条件間の容量差にしきい値の推定ノイズが独立に乗る。
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, fields

import numpy as np

from rc_basics_lab.config import (
    Capacity03Config,
    DriveConfig,
    IpcConfig,
    ReservoirSweepConfig,
)
from rc_basics_lab.diagnostics.base import DiagnosticContext
from rc_basics_lab.diagnostics.ipc import ipc
from rc_basics_lab.diagnostics.memory_capacity import memory_capacity
from rc_basics_lab.experiment.esp import simulate_reference_trajectory
from rc_basics_lab.types import FloatArray

logger = logging.getLogger(__name__)

EXPERIMENT_MC_SWEEP = "3A_mc_sweep"
"""実験 3-A: rho x リーク率 に対する線形メモリ容量 (受け入れ条件1)。"""

EXPERIMENT_IPC_SWEEP = "3B_ipc_sweep"
"""実験 3-B: rho x リーク率 に対する IPC の次数・遅延分解 (受け入れ条件4)。"""

EXPERIMENT_CONSERVATION = "3Bp_conservation"
"""実験 3-B': ノイズ下での保存則 IPC_total <= N (受け入れ条件2)。"""

EXPERIMENT_LENGTH_SWEEP = "3L_length_sweep"
"""系列長 T に対する容量の飽和 (``make saturation-03``)。**本番には含めない**。

``figures-03`` の成果物 (``capacity.csv``) には入らず ``capacity_length.csv``
に別途書く (仕様 §8: 「本番 (figures-03) には含めない」)。それでも実験ラベルを
名乗るのは、``length_sweep.*`` の設定が**他の実験の行を動かしていない**ことを
scope 検査 (``tests/test_config_wiring_capacity.py``) で測るためである。
ラベルを共有すると、T 掃引の設定を変えたときに 3-A の行まで動いても
気づけない。
"""

CAPACITY_EXPERIMENTS: tuple[str, ...] = (
    EXPERIMENT_MC_SWEEP,
    EXPERIMENT_IPC_SWEEP,
    EXPERIMENT_CONSERVATION,
    EXPERIMENT_LENGTH_SWEEP,
)
"""``CapacityRow.experiment`` が取りうる値 (3-C は 3b-2 の T4 が足す)。

``capacity.csv`` に出るのは先頭3つで、``3L_length_sweep`` だけは
``capacity_length.csv`` (``make saturation-03``) 側にしか現れない。
"""

FIGURE_EXPERIMENTS: tuple[str, ...] = (
    EXPERIMENT_MC_SWEEP,
    EXPERIMENT_IPC_SWEEP,
    EXPERIMENT_CONSERVATION,
)
"""``make figures-03`` (``capacity.csv``) が回す実験。予算 900 秒の対象。"""

DIAGNOSTIC_MC = "mc"
"""``CapacityProfileRow.diagnostic``: 線形メモリ容量 (次数は常に1)。"""

DIAGNOSTIC_IPC = "ipc"
"""``CapacityProfileRow.diagnostic``: 情報処理容量 (次数 x 遅延)。"""


@dataclass(frozen=True, slots=True)
class CapacityCondition:
    """容量測定の1条件。掃引の違いはどの軸を振るかだけである。

    02 の ``evaluate_condition`` はキーワード引数で軸を受けていたが、03 は軸が
    8本 (実験ラベル・rho・リーク率・N・状態ノイズ・駆動強度・系列長・
    レプリケート) あり、``n_units`` と ``n_steps`` がセクションごとに違う
    (D-32) ため、条件そのものを1つの値として持ち回る。

    Attributes:
        experiment: ``CAPACITY_EXPERIMENTS`` のいずれか。CSV の ``experiment``
            列になり、3-B' だけ IPC の打ち切りが上書きされる (下記
            ``ipc_config_for``)。
        rho: スペクトル半径。
        leak_rate: リーク率。
        n_units: リザバーのユニット数 N (**セクションが持つ**、D-32)。
        state_noise: tanh 内部に加えるガウスノイズの標準偏差 (D-36)。
        sigma_u: 駆動信号の標準偏差 (D-17)。
        n_steps: 系列長 [ステップ]。
        replicate: レプリケート番号 (0 始まり)。
    """

    experiment: str
    rho: float
    leak_rate: float
    n_units: int
    state_noise: float
    sigma_u: float
    n_steps: int
    replicate: int


@dataclass(frozen=True, slots=True)
class CapacityRow:
    """``capacity.csv`` の1行。**宣言順が CSV の列順の単一の真実**。

    1行 = 1条件で、**全列が常に埋まる** (cfg 依存で本数が変わる
    ``ipc_threshold_degree{d}`` は列にせず、T2 が長形式の
    ``capacity_profile.csv`` に落とす、D-38)。

    ``input_scale`` / ``density`` は ``Capacity03Config.reservoir`` 由来の
    横断共有値、``n_units`` はセクション由来 (D-32)。``washout`` は
    ``DiagnosticContext.washout`` として MC / IPC の ``t0`` に効く値であり、
    実際に使われた基準点は ``t0_mc`` / ``t0_ipc`` に別途出す (D-24)。
    """

    experiment: str
    replicate: int
    seed_reservoir: int
    seed_drive: int
    seed_surrogate: int
    rho: float
    leak_rate: float
    input_scale: float
    sigma_u: float
    input_drive_std: float
    n_units: int
    density: float
    state_noise: float
    n_steps: int
    washout: int
    t0_mc: int
    n_samples_mc: int
    mc_total: float
    mc_total_raw: float
    mc_threshold: float
    mc_effective_delay: float
    mc_ratio: float
    n_delays: int
    t0_ipc: int
    n_samples_ipc: int
    ipc_total: float
    ipc_total_raw: float
    ipc_linear: float
    ipc_nonlinear: float
    ipc_saturation_ratio: float
    n_targets: int
    n_targets_kept: int
    n_degrees: int
    chunk_size_mc_effective: int
    chunk_size_ipc_effective: int
    wall_time_state_s: float
    wall_time_mc_s: float
    wall_time_ipc_s: float
    wall_time_s: float


CAPACITY_CSV_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(CapacityRow))
"""``capacity.csv`` の列順 (``CapacityRow`` の宣言順が単一の真実)。"""


@dataclass(frozen=True, slots=True)
class CapacityOutcome:
    """1条件ぶんの結果。行に加えて図が必要とする配列を持つ。

    02 の ``ConditionOutcome`` と同型である。行だけ返すと、図 (3枚が配列を
    直接使う) のために全条件をもう一度回すことになる。``row`` が
    ``CapacityRow`` である以上、CSV 列順の単一の真実は変わらない。

    Attributes:
        row: ``capacity.csv`` の1行。
        mc_profile: しきい値後の遅延プロファイル ``(mc.max_delay,)``
            (``fig_mc_sweep.png`` の右パネル)。
        ipc_heatmap: (次数, 遅延) のしきい値後の容量 (``fig_ipc_profile.png``)。
        ipc_by_degree: 次数ごとのしきい値後の容量
            (``fig_memory_nonlinearity.png``)。
    """

    row: CapacityRow
    mc_profile: FloatArray
    ipc_heatmap: FloatArray
    ipc_by_degree: FloatArray


def reservoir_config_for(
    config: Capacity03Config, condition: CapacityCondition
) -> ReservoirSweepConfig:
    """1条件ぶんのリザバー構造を組む (**値の組み立てはここ1か所**)。

    03 は 02 の ``ReservoirSweepConfig`` を**設定として**再利用しない (仕様
    §3.2) が、``simulate_reference_trajectory`` の引数型としては再利用する。
    横断共有の3つ (``input_scale`` / ``density`` / ``n_replicates``) は
    ``config.reservoir`` から、``n_units`` は条件から取る (D-32)。
    """
    return ReservoirSweepConfig(
        input_scale=config.reservoir.input_scale,
        n_units=condition.n_units,
        density=config.reservoir.density,
        n_replicates=config.reservoir.n_replicates,
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


def evaluate_capacity_condition(
    config: Capacity03Config, condition: CapacityCondition
) -> CapacityOutcome:
    """1条件を回して MC と IPC の**両方**を取る。

    手順は4つで、順序そのものが設計判断である。

    1. ``simulate_reference_trajectory`` で ``X`` を**1条件につき1回だけ**作る
       (仕様 §5 の禁止構造「条件ごとに X を2回作る」を避ける)。参照軌道の
       生成は 02 から切り出し済みの関数をそのまま呼び、03 側で書き直さない。
    2. ``X`` を読み取り専用にしてから診断へ渡す (D-35)。
    3. 同じ ``X`` と ``u`` で ``memory_capacity`` と ``ipc`` を呼ぶ。
    4. ``ctx`` は1個を両診断で共有する (D-37: サロゲートのシードは全条件で
       共通)。``washout`` も同じ値を使い、``t0`` の違いは各診断が
       ``max(washout, 自分の最大遅延)`` として決める (D-24)。

    Args:
        config: 03 の設定。
        condition: 回す1条件。

    Returns:
        ``capacity.csv`` の1行と、図が使う3本の配列。

    Raises:
        ValueError: 駆動信号の分布が未対応 / 設定が範囲外 / 系列が短すぎる /
            ``ctx.seed`` が要るのに無い場合 (いずれも診断層・ESN 層が投げる)。
    """
    started = time.perf_counter()
    reference = simulate_reference_trajectory(
        reservoir_config_for(config, condition),
        drive_config_for(config, condition),
        reservoir_seed=config.seeds.reservoir,
        drive_seed=config.seeds.drive,
        rho=condition.rho,
        leak_rate=condition.leak_rate,
        sigma_u=condition.sigma_u,
        replicate=condition.replicate,
        state_noise=condition.state_noise,
    )
    wall_time_state_s = time.perf_counter() - started

    states = reference.states
    # D-35: 診断へ渡す前にここで塞ぐ。CapacityProblem は自分が持つビューしか
    # 読み取り専用にできず、元の X への書き込みは黙って gram と desync する。
    states.flags.writeable = False
    u = reference.drive
    ctx = DiagnosticContext(washout=config.drive.washout, seed=config.seeds.surrogate)

    mc_started = time.perf_counter()
    mc = memory_capacity(states, u, ctx=ctx, cfg=config.mc)
    wall_time_mc_s = time.perf_counter() - mc_started

    ipc_started = time.perf_counter()
    ipc_result = ipc(
        states, u, ctx=ctx, cfg=ipc_config_for(config, condition.experiment)
    )
    wall_time_ipc_s = time.perf_counter() - ipc_started

    ipc_by_degree = ipc_result.arrays["ipc_by_degree"]
    row = CapacityRow(
        experiment=condition.experiment,
        replicate=condition.replicate,
        seed_reservoir=config.seeds.reservoir,
        seed_drive=config.seeds.drive,
        seed_surrogate=config.seeds.surrogate,
        rho=condition.rho,
        leak_rate=condition.leak_rate,
        input_scale=config.reservoir.input_scale,
        sigma_u=condition.sigma_u,
        input_drive_std=float(np.std(u)),
        n_units=condition.n_units,
        density=config.reservoir.density,
        state_noise=condition.state_noise,
        n_steps=condition.n_steps,
        washout=config.drive.washout,
        t0_mc=int(mc.params["t0"]),
        n_samples_mc=int(mc.params["n_samples"]),
        mc_total=mc.scalars["mc_total"],
        mc_total_raw=mc.scalars["mc_total_raw"],
        mc_threshold=mc.scalars["mc_threshold"],
        mc_effective_delay=mc.scalars["mc_effective_delay"],
        mc_ratio=mc.scalars["mc_ratio"],
        n_delays=int(mc.scalars["n_delays"]),
        t0_ipc=int(ipc_result.params["t0"]),
        n_samples_ipc=int(ipc_result.params["n_samples"]),
        ipc_total=ipc_result.scalars["ipc_total"],
        ipc_total_raw=ipc_result.scalars["ipc_total_raw"],
        ipc_linear=ipc_result.scalars["ipc_linear"],
        ipc_nonlinear=ipc_result.scalars["ipc_nonlinear"],
        ipc_saturation_ratio=ipc_result.scalars["saturation_ratio"],
        n_targets=int(ipc_result.scalars["n_targets"]),
        n_targets_kept=int(ipc_result.scalars["n_targets_kept"]),
        n_degrees=int(ipc_by_degree.shape[0]),
        chunk_size_mc_effective=int(mc.params["chunk_size_effective"]),
        chunk_size_ipc_effective=int(ipc_result.params["chunk_size_effective"]),
        wall_time_state_s=wall_time_state_s,
        wall_time_mc_s=wall_time_mc_s,
        wall_time_ipc_s=wall_time_ipc_s,
        wall_time_s=time.perf_counter() - started,
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
    return CapacityOutcome(
        row=row,
        mc_profile=mc.arrays["mc_profile"],
        ipc_heatmap=ipc_result.arrays["ipc_heatmap"],
        ipc_by_degree=ipc_by_degree,
    )


__all__ = [
    "CAPACITY_CSV_COLUMNS",
    "CAPACITY_EXPERIMENTS",
    "EXPERIMENT_CONSERVATION",
    "EXPERIMENT_IPC_SWEEP",
    "EXPERIMENT_MC_SWEEP",
    "CapacityCondition",
    "CapacityOutcome",
    "CapacityRow",
    "drive_config_for",
    "evaluate_capacity_condition",
    "ipc_config_for",
    "reservoir_config_for",
]

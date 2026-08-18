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
from dataclasses import dataclass, fields

import numpy as np

from rc_basics_lab.config import (
    Capacity03Config,
    DriveConfig,
    IpcConfig,
    MemoryCapacityConfig,
    ReservoirSweepConfig,
)
from rc_basics_lab.diagnostics.base import DiagnosticContext, DiagnosticResult
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

_MAX_UNITS = 5_000
"""``CapacityCondition.n_units`` の上書き不能な絶対上限 (F-3b1-1-017, CWE-789)。

``ESN`` の重み生成は ``rng.random((N, N))`` (再帰行列) を確保するため、確保量は
``N**2`` に比例する。3a の D-34 (IPC の確保・組合せ計算量の4段の上限) と同じ
threat model —— 設定 YAML の1行変更 (``conservation.n_units_grid: [100000]``)
だけで防御が無い状態だと数十GB の確保に到達しうる (実測: N=100000 で重み行列
だけで約80GB)。本番設定の最大 ``n_units`` は 200 (3-A) で、``_MAX_UNITS=5000``
は25倍の余裕を残しつつ、重み行列を ``8 * 5000**2`` ≈ 200MB に抑える。
"""

_MAX_STATE_ELEMENTS = 200_000_000
"""``n_units * n_steps`` の上書き不能な絶対上限 (F-3b1-1-017, CWE-400/789)。

状態行列 ``X`` は ``(n_steps, n_units)`` の ``float64`` を確保するため、
確保量は ``n_units * n_steps`` に比例する (D-35 の rationale が言う 4GB 予算と
同じ軸)。本番設定の最大は length_sweep (``n_units=50, n_steps=1_000_000`` =
5e7) で、``_MAX_STATE_ELEMENTS=2e8`` は4倍の余裕を残しつつ状態行列を
``8 * 2e8`` = 1.6GB に抑える。``n_steps`` 単体ではなく積で縛るのは、
``n_units`` が小さければ ``n_steps`` を大きく取れる (length_sweep の実際の
使い方) 一方で、両方を同時に大きくする設定変更は個別の軸の検査をすり抜ける
ため (CWE-789 の threat model は D-34 の rationale と同型)。
"""


def _validate_condition_bounds(condition: CapacityCondition) -> None:
    """状態行列・ESN の確保より前に、確保量に上書き不能な絶対上限をかける。

    D-34 の規律 (「確保より前に落とす」) を実験層の確保軸 (``n_units`` /
    ``n_steps``) にも適用する (F-3b1-1-017)。``CapacityCondition`` はこの
    モジュールの全経路 (3-A / 3-B / 3-B' / length_sweep) が最終的に通る単一の
    入口なので、ここ1か所の検査で4経路すべてが守られる。
    """
    if condition.n_units > _MAX_UNITS:
        raise ValueError(
            f"n_units が上限を超えています: {condition.n_units} > {_MAX_UNITS} "
            "(ESN の重み行列の確保量は n_units**2 に比例するため、"
            "確保する前に検査で落とす)"
        )
    n_state_elements = condition.n_units * condition.n_steps
    if n_state_elements > _MAX_STATE_ELEMENTS:
        raise ValueError(
            f"n_units * n_steps が上限を超えています: {n_state_elements} > "
            f"{_MAX_STATE_ELEMENTS} (状態行列の確保量は n_units * n_steps に"
            "比例するため、確保する前に検査で落とす)"
        )


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
        ipc_thresholds: 次数ごとのしきい値 (``ipc_threshold_degree{d}`` を
            次数の昇順に並べたもの)。**cfg 依存で本数が変わる**ため
            ``CapacityRow`` の列にはできず (D-38)、長形式の
            ``CapacityProfileRow.threshold`` に落とす。ここに持たせるのは、
            同じ条件で ``ipc`` をもう一度走らせて取り直すことを禁じるため
            (1条件あたり数秒〜7秒の再計算になる)。

    ``ipc_result.arrays["ipc_by_degree"]`` (次数ごとのしきい値後の容量) は
    フィールドとして運ばない (F-3b1-1-002)。T3 で図を長形式へ切り替えた際
    (D-38) に取り残された前設計の残骸で、``plot_memory_nonlinearity`` は
    ``CapacityRow.ipc_linear`` / ``ipc_nonlinear`` しか読まず、
    ``profile_rows`` も ``mc_profile`` / ``ipc_heatmap`` しか使わない
    (全 outcome を ``-999`` で埋めて成果物を再生成しても capacity.csv /
    capacity_profile.csv / 図4枚は wall_time_* を除きバイト一致することを
    実測済み)。``n_degrees`` の算出には配列そのものではなく形状だけが要るので、
    ``evaluate_capacity_condition`` 内のローカル変数として使い、outcome へは
    運ばない。
    """

    row: CapacityRow
    mc_profile: FloatArray
    ipc_heatmap: FloatArray
    ipc_thresholds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CapacityProfileRow:
    """``capacity_profile.csv`` の1行 (**長形式**、D-38)。宣言順が CSV の列順。

    IPC の ``scalars`` は ``ipc_threshold_degree{d}`` を**次数の本数だけ**持つ
    ため、キー集合が ``cfg.max_delay_by_degree`` に依存する (F-03-1-005)。
    これを列にすると 3-B' (次数4) と 3-A/3-B (次数4) で列数が揃っていても、
    打ち切りを1本増やした瞬間に CSV の列が変わる —— 「列は cfg に依らず一定」
    という単一の真実が壊れる。次数と遅延を**行の値**に落とせば列は静的なままで、
    しきい値も次数ごとの行に自然に乗る。

    MC は次数の概念を持たないので ``degree=1`` に固定する (次数1の線形容量で
    あることは IPC の次数1と同じ意味であり、``diagnostic`` 列で区別する)。

    **書くのはしきい値後の容量が厳密に正のセルだけ**である。全セルを書くと
    本番設定で約6万行になり、``results/`` はコミット対象なのでリポジトリが
    その分だけ重くなる。正値だけに絞る条件 (``capacity > 0``) は IPC の
    ``n_targets_kept`` (``np.count_nonzero(kept)``) が「しきい値を超えた」を
    判定する条件と同じ ``> 0`` だが、**行数が n_targets_kept と一致するとは
    限らない** (F-3b1-1-003)。長形式の行は (次数, 遅延) の**ヒートマップセル
    単位**、``n_targets_kept`` は**目標単位**で単位が異なり、1セルに複数目標
    が畳み込まれるため行数は ``n_targets_kept`` 以下にしかならない
    (本番成果物の実測: 117行すべてで行数 != n_targets_kept、例: 81セル vs
    297目標)。行数の代わりに成果物単体で検算できる不変条件は
    ``capacity`` 列の**総和**が ``ipc_total`` / ``mc_total`` と一致すること
    であり、これは本番成果物117行すべてで成立する
    (``tests/test_capacity_pipeline.py::test_profile_csv_columns_are_static_and_cells_are_positive``
    が両方 (行数の上限・総和の一致) を実測で固定する)。

    Attributes:
        experiment: ``CAPACITY_EXPERIMENTS`` のいずれか。
        replicate: レプリケート番号 (0 始まり)。
        rho: スペクトル半径。
        leak_rate: リーク率。
        n_units: リザバーのユニット数 N。
        state_noise: 状態ノイズの標準偏差。
        diagnostic: ``"mc"`` か ``"ipc"`` (``DIAGNOSTIC_MC`` / ``DIAGNOSTIC_IPC``)。
        degree: 次数 (MC は常に1)。
        delay: 遅延 [ステップ] (1 始まり)。
        capacity: しきい値後の容量 (**厳密に正**)。
        threshold: その次数のしきい値 (MC は ``mc_threshold``)。
    """

    experiment: str
    replicate: int
    rho: float
    leak_rate: float
    n_units: int
    state_noise: float
    diagnostic: str
    degree: int
    delay: int
    capacity: float
    threshold: float


CAPACITY_PROFILE_CSV_COLUMNS: tuple[str, ...] = tuple(
    f.name for f in fields(CapacityProfileRow)
)
"""``capacity_profile.csv`` の列順 (``CapacityProfileRow`` の宣言順が単一の真実)。

**cfg に依らず一定**であることが D-38 の中心で、
``tests/test_capacity_pipeline.py::test_profile_csv_columns_are_static_and_cells_are_positive``
が2つの異なる打ち切り設定で実測する。
"""


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


@dataclass(frozen=True, slots=True)
class CapacityMeasurement:
    """**外部で作られた** ``X`` に対する MC / IPC の測定結果 (行にする前の素材)。

    ``measure_capacity`` の返り値であり、``capacity_row_from`` の入力である。
    ``CapacityOutcome`` (行 + 図の配列) との違いは、**まだ行になっていない**
    ことで、条件の識別子 (実験ラベル・rho・リーク率・…) を1つも持たない。
    行にするために要る値のうち「診断の結果から決まるもの」だけをここに集め、
    「どういう条件で測ったか」は ``capacity_row_from`` のキーワード引数として
    外から与える —— この分け方があるので、``CapacityCondition`` で表現できない
    実験 (3-C は 01 の ``run_task`` が作った状態を測る) でも
    ``CapacityRow`` の約35フィールドを複製せずに行が作れる (F-3b1-1-004)。

    Attributes:
        mc: ``memory_capacity`` の結果。
        ipc: ``ipc`` の結果。
        ipc_thresholds: 次数ごとのしきい値 (次数の昇順)。``ipc.scalars`` の
            ``ipc_threshold_degree{d}`` は**本数が cfg 依存**なので (D-38)、
            ここで一度だけ昇順のタプルに畳んでおく。``n_degrees`` はこの
            タプルの長さであり、``ipc_by_degree`` 配列そのものは運ばない
            (F-3b1-1-002)。
        input_drive_std: 駆動入力の実測標準偏差 (``CapacityRow.input_drive_std``)。
            設定値 ``sigma_u`` と区別するため、実際に診断が見た ``u`` から測る。
        wall_time_mc_s: MC の実行時間 [秒]。
        wall_time_ipc_s: IPC の実行時間 [秒]。
    """

    mc: DiagnosticResult
    ipc: DiagnosticResult
    ipc_thresholds: tuple[float, ...]
    input_drive_std: float
    wall_time_mc_s: float
    wall_time_ipc_s: float


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

    1. ``X`` を読み取り専用にしてから診断へ渡す (D-35)。``CapacityProblem`` は
       ``X`` の**ビュー**を持ち ``gram`` は構築時点のスナップショットなので、
       構築後に ``X`` を書き換えると両者が例外も警告もなく desync する。
       ``CapacityProblem`` は自分のビューしか塞げず**元の ``X`` は塞げない**
       ため、呼び出し側であるここで塞ぐ。診断側でコピーすると T=1e6 で 1.6GB
       増えて 4GB 予算を壊す (F-03-1-012/013)。
    2. 同じ ``X`` と同じ ``u`` で ``memory_capacity`` と ``ipc`` を呼ぶ
       (D-26 / 仕様 §5 の禁止構造「条件ごとに X を2回作る」を避ける)。
    3. ``ctx`` は**呼び出し側が作った1個**をそのまま両診断へ渡す (D-37:
       サロゲートのシードは全条件で共通)。``t0`` の違いは各診断が
       ``max(ctx.washout, 自分の最大遅延)`` として決める (D-24)。

    ``ctx`` を引数で受け取り中で作らないのは、3-C のように条件が
    ``CapacityCondition`` で表現できない経路でも「全条件で ``ctx`` は1個」
    (D-37) を呼び出し側が保てるようにするためである。

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


def capacity_row_from(
    measurement: CapacityMeasurement,
    *,
    experiment: str,
    replicate: int,
    seed_reservoir: int,
    seed_drive: int,
    seed_surrogate: int,
    rho: float,
    leak_rate: float,
    input_scale: float,
    sigma_u: float,
    n_units: int,
    density: float,
    state_noise: float,
    n_steps: int,
    washout: int,
    wall_time_state_s: float,
    wall_time_s: float,
) -> CapacityRow:
    """測定結果と条件の識別子から ``capacity.csv`` の1行を組む (**唯一の経路**)。

    ``CapacityRow`` は約35フィールドあり、``mc`` / ``ipc`` の ``scalars`` と
    ``params`` のどのキーがどの列になるかの対応もここにしかない。この組み立てを
    実験ごとに複製すると「CSV の列順の単一の真実 = 行 dataclass の宣言順」
    (§2.2-1) が実質的に破れる —— 列を1本足したときに複製側が置き去りになり、
    かつ型検査では落ちない (キーワード引数の名前は一致したままなので)。
    3-C (``experiment="3C_narma10"``) は ``CapacityCondition`` で表現できない
    条件を持つが、行の作り方はここを通す (F-3b1-1-004)。

    ``experiment`` 以降がキーワード専用なのは、位置引数で並べると隣接する
    同型の値 (``rho`` / ``leak_rate``、3本の ``seed_*``) を取り違えても
    静かに通ってしまうためである。

    Args:
        measurement: ``measure_capacity`` の返り値。
        experiment: ``CAPACITY_EXPERIMENTS`` のいずれか (CSV の ``experiment``)。
        replicate: レプリケート番号 (0 始まり)。
        seed_reservoir: リザバー重みの基底シード。
        seed_drive: 駆動入力の基底シード。
        seed_surrogate: しきい値サロゲートのシード (``ctx.seed`` と同じ値、D-37)。
        rho: スペクトル半径。
        leak_rate: リーク率。
        input_scale: 入力結合の強さ (横断共有値)。
        sigma_u: 駆動信号の標準偏差の**設定値** (実測は
            ``measurement.input_drive_std``)。
        n_units: リザバーのユニット数 N。
        density: 再帰結合の密度 (横断共有値)。
        state_noise: 状態ノイズの標準偏差。
        n_steps: 系列長 [ステップ]。
        washout: ``ctx.washout`` として渡した値 (実効基準点は ``t0_mc`` /
            ``t0_ipc`` に別途出る、D-24)。
        wall_time_state_s: 状態行列の生成にかかった時間 [秒]。
        wall_time_s: 条件1本の合計時間 [秒]。

    Returns:
        ``capacity.csv`` の1行。
    """
    mc = measurement.mc
    ipc_result = measurement.ipc
    return CapacityRow(
        experiment=experiment,
        replicate=replicate,
        seed_reservoir=seed_reservoir,
        seed_drive=seed_drive,
        seed_surrogate=seed_surrogate,
        rho=rho,
        leak_rate=leak_rate,
        input_scale=input_scale,
        sigma_u=sigma_u,
        input_drive_std=measurement.input_drive_std,
        n_units=n_units,
        density=density,
        state_noise=state_noise,
        n_steps=n_steps,
        washout=washout,
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
        n_degrees=len(measurement.ipc_thresholds),
        chunk_size_mc_effective=int(mc.params["chunk_size_effective"]),
        chunk_size_ipc_effective=int(ipc_result.params["chunk_size_effective"]),
        wall_time_state_s=wall_time_state_s,
        wall_time_mc_s=measurement.wall_time_mc_s,
        wall_time_ipc_s=measurement.wall_time_ipc_s,
        wall_time_s=wall_time_s,
    )


def capacity_outcome_from(
    measurement: CapacityMeasurement, row: CapacityRow
) -> CapacityOutcome:
    """行と測定結果から ``CapacityOutcome`` を組む (図が使う配列を積み替える)。

    ``CapacityOutcome`` は図が必要とする配列を運ぶ役割 (02 の
    ``ConditionOutcome`` と同型) を持ち、``profile_rows`` と
    ``capacity_pipeline`` の入口はこの型である。3-C も同じ型で
    ``capacity.csv`` / ``capacity_profile.csv`` に合流できるよう、
    積み替えをここ1か所に置く。
    """
    return CapacityOutcome(
        row=row,
        mc_profile=measurement.mc.arrays["mc_profile"],
        ipc_heatmap=measurement.ipc.arrays["ipc_heatmap"],
        ipc_thresholds=measurement.ipc_thresholds,
    )


def evaluate_capacity_condition(
    config: Capacity03Config, condition: CapacityCondition
) -> CapacityOutcome:
    """1条件を回して MC と IPC の**両方**を取る (**軌道生成 + 2段の薄い層**)。

    手順は4つで、順序そのものが設計判断である。

    1. ``simulate_reference_trajectory`` で ``X`` を**1条件につき1回だけ**作る
       (仕様 §5 の禁止構造「条件ごとに X を2回作る」を避ける)。参照軌道の
       生成は 02 から切り出し済みの関数をそのまま呼び、03 側で書き直さない。
    2. ``ctx`` を1個だけ作る (D-37: サロゲートのシードは全条件で共通)。
       ``washout`` も同じ値を使い、``t0`` の違いは各診断が
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
    _validate_condition_bounds(condition)
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

    ctx = DiagnosticContext(washout=config.drive.washout, seed=config.seeds.surrogate)
    measurement = measure_capacity(
        reference.states,
        reference.drive,
        ctx=ctx,
        mc_cfg=config.mc,
        ipc_cfg=ipc_config_for(config, condition.experiment),
    )
    row = capacity_row_from(
        measurement,
        experiment=condition.experiment,
        replicate=condition.replicate,
        seed_reservoir=config.seeds.reservoir,
        seed_drive=config.seeds.drive,
        seed_surrogate=config.seeds.surrogate,
        rho=condition.rho,
        leak_rate=condition.leak_rate,
        input_scale=config.reservoir.input_scale,
        sigma_u=condition.sigma_u,
        n_units=condition.n_units,
        density=config.reservoir.density,
        state_noise=condition.state_noise,
        n_steps=condition.n_steps,
        washout=config.drive.washout,
        wall_time_state_s=wall_time_state_s,
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
    return capacity_outcome_from(measurement, row)


def profile_rows(outcome: CapacityOutcome) -> tuple[CapacityProfileRow, ...]:
    """1条件の配列を ``capacity_profile.csv`` の長形式の行に落とす (D-38)。

    **しきい値後の容量が厳密に正のセルだけ**を返す。全セルを書くと本番設定で
    約6万行になり、``results/`` はコミット対象なのでリポジトリがその分だけ
    重くなる。正値だけに絞る条件は IPC の ``n_targets_kept``
    (``np.count_nonzero(kept)``) と同じ ``> 0`` だが、行数と ``n_targets_kept``
    は単位が違う (セル単位 vs 目標単位) ため一致しない (F-3b1-1-003、詳しくは
    ``CapacityProfileRow`` の docstring)。成果物単体で検算できる不変条件は
    ``capacity`` 列の総和が ``ipc_total`` / ``mc_total`` と一致することである。

    MC は ``degree=1`` 固定・遅延は ``mc_profile`` の index+1、IPC は
    ``ipc_heatmap`` の (次数, 遅延) セルをそのまま行にする。しきい値は MC が
    ``row.mc_threshold``、IPC が ``outcome.ipc_thresholds[degree-1]``
    (**再計算しない**。診断を2回走らせることになるため)。
    """
    row = outcome.row
    rows: list[CapacityProfileRow] = []

    def add(
        diagnostic: str, degree: int, delay: int, capacity: float, threshold: float
    ) -> None:
        rows.append(
            CapacityProfileRow(
                experiment=row.experiment,
                replicate=row.replicate,
                rho=row.rho,
                leak_rate=row.leak_rate,
                n_units=row.n_units,
                state_noise=row.state_noise,
                diagnostic=diagnostic,
                degree=degree,
                delay=delay,
                capacity=capacity,
                threshold=threshold,
            )
        )

    for index, capacity in enumerate(outcome.mc_profile):
        if capacity > 0.0:
            add(DIAGNOSTIC_MC, 1, index + 1, float(capacity), row.mc_threshold)
    for degree_index, cells in enumerate(outcome.ipc_heatmap):
        threshold = outcome.ipc_thresholds[degree_index]
        for delay_index, capacity in enumerate(cells):
            if capacity > 0.0:
                add(
                    DIAGNOSTIC_IPC,
                    degree_index + 1,
                    delay_index + 1,
                    float(capacity),
                    threshold,
                )
    return tuple(rows)


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
    "CAPACITY_CSV_COLUMNS",
    "CAPACITY_EXPERIMENTS",
    "CAPACITY_PROFILE_CSV_COLUMNS",
    "DIAGNOSTIC_IPC",
    "DIAGNOSTIC_MC",
    "EXPERIMENT_CONSERVATION",
    "EXPERIMENT_IPC_SWEEP",
    "EXPERIMENT_LENGTH_SWEEP",
    "EXPERIMENT_MC_SWEEP",
    "FIGURE_EXPERIMENTS",
    "CapacityCondition",
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
]

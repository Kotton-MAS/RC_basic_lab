"""実験03 の成果物の行 dataclass と、行を組み立てる関数.

``experiment/capacity.py`` から**行の形と組み立てだけ**を切り出したモジュール。
``experiment/anomaly_rows.py`` (05) と同じ切り口である。

ここが持つのは「CSV の1行がどんな列を持ち、測定結果からどう作られるか」だけで、
**測り方も掃引の組み方も知らない**。列を1つ足すときに読む場所が1つに決まる。
"""

from __future__ import annotations

from dataclasses import dataclass

from rc_basics_lab.diagnostics.base import DiagnosticResult
from rc_basics_lab.experiment.diagnostics_rows import (
    DiagnosticScalarRow,
    condition_key,
    scalar_rows,
)
from rc_basics_lab.types import FloatArray

DIAGNOSTIC_MC = "mc"
"""``CapacityProfileRow.diagnostic``: 線形メモリ容量 (次数は常に1)。"""


DIAGNOSTIC_IPC = "ipc"
"""``CapacityProfileRow.diagnostic``: 情報処理容量 (次数 x 遅延)。"""


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
        diagnostics: 診断が返したスカラの長形式の行 (D-118)。**主表の列を
            増やさずに新しい診断の値を出すための経路**で、``diagnostics.csv``
            へそのまま書ける。
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
    diagnostics: tuple[DiagnosticScalarRow, ...]
    mc_profile: FloatArray
    ipc_heatmap: FloatArray
    ipc_thresholds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CapacityProfileRow:
    """``capacity_profile.csv`` の1行 (**長形式**、D-38)。宣言順が CSV の列順。

    次数と遅延を**行の値**に落とすことで、CSV の列を ``cfg`` に依らず静的に
    保つ (F-03-1-005: IPC の ``scalars`` は次数の本数だけキーを持つため、
    列にすると打ち切りを1本増やした瞬間に列が変わる)。

    MC は次数の概念を持たないので ``degree=1`` に固定する (次数1の線形容量で
    あることは IPC の次数1と同じ意味であり、``diagnostic`` 列で区別する)。

    **書くのはしきい値後の容量が厳密に正のセルだけ**である (全セルだと本番で
    約6万行になり ``results/`` はコミット対象)。行数は ``n_targets_kept`` と
    一致しない —— 行は (次数, 遅延) のセル単位、``n_targets_kept`` は目標単位で、
    1セルに複数目標が畳み込まれる (F-3b1-1-003)。成果物単体で検算できるのは
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

    ``mc`` / ``ipc`` の ``scalars`` のどのキーがどの列になるかの対応はここに
    しかない。実験ごとに複製すると「CSV の列順の単一の真実 = 行 dataclass の
    宣言順」が破れる (F-3b1-1-004)。``experiment`` 以降がキーワード専用なのは、
    隣接する同型の値の取り違えを防ぐため。

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
        wall_time_s: **容量測定 (状態生成 + MC + IPC) の合計時間** [秒]。3-C
            (``run_narma10``) を含む全経路で同じ意味であり、``run_task``
            (3手法 x 全レプリケート) は含まない (F-3b2-1-004/M4)。区間単位の
            ``capacity_pipeline.SectionTiming.wall_time_s`` は3-C だけこれとは
            別の値 (``run_task`` を含む3-C全体) に差し替わる —— 同じ列名
            ``wall_time_s`` が行単位 (ここ) と区間単位 (``SectionTiming``) で
            指す量が3-Cだけ食い違うので、``meta.json`` を読む側は
            ``capacity.csv`` の行の ``wall_time_s`` と
            ``wall_time_breakdown`` の ``wall_time_s`` を同一視しないこと。

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
        # 診断のスカラは長形式へ逃がす (D-118)。主表 (capacity.csv) の 39 列は
        # 1つも動かさないので、診断を足しても指紋も golden も動かない。
        diagnostics=scalar_rows(
            (measurement.mc, measurement.ipc),
            experiment=row.experiment,
            condition_id=condition_key(
                {
                    "rho": row.rho,
                    "leak_rate": row.leak_rate,
                    "n_units": row.n_units,
                    "state_noise": row.state_noise,
                }
            ),
            replicate=row.replicate,
        ),
        mc_profile=measurement.mc.arrays["mc_profile"],
        ipc_heatmap=measurement.ipc.arrays["ipc_heatmap"],
        ipc_thresholds=measurement.ipc_thresholds,
    )


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


__all__ = [
    "DIAGNOSTIC_IPC",
    "DIAGNOSTIC_MC",
    "CapacityMeasurement",
    "CapacityOutcome",
    "CapacityProfileRow",
    "CapacityRow",
    "capacity_outcome_from",
    "capacity_row_from",
    "profile_rows",
]

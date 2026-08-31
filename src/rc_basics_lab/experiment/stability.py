"""実験 4-C (自走の3態マップ) と 4-D (同じ状態行列への MC / IPC) の配線.

**1条件につきリザバーは1つ、状態行列は1本だけ作る** (仕様 §5 禁止する構造4)。
4-C の分類も 4-D の容量も、``plan_replicate`` が作った同じ ``plan.states`` を
見る —— 03 の 3-C が「NARMA10 を解いた ESN」と「容量を測った ESN」を
``plan0`` の共有で同一にしたのとまったく同じ形である。2回作ると「たまたま
一致している」だけになり、片方の経路のシードや ``n_steps`` を変えた瞬間に
黙って別物になる。

**02 の ESP 判定経路 (``simulate_condition``) は呼ばない** (D-47 / ADR 0001 §3.4)。
4-C は ``state_noise`` を掃引軸に持つが、``state_noise > 0`` を比較軌道ループへ
入れると D-14 の3ストリーム分離の外から4本目の変動が混ざり、結果が評価順に
依存する。3態は自走軌道そのものの統計から純関数で決める (D-45) ので、ESP 判定
経路は要らない。呼びたくなったらそれは設計の逸脱であり、D-47 の ``ValueError``
がその場で止める。

**真の軌道は条件ごとに積分し直さない** (仕様 §5 禁止する構造3)。lambda_max も
Lyapunov 時間も ``estimate_lorenz_lyapunov`` が1回だけ推定した値を引数で受ける。

**確保軸5 (条件数) は条件を1つも作る前に検査する** (D-34)。自走は逐次計算で
ベクトル化できない (仕様 §10-1) ので、時間は「格子の積 x レプリケート」に
そのまま比例する。
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rc_basics_lab.config import Chaos04Config, ESNConfig
from rc_basics_lab.diagnostics.base import DiagnosticContext, DiagnosticResult
from rc_basics_lab.experiment.attractor import (
    REGIMES,
    RegimeVerdict,
    classify_regime,
    lyapunov_normalized,
    normalized_error_curve,
    valid_time_from_errors,
    validate_stats_bounds,
)
from rc_basics_lab.experiment.capacity import measure_capacity
from rc_basics_lab.experiment.capacity_bounds import (
    validate_state_matrix_bounds,
    validate_total_step_count,
)
from rc_basics_lab.experiment.capacity_rows import (
    CapacityMeasurement,
    CapacityRow,
    capacity_row_from,
)
from rc_basics_lab.experiment.freerun import run_free_run
from rc_basics_lab.experiment.freerun_tasks import (
    chaos_esn_config,
    lorenz_task_entry,
    task_length,
)
from rc_basics_lab.experiment.report import write_rows_csv
from rc_basics_lab.experiment.runner import ESN_METHOD, TaskEntry
from rc_basics_lab.reservoir.registry import require_esn
from rc_basics_lab.tasks.base import TaskData
from rc_basics_lab.tasks.chaotic import sampling_interval
from rc_basics_lab.types import FloatArray

logger = logging.getLogger(__name__)

EXPERIMENT_STABILITY = "4C_stability"
"""``stability.csv`` の ``experiment`` 列 (4-C)。"""

EXPERIMENT_FREERUN_CAPACITY = "4D_freerun_capacity"
"""``capacity.csv`` (04) の ``experiment`` 列 (4-D)。

03 の ``CAPACITY_EXPERIMENTS`` には**足さない** —— あちらは
``results/03_capacity/capacity.csv`` に出る実験ラベルの集合で、足すと 03 の
成果物の検査 (ラベルの網羅性) が 04 の行を要求しはじめる。04 の容量行は
``results/04_chaotic_freerun/capacity.csv`` に**同じ列で**出す。
"""

STABILITY_CSV = "stability.csv"
"""4-C の成果物名 (列順は ``StabilityRow`` の宣言順が単一の真実)。"""

CAPACITY_CSV = "capacity.csv"
"""4-D の成果物名 (**03 と同じ ``CapacityRow`` の列**)。

03 の ``results/03_capacity/capacity.csv`` はバイト不変でなければならないので、
04 の行は 04 のディレクトリへ出す (D-51)。列の定義は複製せず
``experiment/capacity.py`` の ``CAPACITY_CSV_COLUMNS`` を参照する。
"""

CAPACITY_DRIVE_COMPONENT = 0
"""4-D が MC / IPC の駆動入力として使う成分 (Lorenz の x)。

MC / IPC は**単一変数の駆動信号**から遅延目標を作る (``input_series``)。Lorenz
の入力は3変数なので、どれを駆動と見なすかを決める必要がある。x を選ぶのは
Lorenz のリターンマップ・可視化で慣例的に主軸となる成分であり、y・z と対称な
役割ではないため (x と y は蝶の2枚翅を、z は高さを表す)。

**この選択には測定上の限界がある** (``docs/design.md`` §12):
Lorenz の x は i.i.d. ではなく強く時間相関した決定的な系列なので、MC / IPC が
前提とする「駆動が独立同分布で、遅延目標が互いに直交する」が成り立たない。
実測では ``mc_total`` も ``ipc_total`` も保存則 (<= N = 200) を超える
(例: ``mc_total`` = 258、``ipc_total`` = 601)。相関した駆動では遅延目標が
互いに予測可能なので、容量が重複して数え上げられるためである。

04 は 03 の接ぎ目をそのまま使うこと (仕様 §5 禁止する構造4: 条件ごとに ESN を
2回作らない) を優先しているので、**04 の容量の絶対値は 03 の掃引とも保存則とも
比較できない**。読めるのは**同じ駆動の下での条件間の相対比較**だけである
(4-D の問い「自走が上手くいく領域が容量指標で説明できるか」はそれで足りる)。
"""

CAPACITY_SIGMA_U = 1.0
"""``CapacityRow.sigma_u`` に書く駆動強度の**宣言値**。

04 の駆動は標準化した Lorenz の x なので、訓練区間で標準偏差 1 になるよう
係数を決めてある (D-41)。実測値は ``input_drive_std`` 列に別途出る
(3-C が ``NARMA10_INPUT_STD`` を宣言値として書くのと同じ流儀)。
"""

_MAX_CONDITIONS = 2_000
"""確保軸5: 4-C が回す条件数 (格子の積 x レプリケート) の絶対上限。

自走は逐次計算でベクトル化できない (仕様 §10-1) ので、実行時間はこの数に
比例する。本番設定は 4 x 4 x 4 x 3 = 192 条件で、上限 2000 は 10 倍の余裕を
残しつつ 4-C + 4-D を予算 (300 秒 + 150 秒) の内側に抑える (実測: 1条件
あたり約 0.75 秒)。**上書き不能な定数**であり設定からは動かせない。

``stability.n_replicates`` だけを 10 倍しても、格子だけを 10 倍しても同じ
1つの軸に当たるので、**積で縛る** (片方だけを見る検査はもう一方ですり抜ける)。
"""


def validate_condition_count(n_conditions: int) -> None:
    """確保軸5 を**条件を1つも作る前に**検査する (D-34)。

    Raises:
        ValueError: 条件数が 1 未満、または ``_MAX_CONDITIONS`` 超過。
    """
    if n_conditions < 1:
        raise ValueError(f"条件が1つもありません: {n_conditions}")
    if n_conditions > _MAX_CONDITIONS:
        raise ValueError(
            f"条件数が上限を超えています: {n_conditions} > {_MAX_CONDITIONS} "
            "(格子の積 x レプリケート。自走は逐次計算なので時間がこの数に"
            "比例する)"
        )


@dataclass(frozen=True, slots=True)
class StabilityCondition:
    """4-C の1条件 (ハイパーパラメータ平面の1点 x レプリケート)。

    Attributes:
        rho: スペクトル半径。
        leak_rate: リーク率。
        state_noise: 学習時・自走時の状態ノイズ (D-36)。
        replicate: レプリケート番号。
    """

    rho: float
    leak_rate: float
    state_noise: float
    replicate: int


def stability_conditions(config: Chaos04Config) -> tuple[StabilityCondition, ...]:
    """掃引の全条件を作る (**作る前に確保軸5・積の軸を検査する**)。

    並びは (rho, leak_rate, state_noise, replicate) の入れ子順で、
    ``stability.csv`` の行順もこれに一致する。

    Raises:
        ValueError: 格子が空、確保軸5 を超える、または条件数 x ``stats_steps``
            の積が上限を超える場合。
    """
    stability = config.stability
    if not (
        stability.spectral_radius_grid
        and stability.leak_rate_grid
        and stability.state_noise_grid
    ):
        raise ValueError("stability の格子が空です")
    if stability.n_replicates < 1:
        raise ValueError(
            f"n_replicates は 1 以上である必要があります: {stability.n_replicates}"
        )
    n_conditions = (
        len(stability.spectral_radius_grid)
        * len(stability.leak_rate_grid)
        * len(stability.state_noise_grid)
        * stability.n_replicates
    )
    validate_condition_count(n_conditions)
    # 軸5 と軸4 は単独では上限内でも、**積** が軸検査をすり抜けて膨らみうる。
    validate_total_step_count(n_conditions * config.freerun.stats_steps)
    return tuple(
        StabilityCondition(
            rho=rho, leak_rate=leak, state_noise=noise, replicate=replicate
        )
        for rho in stability.spectral_radius_grid
        for leak in stability.leak_rate_grid
        for noise in stability.state_noise_grid
        for replicate in range(stability.n_replicates)
    )


def condition_esn_config(
    config: Chaos04Config, condition: StabilityCondition
) -> ESNConfig:
    """条件の3軸だけを差し替えた ESN 設定を返す (他の構造 HP は動かさない)。

    D-08 により構造ハイパーパラメータは検証分割で選ばない。掃引で動くのは
    ``spectral_radius`` / ``leak_rate`` / ``state_noise`` の3つだけで、
    ``n_units`` / ``input_scale`` / ``density`` / ``activation`` は 4-A・4-B と
    同じ1点のままである (動かすと「ノイズで領域が変わった」のか
    「別のリザバーだった」のかが分からなくなる)。
    """
    return dataclasses.replace(
        chaos_esn_config(config.base),
        spectral_radius=condition.rho,
        leak_rate=condition.leak_rate,
        state_noise=condition.state_noise,
    )


def condition_task_entry(
    config: Chaos04Config,
    condition: StabilityCondition,
    *,
    trajectory_cache: dict[int, TaskData] | None = None,
) -> TaskEntry:
    """条件ごとの ``TaskEntry`` (課題は Lorenz 固定、ESN 設定だけが動く)。

    4-C / 4-D は Lorenz だけを回す (仕様 §8: 2系で回すと条件数が2倍になり
    900 秒予算を割る)。課題の生成関数は ``lorenz_task_entry`` から借りるので、
    真の軌道の作り方が 4-A / 4-B と1行も違わない。

    **真の軌道は replicate だけで決まり (rho, leak, noise) に依存しない**ので、
    ``trajectory_cache`` を渡すと同じ replicate の2回目以降の呼び出しは
    積分をやり直さずキャッシュした ``TaskData`` をそのまま返す (仕様 §5
    禁止する構造3)。``plan_replicate`` は毎回
    ``make_rng(TASK, replicate)`` を作り直す (同じシード) ので、この
    キャッシュはビット単位で再計算した場合と同一の値になる。
    ``trajectory_cache=None`` (既定) では毎回積分し直し、挙動は導入前と
    変わらない。

    Args:
        config: 04 の設定。
        condition: 回す1条件。
        trajectory_cache: replicate -> ``TaskData`` のキャッシュ (呼び出し側が
            条件をまたいで共有する)。

    Returns:
        条件の ESN 設定を差し替えた ``TaskEntry``。
    """
    base_entry = lorenz_task_entry(config)
    replicate = condition.replicate

    def generate(rng: np.random.Generator) -> TaskData:
        if trajectory_cache is None:
            return base_entry.generate(rng)
        if replicate not in trajectory_cache:
            trajectory_cache[replicate] = base_entry.generate(rng)
        return trajectory_cache[replicate]

    reservoir = condition_esn_config(config, condition)
    return dataclasses.replace(base_entry, reservoir=reservoir, generate=generate)


@dataclass(frozen=True, slots=True)
class StabilityRow:
    """``stability.csv`` の1行 (4-C の1条件)。列順はこの宣言順が単一の真実。

    容量 (MC / IPC) の列は**ここに複製しない**。4-D の行は同じ条件キー
    (``rho`` / ``leak_rate`` / ``state_noise`` / ``replicate``) を持つ
    ``capacity.csv`` (04) 側にあり、2枚を join すれば「自走が上手くいく領域が
    容量指標で説明できるか」を見られる (03 の ``narma10.csv`` と
    ``capacity.csv`` の関係と同じ)。約35列ある ``CapacityRow`` をここへ写すと
    列の単一の真実が2つになる。

    Attributes:
        experiment: ``EXPERIMENT_STABILITY``。
        rho / leak_rate / state_noise / replicate: 条件。
        n_units: リザバーのユニット数 (掃引では動かさない)。
        alpha / val_nrmse: 教師強制で選ばれた読み出し。
        regime: 3態分類 (D-45)。**純関数 + 数値基準**で決まる。
        amplitude_ratio / std_ratio / autocorr_peak: 分類の根拠になった数値。
        diverged / n_completed: 自走の打ち切り。
        stats_steps: 自走させたステップ数 (4-B と同じ窓で測る)。
        valid_time_threshold / valid_time_steps / valid_time_lyapunov /
        valid_time_censored: 有効予測時間 (D-43。4-B と同じ定義)。
        wall_time_s: 条件の実測 wall time [秒] (状態生成 + 学習 + 自走)。
    """

    experiment: str
    rho: float
    leak_rate: float
    state_noise: float
    replicate: int
    n_units: int
    alpha: float
    val_nrmse: float
    regime: str
    amplitude_ratio: float
    std_ratio: float
    autocorr_peak: float
    diverged: bool
    n_completed: int
    stats_steps: int
    valid_time_threshold: float
    valid_time_steps: int
    valid_time_lyapunov: float
    valid_time_censored: bool
    wall_time_s: float


STABILITY_CSV_COLUMNS: tuple[str, ...] = tuple(
    item.name for item in dataclasses.fields(StabilityRow)
)
"""``stability.csv`` の列順 (``StabilityRow`` の宣言順)。"""


@dataclass(frozen=True, slots=True)
class StabilityOutcome:
    """1条件ぶんの 4-C + 4-D の結果。

    Attributes:
        row: ``stability.csv`` の行 (3態分類)。
        capacity: ``capacity.csv`` (04) の行 (**同じ状態行列**への MC / IPC)。
    """

    row: StabilityRow
    capacity: CapacityRow


def capacity_context(config: Chaos04Config) -> DiagnosticContext:
    """4-D の全条件で共有する ``DiagnosticContext`` を1個作る (D-37)。

    ``washout`` は 01 の分割設定の値、``seed`` は
    ``stability.surrogate_seed``。条件ごとにサロゲートのシードを振ると、
    条件間の容量差にしきい値の推定ノイズが独立に乗る (共通乱数法)。
    """
    return DiagnosticContext(
        washout=config.base.split.washout, seed=config.stability.surrogate_seed
    )


def evaluate_stability_condition(
    config: Chaos04Config,
    condition: StabilityCondition,
    *,
    dt: float,
    lyapunov_time: float,
    ctx: DiagnosticContext,
    trajectory_cache: dict[int, TaskData] | None = None,
) -> StabilityOutcome:
    """1条件を回して3態分類 (4-C) と容量 (4-D) を**同じ状態行列から**取る。

    手順と順序そのものが設計判断である。

    1. 確保軸3 (``stats_steps * n_units``) と課題側の確保軸を、状態行列を
       作る前に検査する。
    2. ``run_free_run`` が ``plan_replicate`` で状態行列を**1本だけ**作り、
       教師強制で読み出しを学習し、そのまま自走させる。真の軌道は
       ``trajectory_cache`` (replicate をキーにする) 経由で ``condition_task_
       entry`` から渡すので、(rho, leak, noise) ごとに Lorenz を積分し直さない
       (仕様 §5 禁止する構造3)。
    3. 自走軌道を純関数 (D-45) が3態へ分類する。**図も目視も使わない**。
    4. 2. が作った ``plan.states`` を ``measure_capacity`` へそのまま渡す
       (仕様 §5 禁止する構造4: 条件ごとに ESN を2回作らない)。

    Args:
        config: 04 の設定。
        condition: 回す1条件。
        dt: サンプリング間隔 [時間]。
        lyapunov_time: ``1 / lambda_max`` [時間] (D-42 の数値推定)。
        ctx: 全条件で共有する ``DiagnosticContext`` (D-37)。
        trajectory_cache: replicate -> ``TaskData`` のキャッシュ。呼び出し側
            (``run_stability_experiment``) が全条件で1個を共有する。
            ``None`` (既定) なら毎回積分し直す。

    Returns:
        ``StabilityOutcome``。

    Raises:
        ValueError: 確保軸を超える設定、または課題・診断側の値域違反。
    """
    started = time.perf_counter()
    entry = condition_task_entry(config, condition, trajectory_cache=trajectory_cache)
    validate_state_matrix_bounds(
        entry.reservoir.n_units, task_length(config, entry.name)
    )

    outcome = run_free_run(
        config,
        entry,
        condition.replicate,
        n_steps=config.freerun.stats_steps,
        method=ESN_METHOD,
    )
    plan = outcome.readout.plan
    result = outcome.result
    trajectory: FloatArray = result.predictions[: result.n_completed]
    truth: FloatArray = plan.task.y
    verdict: RegimeVerdict = classify_regime(
        trajectory, reference=truth, diverged=result.diverged
    )
    horizon = min(config.freerun.free_run_steps, result.predictions.shape[0])
    valid = valid_time_from_errors(
        normalized_error_curve(outcome.truth[:horizon], result.predictions[:horizon]),
        config.freerun.valid_time_threshold,
    )
    wall_time_state_s = time.perf_counter() - started

    esn = require_esn(entry.reservoir, "実験4-D (自走と同じ状態行列への容量)")
    row = StabilityRow(
        experiment=EXPERIMENT_STABILITY,
        rho=condition.rho,
        leak_rate=condition.leak_rate,
        state_noise=condition.state_noise,
        replicate=condition.replicate,
        n_units=esn.n_units,
        alpha=outcome.readout.alpha,
        val_nrmse=outcome.readout.val_nrmse,
        regime=verdict.regime,
        amplitude_ratio=verdict.amplitude_ratio,
        std_ratio=verdict.std_ratio,
        autocorr_peak=verdict.autocorr_peak,
        diverged=result.diverged,
        n_completed=result.n_completed,
        stats_steps=int(result.predictions.shape[0]),
        valid_time_threshold=valid.threshold,
        valid_time_steps=valid.steps,
        valid_time_lyapunov=lyapunov_normalized(valid.steps, dt, lyapunov_time),
        valid_time_censored=valid.censored,
        wall_time_s=wall_time_state_s,
    )

    drive: FloatArray = plan.task.u[
        :, CAPACITY_DRIVE_COMPONENT : CAPACITY_DRIVE_COMPONENT + 1
    ]
    measurement: CapacityMeasurement = measure_capacity(
        plan.states, drive, ctx=ctx, mc_cfg=config.mc, ipc_cfg=config.ipc
    )
    capacity = capacity_row_from(
        measurement,
        experiment=EXPERIMENT_FREERUN_CAPACITY,
        replicate=condition.replicate,
        seed_reservoir=config.base.seeds.reservoir,
        # 4-D の駆動は課題の入力そのものなので、基底シードは task (D-06)。
        seed_drive=config.base.seeds.task,
        seed_surrogate=config.stability.surrogate_seed,
        rho=esn.spectral_radius,
        leak_rate=esn.leak_rate,
        input_scale=esn.input_scale,
        sigma_u=CAPACITY_SIGMA_U,
        n_units=esn.n_units,
        density=esn.density,
        state_noise=esn.state_noise,
        n_steps=int(plan.states.shape[0]),
        washout=config.base.split.washout,
        wall_time_state_s=wall_time_state_s,
        wall_time_s=time.perf_counter() - started,
    )
    return StabilityOutcome(row=row, capacity=capacity)


@dataclass(frozen=True, slots=True)
class StabilityResults:
    """実験 4-C + 4-D の結果。

    Attributes:
        outcomes: 条件ごとの結果 (``stability_conditions`` と同じ並び)。
        wall_time_s: 全体の実測 wall time [秒]。
        wall_time_capacity_s: そのうち 4-D (MC + IPC) の合計 [秒]。
    """

    outcomes: tuple[StabilityOutcome, ...]
    wall_time_s: float
    wall_time_capacity_s: float

    @property
    def rows(self) -> tuple[StabilityRow, ...]:
        """``stability.csv`` と同じ行。"""
        return tuple(item.row for item in self.outcomes)

    @property
    def capacity_rows(self) -> tuple[CapacityRow, ...]:
        """``capacity.csv`` (04) と同じ行。"""
        return tuple(item.capacity for item in self.outcomes)


def run_stability_experiment(
    config: Chaos04Config, lyapunov: DiagnosticResult
) -> StabilityResults:
    """実験 4-C と 4-D を回す (**条件ごとに ESN は1つ**)。

    Args:
        config: 04 の設定。
        lyapunov: ``estimate_lorenz_lyapunov`` の結果 (**条件ごとに真の軌道を
            積分し直さない**、仕様 §5 禁止する構造3)。

    Returns:
        ``StabilityResults``。

    Raises:
        ValueError: 確保軸を超える設定、または課題・診断側の値域違反。
    """
    started = time.perf_counter()
    validate_stats_bounds(config.freerun.stats_steps)
    conditions = stability_conditions(config)
    ctx = capacity_context(config)
    dt = sampling_interval(config.lorenz)
    lyapunov_time = lyapunov.scalars["lyapunov_time"]
    # 真の軌道は replicate だけで決まるので全条件でキャッシュを共有する (仕様 §5)。
    trajectory_cache: dict[int, TaskData] = {}

    outcomes = tuple(
        evaluate_stability_condition(
            config,
            condition,
            dt=dt,
            lyapunov_time=lyapunov_time,
            ctx=ctx,
            trajectory_cache=trajectory_cache,
        )
        for condition in conditions
    )
    wall_time_s = time.perf_counter() - started
    wall_time_capacity_s = sum(
        item.capacity.wall_time_mc_s + item.capacity.wall_time_ipc_s
        for item in outcomes
    )
    logger.info(
        "experiment=%s 条件数=%d 3態=%s (4-D 容量 %.1fs) wall_time=%.1fs",
        EXPERIMENT_STABILITY,
        len(outcomes),
        regime_counts(tuple(item.row for item in outcomes)),
        wall_time_capacity_s,
        wall_time_s,
    )
    return StabilityResults(
        outcomes=outcomes,
        wall_time_s=wall_time_s,
        wall_time_capacity_s=wall_time_capacity_s,
    )


def regime_counts(rows: Sequence[StabilityRow]) -> dict[str, int]:
    """3態ごとの行数 (ログと ``meta.json`` の要約)。"""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.regime] = counts.get(row.regime, 0) + 1
    return counts


def regime_map(
    rows: Sequence[StabilityRow], state_noise: float
) -> dict[tuple[float, float], str]:
    """あるノイズ量での ``(rho, leak_rate) -> 代表の3態`` を返す (**純関数**)。

    レプリケートが複数ある格子点は**多数決**で1つに畳む。同数のときは
    ``REGIMES`` の並び (発散 -> 周期 -> アトラクタ) で先にあるものを採る ——
    「悪い方へ倒す」規則を明示しておくと、同数の格子点で図と表が食い違わない。

    受け入れ条件4 (「ノイズ注入の有無で領域が変わる」) は、この関数の出力を
    ノイズ量ごとに比べて実測する (図ではなく行から決める)。
    """
    grouped: dict[tuple[float, float], list[str]] = {}
    for row in rows:
        if row.state_noise != state_noise:
            continue
        grouped.setdefault((row.rho, row.leak_rate), []).append(row.regime)
    return {
        key: min(
            set(values),
            key=lambda regime: (-values.count(regime), REGIMES.index(regime)),
        )
        for key, values in grouped.items()
    }


def write_stability_csv(rows: Sequence[StabilityRow], path: Path) -> Path:
    """4-C の結果を CSV に書く (列順は ``StabilityRow`` の宣言順)。"""
    return write_rows_csv(rows, path, STABILITY_CSV_COLUMNS)


def valid_time_by_regime(rows: Sequence[StabilityRow]) -> dict[str, float]:
    """3態ごとの有効予測時間の中央値 [Lyapunov 時間] (``meta.json`` の要約)。"""
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.regime, []).append(row.valid_time_lyapunov)
    return {
        regime: float(np.median(values)) for regime, values in sorted(grouped.items())
    }


__all__ = [
    "CAPACITY_CSV",
    "CAPACITY_DRIVE_COMPONENT",
    "CAPACITY_SIGMA_U",
    "EXPERIMENT_FREERUN_CAPACITY",
    "EXPERIMENT_STABILITY",
    "STABILITY_CSV",
    "STABILITY_CSV_COLUMNS",
    "StabilityCondition",
    "StabilityOutcome",
    "StabilityResults",
    "StabilityRow",
    "capacity_context",
    "condition_esn_config",
    "condition_task_entry",
    "evaluate_stability_condition",
    "regime_counts",
    "regime_map",
    "run_stability_experiment",
    "stability_conditions",
    "valid_time_by_regime",
    "validate_condition_count",
    "write_stability_csv",
]

"""実験04 (カオス時系列の自由走行予測) の設定 dataclass 群 (D-13).

01 の ``ExperimentConfig`` / 02 の ``Esp02Config`` / 03 の ``Capacity03Config``
とはローダ (``_common.load_config_as``) だけを共有し、フィールドは一切共有
しない。**``ExperimentConfig`` に 04 のフィールドを1個も足さない** (D-13)。

``experiment01`` への import は ``Chaos04Config.base`` (4-A / 4-B が 01 の
``run_task`` を再利用するための内包、D-31 と同じ形) の1本だけで、**一方向**で
ある (D-49)。``esp02`` / ``capacity03`` はここから import しない。

**Mackey-Glass の生成パラメータをここに持たない**のは意図的である。04 の MG は
``tasks/chaotic.py`` が ``tasks/mackey_glass.py`` へ委譲する薄い adapter で
生成するので、パラメータの単一の真実は ``base.mackey_glass``
(01 の ``MackeyGlassConfig``) のままでよい。ここに2本目を置くと「どちらが
効いているか」が設定から読めなくなる。

Lorenz のパラメータ (sigma, rho, beta) = (10, 28, 8/3) は設定フィールドに
**しない** (D-41)。系そのものを定義する定数は ``tasks/chaotic.py`` の
モジュール定数に置く。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rc_basics_lab.config.experiment01 import ExperimentConfig
from rc_basics_lab.diagnostics.ipc import IpcConfig
from rc_basics_lab.diagnostics.lyapunov import MaxLyapunovConfig
from rc_basics_lab.diagnostics.memory_capacity import MemoryCapacityConfig
from rc_basics_lab.tasks.chaotic import LorenzConfig

LORENZ_LYAPUNOV_REFERENCE = 0.9056
"""Lorenz (10, 28, 8/3) の最大 Lyapunov 指数の**照合用**文献値 [1/時間]。

出典: D. Viswanath, *Lyapunov Exponents from Random Fibonacci Sequences to the
Lorenz Equations*, Ph.D. thesis, Cornell University (1998) —— 標準パラメータの
Lorenz 系に対して lambda_max = 0.9056。

**正本は数値推定である** (D-42)。この値は ``MaxLyapunovConfig.reference_value``
として推定値の照合にだけ使い、Lyapunov 時間の正規化 (D-43) には
``diagnostics/lyapunov.py`` が返す推定値を使う。

**判定基準なので ``config`` 層に置く** (D-15: 系そのものを表す量は ``ctx``、
判定基準は ``cfg``)。系の定義 (sigma, rho, beta) は ``tasks/chaotic.py`` にある。
"""


@dataclass(frozen=True, slots=True)
class MackeyGlassStandardizeConfig:
    """04 の MG 課題の標準化 (D-41)。生成パラメータは ``base.mackey_glass``。

    04 は Lorenz と MG を**同じ自走系**に流すため、MG 側にも Lorenz と同じ
    標準化 (訓練区間から推定した1組の係数を全区間で使う) が要る。生成そのものは
    ``tasks/mackey_glass.py`` へ委譲する (再実装しない) ので、04 が名乗るのは
    この1個だけである。

    Attributes:
        standardize_steps: 標準化係数を推定する先頭サンプル数 (D-41)。
    """

    standardize_steps: int = 3000


@dataclass(frozen=True, slots=True)
class FreeRunConfig:
    """自走 (closed-loop) の実行条件 (D-44 / D-50)。

    Attributes:
        warmup_steps: 教師強制で状態を温めるステップ数。自走はこの直後の
            状態から始まる。
        free_run_steps: 自走させるステップ数。**確保軸3**
            (``free_run_steps * n_units``) の一方の項で、自走の入口で
            ``validate_state_matrix_bounds`` が確保より前に検査する。
        stats_steps: 長時間統計 (D-46) に使う自走の総ステップ数。**確保軸4**で
            あり、上書き不能な絶対上限 (``experiment/attractor.py`` の
            ``_MAX_STATS_STEPS``) が 4-B の入口で確保より前に検査する。
            自走は**1本しか回さない** —— 先頭 ``free_run_steps`` を有効予測時間
            (D-43) に、全体を長時間統計に使う。ヒストグラムのビン数と FFT 長は
            この値に**従属**させ、独立した設定軸にしない (確保軸7)。
        valid_time_threshold: 有効予測時間の誤差しきい値 (D-43)。誤差は
            **NRMSE 比** (瞬時 RMSE / 真の軌道の標準偏差、D-02 と同じ正規化) で、
            これを初めて超えたステップまでが有効予測時間である。{0.2, 0.3, 0.4,
            0.5} の感度は ``experiment/attractor.py`` の
            ``VALID_TIME_THRESHOLD_GRID`` が別途まとめて測り、
            ``docs/design.md`` §12 の感度表の一次資料になる。

    Note:
        自走長を削って速くするのは受け入れ条件2 (有効予測時間の分布) を壊す
        ので禁止する (仕様 §10-1)。予算は条件数の側で調整すること。
    """

    warmup_steps: int = 200
    free_run_steps: int = 2000
    stats_steps: int = 20000
    valid_time_threshold: float = 0.4


@dataclass(frozen=True, slots=True)
class StabilityConfig:
    """実験 4-C (自走の3態マップ) の掃引軸 (D-45)。純データ (D-09)。

    格子の積 x ``n_replicates`` が**確保軸5** (条件数) で、上書き不能な絶対
    上限 (``experiment/stability.py`` の ``_MAX_CONDITIONS``) が**条件を1つも
    作る前に**検査する (D-34 の規律)。自走は逐次計算でベクトル化できない
    (仕様 §10-1) ので、時間はこの積にそのまま比例する。

    Attributes:
        spectral_radius_grid: スペクトル半径の格子。
        leak_rate_grid: リーク率の格子。
        state_noise_grid: 状態ノイズ (tanh 内部への加算、D-36) の格子。
            **これを変えると3態マップが変わる**ことが受け入れ条件4 の核心で、
            ``tests/test_experiment_stability.py::test_noise_changes_the_regime_map``
            が実測する。
        n_replicates: 条件あたりのレプリケート数。**予算超過時に落として
            よいのはこの値だけ** (仕様 §5)。格子・自走長は動かさない。
        surrogate_seed: 4-D の MC / IPC のしきい値サロゲート用
            ``DiagnosticContext.seed`` (D-27 / D-37)。**``SeedStream`` では
            ない** —— 診断へ ``ctx.seed`` としてそのまま渡る整数で、
            ``seeds.py`` の既存4ストリームを1つも動かさない。全条件で1個を
            共有する (共通乱数法、D-37)。03 の ``CapacitySeedConfig.surrogate``
            と同じ流儀。
    """

    spectral_radius_grid: tuple[float, ...] = (0.7, 0.9, 1.1, 1.3)
    leak_rate_grid: tuple[float, ...] = (0.1, 0.3, 0.6, 1.0)
    state_noise_grid: tuple[float, ...] = (0.0, 1.0e-4, 1.0e-3, 1.0e-2)
    n_replicates: int = 5
    surrogate_seed: int = 4


@dataclass(frozen=True, slots=True)
class Chaos04Config:
    """実験04 (カオス時系列の自由走行予測) 1本ぶんの設定 (D-13)。

    ``base`` として 01 の ``ExperimentConfig`` をまるごと内包する
    (``Narma10Config.base`` / ``WashoutSweepConfig.base`` と同じ形)。4-A は
    01 の ``run_task`` をそのまま通すので、公平性の3決定 (D-04 / D-05 / D-08)
    と MG の生成パラメータはすべて ``base`` 側が単一の真実である。

    ``lyapunov`` / ``mc`` / ``ipc`` は診断層の設定 (D-15) をそのまま載せた
    もので、効きの実測は各診断のテストへ委譲する。``mc`` / ``ipc`` の消費側は
    4-D (次サイクル) に生えるが、**04 で新しい上限を作らない**(確保軸8) ことを
    今のうちに固定するために、既存の D-34 の4段がそのまま効く形でここに置く。

    Attributes:
        name: 実験名 (``meta.json`` に出る純粋なメタ情報)。
        base: 01 の設定。MG の生成パラメータ (``base.mackey_glass``)・分割・
            alpha 格子・ESN 構造・シードはここが単一の真実。
        lorenz: Lorenz 系の生成パラメータ (04 が足す唯一の課題パラメータ)。
        mackey_glass: 04 の MG 課題の標準化 (生成は ``base.mackey_glass``)。
        freerun: 自走の実行条件 (4-B)。
        stability: 3態マップの掃引軸 (4-C)。
        lyapunov: 最大 Lyapunov 指数の推定条件と文献値との照合基準 (D-42)。
        mc: 線形メモリ容量の設定 (4-D。委譲先は ``diagnostics/``)。
        ipc: 情報処理容量の設定 (4-D。**既存 D-34 の4段を再利用する**)。
    """

    name: str = "04_chaotic_freerun"
    base: ExperimentConfig = field(default_factory=ExperimentConfig)
    lorenz: LorenzConfig = field(default_factory=LorenzConfig)
    mackey_glass: MackeyGlassStandardizeConfig = field(
        default_factory=MackeyGlassStandardizeConfig
    )
    freerun: FreeRunConfig = field(default_factory=FreeRunConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    lyapunov: MaxLyapunovConfig = field(
        default_factory=lambda: MaxLyapunovConfig(
            reference_value=LORENZ_LYAPUNOV_REFERENCE
        )
    )
    mc: MemoryCapacityConfig = field(default_factory=MemoryCapacityConfig)
    ipc: IpcConfig = field(default_factory=IpcConfig)


__all__ = [
    "LorenzConfig",
]

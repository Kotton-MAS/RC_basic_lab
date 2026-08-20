"""実験04 (カオス時系列の自由走行予測) の設定 dataclass 群 (D-13).

01 の ``ExperimentConfig`` / 02 の ``Esp02Config`` / 03 の ``Capacity03Config``
とはローダ (``_common.load_config_as``) だけを共有し、フィールドは一切共有
しない。**``ExperimentConfig`` に 04 のフィールドを1個も足さない** (D-13) ——
足すと 01 の ``tests/test_config_wiring.py::test_each_parameter_changes_output``
(「全フィールドが 01 のパイプライン出力を変える」) を満たせないフィールドが
生まれる。

``experiment01`` への import は ``Chaos04Config.base`` (4-A / 4-B が 01 の
``run_task`` を再利用するための内包、D-31 と同じ形) の1本だけで、**一方向**で
ある (D-49)。``esp02`` / ``capacity03`` はここから import しない。

**Mackey-Glass の生成パラメータをここに持たない**のは意図的である。04 の MG は
``tasks/chaotic.py`` が ``tasks/mackey_glass.py`` へ委譲する薄い adapter で
生成するので、パラメータの単一の真実は ``base.mackey_glass``
(01 の ``MackeyGlassConfig``) のままでよい。ここに2本目を置くと「どちらが
効いているか」が設定から読めなくなる。

Lorenz のパラメータ (sigma, rho, beta) = (10, 28, 8/3) は設定フィールドに
**しない** (D-41)。値を変えるとカオス域かどうかも lambda_max の照合値も
変わり、``LORENZ_LYAPUNOV_REFERENCE`` の意味が失われる。系そのものを定義する
定数は ``tasks/chaotic.py`` のモジュール定数に置く (``tasks/narma.py`` の
係数を設定にしない D-29 と同じ流儀)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rc_basics_lab.config.experiment01 import ExperimentConfig
from rc_basics_lab.diagnostics.ipc import IpcConfig
from rc_basics_lab.diagnostics.lyapunov import MaxLyapunovConfig
from rc_basics_lab.diagnostics.memory_capacity import MemoryCapacityConfig

LORENZ_LYAPUNOV_REFERENCE = 0.9056
"""Lorenz (10, 28, 8/3) の最大 Lyapunov 指数の**照合用**文献値 [1/時間]。

出典: D. Viswanath, *Lyapunov Exponents from Random Fibonacci Sequences to the
Lorenz Equations*, Ph.D. thesis, Cornell University (1998) —— 標準パラメータの
Lorenz 系に対して lambda_max = 0.9056。

**正本は数値推定である** (D-42)。この値は ``MaxLyapunovConfig.reference_value``
として推定値の照合にだけ使い、Lyapunov 時間の正規化 (D-43) には
``diagnostics/lyapunov.py`` が返す推定値を使う。文献値を正本にすると、
積分刻み・サンプリング間隔・burn-in を取り違えても何も落ちない。

**判定基準なので ``config`` 層に置く** (D-15: 系そのものを表す量は ``ctx``、
判定基準は ``cfg``)。系の定義 (sigma, rho, beta) は ``tasks/chaotic.py`` にある。
"""


@dataclass(frozen=True, slots=True)
class LorenzConfig:
    """Lorenz 系の生成パラメータ (D-41)。純データ。値域検証は使う側 (D-09)。

    フィールド名と単位は ``MackeyGlassConfig`` にそろえてある (``rk4_step`` /
    ``sample_interval`` / ``integration_burn_in`` / ``length`` / ``horizon``)。
    2つのカオス系で同じ語が別の意味を持つと、Delta t の較正結果を読み違える。

    Attributes:
        rk4_step: RK4 の積分刻み h [時間]。
        sample_interval: 何積分ステップごとにサンプルするか。
            サンプリング間隔は ``Delta t = rk4_step * sample_interval``。
            **Lyapunov 時間正規化の分母** (D-43) がこの値で動く。
        integration_burn_in: 捨てるサンプル数 (アトラクタへ乗るまでの過渡)。
            単位は**サンプル**で、積分ステップ数ではない (MG と同じ)。
        length: 課題の行数 T。
        horizon: 何ステップ先を予測するか。自走 (D-44) は出力を次時刻の入力へ
            戻すので、``horizon=1`` 以外では自走の意味が変わる。
        standardize_steps: 標準化係数を推定する先頭サンプル数 (D-41)。
            ここでは値域を検証しないが、**訓練区間の内側**でなければならず、
            実験層 (``experiment/freerun.py``) が分割と突き合わせて検査する。
    """

    rk4_step: float = 0.002
    sample_interval: int = 10
    integration_burn_in: int = 1000
    length: int = 8000
    horizon: int = 1
    standardize_steps: int = 3000


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

    Note:
        自走長を削って速くするのは受け入れ条件2 (有効予測時間の分布) を壊す
        ので禁止する (仕様 §10-1)。予算は条件数の側で調整すること。
    """

    warmup_steps: int = 200
    free_run_steps: int = 2000


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
        freerun: 自走の実行条件。
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
    lyapunov: MaxLyapunovConfig = field(
        default_factory=lambda: MaxLyapunovConfig(
            reference_value=LORENZ_LYAPUNOV_REFERENCE
        )
    )
    mc: MemoryCapacityConfig = field(default_factory=MemoryCapacityConfig)
    ipc: IpcConfig = field(default_factory=IpcConfig)

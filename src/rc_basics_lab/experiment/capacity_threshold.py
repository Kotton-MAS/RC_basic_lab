"""しきい値法の比較 —— 既定 (D-27) が総容量をどれだけ削るかの実測 (受け入れ条件3).

受け入れ条件3 は「しきい値処理の有無で総容量がどれだけ変わるかを記録し、
**既定を根拠つきで選ぶ**」である。既定はシャッフルサロゲート (D-27) だが、
その根拠が散文にしか無いと「なぜ生の容量をそのまま足さないのか」を成果物から
たどれない。ここでは**本番の代表条件1つ**を対象に ``threshold_mode`` だけを
振り直し、総容量の差を ``meta.json`` の ``threshold_comparison`` に落とす
(``docs/design.md`` §11.2 の表の一次資料)。

**軌道は1回しか作らない**。しきい値法は診断の中で容量を切る段だけを変える
パラメータなので、02 の閾値感度 (``experiment/threshold.py``) と同じく
「状態を1回作り、判定だけをやり直す」形にできる。素直に3回掃引すると
状態生成が3倍になり、しかも「モードごとに別の X を見た」比較になる。

**代表条件は 3-B (IPC 掃引) の格子の中央点**である (``comparison_condition``)。
掃引の格子から選ぶので ``capacity.csv`` に必ず同じ条件の行が在り、既定モード
(``config.ipc.threshold_mode``) の値が本番成果物と一致することを機械照合できる
—— 比較のためだけの別条件を作ると、この照合ができず「比較表の数字が本番と
無関係」を検出できない。

**MC は ``chi2`` を持たない**。MC の目標は次数1のみで周辺分布が1種類しか
無く、サロゲートで足りるため ``SUPPORTED_THRESHOLD_MODES`` が
``(surrogate, none)`` の2つである (``diagnostics/memory_capacity.py``)。
比較は診断ごとに**その診断が受理するモードだけ**を回す。片方に無いモードを
``None`` や 0.0 で埋めると、成果物を読む側が「chi2 を課したら MC が 0 になった」
と読める形になる。
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass

from rc_basics_lab.config import Capacity03Config

# 定数はモジュールのフルパスから直接引く (D-52 の「正規の入手経路」)。
# 04a T2 以前は ``from rc_basics_lab.diagnostics import ipc`` が
# ``diagnostics/__init__.py`` の再エクスポートした**関数** ipc を返しており、
# フルパスで書くのはその罠を避けるためでもあった (仕様 §10-1)。D-52 で
# 再エクスポートを外したので ``diagnostics.ipc`` はモジュールを指すが、
# 関数・定数の入手経路はフルパス1本のままにする。
from rc_basics_lab.diagnostics.ipc import THRESHOLD_CHI2 as IPC_THRESHOLD_CHI2
from rc_basics_lab.diagnostics.ipc import THRESHOLD_NONE as IPC_THRESHOLD_NONE
from rc_basics_lab.diagnostics.ipc import THRESHOLD_SURROGATE as IPC_THRESHOLD_SURROGATE
from rc_basics_lab.diagnostics.memory_capacity import (
    THRESHOLD_NONE as MC_THRESHOLD_NONE,
)
from rc_basics_lab.diagnostics.memory_capacity import (
    THRESHOLD_SURROGATE as MC_THRESHOLD_SURROGATE,
)
from rc_basics_lab.experiment.capacity import (
    EXPERIMENT_IPC_SWEEP,
    capacity_context,
    ipc_config_for,
    measure_capacity,
    simulate_condition_trajectory,
)
from rc_basics_lab.experiment.capacity_rows import CapacityCondition
from rc_basics_lab.experiment.report import DataclassSummaryMixin

logger = logging.getLogger(__name__)

MC_THRESHOLD_MODES: tuple[str, ...] = (
    MC_THRESHOLD_NONE,
    MC_THRESHOLD_SURROGATE,
)
"""MC で比較するしきい値法 (``diagnostics.memory_capacity`` が受理する全モード)。

並びは「しきい値なし -> 既定」で、表の1行目が上限 (生の容量) になる。
``SUPPORTED_THRESHOLD_MODES`` と過不足なく一致することを配線テストが固定する
(診断側にモードが増えたら比較表にも出す)。
"""

IPC_THRESHOLD_MODES: tuple[str, ...] = (
    IPC_THRESHOLD_NONE,
    IPC_THRESHOLD_SURROGATE,
    IPC_THRESHOLD_CHI2,
)
"""IPC で比較するしきい値法 (``diagnostics.ipc`` が受理する全モード)。"""


@dataclass(frozen=True, slots=True)
class McThresholdRow(DataclassSummaryMixin):
    """MC の総容量をしきい値法ごとに並べた1行。

    Attributes:
        threshold_mode: ``MC_THRESHOLD_MODES`` のいずれか。
        mc_total: しきい値**後**の総容量 (モードによって変わる量)。
        mc_total_raw: しきい値**前**の総容量 (モードに依存しない)。
        mc_threshold: 課したしきい値 (``none`` は 0.0)。
        mc_effective_delay: 容量重心 (実効的な記憶長)。
        mc_ratio: ``mc_total / n_units``。
    """

    threshold_mode: str
    mc_total: float
    mc_total_raw: float
    mc_threshold: float
    mc_effective_delay: float
    mc_ratio: float


@dataclass(frozen=True, slots=True)
class IpcThresholdRow(DataclassSummaryMixin):
    """IPC の総容量をしきい値法ごとに並べた1行。

    Attributes:
        threshold_mode: ``IPC_THRESHOLD_MODES`` のいずれか。
        ipc_total: しきい値**後**の総容量。
        ipc_total_raw: しきい値**前**の総容量 (モードに依存しない)。
        ipc_linear: 次数1の取り分 (しきい値後)。
        ipc_nonlinear: 次数2以上の取り分 (しきい値後)。
        ipc_saturation_ratio: ``ipc_total / n_units`` (保存則の上限に対する割合)。
        n_targets_kept: しきい値を超えた目標の本数。
        ipc_threshold_degree1: 次数1に課したしきい値。``surrogate`` は次数ごとに
            違う値を持つが、``chi2`` は全次数で同じ値になる (次数に依存しない
            近似なので)。表に載せるのは次数1の1本だけにし、次数ごとの内訳は
            ``capacity_profile.csv`` の ``threshold`` 列 (D-38) が持つ。
    """

    threshold_mode: str
    ipc_total: float
    ipc_total_raw: float
    ipc_linear: float
    ipc_nonlinear: float
    ipc_saturation_ratio: float
    n_targets_kept: int
    ipc_threshold_degree1: float


@dataclass(frozen=True, slots=True)
class ThresholdComparison:
    """しきい値法の比較 (``meta.json`` の ``threshold_comparison``)。

    Attributes:
        condition: 測った代表条件 (``comparison_condition``)。
        memory_capacity: MC の行 (``MC_THRESHOLD_MODES`` と同じ並び)。
        ipc: IPC の行 (``IPC_THRESHOLD_MODES`` と同じ並び)。
        default_mc_mode: 本番設定の MC の既定モード。表のどの行が実際に
            ``capacity.csv`` を作ったのかを成果物だけで特定できるようにする。
        default_ipc_mode: 本番設定の IPC の既定モード。
        wall_time_s: 比較に要した実測時間 [秒] (軌道生成 + 5回の診断)。
    """

    condition: CapacityCondition
    memory_capacity: tuple[McThresholdRow, ...]
    ipc: tuple[IpcThresholdRow, ...]
    default_mc_mode: str
    default_ipc_mode: str
    wall_time_s: float

    def to_summary(self) -> dict[str, object]:
        """``meta.json`` に載せるプレーンな dict。"""
        return {
            "condition": dataclasses.asdict(self.condition),
            "default_mc_mode": self.default_mc_mode,
            "default_ipc_mode": self.default_ipc_mode,
            "memory_capacity": [row.to_summary() for row in self.memory_capacity],
            "ipc": [row.to_summary() for row in self.ipc],
            "wall_time_s": self.wall_time_s,
        }


def comparison_condition(config: Capacity03Config) -> CapacityCondition:
    """比較に使う代表条件 = **3-B (IPC 掃引) の格子の中央点、レプリケート0**。

    中央は ``grid[len(grid) // 2]`` で、点数が偶数なら後ろ寄りを採る
    (本番格子では ``rho=0.95`` / ``leak_rate=0.6``)。行の並び順や実行順に
    依存しない決め方であればよく、値そのものに主張は無い。**掃引の格子から
    選ぶ**ことだけが設計判断で、そのおかげで ``capacity.csv`` に同じ条件の
    行が必ず在り、既定モードの値が本番成果物と一致することを照合できる。

    3-B を選ぶのは、しきい値が最も効くのが IPC (目標数 601 本、次数ごとに
    閾値を推定する) であり、その次数分解が受け入れ条件4 の図の中身でもある
    ためである。3-A (N=200 / T=2e4) は次数1しか無く、3-B' (T=2e5・目標
    4,075 本) は同じ主張のために3倍以上の時間を使う。
    """
    section = config.ipc_sweep
    return CapacityCondition(
        experiment=EXPERIMENT_IPC_SWEEP,
        rho=section.rho_grid[len(section.rho_grid) // 2],
        leak_rate=section.leak_rate_grid[len(section.leak_rate_grid) // 2],
        n_units=section.n_units,
        state_noise=0.0,
        sigma_u=section.sigma_u,
        n_steps=section.n_steps,
        replicate=0,
    )


def run_threshold_comparison(config: Capacity03Config) -> ThresholdComparison:
    """代表条件を1回だけ回し、しきい値法だけを振り直して総容量を比べる。

    ``simulate_condition_trajectory`` (D-34/HIGH-1 の上限検査つき) と
    ``capacity_context`` / ``measure_capacity`` を経由するので、確保より前の
    上限検査 (``_validate_condition_bounds``)・read-only 化 (D-35)・``ctx`` の
    共有 (D-37) は掃引とまったく同じ規律で効く (F-3b2-1-001/HIGH-1: 以前は
    この関数だけ上限検査を1回も呼ばずに素通りしていた)。``ctx`` は掃引と同じ
    作り方の1個を5回の診断すべてで共有する —— サロゲートのしきい値を別シード
    で引くと、モード間の差にしきい値の推定ノイズが独立に乗る。

    MC と IPC のモード数が違う (MC は ``chi2`` 非対応) ため、両者を同時に振らず
    「MC を振る間 IPC は既定」「IPC を振る間 MC は既定」とはしない。**組み合わせ
    ごとに ``measure_capacity`` を呼び、使うのは振っている側の結果だけ**にする
    (呼び出し回数は ``max(len(MC_THRESHOLD_MODES), len(IPC_THRESHOLD_MODES))``)。
    診断2本は独立なので、片方のモードがもう片方の結果を変えることはない。

    Args:
        config: 03 の設定。

    Returns:
        代表条件と、しきい値法ごとの総容量。

    Raises:
        ValueError: ``n_units`` / ``n_units * n_steps`` が上限を超える
            (確保より前に検査する) / 診断層が投げるもの (系列が短すぎる /
            ``ctx.seed`` が無い等)。
    """
    started = time.perf_counter()
    condition = comparison_condition(config)
    reference = simulate_condition_trajectory(config, condition)
    ctx = capacity_context(config)
    base_ipc_cfg = ipc_config_for(config, condition.experiment)

    mc_rows: list[McThresholdRow] = []
    ipc_rows: list[IpcThresholdRow] = []
    n_runs = max(len(MC_THRESHOLD_MODES), len(IPC_THRESHOLD_MODES))
    for index in range(n_runs):
        mc_mode = MC_THRESHOLD_MODES[min(index, len(MC_THRESHOLD_MODES) - 1)]
        ipc_mode = IPC_THRESHOLD_MODES[min(index, len(IPC_THRESHOLD_MODES) - 1)]
        measurement = measure_capacity(
            reference.states,
            reference.drive,
            ctx=ctx,
            mc_cfg=dataclasses.replace(config.mc, threshold_mode=mc_mode),
            ipc_cfg=dataclasses.replace(base_ipc_cfg, threshold_mode=ipc_mode),
        )
        if index < len(MC_THRESHOLD_MODES):
            mc_rows.append(_mc_row(mc_mode, measurement.mc.scalars))
        if index < len(IPC_THRESHOLD_MODES):
            ipc_rows.append(
                _ipc_row(ipc_mode, measurement.ipc.scalars, measurement.ipc_thresholds)
            )

    comparison = ThresholdComparison(
        condition=condition,
        memory_capacity=tuple(mc_rows),
        ipc=tuple(ipc_rows),
        default_mc_mode=config.mc.threshold_mode,
        default_ipc_mode=base_ipc_cfg.threshold_mode,
        wall_time_s=time.perf_counter() - started,
    )
    logger.info(
        "しきい値法の比較 (rho=%.2f leak=%.2f N=%d T=%d): "
        "MC %s / IPC %s (wall_time=%.2fs)",
        condition.rho,
        condition.leak_rate,
        condition.n_units,
        condition.n_steps,
        ", ".join(f"{row.threshold_mode}={row.mc_total:.3f}" for row in mc_rows),
        ", ".join(f"{row.threshold_mode}={row.ipc_total:.3f}" for row in ipc_rows),
        comparison.wall_time_s,
    )
    return comparison


def _mc_row(mode: str, scalars: Mapping[str, float]) -> McThresholdRow:
    """MC の scalars から比較行を作る (キー名をここ1か所に閉じる)。"""
    return McThresholdRow(
        threshold_mode=mode,
        mc_total=float(scalars["mc_total"]),
        mc_total_raw=float(scalars["mc_total_raw"]),
        mc_threshold=float(scalars["mc_threshold"]),
        mc_effective_delay=float(scalars["mc_effective_delay"]),
        mc_ratio=float(scalars["mc_ratio"]),
    )


def _ipc_row(
    mode: str, scalars: Mapping[str, float], thresholds: tuple[float, ...]
) -> IpcThresholdRow:
    """IPC の scalars から比較行を作る。

    ``thresholds`` は次数の昇順 (``CapacityMeasurement.ipc_thresholds``)。
    次数1が無い設定は存在しない (``max_delay_by_degree`` が空なら診断側が
    落ちる) が、空タプルを黙って 0.0 と読むと「しきい値を課していない」と
    区別できなくなるので ``IndexError`` のまま通す。
    """
    return IpcThresholdRow(
        threshold_mode=mode,
        ipc_total=float(scalars["ipc_total"]),
        ipc_total_raw=float(scalars["ipc_total_raw"]),
        ipc_linear=float(scalars["ipc_linear"]),
        ipc_nonlinear=float(scalars["ipc_nonlinear"]),
        ipc_saturation_ratio=float(scalars["saturation_ratio"]),
        n_targets_kept=int(scalars["n_targets_kept"]),
        ipc_threshold_degree1=float(thresholds[0]),
    )


__all__ = [
    "IPC_THRESHOLD_MODES",
    "MC_THRESHOLD_MODES",
    "IpcThresholdRow",
    "McThresholdRow",
    "ThresholdComparison",
    "comparison_condition",
    "run_threshold_comparison",
]

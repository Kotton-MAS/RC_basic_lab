"""実験 3-Th: 閾値の選び方が梯子の結論を作っていないか (D-143).

容量はサロゲートのしきい値で切ってから足す (D-27)。既定の
``n_surrogates=100`` / ``surrogate_quantile=0.99`` は**判定基準そのもの**
なので、値によって水準の順位が動くなら、梯子 (D-138〜D-142) が語っているのは
現象ではなく閾値の選び方である。02 の ``abs_tol`` 感度掃引
(``experiment/threshold.py``) と同じ形を容量側でも置く。

``capacity_threshold.py`` との違いは2つある:

- あちらは**モード** (``surrogate`` / ``chi2`` / ``none``) を代表条件1つで
  比べる。こちらは**サロゲートの本数と分位点**を振る
- あちらが見るのは総容量の絶対値、こちらが見るのは**水準間の順位**

**軌道は条件ごとに1回しか作らない。** しきい値は診断の中で容量を切る段だけを
変えるので、状態を1回作って判定だけをやり直せる (仕様 §5 の禁止構造
「条件ごとに X を2回作る」)。
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, fields

from rc_basics_lab.config import Capacity03Config, LadderThresholdConfig
from rc_basics_lab.diagnostics.base import DiagnosticContext
from rc_basics_lab.diagnostics.ipc import ipc
from rc_basics_lab.diagnostics.memory_capacity import memory_capacity
from rc_basics_lab.experiment.capacity import (
    capacity_context,
    drive_config_for,
    ipc_config_for,
    reservoir_config_for,
)
from rc_basics_lab.experiment.esp import simulate_reference_trajectory
from rc_basics_lab.experiment.topology_ladder import (
    _ladder_condition,
    level_name,
    matched_levels,
)
from rc_basics_lab.reservoir.topology import TopologyConfig
from rc_basics_lab.types import FloatArray

logger = logging.getLogger(__name__)

EXPERIMENT_LADDER_THRESHOLD = "3Th_ladder_threshold"
"""``capacity_topology_threshold.csv`` の ``experiment`` 列。"""

THRESHOLD_NONE = "none"
"""しきい値なし (生の容量) を表す ``threshold_mode``。"""

THRESHOLD_SURROGATE = "surrogate"
"""シャッフルサロゲートのしきい値 (既定、D-27)。"""


@dataclass(frozen=True, slots=True)
class ThresholdSetting:
    """1つの判定基準 (**MC と IPC に同じものを課す**)。

    片方だけ変えると「順位が動いたのは MC の閾値か IPC の閾値か」を分けられ
    ない。分けたいのは閾値と現象であって、2つの診断ではない。

    Attributes:
        mode: ``none`` か ``surrogate``。
        n_surrogates: サロゲート本数 (``none`` では使わないが記録は残す)。
        quantile: しきい値に使う分位点 (同上)。
    """

    mode: str
    n_surrogates: int
    quantile: float


@dataclass(frozen=True, slots=True)
class LadderThresholdRow:
    """``capacity_topology_threshold.csv`` の1行。**宣言順が CSV の列順**。

    Attributes:
        experiment: ``EXPERIMENT_LADDER_THRESHOLD``。
        threshold_mode: ``none`` か ``surrogate``。
        n_surrogates: サロゲート本数。
        surrogate_quantile: しきい値の分位点。
        level: 梯子の水準の名前。
        graph: グラフの実現値の番号。
        replicate: 重みの実現値の番号。
        mc_total: 線形メモリ容量 (しきい値後)。
        ipc_total: 総容量 (しきい値後)。
        ipc_total_raw: しきい値前の総容量 (**設定によらず同じはず**)。
        ipc_linear: 次数1の容量。
        ipc_nonlinear: 次数2以上の容量。
        wall_time_s: 実測 wall time [秒]。
    """

    experiment: str
    threshold_mode: str
    n_surrogates: int
    surrogate_quantile: float
    level: str
    graph: int
    replicate: int
    mc_total: float
    ipc_total: float
    ipc_total_raw: float
    ipc_linear: float
    ipc_nonlinear: float
    wall_time_s: float


LADDER_THRESHOLD_CSV_COLUMNS: tuple[str, ...] = tuple(
    item.name for item in fields(LadderThresholdRow)
)
"""``capacity_topology_threshold.csv`` の列順 (宣言順)。"""


def threshold_settings(section: LadderThresholdConfig) -> tuple[ThresholdSetting, ...]:
    """格子を判定基準の並びに展開する (D-143)。

    ``none`` を先頭に置く —— しきい値前の値が基準で、そこからどれだけ削れる
    かを読むためである。``none`` でも本数と分位点を記録に残すのは、CSV の
    列が設定によって欠けないようにするため (欠けると読む側が「その行だけ
    別の実験」と読める)。

    Args:
        section: 3-Th の設定。

    Returns:
        判定基準の並び。

    Raises:
        ValueError: 格子が空の場合。
    """
    if not section.n_surrogates_grid or not section.quantile_grid:
        raise ValueError("しきい値の格子が空です")
    settings: list[ThresholdSetting] = []
    if section.include_no_threshold:
        settings.append(
            ThresholdSetting(
                mode=THRESHOLD_NONE,
                n_surrogates=section.n_surrogates_grid[0],
                quantile=section.quantile_grid[0],
            )
        )
    settings.extend(
        ThresholdSetting(mode=THRESHOLD_SURROGATE, n_surrogates=n, quantile=q)
        for n in section.n_surrogates_grid
        for q in section.quantile_grid
    )
    return tuple(settings)


def run_ladder_threshold(config: Capacity03Config) -> tuple[LadderThresholdRow, ...]:
    """閾値感度を回す (3-Th)。**判定基準 x 水準 x グラフ x 重み**の全件。

    ``ctx`` は全条件で1個を共有する (D-37 の共通乱数法)。軌道は
    (水準, グラフ, 重み) ごとに1回だけ作り、判定基準の数だけ診断をやり直す。

    Args:
        config: 03 の設定 (``ladder_threshold`` と ``topology_ladder`` を読む)。

    Returns:
        行の並び (水準 -> グラフ -> 重み -> 判定基準 の順)。
    """
    section = config.ladder_threshold
    ladder = config.topology_ladder
    ctx = capacity_context(config)
    settings = threshold_settings(section)
    rows: list[LadderThresholdRow] = []
    for topology in matched_levels(ladder.levels, ladder.n_units):
        name = level_name(topology)
        for graph in range(section.n_graphs):
            for replicate in range(section.n_replicates):
                states, drive = _trajectory(config, topology, graph, replicate)
                rows.extend(
                    _measure_at_threshold(
                        config, setting, name, graph, replicate, states, drive, ctx
                    )
                    for setting in settings
                )
    logger.info(
        "3-Th: 判定基準=%d x 水準=%d x グラフ=%d x 重み=%d = %d 行",
        len(settings),
        len(ladder.levels),
        section.n_graphs,
        section.n_replicates,
        len(rows),
    )
    return tuple(rows)


def _trajectory(
    config: Capacity03Config,
    topology: TopologyConfig,
    graph: int,
    replicate: int,
) -> tuple[FloatArray, FloatArray]:
    """1条件ぶんの状態と駆動入力 (**判定基準ごとに作り直さない**)。"""
    ladder = config.topology_ladder
    condition = _ladder_condition(ladder, replicate)
    trajectory = simulate_reference_trajectory(
        reservoir_config_for(config, condition),
        drive_config_for(config, condition),
        reservoir_seed=config.seeds.reservoir,
        drive_seed=config.seeds.drive,
        rho=condition.rho,
        leak_rate=condition.leak_rate,
        sigma_u=condition.sigma_u,
        replicate=replicate,
        state_noise=condition.state_noise,
        topology=topology,
        graph_replicate=graph,
    )
    states = trajectory.states
    states.flags.writeable = False
    return states, trajectory.drive


def _measure_at_threshold(
    config: Capacity03Config,
    setting: ThresholdSetting,
    name: str,
    graph: int,
    replicate: int,
    states: FloatArray,
    drive: FloatArray,
    ctx: DiagnosticContext,
) -> LadderThresholdRow:
    started = time.perf_counter()
    mc_cfg = dataclasses.replace(
        config.mc,
        threshold_mode=setting.mode,
        n_surrogates=setting.n_surrogates,
        surrogate_quantile=setting.quantile,
    )
    ipc_cfg = dataclasses.replace(
        ipc_config_for(config, EXPERIMENT_LADDER_THRESHOLD),
        threshold_mode=setting.mode,
        n_surrogates=setting.n_surrogates,
        surrogate_quantile=setting.quantile,
    )
    mc = memory_capacity(states, drive, ctx=ctx, cfg=mc_cfg)
    capacity = ipc(states, drive, ctx=ctx, cfg=ipc_cfg)
    return LadderThresholdRow(
        experiment=EXPERIMENT_LADDER_THRESHOLD,
        threshold_mode=setting.mode,
        n_surrogates=setting.n_surrogates,
        surrogate_quantile=setting.quantile,
        level=name,
        graph=graph,
        replicate=replicate,
        mc_total=mc.scalars["mc_total"],
        ipc_total=capacity.scalars["ipc_total"],
        ipc_total_raw=capacity.scalars["ipc_total_raw"],
        ipc_linear=capacity.scalars["ipc_linear"],
        ipc_nonlinear=capacity.scalars["ipc_nonlinear"],
        wall_time_s=time.perf_counter() - started,
    )


__all__ = [
    "EXPERIMENT_LADDER_THRESHOLD",
    "LADDER_THRESHOLD_CSV_COLUMNS",
    "LadderThresholdRow",
    "ThresholdSetting",
    "run_ladder_threshold",
    "threshold_settings",
]

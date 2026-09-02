"""実験 3-T: 交絡を1つずつ剥がす対照の梯子 (D-138).

「スケールフリーは記憶容量に効くか」は**既に主張がある問い**なので、BA を足して
MC を測るだけでは追試にしかならない。同じ密度で Erdos-Renyi と Barabasi-Albert
を比べると、次数分布のほかに相互結合率と自己ループも同時に動く (D-131)。
**1つだけ動かした水準**を並べて、何が効いているのかを分ける。

本命は次数保存ランダム化 (D-135) との比較である:

- BA の優位が**残る** -> 効いているのは次数分布 (先行の主張が支持される)
- **消える** -> 効いていたのは生成過程が作る別の何か

グラフと重みを入れ子にするのは、トポロジ比較に2種類の分散があるからである
(グラフの実現値 / 重みの実現値)。片方だけを振ると「BA が良い」がグラフ間分散に
埋もれているかを判定できない。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, fields

from rc_basics_lab.config import Capacity03Config, TopologyLadderConfig
from rc_basics_lab.diagnostics.base import DiagnosticContext
from rc_basics_lab.diagnostics.ipc import ipc
from rc_basics_lab.diagnostics.memory_capacity import memory_capacity
from rc_basics_lab.experiment.capacity import (
    capacity_context,
    drive_config_for,
    ipc_config_for,
    reservoir_config_for,
)
from rc_basics_lab.experiment.capacity_rows import CapacityCondition
from rc_basics_lab.experiment.esp import simulate_reference_trajectory
from rc_basics_lab.reservoir.topology import TopologyConfig, nominal_density

logger = logging.getLogger(__name__)

EXPERIMENT_TOPOLOGY_LADDER = "3T_topology_ladder"
"""``capacity_topology.csv`` の ``experiment`` 列。"""


@dataclass(frozen=True, slots=True)
class TopologyLadderRow:
    """``capacity_topology.csv`` の1行。**宣言順が CSV の列順の単一の真実**。

    Attributes:
        experiment: ``EXPERIMENT_TOPOLOGY_LADDER``。
        level: 梯子の水準の名前 (``erdos_renyi`` / ``control_symmetric`` など)。
        topology_kind: トポロジの判別子 (``level`` は同じ kind を区別する)。
        graph: グラフの実現値の番号 (topology ストリーム)。
        replicate: 重みの実現値の番号 (reservoir ストリーム)。
        n_units: リザバーのユニット数 N。
        n_steps: 系列長 [ステップ]。
        rho: スペクトル半径。
        leak_rate: リーク率。
        sigma_u: 駆動信号の標準偏差。
        nominal_density: 設定から見込まれる密度 (実測ではない)。
        mc_total: 線形メモリ容量。
        ipc_total: しきい値後の総容量。
        ipc_linear: 次数1の容量。
        ipc_nonlinear: 次数2以上の容量。
        wall_time_s: 実測 wall time [秒]。
    """

    experiment: str
    level: str
    topology_kind: str
    graph: int
    replicate: int
    n_units: int
    n_steps: int
    rho: float
    leak_rate: float
    sigma_u: float
    nominal_density: float
    mc_total: float
    ipc_total: float
    ipc_linear: float
    ipc_nonlinear: float
    wall_time_s: float


TOPOLOGY_LADDER_CSV_COLUMNS: tuple[str, ...] = tuple(
    item.name for item in fields(TopologyLadderRow)
)
"""``capacity_topology.csv`` の列順 (``TopologyLadderRow`` の宣言順)。"""


def level_name(topology: TopologyConfig) -> str:
    """水準の名前 (**同じ kind の水準を区別する**)。

    対照 (``control``) は「何を変えたか」で名前が分かれていないと、行を見ても
    対称化なのか自己ループ除去なのか分からない。
    """
    kind = type(topology).KIND
    symmetrize = getattr(topology, "symmetrize", False)
    drop_self_loops = getattr(topology, "drop_self_loops", False)
    if kind != "control":
        return kind
    if symmetrize and drop_self_loops:
        return "control_symmetric_no_self"
    if symmetrize:
        return "control_symmetric"
    if drop_self_loops:
        return "control_no_self"
    return "control_identity"


def _ladder_condition(
    section: TopologyLadderConfig, replicate: int
) -> CapacityCondition:
    return CapacityCondition(
        experiment=EXPERIMENT_TOPOLOGY_LADDER,
        rho=section.rho,
        leak_rate=section.leak_rate,
        n_units=section.n_units,
        state_noise=0.0,
        sigma_u=section.sigma_u,
        n_steps=section.n_steps,
        replicate=replicate,
    )


def run_topology_ladder(config: Capacity03Config) -> tuple[TopologyLadderRow, ...]:
    """梯子を回す (3-T)。**水準 x グラフ x 重み**の全件を返す。

    ``ctx`` は全条件で1個を共有する (D-37 の共通乱数法)。しきい値の推定ノイズが
    条件間の差に独立に乗ると、水準間の差がノイズか実体かを分離できなくなる。

    グラフは topology ストリームから引くので、**同じ重み行列を水準ごとに違う
    マスクで切り出す**ことになる (D-134)。ペアが組めるので、水準差を対応のある
    検定で見られる。

    Args:
        config: 03 の設定 (``topology_ladder`` セクションを読む)。

    Returns:
        行の並び (水準 -> グラフ -> 重み の順)。

    Raises:
        ValueError: 確保軸を超える設定、または診断側の値域違反。
    """
    section = config.topology_ladder
    ctx = capacity_context(config)
    rows: list[TopologyLadderRow] = []
    for topology in section.levels:
        name = level_name(topology)
        for graph in range(section.n_graphs):
            for replicate in range(section.n_replicates):
                rows.append(
                    _measure(config, section, topology, name, graph, replicate, ctx)
                )
    logger.info(
        "3-T: 水準=%d x グラフ=%d x 重み=%d = %d 行",
        len(section.levels),
        section.n_graphs,
        section.n_replicates,
        len(rows),
    )
    return tuple(rows)


def _measure(
    config: Capacity03Config,
    section: TopologyLadderConfig,
    topology: TopologyConfig,
    name: str,
    graph: int,
    replicate: int,
    ctx: DiagnosticContext,
) -> TopologyLadderRow:
    started = time.perf_counter()
    condition = _ladder_condition(section, replicate)
    trajectory = simulate_reference_trajectory(
        reservoir_config_for(config, condition),
        drive_config_for(config, condition),
        reservoir_seed=config.seeds.reservoir,
        drive_seed=config.seeds.drive,
        rho=condition.rho,
        leak_rate=condition.leak_rate,
        sigma_u=condition.sigma_u,
        replicate=replicate,
        topology=topology,
        graph_replicate=graph,
    )
    states = trajectory.states
    states.flags.writeable = False
    mc = memory_capacity(states, trajectory.drive, ctx=ctx, cfg=config.mc)
    capacity = ipc(
        states,
        trajectory.drive,
        ctx=ctx,
        cfg=ipc_config_for(config, EXPERIMENT_TOPOLOGY_LADDER),
    )
    return TopologyLadderRow(
        experiment=EXPERIMENT_TOPOLOGY_LADDER,
        level=name,
        topology_kind=type(topology).KIND,
        graph=graph,
        replicate=replicate,
        n_units=section.n_units,
        n_steps=section.n_steps,
        rho=section.rho,
        leak_rate=section.leak_rate,
        sigma_u=section.sigma_u,
        nominal_density=nominal_density(topology, section.n_units),
        mc_total=mc.scalars["mc_total"],
        ipc_total=capacity.scalars["ipc_total"],
        ipc_linear=capacity.scalars["ipc_linear"],
        ipc_nonlinear=capacity.scalars["ipc_nonlinear"],
        wall_time_s=time.perf_counter() - started,
    )


__all__ = [
    "EXPERIMENT_TOPOLOGY_LADDER",
    "TOPOLOGY_LADDER_CSV_COLUMNS",
    "TopologyLadderRow",
    "level_name",
    "run_topology_ladder",
]

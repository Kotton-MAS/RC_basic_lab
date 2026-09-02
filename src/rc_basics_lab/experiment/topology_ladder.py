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
from rc_basics_lab.reservoir.axes import require_axes, with_axis
from rc_basics_lab.reservoir.topology import (
    TopologyConfig,
    nominal_density,
    rescaled_to_density,
)

logger = logging.getLogger(__name__)

EXPERIMENT_TOPOLOGY_LADDER = "3T_topology_ladder"
"""``capacity_topology.csv`` の ``experiment`` 列。"""


@dataclass(frozen=True, slots=True)
class TopologyLadderRow:
    """``capacity_topology.csv`` の1行。**宣言順が CSV の列順の単一の真実**。

    Attributes:
        experiment: ``EXPERIMENT_TOPOLOGY_LADDER``。
        sweep_axis: この行が属する掃引の軸名 (掃引なしなら空文字)。
        level: 梯子の水準の名前 (``erdos_renyi`` / ``control_symmetric`` など)。
        topology_kind: トポロジの判別子 (``level`` は同じ kind を区別する)。
        graph: グラフの実現値の番号 (topology ストリーム)。
        replicate: 重みの実現値の番号 (reservoir ストリーム)。
        n_units: リザバーのユニット数 N。
        n_steps: 系列長 [ステップ]。
        rho: スペクトル半径。
        leak_rate: リーク率。
        sigma_u: 駆動信号の標準偏差。
        state_noise: 状態ノイズの標準偏差。
        nominal_density: 設定から見込まれる密度 (実測ではない)。
        mc_total: 線形メモリ容量。
        ipc_total: しきい値後の総容量。
        ipc_linear: 次数1の容量。
        ipc_nonlinear: 次数2以上の容量。
        wall_time_s: 実測 wall time [秒]。
    """

    experiment: str
    sweep_axis: str
    level: str
    topology_kind: str
    graph: int
    replicate: int
    n_units: int
    n_steps: int
    rho: float
    leak_rate: float
    sigma_u: float
    state_noise: float
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


DESIGN_AXES: frozenset[str] = frozenset({"n_graphs", "n_replicates"})
"""掃引で振ってはいけない軸 (D-139)。

対の本数を決めるフィールドである。これを振ると掃引点ごとに対の数が変わり、
**対応のある検定が組めなくなる** —— 水準差を見るための入れ子設計そのものが
掃引点によって変わってしまう。
"""


def matched_levels(
    levels: tuple[TopologyConfig, ...], n_units: int
) -> tuple[TopologyConfig, ...]:
    """全水準の見込み密度をそろえた水準列を返す (D-139)。

    **密度を指定できない水準が、密度を決める側になる。** BA は枝数 m が整数
    なので密度が ``2m/N`` に固定される。N を掃引する以上、YAML に固定値を
    書く方式では N=50 でしかそろわない (実測: N=25 で BA が ER の2倍、
    N=100 で半分)。そろっていないと「密度が違うから容量が違う」という
    **一番つまらない交絡**が最初に効いてしまい、梯子が答えたい問いに届かない。

    指定できない水準が複数あるなら、それらは互いにそろっていなければならない
    (BA と、その次数列を借りる次数保存ランダム化は定義上そろう)。1つも無ければ
    先頭の水準の密度に合わせる。

    Args:
        levels: 梯子の水準。
        n_units: そのときのユニット数 N。

    Returns:
        密度をそろえた水準列 (``levels`` と同じ並び)。

    Raises:
        ValueError: 空、または密度を指定できない水準どうしが食い違う場合。
    """
    if not levels:
        raise ValueError("梯子の水準が空です")
    pinned = tuple(
        level
        for level in levels
        if rescaled_to_density(level, nominal_density(level, n_units)) is None
    )
    if pinned:
        targets = {round(nominal_density(level, n_units), 12) for level in pinned}
        if len(targets) != 1:
            raise ValueError(
                f"密度を指定できない水準どうしが N={n_units} で食い違います: "
                f"{sorted(targets)} "
                f"({', '.join(level_name(level) for level in pinned)})"
            )
        target = targets.pop()
    else:
        target = nominal_density(levels[0], n_units)
    return tuple(rescaled_to_density(level, target) or level for level in levels)


def sweep_points(
    section: TopologyLadderConfig,
) -> tuple[tuple[str, TopologyLadderConfig], ...]:
    """掃引を (軸名, その点の設定) の並びに展開する (D-139).

    掃引が無ければ基準の1点だけを ``("", section)`` として返す。掃引が複数
    あれば宣言順に連結する —— 別々の CSV に分けると、基準点が2つのファイルに
    散らばって「同じ条件を測っているのか」を読者が確かめられなくなる。

    Args:
        section: 梯子の設定。

    Returns:
        ``(軸名, 設定)`` の並び。

    Raises:
        ValueError: 存在しない軸、または ``DESIGN_AXES`` を振ろうとした場合。
    """
    points: list[tuple[str, TopologyLadderConfig]] = []
    for sweep in section.sweeps:
        if not sweep.values:
            continue
        if sweep.axis in DESIGN_AXES:
            raise ValueError(
                f"軸 {sweep.axis!r} は掃引できません (対の本数が変わるため。D-139)。"
                f" 振れないのは {sorted(DESIGN_AXES)}"
            )
        require_axes(section, (sweep.axis,), "実験3-T (対照の梯子)")
        current = getattr(section, sweep.axis)
        points.extend(
            (sweep.axis, with_axis(section, sweep.axis, type(current)(value)))
            for value in sweep.values
        )
    return tuple(points) if points else (("", section),)


def _ladder_condition(
    section: TopologyLadderConfig, replicate: int
) -> CapacityCondition:
    return CapacityCondition(
        experiment=EXPERIMENT_TOPOLOGY_LADDER,
        rho=section.rho,
        leak_rate=section.leak_rate,
        n_units=section.n_units,
        state_noise=section.state_noise,
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
    points = sweep_points(section)
    for axis, point in points:
        for topology in matched_levels(point.levels, point.n_units):
            name = level_name(topology)
            for graph in range(point.n_graphs):
                for replicate in range(point.n_replicates):
                    rows.append(
                        _measure(
                            config, point, axis, topology, name, graph, replicate, ctx
                        )
                    )
    logger.info(
        "3-T: 掃引点=%d x 水準=%d x グラフ=%d x 重み=%d = %d 行",
        len(points),
        len(section.levels),
        section.n_graphs,
        section.n_replicates,
        len(rows),
    )
    return tuple(rows)


def _measure(
    config: Capacity03Config,
    section: TopologyLadderConfig,
    axis: str,
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
        state_noise=condition.state_noise,
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
        sweep_axis=axis,
        level=name,
        topology_kind=type(topology).KIND,
        graph=graph,
        replicate=replicate,
        n_units=section.n_units,
        n_steps=section.n_steps,
        rho=section.rho,
        leak_rate=section.leak_rate,
        sigma_u=section.sigma_u,
        state_noise=section.state_noise,
        nominal_density=nominal_density(topology, section.n_units),
        mc_total=mc.scalars["mc_total"],
        ipc_total=capacity.scalars["ipc_total"],
        ipc_linear=capacity.scalars["ipc_linear"],
        ipc_nonlinear=capacity.scalars["ipc_nonlinear"],
        wall_time_s=time.perf_counter() - started,
    )


__all__ = [
    "DESIGN_AXES",
    "EXPERIMENT_TOPOLOGY_LADDER",
    "TOPOLOGY_LADDER_CSV_COLUMNS",
    "TopologyLadderRow",
    "level_name",
    "matched_levels",
    "run_topology_ladder",
    "sweep_points",
]

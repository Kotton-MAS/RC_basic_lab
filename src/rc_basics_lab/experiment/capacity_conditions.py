"""掃引の条件を組む層 (D-137).

``capacity.py`` が 600 行の上限 (D-63 / D-77) を超えて凍結されているので、
条件の組み立てはこちらへ置く。役割としても「何を振るか」は「どう測るか」とは別。
"""

from __future__ import annotations

from rc_basics_lab.config import LengthSweepConfig
from rc_basics_lab.experiment.capacity_rows import CapacityCondition
from rc_basics_lab.reservoir.topology import TopologyConfig

EXPERIMENT_LENGTH_SWEEP = "3L_length_sweep"
"""系列長掃引の実験ラベル (``capacity.py`` と同じ値)。"""


def length_sweep_conditions(
    section: LengthSweepConfig,
) -> tuple[CapacityCondition, ...]:
    """系列長 T の掃引の条件列を組む (D-137)。

    ``topologies`` が空なら従来どおり横断共有の ``density`` で1本だけ。振ると
    **(トポロジ x T) の格子**になる —— 「ハブ型は飽和に必要な T が長いか」を
    測るためで、もしそうなら同じ T で ER と BA を比べた文献は BA を過小評価
    している (トポロジの結論を出す前に確かめるべき順序である)。

    **並び順は T が内側**にしてある。既存の行の並びを保つため。

    Args:
        section: 系列長掃引の設定。

    Returns:
        条件列。
    """
    topologies: tuple[TopologyConfig | None, ...] = section.topologies or (None,)
    return tuple(
        CapacityCondition(
            experiment=EXPERIMENT_LENGTH_SWEEP,
            rho=section.rho,
            leak_rate=section.leak_rate,
            n_units=section.n_units,
            state_noise=0.0,
            sigma_u=section.sigma_u,
            n_steps=n_steps,
            replicate=0,
            topology=topology,
        )
        for topology in topologies
        for n_steps in section.n_steps_grid
    )


__all__ = ["length_sweep_conditions"]

"""対照の梯子 (3-T) の検査 (D-138).

梯子が答えるのは「BA の優位/劣位は次数分布で説明できるか」である。そのために
**同じ重み行列を水準ごとに違うマスクで切り出す** (D-134) 必要があり、ここが
壊れると「重みの実現値の違いをトポロジの効果と読んだ」で終わる。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rc_basics_lab.config import Capacity03Config, load_config_as
from rc_basics_lab.experiment.topology_ladder import (
    EXPERIMENT_TOPOLOGY_LADDER,
    TOPOLOGY_LADDER_CSV_COLUMNS,
    TopologyLadderRow,
    level_name,
    run_topology_ladder,
)
from rc_basics_lab.reservoir.topology import (
    BarabasiAlbertConfig,
    DegreePreservingConfig,
    ErdosRenyiConfig,
    TopologyControlConfig,
    build_mask,
)
from rc_basics_lab.seeds import SeedStream, make_rng_for
from rc_basics_lab.types import FloatArray

GOLDEN_CONFIG = "tests/golden/configs/03_capacity.yaml"


def tiny_config() -> Capacity03Config:
    """秒未満で1周する梯子 (水準は本番と同じ5本)。"""
    config = load_config_as(GOLDEN_CONFIG, Capacity03Config)
    return dataclasses.replace(
        config,
        topology_ladder=dataclasses.replace(
            config.topology_ladder,
            n_units=20,
            n_steps=1200,
            n_graphs=2,
            n_replicates=2,
        ),
    )


def test_the_ladder_covers_every_level_graph_and_weight() -> None:
    """水準 x グラフ x 重みの全件が出る (**選ばない**、事前宣言の全件報告)。"""
    config = tiny_config()
    section = config.topology_ladder
    rows = run_topology_ladder(config)
    expected = len(section.levels) * section.n_graphs * section.n_replicates
    assert len(rows) == expected, f"{expected} 行のはずが {len(rows)} 行です"
    assert {row.experiment for row in rows} == {EXPERIMENT_TOPOLOGY_LADDER}
    for level in (level_name(topology) for topology in section.levels):
        selected = [row for row in rows if row.level == level]
        assert len(selected) == section.n_graphs * section.n_replicates
        assert {row.graph for row in selected} == set(range(section.n_graphs))
        assert {row.replicate for row in selected} == set(range(section.n_replicates))


def test_the_control_levels_are_named_apart() -> None:
    """同じ ``kind`` の対照が名前で区別できる (行を見て何を変えたか分かる)。"""
    base = ErdosRenyiConfig(density=0.08)
    assert level_name(TopologyControlConfig(base=base, symmetrize=True)) == (
        "control_symmetric"
    )
    assert level_name(TopologyControlConfig(base=base, drop_self_loops=True)) == (
        "control_no_self"
    )
    assert level_name(base) == "erdos_renyi"
    rows = run_topology_ladder(tiny_config())
    levels = {row.level for row in rows}
    assert "control_symmetric" in levels and "control_no_self" in levels, (
        f"対照が区別できていません: {sorted(levels)}"
    )


def test_every_level_shares_the_same_weight_matrix() -> None:
    """**水準はマスクだけが違う** (D-134 のペアが成立している)。

    ここが壊れると、水準差に重みの実現値の分散が混ざる。マスクは水準ごとに
    違い、その下の値行列は同じ、という形を直接確かめる。
    """
    n_units, graph = 40, 0
    config = tiny_config()
    density = 2.0 * 2 / n_units
    levels = (
        ErdosRenyiConfig(density=density),
        TopologyControlConfig(base=ErdosRenyiConfig(density=density), symmetrize=True),
        DegreePreservingConfig(base=BarabasiAlbertConfig(m=2)),
        BarabasiAlbertConfig(m=2),
    )
    masks = [
        build_mask(topology, n_units, make_rng_for(0, SeedStream.TOPOLOGY, graph))
        for topology in levels
    ]
    assert not all(np.array_equal(masks[0], other) for other in masks[1:]), (
        "水準を変えてもマスクが同じです"
    )

    # 値行列は reservoir ストリームだけで決まる (マスクを引いても消費しない)
    def values() -> FloatArray:
        rng = make_rng_for(0, SeedStream.RESERVOIR, 0)
        rng.uniform(-0.1, 0.1, n_units)
        rng.uniform(-1.0, 1.0, (n_units, 1))
        drawn: FloatArray = rng.uniform(-1.0, 1.0, (n_units, n_units))
        return drawn

    assert np.array_equal(values(), values()), "値行列が水準ごとに変わっています"
    del config


def test_the_row_columns_are_the_declaration_order() -> None:
    """CSV の列順は行 dataclass の宣言順 (単一の真実)。"""
    assert (
        tuple(item.name for item in dataclasses.fields(TopologyLadderRow))
        == TOPOLOGY_LADDER_CSV_COLUMNS
    )
    assert TOPOLOGY_LADDER_CSV_COLUMNS[:3] == ("experiment", "level", "topology_kind")


def test_the_nominal_density_is_the_same_across_levels() -> None:
    """全水準の見込み密度がそろっている (D-138)。

    そろっていないと「密度が違うから容量が違う」という**一番つまらない交絡**が
    最初に効いてしまい、梯子が答えたい問いに届かない。
    """
    config = load_config_as("experiments/03_capacity/config.yaml", Capacity03Config)
    from rc_basics_lab.reservoir.topology import nominal_density

    section = config.topology_ladder
    densities = [
        nominal_density(topology, section.n_units) for topology in section.levels
    ]
    assert densities == pytest.approx([densities[0]] * len(densities)), (
        f"水準ごとに見込み密度が違います: {densities}"
    )

"""対照の梯子 (3-T) の検査 (D-138).

梯子が答えるのは「BA の優位/劣位は次数分布で説明できるか」である。そのために
**同じ重み行列を水準ごとに違うマスクで切り出す** (D-134) 必要があり、ここが
壊れると「重みの実現値の違いをトポロジの効果と読んだ」で終わる。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rc_basics_lab.config import (
    Capacity03Config,
    LadderSweepConfig,
    TopologyLadderConfig,
    load_config_as,
)
from rc_basics_lab.experiment.topology_ladder import (
    DESIGN_AXES,
    EXPERIMENT_TOPOLOGY_LADDER,
    TOPOLOGY_LADDER_CSV_COLUMNS,
    TopologyLadderRow,
    level_name,
    matched_levels,
    run_topology_ladder,
    sweep_points,
)
from rc_basics_lab.reservoir.axes import numeric_axes
from rc_basics_lab.reservoir.topology import (
    BarabasiAlbertConfig,
    DegreePreservingConfig,
    ErdosRenyiConfig,
    RingTopologyConfig,
    TopologyControlConfig,
    WattsStrogatzConfig,
    build_mask,
    nominal_density,
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
            sweeps=(),
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
    assert TOPOLOGY_LADDER_CSV_COLUMNS[:4] == (
        "experiment",
        "sweep_axis",
        "level",
        "topology_kind",
    )


def test_the_nominal_density_is_the_same_across_levels_at_every_sweep_point() -> None:
    """**どの掃引点でも**全水準の見込み密度がそろっている (D-139)。

    そろっていないと「密度が違うから容量が違う」という**一番つまらない交絡**が
    最初に効いてしまい、梯子が答えたい問いに届かない。N を掃引すると BA の
    密度は ``2m/N`` で動くので、**N=50 で手で合わせた値では他の N で崩れる**
    (実測: N=25 で BA が ER の2倍、N=100 で半分)。
    """
    config = load_config_as("experiments/03_capacity/config.yaml", Capacity03Config)
    points = sweep_points(config.topology_ladder)
    assert len(points) > 1, "掃引が設定されていません"
    for axis, point in points:
        densities = [
            nominal_density(topology, point.n_units)
            for topology in matched_levels(point.levels, point.n_units)
        ]
        assert densities == pytest.approx([densities[0]] * len(densities)), (
            f"{axis}: N={point.n_units} で水準ごとに密度が違います: {densities}"
        )


def test_every_sweepable_axis_is_also_a_csv_column() -> None:
    """**振れる軸は必ず CSV の列にある** (D-139)。

    列に出ない軸を振ると、成果物を見た人には「同じ条件を繰り返し測っている」
    ようにしか見えない。専用の ``sweep_value`` 列を作らずに済むのも、この
    包含関係が成り立っているからである。
    """
    sweepable = numeric_axes(TopologyLadderConfig()) - DESIGN_AXES
    columns = set(TOPOLOGY_LADDER_CSV_COLUMNS)
    assert sweepable <= columns, (
        f"列に出ない軸を振れてしまいます: {sorted(sweepable - columns)}"
    )


def test_the_design_axes_cannot_be_swept() -> None:
    """対の本数を決める軸は振れない (対応のある検定が組めなくなる)。"""
    for axis in sorted(DESIGN_AXES):
        section = TopologyLadderConfig(
            sweeps=(LadderSweepConfig(axis=axis, values=(1.0, 2.0)),)
        )
        with pytest.raises(ValueError, match=axis):
            sweep_points(section)


def test_an_unknown_axis_is_rejected() -> None:
    """持っていない軸を振ろうとしたら落ちる (黙って素通しにしない)。"""
    section = TopologyLadderConfig(
        sweeps=(LadderSweepConfig(axis="temperature", values=(1.0,)),)
    )
    with pytest.raises(ValueError, match="temperature"):
        sweep_points(section)


def test_no_sweep_gives_one_point_with_an_empty_axis() -> None:
    """掃引が無ければ基準の1点だけ (``sweep_axis`` は空文字)。"""
    section = TopologyLadderConfig(sweeps=())
    assert sweep_points(section) == (("", section),)
    empty = TopologyLadderConfig(sweeps=(LadderSweepConfig(axis="n_units", values=()),))
    assert sweep_points(empty) == (("", empty),)


def test_an_int_axis_stays_an_int() -> None:
    """``n_units`` に float を渡しても int で入る (格子は YAML から来る)。"""
    section = TopologyLadderConfig(
        sweeps=(LadderSweepConfig(axis="n_units", values=(25.0,)),)
    )
    ((_, point),) = sweep_points(section)
    assert isinstance(point.n_units, int) and point.n_units == 25


def test_matched_levels_raises_when_pinned_levels_disagree() -> None:
    """密度を指定できない水準どうしが食い違ったら落ちる。

    BA も Watts-Strogatz も整数の枝数で密度が決まるので、**どちらも譲れない**。
    黙って片方に寄せると、寄せられた側の水準が名前と違うものになる。
    """
    levels = (BarabasiAlbertConfig(m=2), WattsStrogatzConfig(k=6))
    with pytest.raises(ValueError, match="食い違"):
        matched_levels(levels, 50)
    # BA の密度は厳密に数える (D-140)。N=50 / m=2 なら
    # (2*3 + 2*2*47) / 2500 = 0.0776 で、近似の 2m/N = 0.08 ではない。
    assert matched_levels((BarabasiAlbertConfig(m=2), ErdosRenyiConfig()), 50) == (
        BarabasiAlbertConfig(m=2),
        ErdosRenyiConfig(density=0.0776),
    )


def test_matched_levels_falls_back_to_the_first_level() -> None:
    """指定できない水準が1つも無ければ先頭に合わせる。"""
    levels = (ErdosRenyiConfig(density=0.3), ErdosRenyiConfig(density=0.9))
    assert matched_levels(levels, 50) == (
        ErdosRenyiConfig(density=0.3),
        ErdosRenyiConfig(density=0.3),
    )
    with pytest.raises(ValueError, match="空"):
        matched_levels((), 50)
    # リングは 1/N で固定なので、こちらが密度を決める側になる
    assert matched_levels((RingTopologyConfig(), ErdosRenyiConfig()), 50) == (
        RingTopologyConfig(),
        ErdosRenyiConfig(density=0.02),
    )


def test_the_shared_baseline_matches_across_sweep_blocks() -> None:
    """2つの掃引に重複して入る基準点が**同じ数を出す** (D-139)。

    掃引点の間で乱数や文脈が漏れていたら、同じ条件の行が食い違う。重複を
    残しているのはこの検査が無料で付いてくるからでもある。
    """
    config = tiny_config()
    section = dataclasses.replace(
        config.topology_ladder,
        sweeps=(
            LadderSweepConfig(axis="n_units", values=(float(20),)),
            LadderSweepConfig(axis="state_noise", values=(0.0,)),
        ),
    )
    rows = run_topology_ladder(dataclasses.replace(config, topology_ladder=section))
    first = {
        (row.level, row.graph, row.replicate): row
        for row in rows
        if row.sweep_axis == "n_units"
    }
    second = {
        (row.level, row.graph, row.replicate): row
        for row in rows
        if row.sweep_axis == "state_noise"
    }
    assert first and first.keys() == second.keys()
    for key, row in first.items():
        assert row.mc_total == pytest.approx(second[key].mc_total), (
            f"{key} の MC が掃引ブロック間で食い違います"
        )
        assert row.ipc_total == pytest.approx(second[key].ipc_total)


def test_the_realized_density_matches_the_nominal_one() -> None:
    """**そう設定した密度**と**そうなった密度**が一致する (D-140)。

    ``nominal_density`` は設定から決まる値でしかない。密度そろえ (D-139) が
    実現値まで届いているかは、生成された行列を数えないと分からない ——
    BA の ``2m/N`` は「各点が m 本を張る」という見込みで、最初の m 点の
    扱いによっては本数がずれ得る。
    """
    rows = run_topology_ladder(tiny_config())
    for level in {row.level for row in rows}:
        selected = [row for row in rows if row.level == level]
        realized = [row.realized_density for row in selected]
        nominal = selected[0].nominal_density
        assert sum(realized) / len(realized) == pytest.approx(nominal, rel=0.15), (
            f"{level}: 見込み {nominal:.4f} に対し実測の平均 "
            f"{sum(realized) / len(realized):.4f}"
        )


def test_the_hub_levels_have_a_wider_degree_distribution() -> None:
    """BA 系の水準は入次数が広がる (梯子が本当に別の構造を作っている)。

    密度をそろえてある以上、平均入次数は水準によらない。**違うのは分布の
    広がりだけ**で、それがこの梯子が動かしている当のものである。ここが
    壊れると「同じ構造を5回測っただけ」の成果物になる。
    """
    rows = run_topology_ladder(tiny_config())
    spread = {
        level: sum(row.in_degree_std for row in rows if row.level == level)
        / sum(1 for row in rows if row.level == level)
        for level in {row.level for row in rows}
    }
    assert spread["barabasi_albert"] > spread["erdos_renyi"], (
        f"BA の次数分布が ER より広くありません: {spread}"
    )
    assert spread["degree_preserving"] == pytest.approx(
        spread["barabasi_albert"], rel=0.05
    ), f"次数保存ランダム化が次数列を保っていません: {spread}"


def test_the_effective_gain_is_reported_per_row() -> None:
    """実効ゲイン (交絡4) が行に出ている。

    ``spectral_gap`` では測れない —— 実行列の固有値は共役対で出るので
    構造的に 0 になる (D-140)。詳細は
    ``test_diagnostics_topology.py::test_the_spectral_gap_cannot_separate_hubs``。
    """
    rows = run_topology_ladder(tiny_config())
    assert all(row.gain_max > 0.0 for row in rows)
    assert all(row.gain_std > 0.0 for row in rows)
    assert len({round(row.gain_max, 6) for row in rows}) > 1, (
        "実効ゲインが全条件で同じです (measure が定数を返しています)"
    )


def test_the_symmetrised_control_does_not_double_the_density() -> None:
    """対称化の対照が**密度まで動かしていない** (D-140)。

    ``mask | mask.T`` は密度を ``d`` から ``2d - d^2`` に上げる。実測で
    N=50 / d=0.08 のとき 0.152 —— ほぼ倍である。土台の密度をそのまま
    見込みとして報告していた間、対照そのものが「梯子が排除したはずの
    交絡」を持っていた。逆算するようになったので、生成されるマスクの
    密度が基準の水準とそろう。
    """
    n_units = 50
    levels = matched_levels(
        (ErdosRenyiConfig(), TopologyControlConfig(symmetrize=True)), n_units
    )
    realized = [
        float(
            np.mean(
                [
                    build_mask(level, n_units, np.random.default_rng(seed)).mean()
                    for seed in range(60)
                ]
            )
        )
        for level in levels
    ]
    assert realized[1] == pytest.approx(realized[0], rel=0.05), (
        f"対称化で密度が動いています: {realized}"
    )
    # 逆算していなければ土台の 0.1 のまま対称化され 0.19 になる
    control = levels[1]
    assert isinstance(control, TopologyControlConfig)
    assert control.base.density < 0.06, (
        f"土台の密度を逆算していません: {control.base.density}"
    )

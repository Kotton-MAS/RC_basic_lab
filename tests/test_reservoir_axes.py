"""掃引軸がモデルによらず扱えること (D-124).

## 何を測るか

1. 軸の集合が**フィールドから決まる**こと (一覧を書き下していないこと)
2. 軸を1つ差し替えると、その軸**だけ**が変わること
3. 持っていない軸を要求したら、**持っている軸を並べて**落ちること
4. 4-C の掃引が ESN 以外でも組めること (計画の狙いそのもの)
"""

from __future__ import annotations

import dataclasses

import pytest

from rc_basics_lab.reservoir.axes import (
    axis_value,
    numeric_axes,
    require_axes,
    with_axis,
)
from rc_basics_lab.reservoir.deep import DeepESNConfig
from rc_basics_lab.reservoir.esn import ESNConfig
from rc_basics_lab.reservoir.protocol import ReservoirConfig
from rc_basics_lab.reservoir.registry import reservoir_density
from rc_basics_lab.reservoir.ring import RingConfig
from rc_basics_lab.reservoir.topology import (
    BarabasiAlbertConfig,
    ErdosRenyiConfig,
)

MODELS: tuple[ReservoirConfig, ...] = (
    ESNConfig(),
    DeepESNConfig(),
    RingConfig(),
)


@pytest.mark.parametrize("config", MODELS, ids=lambda c: type(c).__name__)
def test_axes_come_from_the_fields(config: ReservoirConfig) -> None:
    """軸は数値フィールドそのもの。文字列や入れ子の設定は軸ではない。"""
    axes = numeric_axes(config)
    assert axes, f"{type(config).__name__} に軸がありません"
    for name in axes:
        node: object = config
        for part in name.split("."):
            node = getattr(node, part)
        assert isinstance(node, int | float)
    assert "activation" not in axes  # 文字列
    assert "topology" not in axes  # 入れ子そのものは軸ではない (中身が軸)


@pytest.mark.parametrize("config", MODELS, ids=lambda c: type(c).__name__)
def test_the_three_models_share_the_axes_the_stability_sweep_needs(
    config: ReservoirConfig,
) -> None:
    """4-C の3軸は3モデルとも持っている (だから同じ掃引コードで回る)。"""
    require_axes(config, ("spectral_radius", "leak_rate", "state_noise"), "テスト")


@pytest.mark.parametrize("config", MODELS, ids=lambda c: type(c).__name__)
def test_with_axis_changes_only_that_axis(config: ReservoirConfig) -> None:
    changed = with_axis(config, "leak_rate", 0.123)
    assert type(changed) is type(config)
    assert axis_value(changed, "leak_rate") == pytest.approx(0.123)
    before = dataclasses.asdict(config)
    after = dataclasses.asdict(changed)
    differing = {key for key in before if before[key] != after[key]}
    assert differing == {"leak_rate"}, f"leak_rate 以外も動きました: {differing}"


def test_an_axis_a_model_does_not_have_is_rejected_with_the_available_ones() -> None:
    with pytest.raises(ValueError, match="持っている軸") as raised:
        require_axes(RingConfig(), ("inter_layer_scale",), "4-C の掃引")
    message = str(raised.value)
    assert "inter_layer_scale" in message
    assert "RingConfig" in message
    # 「では何が振れるのか」が同じ行から読めること
    assert "spectral_radius" in message


def test_only_the_deep_model_has_the_layer_axes() -> None:
    """軸名をモデル間で共通化していないこと (集合が違ってよい)。"""
    assert "n_layers" in numeric_axes(DeepESNConfig())
    assert "n_layers" not in numeric_axes(ESNConfig())
    assert "inter_layer_scale" not in numeric_axes(RingConfig())


@pytest.mark.parametrize("config", MODELS, ids=lambda c: type(c).__name__)
def test_density_is_defined_for_every_model(config: ReservoirConfig) -> None:
    """``density`` 列がモデルによらず埋まる (RingConfig は topology を持たない)。"""
    density = reservoir_density(config)
    assert 0.0 < density <= 1.0


def test_the_ring_density_is_one_edge_per_row() -> None:
    assert reservoir_density(RingConfig(n_units=200)) == pytest.approx(1.0 / 200)


def test_axis_value_rejects_an_unknown_axis() -> None:
    with pytest.raises(ValueError, match="持っていません"):
        axis_value(ESNConfig(), "n_layers")


def test_nested_axes_reach_one_level_into_the_topology() -> None:
    """入れ子の設定へ1段だけ潜る (D-130)。

    ``density`` は ``topology`` へ移った (D-122) ので、潜れないと**分離前に
    できた掃引ができなくなる**。BA の ``m`` や WS の ``k`` / ``beta`` も同じ。
    """
    er = ESNConfig()
    assert "topology.density" in numeric_axes(er)
    assert with_axis(er, "topology.density", 0.3).topology == ErdosRenyiConfig(
        density=0.3
    )
    assert axis_value(er, "topology.density") == pytest.approx(0.1)

    ba = ESNConfig(topology=BarabasiAlbertConfig(m=2))
    assert "topology.m" in numeric_axes(ba)
    assert "topology.density" not in numeric_axes(ba), (
        "BA は density を持たない —— 軸はモデルの言葉のままにする (D-124)"
    )
    assert with_axis(ba, "topology.m", 4).topology == BarabasiAlbertConfig(m=4)


def test_the_nesting_stops_at_one_level() -> None:
    """2段目へは潜らない (軸名が長くなるだけで、深さの上限が読めなくなる)。"""
    for name in numeric_axes(ESNConfig()):
        assert name.count(".") <= 1, f"2段以上の軸があります: {name}"


def test_a_nested_axis_a_model_does_not_have_is_rejected() -> None:
    with pytest.raises(ValueError, match="持っていません"):
        with_axis(ESNConfig(), "topology.m", 4)

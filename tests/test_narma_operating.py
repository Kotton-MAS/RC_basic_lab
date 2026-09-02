"""動作点の掃引 (3-C'') の検査 (D-144).

3-C'' が答えるのは「NARMA10 の勝敗が**手法側の動作点**で変わるか」である。
ここが壊れると「動作点を1つ選んだ結果」を現象として報告することになる。

**空虚になる形は「動作点を変えたつもりで課題や分割まで動いていた」**なので、
リザバーを使わない手法 (線形回帰・遅延線) の成績が動作点によらず一定である
ことを正面から測る。
"""

from __future__ import annotations

import dataclasses

import pytest

from rc_basics_lab.config import (
    Capacity03Config,
    MackeyGlassTask,
    NarmaOperatingConfig,
    load_config_as,
    require_task,
)
from rc_basics_lab.experiment.narma_operating import (
    EXPERIMENT_NARMA10_OPERATING,
    OPERATING_CSV_COLUMNS,
    OperatingPointRow,
    base_at,
    operating_points,
    run_narma10_operating_sweep,
)
from rc_basics_lab.reservoir.axes import axis_value

GOLDEN_CONFIG = "tests/golden/configs/03_capacity.yaml"

RESERVOIR_FREE_METHODS = ("linear", "delay_line", "delay_line_ols")
"""リザバーを使わない手法。**動作点で成績が動いてはいけない**。"""


def tiny_config() -> Capacity03Config:
    """秒未満で1周する動作点の掃引 (格子は 2 x 2)。"""
    config = load_config_as(GOLDEN_CONFIG, Capacity03Config)
    return dataclasses.replace(
        config,
        narma_operating=NarmaOperatingConfig(
            n_units_grid=(10, 20), leak_rate_grid=(0.5, 1.0)
        ),
    )


def test_the_grid_is_the_product_of_both_axes() -> None:
    """動作点が N x リーク率の全組合せになる。"""
    section = NarmaOperatingConfig(n_units_grid=(10, 20), leak_rate_grid=(0.5, 1.0))
    assert operating_points(section) == (
        (10, 0.5),
        (10, 1.0),
        (20, 0.5),
        (20, 1.0),
    )
    assert operating_points(dataclasses.replace(section, n_units_grid=())) == ()
    assert operating_points(dataclasses.replace(section, leak_rate_grid=())) == ()


@pytest.mark.parametrize(
    "section",
    [
        NarmaOperatingConfig(n_units_grid=(0,), leak_rate_grid=(0.5,)),
        NarmaOperatingConfig(n_units_grid=(10,), leak_rate_grid=(0.0,)),
        NarmaOperatingConfig(n_units_grid=(10,), leak_rate_grid=(1.5,)),
    ],
)
def test_an_out_of_range_grid_is_rejected(section: NarmaOperatingConfig) -> None:
    """範囲外の格子は落とす (黙って素通りさせない)。"""
    with pytest.raises(ValueError, match=r"n_units|leak_rate"):
        operating_points(section)


def test_base_at_touches_only_the_reservoir() -> None:
    """**動かすのは ESN の2軸だけ** (課題も alpha 格子も分割も触らない)。

    ここが漏れると「動作点を変えた」ではなく「別の実験をした」になり、
    勝敗が動いた原因を動作点に帰せない。
    """
    base = load_config_as(GOLDEN_CONFIG, Capacity03Config).narma.base
    moved = base_at(base, 33, 0.42)
    reservoir = require_task(moved, MackeyGlassTask, "検査").reservoir
    assert int(axis_value(reservoir, "n_units")) == 33
    assert axis_value(reservoir, "leak_rate") == pytest.approx(0.42)
    # リザバー以外は1つも変わっていない
    assert moved.ridge == base.ridge
    assert moved.split == base.split
    assert moved.seeds == base.seeds
    assert moved.n_replicates == base.n_replicates
    original_task = require_task(base, MackeyGlassTask, "検査")
    assert require_task(moved, MackeyGlassTask, "検査").params == original_task.params


def test_only_the_esn_moves_with_the_operating_point() -> None:
    """**リザバーを使わない手法の成績は動作点によらず一定** (D-144)。

    線形回帰も遅延線もリザバーを見ないので、動作点を変えて成績が動いたなら
    課題か分割か乱数が一緒に動いている。「ESN が動作点で勝敗を変える」の
    対照が壊れている状態で、掃引の結論がまるごと信用できなくなる。
    """
    rows = run_narma10_operating_sweep(tiny_config())
    assert rows, "掃引が空です"
    for method in RESERVOIR_FREE_METHODS:
        by_replicate: dict[int, set[float]] = {}
        for row in rows:
            if row.method != method:
                continue
            by_replicate.setdefault(row.replicate, set()).add(round(row.nrmse, 12))
        assert by_replicate, f"{method} の行がありません"
        for replicate, values in by_replicate.items():
            assert len(values) == 1, (
                f"{method} のレプリケート {replicate} が動作点で動いています: "
                f"{sorted(values)}"
            )


def test_the_esn_actually_moves_with_the_operating_point() -> None:
    """ESN の成績は動作点で動く (**掃引が効いていることの確認**)。

    上の検査だけだと「全手法が定数」でも通る。掃引が何も効いていない状態
    (``base_at`` が土台をそのまま返す等) をここで殺す。
    """
    rows = run_narma10_operating_sweep(tiny_config())
    values = {round(row.nrmse, 12) for row in rows if row.method == "esn"}
    assert len(values) > 1, f"ESN の成績が動作点で動きません: {values}"
    capacities = {round(row.ipc_linear, 9) for row in rows}
    assert len(capacities) > 1, "容量が動作点で動きません"


def test_the_row_columns_are_the_declaration_order() -> None:
    """CSV の列順は行 dataclass の宣言順 (単一の真実)。"""
    assert (
        tuple(item.name for item in dataclasses.fields(OperatingPointRow))
        == OPERATING_CSV_COLUMNS
    )
    assert OPERATING_CSV_COLUMNS[:3] == ("experiment", "n_units", "leak_rate")
    rows = run_narma10_operating_sweep(tiny_config())
    assert {row.experiment for row in rows} == {EXPERIMENT_NARMA10_OPERATING}
    assert {(row.n_units, row.leak_rate) for row in rows} == {
        (10, 0.5),
        (10, 1.0),
        (20, 0.5),
        (20, 1.0),
    }


def test_the_nonlinear_share_matches_its_components() -> None:
    """``nonlinear_share`` が ``ipc_nonlinear / ipc_total`` と一致する。

    派生列は元の列から再現できないと、成果物を読む側が検算できない。
    """
    for row in run_narma10_operating_sweep(tiny_config()):
        assert row.nonlinear_share == pytest.approx(row.ipc_nonlinear / row.ipc_total)
        assert row.ipc_linear + row.ipc_nonlinear == pytest.approx(row.ipc_total)

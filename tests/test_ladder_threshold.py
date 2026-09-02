"""閾値感度 (3-Th) の検査 (D-143).

3-Th が答えるのは「梯子の順位が閾値の選び方で動くか」である。ここが壊れると
「閾値をいじって作った図ではない」と言えなくなる。

**空虚になる形は「判定基準を変えても同じ診断を呼んでいる」**なので、生の容量
(``ipc_total_raw``) が設定によらず一定であることと、しきい値なしが必ず最大に
なることの両方を測る。
"""

from __future__ import annotations

import dataclasses

import pytest

from rc_basics_lab.config import (
    Capacity03Config,
    LadderThresholdConfig,
    load_config_as,
)
from rc_basics_lab.experiment import esp as esp_module
from rc_basics_lab.experiment.ladder_threshold import (
    EXPERIMENT_LADDER_THRESHOLD,
    LADDER_THRESHOLD_CSV_COLUMNS,
    LadderThresholdRow,
    run_ladder_threshold,
    threshold_settings,
)

GOLDEN_CONFIG = "tests/golden/configs/03_capacity.yaml"


def tiny_config() -> Capacity03Config:
    """秒未満で1周する閾値感度 (格子は本番より粗い)。

    ``mc.max_delay`` をゴールデンの 20 から 120 へ広げる。20 のままだと
    **全遅延が閾値を超えて何も切られず**、MC の列が条件ごとの定数になる ——
    「順位が閾値で動かない」が『測っていないから動かない』の意味になる。
    本番 (max_delay=400) では切られる側の遅延が必ずある。
    """
    config = load_config_as(GOLDEN_CONFIG, Capacity03Config)
    return dataclasses.replace(
        config,
        mc=dataclasses.replace(config.mc, max_delay=120),
        topology_ladder=dataclasses.replace(
            config.topology_ladder, n_units=20, n_steps=1200, sweeps=()
        ),
        ladder_threshold=LadderThresholdConfig(
            n_surrogates_grid=(5, 10),
            quantile_grid=(0.9, 0.99),
            n_graphs=1,
            n_replicates=1,
        ),
    )


def test_the_threshold_grid_covers_every_setting() -> None:
    """判定基準が (しきい値なし) + 本数 x 分位点 の全件になる。"""
    section = LadderThresholdConfig(
        n_surrogates_grid=(5, 10), quantile_grid=(0.9, 0.99)
    )
    settings = threshold_settings(section)
    assert len(settings) == 1 + 2 * 2
    assert settings[0].mode == "none", "しきい値なしを先頭に置く (基準だから)"
    assert {(s.n_surrogates, s.quantile) for s in settings[1:]} == {
        (5, 0.9),
        (5, 0.99),
        (10, 0.9),
        (10, 0.99),
    }
    with pytest.raises(ValueError, match="空"):
        threshold_settings(dataclasses.replace(section, quantile_grid=()))


def test_the_raw_capacity_does_not_depend_on_the_threshold() -> None:
    """**生の容量は判定基準によらず一定** (D-143)。

    ここが動くなら、しきい値の変更が容量の計算そのものに漏れている。
    「閾値を変えたら順位が動いた」の原因を閾値に帰せなくなる。
    """
    rows = run_ladder_threshold(tiny_config())
    by_condition: dict[tuple[str, int, int], set[float]] = {}
    for row in rows:
        key = (row.level, row.graph, row.replicate)
        by_condition.setdefault(key, set()).add(round(row.ipc_total_raw, 9))
    for key, values in by_condition.items():
        assert len(values) == 1, f"{key} の生の容量が判定基準で動いています: {values}"


def test_no_threshold_is_never_smaller_than_a_threshold() -> None:
    """しきい値なしが必ず最大 (しきい値は容量を**削る**方向にしか効かない)。"""
    rows = run_ladder_threshold(tiny_config())
    for level, graph, replicate in {
        (row.level, row.graph, row.replicate) for row in rows
    }:
        selected = [
            row
            for row in rows
            if (row.level, row.graph, row.replicate) == (level, graph, replicate)
        ]
        bare = next(row for row in selected if row.threshold_mode == "none")
        for row in selected:
            assert row.ipc_total <= bare.ipc_total + 1.0e-9, (
                f"{level}: しきい値が容量を増やしています ({row.threshold_mode} "
                f"n={row.n_surrogates} q={row.surrogate_quantile})"
            )
            assert row.mc_total <= bare.mc_total + 1.0e-9


def test_both_diagnostics_respond_to_the_threshold_setting() -> None:
    """**MC も IPC も判定基準に反応する** (D-143)。

    片方に既定の設定を渡したままにすると、その列は条件ごとに定数になり、
    「順位が閾値で動かない」が**測っていないから動かない**の意味になる。
    しきい値なしと課したときで値が違うことを、両方の診断について測る。
    """
    rows = run_ladder_threshold(tiny_config())
    for column in ("mc_total", "ipc_total"):
        by_condition: dict[tuple[str, int, int], set[float]] = {}
        for row in rows:
            key = (row.level, row.graph, row.replicate)
            by_condition.setdefault(key, set()).add(
                round(float(getattr(row, column)), 9)
            )
        assert any(len(values) > 1 for values in by_condition.values()), (
            f"{column} が判定基準で1度も動きません (しきい値が課されていない)"
        )


def test_the_trajectory_is_built_once_per_condition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**軌道は判定基準ごとに作り直さない** (仕様 §5 の禁止構造)。

    素直に書くと判定基準の数だけ状態生成が走り、しかも「基準ごとに別の X を
    見た」比較になる。回数を数えて固定する。
    """
    calls = 0
    real = esp_module.simulate_reference_trajectory

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)  # type: ignore[arg-type]

    import rc_basics_lab.experiment.ladder_threshold as module

    monkeypatch.setattr(module, "simulate_reference_trajectory", counted)
    config = tiny_config()
    rows = run_ladder_threshold(config)
    section = config.ladder_threshold
    n_conditions = (
        len(config.topology_ladder.levels) * section.n_graphs * section.n_replicates
    )
    assert calls == n_conditions, (
        f"軌道を {calls} 回作りました (条件は {n_conditions} 個)"
    )
    assert len(rows) == n_conditions * len(threshold_settings(section))


def test_the_row_columns_are_the_declaration_order() -> None:
    """CSV の列順は行 dataclass の宣言順 (単一の真実)。"""
    assert (
        tuple(item.name for item in dataclasses.fields(LadderThresholdRow))
        == LADDER_THRESHOLD_CSV_COLUMNS
    )
    assert LADDER_THRESHOLD_CSV_COLUMNS[:4] == (
        "experiment",
        "threshold_mode",
        "n_surrogates",
        "surrogate_quantile",
    )
    rows = run_ladder_threshold(tiny_config())
    assert {row.experiment for row in rows} == {EXPERIMENT_LADDER_THRESHOLD}

"""ESP 判定の閾値感度 (D-16 の既定値が結論を作っていないことの実測).

``docs/design.md`` §9 の感度表の一次資料が
``results/02_esp_and_dynamics/esp_threshold_sensitivity.csv`` であり、
再生成は ``run_02.py --threshold-sweep`` (= ``make threshold-02``)。
ここでは縮小格子で **掃引の性質** を固定する:

- 行数が ``abs_tol`` x ``window`` の組の数と一致する
- 基準 (D-16 の既定値) の行は自分自身からずれない
- ``abs_tol`` を緩めると ESP 成立と判定される条件は減らない (単調性)
- CSV のヘッダと行の対応が崩れない (``critical_rho_by_sigma`` の展開規則)

本番格子での実測は ``docs/design.md`` §9 にある。
"""

from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path

import pytest

from rc_basics_lab.config import (
    DriveConfig,
    Esp02Config,
    EspConfig,
    EspDecayConfig,
    EspMapConfig,
    EspSeedConfig,
    ReservoirSweepConfig,
    TimescaleConfig,
    TimescaleSweepConfig,
)
from rc_basics_lab.experiment.esp import UNIFORM
from rc_basics_lab.experiment.esp_pipeline import (
    ESP_THRESHOLD_SENSITIVITY_CSV,
    run_and_report_threshold_sweep,
    write_threshold_csv,
)
from rc_basics_lab.experiment.threshold import (
    ABS_TOL_GRID,
    REFERENCE_ABS_TOL,
    REFERENCE_WINDOW,
    THRESHOLD_SCALAR_COLUMNS,
    WINDOW_GRID,
    critical_rho,
    run_threshold_sweep,
    sigma_column,
    threshold_csv_columns,
    threshold_row_as_dict,
)

TEST_ABS_TOL_GRID = (1.0e-4, REFERENCE_ABS_TOL, 1.0e-8)
TEST_WINDOW_GRID = (50, REFERENCE_WINDOW)
"""縮小格子。基準の組を必ず含む (含まないと ``run_threshold_sweep`` が拒む)。"""


def small_config() -> Esp02Config:
    """秒未満で 2-C 相当を回せる縮小設定 (構造は本番と同じ)。

    ``n_steps`` は最大の ``window`` (200) より十分長くしてある。
    """
    return Esp02Config(
        name="threshold-test",
        seeds=EspSeedConfig(reservoir=0, drive=1, probe=3),
        drive=DriveConfig(distribution=UNIFORM, n_steps=600, washout=100, n_pairs=6),
        reservoir=ReservoirSweepConfig(
            input_scale=1.0, n_units=40, density=0.2, n_replicates=3
        ),
        decay=EspDecayConfig(rho_grid=(0.5,), sigma_u=0.0, leak_rate=1.0),
        timescale_sweep=TimescaleSweepConfig(
            leak_rate_grid=(1.0,), rho=0.9, sigma_u=0.5
        ),
        esp_map=EspMapConfig(
            rho_grid=(0.6, 0.9, 1.2, 1.8),
            sigma_grid=(0.0, 0.5, 2.0),
            leak_rate=1.0,
        ),
        esp=EspConfig(window=REFERENCE_WINDOW, fit_skip=10),
        timescale=TimescaleConfig(max_lag=50),
    )


def sweep() -> tuple[object, ...]:
    return run_threshold_sweep(
        small_config(),
        abs_tol_grid=TEST_ABS_TOL_GRID,
        window_grid=TEST_WINDOW_GRID,
    )


# --- 臨界 rho の定義 --------------------------------------------------------


def test_critical_rho_is_the_first_rho_below_majority() -> None:
    """収束率が過半数を割った**最初の** rho が境界。"""
    assert critical_rho({0.5: 1.0, 0.9: 1.0, 1.2: 0.0, 1.5: 0.0}) == 1.2
    assert critical_rho({0.5: 1.0, 0.9: 1.0 / 3.0, 1.2: 2.0 / 3.0}) == 0.9


def test_critical_rho_is_nan_when_the_grid_has_no_boundary() -> None:
    """全 rho で収束するなら境界は格子の外。格子上端と混同させない。"""
    assert math.isnan(critical_rho({0.5: 1.0, 0.9: 1.0, 1.2: 1.0}))


# --- 掃引の形 ---------------------------------------------------------------


def test_sweep_has_one_row_per_threshold_combination() -> None:
    """行数と並びが ``abs_tol`` x ``window`` の直積と一致する。"""
    rows = sweep()
    assert len(rows) == len(TEST_ABS_TOL_GRID) * len(TEST_WINDOW_GRID)
    assert [(row.abs_tol, row.window) for row in rows] == [  # type: ignore[attr-defined]
        (abs_tol, window)
        for abs_tol in TEST_ABS_TOL_GRID
        for window in TEST_WINDOW_GRID
    ]


def test_production_grids_are_the_nine_combinations_of_the_spec() -> None:
    """既定の格子は仕様どおり ``abs_tol`` 3点 x ``window`` 3点の9通り。"""
    assert ABS_TOL_GRID == (1.0e-4, 1.0e-6, 1.0e-8)
    assert WINDOW_GRID == (100, 200, 400)
    assert len(ABS_TOL_GRID) * len(WINDOW_GRID) == 9
    assert REFERENCE_ABS_TOL in ABS_TOL_GRID
    assert REFERENCE_WINDOW in WINDOW_GRID


def test_reference_case_does_not_shift_from_itself() -> None:
    """基準 (D-16 の既定値) の行は自分自身と一致する (ずれ 0)。"""
    reference = next(
        row
        for row in sweep()
        if (row.abs_tol, row.window) == (REFERENCE_ABS_TOL, REFERENCE_WINDOW)  # type: ignore[attr-defined]
    )
    assert reference.n_sigma_shifted == 0  # type: ignore[attr-defined]
    assert reference.max_abs_shift == 0.0  # type: ignore[attr-defined]


def test_looser_tolerance_never_shrinks_the_esp_region() -> None:
    """``abs_tol`` を緩めると ESP 成立と判定される条件は減らない (D-16 の単調性)。

    診断単体の単調性は ``test_verdict_is_monotone_in_tolerance`` が見ている。
    ここは掃引全体 (実験層で組んだ格子) でも同じ向きであることを固定する。
    """
    rows = sweep()
    for window in TEST_WINDOW_GRID:
        by_tolerance = [row for row in rows if row.window == window]  # type: ignore[attr-defined]
        by_tolerance.sort(key=lambda row: row.abs_tol)  # type: ignore[attr-defined,no-any-return]
        counts = [row.n_converged for row in by_tolerance]  # type: ignore[attr-defined]
        assert counts == sorted(counts), (window, counts)


def test_every_row_counts_all_conditions() -> None:
    """``n_conditions`` は 2-C の全条件 (rho x sigma_u x レプリケート)。"""
    config = small_config()
    expected = (
        len(config.esp_map.rho_grid)
        * len(config.esp_map.sigma_grid)
        * config.reservoir.n_replicates
    )
    for row in sweep():
        assert row.n_conditions == expected  # type: ignore[attr-defined]
        assert 0 <= row.n_converged <= expected  # type: ignore[attr-defined]
        assert len(row.critical_rho_by_sigma) == len(config.esp_map.sigma_grid)  # type: ignore[attr-defined]


def test_reference_case_must_be_in_the_grid() -> None:
    """基準の組が無い格子は拒む (何からのずれか分からなくなるため)。"""
    with pytest.raises(ValueError, match="基準"):
        run_threshold_sweep(
            small_config(), abs_tol_grid=(1.0e-3,), window_grid=TEST_WINDOW_GRID
        )
    with pytest.raises(ValueError, match="1点以上"):
        run_threshold_sweep(small_config(), abs_tol_grid=(), window_grid=())


# --- CSV -------------------------------------------------------------------


def test_threshold_csv_header_matches_rows(tmp_path: Path) -> None:
    """ヘッダと各行のキーが一致する (``critical_rho_by_sigma`` の展開規則)。"""
    config = small_config()
    rows = sweep()
    path = write_threshold_csv(
        rows,  # type: ignore[arg-type]
        config.esp_map.sigma_grid,
        tmp_path / ESP_THRESHOLD_SENSITIVITY_CSV,
    )
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        written = list(reader)
    assert header == threshold_csv_columns(config.esp_map.sigma_grid)
    assert header[: len(THRESHOLD_SCALAR_COLUMNS)] == THRESHOLD_SCALAR_COLUMNS
    assert len(written) == len(rows)
    for row, record in zip(rows, written, strict=True):
        assert set(record) == set(threshold_row_as_dict(row))  # type: ignore[arg-type]
        assert float(record["abs_tol"]) == row.abs_tol  # type: ignore[attr-defined]
        for sigma, value in row.critical_rho_by_sigma:  # type: ignore[attr-defined]
            written_value = float(record[sigma_column(sigma)])
            assert (written_value == value) or (
                math.isnan(written_value) and math.isnan(value)
            )


def test_threshold_sweep_writes_only_its_own_csv(tmp_path: Path) -> None:
    """``--threshold-sweep`` 相当は本体の7成果物を上書きしない。"""
    out_dir = tmp_path / "out"
    reduced = dataclasses.replace(small_config())
    path = run_and_report_threshold_sweep(reduced, out_dir)
    assert path.name == ESP_THRESHOLD_SENSITIVITY_CSV
    produced = sorted(item.name for item in out_dir.iterdir())
    assert produced == [ESP_THRESHOLD_SENSITIVITY_CSV]

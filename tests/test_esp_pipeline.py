"""02 の「1コマンドで成果物が出る」経路のテスト (受け入れ条件7).

``main.py --experiment 02`` と ``main.py (--experiment 02)``
はどちらも ``esp_pipeline.run_and_report_esp`` を呼ぶ薄い層である。ここでは
縮小設定を一時ディレクトリに書いて**実際に1コマンド相当を走らせ**、
CSV2枚 (``esp_diagnostics.csv`` / ``washout_sensitivity.csv``) と図4枚と
``meta.json`` が出ることと、PNG の実測解像度が retina 相当であることを見る
(01 の ``tests/test_main.py`` と同じ規律)。
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from pathlib import Path

import pytest
from conftest import png_dpi

import main
from rc_basics_lab.config import Esp02Config, load_config_as
from rc_basics_lab.experiment.esp import ESP_CSV_COLUMNS, EXPERIMENT_ESP_MAP
from rc_basics_lab.experiment.esp_pipeline import (
    ESP_ARTIFACTS,
    ESP_DIAGNOSTICS_CSV,
    WASHOUT_SENSITIVITY_CSV,
    run_and_report_esp,
)
from rc_basics_lab.experiment.washout import WASHOUT_CSV_COLUMNS
from rc_basics_lab.plotting.figures_esp import plot_esp_map
from rc_basics_lab.plotting.style import setup_style

RETINA_DPI = 200
EXPECTED_ROWS = 20
"""2課題ぶんの条件数: 2-A 2x2 + 2-B 2x2 + 2-C (2x3)x2。"""

EXPECTED_FIGURES = 4
EXPECTED_CSV = 3
EXPECTED_WASHOUT_ROWS = 24
"""2-D の行数: washout 2点 x 2課題 x 3手法 x 2レプリケート。"""

TINY_CONFIG = """
name: esp_cli_smoke
seeds:
  reservoir: 0
  drive: 1
  probe: 3
drive:
  distribution: uniform
  n_steps: 500
  washout: 60
  n_pairs: 4
reservoir:
  input_scale: 1.0
  n_units: 30
  density: 0.3
  n_replicates: 2
decay:
  rho_grid: [0.5, 1.3]
  sigma_u: 0.0
  leak_rate: 1.0
timescale_sweep:
  leak_rate_grid: [0.2, 1.0]
  rho: 0.9
  sigma_u: 0.5
esp_map:
  rho_grid: [0.8, 1.5]
  sigma_grid: [0.0, 0.5, 2.0]
  leak_rate: 1.0
washout:
  # 2-D。base は 01 の設定そのものなので、既定 (系列長 8000 x 5 レプリケート)
  # のままだと格子ぶん回して数十秒かかる。構造は変えずに規模だけ削る。
  grid: [0, 40]
  pad_series: true
  base:
    name: washout_smoke
    n_replicates: 2
    split:
      washout: 40
      max_start_offset: 40
    ridge:
      alpha_grid: [1.0e-4, 1.0]
      n_lags_grid: [1, 4]
    tasks:
      - kind: mackey_glass
        params:
          length: 500
        reservoir:
          n_units: 30
          topology:
            kind: erdos_renyi
            density: 0.3
      - kind: delay_parity
        params:
          length: 500
        reservoir:
          n_units: 30
          topology:
            kind: erdos_renyi
            density: 0.3
          leak_rate: 1.0
          input_scale: 1.0
esp:
  window: 100
  fit_skip: 10
timescale:
  max_lag: 40
"""


@pytest.fixture
def tiny_experiment(tmp_path: Path) -> tuple[Path, Path]:
    """縮小設定を ``--experiment 02`` に差し替える (本番は 77 秒かかるため)。"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(TINY_CONFIG, encoding="utf-8")
    return config_path, tmp_path / "out"


def test_artifacts_are_regenerated_in_one_command(
    tiny_experiment: tuple[Path, Path],
) -> None:
    """1コマンドで宣言済みの成果物がすべて出る (受け入れ条件7)。"""
    config_path, out_dir = tiny_experiment
    assert (
        main.main(
            ["--experiment", "02", "--config", str(config_path), "--out", str(out_dir)]
        )
        == 0
    )
    for name in ESP_ARTIFACTS:
        assert (out_dir / name).is_file(), f"{name} が生成されていません"
    figures = [name for name in ESP_ARTIFACTS if name.endswith(".png")]
    assert len(figures) == EXPECTED_FIGURES
    for name in figures:
        assert png_dpi(out_dir / name) >= RETINA_DPI


def test_all_four_figures_and_three_csv_in_one_command(
    tiny_experiment: tuple[Path, Path],
) -> None:
    """図4枚 + CSV3枚 + meta.json が1コマンドで出る (受け入れ条件7)。

    3枚目は ``diagnostics.csv`` (診断のスカラの長形式、D-118)。

    ``ESP_ARTIFACTS`` の**中身**を数えるのではなく、出力ディレクトリを実際に
    走査して数える。宣言と実体が食い違ったとき (図を1枚落としたのに
    ``ESP_ARTIFACTS`` から消し忘れた / その逆) に、宣言だけを見るテストは
    黙って通るため。
    """
    config_path, out_dir = tiny_experiment
    assert (
        main.main(
            ["--experiment", "02", "--config", str(config_path), "--out", str(out_dir)]
        )
        == 0
    )
    produced = sorted(path.name for path in out_dir.iterdir() if path.is_file())
    figures = [name for name in produced if name.endswith(".png")]
    csvs = [name for name in produced if name.endswith(".csv")]
    assert len(figures) == EXPECTED_FIGURES, produced
    assert len(csvs) == EXPECTED_CSV, produced
    assert "meta.json" in produced
    assert set(produced) == set(ESP_ARTIFACTS), (
        "生成物と ESP_ARTIFACTS の宣言が食い違っています"
    )
    for name in figures:
        assert png_dpi(out_dir / name) >= RETINA_DPI


def test_washout_csv_has_the_declared_columns(
    tiny_experiment: tuple[Path, Path],
) -> None:
    """``washout_sensitivity.csv`` の列順が ``WashoutRow`` の宣言順と一致する。"""
    config_path, out_dir = tiny_experiment
    assert (
        main.main(
            ["--experiment", "02", "--config", str(config_path), "--out", str(out_dir)]
        )
        == 0
    )
    with (out_dir / WASHOUT_SENSITIVITY_CSV).open(
        encoding="utf-8", newline=""
    ) as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    assert tuple(header) == WASHOUT_CSV_COLUMNS
    assert len(rows) == EXPECTED_WASHOUT_ROWS


def test_csv_has_the_declared_columns(tiny_experiment: tuple[Path, Path]) -> None:
    """``esp_diagnostics.csv`` の列順が ``EspRow`` の宣言順と一致する。"""
    config_path, out_dir = tiny_experiment
    assert (
        main.main(
            ["--experiment", "02", "--config", str(config_path), "--out", str(out_dir)]
        )
        == 0
    )
    with (out_dir / ESP_DIAGNOSTICS_CSV).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    assert tuple(header) == ESP_CSV_COLUMNS
    assert len(rows) == EXPECTED_ROWS


def test_meta_json_records_defaults_and_verdict_agreement(
    tiny_experiment: tuple[Path, Path],
) -> None:
    """``esp_defaults`` と ``verdict_lyapunov_agreement`` が meta.json に残る。

    後者は「λ<0 なのに非収束」がどこで起きたかの一次資料 (多安定性の観測)。
    件数だけでなく sigma_u / rho の分布まで載っていることを固定する。
    """
    config_path, out_dir = tiny_experiment
    assert (
        main.main(
            ["--experiment", "02", "--config", str(config_path), "--out", str(out_dir)]
        )
        == 0
    )
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))

    assert meta["n_rows"] == EXPECTED_ROWS
    assert meta["wall_time_s"] > 0.0
    defaults = meta["esp_defaults"]
    assert defaults["bias_scale"] == 0.0
    assert defaults["esp_distance_washout"] == 0
    assert defaults["input_amplitude_per_sigma"] == pytest.approx(math.sqrt(3.0))
    assert defaults["lyapunov"]["check_propagator"] is True

    agreement = meta["verdict_lyapunov_agreement"]
    assert agreement["n_false_esp"] == 0
    assert agreement["n_rows"] == EXPECTED_ROWS
    assert agreement["n_near_boundary"] + agreement["n_compared"] == EXPECTED_ROWS
    for key in ("disagreement_by_sigma", "disagreement_by_rho"):
        assert isinstance(agreement[key], list)


def test_esp_map_figure_works_without_a_no_input_column(tmp_path: Path) -> None:
    """``sigma_u = 0`` が格子に無くても 2-C の図が描ける。

    無入力パネルは「無入力なら rho<1 が必要条件」を駆動下の主張と混ぜない
    ための別枠であり、格子に 0 が無ければ出ない。その分岐で図が落ちると、
    格子を変えた瞬間に再生成コマンドが死ぬ。
    """
    config = load_config_as(_write(tmp_path, TINY_CONFIG), Esp02Config)
    driven = dataclasses.replace(
        config,
        esp_map=dataclasses.replace(config.esp_map, sigma_grid=(0.5, 2.0)),
    )
    outputs = run_and_report_esp(driven, tmp_path / "driven")
    rows = [row for row in outputs.rows if row.experiment == EXPERIMENT_ESP_MAP]
    assert rows
    assert all(row.sigma_u > 0.0 for row in rows)

    path = plot_esp_map(rows, tmp_path / "no_zero.png", style=setup_style())
    assert png_dpi(path) >= RETINA_DPI


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _without_wall_time(csv_text: str) -> list[list[str]]:
    """CSV から実測時間の列だけ落とす (実行ごとに変わるため)。"""
    rows = [line.split(",") for line in csv_text.strip().splitlines()]
    index = rows[0].index("wall_time_s")
    return [row[:index] + row[index + 1 :] for row in rows]

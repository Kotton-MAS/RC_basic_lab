"""02 の「1コマンドで成果物が出る」経路のテスト (受け入れ条件7 の T3 ぶん).

``main.py --experiment 02`` と ``experiments/02_esp_and_dynamics/run_02.py``
はどちらも ``esp_pipeline.run_and_report_esp`` を呼ぶ薄い層である。ここでは
縮小設定を一時ディレクトリに書いて**実際に1コマンド相当を走らせ**、
``esp_diagnostics.csv`` / 図3枚 / ``meta.json`` が出ることと、PNG の実測解像度が
retina 相当であることを見る (01 の ``tests/test_main.py`` と同じ規律)。
"""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
from conftest import png_dpi

import main
from rc_basics_lab.config import Esp02Config, load_config_as
from rc_basics_lab.experiment.esp import ESP_CSV_COLUMNS, EXPERIMENT_ESP_MAP
from rc_basics_lab.experiment.esp_pipeline import (
    ESP_ARTIFACTS,
    ESP_DIAGNOSTICS_CSV,
    run_and_report_esp,
)
from rc_basics_lab.plotting.figures_esp import plot_esp_map
from rc_basics_lab.plotting.style import setup_style

RETINA_DPI = 200
EXPECTED_ROWS = 20
"""2課題ぶんの条件数: 2-A 2x2 + 2-B 2x2 + 2-C (2x3)x2。"""

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
esp:
  window: 100
  fit_skip: 10
timescale:
  max_lag: 40
"""


@pytest.fixture
def tiny_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """縮小設定を ``--experiment 02`` に差し替える (本番は 77 秒かかるため)。"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(TINY_CONFIG, encoding="utf-8")
    monkeypatch.setitem(
        main.EXPERIMENTS,
        "02",
        dataclasses.replace(main.EXPERIMENTS["02"], config_path=config_path),
    )
    return config_path, tmp_path / "out"


def test_artifacts_are_regenerated_in_one_command(
    tiny_experiment: tuple[Path, Path],
) -> None:
    """1コマンドで CSV1枚・図3枚・meta.json が出る (受け入れ条件7 の T3 ぶん)。"""
    _, out_dir = tiny_experiment
    assert main.main(["--experiment", "02", "--out", str(out_dir)]) == 0
    for name in ESP_ARTIFACTS:
        assert (out_dir / name).is_file(), f"{name} が生成されていません"
    figures = [name for name in ESP_ARTIFACTS if name.endswith(".png")]
    assert len(figures) == 3
    for name in figures:
        assert png_dpi(out_dir / name) >= RETINA_DPI


def test_csv_has_the_declared_columns(tiny_experiment: tuple[Path, Path]) -> None:
    """``esp_diagnostics.csv`` の列順が ``EspRow`` の宣言順と一致する。"""
    _, out_dir = tiny_experiment
    assert main.main(["--experiment", "02", "--out", str(out_dir)]) == 0
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
    _, out_dir = tiny_experiment
    assert main.main(["--experiment", "02", "--out", str(out_dir)]) == 0
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


def test_run_02_and_main_py_agree(tiny_experiment: tuple[Path, Path]) -> None:
    """``run_02.py --config`` と ``main.py --experiment 02`` が同じ CSV を出す。"""
    config_path, out_dir = tiny_experiment
    run_module = _load_run_module()
    assert (
        run_module.main(["--config", str(config_path), "--out", str(out_dir / "run")])
        == 0
    )
    assert main.main(["--experiment", "02", "--out", str(out_dir / "cli")]) == 0
    left = (out_dir / "run" / ESP_DIAGNOSTICS_CSV).read_text(encoding="utf-8")
    right = (out_dir / "cli" / ESP_DIAGNOSTICS_CSV).read_text(encoding="utf-8")
    assert _without_wall_time(left) == _without_wall_time(right)


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


def _load_run_module() -> ModuleType:
    """``experiments/02_esp_and_dynamics/run_02.py`` を読み込む (パッケージ外)。"""
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "02_esp_and_dynamics"
        / "run_02.py"
    )
    spec = importlib.util.spec_from_file_location("experiment_02_run", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"読み込めません: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclass の型解決は sys.modules を引くため、exec 前に登録しておく
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

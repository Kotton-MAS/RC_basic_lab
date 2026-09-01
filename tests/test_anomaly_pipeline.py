"""05 の「1コマンドで成果物が出る」経路のテスト (仕様 §4 T5 受け入れ基準1・5).

``main.py --experiment 05`` と ``main.py (--experiment 05)``
はどちらも ``anomaly_pipeline.run_and_report_anomaly`` を呼ぶ薄い層である。
ここでは縮小設定を一時ディレクトリに書いて**実際に1コマンド相当を走らせ**、
CSV5枚 (``anomaly.csv`` / ``anomaly_threshold.csv`` / ``anomaly_timeline.csv`` /
``anomaly_protocol.csv`` / ``anomaly_size.csv``) と図5枚と ``meta.json`` が出る
ことと、PNG の実測解像度が retina 相当であることを見る (02〜04 のパイプライン
テストと同じ規律)。

**縮小設定の源は合成である** (D-60)。実データ源 (MGAB) を使う本番の
``experiments/05_anomaly_detection/config.yaml`` はキャッシュが要るので、
pytest からは走らせない。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from conftest import png_dpi

import main
from rc_basics_lab.config import Anomaly05Config, load_config_as
from rc_basics_lab.experiment.anomaly_pipeline import (
    ANOMALY_ARTIFACTS,
    SECTION_BUDGETS_S,
    TOTAL_BUDGET_S,
    build_timeline_rows,
    f1_gap_summary,
    preprocessor_uniqueness,
    run_and_report_anomaly,
)
from rc_basics_lab.experiment.anomaly_rows import (
    ANOMALY_CSV,
    ANOMALY_PROTOCOL_CSV,
    ANOMALY_PROTOCOL_CSV_COLUMNS,
    ANOMALY_SIZE_CSV,
    ANOMALY_SIZE_CSV_COLUMNS,
    ANOMALY_THRESHOLD_CSV,
    ANOMALY_THRESHOLD_CSV_COLUMNS,
    ANOMALY_TIMELINE_CSV,
    ANOMALY_TIMELINE_CSV_COLUMNS,
    anomaly_csv_columns,
)
from rc_basics_lab.experiment.anomaly_score import ANOMALY_METHODS
from rc_basics_lab.experiment.anomaly_sources import build_sources

RETINA_DPI = 200
EXPECTED_FIGURES = 5
EXPECTED_CSV = 5
EXPECTED_ROWS = 6
"""1系列 x 1レプリケート x 6系統 (対照は設定から外せない、D-61)。"""

TINY_CONFIG = """
name: anomaly_cli_smoke
dataset:
  source: synthetic
  series: ["s1"]
  max_length: 2000
  train_ratio: 0.25
  calibration_ratio: 0.15
synthetic:
  length: 2000
  n_anomalies: 3
  segment_length: 40
  ignore_margin: 10
  mackey_glass:
    integration_burn_in: 200
preprocess:
  normalize: zscore
  standardize_steps: 200
  input_window: 8
  score_smoothing: 4
reservoir:
  n_units: 30
  washout: 20
  n_replicates: 1
ridge:
  alpha_grid: [1.0e-4, 1.0e-2, 1.0]
threshold:
  target_false_alarm_rate: 0.05
  report_test_optimal: true
  sweep_points: 5
evaluation:
  report_point_adjust: true
  pa_k_grid: [0.0]
  ignore_transition: true
protocol_sweep:
  normalize_grid: [zscore, minmax]
  input_window_grid: [4, 8]
  score_smoothing_grid: [2, 4]
size_sweep:
  n_units_grid: [20, 30]
seeds:
  reservoir: 0
  task: 1
  split: 2
  control: 5
"""


@pytest.fixture
def tiny_experiment(tmp_path: Path) -> tuple[Path, Path]:
    """縮小設定を ``--experiment 05`` に差し替える (本番は実データ源で数分)。"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(TINY_CONFIG, encoding="utf-8")
    return config_path, tmp_path / "out"


def _rows_of(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


def test_artifacts_are_regenerated_in_one_command(
    tiny_experiment: tuple[Path, Path],
) -> None:
    """1コマンドで宣言済みの成果物がすべて出る (受け入れ基準1)。"""
    config_path, out_dir = tiny_experiment
    assert (
        main.main(
            ["--experiment", "05", "--config", str(config_path), "--out", str(out_dir)]
        )
        == 0
    )
    for name in ANOMALY_ARTIFACTS:
        assert (out_dir / name).is_file(), f"{name} が生成されていません"
    figures = [name for name in ANOMALY_ARTIFACTS if name.endswith(".png")]
    assert len(figures) == EXPECTED_FIGURES
    for name in figures:
        assert png_dpi(out_dir / name) >= RETINA_DPI


def test_five_figures_and_five_csv_in_one_command(
    tiny_experiment: tuple[Path, Path],
) -> None:
    """図5枚 + CSV5枚 + meta.json が1コマンドで出る (受け入れ基準1)。

    ``ANOMALY_ARTIFACTS`` の**中身**を数えるのではなく、出力ディレクトリを
    実際に走査して数える。宣言と実体が食い違ったとき (図を1枚落としたのに
    宣言から消し忘れた / その逆) に、宣言だけを見るテストは黙って通るため。
    """
    config_path, out_dir = tiny_experiment
    assert (
        main.main(
            ["--experiment", "05", "--config", str(config_path), "--out", str(out_dir)]
        )
        == 0
    )
    produced = sorted(path.name for path in out_dir.iterdir() if path.is_file())
    assert len([name for name in produced if name.endswith(".png")]) == EXPECTED_FIGURES
    assert len([name for name in produced if name.endswith(".csv")]) == EXPECTED_CSV
    assert "meta.json" in produced
    assert set(produced) == set(ANOMALY_ARTIFACTS), (
        "生成物と ANOMALY_ARTIFACTS の宣言が食い違っています"
    )


def test_every_csv_has_the_declared_columns(
    tiny_experiment: tuple[Path, Path],
) -> None:
    """5枚の CSV の列順が、それぞれの単一の真実と一致する。"""
    config_path, out_dir = tiny_experiment
    config = load_config_as(config_path, Anomaly05Config)
    assert (
        main.main(
            ["--experiment", "05", "--config", str(config_path), "--out", str(out_dir)]
        )
        == 0
    )
    expected: tuple[tuple[str, tuple[str, ...]], ...] = (
        (ANOMALY_CSV, anomaly_csv_columns(config)),
        (ANOMALY_THRESHOLD_CSV, ANOMALY_THRESHOLD_CSV_COLUMNS),
        (ANOMALY_TIMELINE_CSV, ANOMALY_TIMELINE_CSV_COLUMNS),
        (ANOMALY_PROTOCOL_CSV, ANOMALY_PROTOCOL_CSV_COLUMNS),
        (ANOMALY_SIZE_CSV, ANOMALY_SIZE_CSV_COLUMNS),
    )
    for name, columns in expected:
        header, rows = _rows_of(out_dir / name)
        assert tuple(header) == columns, name
        assert rows, f"{name} が空です"
    header, rows = _rows_of(out_dir / ANOMALY_CSV)
    assert len(rows) == EXPECTED_ROWS
    assert {row["method"] for row in rows} == set(ANOMALY_METHODS)


def test_the_timeline_matches_the_headline_row(
    tiny_experiment: tuple[Path, Path],
) -> None:
    """``anomaly_timeline.csv`` が 5-A の行と同じ条件である (D-82)。

    図に出す1例が別条件で作られていると、記事の図と ``anomaly.csv`` の数値が
    違う実験のものになる。閾値・系列・レプリケート・区間の3点で照合する。
    """
    config_path, out_dir = tiny_experiment
    assert (
        main.main(
            ["--experiment", "05", "--config", str(config_path), "--out", str(out_dir)]
        )
        == 0
    )
    _, headline = _rows_of(out_dir / ANOMALY_CSV)
    _, timeline = _rows_of(out_dir / ANOMALY_TIMELINE_CSV)
    first = headline[0]
    assert {row["series"] for row in timeline} == {first["series"]}
    assert {row["replicate"] for row in timeline} == {"0"}
    assert {row["method"] for row in timeline} == set(ANOMALY_METHODS)
    thresholds = {row["method"]: row["threshold"] for row in headline}
    for row in timeline:
        assert row["threshold"] == thresholds[row["method"]], row["method"]
    indices = sorted({int(row["index"]) for row in timeline})
    n_test = int(first["n_test"])
    assert len(indices) <= n_test
    assert any(row["is_anomaly"] == "True" for row in timeline), (
        "図に描く異常が1点も残っていません"
    )


def test_the_timeline_refuses_to_drop_every_anomaly(
    tiny_experiment: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """間引きで異常が全部落ちる設定は ``ValueError`` (静かに帯の無い図にしない)。"""
    config_path, _ = tiny_experiment
    config = load_config_as(config_path, Anomaly05Config)
    monkeypatch.setattr(
        "rc_basics_lab.experiment.anomaly_pipeline.TIMELINE_MAX_POINTS", 2
    )
    with pytest.raises(ValueError, match="異常点が1つも残りません"):
        build_timeline_rows(config, build_sources(config))


def test_meta_json_records_the_section_wall_times(
    tiny_experiment: tuple[Path, Path],
) -> None:
    """区間別の wall time が meta.json に残り、合計が予算内 (受け入れ基準5)。"""
    config_path, out_dir = tiny_experiment
    assert (
        main.main(
            ["--experiment", "05", "--config", str(config_path), "--out", str(out_dir)]
        )
        == 0
    )
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    breakdown = meta["wall_time_breakdown"]
    assert set(breakdown) == set(SECTION_BUDGETS_S)
    for section, budget in SECTION_BUDGETS_S.items():
        assert breakdown[section] > 0.0, section
        assert breakdown[section] < budget, section
    assert sum(breakdown.values()) <= meta["wall_time_s"] + 1e-6
    assert meta["wall_time_s"] < TOTAL_BUDGET_S
    assert meta["total_budget_s"] == TOTAL_BUDGET_S
    assert meta["n_rows"] == EXPECTED_ROWS


def test_meta_json_records_the_acceptance_evidence(
    tiny_experiment: tuple[Path, Path],
) -> None:
    """仕様 §5 の受け入れ条件1・3・4・5 の一次資料が meta.json にある。"""
    config_path, out_dir = tiny_experiment
    assert (
        main.main(
            ["--experiment", "05", "--config", str(config_path), "--out", str(out_dir)]
        )
        == 0
    )
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    # 条件1: 同一前処理・同一基準行で6系統を比較している (D-05 / D-57)。
    uniqueness = meta["preprocessor_uniqueness"]
    assert uniqueness["max_distinct_preprocessor_ids"] == 1
    assert uniqueness["max_distinct_t0"] == 1
    # 条件3: テスト側最適化との差 (D-56)。負の行は定義上あり得ない。
    assert meta["f1_gap"]["n"] == EXPECTED_ROWS
    assert meta["f1_gap"]["n_negative"] == 0.0
    # 条件4: 順位入替の集計 (D-78)。
    protocol = meta["protocol_summary"]
    assert protocol["n_conditions"] == 8
    assert (
        protocol["n_discordant_pairs_distinguishable"]
        <= (protocol["n_discordant_pairs"])
    )
    # 条件5: 劣化点と「格子の端かどうか」(D-80)。
    size = meta["size_summary"]
    assert size["n_units_at_90pct"] in {20, 30}
    assert isinstance(size["saturated"], bool)
    # 5-A の集計 (README の数値表の出どころ)。印の根拠列まで載せる。
    assert set(meta["headline_auprc"]) == set(ANOMALY_METHODS)
    assert "control_sign_p" in meta["headline_auprc"]["esn_residual"]
    assert "cjk_font" in meta


def test_the_summaries_read_only_the_rows(tiny_experiment: tuple[Path, Path]) -> None:
    """meta.json の要約が**行から**計算されている (実験を回し直さない)。

    ``preprocessor_uniqueness`` / ``f1_gap_summary`` を CSV から読んだ行に
    相当するもの (パイプラインの戻り値) へ直接かけて、meta.json と一致する
    ことを見る。
    """
    config_path, out_dir = tiny_experiment
    config = load_config_as(config_path, Anomaly05Config)
    outputs = run_and_report_anomaly(config, out_dir)
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["preprocessor_uniqueness"] == preprocessor_uniqueness(outputs.rows)
    assert meta["f1_gap"] == f1_gap_summary(outputs.rows)
    assert meta["n_timeline_rows"] == len(outputs.timeline_rows)
    assert outputs.paths[-1].name == "meta.json"
    assert tuple(path.name for path in outputs.paths) == ANOMALY_ARTIFACTS

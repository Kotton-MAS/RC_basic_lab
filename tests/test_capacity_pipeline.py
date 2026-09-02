"""03 の「1コマンドで成果物が出る」経路と、本番設定の受け入れ条件のテスト.

前半は縮小設定を一時ディレクトリに書いて**実際に1コマンド相当を走らせ**、
``capacity.csv`` / ``capacity_profile.csv`` / ``meta.json`` が出ることを見る
(01 の ``tests/test_main.py`` / 02 の ``tests/test_esp_pipeline.py`` と同じ規律)。

後半は**コミット済みの本番成果物** (``results/03_capacity/``) を読んで受け入れ
条件1・2 を検査する (``tests/test_readme_summary.py`` が
``results/comparison_summary.csv`` を読むのと同じ形)。本番の 3-A は 54 条件で
30 秒、3-B' は 27 条件で 220 秒かかるため、pytest の中で回すと 03 が足す
テスト時間の予算 (仕様 §5: < 60 秒) を1件で使い切る。受け入れ条件は
「**この設定で実際にこうなった**」という主張なので、縮小設定で回し直しても
主張の裏付けにはならない。
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import cast

import pytest
from test_config_wiring_capacity import PENDING_SECTIONS
from wiring import experiment_config

import main
from rc_basics_lab.config import (
    Capacity03Config,
    CapacityDriveConfig,
    CapacityReservoirConfig,
    ConservationConfig,
    ESNConfig,
    IpcConfig,
    IpcSweepConfig,
    LadderSweepConfig,
    LengthSweepConfig,
    McSweepConfig,
    MemoryCapacityConfig,
    Narma10Config,
    NarmaOperatingConfig,
    RidgeConfig,
    SplitConfig,
    TopologyLadderConfig,
    load_config_as,
)
from rc_basics_lab.config._dump import as_plain_mapping
from rc_basics_lab.experiment.capacity import (
    EXPERIMENT_CONSERVATION,
    EXPERIMENT_IPC_SWEEP,
    EXPERIMENT_LENGTH_SWEEP,
    EXPERIMENT_MC_SWEEP,
    EXPERIMENT_NARMA10,
    FIGURE_EXPERIMENTS,
    n_replicates_for,
    reservoir_config_for,
    run_capacity_experiment,
    run_conservation_sweep,
)
from rc_basics_lab.experiment.capacity_pipeline import (
    CAPACITY_ARTIFACTS,
    CAPACITY_CSV,
    CAPACITY_LENGTH_CSV,
    CAPACITY_PROFILE_CSV,
    run_and_report_capacity,
    run_and_report_length_sweep,
    write_capacity_csv,
    write_capacity_profile_csv,
)
from rc_basics_lab.experiment.capacity_rows import (
    CAPACITY_CSV_COLUMNS,
    CAPACITY_PROFILE_CSV_COLUMNS,
    DIAGNOSTIC_IPC,
    DIAGNOSTIC_MC,
    CapacityCondition,
)
from rc_basics_lab.experiment.narma import run_narma10
from rc_basics_lab.experiment.report import META_JSON
from rc_basics_lab.reservoir.topology import ErdosRenyiConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "03_capacity"

CLI_BUDGET_S = 20.0
"""縮小設定の CLI が成果物を全部作るまでの上限 [秒] (仕様 §4 T2 の受け入れ基準)。"""

CSV_BUDGET_BYTES = 5 * 1024 * 1024
"""``results/03_capacity/*.csv`` の合計サイズの上限 (仕様 §5)。"""

CONSERVATION_TOLERANCE = 1.02
"""保存則 ``ipc_total <= N`` に許す上振れ (受け入れ条件2)。

有限標本のかさ上げはしきい値処理 (D-27) で大半が落ちるが、次数ごとの分位点
推定そのものに揺らぎがあるため厳密な N では割れうる。2% は仕様 §4 T2 の指定値。
"""

NOISE_MARGIN_SIGMAS = 3.0
"""ノイズ有無の差に要求する、レプリケート間 s.d. に対する倍率 (受け入れ条件2)。"""

MIN_DELAY_RATIO = 1.5
"""最大 rho の ``mc_effective_delay`` が最小 rho の何倍以上か (受け入れ条件1)。"""

ESP_BOUNDARY_RHO = 1.0
"""単調性を要求する rho の上限。

これを超える rho (本番格子では 1.1) では駆動が弱いと ESP が成立せず、
記憶容量は**下がる**。詳しくは
``test_mc_effective_delay_increases_with_rho`` の docstring。
"""

MC_RATIO_LOWER_BOUND = 0.10
"""ピーク rho(=1.0) でのリーク率別平均 ``mc_ratio`` の最低ライン (F-3b1-1-024)。

図には ``y=N`` の参照線が描かれているが、それを裏付ける定量的な下限が
実験レベルには存在しなかった (IPC 側の同種の欠落 F-03-1-023 と同型)。この assert が
実際に評価する統計量は ``min(peak_ratio_means)`` (leak_rate ごとに平均してから
最小値を取る) であり、本番実測でのその値は 0.1129 (leak=0.3) で、マージンは約13%
(``saturation_ratio >= 0.5`` や MC 診断単体の下限より薄い)。複数シードで再実測しても
min(leak別平均) は 0.1126〜0.1189 で一貫して薄く、個別 replicate 単体では 0.1057
まで下がる (F-3b1-2-001)。
"""

MC_TOTAL_UPPER_TOLERANCE = 1.02
"""``mc_total <= n_units * MC_TOTAL_UPPER_TOLERANCE`` (trivial 上限。F-3b1-1-024)。

``CONSERVATION_TOLERANCE`` と同じ値だが、3-B' (IPC の保存則) とは別の主張
(3-A: MC の保存則) を固定しているため独立した定数にする。
"""

TINY_CONFIG = """
name: capacity_cli_smoke
seeds:
  reservoir: 0
  drive: 1
  surrogate: 4
drive:
  distribution: uniform
  washout: 40
reservoir:
  input_scale: 1.0
  density: 0.3
  n_replicates: 2
mc_sweep:
  rho_grid: [0.5, 0.95]
  leak_rate_grid: [1.0]
  sigma_u: 0.3
  n_units: 12
  n_steps: 1500
ipc_sweep:
  rho_grid: [0.8]
  leak_rate_grid: [0.6, 1.0]
  sigma_u: 0.35
  n_units: 10
  n_steps: 1400
conservation:
  n_units_grid: [9]
  state_noise_grid: [0.0, 0.05]
  rho: 0.95
  leak_rate: 1.0
  sigma_u: 0.4
  n_steps: 1400
  max_delay_by_degree: [8, 4]
  n_replicates: null
length_sweep:
  n_steps_grid: [1200, 1500]
  rho: 0.9
  leak_rate: 1.0
  sigma_u: 0.3
  n_units: 8
mc:
  max_delay: 20
  n_surrogates: 5
ipc:
  max_delay_by_degree: [8, 4]
  n_surrogates: 5
  n_surrogate_targets: 2
narma:
  length: 900
  n_lags_sweep: [4, 40]
  n_replicates_sweep: 2
  base:
    name: cli_smoke_narma
    n_replicates: 2
    split:
      washout: 50
      max_start_offset: 20
    ridge:
      alpha_grid: [1.0e-2]
      n_lags_grid: [2, 4]
    tasks:
      - kind: mackey_glass
        reservoir:
          n_units: 7
          leak_rate: 1.0
          input_scale: 1.0
          topology:
            kind: erdos_renyi
            density: 0.5

# 3-T / 3-C'': **掃引の値は絶対値なので、縮小設定でも明示しないと本番の格子が
# 走る** (実測 220 秒)。CLI の予算 (20 秒) はここを書き忘れると必ず落ちる。
topology_ladder:
  n_units: 10
  n_steps: 700
  n_graphs: 1
  n_replicates: 1
  sweeps:
    - axis: n_units
      values: [8, 10]
    - axis: rho
      values: [0.8, 0.95]
narma_operating:
  n_units_grid: [20, 30]
  leak_rate_grid: [0.5, 1.0]
"""
"""縮小設定 (構造は本番と同じ)。本番は 330 秒かかるため CLI テストでは使わない。"""

EXPECTED_SWEEP_ROWS = 12
"""3-A (2x1) + 3-B (1x2) + 3-B' (1x2) = 6 条件 x 2 レプリケート。"""

EXPECTED_NARMA_ROWS = 1
"""3-C は1条件 (レプリケート0 の状態行列に対する MC / IPC)。"""

EXPECTED_ROWS = EXPECTED_SWEEP_ROWS + EXPECTED_NARMA_ROWS
"""``capacity.csv`` の行数 (掃引3本 + 3-C)。"""


def tiny_config() -> Capacity03Config:
    """``TINY_CONFIG`` と同じ内容の設定オブジェクト (YAML を経由しない版)。

    **``topology_ladder`` / ``narma_operating`` を必ず縮める。** 掃引の値は
    絶対値なので、書かないと既定 (N=25/50/100・T=5000/20000/80000) がそのまま
    走る —— 実測でこのテストが 220 秒かかった (D-146)。
    """
    return Capacity03Config(
        name="capacity_cli_smoke",
        drive=CapacityDriveConfig(distribution="uniform", washout=40),
        reservoir=CapacityReservoirConfig(input_scale=1.0, density=0.3, n_replicates=2),
        topology_ladder=TopologyLadderConfig(
            n_units=20,
            n_steps=700,
            n_graphs=1,
            n_replicates=1,
            sweeps=(
                LadderSweepConfig(axis="n_units", values=(16.0, 20.0)),
                LadderSweepConfig(axis="rho", values=(0.8, 0.95)),
            ),
        ),
        narma_operating=NarmaOperatingConfig(
            n_units_grid=(20, 30), leak_rate_grid=(0.5, 1.0)
        ),
        mc_sweep=McSweepConfig(
            rho_grid=(0.5, 0.95),
            leak_rate_grid=(1.0,),
            sigma_u=0.3,
            n_units=12,
            n_steps=1500,
        ),
        ipc_sweep=IpcSweepConfig(
            rho_grid=(0.8,),
            leak_rate_grid=(0.6, 1.0),
            sigma_u=0.35,
            n_units=10,
            n_steps=1400,
        ),
        conservation=ConservationConfig(
            n_units_grid=(9,),
            state_noise_grid=(0.0, 0.05),
            rho=0.95,
            leak_rate=1.0,
            sigma_u=0.4,
            n_steps=1400,
            max_delay_by_degree=(8, 4),
        ),
        length_sweep=LengthSweepConfig(
            n_steps_grid=(1200, 1500),
            rho=0.9,
            leak_rate=1.0,
            sigma_u=0.3,
            n_units=8,
        ),
        mc=MemoryCapacityConfig(max_delay=20, n_surrogates=5),
        ipc=IpcConfig(
            max_delay_by_degree=(8, 4), n_surrogates=5, n_surrogate_targets=2
        ),
        narma=Narma10Config(
            n_lags_sweep=(4, 40),
            n_replicates_sweep=2,
            length=900,
            base=experiment_config(
                name="cli_smoke_narma",
                n_replicates=2,
                split=SplitConfig(washout=50, max_start_offset=20),
                ridge=RidgeConfig(alpha_grid=(1e-2,), n_lags_grid=(2, 4)),
                esn_mackey_glass=ESNConfig(
                    n_units=7,
                    leak_rate=1.0,
                    input_scale=1.0,
                    topology=ErdosRenyiConfig(density=0.5),
                ),
            ),
        ),
    )


@pytest.fixture
def tiny_experiment(tmp_path: Path) -> tuple[Path, Path]:
    """縮小設定を ``--experiment 03`` に差し替える (本番は 330 秒かかるため)。"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(TINY_CONFIG, encoding="utf-8")
    return config_path, tmp_path / "out"


VOLATILE_COLUMNS: tuple[str, ...] = (
    "wall_time_state_s",
    "wall_time_mc_s",
    "wall_time_ipc_s",
    "wall_time_s",
)
"""実行ごとに変わる実測時間の列 (**列そのものは CSV に残る**)。"""


def _without_wall_time(csv_text: str) -> list[list[str]]:
    """CSV から実測時間の列だけ落とす (02 の ``_without_wall_time`` と同じ形)。"""
    rows = [line.split(",") for line in csv_text.strip().splitlines()]
    dropped = {rows[0].index(name) for name in VOLATILE_COLUMNS}
    return [
        [field for index, field in enumerate(row) if index not in dropped]
        for row in rows
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# --- 1コマンドで成果物が出る (受け入れ条件7) ---------------------------------


def test_artifacts_are_regenerated_in_one_command_within_the_budget(
    tiny_experiment: tuple[Path, Path],
) -> None:
    """縮小設定の CLI が 20 秒以内に ``CAPACITY_ARTIFACTS`` を全部作る。

    宣言 (``CAPACITY_ARTIFACTS``) と実体 (出力ディレクトリの走査) の**両方**を
    見る。宣言だけを見るテストは、図を1枚落としたのに定数から消し忘れた
    (またはその逆) を黙って通す (02 の
    ``test_all_four_figures_and_two_csv_in_one_command`` と同じ理由)。
    """
    config_path, out_dir = tiny_experiment
    started = time.perf_counter()
    assert (
        main.main(
            ["--experiment", "03", "--config", str(config_path), "--out", str(out_dir)]
        )
        == 0
    )
    elapsed = time.perf_counter() - started
    assert elapsed < CLI_BUDGET_S, f"縮小設定の CLI が {elapsed:.1f}s かかりました"
    for name in CAPACITY_ARTIFACTS:
        assert (out_dir / name).is_file(), f"{name} が生成されていません"
    produced = sorted(path.name for path in out_dir.iterdir() if path.is_file())
    assert set(produced) == set(CAPACITY_ARTIFACTS), (
        "生成物と CAPACITY_ARTIFACTS の宣言が食い違っています"
    )
    # 系列長掃引は本番に含めない (仕様 §8)。CLI が黙って回していないこと。
    assert CAPACITY_LENGTH_CSV not in produced


def test_length_sweep_is_not_part_of_the_production_artifacts(
    tiny_experiment: tuple[Path, Path],
) -> None:
    """``--variant length`` は ``capacity_length.csv`` だけを書く。

    ``threshold-02`` (02 の閾値感度) と同型の分離である。本番
    (``make figures-03``) に含めると T=1e6 の掃引が 900 秒予算に紛れ込む。
    """
    config_path, out_dir = tiny_experiment
    assert (
        main.main(
            [
                "--experiment",
                "03",
                "--variant",
                "length",
                "--config",
                str(config_path),
                "--out",
                str(out_dir),
            ]
        )
        == 0
    )
    produced = sorted(path.name for path in out_dir.iterdir() if path.is_file())
    assert produced == [CAPACITY_LENGTH_CSV]
    rows = _read_csv(out_dir / CAPACITY_LENGTH_CSV)
    assert {row["experiment"] for row in rows} == {EXPERIMENT_LENGTH_SWEEP}
    assert [int(row["n_steps"]) for row in rows] == [1200, 1500, 1200, 1500]


def test_meta_json_records_the_wall_time_breakdown(
    tiny_experiment: tuple[Path, Path],
) -> None:
    """``meta.json`` に区間ごとの実測時間の内訳が載る (仕様 §5)。

    予算 (状態生成の合計 < 60 秒) を割ったときに「診断が重いのか状態生成が
    重いのか」を成果物だけで切り分けられる必要がある。3a の reviewer の警告
    (「予算超過が起きるとすれば診断計算ではなくリザバー状態生成側」) を
    成果物の側で検証可能にするのがこの内訳である。
    """
    config_path, out_dir = tiny_experiment
    assert (
        main.main(
            ["--experiment", "03", "--config", str(config_path), "--out", str(out_dir)]
        )
        == 0
    )
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    breakdown = meta["wall_time_breakdown"]
    assert [item["experiment"] for item in breakdown] == list(FIGURE_EXPERIMENTS)
    expected_conditions = {
        EXPERIMENT_MC_SWEEP: 4,
        EXPERIMENT_IPC_SWEEP: 4,
        EXPERIMENT_CONSERVATION: 4,
        # 3-C は掃引ではなく1条件 (レプリケート0 の状態行列)
        EXPERIMENT_NARMA10: EXPECTED_NARMA_ROWS,
    }
    for item in breakdown:
        assert item["n_conditions"] == expected_conditions[item["experiment"]]
        for key in ("wall_time_state_s", "wall_time_mc_s", "wall_time_ipc_s"):
            assert item[key] > 0.0, (item["experiment"], key)
    assert meta["n_rows"] == EXPECTED_ROWS
    assert meta["n_profile_rows"] > 0
    assert meta["n_narma10_rows"] == 4 * 2  # 4手法 x 2レプリケート (D-90)
    # 3-C は向きを問わないが、どちらに転んでも同じ形で残る (仕様 §4 T4)
    verdict = meta["narma10_verdict"]
    assert set(verdict["nmse_mean"]) == {
        "linear",
        "delay_line",
        "delay_line_ols",
        "esn",
    }
    # D-90: 正則化の有無で結論が変わったかが成果物だけで読める
    assert isinstance(verdict["delay_line_ols_beats_esn"], bool)
    assert verdict["regularisation_changes_the_verdict"] == (
        verdict["delay_line_ols_beats_esn"] != verdict["delay_line_beats_esn"]
    )
    assert verdict["best_method"] in verdict["nmse_mean"]
    assert isinstance(verdict["delay_line_beats_esn"], bool)
    assert verdict["reference_nmse"] == {"linear_ceiling": 0.16, "nonlinear_rc": 0.107}
    assert "未特定" not in verdict["reference_note"]
    assert "Vinckier" in verdict["reference_note"]


# --- CSV の形 (D-38) ---------------------------------------------------------


@pytest.mark.parametrize(
    "max_delay_by_degree",
    [pytest.param((8, 4), id="2degrees"), pytest.param((8, 4, 3), id="3degrees")],
)
def test_profile_csv_columns_are_static_and_cells_are_positive(
    tmp_path: Path, max_delay_by_degree: tuple[int, ...]
) -> None:
    """``capacity_profile.csv`` の列が cfg に依らず一定で、正値セルだけが載る (D-38)。

    IPC の ``scalars`` は ``ipc_threshold_degree{d}`` を次数の本数だけ持つので
    (F-03-1-005)、次数を1本増やすとキー集合が変わる。これを列にすると CSV の
    列が cfg 依存になるため、次数と遅延を**行の値**に落とす。ここでは打ち切りの
    本数が違う2つの設定で**同じ列**が出ることを実測する。

    ``capacity <= 0`` の行が0件であることも同時に測る。全セルを書くと本番設定で
    約6万行になり、``results/`` はコミット対象なのでリポジトリが重くなる。
    絞り込みの条件は IPC の ``n_targets_kept`` (``count_nonzero(kept)``) と
    同じ ``> 0`` だが、**行数は n_targets_kept と一致しない** (F-3b1-1-003)。
    長形式の行は (次数,遅延) のヒートマップセル単位、``n_targets_kept`` は
    目標単位で単位が異なり、1セルに複数目標が畳み込まれるため行数は
    ``n_targets_kept`` 以下にしかならない (assert は ``<=`` だけを要求する)。
    成果物単体で検算できる不変条件は、``capacity`` 列の**総和**が
    ``mc_total`` / ``ipc_total`` と一致することであり、これを直接固定する
    (D-38)。
    """
    config = dataclasses.replace(
        tiny_config(),
        ipc=dataclasses.replace(
            tiny_config().ipc, max_delay_by_degree=max_delay_by_degree
        ),
    )
    results = run_capacity_experiment(config)
    path = write_capacity_profile_csv(
        results.profile_rows, tmp_path / CAPACITY_PROFILE_CSV
    )
    rows = _read_csv(path)
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert tuple(header) == CAPACITY_PROFILE_CSV_COLUMNS
    assert rows, "正値セルが1件も無い設定ではこのテストが空振りします"
    assert [row for row in rows if float(row["capacity"]) <= 0.0] == []
    assert {row["diagnostic"] for row in rows} <= {DIAGNOSTIC_MC, DIAGNOSTIC_IPC}
    for row in rows:
        assert int(row["delay"]) >= 1
        assert int(row["degree"]) == 1 or row["diagnostic"] == DIAGNOSTIC_IPC

    # 行数は n_targets_kept の**上限**にしかならない (セル単位 vs 目標単位)。
    # IPC の heatmap は (次数, max(k_i)) のセルへ足し込むので、行数は
    # n_targets_kept 以下で、少なくとも1セルは埋まる。
    for outcome in results.outcomes:
        ipc_cells = [
            row
            for row in results.profile_rows
            if row.diagnostic == DIAGNOSTIC_IPC
            and row.replicate == outcome.row.replicate
            and row.experiment == outcome.row.experiment
            and row.rho == outcome.row.rho
            and row.leak_rate == outcome.row.leak_rate
            and row.n_units == outcome.row.n_units
            and row.state_noise == outcome.row.state_noise
        ]
        mc_cells = [
            row
            for row in results.profile_rows
            if row.diagnostic == DIAGNOSTIC_MC
            and row.replicate == outcome.row.replicate
            and row.experiment == outcome.row.experiment
            and row.rho == outcome.row.rho
            and row.leak_rate == outcome.row.leak_rate
            and row.n_units == outcome.row.n_units
            and row.state_noise == outcome.row.state_noise
        ]
        assert len(ipc_cells) <= outcome.row.n_targets_kept
        assert bool(ipc_cells) == (outcome.row.n_targets_kept > 0)

        # D-38 の正本の不変条件: capacity 列の総和が ipc_total / mc_total と
        # 一致する (行数の一致ではない)。
        assert math.isclose(
            sum(cell.capacity for cell in ipc_cells),
            outcome.row.ipc_total,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ), outcome.row.experiment
        assert math.isclose(
            sum(cell.capacity for cell in mc_cells),
            outcome.row.mc_total,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ), outcome.row.experiment


def test_capacity_csv_has_no_missing_values(tmp_path: Path) -> None:
    """``capacity.csv`` に空欄・``nan`` が1件も無い (全列が常に埋まる)。

    cfg 依存で本数が変わる量を列にすると、設定によって空欄が生まれる。D-38 の
    長形式はそれを避けるための設計なので、「1行 = 1条件で全列が埋まる」が
    崩れていないことをここで固定する。
    """
    config = tiny_config()
    results = run_capacity_experiment(config)
    # 3-C の行も同じ CSV に合流するので、ここで一緒に検査する (合流させた
    # 行だけ空欄が入っても掃引側のテストでは気づけない)。
    all_rows = (*results.rows, run_narma10(config).capacity.row)
    path = write_capacity_csv(all_rows, tmp_path / CAPACITY_CSV)
    rows = _read_csv(path)
    assert len(rows) == EXPECTED_ROWS
    assert {row["experiment"] for row in rows} == {
        EXPERIMENT_MC_SWEEP,
        EXPERIMENT_IPC_SWEEP,
        EXPERIMENT_CONSERVATION,
        EXPERIMENT_NARMA10,
    }
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert tuple(header) == CAPACITY_CSV_COLUMNS
    for row in rows:
        assert set(row) == set(CAPACITY_CSV_COLUMNS)
        for column, value in row.items():
            assert value != "", f"{column} が空欄です"
            assert value.lower() not in {"nan", "none", "inf", "-inf"}, (
                f"{column} = {value}"
            )
            if column not in {"experiment"}:
                assert math.isfinite(float(value)), f"{column} = {value}"


# --- セクションの scope (仕様 §5 有効性観点) ----------------------------------


def test_conservation_section_does_not_change_the_other_experiments(
    tmp_path: Path,
) -> None:
    """``conservation.*`` を変えても 3-A / 3-B の行が**バイト単位で**変わらない。

    3-B' だけが IPC の打ち切り (``ipc_config_for``) とレプリケート数
    (``n_replicates_for``) を上書きする。上書きが片方向であることは
    ``tests/test_config_wiring_capacity.py`` の scope 検査が値のレベルで測るが、
    ここは**書き出した CSV のバイト列**で測る —— 上書きが ``config.ipc`` を
    その場で書き換える実装 (``dataclasses.replace`` ではなく可変な共有状態) に
    なると、3-A / 3-B の行も静かに変わる。
    """
    base = tiny_config()
    changed = dataclasses.replace(
        base,
        conservation=dataclasses.replace(
            base.conservation,
            n_units_grid=(7, 11),
            state_noise_grid=(0.0, 0.02, 0.08),
            rho=0.8,
            leak_rate=0.5,
            sigma_u=0.6,
            n_steps=1600,
            max_delay_by_degree=(6, 3),
            n_replicates=1,
        ),
    )

    def other_rows(config: Capacity03Config) -> bytes:
        results = run_capacity_experiment(config)
        rows = [
            row
            for row in results.rows
            if row.experiment in {EXPERIMENT_MC_SWEEP, EXPERIMENT_IPC_SWEEP}
        ]
        path = write_capacity_csv(rows, tmp_path / f"{config.conservation.rho}.csv")
        # 実測時間の4列だけは実行ごとに変わるので落とす (列そのものは残る)。
        stripped = _without_wall_time(path.read_text(encoding="utf-8"))
        return "\n".join(",".join(row) for row in stripped).encode("utf-8")

    assert other_rows(base) == other_rows(changed)


def test_conservation_replicates_default_to_the_shared_value_and_can_be_overridden() -> (  # noqa: E501  (仕様 §4 T2 が指定したテスト名。改名すると受け入れ基準との対応が切れる)
    None
):
    """``conservation.n_replicates`` は継承と上書きの**両方**が効く。

    仕様 §7 リスク1 の縮退規則 (「合計見積りが 700 秒を超えた場合に許可される
    調整は ``conservation.n_replicates`` を 3 → 1 に落とすことだけ」) に対応する
    フィールドで、``None`` なら横断共有の ``reservoir.n_replicates`` を継承する。
    継承だけを測ると「上書きが無視されている」実装が、上書きだけを測ると
    「常にセクション側の既定 (None) を int と誤って扱う」実装が通ってしまう。
    """
    base = tiny_config()
    assert base.conservation.n_replicates is None
    assert n_replicates_for(base, EXPERIMENT_CONSERVATION) == 2
    assert n_replicates_for(base, EXPERIMENT_MC_SWEEP) == 2
    assert len(run_conservation_sweep(base)) == 4

    overridden = dataclasses.replace(
        base, conservation=dataclasses.replace(base.conservation, n_replicates=1)
    )
    assert n_replicates_for(overridden, EXPERIMENT_CONSERVATION) == 1
    # 上書きは片方向: 3-A / 3-B は横断共有の値のまま
    assert n_replicates_for(overridden, EXPERIMENT_MC_SWEEP) == 2
    assert len(run_conservation_sweep(overridden)) == 2

    with pytest.raises(ValueError, match="レプリケート数"):
        n_replicates_for(
            dataclasses.replace(
                base,
                conservation=dataclasses.replace(base.conservation, n_replicates=0),
            ),
            EXPERIMENT_CONSERVATION,
        )


def test_reservoir_config_for_carries_the_effective_replicate_count() -> None:
    """``reservoir_config_for`` は実効レプリケート数を渡す (F-3b1-1-006)。

    以前は横断共有の ``reservoir.n_replicates`` を無条件で詰めていたため、
    3-B' で ``conservation.n_replicates`` が上書きされている場合に実効値と
    食い違った ``ReservoirSweepConfig`` が境界を越えていた (実効1に対し3が
    渡る)。``simulate_reference_trajectory`` は ``n_replicates`` を読まないため
    現状は実害が無いが、``drive_config_for`` が ``n_pairs`` を外しているのと
    同じ規律 (『設定したのに効いていないフィールドを作らない』) を
    ``reservoir_config_for`` にも適用する。
    """
    config = dataclasses.replace(
        tiny_config(),
        reservoir=dataclasses.replace(tiny_config().reservoir, n_replicates=3),
        conservation=dataclasses.replace(tiny_config().conservation, n_replicates=1),
    )
    mc_condition = CapacityCondition(
        experiment=EXPERIMENT_MC_SWEEP,
        rho=0.9,
        leak_rate=0.6,
        n_units=config.mc_sweep.n_units,
        state_noise=0.0,
        sigma_u=config.mc_sweep.sigma_u,
        n_steps=config.mc_sweep.n_steps,
        replicate=0,
    )
    conservation_condition = dataclasses.replace(
        mc_condition, experiment=EXPERIMENT_CONSERVATION
    )
    assert reservoir_config_for(config, mc_condition).n_replicates == 3
    assert reservoir_config_for(config, conservation_condition).n_replicates == 1


# --- 本番成果物に対する受け入れ条件 ------------------------------------------


def _production_rows() -> list[dict[str, str]]:
    path = RESULTS / CAPACITY_CSV
    assert path.is_file(), "make figures-03 を実行してください"
    rows = _read_csv(path)
    assert rows, f"{path} が空です"
    return rows


def test_production_config_matches_the_committed_results() -> None:
    """本番 YAML の条件数と ``capacity.csv`` の行数が一致する。

    成果物が古い設定で生成されたまま取り残されると、以下の受け入れ条件の
    テストは「別の設定の結果」を検査することになる。
    """
    config = load_config_as(
        ROOT / "experiments" / "03_capacity" / "config.yaml", Capacity03Config
    )
    expected = {
        EXPERIMENT_MC_SWEEP: len(config.mc_sweep.rho_grid)
        * len(config.mc_sweep.leak_rate_grid)
        * n_replicates_for(config, EXPERIMENT_MC_SWEEP),
        EXPERIMENT_IPC_SWEEP: len(config.ipc_sweep.rho_grid)
        * len(config.ipc_sweep.leak_rate_grid)
        * n_replicates_for(config, EXPERIMENT_IPC_SWEEP),
        EXPERIMENT_CONSERVATION: len(config.conservation.n_units_grid)
        * len(config.conservation.state_noise_grid)
        * n_replicates_for(config, EXPERIMENT_CONSERVATION),
        # 3-C は掃引ではないので常に1行 (レプリケート0 の状態行列)
        EXPERIMENT_NARMA10: 1,
    }
    counts: dict[str, int] = defaultdict(int)
    for row in _production_rows():
        counts[row["experiment"]] += 1
    assert dict(counts) == expected


def _normalize_for_json_comparison(value: object) -> object:
    """タプルをリストへ揃える (``dataclasses.asdict`` と JSON の型差を吸収)。

    ``dataclasses.asdict`` は掃引軸などのタプルフィールドをそのまま ``tuple``
    で返すが、``meta.json`` は JSON 化されているのでリストになる。値そのものの
    比較には無関係な差なので、比較前に両者をリストへ正規化する。
    """
    if isinstance(value, tuple | list):
        return [_normalize_for_json_comparison(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_for_json_comparison(val) for key, val in value.items()}
    return value


def test_production_config_matches_the_committed_meta_json() -> None:
    """本番 YAML の**値**が ``meta.json`` にダンプされた設定と一致する (F-3b1-1-025)。

    ``test_production_config_matches_the_committed_results`` は実験ごとの
    **行数**しか見ておらず、格子の**値**が変わっても検出しない (``rho_grid``
    の1点を差し替えても行数は変わらないため通過してしまう)。この場合、値を
    そのまま真実として使う ``test_mc_effective_delay_increases_with_rho`` /
    ``test_conservation_respects_the_bound`` は設定と成果物が食い違ったまま
    受け入れ条件1・2 の「通過」を続けてしまう。``meta.json['config']`` は
    ``run_and_report_capacity`` が書き出し時点の設定を丸ごとダンプしたもの
    (``config._dump.as_plain_mapping``。D-129 で判別子つきになった) なので、
    現在の ``load_config_as(...)`` と突き合わせれば値ドリフトを丸ごと閉じられる。

    ``PENDING_SECTIONS`` (``tests/test_config_wiring_capacity.py``) に含まれる
    セクション (現状は ``narma`` のみ) は比較対象から除く。``figures-03`` の
    成果物はこのセクションに一切依存しないため (3-C は 3b-2 の T4 で本番経路に
    入るまで不活性)、ここに含めると T4 が narma を配線する最初の一手 (例:
    D-39 に合わせた ``n_units`` の変更) で無関係な CSV・図の再生成を要求して
    しまう。``PENDING_SECTIONS`` から引いているので、T4 が narma を配線して
    ``PENDING_SECTIONS`` から外れれば1か所の変更で自動的に比較対象へ戻る。
    """
    config = load_config_as(
        ROOT / "experiments" / "03_capacity" / "config.yaml", Capacity03Config
    )
    # meta.json と**同じ経路**で落とす (D-129)。asdict は判別子を落とすので、
    # ここだけ asdict のままだと「記録には kind があるのに比較側には無い」で
    # 永久に食い違う。
    current = cast(
        "dict[str, object]",
        _normalize_for_json_comparison(as_plain_mapping(config)),
    )
    meta_path = RESULTS / META_JSON
    assert meta_path.is_file(), "make figures-03 を実行してください"
    committed = json.loads(meta_path.read_text(encoding="utf-8"))["config"]
    for section in PENDING_SECTIONS:
        current.pop(section, None)
        committed.pop(section, None)
    assert current == committed, (
        "experiments/03_capacity/config.yaml の値が"
        " results/03_capacity/meta.json に記録された設定と食い違っています"
        " (make figures-03 で成果物を再生成してください)"
    )


def test_mc_effective_delay_increases_with_rho() -> None:
    """受け入れ条件1: 本番設定で ``mc_effective_delay`` が rho とともに伸びる。

    4つの主張を分けて測る。

    1. **rho <= 1.0 で単調非減少** (リーク率ごとに、レプリケート平均で)。
    2. **格子の最大 rho (1.1) の値が最小 rho (0.5) の 1.5 倍以上**。
    3. (未使用にしない) rho=1.1 で実際に下がる。
    4. **ピーク rho (=1.0) でのリーク率別平均 mc_ratio が最低ラインを上回る**
       (F-3b1-1-024)。図に描かれる ``y=N`` の参照線が『N が上限として意識
       されている』という印象を与えるが、それを裏付ける定量的な下限が
       これまで実験レベルに存在しなかった。あわせて ``mc_total`` の
       trivial 上限 (``<= n_units * 1.02``) も全行で固定する
       (成果物単体で保存則を検算できるようにする)。

    単調性を rho <= 1.0 に限るのは、rho > 1 では駆動が弱いと ESP が成立せず
    記憶容量が**下がる**ためである (記憶容量は臨界点近傍で最大になる)。
    これは 3-A が見せたい現象そのものであり、駆動強度の較正 (T2-5) でも
    解消しない: sigma_u in {0.05, 0.1, 0.2, 0.5} の4点すべてで、どこかの
    リーク率で rho=1.1 が直前の点を下回る (仕様 §7 リスク2 の (a) は切り分け
    済み)。3点目の assert が「rho=1.1 で実際に下がる」ことを固定するので、
    この制限が不要になった (= 物理が変わった) 場合はここが赤くなる。
    """
    by_axis: dict[tuple[float, float], list[float]] = defaultdict(list)
    ratio_by_axis: dict[tuple[float, float], list[float]] = defaultdict(list)
    for row in _production_rows():
        if row["experiment"] == EXPERIMENT_MC_SWEEP:
            key = (float(row["leak_rate"]), float(row["rho"]))
            by_axis[key].append(float(row["mc_effective_delay"]))
            ratio_by_axis[key].append(float(row["mc_ratio"]))
            bound = int(row["n_units"]) * MC_TOTAL_UPPER_TOLERANCE
            assert float(row["mc_total"]) <= bound, row
    assert by_axis, "3-A の行がありません"
    leaks = sorted({leak for leak, _ in by_axis})
    rhos = sorted({rho for _, rho in by_axis})
    assert len(rhos) >= 3

    for leak in leaks:
        means = [statistics.fmean(by_axis[(leak, rho)]) for rho in rhos]
        within_esp = [
            value
            for rho, value in zip(rhos, means, strict=True)
            if rho <= ESP_BOUNDARY_RHO
        ]
        assert within_esp == sorted(within_esp), (
            f"leak={leak}: rho<=1.0 で mc_effective_delay が単調非減少ではありません"
            f" ({within_esp})"
        )
        assert means[-1] >= MIN_DELAY_RATIO * means[0], (
            f"leak={leak}: 最大 rho / 最小 rho = {means[-1] / means[0]:.3f}"
        )
        assert means[-1] < max(means), (
            f"leak={leak}: rho={rhos[-1]} が格子の最大値になっています"
            " (ESP 領域に単調性を限る根拠が消えたので受け入れ条件1 を見直すこと)"
        )

    peak_ratio_means = [
        statistics.fmean(ratio_by_axis[(leak, ESP_BOUNDARY_RHO)]) for leak in leaks
    ]
    assert min(peak_ratio_means) >= MC_RATIO_LOWER_BOUND, (
        f"rho={ESP_BOUNDARY_RHO} でのリーク率別平均 mc_ratio が最低ラインを"
        f" 下回っています ({peak_ratio_means})"
    )


def test_conservation_respects_the_bound() -> None:
    """受け入れ条件2: ``ipc_total <= N``、かつノイズ下では厳密に小さい。

    3つを測る。

    1. 全条件で ``ipc_total <= n_units * 1.02`` (Dambre 2012 の保存則)。
    2. ``state_noise > 0`` の総容量が ``state_noise = 0`` より**小さい**。
    3. その差がレプリケート間 s.d. の 3 倍以上 (ばらつきで説明できない)。

    3 を要求するのは、差の向きだけでは「たまたまそちらに転んだ」と区別できない
    ためである (02 の washout 感度で同じ規律を使っている)。

    **差は同じレプリケート番号どうしで取る (対応のある比較)**。レプリケート
    番号はリザバー重み (``SeedStream.RESERVOIR``) と駆動信号
    (``SeedStream.TASK``) の両方を決めるので、``state_noise`` だけが違う2条件は
    **同じリザバー・同じ入力**を共有する (共通乱数法。D-37 がしきい値のシードに
    ついて言っているのと同じ設計)。対応を無視してセルごとの s.d. を使うと、
    測りたいノイズの効果ではなく「リザバーの引きの良し悪し」がばらつきとして
    分母に乗る —— 実測 (本番 N=25, noise=0.01): 対応なしの s.d. は 2.99
    (差 7.41 の 2.5 倍にしかならない) だが、対応のある差の s.d. は 1.04 で
    比は 7.16 になる。3条件とも同じ向きに動いているのに検出できないのは
    検定の側の問題である。
    """
    rows = [
        row
        for row in _production_rows()
        if row["experiment"] == EXPERIMENT_CONSERVATION
    ]
    assert rows, "3-B' の行がありません"
    for row in rows:
        bound = int(row["n_units"]) * CONSERVATION_TOLERANCE
        assert float(row["ipc_total"]) <= bound, row
        if float(row["state_noise"]) > 0.0:
            # ノイズ下では厳密に N 未満 (受け入れ条件2 の後半)
            assert float(row["ipc_total"]) < int(row["n_units"]), row

    total: dict[tuple[int, float, int], float] = {}
    for row in rows:
        key = (int(row["n_units"]), float(row["state_noise"]), int(row["replicate"]))
        assert key not in total, f"条件が重複しています: {key}"
        total[key] = float(row["ipc_total"])
    units = sorted({n_units for n_units, _, _ in total})
    noises = sorted({noise for _, noise, _ in total})
    replicates = sorted({replicate for _, _, replicate in total})
    assert noises[0] == 0.0, "ノイズ無しの基準点が格子にありません"
    assert len(noises) >= 2
    assert len(replicates) >= 2, "レプリケート間 s.d. を取るには2本以上が要ります"

    for n_units in units:
        for noise in noises[1:]:
            diffs = [
                total[(n_units, 0.0, replicate)] - total[(n_units, noise, replicate)]
                for replicate in replicates
            ]
            gap = statistics.fmean(diffs)
            spread = statistics.stdev(diffs)
            assert gap > 0.0, f"N={n_units}, noise={noise}: 差の向きが逆です ({gap})"
            assert gap >= NOISE_MARGIN_SIGMAS * spread, (
                f"N={n_units}, noise={noise}: 差 {gap:.4f} が"
                f" レプリケート間 s.d. {spread:.4f} の {NOISE_MARGIN_SIGMAS} 倍未満です"
            )


def test_production_csv_files_fit_in_the_budget() -> None:
    """``results/03_capacity/*.csv`` の合計が 5 MB 未満 (仕様 §5)。

    ``results/`` はコミット対象なので、長形式の CSV (D-38) が全セルを書くように
    退行すると、リポジトリの重量として跳ね返る。
    """
    csv_files = sorted(RESULTS.glob("*.csv"))
    assert csv_files, "make figures-03 を実行してください"
    total = sum(path.stat().st_size for path in csv_files)
    assert total < CSV_BUDGET_BYTES, (
        f"CSV の合計が {total / 1024 / 1024:.2f} MB です: "
        + ", ".join(f"{path.name}={path.stat().st_size}" for path in csv_files)
    )


def test_production_profile_rows_are_positive_and_reference_the_same_conditions() -> (
    None
):
    """本番の ``capacity_profile.csv`` も正値のみで、条件が ``capacity.csv`` と揃う。

    縮小設定での D-38 の検査 (上) と対にして、**コミット済みの成果物**が同じ
    性質を持つことを固定する。長形式の行が別の実験・別のレプリケートを指して
    いたら、図 (T3) は存在しない条件を描くことになる。

    D-38 (訂正後) の rule は「成果物から保存則を検算する不変条件は、
    ``capacity_profile.csv`` の capacity 列の総和が ``capacity.csv`` の
    ``ipc_total`` / ``mc_total`` と一致すること」。これまでこの検算は
    ``tiny_config`` を回す
    ``test_profile_csv_columns_are_static_and_cells_are_positive`` の中にしか
    無く、**本番成果物側では検査されていなかった** (round1 で指摘された
    「行数 == n_targets_kept」という本番で成立しない不変条件の再発と同型の
    構図 — 記録が本番成果物について述べているのに、本番成果物側にその検査が
    無い)。条件ごとに diagnostic 別の capacity 列の総和を取り、両 CSV は
    既に読み込み済みなので追加コストはほぼ0で突き合わせる。
    """
    profile = _read_csv(RESULTS / CAPACITY_PROFILE_CSV)
    assert profile, "capacity_profile.csv が空です"
    assert [row for row in profile if float(row["capacity"]) <= 0.0] == []

    def key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
        return (
            row["experiment"],
            row["replicate"],
            row["rho"],
            row["leak_rate"],
            row["n_units"],
            row["state_noise"],
        )

    assert {key(row) for row in profile} <= {key(row) for row in _production_rows()}
    assert {row["diagnostic"] for row in profile} == {DIAGNOSTIC_MC, DIAGNOSTIC_IPC}

    diagnostic_sums: dict[tuple[str, str, str, str, str, str], dict[str, float]] = (
        defaultdict(lambda: defaultdict(float))
    )
    for row in profile:
        diagnostic_sums[key(row)][row["diagnostic"]] += float(row["capacity"])
    for row in _production_rows():
        sums = diagnostic_sums[key(row)]
        assert math.isclose(
            sums.get(DIAGNOSTIC_MC, 0.0),
            float(row["mc_total"]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ), (row["experiment"], row["replicate"])
        assert math.isclose(
            sums.get(DIAGNOSTIC_IPC, 0.0),
            float(row["ipc_total"]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ), (row["experiment"], row["replicate"])


def test_run_and_report_capacity_returns_what_it_wrote(tmp_path: Path) -> None:
    """戻り値の行と書き出した CSV の行数が一致する (図 (T3) が読むのは戻り値)。

    **予算も同時に見る。** 掃引の値は絶対値なので、縮小設定の節を書き忘れると
    ここだけが静かに 220 秒になる (実測、D-146)。行数の検査は通ったままなので、
    時間を見ないと気づけない。
    """
    started = time.perf_counter()
    outputs = run_and_report_capacity(tiny_config(), tmp_path)
    elapsed = time.perf_counter() - started
    assert elapsed < CLI_BUDGET_S, (
        f"縮小設定の合成層が {elapsed:.1f}s かかりました "
        "(topology_ladder / narma_operating を縮め忘れていませんか)"
    )
    assert len(outputs.rows) == EXPECTED_ROWS
    assert len(_read_csv(tmp_path / CAPACITY_CSV)) == EXPECTED_ROWS
    assert len(_read_csv(tmp_path / CAPACITY_PROFILE_CSV)) == len(outputs.profile_rows)
    assert tuple(path.name for path in outputs.paths) == CAPACITY_ARTIFACTS
    assert sum(timing.n_conditions for timing in outputs.timings) == EXPECTED_ROWS


def test_run_and_report_length_sweep_writes_only_the_length_csv(tmp_path: Path) -> None:
    """系列長掃引は ``capacity_length.csv`` 1枚だけを書く。"""
    path = run_and_report_length_sweep(tiny_config(), tmp_path)
    assert path.name == CAPACITY_LENGTH_CSV
    assert sorted(item.name for item in tmp_path.iterdir()) == [CAPACITY_LENGTH_CSV]
    rows = _read_csv(path)
    assert {row["experiment"] for row in rows} == {EXPERIMENT_LENGTH_SWEEP}

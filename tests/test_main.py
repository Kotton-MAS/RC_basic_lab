"""CLI エントリのテスト — 「1コマンドで5成果物」の担保 (受け入れ条件5).

``main.py --experiment 01`` と ``main.py (--experiment 01) --config ...``
はどちらも ``pipeline.run_and_report`` を呼ぶ薄い層である。ここでは縮小設定を
一時ディレクトリに書いて**実際に1コマンド相当を走らせ**、
``comparison.csv`` / ``comparison_summary.csv`` / ``fig_comparison.png`` /
``fig_state_space.png`` / ``meta.json`` の5点が出ることと、PNG の実測解像度が
retina 相当であることを見る。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from conftest import png_dpi

import main
from rc_basics_lab.experiment import catalog
from rc_basics_lab.experiment.pipeline import ARTIFACTS
from rc_basics_lab.experiment.state_space import (
    DELAY_EMBEDDED_INPUT,
    RAW_INPUT,
    RESERVOIR_STATE,
)

RETINA_DPI = 200
EXPECTED_ROWS = 6  # 2課題 x 3手法 x 1レプリケート

TINY_CONFIG = """
name: cli_smoke
n_replicates: 1
seeds:
  reservoir: 0
  task: 1
  split: 2
split:
  train_ratio: 0.5
  val_ratio: 0.25
  test_ratio: 0.25
  washout: 20
  max_start_offset: 10
ridge:
  alpha_grid: [1.0e-4, 1.0e-1]
  n_lags_grid: [1, 4]
tasks:
  - kind: mackey_glass
    params:
      length: 500
      integration_burn_in: 50
    reservoir:
      n_units: 20
      topology:
        kind: erdos_renyi
        density: 0.3
  - kind: delay_parity
    params:
      n_bits: 2
      delay: 1
      length: 500
    reservoir:
      n_units: 20
      topology:
        kind: erdos_renyi
        density: 0.3
      leak_rate: 1.0
      input_scale: 1.0
"""


@pytest.fixture
def tiny_experiment(tmp_path: Path) -> tuple[Path, Path]:
    """縮小設定を書き出す (本番設定は数十秒かかるため)。

    設定の差し替えは ``--config`` で行う (D-125)。かつては
    ``EXPERIMENTS`` の値を monkeypatch していたが、カタログの
    ``config_path`` は ``name`` から導く property になったので、CLI の
    引数で渡すほうが**実際の経路をそのまま通る**。
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(TINY_CONFIG, encoding="utf-8")
    return config_path, tmp_path / "out"


def test_experiment_registry_points_at_existing_configs() -> None:
    """登録済みの実験番号の設定ファイルが実在する。"""
    assert catalog.BY_NUMBER
    for number, spec in catalog.BY_NUMBER.items():
        assert spec.config_path.is_file(), (
            f"実験 {number} の設定が見つかりません: {spec.config_path}"
        )


def test_experiment_registry_covers_the_experiment_directories() -> None:
    """``experiments/`` のディレクトリと登録済みの実験番号が一致する。

    実験を追加したのに ``EXPERIMENTS`` へ足し忘れると
    ``main.py --experiment NN`` から静かに消える (逆に、登録だけして
    ディレクトリが無いと実行時に落ちる)。ディレクトリ名の接頭辞
    (``01_what_is_rc`` -> ``01``) と突き合わせて機械的に固定する。
    """
    directories = {
        path.name.split("_", maxsplit=1)[0]
        for path in (Path(__file__).resolve().parents[1] / "experiments").iterdir()
        if path.is_dir() and not path.name.startswith((".", "_"))
    }
    assert set(catalog.BY_NUMBER) == directories


def test_unknown_experiment_is_rejected() -> None:
    """未登録の実験番号は argparse が弾く (静かに既定へ落ちない)。"""
    with pytest.raises(SystemExit) as excinfo:
        main.parse_args(["--experiment", "99"])
    assert excinfo.value.code != 0


def test_parse_args_defaults() -> None:
    """``--out`` 未指定は ``None`` を返す (実験ごとの既定値は ``main()`` 側で解決)。"""
    args = main.parse_args([])
    assert args.experiment == "01"
    assert args.out is None


def test_experiment_registry_has_unique_default_out_dirs() -> None:
    """登録済みの全実験の既定出力先が互いに異なる (受け入れ条件: データ損失防止)。

    ``--out`` を明示せずに ``main.py --experiment NN`` を実行しても、
    別の実験の成果物 (``meta.json`` など) を黙って上書きしないことを
    レジストリの構造そのものから保証する。03 以降を足したときも
    ``out_dir`` を使い回すとここが赤くなる。
    """
    out_dirs = [spec.scratch_dir for spec in catalog.BY_NUMBER.values()]
    assert len(out_dirs) == len(set(out_dirs)), f"out_dir が重複しています: {out_dirs}"


def test_default_out_dir_does_not_overwrite_other_experiments_meta_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--experiment 02`` を ``--out`` 未指定で実行しても 01 の meta.json は触らない。

    HIGH-1: かつては両実験が ``DEFAULT_OUT = Path("results")`` を共有しており、
    ``--out`` を省略すると 02 が 01 の成果物を黙って上書きしていた。
    実際の実行 (フルの ESN 計算) は縮小できないので、ここでは variant を
    「書き出し先を記録するだけ」のダミーに差し替え、カタログの既定が実際に
    使われることだけを確認する。
    """
    monkeypatch.chdir(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    sentinel = results_dir / "meta.json"
    sentinel.write_text('{"owner": "01"}', encoding="utf-8")

    recorded: list[Path] = []

    def _fake_run(request: catalog.RunRequest) -> None:
        recorded.append(request.out)

    monkeypatch.setitem(
        catalog.BY_NUMBER,
        "02",
        dataclasses.replace(
            catalog.BY_NUMBER["02"], variants={catalog.MAIN: _fake_run}
        ),
    )
    assert main.main(["--experiment", "02"]) == 0
    assert recorded == [catalog.BY_NUMBER["02"].scratch_dir]
    assert sentinel.read_text(encoding="utf-8") == '{"owner": "01"}'


def test_main_writes_the_four_artifacts(tiny_experiment: tuple[Path, Path]) -> None:
    """1コマンドで5成果物が出る (受け入れ条件5)。"""
    config_path, out_dir = tiny_experiment
    assert (
        main.main(
            ["--experiment", "01", "--config", str(config_path), "--out", str(out_dir)]
        )
        == 0
    )
    for name in ARTIFACTS:
        assert (out_dir / name).is_file(), f"{name} が生成されていません"
    for name in (name for name in ARTIFACTS if name.endswith(".png")):
        assert png_dpi(out_dir / name) >= RETINA_DPI


def test_meta_json_records_state_space_comparison(
    tiny_experiment: tuple[Path, Path],
) -> None:
    """``n_components_95`` の比較が meta.json に**数値として**残る (受け入れ条件4)。"""
    config_path, out_dir = tiny_experiment
    assert (
        main.main(
            ["--experiment", "01", "--config", str(config_path), "--out", str(out_dir)]
        )
        == 0
    )
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_rows"] == EXPECTED_ROWS
    assert meta["wall_time_s"] > 0.0
    reports = meta["state_space"]
    assert {report["task"] for report in reports} == {"mackey_glass", "delay_parity"}
    for report in reports:
        spaces = {space["space"]: space for space in report["spaces"]}
        assert set(spaces) == {RAW_INPUT, DELAY_EMBEDDED_INPUT, RESERVOIR_STATE}
        for space in spaces.values():
            assert isinstance(space["n_components_95"], int)
            assert space["n_components_95"] >= 1
            assert space["n_components_95"] <= space["n_features"]


def _without_wall_time(csv_text: str) -> list[list[str]]:
    """CSV から実測時間の列だけ落とす (実行ごとに変わるため)。"""
    rows = [line.split(",") for line in csv_text.strip().splitlines()]
    index = rows[0].index("wall_time_s")
    return [row[:index] + row[index + 1 :] for row in rows]

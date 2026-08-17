"""README の手書きの数値表が生成物と一致すること (F-1-005).

集計ロジックは ``experiment.summary.aggregate_nrmse`` に一本化されているが
(F-1-003)、記事として読みやすい ``README.md`` の表は依然として手で書く運用の
ままにしている。実験を回し直したときに ``results/comparison_summary.csv`` だけ
新しくなり README が古い値のまま取り残されると、この連載の「記事の数値と
リポジトリの実測値を機械的に一致させる」という規律が崩れる。

このテストは README の表の値を ``results/comparison_summary.csv`` (受け入れ
条件3: NRMSE 平均±標準偏差、符号正解率) と突き合わせ、乖離したら落ちる。
README を自動生成する必要はない —— 「どこから引いた数値か」を機械的に
固定するのがここでの目的。
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SUMMARY_CSV = ROOT / "results" / "comparison_summary.csv"

_NRMSE_DECIMALS = 4
_SIGN_DECIMALS = 3

_TASK_LABELS: dict[str, str] = {
    "Mackey-Glass (1ステップ先予測)": "mackey_glass",
    "遅延パリティ `y[t]=u[t-1]u[t-2]`": "delay_parity",
}
_METHOD_COLUMNS: tuple[str, ...] = ("linear", "delay_line", "esn")
"""README の表の列順 (課題列を除く: 線形 / 遅延線 / ESN)。"""

_CELL_RE = re.compile(r"\*{0,2}([\d.]+)\s*±\s*([\d.]+)\*{0,2}")
_SIGN_LINE_RE = re.compile(
    r"線形\s*([\d.]+)\s*/\s*遅延線\s*([\d.]+)\s*/\s*\*{0,2}ESN\s*([\d.]+)\*{0,2}"
)


def _load_summary() -> dict[tuple[str, str], dict[str, float]]:
    """``comparison_summary.csv`` を (課題, 手法) キーの dict にする。"""
    with SUMMARY_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "results/comparison_summary.csv が空です (実験を再実行してください)"
    return {
        (row["task"], row["method"]): {
            "nrmse_mean": float(row["nrmse_mean"]),
            "nrmse_std": float(row["nrmse_std"]),
            "sign_accuracy_mean": float(row["sign_accuracy_mean"]),
        }
        for row in rows
    }


def _parse_readme_table(text: str) -> dict[tuple[str, str], tuple[float, float]]:
    """README の NRMSE 表 (課題 x 手法) を ``{(task, method): (mean, std)}`` にする。"""
    parsed: dict[tuple[str, str], tuple[float, float]] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or cells[0] not in _TASK_LABELS:
            continue
        task = _TASK_LABELS[cells[0]]
        method_cells = cells[1:]
        assert len(method_cells) == len(_METHOD_COLUMNS), (
            f"README の表の列数が想定と違います: {line!r}"
        )
        for method, cell in zip(_METHOD_COLUMNS, method_cells, strict=True):
            match = _CELL_RE.fullmatch(cell)
            assert match, f"README の表のセルを解析できません: {cell!r}"
            parsed[(task, method)] = (float(match.group(1)), float(match.group(2)))
    return parsed


def _parse_readme_sign_accuracy(text: str) -> dict[str, float]:
    """README の「符号正解率」の行を ``{method: value}`` にする (delay_parity 限定)。"""
    for line in text.splitlines():
        if "符号正解率" not in line:
            continue
        match = _SIGN_LINE_RE.search(line)
        if match:
            linear, delay_line, esn = match.groups()
            return {
                "linear": float(linear),
                "delay_line": float(delay_line),
                "esn": float(esn),
            }
    raise AssertionError("README に符号正解率の行が見つかりません")


def test_readme_mentions_the_summary_csv_as_the_source() -> None:
    """README がこの表の出どころ (生成物) を明記していること。"""
    text = README.read_text(encoding="utf-8")
    assert "comparison_summary.csv" in text


def test_readme_nrmse_table_matches_comparison_summary_csv() -> None:
    """README の NRMSE 平均±標準偏差の表が ``comparison_summary.csv`` と一致する。"""
    summary = _load_summary()
    table = _parse_readme_table(README.read_text(encoding="utf-8"))
    assert table.keys() == {
        (task, method) for task in _TASK_LABELS.values() for method in _METHOD_COLUMNS
    }
    for key, (readme_mean, readme_std) in table.items():
        expected = summary[key]
        assert readme_mean == round(expected["nrmse_mean"], _NRMSE_DECIMALS), key
        assert readme_std == round(expected["nrmse_std"], _NRMSE_DECIMALS), key


def test_readme_sign_accuracy_matches_comparison_summary_csv() -> None:
    """README の符号正解率 (遅延パリティ) が ``comparison_summary.csv`` と一致する。"""
    summary = _load_summary()
    readme_values = _parse_readme_sign_accuracy(README.read_text(encoding="utf-8"))
    for method, readme_value in readme_values.items():
        expected = summary[("delay_parity", method)]["sign_accuracy_mean"]
        assert readme_value == round(expected, _SIGN_DECIMALS), method


# --- サイクル02 (実験02 の実測値) -------------------------------------------
#
# 01 と同じ規律を 02 にも適用する。02 の README の数値は
# ``results/02_esp_and_dynamics/`` の生成物 (閾値感度 CSV と meta.json) から
# 引いており、実験を回し直して README が取り残されたらここで落ちる。

ESP_DIR = ROOT / "results" / "02_esp_and_dynamics"
THRESHOLD_CSV = ESP_DIR / "esp_threshold_sensitivity.csv"
ESP_META = ESP_DIR / "meta.json"

_CRITICAL_ROW_LABEL = "ESP が壊れる最小の"
_OUT_OF_GRID = "格子外"
"""格子内に境界が無いこと (CSV では ``nan``) を README で表す語。"""

_META_VALUE_RE = {
    "wall_time_s": re.compile(r"wall_time_s\s*=\s*([\d.]+)"),
    "n_false_esp": re.compile(r"n_false_esp\s*=\s*(\d+)"),
    "n_local_but_not_global": re.compile(r"n_local_but_not_global\s*=\s*(\d+)"),
    "ratio": re.compile(r"washout_sensitivity\.headline\.ratio\s*=\s*([\d.]+)"),
}


def _reference_threshold_row() -> dict[str, str]:
    """閾値感度 CSV のうち、既定値 (D-16) に対応する行。"""
    with THRESHOLD_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "esp_threshold_sensitivity.csv が空です (make threshold-02)"
    for row in rows:
        if (
            float(row["abs_tol"]) == _REFERENCE_ABS_TOL
            and int(row["window"]) == _REFERENCE_WINDOW
        ):
            return row
    raise AssertionError("既定値 (abs_tol=1e-6, window=200) の行が CSV にありません")


def _readme_critical_rho_cells() -> list[str]:
    """README の「ESP が壊れる最小の rho」の行のセル (先頭ラベルを除く)。"""
    for line in README.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and _CRITICAL_ROW_LABEL in line:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            return cells[1:]
    raise AssertionError("README に臨界 rho の行が見つかりません")


def test_readme_mentions_the_experiment_02_artifacts_as_the_source() -> None:
    """02 の数値の出どころ (生成物と再生成コマンド) が README にある。"""
    text = README.read_text(encoding="utf-8")
    assert "esp_threshold_sensitivity.csv" in text
    assert "make figures-02" in text
    assert "make threshold-02" in text


def test_readme_critical_rho_row_matches_the_threshold_csv() -> None:
    """README の臨界 rho の行が閾値感度 CSV の既定値の行と一致する。"""
    row = _reference_threshold_row()
    critical = [
        (name, value)
        for name, value in row.items()
        if name.startswith("critical_rho_sigma_")
    ]
    cells = _readme_critical_rho_cells()
    assert len(cells) == len(critical), (cells, critical)
    for cell, (_, value) in zip(cells, critical, strict=True):
        expected = float(value)
        if math.isnan(expected):
            assert _OUT_OF_GRID in cell, cell
            continue
        assert float(cell) == expected, (cell, expected)


def test_readme_experiment_02_numbers_match_meta_json() -> None:
    """README に書いた 02 の実測値が ``meta.json`` と一致する。"""
    meta = json.loads(ESP_META.read_text(encoding="utf-8"))
    text = README.read_text(encoding="utf-8")
    found: dict[str, float] = {}
    for key, pattern in _META_VALUE_RE.items():
        match = pattern.search(text)
        assert match, f"README に {key} の記述が見つかりません"
        found[key] = float(match.group(1))

    agreement = meta["verdict_lyapunov_agreement"]
    headline = meta["washout_sensitivity"]["headline"]
    assert found["wall_time_s"] == round(meta["wall_time_s"], 2)
    assert found["n_false_esp"] == agreement["n_false_esp"]
    assert found["n_local_but_not_global"] == agreement["n_local_but_not_global"]
    assert found["ratio"] == round(headline["ratio"], 5)

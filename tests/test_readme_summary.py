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

#: README に書いた実測時間と ``meta.json`` の許容比 (D-84)。
#: **厳密一致で縛らない** —— ``wall_time_s`` は実行環境で変わるので、
#: 誰がどのマシンで再生成しても落ちる検査になってしまう
#: (実際に 87.69 → 88.0 → 95.18 と3回落ちた)。
#: 一方で「この実験が10倍遅くなった」は拾いたいので、桁が変わる手前で止める。
WALL_TIME_TOLERANCE = 2.0


def assert_wall_time_is_the_same_order(documented: float, measured: float) -> None:
    """README の実測時間が ``meta.json`` と同じ桁にあること (D-84)。"""
    assert measured > 0, "meta.json の wall_time_s が正の値ではありません"
    ratio = documented / measured
    assert 1 / WALL_TIME_TOLERANCE <= ratio <= WALL_TIME_TOLERANCE, (
        f"README の実測時間 {documented} 秒 と meta.json の {measured:.2f} 秒 が "
        f"{ratio:.2f} 倍ずれています。\n"
        "**実行環境の差では説明できない開き**なので、実験が重くなっていないか"
        "確認してください (単なる再測定なら README の数値を更新してください)。"
    )


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
    """閾値感度 CSV のうち、本番実行が実際に使った基準行 (``is_reference`` 列)。

    基準は D-16 の既定値ではなく ``config.esp.abs_tol`` / ``config.esp.window``
    から取る (``rc_basics_lab.experiment.threshold.run_threshold_sweep``)。
    ここで独自にモジュール定数へ固定してしまうと、``config.esp`` を格子内の
    別点に変えたときに CSV の基準行だけが動き、README 照合は「ずれ 0 でない
    行」を基準として検証し続ける (F-2-002)。CSV 自身が持つ ``is_reference``
    列から引くことでこの経路を塞ぐ。
    """
    with THRESHOLD_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "esp_threshold_sensitivity.csv が空です (make threshold-02)"
    for row in rows:
        if row["is_reference"] == "True":
            return row
    raise AssertionError("is_reference=True の行が CSV にありません")


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
    assert_wall_time_is_the_same_order(found["wall_time_s"], meta["wall_time_s"])
    assert found["n_false_esp"] == agreement["n_false_esp"]
    assert found["n_local_but_not_global"] == agreement["n_local_but_not_global"]
    assert found["ratio"] == round(headline["ratio"], 5)


# --- 実験03 (README の 03 節) vs results/03_capacity/ -------------------------
#
# 02 の ``test_readme_experiment_02_numbers_match_meta_json`` と同じ役割を 03 に
# 対して果たす。03 の節には成果物の行数・実行時間・3-C の成績・容量の3値という
# 「回し直すと動く数値」が並ぶので、README が古いまま取り残されたら落とす。

CAPACITY_RESULTS = ROOT / "results" / "03_capacity"
CAPACITY_META = CAPACITY_RESULTS / "meta.json"
CAPACITY_CSV = CAPACITY_RESULTS / "capacity.csv"
NARMA10_CSV = CAPACITY_RESULTS / "narma10.csv"

_ARTIFACT_ROWS_RE = re.compile(
    r"^\|\s*`(?P<name>[\w.]+\.csv)`\s*\|\s*(?P<rows>[\d,]+)行"
)
_CAPACITY_META_KEYS: dict[str, str] = {
    "capacity.csv": "n_rows",
    "capacity_profile.csv": "n_profile_rows",
    "narma10.csv": "n_narma10_rows",
}
_NMSE_ROW_RE = re.compile(
    r"^\|\s*NMSE\s*\|\s*\*{0,2}([\d.]+)\*{0,2}\s*\|\s*\*{0,2}([\d.]+)\*{0,2}"
    r"\s*\|\s*\*{0,2}([\d.]+)\*{0,2}\s*\|"
)
_MC_RATIO_RE = re.compile(r"到達する最大は (?P<ratio>[\d.]+)%")
_CAPACITY_VALUE_RE = re.compile(
    r"`(?P<name>mc_total|ipc_total|ipc_nonlinear) = (?P<value>[\d.]+)`"
)
_DELAY_LINE_LAGS_RE = re.compile(r"選ばれた k=(?P<lags>\d+)")
_STICKY_RE = re.compile(
    r"(?P<count>\d+)回中(?P<hits>\d+)回で格子の\*\*上端 k=(?P<lags>\d+)"
)


def _experiment_03_section() -> str:
    """README の 03 節だけを切り出す (02 節と数値の書式が同じなので混ざる)。"""
    text = README.read_text(encoding="utf-8")
    start = text.index("## 実験03")
    end = text.index("\n## ", start)
    return text[start:end]


def _capacity_meta_json() -> dict[str, object]:
    loaded: dict[str, object] = json.loads(CAPACITY_META.read_text(encoding="utf-8"))
    return loaded


def _narma10_rows() -> list[dict[str, str]]:
    with NARMA10_CSV.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _capacity_3c_row() -> dict[str, str]:
    with CAPACITY_CSV.open(encoding="utf-8", newline="") as handle:
        return next(
            row for row in csv.DictReader(handle) if row["experiment"] == "3C_narma10"
        )


def test_readme_mentions_the_experiment_03_artifacts_as_the_source() -> None:
    """03 の数値の出どころ (生成物と再生成コマンド) が README にある。"""
    text = README.read_text(encoding="utf-8")
    assert "make figures-03" in text
    assert "make saturation-03" in text
    assert "narma10.csv" in text
    assert "threshold_comparison" in text


def test_readme_experiment_03_artifact_rows_match_meta_json() -> None:
    """README の成果物表の行数が ``meta.json`` と一致する。"""
    meta = _capacity_meta_json()
    found: dict[str, int] = {}
    for line in _experiment_03_section().splitlines():
        match = _ARTIFACT_ROWS_RE.match(line)
        if match and match["name"] in _CAPACITY_META_KEYS:
            found[match["name"]] = int(match["rows"].replace(",", ""))
    assert set(found) == set(_CAPACITY_META_KEYS), found
    for name, key in _CAPACITY_META_KEYS.items():
        assert found[name] == meta[key], name


def test_readme_experiment_03_wall_time_matches_meta_json() -> None:
    """README に書いた `make figures-03` の実測時間が ``meta.json`` と一致する。"""
    meta = _capacity_meta_json()
    match = re.search(r"wall_time_s = ([\d.]+) 秒", _experiment_03_section())
    assert match, "README に 03 の wall_time_s の記述が見つかりません"
    wall_time = meta["wall_time_s"]
    assert isinstance(wall_time, float)
    assert_wall_time_is_the_same_order(float(match.group(1)), wall_time)


def test_readme_narma10_table_matches_the_committed_rows() -> None:
    """README の NARMA10 の NMSE 表が ``narma10.csv`` と一致する。"""
    rows = _narma10_rows()
    for line in _experiment_03_section().splitlines():
        match = _NMSE_ROW_RE.match(line)
        if not match:
            continue
        for method, cell in zip(
            ("linear", "delay_line", "esn"), match.groups(), strict=True
        ):
            subset = [row for row in rows if row["method"] == method]
            assert subset, method
            expected = sum(float(row["nmse"]) for row in subset) / len(subset)
            assert float(cell) == round(expected, len(cell.partition(".")[2])), method
        return
    raise AssertionError("README に NARMA10 の NMSE 表が見つかりません")


def test_readme_experiment_03_capacity_numbers_match_the_csv() -> None:
    """README の 03 節の容量の値・到達率・遅延線の張り付きが一次資料と一致する。

    「容量が成績を説明する」という README の主張の数値そのものなので、
    ここが手書きのまま古くなると記事の根拠が崩れる。
    """
    text = _experiment_03_section()
    row = _capacity_3c_row()
    documented = {
        match["name"]: match["value"] for match in _CAPACITY_VALUE_RE.finditer(text)
    }
    assert set(documented) == {"mc_total", "ipc_total", "ipc_nonlinear"}, documented
    for name, cell in documented.items():
        decimals = len(cell.partition(".")[2])
        assert float(cell) == round(float(row[name]), decimals), name

    with CAPACITY_CSV.open(encoding="utf-8", newline="") as handle:
        ratios = [
            float(record["mc_ratio"])
            for record in csv.DictReader(handle)
            if record["experiment"] == "3A_mc_sweep"
        ]
    ratio_match = _MC_RATIO_RE.search(text)
    assert ratio_match, "README に mc_ratio の到達率が見つかりません"
    assert float(ratio_match["ratio"]) == round(100.0 * max(ratios), 1)

    rows = [row for row in _narma10_rows() if row["method"] == "delay_line"]
    selected = [int(row["n_lags"]) for row in rows]
    top = max(set(selected), key=selected.count)
    lags_match = _DELAY_LINE_LAGS_RE.search(text)
    sticky_match = _STICKY_RE.search(text)
    assert lags_match and sticky_match, "README に遅延線の k の記述が見つかりません"
    assert int(lags_match["lags"]) == top
    assert int(sticky_match["lags"]) == top
    assert int(sticky_match["count"]) == len(selected)
    assert int(sticky_match["hits"]) == selected.count(top)


# --- 実験04 (04b-2 T5) --------------------------------------------------------

CHAOS_RESULTS = ROOT / "results" / "04_chaotic_freerun"
CHAOS_META = CHAOS_RESULTS / "meta.json"
FREERUN_CSV = CHAOS_RESULTS / "freerun.csv"

_VALID_TIME_RE = re.compile(
    r"ESN ([\d.]+) / 遅延線 ([\d.]+) / 線形 ([\d.]+) \[1/lambda_max\]"
)
_REGIME_TREND_RE = re.compile(r"80 条件中 (\d+) -> (\d+)")


def _chaos_meta_json() -> dict[str, object]:
    loaded: dict[str, object] = json.loads(CHAOS_META.read_text(encoding="utf-8"))
    return loaded


def _experiment_04_section() -> str:
    """README の 04 節だけを切り出す。"""
    text = README.read_text(encoding="utf-8")
    start = text.index("## 実験04")
    end = text.index("\n## ", start)
    return text[start:end]


def test_readme_mentions_the_experiment_04_artifacts_as_the_source() -> None:
    """04 の数値の出どころ (生成物と再生成コマンド) が README にある。"""
    text = _experiment_04_section()
    assert "make figures-04" in text
    assert "capacity_note" in text
    assert "docs/design.md" in text


def test_readme_experiment_04_wall_time_matches_meta_json() -> None:
    """README に書いた `make figures-04` の実測時間が ``meta.json`` と一致する。"""
    meta = _chaos_meta_json()
    match = re.search(r"wall_time_s = ([\d.]+) 秒", _experiment_04_section())
    assert match, "README に 04 の wall_time_s の記述が見つかりません"
    wall_time = meta["wall_time_s"]
    assert isinstance(wall_time, float)
    assert float(match.group(1)) == round(wall_time, 1)


def test_readme_experiment_04_valid_time_matches_the_freerun_csv() -> None:
    """README の有効予測時間 (3手法の中央値) が ``freerun.csv`` と一致する。

    「自走では対照が成立しない」という 04 の主張そのものの数値なので、手書きの
    まま古くなると記事の根拠が崩れる。
    """
    with FREERUN_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    match = _VALID_TIME_RE.search(_experiment_04_section())
    assert match, "README に 04 の有効予測時間の記述が見つかりません"
    for cell, method in zip(
        match.groups(), ("esn", "delay_line", "linear"), strict=True
    ):
        values = sorted(
            float(row["valid_time_lyapunov"])
            for row in rows
            if row["task"] == "lorenz" and row["method"] == method
        )
        assert values, method
        middle = len(values) // 2
        median = (
            values[middle]
            if len(values) % 2
            else 0.5 * (values[middle - 1] + values[middle])
        )
        decimals = len(cell.partition(".")[2])
        assert float(cell) == round(median, decimals), method


def test_readme_experiment_04_noise_trend_matches_meta_json() -> None:
    """README の「発散が減る」の数値が ``meta.json`` の3態の内訳と一致する。"""
    counts = _chaos_meta_json()["regime_counts_by_noise"]
    assert isinstance(counts, dict)
    keys = sorted(counts, key=float)
    match = _REGIME_TREND_RE.search(_experiment_04_section())
    assert match, "README に 04 の3態の傾向が見つかりません"
    for cell, key in zip(match.groups(), (keys[0], keys[-1]), strict=True):
        entry = counts[key]
        assert isinstance(entry, dict)
        assert int(cell) == int(entry["diverged"]), key
    total = counts[keys[0]]
    assert isinstance(total, dict)
    assert sum(int(value) for value in total.values()) == 80


# --- 実験05 (05b T5) ----------------------------------------------------------

ANOMALY_RESULTS = ROOT / "results" / "05_anomaly_detection"
ANOMALY_META = ANOMALY_RESULTS / "meta.json"
ANOMALY_CSV_PATH = ANOMALY_RESULTS / "anomaly.csv"
ANOMALY_PROTOCOL_CSV_PATH = ANOMALY_RESULTS / "anomaly_protocol.csv"
ANOMALY_SIZE_CSV_PATH = ANOMALY_RESULTS / "anomaly_size.csv"

_ANOMALY_ROW_RE = re.compile(
    r"^\|\s*\*{0,2}(?P<label>[^|*]+?)\*{0,2}\s*\|\s*"
    r"\*{0,2}(?P<mean>[\d.]+)\s*±\s*(?P<sd>[\d.]+)\*{0,2}\s*\|\s*"
    r"\*{0,2}(?P<ratio>[\d.]+)x\*{0,2}\s*\|\s*"
    r"\*{0,2}(?:あり|なし)\s*\((?P<better>\d+)/(?P<pairs>\d+),"
    r"\s*p=(?P<p_value>[\d.e+-]+)\)\*{0,2}\s*\|\s*$"
)
"""README の 05 の数値表の1行 (系統 / AUPRC / 対照比 / 印と根拠)。"""

_ANOMALY_METHOD_BY_LABEL: dict[str, str] = {
    "ESN 残差": "esn_residual",
    "遅延線 残差": "delay_line_residual",
    "直前値 残差": "persistence_residual",
    "移動統計": "moving_statistics",
    "一様乱数 (対照)": "random_control",
    "入力ノルム (対照)": "input_norm_control",
}

_PROTOCOL_RE = re.compile(
    r"\*\*(?P<conditions>\d+) 点中 (?P<changed>\d+) 点\*\*で変動、"
    r"逆転した系統対は延べ \*\*(?P<pairs>\d+) 組\*\*"
)
_PROTOCOL_MARKED_RE = re.compile(r"組は 0*(?P<marked>\d+) 組")
_SIZE_RE = re.compile(
    r"基準 N=(?P<reference>\d+) の AUPRC (?P<reference_auprc>[\d.]+) に\s*"
    r"対し N=(?P<degraded>\d+) で (?P<degraded_auprc>[\d.]+) "
    r"\((?P<ratio>[\d.]+) 倍\)"
)
_F1_GAP_RE = re.compile(
    r"は 90 行の平均 \*\*(?P<mean>[\d.]+)\*\* \(最大 (?P<max>[\d.]+)"
)
_N_TRAIN_RE = re.compile(r"全 (?P<rows>\d+) 行が `n_train=(?P<n_train>\d+)`")


def _experiment_05_section() -> str:
    """README の 05 節だけを切り出す。"""
    text = README.read_text(encoding="utf-8")
    start = text.index("## 実験05")
    end = text.index("\n## ", start)
    return text[start:end]


def _anomaly_rows() -> list[dict[str, str]]:
    with ANOMALY_CSV_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "results/05_anomaly_detection/anomaly.csv が空です"
    return rows


def _mean_sd(values: list[float]) -> tuple[float, float]:
    """平均と標準偏差 (``ddof=1``。実験層の ``MethodAggregate`` と同じ規則)。"""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def test_readme_mentions_the_experiment_05_artifacts_as_the_source() -> None:
    """05 の数値の出どころ (生成物と再生成コマンド) が README にある。"""
    text = _experiment_05_section()
    assert "make figures-05" in text
    assert "make data-05" in text
    assert "fig_protocol_sensitivity" in text
    assert "docs/design.md" in text


def test_readme_experiment_05_wall_time_matches_meta_json() -> None:
    """README に書いた `make figures-05` の実測時間が ``meta.json`` と一致する。"""
    meta = json.loads(ANOMALY_META.read_text(encoding="utf-8"))
    match = re.search(r"wall_time_s = ([\d.]+) 秒", _experiment_05_section())
    assert match, "README に 05 の wall_time_s の記述が見つかりません"
    wall_time = meta["wall_time_s"]
    assert isinstance(wall_time, float)
    assert float(match.group(1)) == round(wall_time, 1)


def test_readme_experiment_05_auprc_table_matches_the_anomaly_csv() -> None:
    """README の AUPRC 表が ``anomaly.csv`` から計算した値と一致する。

    平均・標準偏差 (``ddof=1``)・一様乱数対照比・符号検定の根拠 (何対中いくつが
    対照を上回ったか) の**4つとも**照合する。表の1列だけを合わせて他が古い、
    という壊れ方を残さない。
    """
    rows = _anomaly_rows()
    parsed = [
        match
        for line in _experiment_05_section().splitlines()
        if (match := _ANOMALY_ROW_RE.match(line)) is not None
    ]
    assert len(parsed) == len(_ANOMALY_METHOD_BY_LABEL), (
        f"05 の表が {len(parsed)} 行しかありません"
    )
    control = [
        float(row["auprc_random"]) for row in rows if row["method"] == "esn_residual"
    ]
    control_mean = sum(control) / len(control)
    for match in parsed:
        method = _ANOMALY_METHOD_BY_LABEL[match["label"].strip()]
        selected = [row for row in rows if row["method"] == method]
        assert selected, method
        mean, sd = _mean_sd([float(row["auprc"]) for row in selected])
        assert float(match["mean"]) == round(mean, 4), method
        assert float(match["sd"]) == round(sd, 4), method
        assert float(match["ratio"]) == round(mean / control_mean, 2), method
        better = sum(
            1 for row in selected if float(row["auprc"]) > float(row["auprc_random"])
        )
        assert int(match["better"]) == better, method
        assert int(match["pairs"]) == len(selected), method


def test_readme_experiment_05_marks_match_the_protocol_csv() -> None:
    """README の p 値と印が ``anomaly_protocol.csv`` の基準条件の行と一致する。"""
    with ANOMALY_PROTOCOL_CSV_PATH.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["is_headline"] == "True"]
    assert rows, "5-C に基準条件の行がありません"
    by_method = {row["method"]: row for row in rows}
    for line in _experiment_05_section().splitlines():
        match = _ANOMALY_ROW_RE.match(line)
        if match is None:
            continue
        method = _ANOMALY_METHOD_BY_LABEL[match["label"].strip()]
        row = by_method[method]
        documented_p = float(match["p_value"])
        actual_p = float(row["control_sign_p"])
        assert abs(documented_p - actual_p) <= 0.02 * actual_p, method
        assert ("あり" in line) == (row["distinguishable"] == "True"), method


def test_readme_experiment_05_protocol_counts_match_the_csv() -> None:
    """README の「27点中23点で変動 / 逆転62組 / 両方に印0組」が CSV と一致する。

    **この3つの数の並びが 5-C の結論そのもの**である (D-78)。延べの逆転数だけを
    書き換えて印の数を古いままにすると、記事の主張が反転する。
    """
    with ANOMALY_PROTOCOL_CSV_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    conditions: dict[tuple[str, str, str], dict[str, str]] = {}
    changed: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["normalize"], row["input_window"], row["score_smoothing"])
        conditions.setdefault(key, row)
        if row["rank_changed"] == "True":
            changed.add(key)
    total_pairs = sum(int(row["n_discordant_pairs"]) for row in conditions.values())
    marked_pairs = sum(
        int(row["n_discordant_pairs_distinguishable"]) for row in conditions.values()
    )
    section = _experiment_05_section()
    match = _PROTOCOL_RE.search(section)
    assert match, "README に 5-C の順位変動の記述が見つかりません"
    assert int(match["conditions"]) == len(conditions)
    assert int(match["changed"]) == len(changed)
    assert int(match["pairs"]) == total_pairs
    marked = _PROTOCOL_MARKED_RE.search(section)
    assert marked, "README に「両方に印がある組」の数が見つかりません"
    assert int(marked["marked"]) == marked_pairs
    esn_ranks = {row["rank"] for row in rows if row["method"] == "esn_residual"}
    assert esn_ranks == {"1"}, "ESN が1位でない格子点があります (README を直す)"


def test_readme_experiment_05_size_numbers_match_the_size_csv() -> None:
    """README の 5-D の数値が ``anomaly_size.csv`` と一致する。"""
    with ANOMALY_SIZE_CSV_PATH.open(encoding="utf-8", newline="") as handle:
        every = list(csv.DictReader(handle))
    by_units = {
        int(row["n_units"]): row for row in every if row["method"] == "esn_residual"
    }
    section = _experiment_05_section()
    match = _SIZE_RE.search(section)
    assert match, "README に 5-D の劣化点の記述が見つかりません"
    reference = by_units[int(match["reference"])]
    degraded = by_units[int(match["degraded"])]
    assert float(match["reference_auprc"]) == round(float(reference["auprc_mean"]), 4)
    assert float(match["degraded_auprc"]) == round(float(degraded["auprc_mean"]), 4)
    assert float(match["ratio"]) == round(float(degraded["auprc_ratio"]), 3)
    assert degraded["below_reference_fraction"] == "True"
    train = _N_TRAIN_RE.search(section)
    assert train, "README に 5-D の学習量の記述が見つかりません"
    assert int(train["rows"]) == len(every)
    assert {row["n_train"] for row in every} == {train["n_train"]}


def test_readme_experiment_05_f1_gap_matches_the_anomaly_csv() -> None:
    """README の `f1_test_optimal - f1_calibrated` が ``anomaly.csv`` と一致する。"""
    rows = _anomaly_rows()
    gaps = [float(row["f1_test_optimal"]) - float(row["f1_calibrated"]) for row in rows]
    match = _F1_GAP_RE.search(_experiment_05_section())
    assert match, "README に 05 の f1 の差の記述が見つかりません"
    assert float(match["mean"]) == round(sum(gaps) / len(gaps), 4)
    assert float(match["max"]) == round(max(gaps), 4)
    assert min(gaps) >= 0.0, "テスト側最適化のほうが低い行があります"


def test_readme_experiment_05_random_control_sits_on_the_anomaly_rate() -> None:
    """README の「乱数の AUPRC が異常率に張り付く」2つの数が CSV と一致する。

    この一致が「主指標が point-adjust を通っていないこと」の実測的な証拠
    (D-54 / D-55) なので、片方だけ古くなると証拠として機能しなくなる。
    """
    rows = _anomaly_rows()
    match = re.search(
        r"\*\*(?P<control>[\d.]+)\*\* vs 異常率\s*\n?\s*\*\*(?P<rate>[\d.]+)\*\*",
        _experiment_05_section(),
    )
    assert match, "README に乱数対照と異常率の対比が見つかりません"
    control = [float(row["auprc"]) for row in rows if row["method"] == "random_control"]
    rate = [float(row["anomaly_rate"]) for row in rows]
    assert float(match["control"]) == round(sum(control) / len(control), 4)
    assert float(match["rate"]) == round(sum(rate) / len(rate), 4)


def test_readme_claims_the_same_experiments_it_documents() -> None:
    """README 冒頭の「実装済みの範囲」が、実際の節見出しと一致すること (D-83)。

    冒頭が「サイクル01〜03 の範囲を実装している」のまま、下に 04 と 05 の節が
    並んでいる状態が実際に起きていた。**最初に読む3行が最も古い**という形の
    ドリフトで、README を開いた人が最初に受け取る情報が間違っていた。

    数値の照合 (この上の各テスト) は厚くしてあったのに、
    「何を実装したか」という一番大きな主張だけが誰にも検査されていなかった。
    """
    text = README.read_text(encoding="utf-8")
    documented = {
        match.group(1) for match in re.finditer(r"^## 実験(\d{2})", text, re.MULTILINE)
    }
    # 01 は「3コマンドで再現する」の節が兼ねているので、節見出しが無くても実装済み。
    documented.add("01")
    claimed = re.search(r"記事(\d{2})〜(\d{2}) の全範囲", text)
    assert claimed, (
        "README 冒頭に「記事NN〜MM の全範囲を実装している」の記述がありません。"
        "範囲を書き換えたなら、この検査の期待形式も合わせてください。"
    )
    span = {f"{n:02d}" for n in range(int(claimed.group(1)), int(claimed.group(2)) + 1)}
    assert span == documented, (
        f"README 冒頭が主張する範囲 {sorted(span)} と、"
        f"実際に節がある実験 {sorted(documented)} が食い違っています。"
    )

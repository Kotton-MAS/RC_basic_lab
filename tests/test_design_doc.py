"""``docs/design.md`` §9 の散文とコードの乖離を機械で殺す.

サイクル02 の設計メモは要件_02 の設計判断3 (「ESP 判定の閾値と窓の既定値と
その根拠を書く」) への回答であり、**書かれた既定値が実装と食い違ったら
記事の根拠そのものが嘘になる**。散文は放っておくと必ずドリフトするので、
2つの対応を機械で固定する:

1. §9 の「既定値」表 —— 3列目に**コード上の出どころ**をドット区切りで書き、
   2列目の値と実際に一致することを検査する (Python リテラルとして評価する)。
   出どころを書けない値は表に載せられないので、根拠の無い数値が表に紛れ込む
   経路も同時に塞がる
2. §9.2 の閾値感度表 —— ``esp_threshold_sensitivity.csv`` と**行数も値も**
   一致することを検査する (仕様 T5 の受け入れ基準: 「CSV が9行以上を持ち
   design.md の表と行数一致」)

``tests/test_readme_summary.py`` (README の表 vs 生成物) と同じ役割を、
design.md と**コード**の間で果たす。
"""

from __future__ import annotations

import ast
import csv
import dataclasses
import importlib
import json
import math
import re
import statistics
from pathlib import Path

import pytest

from rc_basics_lab.experiment.threshold import sigma_column

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "docs" / "design.md"
THRESHOLD_CSV = (
    ROOT / "results" / "02_esp_and_dynamics" / "esp_threshold_sensitivity.csv"
)
UNPADDED_WASHOUT_CSV = (
    ROOT / "results" / "02_esp_and_dynamics" / "washout_sensitivity_unpadded.csv"
)

_PACKAGE = "rc_basics_lab"
_DEFAULTS_ROW = re.compile(
    r"^\|\s*(?P<label>[^|]+?)\s*\|\s*`(?P<value>[^`]+)`\s*\|\s*"
    r"`(?P<path>[A-Za-z_][A-Za-z0-9_.]*)`\s*\|\s*$"
)
_SENSITIVITY_HEADER = re.compile(r"^\|\s*`abs_tol`\s*\|\s*`window`\s*\|")
_NO_BOUNDARY = "\u2014"
"""感度表で「格子内に境界が無い」を表す記号 (CSV では ``nan``)。em dash。"""

_SIGMA_HEAD = "\u03c3="
"""感度表のヘッダで入力強度の列を示す接頭辞 (ギリシャ小文字 sigma + ``=``)。

ruff の RUF001/RUF002 がソース中のギリシャ文字を弾くため、エスケープで書く
(``docs/plans/rc-basics-02.md`` T2 の実装メモ17 と同じ制約)。
"""

_RHO = "\u03c1"
"""ギリシャ文字 rho (スペクトル半径)。``_SIGMA_HEAD`` と同じ理由 (ruff の
RUF001/RUF002 対策) でエスケープで書く (F-3b2-1-003)。"""

MIN_SENSITIVITY_ROWS = 9
"""仕様 T5 の受け入れ基準。``abs_tol`` 3点 x ``window`` 3点。"""


def _text() -> str:
    return DESIGN.read_text(encoding="utf-8")


def _resolve(path: str) -> object:
    """``diagnostics.esp.DEFAULT_ESP.abs_tol`` のような表記を実際の値にする。

    frozen + slots の dataclass は既定値がクラス属性として残らないため、
    dataclass の**型**に対する属性参照はフィールドの既定値として解決する
    (``config.DriveConfig.n_pairs`` が 10 になる)。
    """
    parts = path.split(".")
    module = None
    for stop in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(f"{_PACKAGE}." + ".".join(parts[:stop]))
        except ModuleNotFoundError:
            continue
        parts = parts[stop:]
        break
    assert module is not None, f"モジュールが見つかりません: {path}"
    obj: object = module
    for attr in parts:
        obj = _attribute(obj, attr, path)
    return obj


def _attribute(obj: object, attr: str, path: str) -> object:
    if isinstance(obj, type) and dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            if field.name != attr:
                continue
            if field.default is not dataclasses.MISSING:
                return field.default
            if field.default_factory is not dataclasses.MISSING:
                return field.default_factory()
            raise AssertionError(f"既定値の無いフィールドです: {path}")
    value: object = getattr(obj, attr)
    return value


def _defaults_rows() -> list[tuple[str, str, str]]:
    """§9 の既定値表 (ラベル, 値のリテラル, コード上の出どころ)。"""
    rows: list[tuple[str, str, str]] = []
    for line in _text().splitlines():
        match = _DEFAULTS_ROW.match(line)
        if match:
            rows.append(
                (match["label"], match["value"], match["path"]),
            )
    return rows


def test_design_doc_documents_the_defaults_with_their_source() -> None:
    """§9 に既定値表があり、行数が想定を下回らない (表ごと消えたら落ちる)。"""
    rows = _defaults_rows()
    assert len(rows) >= 15, f"§9 の既定値表が縮んでいます: {len(rows)} 行"
    paths = [path for _, _, path in rows]
    assert len(paths) == len(set(paths)), "同じ出どころが2回書かれています"


@pytest.mark.parametrize("row", _defaults_rows(), ids=lambda row: str(row[2]))
def test_documented_default_matches_the_code(row: tuple[str, str, str]) -> None:
    """§9 の表に書いた既定値が、コード上の値と一致する。"""
    label, literal, path = row
    documented = ast.literal_eval(literal)
    actual = _resolve(path)
    assert documented == actual, (
        f"{label}: design.md は {documented!r} / コードは {actual!r}"
    )


# --- §9.2 閾値感度表 vs 生成物 ----------------------------------------------


def _sensitivity_table() -> tuple[list[str], list[list[str]]]:
    """§9.2 の感度表を (ヘッダ, データ行) に分ける。"""
    lines = _text().splitlines()
    for index, line in enumerate(lines):
        if not _SENSITIVITY_HEADER.match(line):
            continue
        header = [cell.strip() for cell in line.strip("|").split("|")]
        rows: list[list[str]] = []
        for body in lines[index + 2 :]:
            if not body.startswith("|"):
                break
            rows.append([cell.strip() for cell in body.strip("|").split("|")])
        return header, rows
    raise AssertionError("docs/design.md §9.2 の閾値感度表が見つかりません")


def _csv_rows() -> list[dict[str, str]]:
    with THRESHOLD_CSV.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_threshold_sensitivity_csv_has_at_least_nine_rows() -> None:
    """一次資料が仕様どおり9行以上ある (T5 の受け入れ基準)。"""
    assert THRESHOLD_CSV.is_file(), "make threshold-02 で再生成してください"
    assert len(_csv_rows()) >= MIN_SENSITIVITY_ROWS


def test_design_table_row_count_matches_the_threshold_csv() -> None:
    """§9.2 の表の行数が CSV の行数と一致する。"""
    _, table = _sensitivity_table()
    assert len(table) == len(_csv_rows())


def test_design_table_values_match_the_threshold_csv() -> None:
    """§9.2 の表の全セルが CSV と一致する (手書きの表がドリフトしたら落ちる)。

    列の対応は「ヘッダの ``sigma=<値>`` -> CSV の ``critical_rho_sigma_<値>``」
    (ヘッダの実際の表記はギリシャ文字。``_SIGMA_HEAD`` を参照)。
    ``_NO_BOUNDARY`` は「格子内に境界が無い」で、CSV では ``nan`` に対応する。
    """
    header, table = _sensitivity_table()
    records = _csv_rows()
    sigma_columns = {
        index: cell.removeprefix(_SIGMA_HEAD)
        for index, cell in enumerate(header)
        if cell.startswith(_SIGMA_HEAD)
    }
    assert sigma_columns, header

    for cells, record in zip(table, records, strict=True):
        assert float(cells[0]) == float(record["abs_tol"]), cells
        assert int(cells[1]) == int(record["window"]), cells
        assert int(cells[-1]) == int(record["n_converged"]), cells
        for index, sigma in sigma_columns.items():
            expected = float(record[sigma_column(float(sigma))])
            if cells[index] == _NO_BOUNDARY:
                assert math.isnan(expected), (sigma, cells)
                continue
            assert float(cells[index]) == expected, (sigma, cells)


def test_design_doc_points_at_the_regeneration_command() -> None:
    """表の出どころ (再生成コマンドと CSV 名) が §9 に書いてある。"""
    text = _text()
    assert "make threshold-02" in text
    assert "esp_threshold_sensitivity.csv" in text


# --- §9.6 補償なし (pad_series=False) との対比 -------------------------------


def _unpadded_washout_table() -> tuple[list[str], list[list[str]]]:
    """§9.6 の「補償なし」表を (ヘッダ, データ行) に分ける。

    §9.6 には ``| washout | ... |`` 形式の表が2つ (補償あり / 補償なし) ある
    ので、「補償なし」の見出しより後にある方を探す。
    """
    lines = _text().splitlines()
    start = next(index for index, line in enumerate(lines) if "補償なし" in line)
    header_pattern = re.compile(r"^\|\s*washout\s*\|")
    for index in range(start, len(lines)):
        if not header_pattern.match(lines[index]):
            continue
        header = [cell.strip() for cell in lines[index].strip("|").split("|")]
        rows: list[list[str]] = []
        for body in lines[index + 2 :]:
            if not body.startswith("|"):
                break
            rows.append([cell.strip() for cell in body.strip("|").split("|")])
        return header, rows
    raise AssertionError("docs/design.md §9.6 の補償なし表が見つかりません")


def _unpadded_csv_rows() -> list[dict[str, str]]:
    with UNPADDED_WASHOUT_CSV.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mg_esn_unpadded_stats_by_washout() -> dict[int, tuple[int, float]]:
    """``(n_train, レプリケート平均 nrmse)`` を washout ごとに (MG x ESN のみ)。"""
    grouped: dict[int, list[tuple[int, float]]] = {}
    for record in _unpadded_csv_rows():
        if record["task"] != "mackey_glass" or record["method"] != "esn":
            continue
        washout = int(record["washout"])
        grouped.setdefault(washout, []).append(
            (int(record["n_train"]), float(record["nrmse"]))
        )
    return {
        washout: (values[0][0], statistics.fmean(nrmse for _, nrmse in values))
        for washout, values in grouped.items()
    }


def test_unpadded_washout_csv_exists() -> None:
    """補償なし (``pad_series=False``) の一次資料が存在する (D-19 の対比の裏付け)。

    §9.6 の「補償なしでは完全に単調増加する」という記述 —— D-19 の存在意義
    そのもの —— を裏付ける数値がコミット済みの CSV から検証できることを固定する。
    """
    assert UNPADDED_WASHOUT_CSV.is_file(), (
        "washout.pad_series: false で再生成してください (docs/design.md §9.6)"
    )


def test_unpadded_design_table_matches_the_csv() -> None:
    """§9.6 の補償なし表 (``n_train`` / MG x ESN の NRMSE) が CSV の実測と一致する。

    ``test_design_table_values_match_the_threshold_csv`` (§9.2) と同じ役割を
    §9.6 に対して果たす。この一致が崩れたら「補償が無ければ滑らかな単調増加に
    見える」という D-19 の根拠が手書きの数値に戻っている。
    """
    header, table = _unpadded_washout_table()
    stats = _mg_esn_unpadded_stats_by_washout()
    washouts = [int(cell) for cell in header[1:]]
    n_train_row = next(row for row in table if row[0] == "`n_train`")
    nrmse_row = next(row for row in table if "NRMSE" in row[0])
    for washout, documented_n_train, documented_nrmse in zip(
        washouts, n_train_row[1:], nrmse_row[1:], strict=True
    ):
        actual_n_train, actual_nrmse = stats[washout]
        assert int(documented_n_train) == actual_n_train, washout
        assert float(documented_nrmse) == pytest.approx(actual_nrmse, rel=1.0e-3), (
            washout
        )


def test_design_doc_points_at_the_unpadded_regeneration_command() -> None:
    """補償なし表の出どころ (再生成手順と CSV 名) が §9.6 に書いてある。"""
    text = _text()
    assert "washout_sensitivity_unpadded.csv" in text
    assert "pad_series" in text
    assert "false" in text


# --- §11.2〜§11.5 (サイクル03) vs 一次資料 ------------------------------------
#
# 03 の節は「散文の数値が一次資料とずれる」事故を最初から潰す方針で書いてある:
# **表に書いた有効数字はすべて成果物 (meta.json / capacity*.csv / narma10.csv)
# かコード (count_targets / config.py) から機械照合される**。§9.2 / §9.6 と
# 同じ役割を §11 に対して果たす。

CAPACITY_RESULTS = ROOT / "results" / "03_capacity"
CAPACITY_META = CAPACITY_RESULTS / "meta.json"
CAPACITY_CSV = CAPACITY_RESULTS / "capacity.csv"
CAPACITY_PROFILE_CSV = CAPACITY_RESULTS / "capacity_profile.csv"
NARMA10_CSV = CAPACITY_RESULTS / "narma10.csv"
CAPACITY_CONFIG = ROOT / "experiments" / "03_capacity" / "config.yaml"
CONFIG_PY = ROOT / "src" / _PACKAGE / "config.py"

_MODE_CELL = re.compile(r"`(?P<mode>[a-z0-9_]+)`")
_LPAREN = "\uff08"
_RPAREN = "\uff09"
"""全角括弧 (ruff の RUF001 がソース中の全角記号を弾くためエスケープで書く)。"""

_DEFAULT_MARK = f"{_LPAREN}\u65e2\u5b9a{_RPAREN}"
"""§11.2 の表で既定モードの行に付ける印 (全角括弧つきの「既定」)。"""
_BACKTICKED = re.compile(r"^`(?P<name>[a-z0-9_]+)`$")
_FIRST_BACKTICKED = re.compile(r"`(?P<name>[a-z0-9_]+)`")


def _capacity_meta() -> dict[str, object]:
    loaded: dict[str, object] = json.loads(CAPACITY_META.read_text(encoding="utf-8"))
    return loaded


def _capacity_csv_rows() -> list[dict[str, str]]:
    with CAPACITY_CSV.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _entries(mapping: dict[str, object], key: str) -> list[dict[str, object]]:
    """JSON の配列を ``list[dict[str, object]]`` に絞り込む (mypy strict 対応)。"""
    value = mapping[key]
    assert isinstance(value, list), (key, value)
    entries: list[dict[str, object]] = []
    for item in value:
        assert isinstance(item, dict), item
        entries.append(item)
    return entries


def _mapping(mapping: dict[str, object], key: str) -> dict[str, object]:
    value = mapping[key]
    assert isinstance(value, dict), (key, value)
    return value


def _number(mapping: dict[str, object], key: str) -> float:
    value = mapping[key]
    assert isinstance(value, int | float) and not isinstance(value, bool), (key, value)
    return float(value)


def _table_after(pattern: re.Pattern[str]) -> tuple[list[str], list[list[str]]]:
    """``pattern`` に最初に一致するヘッダ行の表を (ヘッダ, データ行) で返す。"""
    lines = _text().splitlines()
    for index, line in enumerate(lines):
        if not pattern.match(line):
            continue
        header = [cell.strip() for cell in line.strip("|").split("|")]
        rows: list[list[str]] = []
        for body in lines[index + 2 :]:
            if not body.startswith("|"):
                break
            rows.append([cell.strip() for cell in body.strip("|").split("|")])
        return header, rows
    raise AssertionError(f"docs/design.md に表が見つかりません: {pattern.pattern}")


def _decimals(cell: str) -> int:
    """セルに書かれた小数点以下の桁数 (整数セルは 0)。"""
    _, _, fraction = cell.partition(".")
    return len(fraction)


def _assert_cell_matches(cell: str, actual: float, label: str) -> None:
    """セルの表記を**その桁数に丸めた実測値**と厳密に比較する。

    丸め桁数をセルから読むので、表の見やすさ (有効数字) と厳密さを両立できる。
    1桁でも書き換えれば落ちる。
    """
    assert float(cell) == round(actual, _decimals(cell)), (
        f"{label}: design.md は {cell} / 一次資料は {actual!r}"
    )


def _threshold_table(name: str) -> list[list[str]]:
    """§11.2 の MC / IPC 比較表 (``name`` は ``mc_total`` / ``ipc_total``)。"""
    header, rows = _table_after(
        re.compile(r"^\|\s*`threshold_mode`\s*\|\s*`" + name + r"`\s*\|")
    )
    return [[*row] for row in rows] if header else rows


def _threshold_header(name: str) -> list[str]:
    header, _ = _table_after(
        re.compile(r"^\|\s*`threshold_mode`\s*\|\s*`" + name + r"`\s*\|")
    )
    return header


@pytest.mark.parametrize(
    ("table_key", "meta_key"),
    [("mc_total", "memory_capacity"), ("ipc_total", "ipc")],
    ids=["mc", "ipc"],
)
def test_threshold_table_matches_meta_json(table_key: str, meta_key: str) -> None:
    """§11.2 のしきい値比較表が ``meta.json`` の ``threshold_comparison`` と一致する。

    受け入れ条件3 の要。列の対応はヘッダのバッククォート付きキー名がそのまま
    JSON のキーで、行の対応は1列目の ``threshold_mode``。**表の数値を1つでも
    書き換えたら落ちる** (セルの桁数に丸めた実測値と厳密比較する)。
    """
    comparison = _capacity_meta()["threshold_comparison"]
    assert isinstance(comparison, dict)
    entries = comparison[meta_key]
    assert isinstance(entries, list)
    header = _threshold_header(table_key)
    table = _threshold_table(table_key)
    assert len(table) == len(entries), (table, entries)

    keys = [
        match["name"]
        for cell in header
        if (match := _BACKTICKED.match(cell)) is not None
    ]
    assert keys[0] == "threshold_mode"
    for cells, entry in zip(table, entries, strict=True):
        assert isinstance(entry, dict)
        mode = _MODE_CELL.match(cells[0])
        assert mode is not None, cells
        assert mode["mode"] == entry["threshold_mode"], cells
        for index, key in enumerate(keys[1:], start=1):
            _assert_cell_matches(cells[index], float(entry[key]), f"{meta_key}.{key}")


def test_threshold_table_marks_the_default_mode() -> None:
    """既定 (D-27) の行に印が付いており、その印が ``meta.json`` と一致する。"""
    comparison = _capacity_meta()["threshold_comparison"]
    assert isinstance(comparison, dict)
    for table_key, default_key in (
        ("mc_total", "default_mc_mode"),
        ("ipc_total", "default_ipc_mode"),
    ):
        marked = [
            cells[0]
            for cells in _threshold_table(table_key)
            if _DEFAULT_MARK in cells[0]
        ]
        assert len(marked) == 1, marked
        assert f"`{comparison[default_key]}`" in marked[0]


def test_threshold_degree_table_matches_the_profile_csv() -> None:
    """§11.2 の次数別しきい値表が ``capacity_profile.csv`` / ``meta.json`` と一致する。

    サロゲートの行は長形式 CSV の ``threshold`` 列 (D-38)、``chi2`` の行は
    次数によらず ``meta.json`` の ``ipc_threshold_degree1`` と同じ値である
    (chi2 は次数に依存しない近似なので、全次数で同じ値になる)。
    """
    comparison = _capacity_meta()["threshold_comparison"]
    assert isinstance(comparison, dict)
    condition = comparison["condition"]
    assert isinstance(condition, dict)
    header, table = _table_after(re.compile(r"^\|\s*次数\s*\|"))
    degrees = [int(cell) for cell in header[1:]]

    with CAPACITY_PROFILE_CSV.open(encoding="utf-8", newline="") as handle:
        surrogate: dict[int, float] = {}
        for record in csv.DictReader(handle):
            if (
                record["experiment"] == condition["experiment"]
                and float(record["rho"]) == condition["rho"]
                and float(record["leak_rate"]) == condition["leak_rate"]
                and int(record["replicate"]) == condition["replicate"]
                and record["diagnostic"] == "ipc"
            ):
                surrogate.setdefault(int(record["degree"]), float(record["threshold"]))
    assert set(surrogate) == set(degrees), (surrogate, degrees)

    ipc_entries = comparison["ipc"]
    assert isinstance(ipc_entries, list)
    chi2 = next(
        float(entry["ipc_threshold_degree1"])
        for entry in ipc_entries
        if isinstance(entry, dict) and entry["threshold_mode"] == "chi2"
    )
    for cells in table:
        expected = (
            surrogate if "surrogate" in cells[0] else dict.fromkeys(degrees, chi2)
        )
        for degree, cell in zip(degrees, cells[1:], strict=True):
            _assert_cell_matches(cell, expected[degree], f"次数{degree} ({cells[0]})")


def test_threshold_section_claims_hold_in_the_data() -> None:
    """§11.2 の散文が主張する大小関係が一次資料でも成り立つ。

    表には有効数字が載っているが、散文は「1% と違わず」「2% に満たない」
    「約1.3倍」のように
    幅で書いてある。**その幅が実測と食い違ったら落とす** (数値を書かない代わりに
    主張が緩くなる、を防ぐ)。
    """
    comparison = _mapping(_capacity_meta(), "threshold_comparison")
    ipc = {str(entry["threshold_mode"]): entry for entry in _entries(comparison, "ipc")}
    mc = {
        str(entry["threshold_mode"]): entry
        for entry in _entries(comparison, "memory_capacity")
    }
    none_ipc = _number(ipc["none"], "ipc_total")
    assert abs(_number(ipc["surrogate"], "ipc_total") - none_ipc) / none_ipc < 0.01
    none_mc = _number(mc["none"], "mc_total")
    assert abs(_number(mc["surrogate"], "mc_total") - none_mc) / none_mc < 0.02
    # 「`surrogate` との差は 0.1% にも満たない」(chi2 の採否)
    surrogate_ipc = _number(ipc["surrogate"], "ipc_total")
    assert (
        abs(_number(ipc["chi2"], "ipc_total") - surrogate_ipc) / surrogate_ipc < 0.001
    )
    # 「`none` では目標が1本も落ちない」
    condition = _mapping(comparison, "condition")
    row = next(
        record
        for record in _capacity_csv_rows()
        if record["experiment"] == condition["experiment"]
        and float(record["rho"]) == condition["rho"]
        and float(record["leak_rate"]) == condition["leak_rate"]
        and int(record["replicate"]) == condition["replicate"]
    )
    assert int(_number(ipc["none"], "n_targets_kept")) == int(row["n_targets"])
    # 「`mc_effective_delay` は `surrogate` の約1.3倍に伸びる」
    ratio = _number(mc["none"], "mc_effective_delay") / _number(
        mc["surrogate"], "mc_effective_delay"
    )
    assert round(ratio, 1) == 1.3, ratio


def test_profile_row_counts_in_the_threshold_section_match_the_artifacts() -> None:
    """§11.2 の「118 条件で 21,812 行 -> 211,916 行」が成果物と一致する。

    `none` にすると全セルが正になるので、長形式の行数は
    ``Σ (n_delays + n_targets)`` になる (D-38 は正値セルだけを書く)。
    """
    meta = _capacity_meta()
    rows = _capacity_csv_rows()
    all_cells = sum(
        int(float(row["n_delays"])) + int(float(row["n_targets"])) for row in rows
    )
    text = _text()
    match = re.search(
        r"本番の (?P<conditions>[\d,]+) 条件で (?P<kept>[\d,]+) 行 -> "
        r"(?P<all>[\d,]+) 行",
        text,
    )
    assert match, "§11.2 に長形式の行数の記述が見つかりません"
    assert int(match["conditions"].replace(",", "")) == meta["n_rows"]
    assert int(match["kept"].replace(",", "")) == meta["n_profile_rows"]
    assert int(match["all"].replace(",", "")) == all_cells


def test_truncation_table_matches_count_targets() -> None:
    """§11.3 の打ち切り表が ``count_targets`` / 本番 YAML と一致する。

    目標数と heatmap セル数は閉形式で数えられる (3a で公開済み) ので、
    表の数値は**実行しなくても**検証できる。打ち切りを変えた瞬間に落ちる。
    """
    ipc_module = importlib.import_module(f"{_PACKAGE}.diagnostics.ipc")
    config_module = importlib.import_module(f"{_PACKAGE}.config")
    config = config_module.load_config_as(
        CAPACITY_CONFIG, config_module.Capacity03Config
    )
    expected = {
        tuple(config.ipc.max_delay_by_degree): config.ipc,
        tuple(config.conservation.max_delay_by_degree): dataclasses.replace(
            config.ipc, max_delay_by_degree=config.conservation.max_delay_by_degree
        ),
    }
    _, table = _table_after(re.compile(r"^\|\s*実験\s*\|\s*`max_delay_by_degree`\s*\|"))
    assert len(table) == len(expected), table
    for cells in table:
        truncation = ast.literal_eval(cells[1].strip("`"))
        assert truncation in expected, cells
        cfg = expected[truncation]
        assert int(cells[2].replace(",", "")) == ipc_module.count_targets(cfg)
        assert int(cells[3].replace(",", "")) == len(cfg.max_delay_by_degree) * max(
            cfg.max_delay_by_degree
        )
        counts: dict[int, int] = {}
        for target in ipc_module.enumerate_targets(cfg):
            degree = sum(order for _, order in target)
            counts[degree] = counts.get(degree, 0) + 1
        documented = [
            int(part.strip().replace(",", "")) for part in cells[4].split("/")
        ]
        assert documented == [counts[degree] for degree in sorted(counts)], cells


def _capacity_rows_for(experiment: str) -> list[dict[str, str]]:
    return [row for row in _capacity_csv_rows() if row["experiment"] == experiment]


def test_mc_sweep_delay_table_matches_the_capacity_csv() -> None:
    """§11.5 の3-A `mc_effective_delay` 表 (リーク率 x rho) が ``capacity.csv`` と一致.

    M3 (3b-2 reviewer-docs, F-3b2-1-003): §11.5 の3表 (3-A/3-B/3-B') は数値
    自体は正確だが、成果物を再生成しても表が古いままだと赤くならない
    (groupby 照合が0件だった)。他の §11.2/§11.5 の表と同じ規律
    (``_table_after`` + ``_assert_cell_matches``) をここにも適用する。
    """
    rows = _capacity_rows_for("3A_mc_sweep")
    assert rows, "3A_mc_sweep の行が capacity.csv に見つかりません"
    header, table = _table_after(
        re.compile(r"^\|\s*リーク率\s*\\\s*" + _RHO + r"\s*\|")
    )
    rho_grid = [float(cell) for cell in header[1:-1]]
    assert len(table) == 3, table
    for cells in table:
        leak_rate = float(cells[0])
        means: list[float] = []
        for index, rho in enumerate(rho_grid, start=1):
            subset = [
                row
                for row in rows
                if float(row["leak_rate"]) == leak_rate and float(row["rho"]) == rho
            ]
            assert subset, (leak_rate, rho)
            mean = statistics.fmean(float(row["mc_effective_delay"]) for row in subset)
            means.append(mean)
            _assert_cell_matches(
                cells[index], mean, f"leak={leak_rate} rho={rho} mc_effective_delay"
            )
        # 表の見出しは「比 (最大rho/最小rho)」で、rho_grid は昇順
        # (先頭=最小、末尾=最大)。
        _assert_cell_matches(
            cells[-1], means[-1] / means[0], f"leak={leak_rate} 比(最大rho/最小rho)"
        )


def test_ipc_sweep_table_matches_the_capacity_csv() -> None:
    """§11.5 の3-B `ipc_total`/`ipc_linear`/`ipc_nonlinear` 表 (リーク率 x rho, 12行)。

    M3 (F-3b2-1-003)。「非線形の割合」列も ``ipc_nonlinear / ipc_total`` の
    再計算と照合する (散文の主張 (a)(b) を裏付ける列そのもの)。
    """
    rows = _capacity_rows_for("3B_ipc_sweep")
    assert rows, "3B_ipc_sweep の行が capacity.csv に見つかりません"
    _, table = _table_after(
        re.compile(r"^\|\s*リーク率\s*\|\s*" + _RHO + r"\s*\|\s*`ipc_total`\s*\|")
    )
    assert len(table) == 12, table
    for cells in table:
        leak_rate = float(cells[0])
        rho = float(cells[1])
        subset = [
            row
            for row in rows
            if float(row["leak_rate"]) == leak_rate and float(row["rho"]) == rho
        ]
        assert subset, (leak_rate, rho)
        ipc_total = statistics.fmean(float(row["ipc_total"]) for row in subset)
        ipc_linear = statistics.fmean(float(row["ipc_linear"]) for row in subset)
        ipc_nonlinear = statistics.fmean(float(row["ipc_nonlinear"]) for row in subset)
        label = f"leak={leak_rate} rho={rho}"
        _assert_cell_matches(cells[2], ipc_total, f"{label} ipc_total")
        _assert_cell_matches(cells[3], ipc_linear, f"{label} ipc_linear")
        _assert_cell_matches(cells[4], ipc_nonlinear, f"{label} ipc_nonlinear")
        _assert_cell_matches(
            cells[5].removesuffix("%"),
            100.0 * ipc_nonlinear / ipc_total,
            f"{label} 非線形の割合",
        )


def test_conservation_table_matches_the_capacity_csv() -> None:
    """§11.5 の3-B' `ipc_total` (`saturation_ratio`) 表 (N x `state_noise`, 9セル)。

    M3 (F-3b2-1-003)。セルは ``"16.91 (0.676)"`` の形 (総容量と括弧内の比) で、
    両方を ``capacity.csv`` の ``ipc_total`` / ``ipc_saturation_ratio`` と照合する。
    """
    rows = _capacity_rows_for("3Bp_conservation")
    assert rows, "3Bp_conservation の行が capacity.csv に見つかりません"
    header, table = _table_after(re.compile(r"^\|\s*N\s*\\\s*`state_noise`\s*\|"))
    noise_grid = [float(cell) for cell in header[1:]]
    assert len(table) == 3, table
    cell_pattern = re.compile(
        r"^(?P<total>[0-9.]+)\s*" + r"\(" + r"\s*(?P<ratio>[0-9.]+)\s*" + r"\)$"
    )
    for cells in table:
        n_units = int(cells[0])
        for index, noise in enumerate(noise_grid, start=1):
            subset = [
                row
                for row in rows
                if int(row["n_units"]) == n_units and float(row["state_noise"]) == noise
            ]
            assert subset, (n_units, noise)
            ipc_total = statistics.fmean(float(row["ipc_total"]) for row in subset)
            ratio = statistics.fmean(
                float(row["ipc_saturation_ratio"]) for row in subset
            )
            match = cell_pattern.match(cells[index])
            assert match is not None, cells[index]
            label = f"N={n_units} state_noise={noise}"
            _assert_cell_matches(match["total"], ipc_total, f"{label} ipc_total")
            _assert_cell_matches(match["ratio"], ratio, f"{label} saturation_ratio")


def test_narma10_table_matches_the_committed_rows() -> None:
    """§11.5 の 3-C の成績表が ``narma10.csv`` / ``meta.json`` と一致する。"""
    with NARMA10_CSV.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    verdict = _capacity_meta()["narma10_verdict"]
    assert isinstance(verdict, dict)
    labels = {"線形": "linear", "遅延線": "delay_line", "ESN": "esn"}
    _, table = _table_after(re.compile(r"^\|\s*手法\s*\|\s*NMSE\s*\|\s*NRMSE\s*\|"))
    assert len(table) == len(labels), table
    for cells in table:
        method = labels[cells[0]]
        subset = [record for record in records if record["method"] == method]
        _assert_cell_matches(
            cells[1],
            statistics.fmean(float(record["nmse"]) for record in subset),
            f"{method}.nmse",
        )
        _assert_cell_matches(
            cells[2],
            statistics.fmean(float(record["nrmse"]) for record in subset),
            f"{method}.nrmse",
        )
        # meta.json の verdict とも突き合わせる (2つの一次資料が食い違ったら落ちる)
        nmse_mean = verdict["nmse_mean"]
        assert isinstance(nmse_mean, dict)
        _assert_cell_matches(cells[1], float(nmse_mean[method]), f"verdict.{method}")


def test_narma10_capacity_table_matches_the_capacity_csv() -> None:
    """§11.5 の 3-C の容量表が ``capacity.csv`` の ``3C_narma10`` 行と一致する。

    「容量が成績を説明する」という §11.5 の読みの根拠そのものなので、
    ここが手書きだと連載の主張が成果物から外れる。
    """
    row = next(
        record
        for record in _capacity_csv_rows()
        if record["experiment"] == "3C_narma10"
    )
    _, table = _table_after(re.compile(r"^\|\s*量\s*\|\s*値\s*\|"))
    assert table, "§11.5 に 3-C の容量表が見つかりません"
    for cells in table:
        match = _FIRST_BACKTICKED.search(cells[0])
        assert match is not None, cells
        _assert_cell_matches(cells[1], float(row[match["name"]]), match["name"])


def test_wall_time_table_matches_the_meta_json() -> None:
    """§11.5 の実行時間表が ``meta.json`` の ``wall_time_breakdown`` と一致する。

    実行時間は再生成のたびに動くので、表を更新せずに成果物だけ差し替えると
    ここが落ちる (§10 の 02 の表は手書きのままだが、03 は一次資料がある)。
    """
    meta = _capacity_meta()
    breakdown = meta["wall_time_breakdown"]
    assert isinstance(breakdown, list)
    comparison = meta["threshold_comparison"]
    assert isinstance(comparison, dict)
    _, table = _table_after(re.compile(r"^\|\s*区間\s*\|\s*条件数\s*\|"))
    labels = {"3-A": 0, "3-B": 1, "3-B'": 2, "3-C": 3}
    seen = 0
    for cells in table:
        if cells[0] in labels:
            entry = breakdown[labels[cells[0]]]
            assert isinstance(entry, dict)
            assert int(cells[1]) == entry["n_conditions"], cells
            for index, key in enumerate(
                ("wall_time_state_s", "wall_time_mc_s", "wall_time_ipc_s"), start=2
            ):
                _assert_cell_matches(cells[index], float(entry[key]), key)
            _assert_cell_matches(
                cells[5].strip("*"), float(entry["wall_time_s"]), "wall_time_s"
            )
            seen += 1
        elif "しきい値比較" in cells[0]:
            _assert_cell_matches(
                cells[5].strip("*"),
                _number(comparison, "wall_time_s"),
                "threshold_comparison.wall_time_s",
            )
            seen += 1
        elif "合計" in cells[0]:
            assert int(cells[1].strip("*")) == meta["n_rows"], cells
            _assert_cell_matches(
                cells[5].strip("*"), _number(meta, "wall_time_s"), "meta.wall_time_s"
            )
            seen += 1
    assert seen == len(labels) + 2, table


def test_artifact_size_table_matches_the_files() -> None:
    """§11.5 の成果物サイズ表が実ファイルと一致する (CSV 予算 5 MB の根拠)。"""
    _, table = _table_after(re.compile(r"^\|\s*ファイル\s*\|\s*行数\s*\|"))
    total = 0
    for cells in table:
        if not cells[0].startswith("`"):
            continue
        name = cells[0].strip("`")
        path = CAPACITY_RESULTS / name
        size = path.stat().st_size
        n_rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
        assert int(cells[1]) == n_rows, cells
        assert int(cells[2].removesuffix(" B").replace(",", "")) == size, cells
        total += size
    documented_total = next(cells for cells in table if "CSV 合計" in cells[0])[2]
    assert int(documented_total.strip("*").removesuffix(" B").replace(",", "")) == total


def test_config_py_line_count_in_the_design_doc_is_current() -> None:
    """§11.5 に書いた ``config.py`` の行数が実ファイルと一致する。

    04 冒頭の package 化の着手条件 (非空 600 行) の根拠なので、分割した瞬間に
    この記述が古くなる —— そのとき赤くする。
    """
    lines = CONFIG_PY.read_text(encoding="utf-8").splitlines()
    nonempty = sum(1 for line in lines if line.strip())
    match = re.search(
        r"\*\*非空 (\d+) 行\*\*" + _LPAREN + r"総 (\d+) 行" + _RPAREN, _text()
    )
    assert match, "§11.5 に config.py の行数の記述が見つかりません"
    assert int(match[1]) == nonempty
    assert int(match[2]) == len(lines)
    assert nonempty > 600, "着手条件 (非空600行) の記述を見直してください"


def test_design_doc_points_at_the_capacity_regeneration_command() -> None:
    """03 の数値の出どころ (成果物と再生成コマンド) が §11 に書いてある。"""
    text = _text()
    assert "make figures-03" in text
    assert "make saturation-03" in text
    assert "threshold_comparison" in text
    assert "capacity_profile.csv" in text

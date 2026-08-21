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
CONFIG_DIR = ROOT / "src" / _PACKAGE / "config"
_CONFIG_TABLE_HEADER = re.compile(
    r"^\|\s*モジュール\s*\|\s*持つもの\s*\|\s*非空行数\s*\|\s*総行数\s*\|"
)
"""§11.5 の ``config/`` 行数表のヘッダ (04 T1 で package 化、D-49)。"""

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


#: 実行時間セルの許容比 (D-89)。**厳密一致で縛らない** ——
#: ``wall_time`` は実行環境で変わるので、誰がどのマシンで再生成しても落ちる
#: 検査になってしまう (図のラベルを直しただけで design.md の4表が落ちた)。
#: 一方で「この区間が10倍遅くなった」は拾いたいので、桁が変わる手前で止める。
WALL_TIME_TOLERANCE = 2.0


def _assert_cell_matches(cell: str, actual: float, label: str) -> None:
    """セルの表記を**その桁数に丸めた実測値**と厳密に比較する。

    丸め桁数をセルから読むので、表の見やすさ (有効数字) と厳密さを両立できる。
    1桁でも書き換えれば落ちる。

    **実行時間のセルにこれを使ってはいけない。** 実行環境で変わる値なので
    ``_assert_wall_time_cell`` を使うこと (D-89)。
    """
    assert float(cell) == round(actual, _decimals(cell)), (
        f"{label}: design.md は {cell} / 一次資料は {actual!r}"
    )


#: CSV 合計サイズの予算 (design.md §11.5 が根拠として挙げている 5 MB)。
CSV_BUDGET_B = 5 * 1024 * 1024

#: サイズセルの許容比。**厳密一致で縛らない** —— CSV には ``wall_time`` 列が
#: 入っており、実行環境で桁数が変わるとファイル全体のバイト数まで動く。
#: 一方でこの表の役目は「CSV 合計が予算 5 MB に収まる根拠」を示すことなので、
#: 桁が変わらない範囲で通し、予算そのものは別に検査する。
SIZE_TOLERANCE = 1.05


def _assert_size_cell(cell: str, actual: int, label: str) -> None:
    """サイズのセルが実ファイルと**同じ桁**にあること。"""
    documented = int(cell.removesuffix(" B").replace(",", ""))
    assert actual > 0, f"{label}: 実ファイルが空です"
    ratio = documented / actual
    assert 1 / SIZE_TOLERANCE <= ratio <= SIZE_TOLERANCE, (
        f"{label}: design.md は {documented} B / 実ファイルは {actual} B で "
        f"{ratio:.3f} 倍ずれています。**wall_time 列の桁では説明できない差**なので、"
        "行数や列が変わっていないか確認してください。"
    )


def _assert_wall_time_cell(cell: str, actual: float, label: str) -> None:
    """実行時間のセルが一次資料と**同じ桁**にあること (D-89)。"""
    documented = float(cell)
    assert actual > 0, f"{label}: 一次資料の実行時間が正の値ではありません"
    ratio = documented / actual
    assert 1 / WALL_TIME_TOLERANCE <= ratio <= WALL_TIME_TOLERANCE, (
        f"{label}: design.md は {cell} 秒 / 一次資料は {actual:.2f} 秒 で "
        f"{ratio:.2f} 倍ずれています。**実行環境の差では説明できない開き**なので、"
        "その区間が重くなっていないか確認してください。"
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
                _assert_wall_time_cell(cells[index], float(entry[key]), key)
            _assert_wall_time_cell(
                cells[5].strip("*"), float(entry["wall_time_s"]), "wall_time_s"
            )
            seen += 1
        elif "しきい値比較" in cells[0]:
            _assert_wall_time_cell(
                cells[5].strip("*"),
                _number(comparison, "wall_time_s"),
                "threshold_comparison.wall_time_s",
            )
            seen += 1
        elif "合計" in cells[0]:
            assert int(cells[1].strip("*")) == meta["n_rows"], cells
            _assert_wall_time_cell(
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
        _assert_size_cell(cells[2], size, name)
        total += size
    documented_total = next(cells for cells in table if "CSV 合計" in cells[0])[2]
    _assert_size_cell(documented_total.strip("*"), total, "CSV 合計")
    assert total < CSV_BUDGET_B, (
        f"CSV 合計が予算 {CSV_BUDGET_B} B を超えました: {total} B"
    )


def test_config_package_line_counts_in_the_design_doc_are_current() -> None:
    """§11.5 の ``config/`` 行数表が実ファイルと一致する (04 T1 / D-49)。

    04 T1 で ``config.py`` (非空 615 行) を package 化するまでは、この検査は
    「単一ファイルの行数が着手条件 600 行を超えているか」を見ていた。分割後に
    見るのは表と実ファイルの一致である:

    **モジュールごとの行数**が表のとおりであること (分割方針の記録がドリフト
    すると、次に割るときの判断材料が嘘になる)。
    表の行が実在するモジュール集合と過不足なく一致することも同時に見るので、
    ``chaos04.py`` (04 T4) を足して表に書き忘れれば赤くなる。

    上限 (非空 300 行) そのものは ``tests/test_config_package_layout.py`` が
    持つ —— 同じ数字を2か所に書くと片方だけ更新されて食い違うため、ここは
    表と実ファイルの一致だけを見る。
    """
    actual = {
        path.name: path.read_text(encoding="utf-8").splitlines()
        for path in sorted(CONFIG_DIR.glob("*.py"))
    }
    documented: dict[str, tuple[int, int]] = {}
    _, table = _table_after(_CONFIG_TABLE_HEADER)
    for cells in table:
        if "合計" in cells[0]:
            continue
        name = cells[0].strip("`")
        documented[name] = (int(cells[2]), int(cells[3]))
    assert set(documented) == set(actual), (
        f"§11.5 の表と config/ の実ファイルが一致しません "
        f"(不足={sorted(set(actual) - set(documented))}, "
        f"余剰={sorted(set(documented) - set(actual))})"
    )

    for name, (doc_nonempty, doc_total) in documented.items():
        lines = actual[name]
        nonempty = sum(1 for line in lines if line.strip())
        assert doc_nonempty == nonempty, f"{name}: 非空行数が表と違います ({nonempty})"
        assert doc_total == len(lines), f"{name}: 総行数が表と違います ({len(lines)})"

    totals = next(cells for cells in table if "合計" in cells[0])
    assert int(totals[2].strip("*")) == sum(
        sum(1 for line in lines if line.strip()) for lines in actual.values()
    )
    assert int(totals[3].strip("*")) == sum(len(lines) for lines in actual.values())


def test_design_doc_points_at_the_capacity_regeneration_command() -> None:
    """03 の数値の出どころ (成果物と再生成コマンド) が §11 に書いてある。"""
    text = _text()
    assert "make figures-03" in text
    assert "make saturation-03" in text
    assert "threshold_comparison" in text
    assert "capacity_profile.csv" in text


# --- 04b-1 T4: カオス系の生成・Delta t の較正・lambda_max の推定 -------------

CHAOS_RESULTS = ROOT / "results" / "04_chaotic_freerun"
CHAOS_META = CHAOS_RESULTS / "meta.json"
CHAOS_CSV = CHAOS_RESULTS / "onestep.csv"

_CHAOS_GENERATION_HEADER = re.compile(r"^\|\s*項目\s*\|\s*値\s*\|\s*出どころ\s*\|")
_CHAOS_CALIBRATION_HEADER = re.compile(
    r"^\|\s*`sample_interval`\s*\|\s*[^|]*\|\s*1ステップ先 NRMSE"
)
_CHAOS_BOUND_HEADER = re.compile(r"^\|\s*#\s*\|\s*軸\s*\|\s*上限\s*\|\s*置き場所\s*\|")
_DIGITS = re.compile(r"\d+")
_LAMBDA = "\u03bb"
"""ギリシャ文字 lambda。``_RHO`` と同じ理由 (ruff の RUF001/RUF002 対策)。"""


def _chaos_meta() -> dict[str, object]:
    """04 の ``meta.json`` (4-A の一次資料)。"""
    loaded: dict[str, object] = json.loads(CHAOS_META.read_text(encoding="utf-8"))
    return loaded


def test_chaos_generation_table_matches_the_config() -> None:
    """§11 の Lorenz 生成表が ``LorenzConfig`` の既定と一致する (D-41)。

    要件書は「積分法・刻み幅・サンプリング間隔を記録」と求めている。記録と
    実装が食い違ったら、記録の方が嘘になる。
    """
    from rc_basics_lab.config import LorenzConfig

    cfg = LorenzConfig()
    _, table = _table_after(_CHAOS_GENERATION_HEADER)
    documented = {cells[0]: cells[1] for cells in table}
    assert "`rk4_step`" in " ".join(documented), documented
    interval_row = next(
        value for key, value in documented.items() if "sample_interval" in key
    )
    assert str(cfg.sample_interval) in interval_row
    assert f"{cfg.rk4_step * cfg.sample_interval:g}" in interval_row.replace(" ", "")
    step_row = next(value for key, value in documented.items() if "rk4_step" in key)
    assert float(step_row) == cfg.rk4_step
    burn_row = next(
        value for key, value in documented.items() if "integration_burn_in" in key
    )
    assert str(cfg.integration_burn_in) in burn_row
    assert str(cfg.integration_burn_in * cfg.sample_interval) in burn_row


def test_chaos_calibration_table_records_the_adopted_and_rejected_values() -> None:
    """Delta t の較正表に採用値と**落選値**の両方が載っている (仕様 §4 T4)。

    「図が良く見えるまで黙って調整する」経路を塞ぐための記録なので、採用値
    だけを書くのは記録として成立しない。採用行が実装の既定と一致することも
    同時に見る。
    """
    from rc_basics_lab.config import LorenzConfig

    _, table = _table_after(_CHAOS_CALIBRATION_HEADER)
    intervals: dict[int, list[str]] = {}
    for cells in table:
        interval = _DIGITS.search(cells[0])
        assert interval, f"較正表の行が sample_interval で始まっていません: {cells}"
        intervals[int(interval[0])] = cells
    assert set(intervals) == {5, 10, 25}, intervals
    adopted = [key for key, cells in intervals.items() if "採用" in cells[0]]
    rejected = [key for key, cells in intervals.items() if "落選" in cells[0]]
    assert len(adopted) == 1 and len(rejected) == 2
    assert adopted[0] == LorenzConfig().sample_interval


def test_chaos_lyapunov_record_matches_the_artifacts() -> None:
    """§11 の lambda_max の記録が ``meta.json`` と設定の文献値に一致する (D-42)。"""
    from rc_basics_lab.config import LORENZ_LYAPUNOV_REFERENCE

    text = _text()
    meta = _chaos_meta()
    lyapunov = meta["lyapunov"]
    assert isinstance(lyapunov, dict)
    estimated = float(lyapunov["lyapunov_per_time"])
    documented = re.search(rf"{_LAMBDA}_max = (\d+\.\d+) ", text)
    assert documented, "lambda_max の実測値が §11 に書かれていません"
    _assert_cell_matches(documented[1], estimated, "lyapunov_per_time")

    relative = re.search(r"相対差は\n?\s*\*\*(\d+\.\d+)%\*\*", text)
    assert relative, "文献値との相対差が §11 に書かれていません"
    _assert_cell_matches(
        relative[1], 100.0 * float(lyapunov["reference_rel_error"]), "rel_error"
    )
    assert f"**{LORENZ_LYAPUNOV_REFERENCE}**" in text, "文献値が §11 に書かれていません"
    assert "Viswanath" in text, "文献値の出典が §11 に書かれていません"


def test_chaos_wall_time_table_matches_the_meta_json() -> None:
    """§11 の 4-A の実行時間表が ``meta.json`` と一致する。

    実行時間は再生成のたびに動くので、表を更新せずに成果物だけ差し替えると
    ここが落ちる。
    """
    meta = _chaos_meta()
    breakdown = meta["wall_time_breakdown"]
    assert isinstance(breakdown, dict)
    _, table = _table_after(re.compile(r"^\|\s*区間\s*\|\s*実測\s*\|\s*予算\s*\|"))
    expected = {
        "真の軌道の生成": float(breakdown["lyapunov_s"]),
        "4-A": float(breakdown["onestep_s"]),
    }
    seen = 0
    for cells in table:
        for label, value in expected.items():
            if cells[0].startswith(label):
                _assert_wall_time_cell(
                    cells[1].strip("*").removesuffix(" 秒"), value, label
                )
                seen += 1
                break
    assert seen == len(expected), table


def test_chaos_allocation_bound_table_matches_the_module_constants() -> None:
    """§11 の確保軸表が実装の上限と一致する (D-34。**04 で新しい上限を作らない**)。"""
    import rc_basics_lab.tasks.chaotic as chaotic

    _, table = _table_after(_CHAOS_BOUND_HEADER)
    rows = {cells[0]: cells for cells in table}
    assert set(rows) == {"1", "2", "3", "8"}, rows
    assert "2e7" in rows["1"][2]
    assert chaotic._MAX_INTEGRATION_STEPS == 20_000_000
    assert "5e7" in rows["2"][2]
    assert chaotic._MAX_TRAJECTORY_ELEMENTS == 50_000_000
    assert "validate_state_matrix_bounds" in rows["3"][2]
    assert "D-34" in rows["8"][2]


def test_chaos_artifact_sizes_are_within_budget() -> None:
    """§12.7 に書いた成果物サイズが実ファイルと一致し、予算 5 MB の内側。"""
    total = sum(path.stat().st_size for path in CHAOS_RESULTS.glob("*.csv"))
    assert total < 5 * 1024 * 1024
    documented = re.search(r"CSV 合計 \| \*\*([\d,]+) KB\*\*", _text())
    assert documented, "§12.7 に CSV 合計サイズが書かれていません"
    assert int(documented[1].replace(",", "")) == round(total / 1024)


# --- §12 (04b-2 T5): 自由走行の設計と実測 --------------------------------------

_VALID_TIME_HEADER = re.compile(
    r"^\|\s*しきい値.*NRMSE 比.*\|\s*中央値 \[λ_max\^-1\]\s*\|"
)
_METHOD_TIME_HEADER = re.compile(
    r"^\|\s*手法\s*\|\s*中央値\s*\|\s*最小\s*\|\s*最大\s*\|"
)
_VERDICT_HEADER = re.compile(r"^\|\s*課題\s*\|\s*手法\s*\|\s*代替より近い\s*\|")
_REGIME_COUNT_HEADER = re.compile(
    r"^\|\s*状態ノイズ\s*\|\s*発散\s*\|\s*周期軌道\s*\|\s*アトラクタ再現\s*\|"
)
_REGIME_RULE_HEADER = re.compile(r"^\|\s*順\s*\|\s*3態\s*\|\s*条件\s*\|\s*定数\s*\|")
_STATS_STEPS_HEADER = re.compile(r"^\|\s*`stats_steps`\s*\|\s*時間単位\s*\|")
_AXES_HEADER = re.compile(
    r"^\|\s*#\s*\|\s*軸\s*\|\s*上限\s*\|\s*置き場所\s*\|\s*検査位置\s*\|"
)
_FULL_TIME_HEADER = re.compile(r"^\|\s*区間\s*\|\s*実測\s*\|\s*予算\s*\|")


def _sensitivity_entries() -> list[dict[str, object]]:
    return _entries(_chaos_meta(), "valid_time_sensitivity")


def _bold(cell: str) -> str:
    """太字と注記を落として中身だけにする (``**0.4 + 全角括弧の注記**`` -> ``0.4``)。"""
    return cell.strip("*").split(_LPAREN)[0].strip()


def test_chaos_valid_time_sensitivity_table_matches_the_meta_json() -> None:
    """§12.2 の閾値感度表が ``meta.json`` と一致する (D-43)。

    仕様 §8 が {0.2, 0.3, 0.4, 0.5} の感度表を要求している。表を成果物から
    切り離すと「その閾値だから出た結論」を否定できない。
    """
    entries = [
        entry
        for entry in _sensitivity_entries()
        if entry["task"] == "lorenz" and entry["method"] == "esn"
    ]
    assert entries, "meta.json に Lorenz / ESN の感度が入っていません"
    by_threshold = {float(str(entry["threshold"])): entry for entry in entries}
    _, table = _table_after(_VALID_TIME_HEADER)
    documented = {float(_bold(row[0])): row for row in table}
    assert set(documented) == set(by_threshold), (documented, by_threshold)
    for threshold, row in documented.items():
        entry = by_threshold[threshold]
        for index, key in enumerate(
            ("median_lyapunov", "min_lyapunov", "max_lyapunov"), start=1
        ):
            _assert_cell_matches(
                _bold(row[index]), float(str(entry[key])), f"{threshold}/{key}"
            )
        assert int(_bold(row[4])) == int(str(entry["n_censored"]))


def test_chaos_method_valid_time_table_matches_the_meta_json() -> None:
    """§12.2 の手法別の表が ``meta.json`` (しきい値 0.4) と一致する。"""
    from rc_basics_lab.config import Chaos04Config

    threshold = Chaos04Config().freerun.valid_time_threshold
    entries = {
        str(entry["method"]): entry
        for entry in _sensitivity_entries()
        if entry["task"] == "lorenz" and entry["threshold"] == threshold
    }
    labels = {"線形": "linear", "遅延線": "delay_line", "ESN": "esn"}
    _, table = _table_after(_METHOD_TIME_HEADER)
    assert {_bold(row[0]) for row in table} == set(labels)
    for row in table:
        entry = entries[labels[_bold(row[0])]]
        for index, key in enumerate(
            ("median_lyapunov", "min_lyapunov", "max_lyapunov"), start=1
        ):
            _assert_cell_matches(_bold(row[index]), float(str(entry[key])), key)


def test_chaos_attractor_verdict_table_matches_the_meta_json() -> None:
    """§12.3 の距離表が ``meta.json`` の ``attractor_verdict`` と一致する (D-46)。"""
    verdicts = {
        (str(entry["task"]), str(entry["method"])): entry
        for entry in _entries(_chaos_meta(), "attractor_verdict")
    }
    labels = {"線形": "linear", "遅延線": "delay_line", "ESN": "esn"}
    tasks = {"Lorenz": "lorenz", "Mackey-Glass": "mackey_glass"}
    _, table = _table_after(_VERDICT_HEADER)
    assert len(table) == len(verdicts)
    for row in table:
        entry = verdicts[(tasks[_bold(row[0])], labels[_bold(row[1])])]
        closer, total = (int(part) for part in _bold(row[2]).split("/"))
        assert closer == int(str(entry["n_closer"]))
        assert total == int(str(entry["n_rows"]))
        for index, keys in (
            (3, ("median_return_map", "median_return_map_surrogate")),
            (4, ("median_spectrum", "median_spectrum_surrogate")),
        ):
            cells = [cell.strip() for cell in _bold(row[index]).split("/")]
            for cell, key in zip(cells, keys, strict=True):
                value = float(str(entry[key]))
                if cell == "nan":
                    assert value != value, key
                else:
                    _assert_cell_matches(cell, value, key)


def test_chaos_regime_count_table_matches_the_meta_json() -> None:
    """§12.5 の3態の内訳が ``meta.json`` と一致する (受け入れ条件4)。"""
    counts = _chaos_meta()["regime_counts_by_noise"]
    assert isinstance(counts, dict)
    _, table = _table_after(_REGIME_COUNT_HEADER)
    assert len(table) == len(counts)
    for row in table:
        key = f"{float(_bold(row[0])):g}"
        actual = counts[key]
        assert isinstance(actual, dict)
        for index, regime in enumerate(("diverged", "periodic", "attractor"), start=1):
            assert int(_bold(row[index])) == int(actual.get(regime, 0))


def test_chaos_regime_rule_table_matches_the_module_constants() -> None:
    """§12.4 の3態の判定表が実装の定数と一致する (D-45)。

    「図から決めていない」の実体は、判定が定数と純関数だけで決まることである。
    表と定数が食い違ったら、表の方が嘘になる。
    """
    from rc_basics_lab.experiment import attractor

    _, table = _table_after(_REGIME_RULE_HEADER)
    assert [row[0] for row in table] == ["1", "2", "3"]
    assert str(attractor.AMPLITUDE_RATIO_MAX) in table[0][2]
    assert "AMPLITUDE_RATIO_MAX" in table[0][3]
    assert str(attractor.COLLAPSE_STD_RATIO) in table[1][2]
    assert str(attractor.PERIODIC_AUTOCORR) in table[1][2]
    assert attractor.COLLAPSE_STD_RATIO == 1.0 / attractor.AMPLITUDE_RATIO_MAX


def test_chaos_stats_steps_table_records_the_adopted_and_rejected_lengths() -> None:
    """§12.3 の ``stats_steps`` 較正表に採用値と**落選値**の両方が載っている。

    Delta t の較正表 (§11) と同じ規律 —— 採用値だけを書くのは記録として成立
    しない。「短い方がきれいな結果になる」ので、選び方を残さないと後から
    「都合の良い長さを選んだ」と区別できない。
    """
    from rc_basics_lab.config import Chaos04Config

    _, table = _table_after(_STATS_STEPS_HEADER)
    lengths = {int(_bold(row[0]).replace(",", "")) for row in table}
    adopted = [row for row in table if "採用" in row[0]]
    assert len(adopted) == 1, table
    assert len(lengths) >= 3, lengths
    assert (
        int(_bold(adopted[0][0]).replace(",", ""))
        == Chaos04Config().freerun.stats_steps
    )


def test_chaos_allocation_axis_table_covers_axes_four_to_seven() -> None:
    """§12.6 の確保軸表が軸4〜7 を1本ずつ挙げ、置き場所が実装と一致する。"""
    from rc_basics_lab.experiment import attractor, stability

    _, table = _table_after(_AXES_HEADER)
    rows = {row[0]: row for row in table}
    assert set(rows) == {"4", "5", "6", "7"}, rows
    assert "attractor.py" in rows["4"][3]
    assert "stability.py" in rows["5"][3]
    assert "freerun.py" in rows["6"][3]
    assert "attractor.py" in rows["7"][3]
    assert attractor._MAX_STATS_STEPS == 1_000_000
    assert stability._MAX_CONDITIONS == 2_000
    assert "条件を作る前" in rows["5"][4]


def test_chaos_full_wall_time_table_matches_the_meta_json() -> None:
    """§12.7 の実行時間表が ``meta.json`` の区間別内訳と一致する。"""
    meta = _chaos_meta()
    breakdown = meta["wall_time_breakdown"]
    assert isinstance(breakdown, dict)
    expected = {
        "真の軌道の生成": float(breakdown["lyapunov_s"]),
        "4-A": float(breakdown["onestep_s"]),
        "4-B": float(breakdown["freerun_s"]),
        "4-C": float(breakdown["stability_s"]),
        "4-D": float(breakdown["capacity_s"]),
        "図5枚": float(breakdown["figures_s"]),
        f"**合計{_LPAREN}`wall_time_s`{_RPAREN}**": _number(meta, "wall_time_s"),
    }
    header, table = _table_after(_FULL_TIME_HEADER)
    # §11 にも同じヘッダの表があるので、区間が7つそろう方 (§12.7) を探す。
    lines = _text().splitlines()
    tables: list[list[list[str]]] = []
    for index, line in enumerate(lines):
        if not _FULL_TIME_HEADER.match(line):
            continue
        rows: list[list[str]] = []
        for body in lines[index + 2 :]:
            if not body.startswith("|"):
                break
            rows.append([cell.strip() for cell in body.strip("|").split("|")])
        tables.append(rows)
    assert header, header
    full = max(tables, key=len)
    assert len(full) >= len(expected), full
    seen = 0
    for row in full:
        for label, value in expected.items():
            if row[0].startswith(label):
                _assert_wall_time_cell(_bold(row[1]).removesuffix(" 秒"), value, label)
                seen += 1
                break
    assert seen == len(expected), full
    assert _number(meta, "wall_time_s") < 900.0, "figures-04 が予算 900 秒を超えました"
    assert table is not None


# --- §13 (05b T5): 異常検知の設計と実測 ----------------------------------------

ANOMALY_META = ROOT / "results" / "05_anomaly_detection" / "meta.json"

_ANOMALY_TIME_HEADER = re.compile(
    r"^\|\s*区間\s*\|\s*実測 \[秒\]\s*\|\s*予算 \[秒\]\s*\|"
)
_ANOMALY_TIME_SECTIONS: dict[str, str] = {
    "5-A + 5-B": "headline_s",
    "時系列図の1例": "timeline_s",
    "5-C": "protocol_s",
    "5-D": "size_s",
    "図5枚": "figures_s",
}


def _anomaly_meta() -> dict[str, object]:
    loaded: dict[str, object] = json.loads(ANOMALY_META.read_text(encoding="utf-8"))
    return loaded


def test_anomaly_wall_time_table_matches_the_meta_json() -> None:
    """§13.3 の実行時間表が ``meta.json`` の区間別内訳・予算と一致する。

    区間ごとの予算は ``experiment.anomaly_pipeline.SECTION_BUDGETS_S`` が単一の
    真実で、``meta.json`` の ``wall_time_budget_s`` にそのまま載る。表だけを
    書き換えて予算を緩める経路を残さない。
    """
    meta = _anomaly_meta()
    breakdown = meta["wall_time_breakdown"]
    budgets = meta["wall_time_budget_s"]
    assert isinstance(breakdown, dict)
    assert isinstance(budgets, dict)
    header, table = _table_after(_ANOMALY_TIME_HEADER)
    assert header and table is not None, "§13.3 の実行時間表が見つかりません"
    seen = 0
    for row in table:
        label = _bold(row[0])
        for prefix, key in _ANOMALY_TIME_SECTIONS.items():
            if not label.startswith(prefix):
                continue
            _assert_wall_time_cell(_bold(row[1]), float(breakdown[key]), prefix)
            assert float(_bold(row[2])) == float(budgets[key]), prefix
            seen += 1
            break
        if label.startswith("合計"):
            _assert_wall_time_cell(_bold(row[1]), _number(meta, "wall_time_s"), "合計")
            assert float(_bold(row[2])) == _number(meta, "total_budget_s")
            seen += 1
    assert seen == len(_ANOMALY_TIME_SECTIONS) + 1, table
    assert _number(meta, "wall_time_s") < 900.0, "figures-05 が予算 900 秒を超えました"


def test_the_design_doc_records_the_dataset_selection_criteria() -> None:
    """§13.1 に選定基準の表があり、落選候補が名前で残っている (要件書 設計判断5)。

    「なぜ MGAB と UCR なのか」を散文だけにすると、次のサイクルが別のデータを
    足すときに基準が再現できない。落選の**理由と候補名**まで表に残す。
    """
    text = _text()
    start = text.index("### 13.1")
    end = text.index("### 13.2", start)
    section = text[start:end]
    rows = [line for line in section.splitlines() if line.startswith("| ")]
    assert len(rows) >= 6, f"§13.1 の選定基準表が縮んでいます: {len(rows)} 行"
    for name in ("SWaT", "Yahoo S5", "Exathlon", "NAB", "SMD"):
        assert name in section, f"落選候補 {name} が §13.1 から消えています"
    assert "MGAB" in section and "UCR" in section


# --- D-89: 実行環境で変わる数値の許容幅 --------------------------------------


def test_the_wall_time_tolerance_accepts_noise_and_rejects_a_drift() -> None:
    """実行時間・サイズの許容幅が**両方向に効いている**こと (D-89)。

    緩めた検査は、緩めすぎれば何も測らなくなる。ここで固定するのは
    「実行環境の揺れは通す」と「桁が変わる差は落とす」の**両方**である。
    片方だけなら、厳密一致に戻すのと、検査を消すのと同じになる。
    """
    from test_readme_summary import assert_wall_time_is_the_same_order

    # 環境差の範囲 (±40%) は通る —— 通らないと再生成のたびに文書を書き換える羽目になる
    _assert_wall_time_cell("7.0", 5.0, "noise")
    _assert_wall_time_cell("5.0", 7.0, "noise")
    assert_wall_time_is_the_same_order(7.0, 5.0)
    _assert_size_cell("1,020 B", 1000, "noise")

    # 桁が変わる差は落ちる —— 落ちないなら検査が無いのと同じ
    for documented, actual in ((30.0, 5.0), (5.0, 30.0)):
        with pytest.raises(AssertionError):
            _assert_wall_time_cell(str(documented), actual, "drift")
        with pytest.raises(AssertionError):
            assert_wall_time_is_the_same_order(documented, actual)
    with pytest.raises(AssertionError):
        _assert_size_cell("2,000 B", 1000, "drift")

    # 一次資料が壊れている (0 秒 / 空ファイル) ときは通さない
    with pytest.raises(AssertionError):
        _assert_wall_time_cell("1.0", 0.0, "broken")
    with pytest.raises(AssertionError):
        _assert_size_cell("1 B", 0, "broken")

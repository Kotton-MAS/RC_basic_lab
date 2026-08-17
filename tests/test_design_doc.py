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
import math
import re
from pathlib import Path

import pytest

from rc_basics_lab.experiment.threshold import sigma_column

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "docs" / "design.md"
THRESHOLD_CSV = (
    ROOT / "results" / "02_esp_and_dynamics" / "esp_threshold_sensitivity.csv"
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

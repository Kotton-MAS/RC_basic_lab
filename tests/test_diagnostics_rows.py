"""診断スカラの長形式書き出し (``experiment/diagnostics_rows.py``) の検査.

守るのは2つ。

1. **``condition_id`` の形式が安定していること** (D-118)。後から形式が変わると
   全実験の CSV が動き、実験をまたいだ突き合わせが静かに壊れる
2. **列が6つのまま増えないこと**。増やせる形にした瞬間に、この層を作った
   理由 (主表の列が診断のたびに増える) がここで再発する
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from rc_basics_lab.diagnostics.base import DiagnosticResult
from rc_basics_lab.experiment.diagnostics_rows import (
    DIAGNOSTICS_CSV_COLUMNS,
    DiagnosticScalarRow,
    condition_key,
    scalar_rows,
    write_diagnostics_csv,
)

# --- condition_id の形式 (D-118) --------------------------------------------


def test_the_condition_key_is_stable_under_dict_order() -> None:
    """**軸を入れる順が変わってもキーが同じ** (D-118 の guard_test)。

    昇順に固定していないと、呼び出し側が dict を作る順で同じ条件が別のキーに
    なる。実験をまたいだ突き合わせ (03 の容量と 04 の容量を merge する) が
    静かに壊れる形なので、ここで固定する。
    """
    forward = condition_key({"leak_rate": 1.0, "rho": 0.9})
    backward = condition_key({"rho": 0.9, "leak_rate": 1.0})
    assert forward == backward == "leak_rate=1|rho=0.9"


@pytest.mark.parametrize(
    ("axes", "expected"),
    [
        ({}, ""),
        ({"rho": 0.9}, "rho=0.9"),
        ({"n_units": 200}, "n_units=200"),
        ({"kind": "esn"}, "kind=esn"),
        ({"rho": 0.30000000000000004}, "rho=0.3"),
        ({"a": 1.0, "b": 2}, "a=1|b=2"),
    ],
)
def test_the_condition_key_format(axes: dict[str, object], expected: str) -> None:
    """形式そのものを固定する。

    浮動小数を ``%g`` で書くのは ``0.30000000000000004`` のような表現差で
    キーが割れないようにするためである。
    """
    assert condition_key(axes) == expected  # type: ignore[arg-type]


def test_different_conditions_get_different_keys() -> None:
    """違う条件は違うキーになる (**空虚でないことの確認**)。"""
    assert condition_key({"rho": 0.9}) != condition_key({"rho": 0.95})
    assert condition_key({"rho": 0.9}) != condition_key({"leak_rate": 0.9})


# --- 列が増えないこと --------------------------------------------------------


def test_the_columns_are_exactly_six() -> None:
    """列は6つのまま。

    **増やせる形にした瞬間に、この層を作った理由が消える。** 診断固有の情報は
    ``condition_id`` と ``key`` の中に入れる。
    """
    assert DIAGNOSTICS_CSV_COLUMNS == (
        "experiment",
        "condition_id",
        "replicate",
        "diagnostic",
        "key",
        "value",
    )


def test_the_row_fields_match_the_columns() -> None:
    """行 dataclass のフィールドと列順が一致する (片方だけ足す事故を塞ぐ)。"""
    import dataclasses

    names = tuple(f.name for f in dataclasses.fields(DiagnosticScalarRow))
    assert names == DIAGNOSTICS_CSV_COLUMNS


# --- 展開 --------------------------------------------------------------------


def test_every_scalar_becomes_one_row() -> None:
    """``scalars`` のキー1つが1行になる。"""
    results = [
        DiagnosticResult(name="memory_capacity", scalars={"mc_total": 20.1, "d": 3.0}),
        DiagnosticResult(name="ipc", scalars={"ipc_total": 5.0}),
    ]
    rows = scalar_rows(results, experiment="3A", condition_id="rho=0.9", replicate=1)
    assert len(rows) == 3
    assert {(r.diagnostic, r.key) for r in rows} == {
        ("memory_capacity", "mc_total"),
        ("memory_capacity", "d"),
        ("ipc", "ipc_total"),
    }
    assert all(r.experiment == "3A" and r.replicate == 1 for r in rows)


def test_the_scalar_order_follows_the_diagnostic() -> None:
    """キーの順は診断の宣言順のまま (昇順に並べ替えない)。

    「総量 -> 内訳」の順で返している診断があり、その順序は読み手にとって
    情報である。
    """
    result = DiagnosticResult(name="d", scalars={"total": 1.0, "part_a": 0.5})
    rows = scalar_rows([result], experiment="e", condition_id="", replicate=0)
    assert [r.key for r in rows] == ["total", "part_a"]


def test_arrays_are_not_written() -> None:
    """``arrays`` は長形式にも出さない (CSV が爆発するため)。"""
    result = DiagnosticResult(
        name="d",
        scalars={"total": 1.0},
        arrays={"profile": np.arange(100, dtype=np.float64)},
    )
    rows = scalar_rows([result], experiment="e", condition_id="", replicate=0)
    assert len(rows) == 1


# --- 書き出し ----------------------------------------------------------------


def test_the_csv_has_the_declared_columns(tmp_path: Path) -> None:
    """書き出した CSV の列が宣言どおりである。"""
    rows = scalar_rows(
        [DiagnosticResult(name="d", scalars={"k": 1.5})],
        experiment="e",
        condition_id="rho=0.9",
        replicate=0,
    )
    path = write_diagnostics_csv(rows, tmp_path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert tuple(reader.fieldnames) == DIAGNOSTICS_CSV_COLUMNS
        written = list(reader)
    assert written == [
        {
            "experiment": "e",
            "condition_id": "rho=0.9",
            "replicate": "0",
            "diagnostic": "d",
            "key": "k",
            "value": "1.5",
        }
    ]


def test_an_empty_result_writes_only_the_header(tmp_path: Path) -> None:
    """診断が1つも無くてもファイルは作る (成果物の有無が条件で変わらない)。"""
    path = write_diagnostics_csv((), tmp_path)
    assert path.is_file()
    assert path.read_text(encoding="utf-8").strip() == ",".join(DIAGNOSTICS_CSV_COLUMNS)

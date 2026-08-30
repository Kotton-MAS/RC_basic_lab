"""実験 3-S: 駆動入力の対称性と IPC の偶数次 (D-116).

記事03 §2.1 の「次数2と4のセルがほぼ空」の理由を**行の値で**判定できることを
固定する。ここで測るのは仮説そのもの (ゼロ対称な入力 + 奇関数 tanh なら偶数次が
消える) が、この配線で実際に検出できるかである。

**図は一切見ない。** 判定は ``even_degree_share_at_offset`` が行から計算する。
"""

from __future__ import annotations

import dataclasses
from dataclasses import fields

import numpy as np
import pytest

from rc_basics_lab.config import Capacity03Config, IpcConfig, SymmetrySweepConfig
from rc_basics_lab.experiment.capacity import EXPERIMENT_SYMMETRY
from rc_basics_lab.experiment.symmetry import (
    SYMMETRY_CSV_COLUMNS,
    SymmetryRow,
    even_degree_share_at_offset,
    run_symmetry_sweep,
)

SYMMETRIC = 0.0
"""ゼロ対称の基準点 (オフセット 0)。"""

SHIFTED = 3.0
"""対称性を壊す側の点 (``3 * sigma_u`` ずらす)。"""


def _small_config() -> Capacity03Config:
    """数秒で回る縮小設定 (2 点 x 1 レプリケート)。"""
    base = Capacity03Config()
    return dataclasses.replace(
        base,
        symmetry_sweep=SymmetrySweepConfig(
            offset_ratio_grid=(SYMMETRIC, SHIFTED),
            rho=0.95,
            leak_rate=1.0,
            sigma_u=0.2,
            n_units=20,
            n_steps=4000,
            n_replicates=1,
        ),
        ipc=IpcConfig(
            max_delay_by_degree=(8, 4, 2, 2),
            max_variables=2,
            n_surrogates=8,
            n_surrogate_targets=2,
            chunk_size=64,
        ),
    )


@pytest.fixture(scope="module")
def rows() -> tuple[SymmetryRow, ...]:
    """縮小設定で 3-S を1回だけ回す。"""
    return run_symmetry_sweep(_small_config())


def test_csv_columns_follow_the_row_declaration_order() -> None:
    """列順の単一の真実は ``SymmetryRow`` の宣言順。"""
    assert tuple(f.name for f in fields(SymmetryRow)) == SYMMETRY_CSV_COLUMNS


def test_every_row_is_labelled_as_the_symmetry_experiment(
    rows: tuple[SymmetryRow, ...],
) -> None:
    """``experiment`` 列が 3-S に固定されている (他の掃引と混ざらない)。"""
    assert {row.experiment for row in rows} == {EXPERIMENT_SYMMETRY}


def test_the_offset_moves_the_mean_but_not_the_spread(
    rows: tuple[SymmetryRow, ...],
) -> None:
    """**平均だけ**が動き、標準偏差は動かない (D-116 の設計そのもの)。

    分布の形を変えていないことがこの列で確かめられる。形を歪めると Legendre
    基底が正規直交でなくなり、容量が二重計上されて比較が無意味になる。
    """
    means = {row.offset_ratio: row.drive_mean for row in rows}
    stds = {row.offset_ratio: row.drive_std for row in rows}
    assert means[SYMMETRIC] == pytest.approx(0.0, abs=0.02)
    assert means[SHIFTED] == pytest.approx(SHIFTED * 0.2, abs=0.02)
    assert stds[SYMMETRIC] == pytest.approx(stds[SHIFTED], rel=1e-3)


def test_even_degrees_are_empty_for_a_symmetric_drive(
    rows: tuple[SymmetryRow, ...],
) -> None:
    """ゼロ対称な入力では偶数次の容量が総容量の 5% 未満 (仮説の前半)。"""
    assert even_degree_share_at_offset(rows, SYMMETRIC) < 0.05


def test_breaking_the_symmetry_makes_the_even_degrees_appear(
    rows: tuple[SymmetryRow, ...],
) -> None:
    """平均をずらすと偶数次が現れる (仮説の後半)。

    **この検査が仮説そのものである。** 落ちたら理由は対称性ではなく、基底の
    構成か打ち切りの側にある (その場合は D-116 の見直しが先)。
    """
    symmetric = even_degree_share_at_offset(rows, SYMMETRIC)
    shifted = even_degree_share_at_offset(rows, SHIFTED)
    assert shifted > 0.20, f"偶数次が現れませんでした: {shifted:.4f}"
    assert shifted > symmetric * 10.0, (
        f"対称 {symmetric:.4f} と非対称 {shifted:.4f} の差が小さすぎます"
    )


def test_degrees_are_long_form_and_cover_every_degree(
    rows: tuple[SymmetryRow, ...],
) -> None:
    """次数が**行の値**で、各条件が全次数ぶんの行を持つ (図に依存しない判定)。"""
    degrees = sorted({row.degree for row in rows})
    assert degrees == [1, 2, 3, 4]
    for offset in (SYMMETRIC, SHIFTED):
        selected = [row for row in rows if row.offset_ratio == offset]
        assert sorted(row.degree for row in selected) == degrees


def test_ipc_total_matches_the_sum_over_degrees(
    rows: tuple[SymmetryRow, ...],
) -> None:
    """``ipc_total`` が次数別容量の合計と一致する (行だけで検算できる)。"""
    for offset in (SYMMETRIC, SHIFTED):
        selected = [row for row in rows if row.offset_ratio == offset]
        total = {row.ipc_total for row in selected}
        assert len(total) == 1
        assert sum(row.capacity for row in selected) == pytest.approx(
            next(iter(total)), rel=1e-9
        )


def test_even_degree_share_at_offset_rejects_an_unknown_offset(
    rows: tuple[SymmetryRow, ...],
) -> None:
    """存在しないオフセットは ``ValueError`` (黙って 0 を返さない)。"""
    with pytest.raises(ValueError, match="の行がありません"):
        even_degree_share_at_offset(rows, 99.0)


def test_capacity_is_not_greater_than_the_raw_capacity(
    rows: tuple[SymmetryRow, ...],
) -> None:
    """しきい値後の容量はしきい値前を超えない。"""
    for row in rows:
        assert row.capacity <= row.capacity_raw + 1e-12


def test_drive_offset_is_the_only_difference_between_conditions(
    rows: tuple[SymmetryRow, ...],
) -> None:
    """オフセット以外の軸が条件間で同一 (対照として成立している)。"""
    axes = {
        (row.rho, row.leak_rate, row.n_units, row.sigma_u, row.n_steps) for row in rows
    }
    assert len(axes) == 1


def test_the_shifted_drive_is_still_uniform(rows: tuple[SymmetryRow, ...]) -> None:
    """一様分布の標準偏差と半値幅の関係が保たれている。

    ``sigma_u`` の一様分布なら半値幅は ``sqrt(3) * sigma_u``。実測の標準偏差が
    設定値と一致することで、オフセットが分布の形を変えていないことを見る。
    """
    for row in rows:
        assert row.drive_std == pytest.approx(row.sigma_u, rel=0.05)
        assert np.isfinite(row.capacity)

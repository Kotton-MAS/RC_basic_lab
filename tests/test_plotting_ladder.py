"""3-T / 3-C'' の図の検査 (D-145).

図が答えるのは「結論が掃引の1点で選んだ結果ではない」ことである。
**空虚になる形は「見出しが固定文で、行が変わっても同じことを言う」**なので、
行を作り変えて見出しが追随することを正面から測る。
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import pytest

from rc_basics_lab.experiment.narma_operating import OperatingPointRow
from rc_basics_lab.experiment.topology_ladder import TopologyLadderRow
from rc_basics_lab.plotting.figures_ladder import (
    AXIS_LABELS,
    BASELINE_LEVEL,
    LADDER_ARTICLE_AXES,
    TARGET_LEVEL,
    draw_ladder_panel,
    ladder_headline,
    paired_sign_test,
)
from rc_basics_lab.plotting.figures_operating import (
    draw_capacity_panel,
    operating_headline,
)
from rc_basics_lab.plotting.style import StyleContext, setup_style

LADDER_CSV = Path("results/03_capacity/capacity_topology.csv")
OPERATING_CSV = Path("results/03_capacity/narma10_operating.csv")


def _ladder_rows() -> tuple[TopologyLadderRow, ...]:
    """``capacity_topology.csv`` を行 dataclass に読み戻す。

    **``**dict`` で組まない。** CSV の値は全部 ``str`` なので、辞書を展開すると
    型が消えて mypy が列の取り違えを見なくなる (実測でそうなった)。列を
    1つずつ書くぶん、列名を変えたらここが落ちる。
    """
    with LADDER_CSV.open(encoding="utf-8", newline="") as handle:
        return tuple(
            TopologyLadderRow(
                experiment=row["experiment"],
                sweep_axis=row["sweep_axis"],
                level=row["level"],
                topology_kind=row["topology_kind"],
                graph=int(row["graph"]),
                replicate=int(row["replicate"]),
                n_units=int(row["n_units"]),
                n_steps=int(row["n_steps"]),
                rho=float(row["rho"]),
                leak_rate=float(row["leak_rate"]),
                sigma_u=float(row["sigma_u"]),
                state_noise=float(row["state_noise"]),
                nominal_density=float(row["nominal_density"]),
                realized_density=float(row["realized_density"]),
                in_degree_max=float(row["in_degree_max"]),
                in_degree_std=float(row["in_degree_std"]),
                gain_max=float(row["gain_max"]),
                gain_std=float(row["gain_std"]),
                mc_total=float(row["mc_total"]),
                mc_effective_delay=float(row["mc_effective_delay"]),
                ipc_total=float(row["ipc_total"]),
                ipc_linear=float(row["ipc_linear"]),
                ipc_nonlinear=float(row["ipc_nonlinear"]),
                wall_time_s=float(row["wall_time_s"]),
            )
            for row in csv.DictReader(handle)
        )


def _operating_rows() -> tuple[OperatingPointRow, ...]:
    """``narma10_operating.csv`` を行 dataclass に読み戻す。"""
    with OPERATING_CSV.open(encoding="utf-8", newline="") as handle:
        return tuple(
            OperatingPointRow(
                experiment=row["experiment"],
                n_units=int(row["n_units"]),
                leak_rate=float(row["leak_rate"]),
                method=row["method"],
                replicate=int(row["replicate"]),
                alpha=float(row["alpha"]),
                n_lags=int(row["n_lags"]),
                nrmse=float(row["nrmse"]),
                nmse=float(row["nmse"]),
                rmse=float(row["rmse"]),
                mc_total=float(row["mc_total"]),
                ipc_total=float(row["ipc_total"]),
                ipc_linear=float(row["ipc_linear"]),
                ipc_nonlinear=float(row["ipc_nonlinear"]),
                nonlinear_share=float(row["nonlinear_share"]),
                wall_time_s=float(row["wall_time_s"]),
            )
            for row in csv.DictReader(handle)
        )


def _style() -> StyleContext:
    return setup_style(commit="0" * 40)


def test_the_ladder_headline_follows_the_rows() -> None:
    """見出しが**行から導かれる** (固定文にしない、D-90 と同じ規律)。

    掃引を広げて結論が変わったときに図が静かに嘘をつくのを防ぐ。BA の値を
    ER より大きく作り変えて、見出しが「上回らない」から変わることを測る。
    """
    rows = _ladder_rows()
    style = _style()
    assert "上回らない" in ladder_headline(rows, style)
    flipped = tuple(
        dataclasses.replace(row, mc_total=row.mc_total + 100.0)
        if row.level == TARGET_LEVEL
        else row
        for row in rows
    )
    assert "上回らない" not in ladder_headline(flipped, style)


def test_the_ladder_panels_follow_the_sweep_axes() -> None:
    """パネルの数と並びが**掃引の出現順**と一致する (アルファベット順にしない)。"""
    rows = _ladder_rows()
    expected = list(dict.fromkeys(row.sweep_axis for row in rows if row.sweep_axis))
    assert expected, "掃引の行がありません"
    for axis_name in expected:
        assert axis_name in AXIS_LABELS, (
            f"見出しの無い軸を描こうとしています: {axis_name}"
        )


def test_an_unknown_sweep_axis_is_rejected() -> None:
    """見出しの決まっていない軸は**描く前に落とす** (FIG-5 と同じ規律)。"""
    from matplotlib.figure import Figure

    rows = tuple(
        dataclasses.replace(row, sweep_axis="temperature") for row in _ladder_rows()
    )
    axis = Figure().subplots(1, 1)
    with pytest.raises(ValueError, match="temperature"):
        draw_ladder_panel(axis, rows, "temperature", "mc_total", _style())


def test_the_article_axes_are_present_in_the_artifact() -> None:
    """記事に出す2軸が成果物に在る (D-146)。

    ここが欠けると図のパネルが静かに空になる。掃引を止めるなら**図の側の
    宣言も直す**、という対応関係をここで固定する。
    """
    present = {row.sweep_axis for row in _ladder_rows()}
    missing = [name for name in LADDER_ARTICLE_AXES if name not in present]
    assert not missing, f"記事に出す軸が成果物にありません: {missing}"


def test_the_paired_sign_test_matches_the_reported_direction() -> None:
    """図が使う対応のある差が、報告した向きと一致する (D-140 / D-144)。

    ここが崩れると、図と本文と CSV が別々のことを言い始める。
    """
    rows = [
        row
        for row in _ladder_rows()
        if row.sweep_axis == "rho" and row.rho == pytest.approx(0.95)
    ]
    mean, p_value = paired_sign_test(rows, BASELINE_LEVEL, "mc_total")
    assert mean < 0.0, f"BA が ER を上回っています: {mean}"
    assert p_value > 0.5, f"BA 優位の符号検定が有意になっています: {p_value}"


def test_the_operating_headline_follows_the_rows() -> None:
    """3-C'' の見出しも行から導かれる。"""
    rows = _operating_rows()
    style = _style()
    assert "12 点のうち 2 点" in operating_headline(rows, style)


def test_a_moving_delay_line_is_rejected() -> None:
    """**遅延線が動作点で動いていたら描かない** (D-144)。

    動いていたら掃引が課題か分割まで動かしており、図の主張 (「動作点で
    勝敗が変わる」) の対照そのものが壊れている。
    """
    rows = tuple(
        dataclasses.replace(row, nrmse=row.nrmse + 0.1 * row.n_units)
        if row.method == "delay_line"
        else row
        for row in _operating_rows()
    )
    from matplotlib.figure import Figure

    axis = Figure().subplots(1, 1)
    with pytest.raises(ValueError, match="動作点で動いています"):
        draw_capacity_panel(axis, rows, _style())


def test_the_narma10_title_names_the_winner_from_the_rows() -> None:
    """``fig_narma10`` の表題が**行から**勝者を数える (D-146)。

    3-C'' のパネルが入った時点で「遅延線が ESN を上回る」は報告する1点で
    しか正しくなくなった。固定文のままだと表題とパネル (c) が食い違い、
    図が読者に嘘をつく。ESN の成績を作り変えて表題が追随することを測る。
    """
    from test_plotting_capacity import narma10_rows

    from rc_basics_lab.experiment.runner import ResultRow
    from rc_basics_lab.plotting.figures_narma10 import narma10_verdict

    def rows(esn_nmse: float) -> tuple[ResultRow, ...]:
        return tuple(
            dataclasses.replace(row, nmse=esn_nmse) if row.method == "esn" else row
            for row in narma10_rows()
        )

    style = _style()
    assert "遅延線" in narma10_verdict(rows(9.0), style)
    assert "ESN" in narma10_verdict(rows(0.001), style)

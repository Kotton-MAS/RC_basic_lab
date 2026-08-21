"""連載通しの図の約束を固定する (docs/図の設計方針_RC基礎編.md / D-84〜D-88).

図は記事から切り離されて流通する。切り離された時点で「何を主張しているか」
「先行研究のどこに位置するか」「どの条件で測ったか」が読めなくなる、という
のがこの検査群の背景である (設計方針の「目的」節)。

ここで測るのは**5記事に共通する約束**だけで、個々の図の中身は
``test_plotting_*.py`` が見る。

- FIG-3 / D-84: 文献由来の参照線は、凡例テキストに出典を持つ
- FIG-5 / D-85: 手法と色の対応は ``style.METHOD_COLORS`` が単一の真実
- FIG-5 / D-86: 参照線は専用色 + 破線で、データ系列と同色にならない
- FIG-6 / D-87: どの図にも再現条件の footnote が焼き込まれる
- FIG-7 / D-88: 打ち切りの外 (未計算) は容量 0 と別の色になる
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from test_plotting_capacity import (
    conservation_rows,
    ipc_sweep_profile,
    ipc_sweep_rows,
    mc_sweep_profile,
    mc_sweep_rows,
    narma10_rows,
)

from rc_basics_lab.plotting import (
    figures_anomaly,
    figures_capacity,
    figures_freerun,
    heatmap,
    style,
)
from rc_basics_lab.plotting.capacity_grids import (
    even_degree_note,
    even_degree_share,
)
from rc_basics_lab.plotting.figures_capacity import (
    plot_ipc_conservation,
    plot_ipc_profile,
    plot_mc_sweep,
    plot_narma10_control,
)
from rc_basics_lab.plotting.labels import METHOD_LABELS, SOURCE_UNIDENTIFIED, cited
from rc_basics_lab.plotting.style import (
    METHOD_COLORS,
    REFERENCE_COLOR,
    StyleContext,
    method_color,
    reference_line_kwargs,
)
from rc_basics_lab.types import BoolArray

PLOTTING_DIR = Path(style.__file__).parent

CITATION = re.compile(r"\[[^\]]*(?:\d{4}|source unidentified|原典未特定)[^\]]*\]")
"""凡例テキストに出典 (``[著者 年]`` / ``[原典未特定]``) が在るか。"""

CONTEXT = StyleContext(cjk_font=None, commit="0123456789abcdef")
"""英語ラベル + 既知のコミットで描く (実行環境の HEAD に依存させない)。"""


@pytest.fixture
def capture_figures(monkeypatch: pytest.MonkeyPatch) -> list[Figure]:
    """``figures_capacity._save`` を包んで描き上がった ``Figure`` を捕まえる。

    PNG の画素からは凡例テキストも軸も読めないので、保存直前の ``Figure`` を
    見る (``test_plotting_capacity.py::captured`` と同じ手口)。
    """
    figures: list[Figure] = []
    original = figures_capacity._save

    def spy(figure: Figure, path: Path) -> Path:
        figures.append(figure)
        return original(figure, path)

    monkeypatch.setattr(figures_capacity, "_save", spy)
    return figures


def _reference_lines(figure: Figure) -> list[Line2D]:
    """参照線の色で描かれた凡例つきの線。"""
    return [
        line
        for line in figure.findobj(Line2D)
        if isinstance(line, Line2D)
        and line.get_label()
        and not str(line.get_label()).startswith("_")
        and _is_reference_colour(line)
    ]


def _is_reference_colour(line: Line2D) -> bool:
    from matplotlib.colors import to_hex

    try:
        return to_hex(line.get_color()) == to_hex(REFERENCE_COLOR)
    except ValueError:  # pragma: no cover - 色指定が壊れているときだけ
        return False


def _texts(figure: Figure) -> list[str]:
    return [
        text.get_text()
        for text in figure.findobj(Text)
        if isinstance(text, Text) and text.get_text()
    ]


# --- FIG-3 / D-84: 参照線に出典 ---------------------------------------------


def test_literature_reference_lines_carry_their_source_in_the_legend(
    tmp_path: Path, capture_figures: list[Figure]
) -> None:
    """3-A / 3-B' / 3-C の参照線の凡例に出典が入っている (FIG-3 / D-84)。

    本文の注は図が切り取られると消える。**凡例テキスト自体**に
    ``[Jaeger 2002 / Dambre 2012]`` / ``[原典未特定]`` を載せる。

    変異注入: ``cited(...)`` を外して素のラベルに戻すと、その図の参照線が
    出典なしになってここが落ちる。
    """
    mc_rows = mc_sweep_rows()
    plot_mc_sweep(mc_rows, mc_sweep_profile(mc_rows), tmp_path / "a.png", style=CONTEXT)
    plot_ipc_conservation(conservation_rows(), tmp_path / "b.png", style=CONTEXT)
    plot_narma10_control(narma10_rows(), tmp_path / "c.png", style=CONTEXT)
    assert len(capture_figures) == 3

    checked: set[str] = set()
    for figure in capture_figures:
        # 凡例は同じ線の複製 (proxy artist) を持つのでラベルの集合で数える
        labels = {str(line.get_label()) for line in _reference_lines(figure)}
        assert labels, "参照線が1本も見つかりません (色の約束が壊れています)"
        for label in labels:
            assert CITATION.search(label), f"参照線に出典がありません: {label!r}"
        checked |= labels
    # 3-A に1本 + 3-B' に1本 + 3-C に2本
    assert len(checked) == 4, f"検査した参照線が {sorted(checked)} です"


def test_cited_refuses_a_reference_line_without_a_source() -> None:
    """出典を書き忘れた参照線は**描く前に**落とす (FIG-3 / D-84)。"""
    assert cited("上限 MC <= N", "Jaeger 2002") == "上限 MC <= N [Jaeger 2002]"
    assert SOURCE_UNIDENTIFIED[0] and SOURCE_UNIDENTIFIED[1]
    with pytest.raises(ValueError, match="出典"):
        cited("上限 MC <= N", "")
    with pytest.raises(ValueError, match="出典"):
        cited("", "Jaeger 2002")


# --- FIG-5 / D-85: 手法の色 --------------------------------------------------


def test_method_colours_are_fixed_for_the_whole_series() -> None:
    """手法4色が ``style`` に固定され、未知の手法は落ちる (FIG-5 / D-85)。"""
    assert set(METHOD_COLORS) == {"linear", "delay_line", "delay_line_ols", "esn"}
    assert len(set(METHOD_COLORS.values())) == len(METHOD_COLORS), "色が重複しています"
    assert set(METHOD_LABELS) == set(METHOD_COLORS), (
        "ラベルの表と色の表で手法の集合が違います"
    )
    assert method_color("esn") == METHOD_COLORS["esn"]
    with pytest.raises(ValueError, match="手法の色"):
        method_color("nonexistent_method")


def test_no_plotting_module_picks_a_tab_colour_by_hand() -> None:
    """作図層に ``tab:`` 直書きが残っていない (FIG-5 / D-85)。

    色の対応表を ``style`` に置いても、各モジュールが自分で ``tab:blue`` を
    選び直せば連載通しの約束は静かに崩れる。**直書きの禁止まで含めて**
    1つの約束である。
    """
    literal = re.compile(r"[\"']tab:")
    offenders = {
        path.name: [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if literal.search(line)
        ]
        for path in sorted(PLOTTING_DIR.glob("*.py"))
        if literal.search(path.read_text(encoding="utf-8"))
    }
    assert not offenders, f"matplotlib の tab: 色が直書きされています: {offenders}"


# --- FIG-5 / D-86: 参照線の体裁 ---------------------------------------------


def test_the_reference_colour_is_never_a_data_series_colour() -> None:
    """参照線の色が、どのデータ系列の色とも重ならない (FIG-5 / D-86)。"""
    palettes = {
        "style.METHOD_COLORS": set(METHOD_COLORS.values()),
        "figures_anomaly.METHOD_COLORS": set(figures_anomaly.METHOD_COLORS.values()),
        "figures_freerun.REGIME_COLORS": set(figures_freerun.REGIME_COLORS.values()),
        "figures_freerun.SOURCE_STYLE": {
            value[0] for value in figures_freerun.SOURCE_STYLE.values()
        },
    }
    clashes = {
        name: colours
        for name, colours in palettes.items()
        if REFERENCE_COLOR in colours
    }
    assert not clashes, f"参照線の色がデータ系列と同色です: {clashes}"


def test_reference_lines_are_dashed_and_share_one_colour() -> None:
    """参照線は常に同色 + 破線で、複数本あれば刻みで区別する (FIG-5 / D-86)。"""
    first = reference_line_kwargs(0)
    second = reference_line_kwargs(1)
    assert first["color"] == second["color"] == REFERENCE_COLOR
    assert first["linestyle"] != second["linestyle"], "参照線が2本とも同じ刻みです"
    for kwargs in (first, second):
        offset, dashes = kwargs["linestyle"]
        assert offset == 0.0
        assert len(dashes) == 2 and all(value > 0.0 for value in dashes)
    # 何本目でも巡回するだけで色は変わらない
    assert reference_line_kwargs(7)["color"] == REFERENCE_COLOR


# --- FIG-6 / D-87: 再現条件の footnote --------------------------------------


def _plot_functions() -> dict[str, ast.FunctionDef]:
    """作図層の ``plot_*`` 関数 (AST)。"""
    found: dict[str, ast.FunctionDef] = {}
    for path in sorted(PLOTTING_DIR.glob("figures*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("plot_"):
                found[f"{path.name}::{node.name}"] = node
    return found


def _calls(node: ast.AST) -> set[str]:
    return {
        child.func.id
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }


def test_every_figure_burns_in_its_reproduction_conditions() -> None:
    """全 21 図が footnote を書いている (FIG-6 / D-87)。

    ``meta.json`` にある条件は、図が単体で流通した時点で失われる。
    **図を1枚足したときに footnote を忘れる**のがこの約束の壊れ方なので、
    実行時ではなく AST で全 ``plot_*`` を数える (描画を通さなくても効く)。
    """
    functions = _plot_functions()
    assert len(functions) >= 19, f"作図関数が {len(functions)} 個しかありません"
    missing = sorted(
        name
        for name, node in functions.items()
        if not ({"add_provenance", "add_footnote"} & _calls(node))
        # 図を組み立てる側 (build_*) に委譲している場合は呼び出し先が持つ
        and "build_protocol_sensitivity_figure" not in _calls(node)
    )
    assert not missing, f"再現条件の footnote が無い図があります: {missing}"


def test_the_footnote_text_names_the_conditions_and_the_commit(
    tmp_path: Path, capture_figures: list[Figure]
) -> None:
    """footnote に条件・レプリケート数・コミットが ASCII で載る (FIG-6 / D-87)。"""
    rows = mc_sweep_rows()
    plot_mc_sweep(rows, mc_sweep_profile(rows), tmp_path / "mc.png", style=CONTEXT)
    footnotes = [
        text
        for text in _texts(capture_figures[0])
        if "commit=" in text and "rep (replicate" in text
    ]
    assert len(footnotes) == 1, f"footnote が1行ではありません: {footnotes}"
    footnote = footnotes[0]
    assert footnote.startswith("N = 20, sigma_u = ")
    assert "2 rep (replicate 0-1)" in footnote
    assert footnote.endswith("commit=0123456")  # 先頭7桁だけを載せる
    assert footnote.isascii(), "footnote は ASCII だけで組む (CJK フォント非依存)"


def test_the_footnote_is_omitted_when_the_commit_is_unknown(
    tmp_path: Path, capture_figures: list[Figure]
) -> None:
    """コミットが分からないときは ``commit=`` の項だけが落ちる (FIG-6)。"""
    rows = mc_sweep_rows()
    plot_mc_sweep(
        rows,
        mc_sweep_profile(rows),
        tmp_path / "mc.png",
        style=StyleContext(cjk_font=None),
    )
    footnotes = [
        text for text in _texts(capture_figures[0]) if "rep (replicate" in text
    ]
    assert len(footnotes) == 1
    assert "commit=" not in footnotes[0]


# --- FIG-4: 上限が遠いときの正規化軸 ----------------------------------------


def test_the_mc_sweep_shows_the_capacity_normalised_by_the_bound(
    tmp_path: Path, capture_figures: list[Figure]
) -> None:
    """3-A に ``MC / N`` の第2軸が在る (FIG-4)。

    上限 200 と実測 10〜36 は対数軸でも離れすぎていて、「上限からどれだけ
    遠いか」が読めない。第2軸が消えたらここが落ちる。
    """
    rows = mc_sweep_rows(leaks=(0.3,))
    n_units = float(rows[0].n_units)
    plot_mc_sweep(rows, mc_sweep_profile(rows), tmp_path / "mc.png", style=CONTEXT)
    panel = capture_figures[0].axes[0]
    children = list(panel.child_axes)
    assert len(children) == 1, f"第2軸が {len(children)} 本です"
    # 目盛の範囲が主軸を N で割ったものになっている (ラベルではなく変換を見る)
    low, high = panel.get_ylim()
    assert children[0].get_ylim() == pytest.approx((low / n_units, high / n_units))
    # 「右軸が何か」はパネルのタイトルに書く (第2軸は constrained layout の
    # 管理外で、軸ラベルを付けると隣のパネルに重なるため)
    assert "MC / N" in panel.get_title()


# --- FIG-7 / D-88: 未計算と 0 を区別する ------------------------------------


def test_uncomputed_cells_are_masked_instead_of_being_drawn_as_zero() -> None:
    """打ち切りの外がマスクされる (FIG-7 / D-88)。"""
    cells = np.ones((2, 5), dtype=np.float64)
    masked = heatmap.masked_beyond_truncation(cells, {1: 5, 2: 2})
    assert np.ma.getmaskarray(masked)[1, 2:].all(), "次数2 の打ち切り外が未マスクです"
    assert not np.ma.getmaskarray(masked)[0].any(), "次数1 は打ち切りに達していません"
    # 打ち切りが分からないときは「未計算」を捏造しない
    assert np.ma.getmaskarray(heatmap.masked_beyond_truncation(cells, None)).sum() == 0


def test_the_ipc_profile_separates_uncomputed_cells_from_zero_capacity(
    tmp_path: Path, capture_figures: list[Figure]
) -> None:
    """3-B のヒートマップが未計算を 0 と別の色で描く (FIG-7 / D-88)。

    面積の約7割が 0 のこの図で打ち切りの外を 0 と同じ配色にすると、読者は
    打ち切り設定の形を系のデータ構造として読む。**変異注入**:
    ``max_delay_by_degree`` を渡さない (= 全セル計算済みとして描く) と
    マスクが 0 個になり、境界線も 0 本になってここが落ちる。
    """
    rows = ipc_sweep_rows()
    plot_ipc_profile(
        rows,
        ipc_sweep_profile(rows),
        tmp_path / "ipc.png",
        style=CONTEXT,
        max_delay_by_degree={1: 3, 2: 1},
    )
    figure = capture_figures[0]
    meshes = [mesh for mesh in figure.findobj(QuadMesh) if isinstance(mesh, QuadMesh)]
    assert meshes, "ヒートマップがありません"
    masked_cells = 0
    for mesh in meshes:
        values = mesh.get_array()
        assert values is not None
        mask: BoolArray = np.ma.getmaskarray(np.ma.asarray(values))
        masked_cells += int(np.count_nonzero(mask))
    assert masked_cells > 0, "未計算のセルが1つもマスクされていません"
    # 未計算の色は配色 (viridis) のどの値とも違う無彩色
    bad = meshes[0].get_cmap().get_bad()
    assert bad[3] > 0.0, "未計算のセルが透明になっています (0 と区別できません)"
    # 打ち切り境界の線が描かれている
    edges = [
        line
        for line in figure.findobj(Line2D)
        if isinstance(line, Line2D)
        and line.get_color() == heatmap.TRUNCATION_EDGE_COLOR
    ]
    assert edges, "打ち切り境界の線が引かれていません"


def test_the_ipc_profile_explains_why_the_even_degrees_are_empty(
    tmp_path: Path, capture_figures: list[Figure]
) -> None:
    """3-B の図に「偶数次が空である理由」の注がある (D-94)。

    FIG-7 で「未計算」を 0 と区別できるようにした以上、「0 である理由」も
    要る —— 読者が最初に目を引かれるのは「なぜ次数2と4だけ空なのか」である。

    注の割合は**行から数え直す**ので、掃引の設定が変われば注も変わる
    (固定文は結果が変わったときに静かに嘘をつく。3-C のタイトルと同じ理由)。
    **変異注入**: ``even_degree_share`` の偶奇判定を反転すると、注の割合が
    偶数次ではなく奇数次のものになり、ここが落ちる。
    """
    rows = ipc_sweep_rows()
    profile = ipc_sweep_profile(rows)
    plot_ipc_profile(
        rows,
        profile,
        tmp_path / "ipc.png",
        style=CONTEXT,
        max_delay_by_degree={1: 3, 2: 1},
    )
    figure = capture_figures[0]
    # 言語に依存させない (CONTEXT は英語ラベル)。日英どちらでも同じ注が出る。
    expected = CONTEXT.label(*even_degree_note(profile))
    notes = [
        text.get_text()
        for text in figure.findobj(Text)
        if isinstance(text, Text) and text.get_text() == expected
    ]
    assert notes, "偶数次が空である理由の注がありません。期待した文字列:\n" + expected
    note = notes[0]
    assert "tanh" in note and "symmetric" in note, note

    # 割合が実測から生成されている (手書きの固定文なら一致しない)
    share = even_degree_share(profile)
    assert f"{share:.1%}" in note, (note, share)
    total = sum(row.capacity for row in profile)
    even = sum(row.capacity for row in profile if row.degree % 2 == 0)
    assert share == pytest.approx(even / total)

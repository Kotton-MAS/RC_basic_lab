"""連載通しの図の約束を固定する (docs/series/図の設計方針_RC基礎編.md / D-84〜D-88).

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
import inspect
import re
import struct
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from test_esp_pipeline import TINY_CONFIG as _ESP_TINY
from test_esp_pipeline import _write as _esp_write
from test_plotting_anomaly import anomaly_rows, threshold_rows
from test_plotting_capacity import (
    conservation_rows,
    ipc_sweep_profile,
    ipc_sweep_rows,
    mc_sweep_profile,
    mc_sweep_rows,
    narma10_rows,
    plot_narma10_control,
)
from test_plotting_freerun import freerun_rows

from rc_basics_lab.config import Esp02Config, load_config_as
from rc_basics_lab.experiment.esp import EXPERIMENT_ESP_MAP
from rc_basics_lab.experiment.esp_pipeline import run_and_report_esp
from rc_basics_lab.experiment.freerun_rows import (
    FreeRunProfileRow,
)
from rc_basics_lab.experiment.horizon import HORIZON_STEPS, HorizonRow
from rc_basics_lab.experiment.narma import (
    NARMA10_REFERENCE_NOTE,
    NARMA10_REFERENCE_NOTE_EN,
)
from rc_basics_lab.experiment.runner import ResultRow
from rc_basics_lab.experiment.waveform_data import WaveformPanel
from rc_basics_lab.plotting import (
    figures_anomaly,
    figures_capacity,
    figures_esp,
    figures_freerun,
    figures_ipc_profile,
    figures_mc_sweep,
    heatmap,
    style,
    waveforms,
)
from rc_basics_lab.plotting.capacity_grids import (
    even_degree_note,
    even_degree_share,
)
from rc_basics_lab.plotting.figures_anomaly import (
    RANDOM_SCORE_F1_CONDITIONS,
    RANDOM_SCORE_PLAIN_F1,
    plot_threshold_tradeoff,
)
from rc_basics_lab.plotting.figures_capacity import (
    plot_ipc_conservation,
)
from rc_basics_lab.plotting.figures_esp import plot_esp_map
from rc_basics_lab.plotting.figures_freerun import plot_valid_time
from rc_basics_lab.plotting.figures_horizon import (
    JAEGER_CONDITIONS,
    JAEGER_LOG10_NRMSE84,
    JAEGER_PREVIOUS_LOG10,
    draw_horizon_panel,
)
from rc_basics_lab.plotting.figures_ipc_profile import plot_ipc_profile
from rc_basics_lab.plotting.figures_mc_sweep import plot_mc_sweep
from rc_basics_lab.plotting.freerun_headlines import (
    LITERATURE_VALID_TIME,
    LITERATURE_VALID_TIME_CONDITIONS,
)
from rc_basics_lab.plotting.labels import (
    APPELTANT_2011,
    GAUTHIER_2021,
    JAEGER_HAAS_2004,
    KIM_2022,
    METHOD_LABELS,
    SOURCE_UNIDENTIFIED,
    VINCKIER_2015,
    cited_bound,
    cited_measurement,
)
from rc_basics_lab.plotting.narma10_panel import (
    REFERENCE_CONDITIONS,
    REFERENCE_LABELS,
    REFERENCE_SOURCES,
)
from rc_basics_lab.plotting.style import (
    MAX_ASPECT_RATIO,
    METHOD_COLORS,
    MIN_ASPECT_RATIO,
    REFERENCE_COLOR,
    StyleContext,
    method_color,
    reference_line_kwargs,
)
from rc_basics_lab.types import BoolArray, FloatArray

PLOTTING_DIR = Path(style.__file__).parent

CITATION = re.compile(r"\[[^\]]*(?:\d{4}|source unidentified|原典未特定)[^\]]*\]")
"""凡例テキストに出典 (``[著者 年]`` / ``[原典未特定]``) が在るか。"""

ROOT = Path(__file__).resolve().parents[1]

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
    # ``plot_ipc_profile`` は行数上限 (D-77) で figures_ipc_profile へ移り、
    # 保存関数の名前も ``save_png`` になった。**両方を差し替える** ——
    # 片方だけだと、その図の検査が「図が保存されませんでした」で空振りする。
    monkeypatch.setattr(figures_ipc_profile, "save_png", spy)
    # ``plot_mc_sweep`` も figures_mc_sweep へ移った (D-77)。同じ理由で足す。
    monkeypatch.setattr(figures_mc_sweep, "_save", spy)
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


def _legend_texts(figure: Figure) -> list[str]:
    """図の凡例テキストを集める。

    **軸の凡例と図の凡例の両方**を見る。FIG-17 で系列の多い図の凡例を
    ``figure.legend`` へ移したので、軸だけを見ていると「凡例が無い」と誤判定する
    (実測: fig_threshold_tradeoff で参照線の検査が空振りした)。
    """
    texts: list[str] = []
    for axis in figure.axes:
        legend = axis.get_legend()
        if legend is not None:
            texts.extend(text.get_text() for text in legend.get_texts())
    for legend in figure.legends:
        texts.extend(text.get_text() for text in legend.get_texts())
    return texts


def _capture_saves(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, name: str
) -> list[Figure]:
    """``module.<name>`` を包んで、保存直前の ``Figure`` を集める。

    ``capture_figures`` は ``figures_capacity`` 専用なので、他のモジュールは
    ここを通す。``getattr`` で引くのは、保存関数の名前がモジュールごとに
    違う (``_save`` / ``save_png``) ためである。
    """
    captured: list[Figure] = []
    original = getattr(module, name)

    def spy(figure: Figure, path: Path) -> Path:
        captured.append(figure)
        result: Path = original(figure, path)
        return result

    monkeypatch.setattr(module, name, spy)
    return captured


def test_literature_reference_lines_carry_their_source_in_the_legend(
    tmp_path: Path, capture_figures: list[Figure]
) -> None:
    """3-A / 3-B' / 3-C の参照線の凡例に出典が入っている (FIG-3 / D-84)。

    本文の注は図が切り取られると消える。**凡例テキスト自体**に
    ``[Jaeger 2002 / Dambre 2012]`` / ``[原典未特定]`` を載せる。

    変異注入: ``cited_*(...)`` を外して素のラベルに戻すと、その図の参照線が
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
    assert cited_bound("上限 MC <= N", "Jaeger 2002") == "上限 MC <= N [Jaeger 2002]"
    assert SOURCE_UNIDENTIFIED[0] and SOURCE_UNIDENTIFIED[1]
    for text, source in (("上限 MC <= N", ""), ("", "Jaeger 2002")):
        with pytest.raises(ValueError, match="出典"):
            cited_bound(text, source)
        with pytest.raises(ValueError, match="出典"):
            cited_measurement(text, source, "N = 50 規模")


def test_a_literature_measurement_must_carry_its_operating_point() -> None:
    """文献の実測値は**動作点**なしに引けない (D-97)。

    実測値は測った条件でしか意味を持たない。条件を書かずに値だけ引くと、
    読者は「その数字と自分の数字が比較可能か」を判断できないまま、
    比較可能だと受け取る。実例 (D-95): 3-C は先行の「正則化なし」だけを
    再現して「先行の対照を足した」と書いたが、動作点は再現していなかった。

    **理論上限は別の入口** (``cited_bound``) にしてある —— 上限は動作点に
    依らないので、そちらに条件を求めると意味の無い文字列を書かせることになる。
    """
    assert (
        cited_measurement("k/n = 0.91", "Goudarzi et al. 2014", "1,810 タップ")
        == "k/n = 0.91 [Goudarzi et al. 2014; 1,810 タップ]"
    )
    with pytest.raises(ValueError, match="動作点"):
        cited_measurement("k/n = 0.91", "Goudarzi et al. 2014", "")

    # 上限は条件を求められない (求めると空文字を埋めるだけの作業になる)
    import inspect

    assert "conditions" not in inspect.signature(cited_bound).parameters


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
    """3-A に ``MC / N`` の第2軸が在り、上限が遠いときは縦軸が破断する (FIG-4 / 2-7)。

    上限 200 と実測 10〜36 は 5.5 倍離れていて、対数軸にしても実測の帯より
    広い空白が上に残る。上段に上限線、下段に実測を置いて破断し、隔たりは
    ``MC / N`` の第2軸で数値として読む。第2軸か破断のどちらかが消えたら
    ここが落ちる。
    """
    rows = mc_sweep_rows(leaks=(0.3,))
    n_units = float(rows[0].n_units)
    plot_mc_sweep(rows, mc_sweep_profile(rows), tmp_path / "mc.png", style=CONTEXT)
    figure = capture_figures[0]
    data_axis = next(axis for axis in figure.axes if "MC_total" in axis.get_ylabel())
    children = list(data_axis.child_axes)
    assert len(children) == 1, f"第2軸が {len(children)} 本です"
    # 目盛の範囲が主軸を N で割ったものになっている (ラベルではなく変換を見る)
    low, high = data_axis.get_ylim()
    assert children[0].get_ylim() == pytest.approx((low / n_units, high / n_units))
    # 上限線は実測パネルではなく上段に在る (= 破断している)
    assert not _reference_line_values(data_axis), "上限線が実測パネルに残っています"
    bound_axis = next(
        axis
        for axis in figure.axes
        if axis is not data_axis and _reference_line_values(axis)
    )
    assert _reference_line_values(bound_axis) == pytest.approx([n_units])
    # 実測パネルの上端は上限に届かない (破断の意味が消えていない)
    assert high < n_units
    # 「右軸が何か」はタイトルに書く (第2軸は constrained layout の管理外で、
    # 軸ラベルを付けると隣のパネルに重なるため)。破断時は上段が見出しになる。
    assert "MC / N" in bound_axis.get_title()


def _reference_line_values(axis: Axes) -> list[float]:
    """``axis`` に引かれた水平参照線の y 値。

    ``axhline`` は端から端までの ``Line2D`` になるので、y が定数の破線を拾う。
    データ系列 (誤差棒) は実線なので混ざらない。破断記号も線を描かない
    (``linestyle="none"``) ので除く —— 記号は軸座標の y=1 に置かれており、
    これを数えると「上限線が残っている」と誤判定する。
    """
    drawn = {"-", "none", "None"}
    values: list[float] = []
    for line in axis.get_lines():
        ydata = np.asarray(line.get_ydata(), dtype=np.float64)
        if ydata.size != 2 or ydata[0] != ydata[1]:
            continue
        if line.get_linestyle() not in drawn:
            values.append(float(ydata[0]))
    return values


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
    # **改行を無視して比べる**。長い注記は layout.wrapped_note が折り返して
    # 図に載せる (折り返さないと tight bbox が横に伸びて図そのものが歪む)。
    # 比べたいのは注記の**内容**であって行の切り方ではない。
    notes = [
        text.get_text()
        for text in figure.findobj(Text)
        if isinstance(text, Text)
        and text.get_text().replace("\n", "") == expected.replace("\n", "")
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


# --- FIG-2: 各実験に文献照合が1枚以上あるか (D-96) -----------------------------

#: 実験 -> その実験の図を描くモジュール (``src/rc_basics_lab/plotting/`` 相対)。
FIGURE_MODULES: dict[str, tuple[str, ...]] = {
    "01": ("figures.py", "figures_horizon.py"),
    "02": ("figures_esp.py", "esp_references.py"),
    "03": ("figures_capacity.py", "figures_narma_taps.py", "narma10_panel.py"),
    "04": ("figures_freerun.py", "freerun_headlines.py"),
    "05": ("figures_anomaly.py", "figures_anomaly_sweep.py"),
}

#: **文献照合図がまだ無い実験** (2026-08-21 の実測)。
#: FIG-2 は「各記事に文献照合を最低1枚」を求めているが、引くべき文献値と条件が
#: 特定できていない実験がある。**この集合は増やせない** —— 増えるということは
#: 新しい実験を文献照合なしで足したということで、それを黙って通さない。
#: 減らしたらここから外す (下の検査が外し忘れを拾う)。
KNOWN_WITHOUT_CITATION: frozenset[str] = frozenset()


def _cites_literature(module: str) -> bool:
    """そのモジュールが**文献の実測値**と照合しているか (D-97)。

    数えるのは ``cited_measurement`` だけで、``cited_bound`` は数えない。
    理論上限 (``MC <= N``) に出典を付けるのは正しい作法だが、それは
    **文献の数字とこちらの数字を並べたことにはならない** —— FIG-2 が
    求めているのは後者である。
    """
    path = ROOT / "src" / "rc_basics_lab" / "plotting" / module
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "cited_measurement"
        for node in ast.walk(tree)
    )


def _experiments_with_citation() -> set[str]:
    return {
        experiment
        for experiment, modules in FIGURE_MODULES.items()
        if any(_cites_literature(module) for module in modules)
    }


def test_the_figure_modules_table_matches_the_package() -> None:
    """対応表が実在のモジュールと一致する。

    表が古いと下の2つの検査が黙って対象を取りこぼす。**測れない状態を
    先に潰す** (D-93 と同じ規律)。
    """
    listed = {module for modules in FIGURE_MODULES.values() for module in modules}
    package = ROOT / "src" / "rc_basics_lab" / "plotting"
    missing = sorted(module for module in listed if not (package / module).is_file())
    assert not missing, f"対応表にあるが実在しないモジュール: {missing}"


def test_no_experiment_loses_its_literature_comparison() -> None:
    """文献照合を持つ実験が減っていないこと (FIG-2 / D-96)。

    実測 (2026-08-21): 03 だけが持つ (``figures_capacity`` の参照線3本と
    ``figures_narma_taps`` の先行の動作点)。残り4本はまだ持っていない ——
    引くべき文献値と条件 (N・入力次元・観測ノイズ) が未特定であり、
    **無いものを引くわけにはいかない**。

    そこで「持っていない実験の集合は増やせない」形にする。新しい実験を
    文献照合なしで足すとここが赤くなり、既存の穴は見えたまま残る。
    """
    without = set(FIGURE_MODULES) - _experiments_with_citation()
    new_gaps = sorted(without - KNOWN_WITHOUT_CITATION)
    assert not new_gaps, (
        f"文献照合図を持たない実験が増えました: {new_gaps}\n"
        "FIG-2 は各記事に文献照合を最低1枚求めています "
        "(docs/series/図の設計方針_RC基礎編.md)。\n"
        "**KNOWN_WITHOUT_CITATION に追記して通すのはラチェットを外す操作です。**"
    )


def test_the_known_gaps_list_has_no_stale_entries() -> None:
    """文献照合が付いた実験が既知の穴に残っていないこと。

    残したままだと、その実験から参照線を消しても検査が通ってしまう。
    """
    stale = sorted(KNOWN_WITHOUT_CITATION & _experiments_with_citation())
    assert not stale, (
        f"文献照合が付いたのに既知の穴に残っています: {stale}\n"
        "KNOWN_WITHOUT_CITATION から外してください (ラチェットが1段締まります)。"
    )


def test_every_narma10_reference_line_names_a_real_source() -> None:
    """3-C の参照値が**特定済みの出典**を名乗ること (D-100)。

    0.16 / 0.107 は長く「原典未特定」のまま引かれていた。実際に辿ると
    どちらも Vinckier et al. 2015 (Optica 2:438) に行き着く ——
    0.107 はその論文自身の実験値 (N = 50、訓練/テスト各 1000 ステップ、
    10 回反復で s.d. 0.012)、0.16 は同論文が Appeltant et al. 2011
    (Nat. Commun. 2:468) に帰す「線形シフトレジスタで得られる最良値」である。

    **`SOURCE_UNIDENTIFIED` へ戻す変更をここで落とす。** 出典が分かった値を
    「未特定」に戻すのは情報を捨てる操作であり、静かに起こり得る
    (参照値を1本足すときに、既存の書き方をコピーすると起こる)。
    """
    assert set(REFERENCE_SOURCES) == set(REFERENCE_LABELS), (
        "参照線のキーと出典のキーが一致していません: "
        f"{sorted(REFERENCE_SOURCES)} vs {sorted(REFERENCE_LABELS)}"
    )
    assert set(REFERENCE_CONDITIONS) == set(REFERENCE_LABELS), (
        "参照線のキーと動作点のキーが一致していません"
    )
    unidentified = sorted(
        key
        for key, source in REFERENCE_SOURCES.items()
        if source in SOURCE_UNIDENTIFIED or not source.strip()
    )
    assert not unidentified, (
        f"原典が特定済みの参照値が「未特定」に戻っています: {unidentified}\n"
        "0.16 -> Appeltant et al. 2011 / 0.107 -> Vinckier et al. 2015 (D-100)。"
    )
    # 出典が実際に凡例へ届いていること (定数だけ直して配線を忘れる形を落とす)
    assert REFERENCE_SOURCES["linear_ceiling"] == APPELTANT_2011
    assert REFERENCE_SOURCES["nonlinear_rc"] == VINCKIER_2015


def test_the_reference_note_records_the_conditions_of_each_source() -> None:
    """図の注が出典と**測定条件**の両方を持つこと (D-100)。"""
    for note in (NARMA10_REFERENCE_NOTE, NARMA10_REFERENCE_NOTE_EN):
        assert "Vinckier" in note and "Appeltant" in note, note
        assert "0.107" in note and "0.16" in note, note
        assert "50" in note and "1000" in note, note
    assert "未特定" not in NARMA10_REFERENCE_NOTE


def test_the_valid_time_figure_carries_a_literature_reference_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """4-B に文献の有効予測時間の参照線がある (FIG-2 / D-102)。

    参照線が無いままタイトルで水準を主張していたのが指摘 B-3 だった。
    一次資料 (Gauthier et al. 2021, Nat. Commun. 12:5564) の本文に
    「The NG-RC forecasts well out to ~5 Lyapunov times.」とあり、
    同じ段落が「100s to 1000s of reservoir nodes を持つ最適化された
    従来型 RC に匹敵する」と書いている。こちらの N = 200 はその範囲で、
    Lyapunov 時間も原典 1.1 / こちらの数値推定 1.09 とそろっている。

    **出典と動作点の両方が凡例に届いていること**を見る (D-97)。
    判定基準が原典と同じでないことも動作点に含める —— 原典は
    「forecasts well」と定性的に述べており、こちらの閾値 0.4 とは違う。
    """
    # capture_figures は figures_capacity._save だけを見ているので、
    # 04 の保存経路は自前で包む。
    captured = _capture_saves(monkeypatch, figures_freerun, "_save")
    plot_valid_time(freerun_rows(), tmp_path / "valid.png", style=CONTEXT)
    assert captured, "図が保存されませんでした"
    figure = captured[0]
    labels = _legend_texts(figure)
    cited = [label for label in labels if GAUTHIER_2021 in label]
    assert cited, f"文献の参照線が凡例にありません: {labels}"
    legend = cited[0]
    assert str(LITERATURE_VALID_TIME).rstrip("0").rstrip(".") in legend, legend
    # 動作点が付いている (出典だけでは比較可能かを読者が判断できない)
    assert ";" in legend, legend
    assert CONTEXT.label(*LITERATURE_VALID_TIME_CONDITIONS) in legend, legend


def test_the_threshold_tradeoff_figure_carries_a_literature_reference_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """05 の F1 パネルに文献の参照線がある (FIG-2 / D-103)。

    一次資料は Kim, Choi, Choi, Lee, Yoon (2022) *Towards a Rigorous
    Evaluation of Time-series Anomaly Detection*, AAAI-22。Table 2 の
    Case 1 (Random anomaly score) が、**素の F1** で SWaT 0.216 /
    WADI 0.109 / MSL 0.190 / SMAP 0.227 / SMD 0.080 を報告している。

    比べる相手を ``F1PA`` 列 (0.804〜0.969) にしない。この図の F1 は
    point-adjust を通していない (D-54 / D-55) ので、取り違えると
    「乱数と同程度」を「最先端と同程度」と読むことになる。
    """
    captured = _capture_saves(monkeypatch, figures_anomaly, "save_png")
    plot_threshold_tradeoff(
        anomaly_rows(), threshold_rows(), tmp_path / "t.png", style=CONTEXT
    )
    assert captured, "図が保存されませんでした"
    labels = _legend_texts(captured[0])
    cited = [label for label in labels if KIM_2022 in label]
    assert cited, f"文献の参照線が凡例にありません: {labels}"
    legend = cited[0]
    assert ";" in legend, legend
    assert CONTEXT.label(*RANDOM_SCORE_F1_CONDITIONS) in legend, legend
    # PA 側の値 (0.80 以上) と取り違えていない
    low, high = RANDOM_SCORE_PLAIN_F1
    assert high < 0.5, (low, high)


def test_the_horizon_figure_carries_a_literature_reference_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """01' に文献の NRMSE84 の参照線がある (FIG-2 / D-105)。

    一次資料は Jaeger & Haas (2004) *Harnessing Nonlinearity*,
    Science **304**:78-80。本文に

        For testing, a 84-step continuation d(3001),...,d(3084) ...
        The network output y(3084) was compared with the correct
        continuation d(3084).

    とあり、報告値は ``log10(NRMSE84) = -4.2``、従来手法はその 700 倍悪い。

    **予測長を合わせたから並べられる。** 01 本体は1ステップ先なので、
    そのままでは比較できない (D-105 がこの実験を足した理由)。
    """
    from matplotlib.figure import Figure

    from rc_basics_lab.plotting.figures_horizon import horizon_reference_note

    figure = Figure()
    draw_horizon_panel(figure.subplots(1, 1), _horizon_rows(), CONTEXT)
    labels = _legend_texts(figure)
    values = [label for label in labels if "-4.20" in label or "-1.35" in label]
    assert len(values) == 2, f"文献の参照線が2本ありません: {labels}"
    # **出典と条件は凡例ではなく注記が持つ** (FIG-14 で移した)。
    # 凡例に入れると幅が軸の4倍になり、隣のパネルを潰した (実測)。
    note = horizon_reference_note(CONTEXT)
    assert JAEGER_HAAS_2004 in note, note
    assert ";" in note, note
    assert CONTEXT.label(*JAEGER_CONDITIONS) in note, note
    # 従来手法の線は 700 倍悪い側 (取り違えると比較の向きが反転する)
    assert JAEGER_PREVIOUS_LOG10 > JAEGER_LOG10_NRMSE84
    assert pytest.approx(700.0) == 10.0 ** (
        JAEGER_PREVIOUS_LOG10 - JAEGER_LOG10_NRMSE84
    )


def _horizon_rows() -> tuple[HorizonRow, ...]:
    """図を描くのに足りる最小の行 (値そのものは検査しない)。"""
    return tuple(
        HorizonRow(
            task="mackey_glass",
            method="esn",
            replicate=index,
            n_units=200,
            horizon=HORIZON_STEPS,
            nrmse_horizon=0.006,
            log10_nrmse_horizon=-2.2 + 0.1 * index,
            nrmse_mean_to_horizon=0.004,
            diverged=False,
            wall_time_s=0.1,
        )
        for index in range(3)
    )


# --- FIG-14: 凡例が軸からはみ出さない (D-106) ---------------------------------

#: 凡例が**軸の外へ出ている**軸 (2026-08-22 の実測)。
#: この集合は増やせない。減らしたら外す。
#: 幅比が 1 を超えると、軸ラベルや隣のパネルを潰す (実測: fig_esp_map の
#: 無入力パネルで 4.19 倍になり、y 軸の目盛りが読めなくなっていた)。
LEGENDS_OUTSIDE_THE_AXES: frozenset[str] = frozenset()


def _legend_fits(axis: Axes) -> bool:
    """凡例が軸の内側に収まっているか (FIG-14)。"""
    legend = axis.get_legend()
    if legend is None:
        return True
    legend_box = legend.get_window_extent()
    axis_box = axis.get_window_extent()
    return bool(
        legend_box.x0 >= axis_box.x0 - 1
        and legend_box.x1 <= axis_box.x1 + 1
        and legend_box.y0 >= axis_box.y0 - 1
        and legend_box.y1 <= axis_box.y1 + 1
    )


def test_no_legend_spills_out_of_its_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """凡例が軸からはみ出していないこと (FIG-14 / D-106)。

    **重なりは目視でしか見つからないので機械に置く** (方針文書 FIG-14)。
    実際に起きた形: 出典つきのラベル (FIG-3 で長くなる) を幅の狭いパネルの
    凡例に置いたところ、**軸の 4.19 倍**の幅になり、はみ出して y 軸の
    目盛りを潰した。直し方は凡例をやめて出典を注へ移すこと。

    ここで描くのは、幅の狭いパネルを持つ 2-C (``plot_esp_map``) である ——
    そこが実際に壊れた図であり、同じ壊れ方をもう一度作らないための検査になる。
    """
    config = load_config_as(_esp_write(tmp_path, _ESP_TINY), Esp02Config)
    outputs = run_and_report_esp(config, tmp_path / "out")
    rows = [row for row in outputs.rows if row.experiment == EXPERIMENT_ESP_MAP]
    assert rows, "2-C の行が出ませんでした"
    captured = _capture_saves(monkeypatch, figures_esp, "_save")
    plot_esp_map(rows, tmp_path / "map.png", style=CONTEXT)
    assert captured, "図が保存されませんでした"
    figure = captured[0]
    figure.canvas.draw()
    spilled = [
        index for index, axis in enumerate(figure.axes) if not _legend_fits(axis)
    ]
    assert not spilled, (
        f"凡例が軸からはみ出しています (軸 {spilled})。\n"
        "出典つきのラベルは長くなるので、幅の狭いパネルでは"
        "**凡例をやめて注へ移してください** (FIG-14 / D-106)。"
    )


# --- FIG-13: アスペクト比を範囲に収める (D-108) --------------------------------

#: **範囲の外にある図** (2026-08-22 の実測)。この集合は増やせない。
#: 直したら外す (下の陳腐化検査が外し忘れを拾う)。
ASPECT_RATIO_EXCEPTIONS: frozenset[str] = frozenset()
"""アスペクト比の上限を外れてよい図。**空である。**

かつては3枚あった (``fig_stability_map`` 4.03 : 1 / ``fig_ipc_profile``
3.84 : 1 / ``fig_score_timeline`` 0.87 : 1)。FIG-13 で全部畳んだので、
**ここへ追記して通すのはラチェットを外す操作である**。図が上限を外れたら、
例外に足すのではなくパネルの並べ方を変える (2段に折る / 高さを取る)。
"""


def _png_aspect(path: Path) -> float:
    """PNG の横縦比 (ヘッダから読む。画像を展開しない)。"""
    header = path.read_bytes()[16:24]
    width, height = struct.unpack(">II", header)
    return float(width) / float(height)


def _artifact_pngs() -> list[Path]:
    return sorted((ROOT / "results").rglob("*.png"))


def test_the_aspect_ratio_exception_list_has_no_stale_entries() -> None:
    """例外リストが実在の図を指していること (FIG-13 / D-108)。

    実在しない図が残っていると、その分だけ黙って例外を増やせる。
    """
    names = {path.name for path in _artifact_pngs()}
    missing = sorted(ASPECT_RATIO_EXCEPTIONS - names)
    assert not missing, f"例外リストに実在しない図があります: {missing}"
    fixed = sorted(
        name
        for name in ASPECT_RATIO_EXCEPTIONS & names
        if MIN_ASPECT_RATIO
        <= _png_aspect(next(p for p in _artifact_pngs() if p.name == name))
        <= MAX_ASPECT_RATIO
    )
    assert not fixed, (
        f"範囲に収まったのに例外リストに残っています: {fixed}\n"
        "ASPECT_RATIO_EXCEPTIONS から外してください (ラチェットが1段締まります)。"
    )


def test_no_new_figure_falls_outside_the_aspect_ratio_range() -> None:
    """図の横縦比が ``MIN``〜``MAX`` に収まること (FIG-13 / D-108)。

    実測 (2026-08-22): 23 枚が **21 種類**の比にばらけ、0.87 : 1 から
    4.03 : 1 まであった。記事に順に並べたとき、図ごとに幅と高さが変わって
    視線のリズムが崩れる。

    **既存の範囲外3枚は凍結してある。** 直すにはパネル構成を変える必要が
    あり、それは FIG-12 の統合と同じ回にやるほうが手戻りが少ない。
    ここが止めるのは**新しく範囲外の図を足すこと**である。
    """
    offenders = {
        path.name: round(_png_aspect(path), 2)
        for path in _artifact_pngs()
        if path.name not in ASPECT_RATIO_EXCEPTIONS
        and not (MIN_ASPECT_RATIO <= _png_aspect(path) <= MAX_ASPECT_RATIO)
    }
    assert not offenders, (
        f"横縦比が範囲外の図があります: {offenders}\n"
        f"許容は {MIN_ASPECT_RATIO} 〜 {MAX_ASPECT_RATIO} : 1 です。"
        "**超えるならパネルを2段に折り、縦長なら横に折ってください** "
        "(FIG-13 / D-108)。\n"
        "**ASPECT_RATIO_EXCEPTIONS に追記して通すのはラチェットを外す操作です。**"
    )


# --- FIG-11: 波形の選び方は自由変数にしない (D-107) ---------------------------


def test_the_waveform_selection_is_not_a_free_parameter() -> None:
    """波形の切り出しが**呼び出し側から選べない**こと (D-107)。

    「よく当たっている区間」を選べる図にすると、同じデータから好きな結論の
    図が作れる —— 仕様 §5 の禁止する構造そのものである。だから区間・長さ・
    レプリケートはモジュール定数であり、引数として受け取らない。

    **引数に足す変更をここで落とす。** 定数は残したまま引数を優先させる、
    という形で決定が骨抜きになるのが一番起きやすい壊れ方なので、
    シグネチャを直接見る。
    """
    selection = dict(waveforms.selection_is_fixed())
    assert selection == {
        "offset": 0,
        "steps": 300,
        "replicate": 0,
        # 課題ごとの長さも定数表である (D-107)。反転や高周波の課題は
        # 300 ステップでは塗り潰れ、図の主張が読めなかった (C-4)。
        "delay_parity": 40,
        "narma10": 100,
    }, selection

    for name in ("offset", "start", "window", "replicate", "steps"):
        parameters = inspect.signature(waveforms.plot_prediction_waveform).parameters
        assert name not in parameters, (
            f"波形の選び方が引数になっています: {name} (D-107)。\n"
            "定数のままにしてください —— 引数にすると"
            "「この図だけ別の区間」が書けてしまい、決定が実質無くなります。"
        )

    # 切り出しは常に同じ長さ・同じ位置 (start だけが外から来る)
    series = np.arange(1000, dtype=np.float64)
    first = waveforms.slice_window(series, 100)
    again = waveforms.slice_window(series, 100)
    assert np.array_equal(first, again)
    assert first[0] == 100 + waveforms.WAVEFORM_OFFSET
    assert first.size == waveforms.WAVEFORM_STEPS


def _comparison_rows() -> tuple[ResultRow, ...]:
    """``plot_comparison`` に足りる最小の行 (2 課題 x 3 手法)。"""
    return tuple(
        ResultRow(
            task=task,
            method=method,
            replicate=replicate,
            seed_reservoir=0,
            seed_task=1,
            seed_split=2,
            alpha=1e-4,
            n_lags=0,
            rmse=0.1,
            nrmse=0.1,
            nmse=0.01,
            sign_accuracy=0.5,
            n_train=100,
            n_val=20,
            n_test=50,
            t0=1,
            wall_time_s=0.1,
        )
        for task in ("mackey_glass", "delay_parity")
        for method in ("linear", "delay_line", "esn")
        for replicate in range(2)
    )


def _comparison_waveforms() -> tuple[WaveformPanel, ...]:
    """波形パネル2枚 (01 は課題を2つ扱う) に足りる最小の入力。"""
    truth: FloatArray = np.linspace(0.0, 1.0, 16)
    return tuple(
        WaveformPanel(task=task, truth=truth, predictions={"esn": truth * 0.99})
        for task in ("mackey_glass", "delay_parity")
    )


def _profile_rows_for_two_tasks() -> tuple[FreeRunProfileRow, ...]:
    """位相図パネル2枚に足りる最小の行。"""
    return tuple(
        FreeRunProfileRow(
            experiment="4B_freerun",
            task=task,
            method="esn",
            replicate=0,
            source=source,
            kind="phase",
            index=index,
            x=float(index),
            y=float(index) * 0.5,
        )
        for task in ("lorenz", "mackey_glass")
        for source in ("truth", "freerun")
        for index in range(4)
    )


def _onestep_rows_for_two_tasks() -> tuple[ResultRow, ...]:
    """4-A パネルに足りる最小の行。"""
    return tuple(
        ResultRow(
            task=task,
            method=method,
            replicate=replicate,
            seed_reservoir=0,
            seed_task=1,
            seed_split=2,
            alpha=1e-4,
            n_lags=0,
            rmse=0.1,
            nrmse=0.1,
            nmse=0.01,
            sign_accuracy=0.5,
            n_train=100,
            n_val=20,
            n_test=50,
            t0=1,
            wall_time_s=0.1,
        )
        for task in ("lorenz", "mackey_glass")
        for method in ("linear", "delay_line", "esn")
        for replicate in range(2)
    )


# --- FIG-12: 点の少ないスカラー比較を単独図にしない (D-109) --------------------


def test_sparse_scalar_figures_are_panels_not_standalone_figures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """01 / 04 の主図が、スカラー比較を**パネルとして**抱えていること (D-109)。

    かつては ``fig_onestep`` (6 点) と ``fig_horizon`` (5 点) が単独の figure
    だった。**点の数を数えるのではなく、軸の数を数える** —— 単独図に戻せば
    軸が減るので、その形で退行が捕まる。

    ``fig_waveform`` も同じ理由で ``fig_comparison`` のパネルになっている。
    """
    from rc_basics_lab.plotting import figures, figures_freerun

    captured = _capture_saves(monkeypatch, figures, "_save")
    figures.plot_comparison(
        _comparison_rows(),
        tmp_path / "fig_comparison.png",
        waveforms=_comparison_waveforms(),
        horizon_rows=_horizon_rows(),
        style=CONTEXT,
    )
    assert captured, "01 の主図が保存されませんでした"
    # 波形パネルは**上段と残差の段**の2軸になる (C-3) ので、
    # 4 パネル = 1 + 2x2 + 1 = 6 軸である。
    assert len(captured[0].axes) == 6, (
        "01 の fig_comparison は 課題別の誤差 / 波形2枚 (各2段) / 自走 の"
        f"6軸であるべきです (実測 {len(captured[0].axes)})。"
        "単独の figure に戻すのは D-109 の取り消しです。"
    )

    captured_freerun = _capture_saves(monkeypatch, figures_freerun, "_save")
    figures_freerun.plot_freerun_attractor(
        _profile_rows_for_two_tasks(),
        tmp_path / "fig_freerun_attractor.png",
        onestep_rows=_onestep_rows_for_two_tasks(),
        style=CONTEXT,
    )
    assert captured_freerun, "04 の主図が保存されませんでした"
    assert len(captured_freerun[0].axes) == 3, (
        "04 の fig_freerun_attractor は 位相図2枚 + 4-A のスカラー比較の"
        f"3パネルであるべきです (実測 {len(captured_freerun[0].axes)})。"
    )


def test_the_figures_that_fig12_folded_away_are_gone() -> None:
    """畳んだ3枚が成果物として復活していないこと (D-109)。

    **図を足すのは自由だが、この3つの名前で単独図に戻すのは統合の取り消し**
    なので、名前で止める。
    """
    folded = ("fig_onestep.png", "fig_horizon.png", "fig_waveform.png")
    present = [name for name in folded if list((ROOT / "results").rglob(name))]
    assert not present, (
        f"FIG-12 で畳んだ図が単独で復活しています: {present}。"
        "パネルとして親figureへ入れてください (D-109)。"
    )


def test_the_state_waveform_units_are_not_selectable() -> None:
    """状態波形のユニットが**選べない**こと (FIG-11 追加図5 / D-107)。

    「よく散っているユニット」を選べる図にすると、同じリザバーから
    好きな結論の図が作れる。だから先頭から番号順に取る。
    """
    from rc_basics_lab.experiment import state_waveform as module

    assert module.STATE_WAVEFORM_UNITS == 8, module.STATE_WAVEFORM_UNITS
    parameters = inspect.signature(module.state_waveform).parameters
    for name in ("units", "unit_indices", "n_units", "offset", "replicate"):
        assert name not in parameters, (
            f"ユニットの選び方が引数になっています: {name} (D-107)。\n"
            "定数のままにしてください —— 引数にすると"
            "「この図だけ別のユニット」が書けてしまいます。"
        )


# --- FIG-14: 出典入りの凡例が隣のパネルを潰さない (D-110) ----------------------


def test_no_legend_spills_out_of_its_axes_in_the_main_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """01 の主図でも凡例が軸に収まっていること (FIG-14 / D-110)。

    D-106 の検査は 2-C しか描いていなかった。**壊れ方は同じでも、
    描いていない図は守られない** —— 実際 FIG-12 でパネルを詰めたとき、
    自走パネルの出典つき凡例が軸をはみ出して隣のパネルにかぶった。
    パネルを増やすと1枚あたりの幅が縮むので、この図が一番先に壊れる。

    直し方は凡例をやめて注へ移すこと (``horizon_reference_note``)。
    出典を消すのではなく置き場所を変えるだけなので、D-97 は満たせる。
    """
    from rc_basics_lab.plotting import figures

    captured = _capture_saves(monkeypatch, figures, "_save")
    figures.plot_comparison(
        _comparison_rows(),
        tmp_path / "fig_comparison.png",
        waveforms=_comparison_waveforms(),
        horizon_rows=_horizon_rows(),
        style=CONTEXT,
    )
    assert captured, "図が保存されませんでした"
    figure = captured[0]
    figure.canvas.draw()
    spilled = [
        index for index, axis in enumerate(figure.axes) if not _legend_fits(axis)
    ]
    assert not spilled, (
        f"凡例が軸からはみ出しています (軸 {spilled})。\n"
        "出典つきのラベルは長くなるので、パネルが増えて幅が縮んだ図では"
        "**凡例をやめて注へ移してください** (FIG-14 / D-110)。"
    )


def test_legends_are_opaque_so_lines_do_not_show_through() -> None:
    """凡例の背景が不透明であること (FIG-14 / D-110)。

    掃引図では線が軸全体に渡るので、凡例をどこへ置いても下を線が通る
    (実測: fig_esp_decay で 6 本)。**位置では解決できないので、
    読めることのほうを保証する**。
    """
    params = style.rc_params_for(CONTEXT)
    assert params["legend.frameon"] is True, params.get("legend.frameon")
    assert params["legend.framealpha"] == 1.0, params.get("legend.framealpha")


# --- FIG-1: タイトルは結論文 / FIG-5: 手法名は対応表を通す (D-111) --------------

QUESTION_ENDINGS: tuple[str, ...] = (
    "か",
    "か。",
    "?",
    "\uff1f",  # 全角の疑問符。字面が半角と紛らわしいのでコードポイントで書く
    "か?",
    "か\uff1f",
)
"""疑問形と判定する語尾 (FIG-1)。

**「結論文か」は機械で判定できないので、疑問形の禁止という下限だけを置く。**
粗いが、実際に破れた 2 件 (「どう外れていくか」「予測がどう見えるか」) は
これで落ちる。図が何を示したかを言えないなら、その図はまだ主張していない。
"""


def _titles(figure: Figure) -> list[str]:
    """figure と各軸の見出しを集める。

    Args:
        figure: 描画済みの figure。

    ``figure.texts`` を見るのは、``suptitle`` が private 属性でしか
    取れないためである (注 ``supxlabel`` も同じ列に入るが、
    そちらは疑問形にならないので判定に影響しない)。

    Returns:
        空でない見出しの並び。
    """
    found = [text.get_text() for text in figure.texts]
    found.extend(axis.get_title() for axis in figure.axes)
    return [text.strip() for text in found if text.strip()]


def _is_question(title: str) -> bool:
    """見出しが疑問形で終わるか。

    Args:
        title: 見出し。

    Returns:
        疑問形なら ``True``。
    """
    stripped = title.strip().rstrip("。 ")
    return stripped.endswith(QUESTION_ENDINGS)


def test_no_figure_title_is_a_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """図と各パネルの見出しが疑問形で終わらないこと (FIG-1 / D-111)。

    **FIG-1 だけ機械検査が無く、そこが新規図で破れた。** 既存 19 枚は
    結論文だったのに、検査の無い 1 本だけが崩れている。文書だけの規約は
    次に破れる、というこのリポジトリの繰り返しの確認の、図の側での再現。
    """
    from rc_basics_lab.plotting import figures, waveforms

    captured = _capture_saves(monkeypatch, figures, "_save")
    figures.plot_comparison(
        _comparison_rows(),
        tmp_path / "fig_comparison.png",
        waveforms=_comparison_waveforms(),
        horizon_rows=_horizon_rows(),
        style=CONTEXT,
    )
    captured += _capture_saves(monkeypatch, waveforms, "save_png")
    waveforms.plot_prediction_waveform(
        _comparison_waveforms()[0].truth,
        _comparison_waveforms()[0].predictions,
        tmp_path / "fig_waveform.png",
        task_label=("NARMA10", "NARMA10"),
        style=CONTEXT,
    )
    assert captured, "図が保存されませんでした"
    questions = [
        title for figure in captured for title in _titles(figure) if _is_question(title)
    ]
    assert not questions, (
        f"疑問形の見出しがあります: {questions} (FIG-1 / D-111)。\n"
        "**図が何を示したかを書いてください。** 「どう見えるか」は問いであって"
        "結論ではありません。結論文は行から導くこと (D-90 と同じ規律)。"
    )


def test_no_legend_shows_a_raw_method_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """凡例に手法の生のキーが出ていないこと (FIG-5 / D-111)。

    ``label=name`` で生のキーを渡すと、同じ図の x 軸ラベルが「線形」、
    凡例が ``linear`` になり、**同じものが2通りに呼ばれる**。
    ``style.label`` を通らないので、CJK フォントが無い環境の英語
    フォールバックも効かない (豆腐の図が「正常に」生成される)。
    """
    from rc_basics_lab.plotting import figures

    captured = _capture_saves(monkeypatch, figures, "_save")
    figures.plot_comparison(
        _comparison_rows(),
        tmp_path / "fig_comparison.png",
        waveforms=_comparison_waveforms(),
        horizon_rows=_horizon_rows(),
        style=CONTEXT,
    )
    assert captured, "図が保存されませんでした"
    raw = [
        text
        for figure in captured
        for axis in figure.axes
        for text in _legend_texts(figure)
        if text in METHOD_LABELS
    ]
    assert not raw, (
        f"凡例に生のキーが出ています: {sorted(set(raw))} (FIG-5 / D-111)。\n"
        "``label_of(METHOD_LABELS, name, style)`` を通してください。"
    )


# --- FIG-12: 1記事あたりの図の枚数 (D-112) ------------------------------------

MAX_FIGURES_PER_ARTICLE = 4
"""``連載構成案_RC基礎編.md`` の想定 (2〜4 枚) の上限。"""

#: 上限を超えている記事の**現在値**。**この辞書は増やせない** (2026-08-23 実測)。
#: 減らすのは自由で、減らしたら値を下げる (下の陳腐化検査が下げ忘れを拾う)。
FIGURE_COUNT_FROZEN: dict[str, int] = {
    "03_capacity": 5,
    "04_chaotic_freerun": 5,
    "05_anomaly_detection": 5,
}
"""上限超過を凍結した記事。

**ここへ追記して通すのはラチェットを外す操作である。** 図が増えたら、
凍結値を上げるのではなくパネルへ畳む (03 は NARMA10 の 3 枚を 1 枚に
畳んで 7 -> 5 枚にした)。
"""


def _figures_by_article() -> dict[str, int]:
    """記事ごとの PNG 枚数。

    Returns:
        ``{記事ディレクトリ名: 枚数}``。01 は ``results/`` 直下にあるので
        ``"01"`` というキーで数える。
    """
    root = ROOT / "results"
    counts = {"01": len(list(root.glob("*.png")))}
    for directory in sorted(root.iterdir()):
        if directory.is_dir():
            counts[directory.name] = len(list(directory.glob("*.png")))
    return {name: count for name, count in counts.items() if count}


def test_no_article_exceeds_its_figure_budget() -> None:
    """1記事あたりの図が上限か凍結値を超えないこと (FIG-12 / D-112)。

    **「足すだけでなく減らす」が FIG-12 の眼目だった**が、第3版の作業では
    正味で 23 -> 24 枚と増えていた。枚数は目視では追えないので機械に置く。
    """
    over = {
        name: count
        for name, count in _figures_by_article().items()
        if count > FIGURE_COUNT_FROZEN.get(name, MAX_FIGURES_PER_ARTICLE)
    }
    assert not over, (
        f"記事あたりの図が上限を超えています: {over} (FIG-12 / D-112)。\n"
        f"上限は {MAX_FIGURES_PER_ARTICLE} 枚 (凍結済みの記事は"
        f" FIGURE_COUNT_FROZEN の値)。\n"
        "**凍結値を上げて通すのはラチェットを外す操作です。** "
        "同じ主張を支えている図はパネルへ畳んでください。"
    )


def test_the_frozen_figure_counts_have_no_stale_entries() -> None:
    """減ったのに凍結値が古いままになっていないこと (D-112)。

    減らしたのに値が古いと、その差分だけ**また増やせてしまう**。
    """
    counts = _figures_by_article()
    slack = {
        name: (frozen, counts[name])
        for name, frozen in FIGURE_COUNT_FROZEN.items()
        if name in counts and counts[name] < frozen
    }
    assert not slack, (
        f"凍結値より少なくなった記事があります (凍結値, 実測): {slack}。\n"
        "FIGURE_COUNT_FROZEN を実測値まで下げてください"
        f" (上限 {MAX_FIGURES_PER_ARTICLE} 枚に収まったなら項目ごと消す)。"
    )

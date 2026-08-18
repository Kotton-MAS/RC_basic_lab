"""記事03の図4枚のテスト (仕様 §4 T3).

図は「生成できた」だけでは検証にならない。この4枚が主張しているのは
受け入れ条件1・2・4 そのものなので、

- **上限線 y=N が実際に描かれている** (``fig_ipc_conservation``、受け入れ条件2)
- ラベルが ``StyleContext`` を通っている (CJK フォントが無い環境で英語に
  落ち、ある環境では日本語になる。D-10)
- 縮小データ (2条件) と**縮退ケース** (条件が1つだけ / 全容量が 0) でも
  例外なく描ける

の3点を固定する。1つ目は ``figures_capacity._save`` を差し替えて ``Figure`` を
捕まえ、``conservation_bound`` が返す座標と一致する線が軸に在ることを見る
(PNG の画素を見ると閾値の選び方でいくらでも偽の緑になる)。

診断はここでは一切走らせない。図は ``capacity.csv`` / ``capacity_profile.csv``
と同じ行 (``CapacityRow`` / ``CapacityProfileRow``) だけを読む。
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
from conftest import png_dpi
from matplotlib.figure import Figure
from matplotlib.text import Text

from rc_basics_lab.experiment.capacity import (
    DIAGNOSTIC_IPC,
    DIAGNOSTIC_MC,
    EXPERIMENT_CONSERVATION,
    EXPERIMENT_IPC_SWEEP,
    EXPERIMENT_MC_SWEEP,
    CapacityProfileRow,
    CapacityRow,
)
from rc_basics_lab.plotting import figures_capacity, style
from rc_basics_lab.plotting.figures_capacity import (
    conservation_bound,
    ipc_heatmap_means,
    mc_profile_means,
    plot_ipc_conservation,
    plot_ipc_profile,
    plot_mc_sweep,
    plot_memory_nonlinearity,
)
from rc_basics_lab.plotting.style import StyleContext, setup_style

RETINA_DPI = 200
"""仕様 §4 T3: 図は 200 dpi。"""

N_DELAYS = 6
"""縮小データの MC 遅延数 (``mc.max_delay`` に相当)。"""

N_DEGREES = 2
"""縮小データの IPC 次数の本数。"""

_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3040, 0x30FF),  # ひらがな / カタカナ
    (0x3400, 0x4DBF),  # CJK 拡張A
    (0x4E00, 0x9FFF),  # CJK 統合漢字
    (0xFF00, 0xFFEF),  # 全角形
)
"""日本語ラベルの検出に使う符号位置の範囲 (D-10 の「豆腐文字」判定)。"""


def _row(
    *,
    experiment: str,
    rho: float,
    leak_rate: float,
    replicate: int,
    n_units: int = 20,
    state_noise: float = 0.0,
    mc_total: float = 3.0,
    mc_effective_delay: float = 2.0,
    ipc_linear: float = 2.0,
    ipc_nonlinear: float = 1.0,
) -> CapacityRow:
    """図が読む列だけを意味のある値にした1行 (残りは固定)。"""
    return CapacityRow(
        experiment=experiment,
        replicate=replicate,
        seed_reservoir=0,
        seed_drive=1,
        seed_surrogate=4,
        rho=rho,
        leak_rate=leak_rate,
        input_scale=1.0,
        sigma_u=0.2,
        input_drive_std=0.2,
        n_units=n_units,
        density=0.1,
        state_noise=state_noise,
        n_steps=1000,
        washout=40,
        t0_mc=40,
        n_samples_mc=900,
        mc_total=mc_total,
        mc_total_raw=mc_total + 0.5,
        mc_threshold=0.01,
        mc_effective_delay=mc_effective_delay,
        mc_ratio=mc_total / n_units,
        n_delays=N_DELAYS,
        t0_ipc=40,
        n_samples_ipc=900,
        ipc_total=ipc_linear + ipc_nonlinear,
        ipc_total_raw=ipc_linear + ipc_nonlinear + 0.5,
        ipc_linear=ipc_linear,
        ipc_nonlinear=ipc_nonlinear,
        ipc_saturation_ratio=0.5,
        n_targets=10,
        n_targets_kept=4,
        n_degrees=N_DEGREES,
        chunk_size_mc_effective=256,
        chunk_size_ipc_effective=256,
        wall_time_state_s=0.1,
        wall_time_mc_s=0.1,
        wall_time_ipc_s=0.1,
        wall_time_s=0.3,
    )


def _profile(
    row: CapacityRow, *, diagnostic: str, degree: int, delay: int, capacity: float
) -> CapacityProfileRow:
    """``row`` と同じ条件を指す長形式の1行 (正値セルのみが書かれる、D-38)。"""
    return CapacityProfileRow(
        experiment=row.experiment,
        replicate=row.replicate,
        rho=row.rho,
        leak_rate=row.leak_rate,
        n_units=row.n_units,
        state_noise=row.state_noise,
        diagnostic=diagnostic,
        degree=degree,
        delay=delay,
        capacity=capacity,
        threshold=0.01,
    )


def mc_sweep_rows(
    rhos: Sequence[float] = (0.5, 0.9),
    leaks: Sequence[float] = (0.3, 1.0),
    replicates: Sequence[int] = (0, 1),
) -> tuple[CapacityRow, ...]:
    """3-A 相当の行 (既定は 2 rho x 2 leak x 2 レプリケート)。"""
    return tuple(
        _row(
            experiment=EXPERIMENT_MC_SWEEP,
            rho=rho,
            leak_rate=leak,
            replicate=replicate,
            n_units=20,
            mc_total=2.0 + rho + leak + 0.1 * replicate,
            mc_effective_delay=1.0 + 2.0 * rho,
        )
        for replicate in replicates
        for rho in rhos
        for leak in leaks
    )


def mc_sweep_profile(rows: Sequence[CapacityRow]) -> tuple[CapacityProfileRow, ...]:
    """遅延 1..3 だけが正の MC プロファイル (残りのセルは長形式に**存在しない**)。"""
    return tuple(
        _profile(
            row, diagnostic=DIAGNOSTIC_MC, degree=1, delay=delay, capacity=1.0 / delay
        )
        for row in rows
        for delay in (1, 2, 3)
    )


def ipc_sweep_rows(
    rhos: Sequence[float] = (0.5, 1.1),
    leaks: Sequence[float] = (0.3, 1.0),
    replicates: Sequence[int] = (0, 1),
) -> tuple[CapacityRow, ...]:
    """3-B 相当の行 (rho が上がると非線形の取り分が減る)。"""
    return tuple(
        _row(
            experiment=EXPERIMENT_IPC_SWEEP,
            rho=rho,
            leak_rate=leak,
            replicate=replicate,
            n_units=10,
            ipc_linear=1.0 + rho,
            ipc_nonlinear=max(3.0 - 2.0 * rho, 0.0) + 0.1 * replicate,
        )
        for replicate in replicates
        for rho in rhos
        for leak in leaks
    )


def ipc_sweep_profile(rows: Sequence[CapacityRow]) -> tuple[CapacityProfileRow, ...]:
    """次数1が遅延 1..3、次数2が遅延1 だけ正のヒートマップ。"""
    return tuple(
        _profile(
            row,
            diagnostic=DIAGNOSTIC_IPC,
            degree=degree,
            delay=delay,
            capacity=1.0 / (degree * delay),
        )
        for row in rows
        for degree, delay in ((1, 1), (1, 2), (1, 3), (2, 1))
    )


def conservation_rows(
    units: Sequence[int] = (10, 20),
    noises: Sequence[float] = (0.0, 0.05),
    replicates: Sequence[int] = (0, 1),
) -> tuple[CapacityRow, ...]:
    """3-B' 相当の行 (ノイズを入れると ``ipc_total`` が N から離れる)。"""
    return tuple(
        _row(
            experiment=EXPERIMENT_CONSERVATION,
            rho=0.95,
            leak_rate=1.0,
            replicate=replicate,
            n_units=n_units,
            state_noise=noise,
            ipc_linear=n_units * 0.4 * (1.0 - 5.0 * noise),
            ipc_nonlinear=n_units * 0.3 * (1.0 - 5.0 * noise) + 0.1 * replicate,
        )
        for replicate in replicates
        for n_units in units
        for noise in noises
    )


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[Figure]:
    """``_save`` を包んで描き上がった ``Figure`` を捕まえる。

    図の主張 (上限線が在る / ラベルが英語に落ちた) は PNG の画素からは測れない。
    保存直前の ``Figure`` を掴んで artist を直接見る。
    """
    figures: list[Figure] = []
    original = figures_capacity._save

    def spy(figure: Figure, path: Path) -> Path:
        figures.append(figure)
        return original(figure, path)

    monkeypatch.setattr(figures_capacity, "_save", spy)
    return figures


def _draw_all(tmp_path: Path, context: StyleContext) -> tuple[Path, ...]:
    """縮小データ (2条件相当) で図4枚を描く。"""
    mc_rows = mc_sweep_rows()
    ipc_rows = ipc_sweep_rows()
    return (
        plot_mc_sweep(
            mc_rows, mc_sweep_profile(mc_rows), tmp_path / "mc.png", style=context
        ),
        plot_ipc_profile(
            ipc_rows, ipc_sweep_profile(ipc_rows), tmp_path / "ipc.png", style=context
        ),
        plot_memory_nonlinearity(ipc_rows, tmp_path / "split.png", style=context),
        plot_ipc_conservation(
            conservation_rows(), tmp_path / "bound.png", style=context
        ),
    )


def _texts(figure: Figure) -> list[str]:
    """図に載っている文字列 (タイトル・軸ラベル・凡例・目盛) を集める。"""
    return [
        text.get_text()
        for text in figure.findobj(Text)
        if isinstance(text, Text) and text.get_text()
    ]


def _has_cjk(value: str) -> bool:
    return any(
        low <= ord(char) <= high for char in value for low, high in _CJK_RANGES
    )


def _no_fonts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(style, "_available_font_names", frozenset)


def _with_cjk_font(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        style, "_available_font_names", lambda: frozenset({"IPAexGothic"})
    )


# --- 受け入れ条件2: 上限線が図に在る -----------------------------------------


def test_conservation_figure_draws_the_bound_line(
    tmp_path: Path, captured: list[Figure]
) -> None:
    """``fig_ipc_conservation`` に y=N の参照線 (傾き1の対角線) が実在する。

    この対角線が図の主張そのもの (受け入れ条件2: ``IPC_total <= N``) なので、
    線の有無をテストで固定する。**変異試験**: ``_draw_conservation_bound`` の
    ``axis.plot`` を消すと、一致する線が0本になってここが落ちる (他のテストは
    緑のまま通る)。座標は描画側と同じ ``conservation_bound`` から取り、
    「線は在るが別の場所に引かれている」も同時に排除する。
    """
    rows = conservation_rows()
    units = sorted({row.n_units for row in rows})
    path = plot_ipc_conservation(rows, tmp_path / "bound.png", style=setup_style())
    assert path.is_file()
    assert len(captured) == 1

    expected_x, expected_y = conservation_bound(units)
    axis = captured[0].axes[0]
    matched = [
        line
        for line in axis.get_lines()
        if np.array_equal(np.asarray(line.get_xdata(), dtype=np.float64), expected_x)
        and np.array_equal(np.asarray(line.get_ydata(), dtype=np.float64), expected_y)
    ]
    assert len(matched) == 1, (
        "y=N の参照線が見つかりません: "
        f"{[list(line.get_xdata()) for line in axis.get_lines()]}"
    )
    # 傾き1 (対角線) であること。水平線に退化していたら受け入れ条件2 を示せない。
    assert expected_y[1] - expected_y[0] == pytest.approx(expected_x[1] - expected_x[0])
    assert expected_x[1] > expected_x[0] > 0.0


def test_conservation_bound_spans_a_single_size_grid() -> None:
    """``n_units`` が1点しかなくても対角線が長さを持つ (縮小設定の縮退ケース)。

    1点だけを結ぶと長さ 0 の線分になり、上限線が図から消える。
    """
    x, y = conservation_bound([9])
    assert x[0] < 9.0 < x[1]
    assert np.array_equal(x, y)
    with pytest.raises(ValueError, match="units"):
        conservation_bound([])


# --- D-10: ラベルが StyleContext を通っている --------------------------------


def test_figures_use_the_style_context_labels(
    tmp_path: Path, captured: list[Figure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """CJK フォントが無い環境ではラベルが英語に落ちる (D-10 の既存機構)。

    フォント設定だけを戻す実装だと、CI 上で豆腐文字 (□□□) の図が「正常に」
    生成される。図の中の文字列を実際に走査し、

    - フォントが無い環境: 日本語の符号位置が**1文字も無い**
    - フォントが在る環境: 日本語が**在る**

    の両方を測る。片方だけだと「全部英語で直書き」した実装が通ってしまう。
    """
    _no_fonts(monkeypatch)
    context = setup_style()
    assert context.cjk_available is False
    _draw_all(tmp_path / "en", context)
    assert len(captured) == 4
    for figure in captured:
        texts = _texts(figure)
        assert texts
        assert [value for value in texts if _has_cjk(value)] == []

    captured.clear()
    _with_cjk_font(monkeypatch)
    japanese = setup_style()
    assert japanese.cjk_available is True
    with warnings.catch_warnings():
        # 差し替えたフォント名は実在しないので、日本語の字形が見つからない
        # という警告が savefig のたびに出る。ここで測るのは**ラベル文字列**
        # であって字形ではない (字形が在る環境で描くのは記事用の生成時)。
        warnings.simplefilter("ignore", UserWarning)
        _draw_all(tmp_path / "ja", japanese)
    assert len(captured) == 4
    for figure in captured:
        assert any(_has_cjk(value) for value in _texts(figure))


# --- 縮小データ・縮退ケースで描ける -------------------------------------------


def test_all_four_figures_are_written_at_retina_resolution(tmp_path: Path) -> None:
    """縮小データ (2条件) で図4枚が 200 dpi で書き出される。"""
    paths = _draw_all(tmp_path, setup_style())
    assert len(paths) == 4
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 0
        assert png_dpi(path) >= RETINA_DPI


def test_figures_render_with_a_single_condition(tmp_path: Path) -> None:
    """条件が1つしかない縮退ケースでも例外なく描ける。

    レプリケートが1本だと標準偏差の母数が1になり、格子が1点だとヒートマップの
    パネルが1枚・上限線の両端が同じ値になる。どれも縮小設定 (と、掃引を
    1点に絞った手元の確認) で普通に起きる。
    """
    mc_rows = mc_sweep_rows(rhos=(0.9,), leaks=(1.0,), replicates=(0,))
    ipc_rows = ipc_sweep_rows(rhos=(0.9,), leaks=(1.0,), replicates=(0,))
    conservation = conservation_rows(units=(9,), noises=(0.0,), replicates=(0,))
    context = setup_style()
    paths = (
        plot_mc_sweep(
            mc_rows, mc_sweep_profile(mc_rows), tmp_path / "mc.png", style=context
        ),
        plot_ipc_profile(
            ipc_rows, ipc_sweep_profile(ipc_rows), tmp_path / "ipc.png", style=context
        ),
        plot_memory_nonlinearity(ipc_rows, tmp_path / "split.png", style=context),
        plot_ipc_conservation(conservation, tmp_path / "bound.png", style=context),
    )
    for path in paths:
        assert png_dpi(path) >= RETINA_DPI


def test_figures_render_when_every_capacity_is_zero(tmp_path: Path) -> None:
    """全容量が 0 (長形式が1行も無い) 縮退ケースでも例外なく描ける。

    しきい値を超えたセルが1つも無ければ ``capacity_profile.csv`` は空になる
    (D-38 は正値セルだけを書く)。強すぎる駆動や短すぎる系列で普通に起こるので、
    そこで再生成コマンドが死ぬと図の更新ができなくなる。
    """
    mc_rows = tuple(
        _row(
            experiment=EXPERIMENT_MC_SWEEP,
            rho=rho,
            leak_rate=1.0,
            replicate=0,
            mc_total=0.0,
            mc_effective_delay=0.0,
        )
        for rho in (0.5, 0.9)
    )
    ipc_rows = tuple(
        _row(
            experiment=EXPERIMENT_IPC_SWEEP,
            rho=rho,
            leak_rate=1.0,
            replicate=0,
            ipc_linear=0.0,
            ipc_nonlinear=0.0,
        )
        for rho in (0.5, 0.9)
    )
    conservation = tuple(
        _row(
            experiment=EXPERIMENT_CONSERVATION,
            rho=0.95,
            leak_rate=1.0,
            replicate=0,
            n_units=n_units,
            state_noise=0.1,
            ipc_linear=0.0,
            ipc_nonlinear=0.0,
        )
        for n_units in (10, 20)
    )
    context = setup_style()
    paths = (
        plot_mc_sweep(mc_rows, (), tmp_path / "mc.png", style=context),
        plot_ipc_profile(ipc_rows, (), tmp_path / "ipc.png", style=context),
        plot_memory_nonlinearity(ipc_rows, tmp_path / "split.png", style=context),
        plot_ipc_conservation(conservation, tmp_path / "bound.png", style=context),
    )
    for path in paths:
        assert png_dpi(path) >= RETINA_DPI


@pytest.mark.parametrize(
    "draw",
    [
        pytest.param(
            lambda path, context: plot_mc_sweep((), (), path, style=context), id="mc"
        ),
        pytest.param(
            lambda path, context: plot_ipc_profile((), (), path, style=context),
            id="ipc",
        ),
        pytest.param(
            lambda path, context: plot_memory_nonlinearity((), path, style=context),
            id="split",
        ),
        pytest.param(
            lambda path, context: plot_ipc_conservation((), path, style=context),
            id="bound",
        ),
    ],
)
def test_plot_functions_reject_empty_rows(
    tmp_path: Path, draw: Callable[[Path, StyleContext], Path]
) -> None:
    """行が空なら黙って空の図を書かず ``ValueError`` (01・02 の図と同じ規律)。"""
    with pytest.raises(ValueError, match="rows"):
        draw(tmp_path / "unused.png", StyleContext())


# --- 長形式 -> 格子の復元 (D-38) ---------------------------------------------


def test_profile_grids_fill_missing_cells_with_zero_and_average_over_replicates() -> (
    None
):
    """欠けているセルは 0 として埋め、平均の分母は**レプリケート数**にする。

    長形式には正値セルしか無い (D-38) ので、「行が無い」= 「容量 0」である。
    平均を「在る行の数」で割ると、容量 0 のレプリケートを無視した過大評価に
    なる —— ここでは片方のレプリケートにだけ行がある状態を作り、平均が
    半分になることで分母を固定する。
    """
    rows = mc_sweep_rows(rhos=(0.9,), leaks=(1.0,), replicates=(0, 1))
    only_first = tuple(
        row for row in mc_sweep_profile(rows) if row.replicate == 0
    )
    means = mc_profile_means(rows, only_first, 1.0)
    assert set(means) == {0.9}
    profile = means[0.9]
    assert profile.shape == (N_DELAYS,)
    # 遅延 1..3 は 1/delay を1レプリケートぶんだけ持つ -> 2 で割られる
    assert profile[:3] == pytest.approx([0.5, 0.25, 1.0 / 6.0])
    # 遅延 4..6 は行が無い -> 0 (nan にしない)
    assert np.array_equal(profile[3:], np.zeros(N_DELAYS - 3))

    ipc_rows = ipc_sweep_rows(rhos=(0.9,), leaks=(1.0,), replicates=(0, 1))
    heatmap = ipc_heatmap_means(ipc_rows, ipc_sweep_profile(ipc_rows), 1.0)[0.9]
    assert heatmap.shape == (N_DEGREES, 3)
    assert heatmap[0] == pytest.approx([1.0, 0.5, 1.0 / 3.0])
    assert heatmap[1] == pytest.approx([0.5, 0.0, 0.0])


def test_profile_rows_of_other_experiments_are_ignored() -> None:
    """別の実験の長形式の行が混ざっても図には入らない。

    3-A と 3-B は (rho, leak_rate) の格子が一部重なるため、実験ラベルで
    絞らずに読むと 3-B のセルが 3-A のプロファイルに足し込まれる。
    """
    rows = mc_sweep_rows(rhos=(0.9,), leaks=(1.0,), replicates=(0,))
    intruder = ipc_sweep_rows(rhos=(0.9,), leaks=(1.0,), replicates=(0,))
    mixed = mc_sweep_profile(rows) + tuple(
        _profile(
            row, diagnostic=DIAGNOSTIC_MC, degree=1, delay=1, capacity=100.0
        )
        for row in intruder
    )
    profile = mc_profile_means(rows, mixed, 1.0)[0.9]
    assert profile[0] == pytest.approx(1.0)


def test_representative_leak_rate_picks_the_largest_total() -> None:
    """代表リーク率は総容量の平均が最大のもの (同点なら小さい方)。"""
    rows = mc_sweep_rows(rhos=(0.5, 0.9), leaks=(0.3, 1.0), replicates=(0, 1))
    assert figures_capacity.representative_leak_rate(rows, "mc_total") == 1.0
    tied = tuple(
        _row(
            experiment=EXPERIMENT_MC_SWEEP,
            rho=0.9,
            leak_rate=leak,
            replicate=0,
            mc_total=5.0,
        )
        for leak in (0.3, 1.0)
    )
    assert figures_capacity.representative_leak_rate(tied, "mc_total") == 0.3
    with pytest.raises(ValueError, match="rows"):
        figures_capacity.representative_leak_rate((), "mc_total")

"""作図層のテスト (D-10).

日本語フォントのために新規依存を足さないと決めた以上、**フォントが無い環境で
何が起きるか**が設計の本体になる。フォント設定だけを戻すと CI 上で豆腐文字の図が
「正常に」生成されてしまうため、ラベル文字列ごと英語へ切り替わることを固定する。

図そのものは Agg キャンバスで書く (ディスプレイ非依存)。解像度は rcParams では
なく**保存した PNG から実測**する。``setup_style()`` はプロセス全体の rcParams
を書き換えないため (F-1-008)、rcParams への反映は ``rc_params_for`` が返す辞書と
``matplotlib.rc_context`` の組で確認する。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np
import pytest
from conftest import png_dpi

from rc_basics_lab.experiment.runner import ResultRow
from rc_basics_lab.experiment.state_space import (
    DELAY_EMBEDDED_INPUT,
    RAW_INPUT,
    RESERVOIR_STATE,
    SpaceSummary,
    StateSpaceReport,
)
from rc_basics_lab.plotting import figures, style
from rc_basics_lab.plotting.labels import label
from rc_basics_lab.plotting.style import (
    CJK_FONT_CANDIDATES,
    SAVEFIG_DPI,
    StyleContext,
    find_cjk_font,
    rc_params_for,
    setup_style,
)

RETINA_DPI = 200


def _no_fonts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(style, "_available_font_names", frozenset)


def _only(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    monkeypatch.setattr(style, "_available_font_names", lambda: frozenset(names))


def test_labels_fall_back_to_english_without_cjk_font(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """CJK フォントが1つも無いとき、例外を出さず英語ラベルに切り替わる (D-10)。"""
    _no_fonts(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=style.logger.name):
        context = setup_style()
    assert context.cjk_font is None
    assert context.cjk_available is False
    assert context.label("リザバー状態", "reservoir state") == "reservoir state"
    assert label("累積寄与率", "cumulative ratio", cjk=False) == "cumulative ratio"
    assert "CJK" in caplog.text
    # フォント設定だけを触って日本語ラベルを残す (= 豆腐文字) 実装ではないこと
    assert find_cjk_font() is None


def test_cjk_font_is_used_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """候補フォントが1つでもあれば日本語ラベルを使い、rc_context 経由で反映される。

    ``setup_style()`` はもうプロセス全体の rcParams を書き換えない (F-1-008)。
    ``rc_params_for`` が返す辞書の中身と、``matplotlib.rc_context`` で
    一時適用したときに実際に効くことの両方を見る。
    """
    _only(monkeypatch, "IPAexGothic", "DejaVu Sans")
    context = setup_style()
    assert context.cjk_font == "IPAexGothic"
    assert context.label("リザバー状態", "reservoir state") == "リザバー状態"

    params = rc_params_for(context)
    sans_serif = cast("list[str]", params["font.sans-serif"])
    assert sans_serif[0] == "IPAexGothic"
    assert params["axes.unicode_minus"] is False

    # 適用前は既定値のまま (グローバルを汚していないことの確認)
    assert matplotlib.rcParams["font.sans-serif"][0] != "IPAexGothic"
    with matplotlib.rc_context(params):  # type: ignore[arg-type]
        assert matplotlib.rcParams["font.sans-serif"][0] == "IPAexGothic"
        assert matplotlib.rcParams["axes.unicode_minus"] is False
    # rc_context を抜けたら既定値に戻る
    assert matplotlib.rcParams["font.sans-serif"][0] != "IPAexGothic"


def test_font_candidates_are_tried_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """候補の並び順が優先順位そのものであること。"""
    _only(monkeypatch, *CJK_FONT_CANDIDATES)
    assert find_cjk_font() == CJK_FONT_CANDIDATES[0]
    _only(monkeypatch, CJK_FONT_CANDIDATES[-1], CJK_FONT_CANDIDATES[-2])
    assert find_cjk_font() == CJK_FONT_CANDIDATES[-2]


def test_savefig_dpi_is_retina(tmp_path: Path) -> None:
    """保存される図が retina 相当の dpi であること (受け入れ条件5)。

    ``setup_style()`` はもう rcParams を書き換えない (F-1-008) ので、
    グローバル rcParams ではなく**実際に保存した PNG から実測**する。
    """
    assert SAVEFIG_DPI >= RETINA_DPI
    context = setup_style()
    path = figures.plot_comparison(
        _rows(), tmp_path / "dpi.png", style=context, **_comparison_extras()
    )
    assert png_dpi(path) >= RETINA_DPI


def test_label_requires_both_languages() -> None:
    """英語ラベルの書き忘れは静かに通さない。"""
    with pytest.raises(ValueError, match="ja / en"):
        label("日本語だけ", "", cjk=False)


def _comparison_extras() -> dict[str, object]:
    """``plot_comparison`` の追加パネル (FIG-12) に渡す最小の入力。

    01 の主図は波形と自走 (01') を**同じ figure のパネル**として持つように
    なった。**1箇所にまとめる** —— 呼び出しごとに書くと、パネルが増えたときに
    テストだけが取り残される。
    """
    import numpy as np

    from rc_basics_lab.experiment.horizon import HORIZON_STEPS, HorizonRow

    truth = np.linspace(0.0, 1.0, 8)
    return {
        "waveform": (truth, {"esn": truth * 0.99}),
        "horizon_rows": tuple(
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
        ),
    }


def _rows() -> tuple[ResultRow, ...]:
    def row(task: str, method: str, replicate: int, nrmse: float) -> ResultRow:
        return ResultRow(
            task=task,
            method=method,
            replicate=replicate,
            seed_reservoir=0,
            seed_task=1,
            seed_split=2,
            alpha=1e-4,
            n_lags=0,
            rmse=nrmse,
            nrmse=nrmse,
            nmse=nrmse**2,
            sign_accuracy=0.5,
            n_train=10,
            n_val=5,
            n_test=5,
            t0=1,
            wall_time_s=0.1,
        )

    return tuple(
        row(task, method, replicate, value + 0.01 * replicate)
        for task, values in (
            ("mackey_glass", (0.15, 0.02, 0.03)),
            ("delay_parity", (1.0, 1.0, 0.09)),
        )
        for method, value in zip(("linear", "delay_line", "esn"), values, strict=True)
        for replicate in range(2)
    )


def _report(task: str) -> StateSpaceReport:
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((50, 2))
    curve = np.array([0.6, 0.9, 0.97, 1.0])
    spaces = (
        SpaceSummary(
            RAW_INPUT, 1, 1, 1.0, np.array([1.0]), rng.standard_normal((50, 1))
        ),
        SpaceSummary(DELAY_EMBEDDED_INPUT, 17, 3, 2.5, curve, scores),
        SpaceSummary(RESERVOIR_STATE, 200, 2, 1.8, curve, scores),
    )
    return StateSpaceReport(task=task, replicate=0, n_lags=16, n_rows=50, spaces=spaces)


def test_figures_are_written_at_retina_resolution(tmp_path: Path) -> None:
    """図2枚が実際に書き出され、PNG の実測 dpi が 200 以上 (受け入れ条件5)。"""
    context = setup_style()
    comparison = figures.plot_comparison(
        _rows(),
        tmp_path / "fig_comparison.png",
        style=context,
        **_comparison_extras(),
    )
    state_space = figures.plot_state_space(
        [_report("mackey_glass"), _report("delay_parity")],
        tmp_path / "fig_state_space.png",
        style=context,
    )
    for path in (comparison, state_space):
        assert path.is_file()
        assert path.stat().st_size > 0
        assert png_dpi(path) >= RETINA_DPI


def test_figures_render_without_cjk_font(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CJK フォントが無い環境でも図の生成が落ちない (CI が見る唯一の性質)。"""
    _no_fonts(monkeypatch)
    context = setup_style()
    assert context.cjk_available is False
    figures.plot_comparison(
        _rows(), tmp_path / "a.png", style=context, **_comparison_extras()
    )
    figures.plot_state_space(
        [_report("mackey_glass")], tmp_path / "b.png", style=context
    )
    assert (tmp_path / "a.png").is_file()
    assert (tmp_path / "b.png").is_file()


def test_aggregate_nrmse_matches_manual_mean_and_std() -> None:
    """図に出る平均±標準偏差が手計算と一致する (受け入れ条件3 の数値の出どころ)。"""
    stats = figures.aggregate_nrmse(_rows())
    aggregate = stats[("mackey_glass", "linear")]
    assert aggregate.n == 2
    assert aggregate.mean == pytest.approx(0.155)
    assert aggregate.std == pytest.approx(0.01 / np.sqrt(2))


def test_plot_comparison_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="rows"):
        figures.plot_comparison(
            [], Path("unused.png"), style=StyleContext(), **_comparison_extras()
        )

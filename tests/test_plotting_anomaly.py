"""記事05の図5枚のテスト (仕様 §4 T5 受け入れ基準4 / D-81).

図は**成果物 CSV の行だけを読む** (仕様 §5 禁止する構造7) ので、ここでは
行 dataclass を手で組んで5枚を描く。実験も掃引もこのファイルからは1回も
走らせない (走らせようとしたら落ちることを
``test_figures_never_run_an_experiment`` が実測する)。

``StyleContext(cjk_font=None)`` で描くのが既定である —— CI に CJK フォントが
無い環境で英語ラベルに切り替わること (D-10) が、この層で最も落としたくない
性質だからである。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from conftest import png_dpi

from rc_basics_lab.experiment.anomaly_rows import (
    AnomalyRow,
    ProtocolSweepRow,
    SizeSweepRow,
    ThresholdSweepRow,
    TimelineRow,
)
from rc_basics_lab.experiment.anomaly_score import (
    ANOMALY_METHODS,
    DELAY_LINE_RESIDUAL,
    ESN_RESIDUAL,
    RANDOM_CONTROL,
)
from rc_basics_lab.plotting.figures_anomaly import (
    MARKED_STYLE,
    MARKED_SUFFIX,
    METHOD_COLORS,
    METHOD_LABELS,
    UNMARKED_STYLE,
    UNMARKED_SUFFIX,
    method_label,
    method_line_style,
    plot_pr_curves,
    plot_score_timeline,
    plot_threshold_tradeoff,
)
from rc_basics_lab.plotting.figures_anomaly_sweep import (
    build_protocol_sensitivity_figure,
    plot_protocol_sensitivity,
    plot_size_vs_performance,
)
from rc_basics_lab.plotting.style import StyleContext, setup_style

STYLE = StyleContext(cjk_font=None)
RETINA_DPI = 200

MARKED_METHODS = (ESN_RESIDUAL, DELAY_LINE_RESIDUAL)
"""印を付ける系統 (実測の MGAB / 合成源で対照から離れる2系統に相当)。"""

CONDITIONS: tuple[tuple[str, int, int], ...] = (
    ("zscore", 16, 8),
    ("minmax", 16, 8),
    ("robust", 8, 1),
)


def _auprc_of(method: str) -> float:
    """系統ごとの作り物の AUPRC (順位が決まるように差を付ける)。"""
    return {
        ESN_RESIDUAL: 0.40,
        DELAY_LINE_RESIDUAL: 0.19,
    }.get(method, 0.08)


def anomaly_rows() -> tuple[AnomalyRow, ...]:
    """``anomaly.csv`` 相当の行 (1系列 x 2レプリケート x 6系統)。"""
    return tuple(
        AnomalyRow(
            dataset="synthetic",
            series="s1",
            method=method,
            replicate=replicate,
            seed_reservoir=0,
            seed_task=1,
            seed_split=2,
            seed_control=5,
            normalize="zscore",
            preprocessor_id="abc123456789",
            selected_alpha=1e-3,
            auprc=_auprc_of(method) + 0.01 * replicate,
            auprc_random=0.08,
            anomaly_rate=0.078,
            threshold=1.5,
            f1_calibrated=0.3,
            precision_calibrated=0.4,
            recall_calibrated=0.24,
            far_test=0.012,
            n_evaluated=1000,
            n_train=400,
            n_calibration=200,
            n_test=1000,
            t0=20,
            split_offset=3,
            wall_time_s=0.01,
            f1_test_optimal=0.45,
        )
        for method in ANOMALY_METHODS
        for replicate in range(2)
    )


def threshold_rows() -> tuple[ThresholdSweepRow, ...]:
    """``anomaly_threshold.csv`` 相当の行 (系統 x レプリケート x 予算5点)。"""
    return tuple(
        ThresholdSweepRow(
            dataset="synthetic",
            series="s1",
            method=method,
            replicate=replicate,
            target_false_alarm_rate=(index + 1) / 5.0,
            threshold=2.0 - 0.2 * index,
            precision=_auprc_of(method) * (1.0 - 0.1 * index),
            recall=0.2 * (index + 1),
            f1=0.3 - 0.02 * index,
            false_alarm_rate=(index + 1) / 5.0,
            n_alarms=10 * (index + 1),
            calibrated_threshold=1.5,
        )
        for method in ANOMALY_METHODS
        for replicate in range(2)
        for index in range(5)
    )


def timeline_rows() -> tuple[TimelineRow, ...]:
    """``anomaly_timeline.csv`` 相当の行 (6系統 x 60 点、うち 10 点が異常)。"""
    return tuple(
        TimelineRow(
            dataset="synthetic",
            series="s1",
            method=method,
            replicate=0,
            index=1000 + index,
            score=math.sin(0.3 * index) + (2.0 if 20 <= index < 30 else 0.0),
            is_anomaly=20 <= index < 30,
            is_ignored=index in {19, 30},
            threshold=1.5,
        )
        for method in ANOMALY_METHODS
        for index in range(60)
    )


def protocol_rows() -> tuple[ProtocolSweepRow, ...]:
    """``anomaly_protocol.csv`` 相当の行 (3格子点 x 6系統)。

    **印のある2系統の順位は不動**、印の無い4系統だけが入れ替わる —— T4 の
    実測 (27格子点中21で順位が動いたが、逆転46組はすべて印の無い系統の内部)
    と同じ形にしてある。
    """
    rows: list[ProtocolSweepRow] = []
    for position, (normalize, window, smoothing) in enumerate(CONDITIONS):
        others = [method for method in ANOMALY_METHODS if method not in MARKED_METHODS]
        shuffled = others[position:] + others[:position]
        order = [*MARKED_METHODS, *shuffled]
        for rank, method in enumerate(order, start=1):
            marked = method in MARKED_METHODS
            rows.append(
                ProtocolSweepRow(
                    dataset="synthetic",
                    normalize=normalize,
                    input_window=window,
                    score_smoothing=smoothing,
                    is_headline=position == 0,
                    method=method,
                    auprc_mean=_auprc_of(method) + 0.001 * (6 - rank),
                    auprc_sd=0.01,
                    auprc_random_mean=0.08,
                    n_pairs=15,
                    n_better_than_control=15 if marked else 8,
                    control_sign_p=3.05e-05 if marked else 0.5,
                    distinguishable=marked,
                    rank=rank,
                    reference_rank=rank if marked else 3,
                    rank_changed=not marked and rank != 3,
                    reference_distinguishable=marked,
                    kendall_tau=0.6,
                    n_discordant_pairs=0 if position == 0 else 3,
                    n_discordant_pairs_distinguishable=0,
                )
            )
    return tuple(rows)


def size_rows() -> tuple[SizeSweepRow, ...]:
    """``anomaly_size.csv`` 相当の行 (N 4点 x 6系統)。"""
    grid = (25, 50, 100, 200)
    return tuple(
        SizeSweepRow(
            dataset="synthetic",
            n_units=n_units,
            method=method,
            auprc_mean=_auprc_of(method)
            * (n_units / 200.0 if method == ESN_RESIDUAL else 1.0),
            auprc_sd=0.01,
            auprc_random_mean=0.08,
            n_pairs=15,
            n_better_than_control=15 if method in MARKED_METHODS else 8,
            control_sign_p=3.05e-05 if method in MARKED_METHODS else 0.5,
            distinguishable=method in MARKED_METHODS,
            reference_n_units=200,
            auprc_reference=_auprc_of(method),
            auprc_ratio=(n_units / 200.0 if method == ESN_RESIDUAL else 1.0),
            below_reference_fraction=(method == ESN_RESIDUAL and n_units / 200.0 < 0.9),
            n_train=4900,
        )
        for n_units in grid
        for method in ANOMALY_METHODS
    )


def _write_all(directory: Path, style: StyleContext) -> tuple[Path, ...]:
    return (
        plot_pr_curves(
            anomaly_rows(),
            threshold_rows(),
            directory / "fig_pr_curves.png",
            style=style,
        ),
        plot_score_timeline(
            timeline_rows(), directory / "fig_score_timeline.png", style=style
        ),
        plot_threshold_tradeoff(
            anomaly_rows(),
            threshold_rows(),
            directory / "fig_threshold_tradeoff.png",
            style=style,
        ),
        plot_protocol_sensitivity(
            protocol_rows(), directory / "fig_protocol_sensitivity.png", style=style
        ),
        plot_size_vs_performance(
            size_rows(), directory / "fig_size_vs_performance.png", style=style
        ),
    )


def test_all_five_figures_are_written_without_a_cjk_font(tmp_path: Path) -> None:
    """CJK フォントが無い環境でも5枚が描ける (受け入れ基準4 / D-10)。"""
    paths = _write_all(tmp_path, STYLE)
    assert len({path.name for path in paths}) == 5
    for path in paths:
        assert path.stat().st_size > 0


def test_figures_are_written_at_retina_resolution(tmp_path: Path) -> None:
    """5枚の PNG の実測 dpi が 200 以上 (02〜04 と同じ規律)。"""
    for path in _write_all(tmp_path, setup_style()):
        assert png_dpi(path) >= RETINA_DPI


def test_figures_never_run_an_experiment(tmp_path: Path) -> None:
    """**図が実験・掃引を走らせない** (仕様 §5 禁止する構造7)。

    実験の入口を「呼ばれたら落ちる」ものに差し替えたまま5枚を描く。
    """
    import rc_basics_lab.experiment.anomaly as anomaly_module
    import rc_basics_lab.experiment.anomaly_sweep as sweep_module

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("図が実験を走らせました")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(anomaly_module, "run_anomaly_headline", forbidden)
        patch.setattr(anomaly_module, "run_anomaly_replicate", forbidden)
        patch.setattr(sweep_module, "run_protocol_sweep", forbidden)
        patch.setattr(sweep_module, "run_size_sweep", forbidden)
        _write_all(tmp_path, STYLE)


def test_the_protocol_figure_marks_the_methods_that_are_distinguishable() -> None:
    """5-C の図が印を可視化している (D-81 の guard_test)。

    ここが守るのは「印のある系統と無い系統が、図の上で**必ず違って見える**」
    ことである。T4 の実測では 27 格子点中 21 で順位が動いたが、逆転した 46 組は
    すべて印の無い4系統の内部だった —— 印を描かない図は「プロトコルに敏感」と
    いう**逆の結論**を伝える。

    凡例の文言・線種・線幅・マーカーの塗りの4つすべてで区別する。全系統を同じ
    体裁で描く実装に変異させると、どれか1つを直しても残りで落ちる。
    """
    figure = build_protocol_sensitivity_figure(protocol_rows(), style=STYLE)
    lines = {str(line.get_label()): line for line in figure.axes[0].get_lines()}
    marked = {label: line for label, line in lines.items() if MARKED_SUFFIX[1] in label}
    unmarked = {
        label: line for label, line in lines.items() if UNMARKED_SUFFIX[1] in label
    }
    assert len(marked) == len(MARKED_METHODS), lines
    assert len(unmarked) == len(ANOMALY_METHODS) - len(MARKED_METHODS), lines
    assert set(marked) | set(unmarked) == set(lines)
    for line in marked.values():
        assert line.get_linestyle() == MARKED_STYLE.linestyle
        assert line.get_linewidth() == MARKED_STYLE.linewidth
        assert line.get_fillstyle() == MARKED_STYLE.fillstyle
    for line in unmarked.values():
        assert line.get_linestyle() != MARKED_STYLE.linestyle
        assert line.get_linewidth() < MARKED_STYLE.linewidth
        assert line.get_fillstyle() == UNMARKED_STYLE.fillstyle


def test_the_protocol_figure_separates_the_reversals_that_carry_a_mark() -> None:
    """逆転の「延べ」と「両方に印がある」を**別の棒**で描く (D-81)。

    延べだけを描くと 46 組が結論として読まれる。実測では印のある組は 0 組で、
    2本を並べて初めて「測っていたのは雑音の順位」だと分かる。
    """
    rows = protocol_rows()
    figure = build_protocol_sensitivity_figure(rows, style=STYLE)
    labels = [str(container.get_label()) for container in figure.axes[1].containers]
    assert len(labels) == 2, labels
    total = sum(row.n_discordant_pairs for row in rows if row.method == ESN_RESIDUAL)
    marked = sum(
        row.n_discordant_pairs_distinguishable
        for row in rows
        if row.method == ESN_RESIDUAL
    )
    assert f"total {total}" in labels[0]
    assert f"both distinguishable ({marked})" in labels[1]


def test_method_line_style_and_label_differ_by_the_mark() -> None:
    """印の有無で体裁と文言が必ず変わる (D-81 の最小単位)。"""
    assert method_line_style(True) != method_line_style(False)
    assert MARKED_STYLE.linewidth > UNMARKED_STYLE.linewidth
    assert MARKED_STYLE.fillstyle != UNMARKED_STYLE.fillstyle
    marked = method_label(ESN_RESIDUAL, STYLE, mark=True)
    unmarked = method_label(ESN_RESIDUAL, STYLE, mark=False)
    assert marked != unmarked
    assert MARKED_SUFFIX[1] in marked
    assert UNMARKED_SUFFIX[1] in unmarked
    assert method_label(ESN_RESIDUAL, STYLE) == METHOD_LABELS[ESN_RESIDUAL][1]


def test_unknown_method_label_fails_before_drawing() -> None:
    """対応表に無い系統名は ``ValueError`` (図から静かに消さない、D-10)。"""
    with pytest.raises(ValueError, match="ラベルの対応表にありません"):
        method_label("unknown_method", STYLE)


def test_every_method_has_a_label_and_a_color() -> None:
    """``ANOMALY_METHODS`` の全要素が対応表にある (系統を足したら図も直す)。"""
    assert set(METHOD_LABELS) == set(ANOMALY_METHODS)
    assert set(METHOD_COLORS) == set(ANOMALY_METHODS)
    assert len(set(METHOD_COLORS.values())) == len(ANOMALY_METHODS)


def test_the_control_is_never_dropped_from_the_size_figure(tmp_path: Path) -> None:
    """5-D の図が対照の行も描く (D-61 の作図側)。"""
    rows = size_rows()
    assert any(row.method == RANDOM_CONTROL for row in rows)
    path = plot_size_vs_performance(rows, tmp_path / "size.png", style=STYLE)
    assert path.stat().st_size > 0


def test_figures_reject_empty_rows(tmp_path: Path) -> None:
    """行が空なら描く前に落ちる (空の図を成果物にしない)。"""
    with pytest.raises(ValueError, match="rows が空です"):
        plot_score_timeline((), tmp_path / "a.png", style=STYLE)
    with pytest.raises(ValueError, match="rows が空です"):
        plot_protocol_sensitivity((), tmp_path / "b.png", style=STYLE)
    with pytest.raises(ValueError, match="rows が空です"):
        plot_size_vs_performance((), tmp_path / "c.png", style=STYLE)
    with pytest.raises(ValueError, match="rows が空です"):
        plot_pr_curves((), threshold_rows(), tmp_path / "d.png", style=STYLE)

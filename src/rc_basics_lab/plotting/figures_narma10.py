"""NARMA10 の3枚を1枚に畳んだ図 (FIG-12 / C-6).

03 は図が 7 枚あり、``連載構成案_RC基礎編.md`` の想定 (2〜4 枚) の倍近く
あった。うち NARMA10 の 3 枚は**同じ主張を支えている** ——
「遅延線は正則化とタップ数の効きで ESN を上回れる」であり、
成績 (3-C) → その効きの内訳 (3-C') → 実際の波形、という一続きの流れになる。

各パネルの描画は元の図の関数をそのまま呼ぶ (書き写さない)。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from statistics import mean, stdev

from rc_basics_lab.experiment.narma import (
    NARMA10_REFERENCE_NOTE,
    NARMA10_REFERENCE_NOTE_EN,
)
from rc_basics_lab.experiment.narma_operating import OperatingPointRow
from rc_basics_lab.experiment.narma_taps import TapSweepRow
from rc_basics_lab.experiment.runner import ResultRow
from rc_basics_lab.plotting.figures_capacity import draw_narma10_control_panel
from rc_basics_lab.plotting.figures_narma_taps import draw_narma10_taps_panel
from rc_basics_lab.plotting.figures_operating import (
    draw_capacity_panel,
    operating_headline,
)
from rc_basics_lab.plotting.labels import METHOD_LABELS
from rc_basics_lab.plotting.layout import label_panels, wrapped_note
from rc_basics_lab.plotting.narma10_panel import narma10_subtitle
from rc_basics_lab.plotting.style import (
    PANEL_TITLE_SIZE,
    StyleContext,
    add_provenance,
    new_figure,
    rc_context_for,
    save_png,
)
from rc_basics_lab.plotting.waveforms import (
    draw_prediction_waveform,
    waveform_headline,
)
from rc_basics_lab.types import FloatArray


def narma10_verdict(rows: Sequence[ResultRow], style: StyleContext) -> str:
    """報告する1点での勝敗を**行から数えて**返す (固定文にしない)。

    3-C'' のパネルが入った以上、表題は「どちらが勝つか」ではなく
    「どこで勝つか」を言わなければならない。

    **実測の s.d. の範囲で並ぶものに「最良」と言わない。** 許容差なしの
    argmin を取ると、0.1534 (OLS) と 0.1538 (リッジ) の差 0.0004 —— 同じ
    条件の s.d. 0.0071 の**19分の1** —— に「最良」の語が付き、記事の主張
    (「小さいタップ数では正則化の有無で結論が変わらない」) と矛盾する。
    最良の手法の s.d. 以内に収まるものは**並んでいる**と書く。
    """
    stats = {
        method: (
            mean([row.nmse for row in rows if row.method == method]),
            _spread([row.nmse for row in rows if row.method == method]),
        )
        for method in {row.method for row in rows}
    }
    best = min(stats, key=lambda key: stats[key][0])
    best_mean, best_spread = stats[best]
    tied = sorted(
        method
        for method, (value, _) in stats.items()
        if value - best_mean <= best_spread
    )
    if len(tied) == 1:
        return style.label(
            f"この動作点で最良なのは {METHOD_LABELS[best][0]} である",
            f"the best method at this operating point is {METHOD_LABELS[best][1]}",
        )
    names_ja = " / ".join(METHOD_LABELS[method][0] for method in tied)
    names_en = " / ".join(METHOD_LABELS[method][1] for method in tied)
    return style.label(
        f"この動作点では {names_ja} が s.d. の範囲で並ぶ",
        f"at this operating point {names_en} tie within one s.d.",
    )


def _spread(values: Sequence[float]) -> float:
    """レプリケート間の標準偏差 (1本しか無ければ 0)。"""
    return float(stdev(values)) if len(values) > 1 else 0.0


def plot_narma10(
    rows: Sequence[ResultRow],
    tap_rows: Sequence[TapSweepRow],
    operating_rows: Sequence[OperatingPointRow],
    waveform: tuple[FloatArray, dict[str, FloatArray]],
    path: Path,
    *,
    style: StyleContext,
) -> Path:
    """実験 3-C / 3-C' と予測波形を1枚に並べる (FIG-12 / C-6)。

    Args:
        rows: ``narma10.csv`` と同じ行 (左上のパネル)。
        tap_rows: ``narma10_taps.csv`` と同じ行 (上段中央のパネル)。
        operating_rows: ``narma10_operating.csv`` と同じ行 (右上のパネル、
            3-C'' / D-146)。
        waveform: ``(真値, 手法 -> 予測)`` (下段のパネル)。
        path: 出力先の PNG。
        style: 配色・言語・commit。

    Returns:
        書き出した PNG のパス。

    Raises:
        ValueError: いずれかの入力が空の場合。
    """
    if not rows:
        raise ValueError("rows が空です")
    truth, predictions = waveform
    with rc_context_for(style):
        # 上段に2つ、下段に波形。下段を全幅にするのは、波形が横軸に
        # 100 ステップあり、半幅では線が重なって読めなくなるためである。
        figure = new_figure(18.0, 9.0)
        grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 1.15))
        control = figure.add_subplot(grid[0, 0])
        taps = figure.add_subplot(grid[0, 1])
        operating = figure.add_subplot(grid[0, 2])
        draw_narma10_control_panel(control, rows, style)
        draw_narma10_taps_panel(taps, tap_rows, style)
        # 3-C'' は**動作点で勝敗が変わる**ことを示す (D-144)。3-C 本体の
        # パネルの隣に置くのは、あちらが報告する1点がこちらの面の1点だから
        # である —— 別の図にすると「その1点を選んだ」ことが読者に見えない。
        draw_capacity_panel(operating, operating_rows, style)
        operating.set_title(
            operating_headline(operating_rows, style), fontsize=PANEL_TITLE_SIZE
        )
        drawn = draw_prediction_waveform(
            figure, figure.add_subplot(grid[1, :]), truth, predictions, style
        )
        drawn.top.set_title(
            f"NARMA10: {waveform_headline(truth, predictions, style)}",
            fontsize=PANEL_TITLE_SIZE,
        )
        # 記号は**波形を2段に割った後**に振る (FIG-16)。
        # ``draw_prediction_waveform`` は渡された軸を ``remove()`` するので、
        # 先に振ると下段の記号が消える (``fig_comparison`` で実際に消えていた)。
        label_panels([control, taps, operating, drawn.top], style=style)
        # **固定文にしない** (D-145 と同じ規律)。3-C'' のパネルが入った時点で
        # 「遅延線が ESN を上回る」は (a) の1点でしか正しくなくなった ——
        # 表題とパネルが食い違う図は、読者に嘘をつく。
        figure.suptitle(
            style.label(
                "実験 3-C / 3-C' / 3-C'': NARMA10 の勝敗は動作点で決まる\n"
                f"報告する1点では遅延線が上回るが、{narma10_verdict(rows, style)}",
                "Experiments 3-C / 3-C' / 3-C'': the NARMA10 winner is set by"
                " the operating point\nAt the reported point the delay line"
                f" wins, but {narma10_verdict(rows, style)}",
            )
        )
        # **原典未特定の注を落とさない。** 参照値 (0.16 / 0.107) は出典が
        # 特定できていないので、数字だけを置くと後から図の側から辿れない。
        # **行ごとに折り返す** (1-8)。この注記は2行構成なので、まとめて折ると
        # 段落の切れ目が消える。実測でこの図の最終行が図幅いっぱいだった。
        figure.supxlabel(
            wrapped_note(narma10_subtitle(style))
            + "\n"
            + wrapped_note(
                style.label(NARMA10_REFERENCE_NOTE, NARMA10_REFERENCE_NOTE_EN)
            ),
            fontsize=8,
        )
        conditions = (
            f"n_train = {rows[0].n_train}, "
            f"k = {min(row.n_lags for row in tap_rows)}"
            f"..{max(row.n_lags for row in tap_rows)}, "
            f"waveform steps = 0..{drawn.length}"
        )
        add_provenance(figure, conditions, rows, style=style)
        return save_png(figure, path)


__all__ = ["plot_narma10"]

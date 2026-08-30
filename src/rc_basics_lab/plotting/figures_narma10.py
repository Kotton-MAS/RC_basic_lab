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

from rc_basics_lab.experiment.narma import (
    NARMA10_REFERENCE_NOTE,
    NARMA10_REFERENCE_NOTE_EN,
)
from rc_basics_lab.experiment.narma_taps import TapSweepRow
from rc_basics_lab.experiment.runner import ResultRow
from rc_basics_lab.plotting.figures_capacity import draw_narma10_control_panel
from rc_basics_lab.plotting.figures_narma_taps import draw_narma10_taps_panel
from rc_basics_lab.plotting.layout import wrapped_note
from rc_basics_lab.plotting.narma10_panel import narma10_subtitle
from rc_basics_lab.plotting.style import (
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


def plot_narma10(
    rows: Sequence[ResultRow],
    tap_rows: Sequence[TapSweepRow],
    waveform: tuple[FloatArray, dict[str, FloatArray]],
    path: Path,
    *,
    style: StyleContext,
) -> Path:
    """実験 3-C / 3-C' と予測波形を1枚に並べる (FIG-12 / C-6)。

    Args:
        rows: ``narma10.csv`` と同じ行 (左上のパネル)。
        tap_rows: ``narma10_taps.csv`` と同じ行 (右上のパネル)。
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
        figure = new_figure(13.0, 9.0)
        grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.15))
        draw_narma10_control_panel(figure.add_subplot(grid[0, 0]), rows, style)
        draw_narma10_taps_panel(figure.add_subplot(grid[0, 1]), tap_rows, style)
        drawn = draw_prediction_waveform(
            figure, figure.add_subplot(grid[1, :]), truth, predictions, style
        )
        drawn.top.set_title(
            f"NARMA10: {waveform_headline(truth, predictions, style)}", fontsize=9
        )
        figure.suptitle(
            style.label(
                "実験 3-C / 3-C': NARMA10 では遅延線が ESN を上回る",
                "Experiments 3-C / 3-C': on NARMA10 the delay line beats the ESN",
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

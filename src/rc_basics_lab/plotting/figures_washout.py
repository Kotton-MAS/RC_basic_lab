"""実験 2-D: washout 長への性能感度 (``fig_washout_sensitivity``).

``figures_esp.py`` から分けてあるのは行数上限 (D-77) のためである。
**上限のほうは緩めない**。2-D は他の3枚 (2-A/2-B/2-C) と違って 01 の
``run_experiment`` を回した行 (``WashoutRow``) を読むので、まとまりとして切れる。

**比はレプリケートで対応をとって作る** (``_paired_ratio_series``、2-15) ——
平均の比に生のばらつきを流用すると、レプリケート間の水準差 (この掃引では ±11%)
がそのまま誤差棒になり、washout の効果 (1% 未満) と桁が合わない。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from rc_basics_lab.experiment.washout import (
    HEADLINE_METHOD,
    HEADLINE_TASK,
    WashoutRow,
    WashoutSensitivity,
    mean_nrmse_by_washout,
)
from rc_basics_lab.plotting.labels import METHOD_LABELS
from rc_basics_lab.plotting.style import (
    StyleContext,
    add_provenance,
    method_color,
    new_figure,
    rc_context_for,
    reference_line_kwargs,
    require_rows,
    save_png,
)
from rc_basics_lab.types import FloatArray

_TASK_LABELS: Mapping[str, tuple[str, str]] = {
    "mackey_glass": ("Mackey-Glass", "Mackey-Glass"),
    "delay_parity": ("遅延パリティ", "delay parity"),
}
"""課題名の表示ラベル (D-10: ja/en の対)。未知の課題は名前をそのまま出す。"""


_CONTROL_ALPHA = 0.40
"""対照 (主役でない課題) の線の不透明度。「薄く重ねる」の実体。"""

_HEADLINE_LINEWIDTH = 2.0
_CONTROL_LINEWIDTH = 1.2


def _mark_reference(
    axis: Axes, sensitivity: WashoutSensitivity, style: StyleContext
) -> None:
    """01 の本番値に垂直線を引く (仕様 §4 T4)。"""
    axis.axvline(
        float(sensitivity.reference_washout),
        **reference_line_kwargs(1),
        label=style.label(
            f"01 の本番値 (washout = {sensitivity.reference_washout})",
            f"production value used in 01 (washout = {sensitivity.reference_washout})",
        ),
    )


def _variation_note(sensitivity: WashoutSensitivity, style: StyleContext) -> str:
    """変動幅の数値注記 (仕様 §4 T4: 「変動幅を数値注記する」)。

    比だけでなく**レプリケート間のばらつきと比べてどうか**まで書く。比が
    1.0 でないことだけを見て「効果があった」と読まれるのを防ぐため。
    """
    headline = sensitivity.headline
    verdict_ja, verdict_en = (
        ("レプリケート間のばらつきを超える", "exceeds the replicate spread")
        if headline.exceeds_replicate_noise
        else ("レプリケート間のばらつき以下", "within the replicate spread")
    )
    task_ja, task_en = _TASK_LABELS.get(headline.task, (headline.task, headline.task))
    method_ja, method_en = METHOD_LABELS.get(
        headline.method, (headline.method, headline.method)
    )
    return style.label(
        f"{task_ja} x {method_ja}: 変動幅 (最大/最小) = {headline.ratio:.4f} 倍\n"
        f"({headline.nrmse_min:.3g} .. {headline.nrmse_max:.3g}、"
        f"レプリケート間 s.d. 最大 {headline.replicate_std_max:.3g})\n"
        f"-> {verdict_ja}",
        f"{task_en} x {method_en}: max/min = {headline.ratio:.4f}\n"
        f"({headline.nrmse_min:.3g} .. {headline.nrmse_max:.3g}; "
        f"max replicate s.d. {headline.replicate_std_max:.3g})\n"
        f"-> {verdict_en}",
    )


def _series_label(style: StyleContext, task: str, method: str) -> str:
    """凡例のラベル ``課題 x 手法``。"""
    task_ja, task_en = _TASK_LABELS.get(task, (task, task))
    method_ja, method_en = METHOD_LABELS.get(method, (method, method))
    return style.label(f"{task_ja} x {method_ja}", f"{task_en} x {method_en}")


def _washout_series(
    rows: Sequence[WashoutRow], task: str, method: str
) -> tuple[list[float], FloatArray, FloatArray]:
    """(washout, レプリケート平均 NRMSE, レプリケート間標準偏差) の3点セット。

    平均は ``mean_nrmse_by_washout`` (実験層) と共有する。標準偏差は
    ``WashoutRow.nrmse_std`` がすでに (課題, 手法, washout) 単位の値なので、
    同じ組の行から1つ拾えばよい。
    """
    means = mean_nrmse_by_washout(rows, task, method)
    stds = {
        row.washout: row.nrmse_std
        for row in rows
        if row.task == task and row.method == method
    }
    washouts = list(means)
    values: FloatArray = np.array(
        [means[washout] for washout in washouts], dtype=np.float64
    )
    errors: FloatArray = np.array(
        [stds[washout] for washout in washouts], dtype=np.float64
    )
    return [float(washout) for washout in washouts], values, errors


def _plot_absolute_panel(
    axis: Axes,
    rows: Sequence[WashoutRow],
    pairs: Sequence[tuple[str, str]],
    sensitivity: WashoutSensitivity,
    style: StyleContext,
) -> None:
    """左パネル: NRMSE の絶対値 (課題間の水準差を示す文脈)。"""
    for task, method in pairs:
        washouts, values, errors = _washout_series(rows, task, method)
        headline = task == HEADLINE_TASK
        axis.errorbar(
            washouts,
            values,
            yerr=errors,
            fmt="o-" if headline else "s--",
            capsize=3,
            color=method_color(method),
            alpha=1.0 if headline else _CONTROL_ALPHA,
            linewidth=_HEADLINE_LINEWIDTH if headline else _CONTROL_LINEWIDTH,
            label=_series_label(style, task, method),
        )
    axis.set_yscale("log")
    axis.set_xlabel(style.label("washout [ステップ]", "washout [steps]"))
    axis.set_ylabel(
        style.label("テスト NRMSE (レプリケート平均)", "test NRMSE (mean over reps)")
    )
    axis.set_title(
        style.label(
            "絶対値 (誤差棒はレプリケート間の標準偏差)",
            "Absolute NRMSE (error bars: spread over replicates)",
        ),
        fontsize=10,
    )
    _mark_reference(axis, sensitivity, style)
    axis.legend(loc="center right", fontsize=7, ncols=1)


def _paired_ratio_series(
    rows: Sequence[WashoutRow], task: str, method: str, reference_washout: int
) -> tuple[list[float], FloatArray, FloatArray] | None:
    """**レプリケートで対応をとった**比の (washout, 平均, 標準偏差) (2-15)。

    比を「平均の比」で作り、誤差棒に生の NRMSE のばらつきを流用すると、
    レプリケート間の水準差 (この掃引では ±11%) がそのまま誤差棒になる。
    washout の効果は 1% 未満なので、**図が主張と逆の印象**になっていた。

    同じレプリケートは全 washout で同じ種 (リザバー・課題・分割) を使うので、
    レプリケートごとに ``NRMSE(w) / NRMSE(基準)`` を作れば水準差が相殺される。
    実測ではこの操作で誤差棒が ±0.11 -> ±0.01 になり、比そのもの
    (0.999〜1.007) と同じ桁になる。**見え方を主張に寄せたのではなく、
    測っている量を「washout の効果」に合わせた。**

    Returns:
        ``(washout, 比の平均, 比の標準偏差)``。基準の washout が欠けている
        レプリケートが1つでもあれば ``None`` (対応が取れないので比を作らない)。
    """
    per_replicate: dict[int, dict[int, float]] = {}
    for row in rows:
        if row.task == task and row.method == method:
            per_replicate.setdefault(row.replicate, {})[row.washout] = row.nrmse
    if not per_replicate:
        return None
    washouts = sorted(
        {washout for series in per_replicate.values() for washout in series}
    )
    ratios: list[list[float]] = []
    for washout in washouts:
        column: list[float] = []
        for series in per_replicate.values():
            reference = series.get(reference_washout)
            value = series.get(washout)
            if reference is None or value is None or reference <= 0.0:
                return None
            column.append(value / reference)
        ratios.append(column)
    means: FloatArray = np.array([float(np.mean(c)) for c in ratios], dtype=np.float64)
    stds: FloatArray = np.array([float(np.std(c)) for c in ratios], dtype=np.float64)
    return [float(washout) for washout in washouts], means, stds


def _plot_relative_panel(
    axis: Axes,
    rows: Sequence[WashoutRow],
    pairs: Sequence[tuple[str, str]],
    sensitivity: WashoutSensitivity,
    style: StyleContext,
) -> None:
    """右パネル: 01 の本番値を 1 とした相対 NRMSE。

    絶対値のパネルでは水準差 (MG の 7e-4 と パリティの 1.0) が支配的で、
    washout による 1% 未満の変動が読めない。**この図の主張は変動の大きさ**な
    ので、基準点で割った比を別パネルにする。

    比は**レプリケートで対応をとって**作る (``_paired_ratio_series``、2-15)。
    """
    for task, method in pairs:
        series = _paired_ratio_series(rows, task, method, sensitivity.reference_washout)
        if series is None:
            continue
        washouts, ratios, errors = series
        headline = task == HEADLINE_TASK
        axis.errorbar(
            washouts,
            ratios,
            yerr=errors,
            fmt="o-" if headline else "s--",
            capsize=3,
            color=method_color(method),
            alpha=1.0 if headline else _CONTROL_ALPHA,
            linewidth=_HEADLINE_LINEWIDTH if headline else _CONTROL_LINEWIDTH,
            label=_series_label(style, task, method),
        )
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.set_xlabel(style.label("washout [ステップ]", "washout [steps]"))
    axis.set_ylabel(
        style.label(
            f"NRMSE / NRMSE(washout={sensitivity.reference_washout})",
            f"NRMSE / NRMSE(washout={sensitivity.reference_washout})",
        )
    )
    axis.set_title(
        style.label(
            "01 の本番値で正規化した比",
            "Normalised by the production value used in 01",
        ),
        fontsize=10,
    )
    _mark_reference(axis, sensitivity, style)
    axis.text(
        0.02,
        0.02,
        _variation_note(sensitivity, style),
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
    )


def plot_washout_sensitivity(
    rows: Sequence[WashoutRow],
    path: Path,
    *,
    style: StyleContext,
    sensitivity: WashoutSensitivity,
) -> Path:
    """washout 長への性能感度を描く (受け入れ条件5、D-19)。

    主役は Mackey-Glass、遅延パリティは「washout に反応しない対照」。01 の本番値
    に垂直線を引き、変動幅を注記する。副題に ``pad_series`` を書くのは、この図が
    「washout の効果」を示すのか「washout + 訓練量の効果」を示すのかが、その1点
    で変わるため (D-19)。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    require_rows(rows)
    pairs = sorted(
        {(row.task, row.method) for row in rows},
        key=lambda pair: (pair[0] != HEADLINE_TASK, pair[1] != HEADLINE_METHOD, pair),
    )

    with rc_context_for(style):
        figure = new_figure(11.0, 5.8)  # 高さは 1-6 (Zenn 幅で潰れない比)
        axes = figure.subplots(1, 2, squeeze=False)
        _plot_absolute_panel(axes[0][0], rows, pairs, sensitivity, style)
        _plot_relative_panel(axes[0][1], rows, pairs, sensitivity, style)
        design_ja, design_en = (
            (
                "訓練/検証/テストの行数を格子全体で一定に保つ補償あり",
                "with compensation: train/val/test sizes held constant",
            )
            if sensitivity.pad_series
            else (
                "補償なし (washout と訓練データ量が交絡した設計)",
                "no compensation: washout is confounded with the training size",
            )
        )
        figure.suptitle(
            style.label(
                f"実験 2-D: washout 長への性能感度\n{design_ja}",
                f"Experiment 2-D: sensitivity to the washout length\n{design_en}",
            )
        )
        conditions = (
            f"n_train = {rows[0].n_train}, pad_series = {sensitivity.pad_series}"
        )
        add_provenance(figure, conditions, rows, style=style)
        return save_png(figure, path)


__all__ = ["plot_washout_sensitivity"]

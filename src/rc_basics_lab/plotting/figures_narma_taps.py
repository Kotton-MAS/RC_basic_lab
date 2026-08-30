"""実験 3-C' の図 —— タップ数と正則化の効き目 (D-95).

3-C 本体の図 (``figures_capacity.plot_narma10_control``) は動作点1つの
比較なので、「正則化の有無が結論を変えるのはどこからか」を答えられない。
ここが描くのは**タップ数の軸**である。

横軸を ``k / n_train`` にするのは、先行 (Goudarzi et al. 2014) の対照が
「正則化なし かつ k ≈ n_train」だったからで、タップ数の絶対値ではなく
**訓練長に対する比**が壊れ方を決める。先行の動作点を参照線で示す。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes

from rc_basics_lab.experiment.narma_taps import TapSweepRow
from rc_basics_lab.experiment.runner import DELAY_LINE, DELAY_LINE_OLS
from rc_basics_lab.plotting.capacity_grids import mean_std
from rc_basics_lab.plotting.labels import (
    GOUDARZI_2014,
    METHOD_LABELS,
    cited_measurement,
)
from rc_basics_lab.plotting.style import (
    REFERENCE_COLOR,
    REFERENCE_DASHES,
    StyleContext,
    method_color,
)

GOUDARZI_TAPS_PER_TRAIN = 1810.0 / 2000.0
"""先行の動作点 ``k / n_train`` (NARMA10 で 1,810 タップ / 訓練 2,000 点)。

``docs/series/rc-basics-survey.md`` Q2 が記録している値。**タップ数そのものではなく
比を引く**のは、こちらの訓練長が違うので絶対値では重ならないからである。
"""


def _series(
    rows: Sequence[TapSweepRow], method: str
) -> tuple[list[float], list[float], list[float]]:
    """1手法の (k/n_train, NMSE 平均, s.d.) を k の昇順で返す。"""
    lags = sorted({row.n_lags for row in rows if row.method == method})
    ratios: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    for n_lags in lags:
        group = [row for row in rows if row.method == method and row.n_lags == n_lags]
        mean, std = mean_std([row.nmse for row in group])
        ratios.append(group[0].taps_per_train)
        means.append(mean)
        stds.append(std)
    return ratios, means, stds


def headline(rows: Sequence[TapSweepRow], style: StyleContext) -> str:
    """タイトルの結論文を**行から導く** (D-90 と同じ規律)。

    固定文にすると、掃引の範囲を変えて壊れ方が消えたときに図が静かに
    嘘をつく。OLS がリッジの何倍まで悪化したかを毎回数え直す。
    """
    worst = 1.0
    for n_lags in {row.n_lags for row in rows}:
        at_k = [row for row in rows if row.n_lags == n_lags]
        ridge, _ = mean_std([r.nmse for r in at_k if r.method == DELAY_LINE])
        ols, _ = mean_std([r.nmse for r in at_k if r.method == DELAY_LINE_OLS])
        if ridge > 0.0:
            worst = max(worst, ols / ridge)
    if worst < 1.5:
        return style.label(
            "この掃引の範囲では正則化の有無が成績を変えない",
            "regularisation does not change the error over this sweep",
        )
    return style.label(
        f"タップ数が訓練長に近づくと正則化なしだけが壊れる (最大 {worst:.1f} 倍)",
        "only the unregularised fit breaks as the taps approach the"
        f" training length (up to {worst:.1f}x)",
    )


def draw_narma10_taps_panel(
    axis: Axes, rows: Sequence[TapSweepRow], style: StyleContext
) -> None:
    """実験 3-C' (``k / n_train`` に対する NMSE) を1つの軸に描く。

    単独の figure をやめてパネルにしたのは FIG-12 / C-6 による。

    Args:
        axis: 描画先。
        rows: 掃引の行 (``run_narma10_tap_sweep`` の出力)。
        style: 配色・言語。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    if not rows:
        raise ValueError("rows が空です")
    for method in (DELAY_LINE, DELAY_LINE_OLS):
        ratios, means, stds = _series(rows, method)
        if not ratios:
            continue
        axis.errorbar(
            ratios,
            means,
            yerr=np.asarray(stds, dtype=np.float64),
            fmt="o-",
            capsize=4,
            color=method_color(method),
            label=style.label(*METHOD_LABELS[method]),
        )
    axis.axvline(
        GOUDARZI_TAPS_PER_TRAIN,
        color=REFERENCE_COLOR,
        dashes=REFERENCE_DASHES[0],
        label=cited_measurement(
            style.label(
                f"先行の動作点 k/n = {GOUDARZI_TAPS_PER_TRAIN:.2f}",
                f"prior operating point k/n = {GOUDARZI_TAPS_PER_TRAIN:.2f}",
            ),
            GOUDARZI_2014,
            style.label(
                "1,810 タップ / 訓練 2,000 点",
                "1,810 taps / 2,000 training points",
            ),
        ),
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    # 対数軸の自動範囲は誤差棒の上端を切ることがある (実測: k/n=0.94 の
    # OLS が枠外に出た)。壊れ方そのものが主題なので余白を明示する。
    axis.margins(y=0.15)
    axis.set_xlabel(
        style.label(
            "タップ数 / 訓練長 (k / n_train)",
            "taps per training point (k / n_train)",
        )
    )
    n_replicates = len({row.replicate for row in rows})
    axis.set_ylabel(
        style.label(
            f"NMSE (テスト区間・{n_replicates}レプリケートの平均±標準偏差)",
            f"NMSE (test split, mean +- s.d. of {n_replicates} replicates)",
        )
    )
    axis.legend(loc="best", fontsize=8)
    axis.set_title(headline(rows, style), fontsize=9)


__all__ = ["GOUDARZI_TAPS_PER_TRAIN", "draw_narma10_taps_panel", "headline"]

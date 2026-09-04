"""記事04の図5枚 (実験 4-A / 4-B / 4-C / 4-D).

  (受け入れ条件3 の前半: **差が小さい**ことを見せる図)。
- ``plot_freerun_attractor``: **記事の主図**。自走軌道と真の軌道の位相図の
  重ね描き (Lorenz は (x, z)、1変数系は遅延座標埋め込み)。
- ``plot_valid_time``: 有効予測時間 (Lyapunov 時間で正規化) のシード分布。
  対照 (線形・遅延線) も同じ軸に載せる (受け入れ条件3 の後半)。
- ``plot_stability_map``: 状態ノイズ量ごとに (rho x リーク率) の3態マップを
  並べ、右端に 4-D の容量 (MC) と有効予測時間の関係を置く (受け入れ条件4)。
- ``plot_freerun_stats``: 長時間自走後のリターンマップとパワースペクトルの
  比較 (受け入れ条件5)。

``figures.py`` / ``figures_esp.py`` / ``figures_capacity.py`` と同じ規律に従う:
pyplot を使わず ``Figure`` + ``FigureCanvasAgg`` を直接組み、描画設定は
``matplotlib.rc_context`` で描画中だけ一時適用する (F-1-008)。ラベルは必ず
``style.label(ja, en)`` を通す (D-10)。

**図は成果物 CSV の行だけを読む** (仕様 §5 禁止する構造7)。位相図・リターン
マップ・スペクトルのように「配列」で表現される量も、``freerun_profile.csv`` に
書き出したのと同じ長形式の行 (``FreeRunProfileRow``) から復元する。診断も実験も
ここでは1回も走らせない。

**ギリシャ文字は書かない**: ruff の RUF001/RUF002 が ASCII と紛らわしい文字を
弾くため、ソース中では ``rho`` と綴る (02・03 の図と同じ)。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from rc_basics_lab.experiment.attractor import (
    REGIME_ATTRACTOR,
    REGIME_DIVERGED,
    REGIME_PERIODIC,
)
from rc_basics_lab.experiment.freerun import (
    KIND_PHASE,
    KIND_RETURN_MAP,
    KIND_SPECTRUM,
    SOURCE_FREERUN,
    SOURCE_TRUTH,
)
from rc_basics_lab.experiment.freerun_rows import (
    FreeRunProfileRow,
    FreeRunRow,
)
from rc_basics_lab.experiment.runner import DELAY_LINE, ESN_METHOD, LINEAR, ResultRow
from rc_basics_lab.plotting.freerun_grids import (
    label_of,
    mean_std,
    profile_points,
    tasks_of,
)
from rc_basics_lab.plotting.freerun_headlines import (
    LITERATURE_VALID_TIME,
    LITERATURE_VALID_TIME_CONDITIONS,
    valid_time_headline,
)
from rc_basics_lab.plotting.labels import (
    GAUTHIER_2021,
    METHOD_LABELS,
    cited_measurement,
)
from rc_basics_lab.plotting.layout import label_panels, legend_below, wrapped_note
from rc_basics_lab.plotting.style import (
    METHOD_COLORS,
    REFERENCE_COLOR,
    REFERENCE_DASHES,
    StyleContext,
    add_provenance,
    method_color,
    rc_context_for,
    require_rows,
)
from rc_basics_lab.plotting.style import new_figure as _new_figure
from rc_basics_lab.plotting.style import save_png as _save

TASK_LABELS: dict[str, tuple[str, str]] = {
    "lorenz": ("Lorenz", "Lorenz"),
    "mackey_glass": ("Mackey-Glass", "Mackey-Glass"),
}
"""課題名の表示。"""

REGIME_LABELS: dict[str, tuple[str, str]] = {
    REGIME_DIVERGED: ("発散", "diverged"),
    REGIME_PERIODIC: ("周期軌道", "periodic"),
    REGIME_ATTRACTOR: ("アトラクタ再現", "attractor"),
}
"""3態の表示 (``REGIMES`` の全要素を持つ。欠けたら描く前に落とす)。"""

CENSORED_EDGE_COLOR = "#b2182b"
"""打ち切られた観測の縁の色 (D-43。参照線の色とは別にする)。"""

REGIME_COLORS: dict[str, str] = {
    REGIME_DIVERGED: "#b2182b",
    REGIME_PERIODIC: "#2166ac",
    REGIME_ATTRACTOR: "#1a9850",
}
"""3態の色 (発散=赤 / 周期=青 / 再現=緑)。"""

REGIME_MARKERS: dict[str, str] = {
    REGIME_DIVERGED: "X",
    REGIME_PERIODIC: "^",
    REGIME_ATTRACTOR: "s",
}
"""3態のマーカー形状 (発散=x印 / 周期=三角 / 再現=四角)。

**色だけで符号化しない** (FIG-18)。赤・青・緑は色覚多様性で最も区別が難しい
組み合わせで、3態マップはこの3色だけで塗られていた。形を併用すれば、色が
区別できなくても読める。凡例も同じ形で出す。
"""

SOURCE_STYLE: dict[str, tuple[str, str]] = {
    SOURCE_TRUTH: ("#444444", "真の軌道"),
    SOURCE_FREERUN: (METHOD_COLORS[ESN_METHOD], "自走 (ESN)"),
}
"""軌道の源の色とラベル。

**自走は ESN 色 (緑) にする。** かつては橙 (#d95f02) で、同じ図の 4-A パネル
では橙が Mackey-Glass を指しており、**1枚の図の中で橙の意味が2つあった**。
自走は ESN の自走なので、連載共通の手法色 (D-85) をそのまま使えば衝突が消え、
「どの手法の自走か」もラベルを読まずに分かる。
"""
"""重ね描きの色と日本語ラベル (英語は ``_source_label``)。"""

TASK_COLORS: dict[str, str] = {"lorenz": "#d95f02", "mackey_glass": "#7570b3"}
"""4-A パネルで**課題**を区別する色 (``METHOD_COLORS`` とは別の次元)。

色を指定していなかったため既定循環色 (青・橙) が当たり、同じ図の位相図で橙が
「自走」を指していたのと衝突していた。形も変える (FIG-18)。
"""

TASK_MARKERS: dict[str, str] = {"lorenz": "o", "mackey_glass": "s"}
"""4-A パネルで課題を区別するマーカー形状。"""

_SOURCE_LABELS_EN: dict[str, str] = {
    SOURCE_TRUTH: "true trajectory",
    SOURCE_FREERUN: "free run (ESN)",
}


def source_label(source: str, style: StyleContext) -> str:
    """出典 (真の軌道 / 予測など) の表示ラベル。

    ``SOURCE_STYLE`` / ``_SOURCE_LABELS_EN`` が 04 固有の定数なので、
    ``freerun_grids`` へは出さずここに残す。
    """
    return style.label(SOURCE_STYLE[source][1], _SOURCE_LABELS_EN[source])


def _draw_onestep_panel(
    axis: Axes, rows: Sequence[ResultRow], style: StyleContext
) -> None:
    """4-A (教師強制の1ステップ先) を **位相図と同じ figure のパネル**に描く。

    単独の figure にしないのは FIG-12 による。点は 6 個しかなく、単独図では
    面積の9割が空白になっていた。**「教師強制では差が出ない」(4-A) と
    「自走にすると差が出る」(4-B) は一つの主張の表と裏**なので、並べたほうが
    読み手の往復も減る。

    Args:
        axis: 描画先。
        rows: ``onestep.csv`` と同じ行。
        style: 配色・言語。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    require_rows(rows)
    tasks = tasks_of(rows)
    methods = [LINEAR, DELAY_LINE, ESN_METHOD]
    positions = np.arange(len(methods), dtype=np.float64)
    for offset, task in enumerate(tasks):
        stats = [
            mean_std(
                [row.nrmse for row in rows if row.task == task and row.method == method]
            )
            for method in methods
        ]
        means = np.asarray([mean for mean, _ in stats], dtype=np.float64)
        stds = np.asarray([std for _, std in stats], dtype=np.float64)
        axis.errorbar(
            positions + 0.12 * (offset - 0.5 * (len(tasks) - 1)),
            means,
            yerr=np.vstack([np.minimum(stds, means * 0.999), stds]),
            fmt=TASK_MARKERS.get(task, "o"),
            color=TASK_COLORS.get(task, REFERENCE_COLOR),
            capsize=5,
            label=label_of(TASK_LABELS, task, style),
        )
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels([label_of(METHOD_LABELS, method, style) for method in methods])
    axis.set_xlim(-0.5, len(methods) - 0.5)
    n_replicates = len({row.replicate for row in rows})
    axis.set_title(
        style.label(
            "4-A: 教師強制では遅延線と ESN に差が出ない",
            "4-A: with teacher forcing the delay line and the ESN"
            "\nare indistinguishable",
        )
    )
    axis.set_ylabel(
        style.label(
            f"NRMSE (テスト区間・{n_replicates}レプリケートの平均±標準偏差)",
            f"NRMSE (test split, mean +- s.d. of {n_replicates} replicates)",
        )
    )
    axis.legend(loc="best", fontsize=8)


def plot_freerun_attractor(
    rows: Sequence[FreeRunProfileRow],
    path: Path,
    *,
    onestep_rows: Sequence[ResultRow],
    style: StyleContext,
) -> Path:
    """**記事の主図**: 自走軌道と真の軌道の位相図 + 4-A のスカラー比較。

    Lorenz は (x, z) 平面、1変数系は遅延座標埋め込み (遅延は真の軌道の自己
    相関から決めた1個を両者に使う)。図は ``freerun_profile.csv`` の行だけを
    読み、軌道を作り直さない。

    最後のパネルに 4-A (教師強制の1ステップ先) を同居させる (FIG-12)。
    **かつては単独の figure だったが、点が 6 個しかなく面積の9割が空白だった。**
    「教師強制では差が出ない」と「自走にすると差が出る」は一つの主張の表と裏で、
    並べて初めて対比になる。

    Args:
        rows: ``freerun_profile.csv`` と同じ行。
        path: 出力先の PNG。
        onestep_rows: ``onestep.csv`` と同じ行 (最後のパネルに使う)。
        style: 配色・言語・commit。

    Returns:
        書き出した PNG のパス。

    Raises:
        ValueError: 位相図の行が1つも無い場合、``onestep_rows`` が空の場合。
    """
    tasks = tasks_of(rows)
    if not tasks:
        raise ValueError("profile 行が空です")
    require_rows(onestep_rows)
    with rc_context_for(style):
        # パネルが1つ増えるぶん幅も増やす。1枚あたりの幅を保たないと
        # 位相図が潰れ、蝶形が読めなくなる (FIG-13 の上限 3.2:1 には収まる)。
        figure = _new_figure(
            5.0 * (len(tasks) + 1), 7.2
        )  # 1-6: Zenn の本文幅 700px で潰れないよう比を 2.0 前後に抑える
        axes = np.atleast_1d(figure.subplots(1, len(tasks) + 1))
        label_panels(list(axes), style=style)
        drawn = 0
        for axis, task in zip(axes[: len(tasks)], tasks, strict=True):
            for source in (SOURCE_TRUTH, SOURCE_FREERUN):
                points = profile_points(rows, task, KIND_PHASE, source)
                if points.shape[0] == 0:
                    continue
                # 線でつなぐ。**点だけで描くと蝶形の2枚翅が雲に潰れる** (実測)。
                # 間引き後も隣接点の間隔は 0.05 時間単位しかなく折れ線は軌道をなぞる。
                axis.plot(
                    points[:, 0],
                    points[:, 1],
                    linewidth=0.5,
                    alpha=0.8,
                    color=SOURCE_STYLE[source][0],
                    label=source_label(source, style),
                )
                drawn += 1
            axis.set_title(label_of(TASK_LABELS, task, style))
            axis.set_xlabel(
                style.label("第1成分 (標準化)", "component 1 (standardized)")
            )
            axis.set_ylabel(
                style.label("第2成分 / 遅延座標", "component 2 / delay coordinate")
            )
            # アトラクタの**形**を見せる図なので縦横を等尺にする。歪めると
            # 蝶形かどうかを目で判定できない。
            axis.set_aspect("equal", adjustable="datalim")
        if drawn == 0:
            raise ValueError("位相図に描く点がありません")
        _draw_onestep_panel(axes[len(tasks)], onestep_rows, style)
        # 同じ凡例が位相図の枚数ぶん繰り返されていたので図の外へ1つに統合する。
        legend_below(figure, list(axes), style=style, ncol=4)
        figure.suptitle(
            style.label(
                "実験 4-A / 4-B: 教師強制では遅延線と ESN に差が出ないが、"
                "入力を切ると ESN だけがアトラクタの形を保つ",
                "Experiments 4-A / 4-B: teacher forcing leaves the delay line"
                " and the ESN indistinguishable, but with the input switched"
                " off only the ESN keeps the shape of the attractor",
            )
        )
        # 位相図はレプリケート0の1本、4-A のパネルは全レプリケートの平均である。
        # ``rows`` だけを渡すと脚注が「1 rep」になり、右のパネルを取り違える。
        onestep_replicates = len({row.replicate for row in onestep_rows})
        conditions = (
            f"tasks = {'/'.join(tasks)}, 4-A panel = {onestep_replicates} replicates"
        )
        add_provenance(figure, conditions, rows, style=style)
        return _save(figure, path)


def plot_valid_time(
    rows: Sequence[FreeRunRow], path: Path, *, style: StyleContext
) -> Path:
    """有効予測時間 (Lyapunov 時間で正規化) のシード分布 (受け入れ条件2 / 3)。

    横軸が手法、縦軸が ``valid_time_lyapunov``。シードごとの点をそのまま重ねる
    (箱ひげだけにすると10本という本数が図から読めない)。**打ち切られた行は
    別のマーカー**で描く —— 打ち切りを普通の観測値として描くと分布の右端が
    実際より小さく見える (D-43)。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    require_rows(rows)
    # **lambda_max を数値推定してある系だけを描く** (D-42 / D-43)。04 が
    # 推定しているのは Lorenz だけで、Mackey-Glass の行は Lyapunov 列が nan
    # である (推定していない量を他の系の値で埋めない)。縦軸が Lyapunov 時間の
    # この図に nan の系を並べると、空のパネルが「有効予測時間が 0」に見える。
    normalized = [row for row in rows if math.isfinite(row.lyapunov_time)]
    if not normalized:
        raise ValueError("lyapunov_time が有限な行がありません")
    tasks = tasks_of(normalized)
    rows = normalized
    methods = [LINEAR, DELAY_LINE, ESN_METHOD]
    with rc_context_for(style):
        figure = _new_figure(6.4 * len(tasks), 5.0)
        axes = np.atleast_1d(figure.subplots(1, len(tasks), sharey=True))
        label_panels(list(axes), style=style)
        for axis, task in zip(axes, tasks, strict=True):
            _valid_time_panel(axis, rows, task, methods, style)
        axes[0].set_ylabel(
            style.label(
                "有効予測時間 [Lyapunov 時間]", "valid prediction time [1 / lambda_max]"
            )
        )
        axis.axhline(
            LITERATURE_VALID_TIME,
            color=REFERENCE_COLOR,
            dashes=REFERENCE_DASHES[0],
            label=cited_measurement(
                style.label(
                    f"文献の有効予測時間 ~{LITERATURE_VALID_TIME:g} Lyapunov 時間",
                    f"literature valid time ~{LITERATURE_VALID_TIME:g} Lyapunov times",
                ),
                GAUTHIER_2021,
                style.label(*LITERATURE_VALID_TIME_CONDITIONS),
            ),
        )
        # 1 行の凡例が軸幅いっぱいに広がっていたので図の外へ出す。
        legend_below(figure, list(axes), style=style, ncol=2)
        figure.suptitle(
            style.label(
                f"実験 4-B: {valid_time_headline(rows, style)}",
                f"Experiment 4-B: {valid_time_headline(rows, style)}",
            )
        )
        # **折り返してから渡す**。保存は bbox_inches="tight" なので、軸より横に
        # 長い supxlabel があると tight bbox がその幅まで広がり、軸が中央に
        # 取り残される (実測: 軸 1000px に対し canvas 2883px だった)。
        # matplotlib は Markdown を解釈しないので ** は書かない (そのまま出る)。
        figure.supxlabel(
            wrapped_note(
                style.label(
                    "注: lambda_max の数値推定は Lorenz だけ (D-42)。"
                    "参照線の原典は Lyapunov 時間を 1.1 としており、"
                    "こちらの数値推定 (1.09) と実質一致するため単位はそろっている。"
                    "ただし原典は「forecasts well」と定性的に述べており、"
                    "こちらの閾値 0.4 と同じ基準ではない (D-102)。",
                    "Note: lambda_max is estimated only for Lorenz (D-42)."
                    " The cited work uses a Lyapunov time of 1.1, matching our"
                    " numerical estimate (1.09), so the units agree. But it states"
                    " the horizon qualitatively, which is not the same criterion as"
                    " our threshold of 0.4 (D-102).",
                )
            ),
            fontsize=8,
        )
        conditions = (
            f"free_run = {rows[0].free_run_steps} steps,"
            f" threshold = {rows[0].valid_time_threshold:g}"
        )
        add_provenance(figure, conditions, rows, style=style)
        return _save(figure, path)


def _valid_time_panel(
    axis: Axes,
    rows: Sequence[FreeRunRow],
    task: str,
    methods: Sequence[str],
    style: StyleContext,
) -> None:
    """1課題ぶんの有効予測時間パネル (点 + 中央値の横線)。"""
    generator = np.random.default_rng(0)
    for position, method in enumerate(methods):
        selected = [row for row in rows if row.task == task and row.method == method]
        if not selected:
            continue
        values = np.asarray(
            [row.valid_time_lyapunov for row in selected], dtype=np.float64
        )
        jitter = generator.uniform(-0.12, 0.12, values.size)
        censored = np.asarray([row.valid_time_censored for row in selected], dtype=bool)
        axis.scatter(
            position + jitter[~censored],
            values[~censored],
            s=28,
            color=method_color(method),
            alpha=0.8,
        )
        if bool(np.any(censored)):
            axis.scatter(
                position + jitter[censored],
                values[censored],
                s=44,
                marker="^",
                facecolors="none",
                edgecolors=CENSORED_EDGE_COLOR,
                label=style.label("打ち切り (自走長に到達)", "censored (run ended)"),
            )
        median = float(np.median(values))
        axis.plot([position - 0.25, position + 0.25], [median, median], color="black")
        # 中央値ラベルは**右へ**逃がす。真上に置くと文献の参照線 (5.0) や
        # 点群そのものと重なる (実測: 4.83 が参照線に、0.18 が点群に被った)。
        axis.annotate(
            f"{median:.2f}",
            (position, median),
            textcoords="offset points",
            xytext=(16, 0),
            ha="center",
            fontsize=8,
        )
    axis.set_xticks(np.arange(len(methods), dtype=np.float64))
    axis.set_xticklabels([label_of(METHOD_LABELS, method, style) for method in methods])
    axis.set_xlim(-0.5, len(methods) - 0.5)
    axis.set_title(label_of(TASK_LABELS, task, style))
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles[:1], labels[:1], loc="best", fontsize=8)


def _annotate_distance(
    axis: Axes,
    rows: Sequence[FreeRunRow],
    task: str,
    column: str,
    style: StyleContext,
) -> None:
    """自走とシャッフル代替の距離をパネルの隅に注記する (D-46)。

    レプリケートの中央値を使う。1本だけ選ぶと、たまたま良い/悪い走りが
    そのまま結論に見える。

    Args:
        axis: 注記を置く軸。
        rows: ``freerun.csv`` と同じ行。
        task: 対象の課題名。
        column: ``"return_map_distance"`` か ``"spectrum_distance"``。
        style: 描画コンテキスト。
    """
    selected = [row for row in rows if row.task == task and row.method == ESN_METHOD]
    values = [float(getattr(row, column)) for row in selected]
    surrogate = [float(getattr(row, f"{column}_surrogate")) for row in selected]
    finite = [value for value in values if math.isfinite(value)]
    finite_surrogate = [value for value in surrogate if math.isfinite(value)]
    if not finite or not finite_surrogate:
        return
    count = len(finite)  # 描画は1本、注記は全本の中央値。注記に本数と中央値を明記する
    axis.text(
        0.03,
        0.97,
        style.label(
            f"自走 {float(np.median(finite)):.3f}"
            f" / 代替 {float(np.median(finite_surrogate)):.3f}"
            f" ({count} 本の中央値)",
            f"free run {float(np.median(finite)):.3f}"
            f" / surrogate {float(np.median(finite_surrogate)):.3f}"
            f" (median of {count})",
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "lw": 0.0},
    )


def plot_freerun_stats(
    rows: Sequence[FreeRunProfileRow],
    distance_rows: Sequence[FreeRunRow],
    path: Path,
    *,
    style: StyleContext,
) -> Path:
    """実験 4-B: 長時間統計の比較 (リターンマップ + パワースペクトル)。

    上段がリターンマップ ``(z_n, z_(n+1))``、下段が正規化パワースペクトル
    (対数)。どちらも真の軌道と自走を重ねる。

    **シャッフル代替との距離を各パネルに注記する** (D-46)。見出しは「代替より
    はるかに真の軌道へ近い」と主張しているのに、以前は代替がどのパネルにも
    描かれておらず**図が主張を裏づけていなかった**。代替の軌道そのものを第3の
    系統として重ねると点が三重になって読めなくなるので、比較の結論だけを
    数値で置く。値は ``freerun.csv`` の ``*_distance`` 列 (図は距離を作らない)。

    Raises:
        ValueError: 描く点が1つも無い場合。
    """
    tasks = tasks_of(rows)
    if not tasks:
        raise ValueError("profile 行が空です")
    with rc_context_for(style):
        figure = _new_figure(5.4 * len(tasks), 8.0)
        axes = np.atleast_2d(figure.subplots(2, len(tasks), squeeze=False))
        drawn = 0
        for column, task in enumerate(tasks):
            for source in (SOURCE_TRUTH, SOURCE_FREERUN):
                points = profile_points(rows, task, KIND_RETURN_MAP, source)
                if points.shape[0] > 0:
                    axes[0][column].scatter(
                        points[:, 0],
                        points[:, 1],
                        s=12,
                        alpha=0.7,
                        color=SOURCE_STYLE[source][0],
                        label=source_label(source, style),
                    )
                    drawn += 1
                spectrum = profile_points(rows, task, KIND_SPECTRUM, source)
                if spectrum.shape[0] > 0:
                    axes[1][column].plot(
                        spectrum[:, 0],
                        np.maximum(spectrum[:, 1], 1.0e-12),
                        linewidth=1.0,
                        color=SOURCE_STYLE[source][0],
                        label=source_label(source, style),
                    )
                    drawn += 1
            axes[0][column].set_title(label_of(TASK_LABELS, task, style))
            axes[0][column].set_xlabel(style.label("極大値 z_n", "maximum z_n"))
            axes[0][column].set_ylabel(
                style.label("次の極大値 z_(n+1)", "next maximum z_(n+1)")
            )
            # リターンマップは形を見せる図なので縦横を等尺にする。歪めると
            # 「同じ曲線に乗っているか」が目で判定できない。
            axes[0][column].set_aspect("equal", adjustable="datalim")
            _annotate_distance(
                axes[0][column],
                distance_rows,
                task,
                "return_map_distance",
                style,
            )
            axes[1][column].set_yscale("log")
            axes[1][column].set_xlabel(style.label("周波数 [1/時間]", "frequency"))
            axes[1][column].set_ylabel(style.label("正規化パワー", "normalized power"))
            _annotate_distance(
                axes[1][column], distance_rows, task, "spectrum_distance", style
            )
        if drawn == 0:
            raise ValueError("長時間統計に描く点がありません")
        # 4 パネルすべてに同じ凡例が出ていたので1つに統合する。
        label_panels(list(axes.ravel()), style=style)
        legend_below(figure, list(axes.ravel()), style=style, ncol=2)
        figure.suptitle(
            style.label(
                "実験 4-B: 長時間自走後もリターンマップとスペクトルは"
                "シャッフル代替よりはるかに真の軌道へ近い",
                "Experiment 4-B: after a long free run the return map and the"
                " spectrum stay far closer to the truth than the shuffled"
                " surrogate",
            )
        )
        conditions = f"tasks = {'/'.join(tasks)}"
        add_provenance(figure, conditions, rows, style=style)
        return _save(figure, path)


__all__ = [
    "METHOD_LABELS",
    "REGIME_COLORS",
    "REGIME_LABELS",
    "REGIME_MARKERS",
    "TASK_COLORS",
    "TASK_LABELS",
    "TASK_MARKERS",
    "plot_freerun_attractor",
    "plot_freerun_stats",
    "plot_valid_time",
    "profile_points",
]

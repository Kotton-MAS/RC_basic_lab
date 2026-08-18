"""記事03の図4枚 (実験 3-A / 3-B / 3-B').

- ``plot_mc_sweep``: rho x リーク率 に対する線形メモリ容量と、遅延プロファイル
  の伸び (受け入れ条件1)。左は上限線 y=N つきの ``mc_total``、右は代表リーク率
  での遅延プロファイルを rho 別に重ねる。
- ``plot_ipc_profile``: (次数 x 遅延) の容量ヒートマップを rho 別のパネルに
  並べる (受け入れ条件4)。
- ``plot_memory_nonlinearity``: ``ipc_linear`` と ``ipc_nonlinear`` の積み上げで
  線形/非線形の配分の移動を見せる (受け入れ条件4)。
- ``plot_ipc_conservation``: x=N, y=``ipc_total`` に**傾き1の対角線 y=N** を
  重ねる (受け入れ条件2)。この対角線は図の主張そのものなので、線が実際に
  描かれていることを ``test_conservation_figure_draws_the_bound_line`` が固定する。

``figures.py`` / ``figures_esp.py`` と同じ規律に従う: pyplot を使わず ``Figure``
+ ``FigureCanvasAgg`` を直接組み、描画設定は ``matplotlib.rc_context`` で描画中
だけ一時適用する (F-1-008)。ラベルは必ず ``style.label(ja, en)`` を通す (D-10)。

**ギリシャ文字は書かない**: ruff の RUF001/RUF002 が ASCII と紛らわしい文字を
弾くため、ソース中では ``rho`` / ``sigma_u`` と綴る (02 の図と同じ)。

**配列ではなく長形式の行を読む** (D-38): 遅延プロファイルとヒートマップは
``CapacityProfileRow`` (= ``capacity_profile.csv`` の行) から復元する。
長形式には**しきい値後の容量が厳密に正のセルだけ**が在るので、格子に戻すときは
**欠けているセルを 0 で埋める** (「未測定」と「容量 0」を区別する必要は無い ——
しきい値を超えなかったセルの容量は 0 である)。診断はここでは一切走らせない。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import QuadMesh
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from rc_basics_lab.experiment.capacity import (
    DIAGNOSTIC_IPC,
    DIAGNOSTIC_MC,
    CapacityProfileRow,
    CapacityRow,
)
from rc_basics_lab.plotting.style import StyleContext, rc_params_for
from rc_basics_lab.types import FloatArray

BOUND_MARGIN = 0.1
"""上限線 y=N (対角線) を格子の外へ伸ばす割合。

``n_units`` が1点しかない縮退ケース (縮小設定) でも**線**として描けるように
両端へ余白を取る。1点だけを結ぶと長さ 0 の線分になり、図の主張である
「傾き1の対角線」が消える。
"""

_MIN_COLOR_MAX = 1.0e-12
"""ヒートマップの配色上限の下駄。

全セルが 0 の縮退ケースで ``Normalize(0, 0)`` を作るとゼロ除算になるため、
上限が正でないときだけ 1.0 に読み替える (下駄そのものは描画に出ない)。
"""

_HEATMAP_CMAP = "viridis"
_STACK_COLORS = ("tab:blue", "tab:orange")
"""積み上げ棒 (線形, 非線形) の色。"""


def _new_figure(width: float, height: float) -> Figure:
    """constrained layout の Figure を作る (``figures.py`` と同じ規律)。"""
    figure = Figure(figsize=(width, height))
    figure.set_layout_engine("constrained")
    return figure


def _save(figure: Figure, path: Path) -> Path:
    """Agg キャンバスで PNG を書く (ディスプレイに依存しない)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)
    figure.savefig(path, format="png")
    return path


def _unique_sorted(values: Sequence[float]) -> tuple[float, ...]:
    """重複を除いて昇順に並べる (格子の軸を行から復元する)。"""
    return tuple(sorted(set(values)))


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    """レプリケート平均と標準偏差 (母標準偏差、``ddof=0``)。

    レプリケートが1本しかない縮退ケースでも落ちないよう ``np.std`` を使う
    (``statistics.stdev`` は n=1 で例外になる)。
    """
    if not values:
        return math.nan, math.nan
    array: FloatArray = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array))


def _n_replicates(rows: Sequence[CapacityRow]) -> int:
    """行に現れるレプリケートの本数。

    長形式の行は**正値セルだけ**なので、容量が 0 のレプリケートは1行も
    書かれていない。平均を「在る行の数」で割ると、0 のレプリケートを無視した
    過大評価になる。分母はここ (``capacity.csv`` 側の行) から取る。
    """
    return max(len({row.replicate for row in rows}), 1)


def _for_rows(
    profile: Sequence[CapacityProfileRow], rows: Sequence[CapacityRow]
) -> tuple[CapacityProfileRow, ...]:
    """``rows`` と同じ実験ラベルの長形式の行だけに絞る。

    3-A と 3-B は (rho, leak_rate) の格子が一部重なるため、実験ラベルで
    絞らずに読むと別の実験のセルが図に混ざりうる。呼び出し側の絞り込み漏れを
    ここで塞ぐ (絞り込み済みの並びを渡しても結果は変わらない)。
    """
    experiments = {row.experiment for row in rows}
    return tuple(row for row in profile if row.experiment in experiments)


def representative_leak_rate(rows: Sequence[CapacityRow], key: str) -> float:
    """パネルに出す代表リーク率を選ぶ (``key`` の平均が最大のもの)。

    3-A の右パネルと 3-B のヒートマップは1つのリーク率しか描けない。**総容量が
    最も大きい動作点**を代表にするのは、その図が見せたい構造 (プロファイルの
    伸び / 次数の配分) が最も読み取れる点だからである。同点のときは小さい方を
    採り、選択が行の並び順に依存しないようにする。

    Args:
        rows: 1実験ぶんの行。
        key: 比較に使う列名 (``"mc_total"`` / ``"ipc_total"``)。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    if not rows:
        raise ValueError("rows が空です")
    totals: dict[float, list[float]] = {}
    for row in rows:
        totals.setdefault(row.leak_rate, []).append(float(getattr(row, key)))
    return min(totals, key=lambda leak: (-float(np.mean(totals[leak])), leak))


def mc_profile_means(
    rows: Sequence[CapacityRow], profile: Sequence[CapacityProfileRow], leak_rate: float
) -> dict[float, FloatArray]:
    """MC の遅延プロファイルを rho ごとにレプリケート平均する (D-38 の長形式から)。

    長形式には正値セルしか無いので、``(n_delays,)`` の 0 配列に足し込んでから
    レプリケート数で割る。``n_delays`` は ``capacity.csv`` 側の列
    (``mc.max_delay``) を使う —— 長形式の最大遅延から決めると、条件によって
    横軸の長さが変わって rho 間の比較ができなくなる。
    """
    n_delays = max((row.n_delays for row in rows), default=0)
    divisor = float(_n_replicates(rows))
    means = {
        rho: np.zeros(n_delays, dtype=np.float64)
        for rho in _unique_sorted([row.rho for row in rows])
    }
    for row in _for_rows(profile, rows):
        if row.diagnostic != DIAGNOSTIC_MC or row.leak_rate != leak_rate:
            continue
        cells = means.get(row.rho)
        if cells is None or not 1 <= row.delay <= n_delays:
            continue
        cells[row.delay - 1] += row.capacity / divisor
    return means


def ipc_heatmap_means(
    rows: Sequence[CapacityRow], profile: Sequence[CapacityProfileRow], leak_rate: float
) -> dict[float, FloatArray]:
    """IPC の (次数 x 遅延) 容量を rho ごとにレプリケート平均する (D-38 の長形式から)。

    形は全パネルで共通の ``(max(n_degrees), 最大遅延)`` にそろえる。遅延の
    打ち切りは次数ごとに違う (本番は 60/20/10/6) ので、打ち切りの外は 0 のまま
    残り、ヒートマップの中に打ち切りの形がそのまま出る。
    """
    n_degrees = max((row.n_degrees for row in rows), default=1)
    cells = _for_rows(profile, rows)
    n_delays = max(
        (row.delay for row in cells if row.diagnostic == DIAGNOSTIC_IPC), default=1
    )
    divisor = float(_n_replicates(rows))
    means = {
        rho: np.zeros((n_degrees, n_delays), dtype=np.float64)
        for rho in _unique_sorted([row.rho for row in rows])
    }
    for row in cells:
        if row.diagnostic != DIAGNOSTIC_IPC or row.leak_rate != leak_rate:
            continue
        grid = means.get(row.rho)
        if grid is None or not 1 <= row.degree <= n_degrees:
            continue
        grid[row.degree - 1, row.delay - 1] += row.capacity / divisor
    return means


def conservation_bound(units: Sequence[int]) -> tuple[FloatArray, FloatArray]:
    """上限線 ``y = N`` (傾き1の対角線) の端点を返す (受け入れ条件2)。

    描画とテストが**同じ1か所**からこの座標を取る。図の主張は
    「``ipc_total`` はこの対角線を超えない」なので、線が消えたことを
    ``test_conservation_figure_draws_the_bound_line`` が検出できる必要がある。

    Raises:
        ValueError: ``units`` が空の場合。
    """
    if not units:
        raise ValueError("units が空です")
    low = min(units) * (1.0 - BOUND_MARGIN)
    high = max(units) * (1.0 + BOUND_MARGIN)
    edge: FloatArray = np.array([low, high], dtype=np.float64)
    return edge, edge


# --- 3-A: 線形メモリ容量の掃引 ---------------------------------------------


def _plot_mc_total_panel(
    axis: Axes, rows: Sequence[CapacityRow], style: StyleContext
) -> None:
    """左パネル: rho x リーク率 の ``mc_total`` と上限線 y=N。

    縦軸を対数にするのは、上限線 (本番 N=200) と実測 (10〜36) を1枚に載せる
    ためである。線形軸だと上限線を入れた瞬間に実測の差が潰れて、受け入れ条件1
    の「rho とともに伸びる」が読めなくなる。
    """
    rhos = _unique_sorted([row.rho for row in rows])
    leaks = _unique_sorted([row.leak_rate for row in rows])
    colors = matplotlib.colormaps["viridis"](np.linspace(0.0, 0.9, len(leaks)))
    for index, leak in enumerate(leaks):
        stats = [
            _mean_std(
                [row.mc_total for row in rows if row.rho == rho and row.leak_rate == leak]
            )
            for rho in rhos
        ]
        axis.errorbar(
            list(rhos),
            [mean for mean, _ in stats],
            yerr=[std for _, std in stats],
            fmt="o-",
            capsize=4,
            color=colors[index],
            label=style.label(f"a = {leak:g}", f"a = {leak:g}"),
        )
    for n_units in sorted({row.n_units for row in rows}):
        axis.axhline(
            float(n_units),
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label=style.label(
                f"上限 MC <= N = {n_units}", f"bound MC <= N = {n_units}"
            ),
        )
    axis.set_yscale("log")
    axis.set_xlabel(style.label("スペクトル半径 rho", "spectral radius rho"))
    axis.set_ylabel(
        style.label(
            "MC_total (レプリケート平均±s.d.)", "MC_total (mean +- s.d. over reps)"
        )
    )
    axis.set_title(
        style.label(
            "線形メモリ容量と上限 N", "Linear memory capacity and the bound N"
        ),
        fontsize=10,
    )
    axis.legend(loc="lower right", fontsize=8, ncols=2)


def _plot_mc_profile_panel(
    axis: Axes,
    rows: Sequence[CapacityRow],
    profile: Sequence[CapacityProfileRow],
    leak_rate: float,
    style: StyleContext,
) -> None:
    """右パネル: 代表リーク率での遅延プロファイルを rho 別に重ねる。

    横軸を対数にすると「rho を上げるとプロファイルが右へ伸びる」が形として
    読める。各 rho の容量重心 (``mc_effective_delay``、受け入れ条件1 が測る量
    そのもの) を同色の縦点線で入れ、図と受け入れ条件の対応を明示する。
    """
    means = mc_profile_means(rows, profile, leak_rate)
    rhos = tuple(means)
    colors = matplotlib.colormaps["viridis"](np.linspace(0.0, 0.9, max(len(rhos), 1)))
    for index, rho in enumerate(rhos):
        cells = means[rho]
        delays = np.arange(1, cells.shape[0] + 1, dtype=np.float64)
        axis.plot(
            delays,
            cells,
            color=colors[index],
            linewidth=1.4,
            label=style.label(f"rho = {rho:g}", f"rho = {rho:g}"),
        )
        effective, _ = _mean_std(
            [
                row.mc_effective_delay
                for row in rows
                if row.rho == rho and row.leak_rate == leak_rate
            ]
        )
        if math.isfinite(effective) and effective > 0.0:
            axis.axvline(effective, color=colors[index], linestyle=":", linewidth=1.0)
    axis.set_xscale("log")
    axis.set_xlabel(style.label("遅延 k [ステップ]", "delay k [steps]"))
    axis.set_ylabel(
        style.label(
            "遅延ごとの容量 (しきい値後)", "capacity per delay (after thresholding)"
        )
    )
    axis.set_title(
        style.label(
            f"遅延プロファイル (a = {leak_rate:g}、縦点線は容量重心)",
            f"Delay profile (a = {leak_rate:g}; dotted: centre of mass)",
        ),
        fontsize=10,
    )
    axis.legend(loc="upper right", fontsize=8, ncols=2)


def plot_mc_sweep(
    rows: Sequence[CapacityRow],
    profile: Sequence[CapacityProfileRow],
    path: Path,
    *,
    style: StyleContext,
) -> Path:
    """実験 3-A の図を書く (受け入れ条件1)。

    Args:
        rows: 3-A の行 (``capacity.csv`` と同じ)。
        profile: 3-A の長形式の行 (``capacity_profile.csv`` と同じ、D-38)。
        path: 出力先 PNG。
        style: ``setup_style()`` の戻り値 (ラベル言語の決定に使う)。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    if not rows:
        raise ValueError("rows が空です")
    leak_rate = representative_leak_rate(rows, "mc_total")
    with matplotlib.rc_context(rc_params_for(style)):  # type: ignore[arg-type]
        figure = _new_figure(12.0, 4.8)
        axes = figure.subplots(1, 2, squeeze=False)
        _plot_mc_total_panel(axes[0][0], rows, style)
        _plot_mc_profile_panel(axes[0][1], rows, profile, leak_rate, style)
        first = rows[0]
        figure.suptitle(
            style.label(
                "実験 3-A: スペクトル半径とリーク率が決める線形メモリ容量"
                f" (N = {first.n_units}, sigma_u = {first.sigma_u:g})",
                "Experiment 3-A: linear memory capacity versus rho and leak rate"
                f" (N = {first.n_units}, sigma_u = {first.sigma_u:g})",
            )
        )
        return _save(figure, path)


# --- 3-B: 次数 x 遅延 のヒートマップ ---------------------------------------


def _plot_heatmap_panel(
    axis: Axes,
    cells: FloatArray,
    rho: float,
    norm: Normalize,
    style: StyleContext,
    *,
    show_ylabel: bool,
) -> QuadMesh:
    """1つの rho ぶんの (次数 x 遅延) ヒートマップ (配色は呼び出し側と共通)。"""
    n_degrees, n_delays = cells.shape
    mesh = axis.pcolormesh(
        np.arange(n_delays + 1, dtype=np.float64) + 0.5,
        np.arange(n_degrees + 1, dtype=np.float64) + 0.5,
        cells,
        cmap=_HEATMAP_CMAP,
        norm=norm,
        shading="flat",
    )
    axis.set_yticks(np.arange(1, n_degrees + 1, dtype=np.float64))
    axis.set_xlabel(style.label("遅延 k [ステップ]", "delay k [steps]"))
    if show_ylabel:
        axis.set_ylabel(style.label("次数 d", "degree d"))
    axis.set_title(
        style.label(f"rho = {rho:g}", f"rho = {rho:g}"),
        fontsize=10,
    )
    axis.grid(visible=False)
    return mesh


def plot_ipc_profile(
    rows: Sequence[CapacityRow],
    profile: Sequence[CapacityProfileRow],
    path: Path,
    *,
    style: StyleContext,
) -> Path:
    """実験 3-B の (次数 x 遅延) ヒートマップを rho 別に並べる (受け入れ条件4)。

    パネルは代表リーク率 1本 x rho 4点。配色は全パネルで共通の上限を使う
    (パネルごとに正規化すると「rho を上げると非線形が減る」という主張が
    色の付け替えで消える)。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    if not rows:
        raise ValueError("rows が空です")
    leak_rate = representative_leak_rate(rows, "ipc_total")
    means = ipc_heatmap_means(rows, profile, leak_rate)
    rhos = tuple(means)
    ceiling = max((float(cells.max()) for cells in means.values()), default=0.0)
    norm = Normalize(vmin=0.0, vmax=max(ceiling, _MIN_COLOR_MAX))

    with matplotlib.rc_context(rc_params_for(style)):  # type: ignore[arg-type]
        figure = _new_figure(3.4 * len(rhos) + 1.2, 3.6)
        axes = figure.subplots(1, len(rhos), squeeze=False)
        meshes = [
            _plot_heatmap_panel(
                axes[0][index], means[rho], rho, norm, style, show_ylabel=index == 0
            )
            for index, rho in enumerate(rhos)
        ]
        figure.colorbar(
            meshes[-1],
            ax=list(axes[0]),
            label=style.label(
                "容量 (しきい値後・レプリケート平均)",
                "capacity (after thresholding, mean over reps)",
            ),
        )
        first = rows[0]
        figure.suptitle(
            style.label(
                "実験 3-B: 次数 x 遅延 で見た情報処理容量の配分"
                f" (a = {leak_rate:g}, N = {first.n_units})",
                "Experiment 3-B: information processing capacity by degree and delay"
                f" (a = {leak_rate:g}, N = {first.n_units})",
            )
        )
        return _save(figure, path)


# --- 3-B: 線形 / 非線形 の配分 ----------------------------------------------


def _plot_stack_panel(
    axis: Axes,
    rows: Sequence[CapacityRow],
    leak_rate: float,
    style: StyleContext,
    *,
    show_ylabel: bool,
) -> None:
    """1つのリーク率ぶんの積み上げ棒 (線形 + 非線形)。"""
    rhos = _unique_sorted([row.rho for row in rows])
    positions = np.arange(len(rhos), dtype=np.float64)
    selected = [
        [row for row in rows if row.rho == rho and row.leak_rate == leak_rate]
        for rho in rhos
    ]
    linear = [_mean_std([row.ipc_linear for row in group])[0] for group in selected]
    nonlinear = [
        _mean_std([row.ipc_nonlinear for row in group])[0] for group in selected
    ]
    total_std = [_mean_std([row.ipc_total for row in group])[1] for group in selected]
    axis.bar(
        positions,
        linear,
        width=0.6,
        color=_STACK_COLORS[0],
        label=style.label("線形 (次数1)", "linear (degree 1)"),
    )
    axis.bar(
        positions,
        nonlinear,
        width=0.6,
        bottom=linear,
        color=_STACK_COLORS[1],
        label=style.label("非線形 (次数2以上)", "nonlinear (degree >= 2)"),
    )
    axis.errorbar(
        positions,
        [a + b for a, b in zip(linear, nonlinear, strict=True)],
        yerr=total_std,
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    for position, low, high in zip(positions, linear, nonlinear, strict=True):
        total = low + high
        if total <= 0.0:
            continue
        axis.annotate(
            f"{high / total:.0%}",
            (position, total),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{rho:g}" for rho in rhos])
    axis.set_xlabel(style.label("スペクトル半径 rho", "spectral radius rho"))
    if show_ylabel:
        axis.set_ylabel(
            style.label(
                "容量 (レプリケート平均、上の数字は非線形の割合)",
                "capacity (mean over reps; label: nonlinear share)",
            )
        )
    axis.set_title(
        style.label(f"a = {leak_rate:g}", f"a = {leak_rate:g}"),
        fontsize=10,
    )
    axis.legend(loc="upper right", fontsize=8)


def plot_memory_nonlinearity(
    rows: Sequence[CapacityRow], path: Path, *, style: StyleContext
) -> Path:
    """線形容量と非線形容量の配分の移動を積み上げで描く (受け入れ条件4)。

    パネルはリーク率ごと、横軸は rho。棒の上の数値は非線形の割合で、
    「rho を上げると総容量のうち非線形が減る」という主張を数値でも読ませる
    (積み上げの高さだけだと、総容量の減少と配分の移動が分離できない)。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    if not rows:
        raise ValueError("rows が空です")
    leaks = _unique_sorted([row.leak_rate for row in rows])
    with matplotlib.rc_context(rc_params_for(style)):  # type: ignore[arg-type]
        figure = _new_figure(4.0 * len(leaks) + 1.0, 4.4)
        axes = figure.subplots(1, len(leaks), squeeze=False, sharey=True)
        for index, leak in enumerate(leaks):
            _plot_stack_panel(
                axes[0][index], rows, leak, style, show_ylabel=index == 0
            )
        first = rows[0]
        figure.suptitle(
            style.label(
                "実験 3-B: rho とリーク率で動く線形/非線形の配分"
                f" (N = {first.n_units}, sigma_u = {first.sigma_u:g})",
                "Experiment 3-B: how rho and the leak rate move the linear /"
                f" nonlinear split (N = {first.n_units}, sigma_u = {first.sigma_u:g})",
            )
        )
        return _save(figure, path)


# --- 3-B': 保存則 IPC_total <= N --------------------------------------------


def _draw_conservation_bound(
    axis: Axes, units: Sequence[int], style: StyleContext
) -> None:
    """上限線 y=N (傾き1の対角線) を描く (受け入れ条件2、**図の主張そのもの**)。

    ``conservation_bound`` が返す座標をそのまま描く。テストは同じ関数から
    座標を取り、この線が軸に実在することを確かめる。
    """
    x, y = conservation_bound(units)
    axis.plot(
        x,
        y,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=style.label(
            "上限 IPC_total = N (傾き1)", "bound IPC_total = N (slope 1)"
        ),
    )


def plot_ipc_conservation(
    rows: Sequence[CapacityRow], path: Path, *, style: StyleContext
) -> Path:
    """実験 3-B' の保存則 ``IPC_total <= N`` を描く (受け入れ条件2)。

    横軸が N、縦軸が ``ipc_total``、線は ``state_noise`` 別。**傾き1の対角線**
    が上限で、ノイズを入れると点が対角線から下へ離れる (ノイズがリザバーの
    自由度を潰し、線形読み出しで取り出せる容量が N に届かなくなる)。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    if not rows:
        raise ValueError("rows が空です")
    units = sorted({row.n_units for row in rows})
    noises = _unique_sorted([row.state_noise for row in rows])
    colors = matplotlib.colormaps["plasma"](np.linspace(0.0, 0.75, len(noises)))

    with matplotlib.rc_context(rc_params_for(style)):  # type: ignore[arg-type]
        figure = _new_figure(7.2, 5.0)
        axis = figure.subplots(1, 1)
        for index, noise in enumerate(noises):
            stats = [
                _mean_std(
                    [
                        row.ipc_total
                        for row in rows
                        if row.n_units == n_units and row.state_noise == noise
                    ]
                )
                for n_units in units
            ]
            axis.errorbar(
                [float(n_units) for n_units in units],
                [mean for mean, _ in stats],
                yerr=[std for _, std in stats],
                fmt="o-",
                capsize=4,
                color=colors[index],
                label=style.label(
                    f"状態ノイズ = {noise:g}", f"state noise = {noise:g}"
                ),
            )
        _draw_conservation_bound(axis, units, style)
        axis.set_xlabel(style.label("リザバーのユニット数 N", "reservoir size N"))
        axis.set_ylabel(
            style.label(
                "IPC_total (レプリケート平均±s.d.)",
                "IPC_total (mean +- s.d. over reps)",
            )
        )
        first = rows[0]
        figure.suptitle(
            style.label(
                "実験 3-B': 情報処理容量の保存則 IPC_total <= N"
                f" (rho = {first.rho:g}, a = {first.leak_rate:g})",
                "Experiment 3-B': the capacity bound IPC_total <= N"
                f" (rho = {first.rho:g}, a = {first.leak_rate:g})",
            )
        )
        axis.legend(loc="upper left", fontsize=8)
        return _save(figure, path)


__all__ = [
    "BOUND_MARGIN",
    "conservation_bound",
    "ipc_heatmap_means",
    "mc_profile_means",
    "plot_ipc_conservation",
    "plot_ipc_profile",
    "plot_mc_sweep",
    "plot_memory_nonlinearity",
    "representative_leak_rate",
]

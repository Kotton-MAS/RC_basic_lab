"""梯子 (3-T / 3-Th) の実行と成果物の書き出し.

``capacity_pipeline.py`` が上限 (600 行) に達したので分けた。役割としても、
梯子は ``make figures-03`` の予算の外で手動実行する補助実験であり、本番の
成果物 (``CAPACITY_ARTIFACTS``) には含まれない。

- 3-T (``capacity_topology.csv``): 交絡を1つずつ剥がす対照の梯子 (D-138)
- 3-Th (``capacity_topology_threshold.csv``): その結論が閾値の選び方で
  動かないことの実測 (D-143)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from statistics import mean

from rc_basics_lab.config import Capacity03Config
from rc_basics_lab.experiment.ladder_threshold import (
    LADDER_THRESHOLD_CSV_COLUMNS,
    LadderThresholdRow,
    run_ladder_threshold,
)
from rc_basics_lab.experiment.rows_csv import write_rows_csv
from rc_basics_lab.experiment.topology_ladder import (
    TOPOLOGY_LADDER_CSV_COLUMNS,
    TopologyLadderRow,
    run_topology_ladder,
)

logger = logging.getLogger(__name__)

CAPACITY_TOPOLOGY_CSV = "capacity_topology.csv"
"""3-T の成果物 (``results/03_capacity/`` 配下)。"""

CAPACITY_TOPOLOGY_THRESHOLD_CSV = "capacity_topology_threshold.csv"
"""3-Th の成果物 (``results/03_capacity/`` 配下)。"""

FIG_LADDER = "fig_topology_ladder.png"
"""3-T の図 (D-145)。CSV だけだと 1680 行を読まないと結論が出てこない。"""


def _ladder_group(row: TopologyLadderRow) -> tuple[str, float, str]:
    """log をまとめる単位 (掃引軸, その値, 水準)。

    掃引した値は**行の既存の列から読む** (``n_units`` を振ったなら
    ``n_units`` 列)。専用の値の列を作らないのは、CSV だけを見た人が
    「何を振ったのか」を設定と突き合わせずに読めるようにするためである。
    """
    if not row.sweep_axis:
        return ("", 0.0, row.level)
    return (row.sweep_axis, float(getattr(row, row.sweep_axis)), row.level)


def run_and_report_topology_ladder(config: Capacity03Config, out_dir: Path) -> Path:
    """対照の梯子を回し ``capacity_topology.csv`` に書く (3-T、D-138)。

    本体の成果物とは独立に走る (``CAPACITY_ARTIFACTS`` に含めない)。
    ``make ladder-03`` として手動実行する (``symmetry-03`` と同じ扱い)。

    水準ごとの平均を log に出す —— **CSV を開かなくても梯子の形が読める**
    ようにするためで、判定 (対応のある検定) は行を読む側の仕事である。
    """
    started = time.perf_counter()
    rows = run_topology_ladder(config)
    path = write_rows_csv(
        rows, out_dir / CAPACITY_TOPOLOGY_CSV, TOPOLOGY_LADDER_CSV_COLUMNS
    )
    # 作図層の import は関数本体に置く (D-53)。
    from rc_basics_lab.meta import git_commit
    from rc_basics_lab.plotting.figures_ladder import plot_ladder
    from rc_basics_lab.plotting.style import setup_style

    plot_ladder(rows, out_dir / FIG_LADDER, style=setup_style(commit=git_commit()))
    for key in dict.fromkeys(_ladder_group(row) for row in rows):
        selected = [row for row in rows if _ladder_group(row) == key]
        axis, value, level = key
        where = f"{axis}={value:g} " if axis else ""
        logger.info(
            "3-T %s%s: MC=%.3f IPC=%.3f (線形 %.3f / 非線形 %.3f) n=%d",
            where,
            level,
            mean(row.mc_total for row in selected),
            mean(row.ipc_total for row in selected),
            mean(row.ipc_linear for row in selected),
            mean(row.ipc_nonlinear for row in selected),
            len(selected),
        )
    logger.info(
        "対照の梯子: %d 行 / wall_time=%.2fs / 出力=%s",
        len(rows),
        time.perf_counter() - started,
        path,
    )
    return path


def run_and_report_ladder_threshold(config: Capacity03Config, out_dir: Path) -> Path:
    """閾値感度を回し ``capacity_topology_threshold.csv`` に書く (3-Th、D-143)。

    本体の成果物とは独立に走る。``make ladder-threshold-03`` として手動実行
    する (``ladder-03`` と同じ扱い)。

    **判定基準ごとに BA と ER の差を log に出す。** 見たいのは総容量の絶対値
    ではなく**順位が動くかどうか**なので、差のほうを出さないと CSV を開かない
    限り結論が読めない。
    """
    started = time.perf_counter()
    rows = run_ladder_threshold(config)
    path = write_rows_csv(
        rows, out_dir / CAPACITY_TOPOLOGY_THRESHOLD_CSV, LADDER_THRESHOLD_CSV_COLUMNS
    )
    for key in dict.fromkeys(_threshold_group(row) for row in rows):
        selected = [row for row in rows if _threshold_group(row) == key]
        mode, n_surrogates, quantile = key
        logger.info(
            "3-Th %s n=%d q=%g: BA-ER の MC=%+.3f IPC=%+.3f (対 %d)",
            mode,
            n_surrogates,
            quantile,
            _paired_gap(selected, "mc_total"),
            _paired_gap(selected, "ipc_total"),
            sum(1 for row in selected if row.level == "erdos_renyi"),
        )
    logger.info(
        "閾値感度: %d 行 / wall_time=%.2fs / 出力=%s",
        len(rows),
        time.perf_counter() - started,
        path,
    )
    return path


def _threshold_group(row: LadderThresholdRow) -> tuple[str, int, float]:
    """log をまとめる単位 (モード, サロゲート本数, 分位点)。"""
    return (row.threshold_mode, row.n_surrogates, row.surrogate_quantile)


def _paired_gap(rows: list[LadderThresholdRow], column: str) -> float:
    """同じ (グラフ, 重み) で対にした BA - ER の平均。

    対にしないと、グラフの実現値の分散が差に混ざる (D-134 と同じ理由)。
    """
    by_level = {
        (row.level, row.graph, row.replicate): float(getattr(row, column))
        for row in rows
    }
    gaps = [
        by_level[("barabasi_albert", graph, replicate)] - value
        for (level, graph, replicate), value in by_level.items()
        if level == "erdos_renyi" and ("barabasi_albert", graph, replicate) in by_level
    ]
    return mean(gaps) if gaps else float("nan")


__all__ = [
    "CAPACITY_TOPOLOGY_CSV",
    "CAPACITY_TOPOLOGY_THRESHOLD_CSV",
    "FIG_LADDER",
    "run_and_report_ladder_threshold",
    "run_and_report_topology_ladder",
]

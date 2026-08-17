"""1コマンドで5成果物を作る経路 (受け入れ条件5).

``comparison.csv`` / ``comparison_summary.csv`` / ``fig_comparison.png`` /
``fig_state_space.png`` / ``meta.json`` の5点をここで一括生成する。CLI
(``main.py`` と ``experiments/01_what_is_rc/run.py``) はこの関数を呼ぶだけの
薄い層にして、「どのコマンドから走らせても同じ成果物が出る」を構造で保証する。

レプリケート0の ``ReplicatePlan`` はここで1回だけ作り、``run_experiment`` と
``collect_state_space`` の両方へ明示的に渡す (F-1-009: 図と CSV が同じ乱数列を
見ているという暗黙の依存を、明示的な受け渡しに変える)。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from rc_basics_lab.config import ExperimentConfig
from rc_basics_lab.experiment.report import (
    COMPARISON_CSV,
    COMPARISON_SUMMARY_CSV,
    META_JSON,
    write_comparison_csv,
    write_meta,
)
from rc_basics_lab.experiment.runner import (
    ResultRow,
    run_experiment,
)
from rc_basics_lab.experiment.state_space import (
    DELAY_EMBEDDED_INPUT,
    RESERVOIR_STATE,
    StateSpaceReport,
    collect_state_space,
    summarize,
)
from rc_basics_lab.plotting.figures import plot_comparison, plot_state_space
from rc_basics_lab.plotting.style import setup_style

logger = logging.getLogger(__name__)

FIG_COMPARISON = "fig_comparison.png"
FIG_STATE_SPACE = "fig_state_space.png"

ARTIFACTS: tuple[str, ...] = (
    COMPARISON_CSV,
    COMPARISON_SUMMARY_CSV,
    FIG_COMPARISON,
    FIG_STATE_SPACE,
    META_JSON,
)
"""1コマンドで必ず出る成果物のファイル名 (受け入れ条件5 の検査対象)。"""


@dataclass(frozen=True, slots=True)
class ExperimentOutputs:
    """``run_and_report`` の成果物。

    Attributes:
        rows: ``comparison.csv`` と同じ長形式の行。
        state_space: 課題ごとの入力空間 / 状態空間の PCA 比較 (実験1-B)。
        paths: 生成したファイル (``ARTIFACTS`` と同じ並び)。
        wall_time_s: 計算部分の実測 wall time (図の書き出しは含まない)。
    """

    rows: tuple[ResultRow, ...]
    state_space: tuple[StateSpaceReport, ...]
    paths: tuple[Path, ...]
    wall_time_s: float


def _log_state_space(reports: tuple[StateSpaceReport, ...]) -> None:
    """``n_components_95`` の比較を**数値として**ログに残す (受け入れ条件4)。"""
    for report in reports:
        state = report.space(RESERVOIR_STATE)
        embedded = report.space(DELAY_EMBEDDED_INPUT)
        logger.info(
            "task=%s n_components_95: %s=%d (%d次元) vs %s=%d (%d次元) -> %s",
            report.task,
            RESERVOIR_STATE,
            state.n_components_95,
            state.n_features,
            DELAY_EMBEDDED_INPUT,
            embedded.n_components_95,
            embedded.n_features,
            "state > input"
            if state.n_components_95 > embedded.n_components_95
            else "state <= input",
        )


def run_and_report(config: ExperimentConfig, out_dir: Path) -> ExperimentOutputs:
    """実験を実行し、CSV・図2枚・meta.json を書き出す。

    Args:
        config: 実験設定。
        out_dir: 出力ディレクトリ (無ければ作る)。

    Returns:
        生成した行・PCA 比較・ファイルパス・実測 wall time。
    """
    started = time.perf_counter()
    rows = tuple(run_experiment(config))
    reports = collect_state_space(config)
    wall_time_s = time.perf_counter() - started
    _log_state_space(reports)

    style = setup_style()
    paths = (
        write_comparison_csv(rows, out_dir / COMPARISON_CSV),
        plot_comparison(rows, out_dir / FIG_COMPARISON, style=style),
        plot_state_space(reports, out_dir / FIG_STATE_SPACE, style=style),
        write_meta(
            config,
            wall_time_s,
            len(rows),
            out_dir / META_JSON,
            extra={"state_space": summarize(reports), "cjk_font": style.cjk_font},
        ),
    )
    logger.info(
        "完了: %d 行 / wall_time=%.2fs / 出力=%s",
        len(rows),
        wall_time_s,
        ", ".join(str(path) for path in paths),
    )
    return ExperimentOutputs(
        rows=rows, state_space=reports, paths=paths, wall_time_s=wall_time_s
    )


__all__ = [
    "ARTIFACTS",
    "FIG_COMPARISON",
    "FIG_STATE_SPACE",
    "ExperimentOutputs",
    "run_and_report",
]

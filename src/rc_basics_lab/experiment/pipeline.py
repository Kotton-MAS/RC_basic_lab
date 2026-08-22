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
from rc_basics_lab.experiment.horizon import CSV_COLUMNS as HORIZON_CSV_COLUMNS
from rc_basics_lab.experiment.horizon import TASK_NAME as HORIZON_TASK
from rc_basics_lab.experiment.horizon import run_horizon, summarize_horizon
from rc_basics_lab.experiment.report import (
    COMPARISON_CSV,
    COMPARISON_SUMMARY_CSV,
    META_JSON,
    write_comparison_csv,
    write_comparison_summary_csv,
    write_meta,
    write_rows_csv,
)
from rc_basics_lab.experiment.runner import (
    ReplicatePlan,
    ResultRow,
    build_tasks,
    plan_replicate,
    run_experiment,
)
from rc_basics_lab.experiment.state_space import (
    DELAY_EMBEDDED_INPUT,
    RESERVOIR_STATE,
    StateSpaceReport,
    collect_state_space,
    summarize,
)
from rc_basics_lab.experiment.summary import aggregate_nrmse
from rc_basics_lab.experiment.waveform_data import waveform_predictions

logger = logging.getLogger(__name__)

HORIZON_CSV = "horizon.csv"
FIG_HORIZON = "fig_horizon.png"
FIG_WAVEFORM = "fig_waveform.png"
FIG_COMPARISON = "fig_comparison.png"
FIG_STATE_SPACE = "fig_state_space.png"

ARTIFACTS: tuple[str, ...] = (
    COMPARISON_CSV,
    COMPARISON_SUMMARY_CSV,
    HORIZON_CSV,
    FIG_COMPARISON,
    FIG_HORIZON,
    FIG_WAVEFORM,
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


def _plan_zero_by_task(config: ExperimentConfig) -> dict[str, ReplicatePlan]:
    """タスク名 -> レプリケート0の ``ReplicatePlan``。

    ``run_experiment`` と ``collect_state_space`` の両方がレプリケート0を
    使うため、ここで1回だけ作って両方へ渡す (F-1-009)。
    """
    return {
        entry.name: plan_replicate(config, entry, 0) for entry in build_tasks(config)
    }


def run_and_report(config: ExperimentConfig, out_dir: Path) -> ExperimentOutputs:
    """実験を実行し、CSV2枚・図2枚・meta.json を書き出す。

    Args:
        config: 実験設定。
        out_dir: 出力ディレクトリ (無ければ作る)。

    Returns:
        生成した行・PCA 比較・ファイルパス・実測 wall time。
    """
    # 作図層の import を関数本体に置くのは D-53。合成層 (experiment) が
    # plotting を module-level で import すると
    # plotting/__init__ -> plotting.figures -> experiment.runner ->
    # experiment/__init__ -> experiment.pipeline -> plotting.figures
    # の循環になり ``import rc_basics_lab.plotting`` 単独が ImportError になる。
    # 先頭へ戻すと tests/test_layer_boundaries.py の AST guard と
    # subprocess guard の両方が落ちる。
    from rc_basics_lab.meta import git_commit
    from rc_basics_lab.plotting.figures import plot_comparison, plot_state_space
    from rc_basics_lab.plotting.figures_horizon import plot_horizon
    from rc_basics_lab.plotting.style import setup_style
    from rc_basics_lab.plotting.waveforms import plot_prediction_waveform

    started = time.perf_counter()
    plans0 = _plan_zero_by_task(config)
    rows = tuple(run_experiment(config, plans0=plans0))
    reports = collect_state_space(config, plans=plans0)
    wall_time_s = time.perf_counter() - started
    _log_state_space(reports)

    stats = aggregate_nrmse(rows)
    # 01' (D-105): 1ステップ先の行は1つも触らず、自走だけを別の CSV へ出す。
    horizon_entry = next(
        entry for entry in build_tasks(config) if entry.name == HORIZON_TASK
    )
    horizon_rows = run_horizon(config, horizon_entry)
    # commit は meta.json と図の footnote (FIG-6 / D-87) で同じ値を使う。
    style = setup_style(commit=git_commit())
    paths = (
        write_comparison_csv(rows, out_dir / COMPARISON_CSV),
        write_comparison_summary_csv(stats, out_dir / COMPARISON_SUMMARY_CSV),
        write_rows_csv(horizon_rows, out_dir / HORIZON_CSV, HORIZON_CSV_COLUMNS),
        plot_prediction_waveform(
            *waveform_predictions(config, plans0[HORIZON_TASK]),
            out_dir / FIG_WAVEFORM,
            task_label=("Mackey-Glass", "Mackey-Glass"),
            style=style,
        ),
        plot_comparison(rows, out_dir / FIG_COMPARISON, style=style),
        plot_horizon(horizon_rows, out_dir / FIG_HORIZON, style=style),
        plot_state_space(reports, out_dir / FIG_STATE_SPACE, style=style),
        write_meta(
            config,
            wall_time_s,
            len(rows),
            out_dir / META_JSON,
            extra={
                "state_space": summarize(reports),
                "cjk_font": style.cjk_font,
                "horizon": summarize_horizon(horizon_rows),
            },
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
    "FIG_HORIZON",
    "FIG_STATE_SPACE",
    "FIG_WAVEFORM",
    "HORIZON_CSV",
    "ExperimentOutputs",
    "run_and_report",
]

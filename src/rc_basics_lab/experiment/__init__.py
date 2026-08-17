"""実験層 — 課題・リザバー・読み出しを配線して結果の行を作る.

- ``split``: 連続分割と全手法共通の基準行 ``t0`` (D-05 / D-06)
- ``runner``: (課題 x 手法 x レプリケート) の実行 (D-04 / D-08)
- ``report``: ``comparison.csv`` / ``meta.json`` への書き出し
"""

from rc_basics_lab.experiment.report import write_comparison_csv, write_meta
from rc_basics_lab.experiment.runner import (
    CSV_COLUMNS,
    Method,
    ReplicatePlan,
    ResultRow,
    TaskEntry,
    build_methods,
    build_tasks,
    plan_replicate,
    run_experiment,
    run_task,
)
from rc_basics_lab.experiment.split import Split, compute_t0, make_split

__all__ = [
    "CSV_COLUMNS",
    "Method",
    "ReplicatePlan",
    "ResultRow",
    "Split",
    "TaskEntry",
    "build_methods",
    "build_tasks",
    "compute_t0",
    "make_split",
    "plan_replicate",
    "run_experiment",
    "run_task",
    "write_comparison_csv",
    "write_meta",
]

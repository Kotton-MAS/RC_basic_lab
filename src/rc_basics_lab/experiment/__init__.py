"""実験層 — 課題・リザバー・読み出しを配線して結果の行を作る.

- ``split``: 連続分割と全手法共通の基準行 ``t0`` (D-05 / D-06)
- ``runner``: (課題 x 手法 x レプリケート) の実行 (D-04 / D-08)
- ``state_space``: 入力空間とリザバー状態空間の PCA 比較 (実験1-B)
- ``report``: ``comparison.csv`` / ``meta.json`` への書き出し
- ``pipeline``: 1コマンドで4成果物をそろえる経路 (CLI はここを呼ぶだけ)
"""

from rc_basics_lab.experiment.pipeline import (
    ARTIFACTS,
    ExperimentOutputs,
    run_and_report,
)
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
from rc_basics_lab.experiment.state_space import (
    StateSpaceReport,
    collect_state_space,
)

__all__ = [
    "ARTIFACTS",
    "CSV_COLUMNS",
    "ExperimentOutputs",
    "Method",
    "ReplicatePlan",
    "ResultRow",
    "Split",
    "StateSpaceReport",
    "TaskEntry",
    "build_methods",
    "build_tasks",
    "collect_state_space",
    "compute_t0",
    "make_split",
    "plan_replicate",
    "run_and_report",
    "run_experiment",
    "run_task",
    "write_comparison_csv",
    "write_meta",
]

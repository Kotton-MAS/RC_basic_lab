"""実験層 — 課題・リザバー・読み出しを配線して結果の行を作る.

- ``split``: 連続分割と全手法共通の基準行 ``t0`` (D-05 / D-06)
- ``runner``: (課題 x 手法 x レプリケート) の実行 (D-04 / D-08)
- ``state_space``: 入力空間とリザバー状態空間の PCA 比較 (実験1-B)
- ``summary``: (課題, 手法) ごとの NRMSE 集計 (F-1-003)
- ``report``: ``comparison.csv`` / ``comparison_summary.csv`` / ``meta.json``
  への書き出し
- ``pipeline``: 1コマンドで5成果物をそろえる経路 (CLI はここを呼ぶだけ)
- ``esp``: 実験 2-A / 2-B / 2-C の配線 (ESN と診断層をつなぐ場所)
- ``capacity``: 実験 3-A / 3-B / 3-B' の配線 (MC / IPC を同じ X で測る)
- ``washout``: 実験 2-D の washout 感度 (01 の ``run_experiment`` を再利用)
- ``esp_pipeline``: 02 の成果物 (CSV2枚 + 図4枚 + meta.json) をそろえる経路
"""

from rc_basics_lab.experiment.capacity import (
    CAPACITY_CSV_COLUMNS,
    CAPACITY_EXPERIMENTS,
    CapacityCondition,
    CapacityOutcome,
    CapacityRow,
    evaluate_capacity_condition,
    ipc_config_for,
)
from rc_basics_lab.experiment.esp import (
    ESP_CSV_COLUMNS,
    ConditionOutcome,
    EspResults,
    EspRow,
    VerdictAgreement,
    esn_propagator,
    evaluate_condition,
    make_drive,
    make_initial_states,
    run_esp_experiment,
    summarize_verdict_agreement,
)
from rc_basics_lab.experiment.esp_pipeline import (
    ESP_ARTIFACTS,
    EspOutputs,
    run_and_report_esp,
    write_esp_csv,
    write_washout_csv,
)
from rc_basics_lab.experiment.pipeline import (
    ARTIFACTS,
    ExperimentOutputs,
    run_and_report,
)
from rc_basics_lab.experiment.report import (
    write_comparison_csv,
    write_comparison_summary_csv,
    write_meta,
    write_meta_for,
)
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
from rc_basics_lab.experiment.summary import Aggregate, aggregate_nrmse
from rc_basics_lab.experiment.washout import (
    WASHOUT_CSV_COLUMNS,
    MethodSensitivity,
    WashoutRow,
    WashoutSensitivity,
    mean_nrmse_by_washout,
    predicted_t0,
    run_washout_sweep,
    summarize_washout_sensitivity,
    variant_for,
)

__all__ = [
    "ARTIFACTS",
    "CAPACITY_CSV_COLUMNS",
    "CAPACITY_EXPERIMENTS",
    "CSV_COLUMNS",
    "ESP_ARTIFACTS",
    "ESP_CSV_COLUMNS",
    "WASHOUT_CSV_COLUMNS",
    "Aggregate",
    "CapacityCondition",
    "CapacityOutcome",
    "CapacityRow",
    "ConditionOutcome",
    "EspOutputs",
    "EspResults",
    "EspRow",
    "ExperimentOutputs",
    "Method",
    "MethodSensitivity",
    "ReplicatePlan",
    "ResultRow",
    "Split",
    "StateSpaceReport",
    "TaskEntry",
    "VerdictAgreement",
    "WashoutRow",
    "WashoutSensitivity",
    "aggregate_nrmse",
    "build_methods",
    "build_tasks",
    "collect_state_space",
    "compute_t0",
    "esn_propagator",
    "evaluate_capacity_condition",
    "evaluate_condition",
    "ipc_config_for",
    "make_drive",
    "make_initial_states",
    "make_split",
    "mean_nrmse_by_washout",
    "plan_replicate",
    "predicted_t0",
    "run_and_report",
    "run_and_report_esp",
    "run_esp_experiment",
    "run_experiment",
    "run_task",
    "run_washout_sweep",
    "summarize_verdict_agreement",
    "summarize_washout_sensitivity",
    "variant_for",
    "write_comparison_csv",
    "write_comparison_summary_csv",
    "write_esp_csv",
    "write_meta",
    "write_meta_for",
    "write_washout_csv",
]

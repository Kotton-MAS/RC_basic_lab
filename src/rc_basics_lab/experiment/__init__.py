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
- ``capacity_pipeline``: 03 の成果物 (CSV3枚 + 図5枚 + meta.json) をそろえる経路
- ``capacity_threshold``: しきい値法の比較 (受け入れ条件3、design.md §11.2 の一次資料)
- ``narma``: 実験 3-C (NARMA10) の配線 (01 の ``run_task`` を再利用、D-31)
- ``freerun``: 実験 4-A (1ステップ先予測) と自走の入口 (D-31 / D-44 / D-50)
- ``washout``: 実験 2-D の washout 感度 (01 の ``run_experiment`` を再利用)
- ``esp_pipeline``: 02 の成果物 (CSV2枚 + 図4枚 + meta.json) をそろえる経路
"""

from rc_basics_lab.experiment.capacity import (
    CAPACITY_CSV_COLUMNS,
    CAPACITY_EXPERIMENTS,
    CAPACITY_PROFILE_CSV_COLUMNS,
    CapacityCondition,
    CapacityOutcome,
    CapacityProfileRow,
    CapacityResults,
    CapacityRow,
    evaluate_capacity_condition,
    ipc_config_for,
    n_replicates_for,
    profile_rows,
    run_capacity_experiment,
    run_conservation_sweep,
    run_ipc_sweep,
    run_length_sweep,
    run_mc_sweep,
)
from rc_basics_lab.experiment.capacity_pipeline import (
    CAPACITY_ARTIFACTS,
    CapacityOutputs,
    run_and_report_capacity,
    run_and_report_length_sweep,
    write_capacity_csv,
    write_capacity_profile_csv,
)
from rc_basics_lab.experiment.capacity_threshold import (
    IPC_THRESHOLD_MODES,
    MC_THRESHOLD_MODES,
    IpcThresholdRow,
    McThresholdRow,
    ThresholdComparison,
    comparison_condition,
    run_threshold_comparison,
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
from rc_basics_lab.experiment.freerun import (
    ONESTEP_CSV,
    FreeRunOutcome,
    TeacherForcedReadout,
    chaos_task_entries,
    esn_state_updater,
    estimate_lorenz_lyapunov,
    fit_teacher_forced,
    run_free_run,
    run_onestep,
    validate_free_run_bounds,
)
from rc_basics_lab.experiment.narma import (
    Narma10Results,
    Narma10Verdict,
    narma_task_entry,
    run_narma10,
    summarize_narma10,
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
    "CAPACITY_ARTIFACTS",
    "CAPACITY_CSV_COLUMNS",
    "CAPACITY_EXPERIMENTS",
    "CAPACITY_PROFILE_CSV_COLUMNS",
    "CSV_COLUMNS",
    "ESP_ARTIFACTS",
    "ESP_CSV_COLUMNS",
    "IPC_THRESHOLD_MODES",
    "MC_THRESHOLD_MODES",
    "ONESTEP_CSV",
    "WASHOUT_CSV_COLUMNS",
    "Aggregate",
    "CapacityCondition",
    "CapacityOutcome",
    "CapacityOutputs",
    "CapacityProfileRow",
    "CapacityResults",
    "CapacityRow",
    "ConditionOutcome",
    "EspOutputs",
    "EspResults",
    "EspRow",
    "ExperimentOutputs",
    "FreeRunOutcome",
    "IpcThresholdRow",
    "McThresholdRow",
    "Method",
    "MethodSensitivity",
    "Narma10Results",
    "Narma10Verdict",
    "ReplicatePlan",
    "ResultRow",
    "Split",
    "StateSpaceReport",
    "TaskEntry",
    "TeacherForcedReadout",
    "ThresholdComparison",
    "VerdictAgreement",
    "WashoutRow",
    "WashoutSensitivity",
    "aggregate_nrmse",
    "build_methods",
    "build_tasks",
    "chaos_task_entries",
    "collect_state_space",
    "comparison_condition",
    "compute_t0",
    "esn_propagator",
    "esn_state_updater",
    "estimate_lorenz_lyapunov",
    "evaluate_capacity_condition",
    "evaluate_condition",
    "fit_teacher_forced",
    "ipc_config_for",
    "make_drive",
    "make_initial_states",
    "make_split",
    "mean_nrmse_by_washout",
    "n_replicates_for",
    "narma_task_entry",
    "plan_replicate",
    "predicted_t0",
    "profile_rows",
    "run_and_report",
    "run_and_report_capacity",
    "run_and_report_esp",
    "run_and_report_length_sweep",
    "run_capacity_experiment",
    "run_conservation_sweep",
    "run_esp_experiment",
    "run_experiment",
    "run_free_run",
    "run_ipc_sweep",
    "run_length_sweep",
    "run_mc_sweep",
    "run_narma10",
    "run_onestep",
    "run_task",
    "run_threshold_comparison",
    "run_washout_sweep",
    "summarize_narma10",
    "summarize_verdict_agreement",
    "summarize_washout_sensitivity",
    "validate_free_run_bounds",
    "variant_for",
    "write_capacity_csv",
    "write_capacity_profile_csv",
    "write_comparison_csv",
    "write_comparison_summary_csv",
    "write_esp_csv",
    "write_meta",
    "write_meta_for",
    "write_washout_csv",
]

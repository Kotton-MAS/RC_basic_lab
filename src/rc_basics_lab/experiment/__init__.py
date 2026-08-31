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
- ``attractor``: 自走軌道の評価 (有効予測時間 / 3態分類 / 長時間統計) の
  純関数層 (D-43 / D-45 / D-46)
- ``freerun``: 実験 4-A (1ステップ先予測) と 4-B (自走) の配線
  (D-31 / D-43 / D-44 / D-46 / D-50)
- ``stability``: 実験 4-C (3態マップ) と 4-D (同じ状態行列への MC / IPC)
- ``freerun_pipeline``: 04 の成果物 (CSV5枚 + 図5枚 + meta.json) をそろえる経路
- ``washout``: 実験 2-D の washout 感度 (01 の ``run_experiment`` を再利用)
- ``esp_pipeline``: 02 の成果物 (CSV2枚 + 図4枚 + meta.json) をそろえる経路
- ``anomaly_score``: 05 の異常スコア6系統を分ける唯一の場所 (D-61)
- ``anomaly_threshold``: 運用閾値と 5-B の掃引 (D-56。**ラベルを取らない**署名)
- ``anomaly_rows``: 05 の行 dataclass と CSV 列 (D-55 の列の対)
- ``anomaly_sources``: ``dataset.source`` -> ``SeriesSource`` の辞書 (D-71)。
  ``experiment/`` で ``datasets`` を import する唯一のモジュール
- ``anomaly``: 実験 5-A / 5-B の配線 (D-05 / D-57)
- ``anomaly_ranking``: 順位と「対照と区別できるか」の印 (D-78)。行から数を
  作るだけの純関数層
- ``anomaly_sweep``: 実験 5-C (プロトコル感度) / 5-D (N と性能) の掃引
  (D-78 / D-79)。格子点ごとに 5-A をそのまま回して集計するだけの層
- ``anomaly_pipeline``: 05 の成果物 (CSV5枚 + 図5枚 + meta.json) をそろえる経路
"""

from rc_basics_lab.experiment import (
    anomaly,
    anomaly_pipeline,
    anomaly_ranking,
    anomaly_rows,
    anomaly_score,
    anomaly_sources,
    anomaly_sweep,
    anomaly_threshold,
    attractor,
    capacity,
    capacity_bounds,
    capacity_pipeline,
    capacity_rows,
    capacity_threshold,
    esp,
    esp_pipeline,
    freerun,
    freerun_pipeline,
    horizon,
    narma,
    narma_taps,
    pipeline,
    report,
    runner,
    split,
    stability,
    state_space,
    state_updaters,
    state_waveform,
    summary,
    symmetry,
    threshold,
    valid_time,
    washout,
    waveform_data,
)

__all__ = [
    "anomaly",
    "anomaly_pipeline",
    "anomaly_ranking",
    "anomaly_rows",
    "anomaly_score",
    "anomaly_sources",
    "anomaly_sweep",
    "anomaly_threshold",
    "attractor",
    "capacity",
    "capacity_bounds",
    "capacity_pipeline",
    "capacity_rows",
    "capacity_threshold",
    "esp",
    "esp_pipeline",
    "freerun",
    "freerun_pipeline",
    "horizon",
    "narma",
    "narma_taps",
    "pipeline",
    "report",
    "runner",
    "split",
    "stability",
    "state_space",
    "state_updaters",
    "state_waveform",
    "summary",
    "symmetry",
    "threshold",
    "valid_time",
    "washout",
    "waveform_data",
]

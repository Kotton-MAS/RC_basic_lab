"""作図層 — 記事用の図をヘッドレス環境でも同じ手順で再生成する.

- ``style``: rcParams (savefig.dpi=200) と CJK フォント探索 (D-10)
- ``labels``: 日本語 / 英語ラベルの切り替え (豆腐文字を出さない)
- ``figures``: 記事01の図2枚 (``fig_comparison`` / ``fig_state_space``)
- ``figures_esp``: 記事02の図4枚 (``fig_esp_decay`` / ``fig_leak_timescale`` /
  ``fig_esp_map`` / ``fig_washout_sensitivity``)
- ``figures_capacity``: 記事03の図5枚 (``fig_mc_sweep`` / ``fig_ipc_profile`` /
  ``fig_memory_nonlinearity`` / ``fig_ipc_conservation`` /
  ``fig_narma10_control``)
- ``figures_freerun``: 記事04の図5枚 (``fig_onestep`` /
  ``fig_freerun_attractor`` / ``fig_valid_time`` / ``fig_stability_map`` /
  ``fig_freerun_stats``)
- ``figures_anomaly``: 記事05の図のうち 5-A / 5-B の3枚 (``fig_pr_curves`` /
  ``fig_score_timeline`` / ``fig_threshold_tradeoff``) と、印の体裁の単一の真実
- ``capacity_grids``: 記事03の図が読む「行 → 格子」の復元 (描画を含まない)
- ``narma10_panel``: 実験 3-C の横軸ラベルと結論文の生成 (描画を含まない)
- ``heatmap``: 打ち切り・欠測を 0 と別の色で描くための補助 (FIG-7 / D-88)
- ``figures_anomaly_sweep``: 記事05の掃引の2枚 (``fig_protocol_sensitivity`` /
  ``fig_size_vs_performance``)。**印を必ず可視化する** (D-81)

pyplot は使わない (``Figure`` + ``FigureCanvasAgg`` を直接組む)。CI には
ディスプレイが無いため、既定バックエンドに依存しない経路にそろえる。
"""

from rc_basics_lab.plotting import (
    capacity_grids,
    esp_references,
    figures,
    figures_anomaly,
    figures_anomaly_sweep,
    figures_capacity,
    figures_esp,
    figures_freerun,
    figures_freerun_time,
    figures_horizon,
    figures_narma_taps,
    figures_stability,
    freerun_grids,
    freerun_headlines,
    heatmap,
    labels,
    narma10_panel,
    style,
    waveforms,
)

__all__ = [
    "capacity_grids",
    "esp_references",
    "figures",
    "figures_anomaly",
    "figures_anomaly_sweep",
    "figures_capacity",
    "figures_esp",
    "figures_freerun",
    "figures_freerun_time",
    "figures_horizon",
    "figures_narma_taps",
    "figures_stability",
    "freerun_grids",
    "freerun_headlines",
    "heatmap",
    "labels",
    "narma10_panel",
    "style",
    "waveforms",
]

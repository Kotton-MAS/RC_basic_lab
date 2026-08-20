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

pyplot は使わない (``Figure`` + ``FigureCanvasAgg`` を直接組む)。CI には
ディスプレイが無いため、既定バックエンドに依存しない経路にそろえる。
"""

from rc_basics_lab.plotting import (
    figures,
    figures_capacity,
    figures_esp,
    figures_freerun,
    labels,
    style,
)

__all__ = [
    "figures",
    "figures_capacity",
    "figures_esp",
    "figures_freerun",
    "labels",
    "style",
]

"""作図層 — 記事用の図をヘッドレス環境でも同じ手順で再生成する.

- ``style``: rcParams (savefig.dpi=200) と CJK フォント探索 (D-10)
- ``labels``: 日本語 / 英語ラベルの切り替え (豆腐文字を出さない)
- ``figures``: 記事01の図2枚 (``fig_comparison`` / ``fig_state_space``)
- ``figures_esp``: 記事02の図4枚 (``fig_esp_decay`` / ``fig_leak_timescale`` /
  ``fig_esp_map`` / ``fig_washout_sensitivity``)
- ``figures_capacity``: 記事03の図4枚 (``fig_mc_sweep`` / ``fig_ipc_profile`` /
  ``fig_memory_nonlinearity`` / ``fig_ipc_conservation``)

pyplot は使わない (``Figure`` + ``FigureCanvasAgg`` を直接組む)。CI には
ディスプレイが無いため、既定バックエンドに依存しない経路にそろえる。
"""

from rc_basics_lab.plotting.figures import plot_comparison, plot_state_space
from rc_basics_lab.plotting.figures_capacity import (
    plot_ipc_conservation,
    plot_ipc_profile,
    plot_mc_sweep,
    plot_memory_nonlinearity,
)
from rc_basics_lab.plotting.figures_esp import (
    plot_esp_decay,
    plot_esp_map,
    plot_leak_timescale,
    plot_washout_sensitivity,
)
from rc_basics_lab.plotting.labels import label
from rc_basics_lab.plotting.style import (
    StyleContext,
    find_cjk_font,
    rc_params_for,
    setup_style,
)

__all__ = [
    "StyleContext",
    "find_cjk_font",
    "label",
    "plot_comparison",
    "plot_esp_decay",
    "plot_esp_map",
    "plot_ipc_conservation",
    "plot_ipc_profile",
    "plot_leak_timescale",
    "plot_mc_sweep",
    "plot_memory_nonlinearity",
    "plot_state_space",
    "plot_washout_sensitivity",
    "rc_params_for",
    "setup_style",
]

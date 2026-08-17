"""作図層 — 記事用の図をヘッドレス環境でも同じ手順で再生成する.

- ``style``: rcParams (savefig.dpi=200) と CJK フォント探索 (D-10)
- ``labels``: 日本語 / 英語ラベルの切り替え (豆腐文字を出さない)
- ``figures``: 記事01の図2枚 (``fig_comparison`` / ``fig_state_space``)

pyplot は使わない (``Figure`` + ``FigureCanvasAgg`` を直接組む)。CI には
ディスプレイが無いため、既定バックエンドに依存しない経路にそろえる。
"""

from rc_basics_lab.plotting.figures import plot_comparison, plot_state_space
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
    "plot_state_space",
    "rc_params_for",
    "setup_style",
]

"""4-C が振る軸と、条件からリザバー設定を組む場所 (D-124).

``stability.py`` が 600 行の上限 (D-63 / D-77) に達したので分けた。役割としても
「何を振るか」は「どう測るか」とは別である。
"""

from __future__ import annotations

from rc_basics_lab.config import Chaos04Config
from rc_basics_lab.experiment.freerun_tasks import chaos_reservoir_config
from rc_basics_lab.experiment.stability_rows import StabilityCondition
from rc_basics_lab.reservoir.axes import require_axes, with_axis
from rc_basics_lab.reservoir.protocol import ReservoirConfig

STABILITY_AXES: tuple[str, ...] = ("spectral_radius", "leak_rate", "state_noise")
"""4-C が振る軸 (**モデルの言葉のまま**。D-124)。

軸名をモデル間で共通化しない —— ``leak_rate`` を持たないモデルに別名を
生やすと「同じ名前だが意味が違う軸」ができ、図をまたいだ比較が静かに壊れる。
持っていないモデルには ``require_axes`` が「持っている軸はこれ」と言って落ちる。
"""


def condition_reservoir_config(
    config: Chaos04Config, condition: StabilityCondition
) -> ReservoirConfig:
    """条件の3軸だけを差し替えたリザバー設定を返す (他の構造 HP は動かさない)。

    D-08 により構造ハイパーパラメータは検証分割で選ばない。掃引で動くのは
    ``STABILITY_AXES`` の3つだけで、``n_units`` / ``input_scale`` /
    ``density`` / ``activation`` は 4-A・4-B と同じ1点のままである
    (動かすと「ノイズで領域が変わった」のか「別のリザバーだった」のかが
    分からなくなる)。

    **モデルは問わない** (D-124)。``kind: ring`` でも 4-C の掃引はそのまま
    回る —— リングも同じ3軸を持つため。持たないモデルなら
    ``require_axes`` が落とす。
    """
    base = chaos_reservoir_config(config.base)
    require_axes(base, STABILITY_AXES, "実験4-C (3態マップ)")
    values = (condition.rho, condition.leak_rate, condition.state_noise)
    for name, value in zip(STABILITY_AXES, values, strict=True):
        base = with_axis(base, name, value)
    return base


__all__ = ["STABILITY_AXES", "condition_reservoir_config"]

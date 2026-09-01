"""課題の接合面 — **課題を足すときに満たすべき形** (D-123).

``reservoir/protocol.py`` と同じ役割である。``experiment`` 層は
``TaskConfig`` と ``registry.build_task`` だけを見るので、課題を1つ足しても
生成箇所を触らずに済む。

課題は**純関数**である (D-59)。設定を受け取って ``TaskData`` を返すだけで、
ネットワークも I/O も持たない。外部データを読む課題は ``datasets/`` が
取得して ``tasks/`` に渡す。

``TaskConfig`` に入れてよいのは**生成パラメータだけ**である。リザバーの設定を
入れてはいけない —— 同じ課題を別のモデルで回すのが実験であり、課題が
モデルを持つと 04 / 05 がその課題を使うたびに使わないリザバー設定を
引きずることになる。組み合わせは ``config`` 側の ``TaskSpec`` が持つ。
"""

from __future__ import annotations

from rc_basics_lab.config import DelayParityConfig, MackeyGlassConfig

type TaskConfig = MackeyGlassConfig | DelayParityConfig
"""課題の生成設定。課題を足したら union に足す (``ReservoirConfig`` と同じ流儀)。

**先頭が既定である。** ``registry.build_task`` の ``match`` が唯一の分岐点で、
ここに足して ``case`` を書き忘れたら mypy の網羅性検査が落とす。
"""


__all__ = ["TaskConfig"]

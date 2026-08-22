"""リザバー層 — 入力系列から状態系列 ``X`` を作る側.

この層は ``rc_basics_lab.diagnostics`` を **import しない** (依存の向きは
「診断層が独立側」)。診断は外部素子由来の状態系列にもそのまま適用できる必要が
あるため、両者は片方向にも結合させない。

サイクル 02〜05 でこのパッケージに手を入れるときも、``ESN.step`` / ``ESN.run`` の
公開署名は変更しない (02 の2初期状態は ``x0``、04 の閉ループは ``step``、
04 のノイズ注入は ``ESNConfig.state_noise`` で既に配線済み)。
"""

from rc_basics_lab.reservoir.esn import ESN, ESNConfig, spectral_radius

__all__ = ["ESN", "ESNConfig", "spectral_radius"]

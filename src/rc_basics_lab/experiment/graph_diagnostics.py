"""結合行列そのものを測る診断を実験へ配線する層 (D-122 / D-132).

``diagnostics/topology.py`` は ``W`` を取る族で、``reservoir`` を知らない。
モデルから行列を取り出して族をまとめて回すのは実験側の仕事なので、その1手順を
ここに置く (``capacity.py`` は 600 行の上限を超えて凍結されており、足せない)。
"""

from __future__ import annotations

from rc_basics_lab.diagnostics.base import DiagnosticResult
from rc_basics_lab.diagnostics.topology import (
    degree_distribution,
    small_world,
    spectral_profile,
)
from rc_basics_lab.reservoir.protocol import Reservoir
from rc_basics_lab.reservoir.registry import require_graph


def graph_diagnostics(reservoir: Reservoir) -> tuple[DiagnosticResult, ...]:
    """結合行列そのものを測る診断を回す (D-122 の族を実験へ配線する)。

    状態行列とは別に**1条件につき1回**、リザバーの結合を測る。同じ密度でも
    Erdos-Renyi と Barabasi-Albert では次数分布もスモールワールド性も違うので、
    「容量が高いのはトポロジのおかげか」を後から問えるようにしておく。

    行列を持たないモデルなら ``require_graph`` が落とす —— 黙って空を返すと
    「トポロジの行だけが静かに消えた成果物」ができる。

    Args:
        reservoir: その条件で作ったリザバー。

    Returns:
        ``degree_distribution`` / ``spectral_profile`` / ``small_world`` の結果。
    """
    matrix = require_graph(reservoir, used_by="実験03 (容量とトポロジ)")
    return (
        degree_distribution(matrix),
        spectral_profile(matrix),
        small_world(matrix),
    )


__all__ = ["graph_diagnostics"]

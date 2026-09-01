"""リザバーの接合面 — **モデルを足すときに満たすべき面**.

``experiment`` 層はこの ``Reservoir`` だけを見る。``ESN`` を直接名指しすると、
モデルを1つ足すたびに生成箇所 (実測で5箇所) を全部触ることになる。

**面は実測で決めてある。** ``experiment`` 層から呼ばれているのは ``run`` /
``step`` / ``config`` / ``n_units`` / ``n_inputs`` の5つだけである。
``ESN`` はこれに加えて ``W`` / ``W_in`` / ``initial_state`` を公開しているが、
**それらは面に入れない** —— 使うのは ESN 固有の検査 (スペクトル半径が設定値に
正規化されているか、同一シードで重みがバイト一致するか) だけで、リザバー一般に
要求できる性質ではない。面に入れると、新しいモデルが「再帰行列 W を持つ」
義務まで負うことになる (持たないモデルはいくらでもある)。

ESN 固有の検査が要るときは ``ESN`` を直接名指しする。**それが正しい**ので、
``tests/test_reservoir.py`` はこの Protocol を経由しない。

新しいモデルを足す手順は ``docs/guide/リザバーを足す.md`` にある。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from rc_basics_lab.reservoir.deep import DeepESNConfig
from rc_basics_lab.reservoir.esn import ESNConfig
from rc_basics_lab.reservoir.ring import RingConfig
from rc_basics_lab.types import FloatArray

type ReservoirConfig = ESNConfig | DeepESNConfig | RingConfig
"""リザバーの構造設定。モデルを足したら union に足す (``FeatureSpec`` と同じ流儀)。

**先頭が既定である。** YAML で ``kind`` を省いたら先頭 (``ESNConfig``) になるので、
並び順を変えると既存の設定の意味が変わる。足すときは末尾へ。

``registry.build_reservoir`` の ``match`` が唯一の分岐点で、ここに足して
``case`` を書き忘れたら mypy の網羅性検査が落とす。
"""


@runtime_checkable
class Reservoir(Protocol):
    """入力系列を状態系列へ写すもの。

    ``runtime_checkable`` にしてあるのはテストで ``isinstance`` を使うためで、
    本番経路の分岐には使わない (Protocol の isinstance はメソッドの有無しか
    見ず、署名は見ないため、判定に使うと署名違いを取りこぼす)。
    """

    @property
    def config(self) -> ReservoirConfig:
        """構造ハイパーパラメータ (``meta.json`` と図の footnote が読む)。"""
        ...

    @property
    def n_units(self) -> int:
        """状態の次元 N。``run`` が返す配列の列数と必ず一致する。"""
        ...

    @property
    def n_inputs(self) -> int:
        """入力次元 D_in。"""
        ...

    def step(
        self,
        x: FloatArray,
        u: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> FloatArray:
        """1ステップ更新して次状態を返す (自走の閉ループが使う)。"""
        ...

    def run(
        self,
        u: FloatArray,
        x0: FloatArray | None = None,
        rng: np.random.Generator | None = None,
    ) -> FloatArray:
        """入力系列 ``(T, D_in)`` を流して状態系列 ``(T, N)`` を返す。"""
        ...


@runtime_checkable
class GraphReservoir(Protocol):
    """結合を明示的な行列として持つモデルだけが満たす**追加の**面 (D-122).

    ``Reservoir`` の5面は広げない。「再帰行列を持つ」はリザバー一般に要求できる
    性質ではないからで、外部素子のモデルは持たないことがある。

    **解析側がこの面で分岐してはいけない。** ``isinstance(res, GraphReservoir)``
    で挙動を変えると、行列を持たないモデルを渡したとき「トポロジ診断だけが
    静かに空になる」実験ができてしまう。トポロジを見る実験の側が
    ``registry.require_graph(res, used_by=...)`` で**要求する** ——
    ``require_esn`` と同じ流儀である。
    """

    def adjacency(self) -> FloatArray:
        """状態どうしの結合を ``(N, N)`` で返す。

        向きの規約は ``W[i, j] != 0`` が **j から i への辺**
        (``x_{t+1} = f(W x_t + ...)`` なので行が受け手)。
        """
        ...


__all__ = ["GraphReservoir", "Reservoir", "ReservoirConfig"]

"""自走の状態更新器 (``StateUpdater`` アダプタ, D-50).

``free_run`` は「状態をどう進めるか」を関数1つで受け取る。ESN・遅延線・
線形ベースラインの3つは、**どれも ESN を参照せずに書ける**というのが
D-50 の主張であり、その実例をここに並べる。

``freerun.py`` から分けてあるのは行数上限 (D-77) のためである。
**上限のほうを緩めない**。
"""

from __future__ import annotations

import numpy as np

from rc_basics_lab.readout.autoregressive import StateUpdater
from rc_basics_lab.reservoir.protocol import Reservoir
from rc_basics_lab.types import FloatArray


def esn_state_updater(
    esn: Reservoir, rng: np.random.Generator | None = None
) -> StateUpdater:
    """ESN を ``StateUpdater`` (D-50) に適合させるアダプタ。

    **``ESN.run`` ではなく ``ESN.step`` を使う** (仕様 §5 禁止する構造8)。自走は
    ``u[t+1]`` が ``y_hat[t]`` に依存するので、入力系列が既知でないと動かない
    ``run`` では書けない。

    **``state_noise > 0`` なら ``rng`` を渡すのが正しい** —— 自走は伝播器では
    なく**軌道を作る**呼び出しであり、学習時の状態にノイズを入れた設定で自走中
    だけノイズを外すと、学習時と評価時で別の系を測ることになる (D-36)。02 の
    ``esn_propagator`` が決定的でなければならない (D-48) のは、条件付き
    Lyapunov 指数が「同じ軌道のまわりの摂動の成長率」を測るからであって、
    「ESN は常に決定的に回す」という規則ではない。

    Raises:
        ValueError: ``state_noise > 0`` なのに ``rng`` が ``None`` の場合。
    """
    if esn.config.state_noise > 0.0 and rng is None:
        raise ValueError(
            "state_noise > 0 の自走には rng が必要です (D-36)。"
            "黙ってノイズ無しで自走すると、学習時とは別の系を評価することに"
            "なる。決定性が要るのは 02 の伝播器 (D-48) であって自走ではない"
        )

    def update(x: FloatArray, u: FloatArray) -> FloatArray:
        return esn.step(x, u, rng)

    return update


def delay_line_state_updater(n_inputs: int) -> StateUpdater:
    """遅延線を ``StateUpdater`` (D-50) に適合させるアダプタ (シフトレジスタ)。

    遅延線には内部状態が無い、というのは**設計行列から見た話**にすぎない。
    閉ループにすると「直前まで自分が吐いた出力」を保持する必要があり、それは
    シフトレジスタという**状態**である。``x[k] = [u[k], u[k-1], ..., u[k-K]]``
    と置けば ``[1, x[k]]`` (``ReservoirSpec(include_input=False)``) が
    ``DelayLineSpec(n_lags=K)`` の ``[1, u[k], ..., u[k-K]]`` と**同じ列**に
    なるので、教師強制で学んだ係数をそのまま流せる (D-44)。

    これは受け入れ条件3 の後半 (「自走では対照が成立しない」) を**数値で**
    測るための配線である。対照を自走させずに「原理的に不利」とだけ書くと、
    主張が実測から切り離される。同時に、ESN を1行も参照しない外部状態生成器で
    ``free_run`` が動くこと (D-50) の2つ目の実例でもある。

    Args:
        n_inputs: 入力次元 ``D_in`` (レジスタは ``D_in`` ずつずれる)。

    Raises:
        ValueError: ``n_inputs`` が 1 未満の場合。
    """
    if n_inputs < 1:
        raise ValueError(f"n_inputs は 1 以上である必要があります: {n_inputs}")

    def update(x: FloatArray, u: FloatArray) -> FloatArray:
        shifted: FloatArray = np.concatenate((u, x[:-n_inputs]))
        return shifted

    return update


def passthrough_state_updater() -> StateUpdater:
    """線形ベースラインの ``StateUpdater`` (状態を持たない = 恒等写像)。

    ``PassthroughSpec`` は状態を1列も使わないので、``free_run`` に渡す状態は
    形だけのダミー1要素でよい。**恒等写像であること自体が主張**である ——
    記憶を持たない手法を閉ループに入れると ``u[k+1] = W [1, u[k]]`` という
    1次のアフィン写像になり、不動点へ落ちるか発散するかしかない
    (要件書 位置づけ(b))。
    """

    def update(x: FloatArray, u: FloatArray) -> FloatArray:
        return x

    return update


__all__ = [
    "delay_line_state_updater",
    "esn_state_updater",
    "passthrough_state_updater",
]

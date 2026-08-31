"""リング結合リザバー (SCR: Simple Cycle Reservoir).

Rodan & Tino (2011) "Minimum Complexity Echo State Network" の構成。
**ランダム性をほとんど使わない**リザバーで、標準的な ESN と同等の性能が出る
ことを示した文献であり、「リザバーの性能はランダムな結合そのものから来るのか」
という問いに対する対照になる。

構成は極端に単純である:

- ``W`` は**一方向の単一閉路**。``W[i, i-1] = r`` と ``W[0, N-1] = r`` だけが
  非零で、値はすべて同じ ``r``
- ``W_in`` は**大きさがすべて同じ**。符号だけが変わる

閉路行列のスペクトル半径は厳密に ``|r|`` なので、``spectral_radius`` の設定値が
そのまま ``r`` になる (ESN のように生成後に測って割る操作が要らない)。

**文献との違いを1つ明記する。** 原論文は符号を円周率の桁から決める完全に
決定論的な構成だが、ここでは ``rng`` から引く。このリポジトリの再現性は
「シードを固定すれば同じ」で統一されており (D-06)、そこだけ別の規律を持ち込むと
レプリケート間の比較が別の意味になるためである。**大きさが一定であること**が
SCR の主張の核で、符号の決め方はそこではない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from rc_basics_lab.types import FloatArray


@dataclass(frozen=True, slots=True)
class RingConfig:
    """リング結合リザバーの構造ハイパーパラメータ。

    Attributes:
        n_units: リザバーのユニット数 N (閉路の長さ)。
        spectral_radius: 閉路の重み ``r`` そのもの。閉路行列のスペクトル半径は
            厳密に ``|r|`` なので、測って割り直す操作が要らない。
        leak_rate: 漏れ率 a。1.0 で漏れなし。
        input_scale: 入力重みの**大きさ** (符号だけが ``rng`` で変わる)。
        bias_scale: 定数入力に対応する重みの大きさ。
        state_noise: tanh 内部に加えるガウスノイズの標準偏差。

    ``density`` を持たないのは、閉路の密度が ``1/N`` で構造から決まるためである。
    設定できるように見せると「効かない設定」になる (D-13)。
    """

    KIND: ClassVar[str] = "ring"

    n_units: int = 200
    spectral_radius: float = 0.9
    leak_rate: float = 0.3
    input_scale: float = 0.5
    bias_scale: float = 0.1
    state_noise: float = 0.0


def _validate_ring_config(config: RingConfig, n_inputs: int) -> None:
    """設定値の範囲を検査する (``ESN._validate_config`` と同じ規律)。"""
    if config.n_units < 2:
        raise ValueError(
            f"n_units は 2 以上である必要があります (閉路のため): {config.n_units}"
        )
    if not 0.0 < config.leak_rate <= 1.0:
        raise ValueError(
            f"leak_rate は (0, 1] である必要があります: {config.leak_rate}"
        )
    if config.spectral_radius <= 0.0:
        raise ValueError(
            f"spectral_radius は正である必要があります: {config.spectral_radius}"
        )
    if config.input_scale < 0.0 or config.bias_scale < 0.0:
        raise ValueError(
            "input_scale / bias_scale は 0 以上である必要があります: "
            f"{config.input_scale}, {config.bias_scale}"
        )
    if config.state_noise < 0.0:
        raise ValueError(
            f"state_noise は 0 以上である必要があります: {config.state_noise}"
        )
    if n_inputs < 1:
        raise ValueError(f"n_inputs は 1 以上である必要があります: {n_inputs}")


def cycle_matrix(n_units: int, weight: float) -> FloatArray:
    """一方向の単一閉路 ``(N, N)`` を返す。

    ``W[i, i-1] = weight`` と ``W[0, N-1] = weight`` だけが非零である。
    **この行列のスペクトル半径は厳密に ``|weight|``** —— 固有値は
    ``weight`` に 1 の N 乗根を掛けたもので、絶対値が全部同じになる。

    Args:
        n_units: 閉路の長さ N (2 以上)。
        weight: 閉路の重み。

    Returns:
        ``(N, N)`` の密行列 (N <= 1000 なので疎行列は使わない)。
    """
    matrix: FloatArray = np.zeros((n_units, n_units), dtype=np.float64)
    rows = np.arange(n_units)
    matrix[rows, rows - 1] = weight
    return matrix


class RingReservoir:
    """リング結合リザバー (``Reservoir`` を満たす)。

    更新式は ESN と同じ漏れ積分 tanh である。違うのは ``W`` と ``W_in`` の
    作り方だけで、**同じ更新式のもとで構造だけを比べられる**ようにしてある。
    """

    def __init__(
        self,
        config: RingConfig,
        rng: np.random.Generator,
        *,
        n_inputs: int = 1,
    ) -> None:
        """重みを生成する。

        Args:
            config: 構造ハイパーパラメータ。
            rng: 符号を引く Generator (``seeds.make_rng`` の reservoir ストリーム)。
            n_inputs: 入力次元 D_in。

        Raises:
            ValueError: 設定値が範囲外の場合。
        """
        _validate_ring_config(config, n_inputs)
        self._config = config
        self._n_inputs = n_inputs

        # 大きさは一定、符号だけ引く (SCR の主張の核は「大きさが一定」)。
        signs: FloatArray = np.where(
            rng.random((config.n_units, 1 + n_inputs)) < 0.5, -1.0, 1.0
        )
        scales: FloatArray = np.full(
            (config.n_units, 1 + n_inputs), config.input_scale, dtype=np.float64
        )
        scales[:, 0] = config.bias_scale
        weights_in: FloatArray = signs * scales
        recurrent = cycle_matrix(config.n_units, config.spectral_radius)

        weights_in.setflags(write=False)
        recurrent.setflags(write=False)
        self._weights_in = weights_in
        self._recurrent = recurrent

    @property
    def config(self) -> RingConfig:
        """生成に使った設定。"""
        return self._config

    @property
    def n_units(self) -> int:
        """リザバーのユニット数 N。"""
        return self._config.n_units

    @property
    def n_inputs(self) -> int:
        """入力次元 D_in。"""
        return self._n_inputs

    @property
    def W_in(self) -> FloatArray:
        """入力重み ``(N, 1 + D_in)``。読み取り専用。

        ``Reservoir`` の面には入っていない (``ESN`` と同じ扱い)。**この構成が
        主張する「大きさが一定」を検査するために公開する** —— 振る舞いからでは
        「符号だけが違う」ことを直接は測れない。
        """
        return self._weights_in

    @property
    def W(self) -> FloatArray:
        """再帰重み ``(N, N)``。読み取り専用 (単一閉路)。"""
        return self._recurrent

    def initial_state(self) -> FloatArray:
        """既定の初期状態 (零ベクトル)。"""
        return np.zeros(self._config.n_units, dtype=np.float64)

    def _update(
        self, x: FloatArray, drive: FloatArray, rng: np.random.Generator | None
    ) -> FloatArray:
        """更新式の唯一の実装。``step`` と ``run`` がともにここを通る。"""
        pre_activation = drive + self._recurrent @ x
        noise = self._config.state_noise
        if noise > 0.0:
            if rng is None:
                raise ValueError(
                    "state_noise > 0 のときは rng が必要です "
                    "(黙ってノイズ無しにすると設定が効かない実験になる)"
                )
            pre_activation = pre_activation + noise * rng.standard_normal(
                self._config.n_units
            )
        leak = self._config.leak_rate
        activated: FloatArray = np.tanh(pre_activation)
        return (1.0 - leak) * x + leak * activated

    def _check_state(self, x: FloatArray) -> FloatArray:
        state = np.asarray(x, dtype=np.float64)
        if state.shape != (self._config.n_units,):
            raise ValueError(
                f"状態は ({self._config.n_units},) である必要があります: {state.shape}"
            )
        return state

    def step(
        self,
        x: FloatArray,
        u: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> FloatArray:
        """1ステップ更新して次状態を返す。"""
        state = self._check_state(x)
        inputs = np.asarray(u, dtype=np.float64)
        if inputs.shape != (self._n_inputs,):
            raise ValueError(
                f"入力は ({self._n_inputs},) である必要があります: {inputs.shape}"
            )
        drive = self._weights_in[:, 0] + self._weights_in[:, 1:] @ inputs
        return self._update(state, drive, rng)

    def run(
        self,
        u: FloatArray,
        x0: FloatArray | None = None,
        rng: np.random.Generator | None = None,
    ) -> FloatArray:
        """入力系列 ``(T, D_in)`` を流して状態系列 ``(T, N)`` を返す。"""
        inputs = np.asarray(u, dtype=np.float64)
        if inputs.ndim != 2:
            raise ValueError(f"u は (T, D_in) の2次元配列が必要です: {inputs.shape}")
        n_steps, n_inputs = inputs.shape
        if n_inputs != self._n_inputs:
            raise ValueError(f"入力次元が一致しません: {n_inputs} != {self._n_inputs}")
        if n_steps == 0:
            raise ValueError("u が空です")
        state = self.initial_state() if x0 is None else self._check_state(x0)
        states: FloatArray = np.empty((n_steps, self._config.n_units), dtype=np.float64)
        drives = self._weights_in[:, 0] + inputs @ self._weights_in[:, 1:].T
        for index in range(n_steps):
            state = self._update(state, drives[index], rng)
            states[index] = state
        return states


__all__ = ["RingConfig", "RingReservoir", "cycle_matrix"]

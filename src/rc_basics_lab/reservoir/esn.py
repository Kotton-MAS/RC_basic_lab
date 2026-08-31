"""Echo State Network の最小実装.

更新式は1本だけである::

    x[t] = (1 - a) * x[t-1] + a * tanh(W_in @ [1; u[t]] + W @ x[t-1] + noise)

``step`` と ``run`` を**両方このサイクルで公開する**。``x0`` (02 の2初期状態)、
``state_noise`` (04 のノイズ注入)、``step`` (04 の閉ループ)を先に配線しておき、
サイクル 02/04 で公開 API を変更しなくて済むようにするためである。

``run`` は ``step`` を逐次呼ぶのと**ビット単位で同一の結果**を返す
(``tests/test_reservoir.py::test_run_equals_repeated_step``)。閉ループ実験で
``step`` に切り替えた瞬間に軌道が変わる、という事故を防ぐ。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from rc_basics_lab.types import FloatArray

TANH = "tanh"
"""現在サポートする唯一の活性化関数名。"""


@dataclass(frozen=True, slots=True)
class ESNConfig:
    """リザバーの構造ハイパーパラメータ (D-08: 検証分割で調整しない)。

    Attributes:
        n_units: リザバーのユニット数 N。
        spectral_radius: 再帰行列 W のスペクトル半径 (実測値で正規化する)。
        leak_rate: 漏れ率 a。1.0 で漏れなし (通常の tanh 更新)。
        input_scale: 入力重みの一様分布の幅 ``U[-input_scale, input_scale]``。
        bias_scale: 定数入力 (``[1; u]`` の先頭) に対応する重みの幅。
        density: W の非零率。
        activation: 活性化関数名。現在は ``"tanh"`` のみ。
        state_noise: tanh 内部に加えるガウスノイズの標準偏差 (04 用。既定 0)。

    値の検証は ``ESN.__init__`` で行う (設定 dataclass 群は T1 と同じく
    純粋なデータ保持に留める)。

    ``KIND`` は YAML の ``kind: esn`` に対応する判別子である。**``ClassVar``
    なので ``dataclasses.asdict`` に現れない** —— フィールドにすると
    ``meta.json`` が変わり、既存の成果物の指紋が壊れる。判別子は「どの型を
    作るか」の情報であって、その型の設定値ではない。
    """

    KIND: ClassVar[str] = "esn"

    n_units: int = 200
    spectral_radius: float = 0.9
    leak_rate: float = 0.3
    input_scale: float = 0.5
    bias_scale: float = 0.1
    density: float = 0.1
    activation: str = TANH
    state_noise: float = 0.0


def spectral_radius(matrix: FloatArray) -> float:
    """行列のスペクトル半径 (固有値の絶対値の最大)。

    ``numpy.linalg.eigvals`` (密・決定論的) を使う。ARPACK は反復初期値で微小に
    揺れて再現性が落ちるため、N <= 1000 の本連載規模では密で押す (仕様 §3)。
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"正方行列が必要です: {matrix.shape}")
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def _validate_config(config: ESNConfig) -> None:
    if config.n_units < 1:
        raise ValueError(f"n_units は 1 以上である必要があります: {config.n_units}")
    if config.activation != TANH:
        raise ValueError(f"未対応の活性化関数です: {config.activation!r}")
    if not 0.0 < config.density <= 1.0:
        raise ValueError(f"density は (0, 1] である必要があります: {config.density}")
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


class ESN:
    """漏れ積分型 tanh リザバー。

    重みは ``__init__`` で1度だけ引き、以後は読み取り専用にする
    (学習は読み出し層 ``rc_basics_lab.readout`` 側だけで行う)。
    """

    def __init__(
        self,
        config: ESNConfig,
        rng: np.random.Generator,
        *,
        n_inputs: int = 1,
    ) -> None:
        """重みを生成する。

        Args:
            config: 構造ハイパーパラメータ。
            rng: 重み生成用の Generator (``seeds.make_rng`` の reservoir ストリーム)。
            n_inputs: 入力次元 D_in。課題側が決める量なので YAML ではなくここで渡す。

        Raises:
            ValueError: 設定値が範囲外、または生成した W が零行列だった場合。
        """
        _validate_config(config)
        if n_inputs < 1:
            raise ValueError(f"n_inputs は 1 以上である必要があります: {n_inputs}")
        self._config = config
        self._n_inputs = n_inputs

        n_units = config.n_units
        # W_in: (N, 1 + D_in)。先頭列が定数入力 1 に対応する。
        weights_in: FloatArray = np.empty((n_units, 1 + n_inputs), dtype=np.float64)
        weights_in[:, 0] = rng.uniform(-config.bias_scale, config.bias_scale, n_units)
        weights_in[:, 1:] = rng.uniform(
            -config.input_scale, config.input_scale, (n_units, n_inputs)
        )

        # W: 密度 density のスパース構造を密行列として保持する (N <= 1000 なので
        # scipy.sparse を持ち込むより単純で、eigvals もそのまま使える)。
        mask = rng.random((n_units, n_units)) < config.density
        values: FloatArray = rng.uniform(-1.0, 1.0, (n_units, n_units))
        recurrent: FloatArray = np.where(mask, values, 0.0)
        measured = spectral_radius(recurrent)
        if measured == 0.0:
            raise ValueError(
                "生成した W のスペクトル半径が 0 です "
                f"(density={config.density}, n_units={n_units})"
            )
        recurrent *= config.spectral_radius / measured

        weights_in.setflags(write=False)
        recurrent.setflags(write=False)
        self._weights_in = weights_in
        self._recurrent = recurrent

    @property
    def config(self) -> ESNConfig:
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
        """入力重み ``(N, 1 + D_in)``。読み取り専用 (名前は更新式の記号に合わせる)。"""
        return self._weights_in

    @property
    def W(self) -> FloatArray:
        """再帰重み ``(N, N)``。読み取り専用 (名前は更新式の記号に合わせる)。"""
        return self._recurrent

    def initial_state(self) -> FloatArray:
        """既定の初期状態 (零ベクトル)。"""
        return np.zeros(self._config.n_units, dtype=np.float64)

    def _input_drive(self, u: FloatArray) -> FloatArray:
        """``W_in @ [1; u]``。"""
        return self._weights_in[:, 0] + self._weights_in[:, 1:] @ u

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
        """1ステップ更新して次状態を返す (04 の閉ループ用)。

        Args:
            x: 現在の状態 ``(N,)``。
            u: 現在の入力 ``(D_in,)``。
            rng: ``state_noise > 0`` のときに必要なノイズ用 Generator。

        Raises:
            ValueError: 形状不整合、または ``state_noise > 0`` かつ ``rng is None``。
        """
        state = self._check_state(x)
        inputs = np.asarray(u, dtype=np.float64)
        if inputs.shape != (self._n_inputs,):
            raise ValueError(
                f"入力は ({self._n_inputs},) である必要があります: {inputs.shape}"
            )
        return self._update(state, self._input_drive(inputs), rng)

    def run(
        self,
        u: FloatArray,
        x0: FloatArray | None = None,
        rng: np.random.Generator | None = None,
    ) -> FloatArray:
        """入力系列を流して状態系列を返す。

        Args:
            u: 入力系列 ``(T, D_in)``。1次元は受理しない (診断層と同じ規約)。
            x0: 初期状態 ``(N,)``。``None`` なら零ベクトル (02 の2初期状態用)。
            rng: ``state_noise > 0`` のときに必要なノイズ用 Generator。

        Returns:
            状態系列 ``(T, N)``。``X[t]`` は ``u[t]`` を処理した**後**の状態。
        """
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
        for index in range(n_steps):
            state = self._update(state, self._input_drive(inputs[index]), rng)
            states[index] = state
        return states


__all__ = ["ESN", "TANH", "ESNConfig", "spectral_radius"]

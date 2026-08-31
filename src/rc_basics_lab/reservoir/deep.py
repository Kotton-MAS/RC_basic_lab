"""深層 ESN (層を積んだリザバー).

Gallicchio & Micheli (2017) "Deep Echo State Network (DeepESN)" の構成。
層を積むと**層ごとに違う時間スケールが自然に現れる** (深い層ほど遅い) という
主張で、単層の ESN に対する対照になる。

構成:

- 第1層は外部入力 ``u[t]`` を受ける
- 第 l 層 (l >= 2) は**直前の層の状態** ``x^(l-1)[t]`` を入力として受ける
  (同時刻。層をまたぐ遅延は入れない)
- 状態は全層の連結 ``[x^(1); x^(2); ...; x^(L)]``

``n_units`` は**連結後の総次元**である。層ごとの数ではない —— 読み出し層も
診断層も「状態の次元」として ``n_units`` を読み、容量の上限 ``MC <= N`` も
その N を指すためである。``n_layers`` で割り切れることを検査する。

**層ごとにリーク率を変えない。** 原論文は層ごとに違う値を許すが、ここでは
1つの値を全層で共有する。時間スケールの分化が「リーク率を層ごとに設定した
から」ではなく**積んだこと自体から出る**ことを見たいためで、層ごとに振ると
その区別がつかなくなる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from rc_basics_lab.reservoir._kernel import (
    check_common_config,
    check_state,
    leaky_tanh_update,
)
from rc_basics_lab.reservoir.esn import spectral_radius
from rc_basics_lab.reservoir.topology import (
    ErdosRenyiConfig,
    TopologyConfig,
    build_mask,
    nominal_density,
)
from rc_basics_lab.types import FloatArray


@dataclass(frozen=True, slots=True)
class DeepESNConfig:
    """深層 ESN の構造ハイパーパラメータ。

    Attributes:
        n_units: **連結後の総ユニット数** N。``n_layers`` で割り切れること。
        n_layers: 層の数 L。1 なら単層の ESN と同じ構造になる。
        spectral_radius: 各層の再帰行列のスペクトル半径 (全層共通)。
        leak_rate: 漏れ率 a (全層共通。上の注を参照)。
        input_scale: 第1層が外部入力に掛ける重みの幅。
        inter_layer_scale: 第 l 層が直前の層の状態に掛ける重みの幅。
        bias_scale: 定数入力に対応する重みの幅 (全層共通)。
        topology: 各層の結合構造 (既定は密度 0.1 の Erdos-Renyi)。**層ごとに
            独立に生成**するので、同じ設定でも層ごとに違う実現になる。
        state_noise: tanh 内部に加えるガウスノイズの標準偏差。
    """

    KIND: ClassVar[str] = "deep_esn"

    n_units: int = 200
    n_layers: int = 2
    spectral_radius: float = 0.9
    leak_rate: float = 0.3
    input_scale: float = 0.5
    inter_layer_scale: float = 0.5
    bias_scale: float = 0.1
    topology: TopologyConfig = field(default_factory=ErdosRenyiConfig)
    state_noise: float = 0.0


def _validate_deep_config(config: DeepESNConfig, n_inputs: int) -> None:
    """設定値の範囲を検査する。"""
    if config.n_layers < 1:
        raise ValueError(f"n_layers は 1 以上である必要があります: {config.n_layers}")
    if config.n_units < config.n_layers:
        raise ValueError(
            f"n_units は n_layers 以上である必要があります: "
            f"{config.n_units} < {config.n_layers}"
        )
    if config.n_units % config.n_layers != 0:
        raise ValueError(
            "n_units は n_layers で割り切れる必要があります "
            f"({config.n_units} / {config.n_layers})。"
            "層ごとの数を暗黙に丸めると、設定した総次元と実際がずれる"
        )
    check_common_config(
        n_units=config.n_units,
        min_units=1,
        leak_rate=config.leak_rate,
        spectral_radius=config.spectral_radius,
        scales={
            "input_scale": config.input_scale,
            "inter_layer_scale": config.inter_layer_scale,
            "bias_scale": config.bias_scale,
        },
        state_noise=config.state_noise,
        n_inputs=n_inputs,
    )
    layer_units = config.n_units // config.n_layers
    # 層が小さいと density * layer_units が 1 を割り、W が冪零 (再帰の無い
    # リザバー) になる確率が無視できなくなる。**シード次第で通ったり落ちたり
    # する**のは設定として最悪なので、条件そのものを先に落とす。
    # 実測: n_units=12 / n_layers=3 / density=0.1 は layer_units=4 で
    # 期待非零が 1.6、シード 0 で冪零になった。
    density = nominal_density(config.topology, layer_units)
    if density * layer_units < 1.0:
        raise ValueError(
            f"density * (n_units / n_layers) は 1 以上が必要です "
            f"({density} * {layer_units} = {density * layer_units:.2f})。"
            "層あたりのユニット数が少なすぎるか density が低すぎます —— "
            "この条件では再帰の無い W (冪零) がシード次第で生まれます。"
            "n_layers を減らすか density を上げてください"
        )
    if not 0.0 < config.leak_rate <= 1.0:
        raise ValueError(
            f"leak_rate は (0, 1] である必要があります: {config.leak_rate}"
        )


def _random_recurrent(
    n_units: int,
    topology: TopologyConfig,
    target_radius: float,
    rng: np.random.Generator,
) -> FloatArray:
    """1層ぶんの再帰行列 (``ESN`` と同じ作り方・同じ引き方)。

    結合の有無は ``topology`` 層が決め、値はここで引く (拡張性方針 §2-1)。
    **層ごとに独立に引く**ので、同じトポロジ設定でも層ごとに違う実現になる。
    """
    mask = build_mask(topology, n_units, rng)
    values: FloatArray = rng.uniform(-1.0, 1.0, (n_units, n_units))
    recurrent: FloatArray = np.where(mask, values, 0.0)
    measured = spectral_radius(recurrent)
    if measured == 0.0:
        raise ValueError(
            f"生成した W のスペクトル半径が 0 です (topology={topology}, N={n_units})"
        )
    return recurrent * (target_radius / measured)


class DeepESN:
    """層を積んだ漏れ積分 tanh リザバー (``Reservoir`` を満たす)。

    ``run`` が返すのは**全層を連結した** ``(T, n_units)`` である。
    層ごとの状態を見たいときは ``layer_slice`` で切り出す。
    """

    def __init__(
        self,
        config: DeepESNConfig,
        rng: np.random.Generator,
        *,
        n_inputs: int = 1,
    ) -> None:
        """層ごとの重みを生成する。

        重みは**第1層から順に**引く。順序を変えると同じシードでも別の重みに
        なるので、層の数を増やしたときに前の層が変わらないよう浅い側から引く。

        Args:
            config: 構造ハイパーパラメータ。
            rng: 重み生成用の Generator。
            n_inputs: 入力次元 D_in。

        Raises:
            ValueError: 設定値が範囲外、または生成した W が零行列だった場合。
        """
        _validate_deep_config(config, n_inputs)
        self._config = config
        self._n_inputs = n_inputs
        self._layer_units = config.n_units // config.n_layers

        weights_in: list[FloatArray] = []
        recurrent: list[FloatArray] = []
        for layer in range(config.n_layers):
            # 第1層だけ外部入力、以降は直前の層の状態を受ける。
            fan_in = n_inputs if layer == 0 else self._layer_units
            scale = config.input_scale if layer == 0 else config.inter_layer_scale
            block: FloatArray = np.empty(
                (self._layer_units, 1 + fan_in), dtype=np.float64
            )
            block[:, 0] = rng.uniform(
                -config.bias_scale, config.bias_scale, self._layer_units
            )
            block[:, 1:] = rng.uniform(-scale, scale, (self._layer_units, fan_in))
            block.setflags(write=False)
            weights_in.append(block)
            matrix = _random_recurrent(
                self._layer_units,
                config.topology,
                config.spectral_radius,
                rng,
            )
            matrix.setflags(write=False)
            recurrent.append(matrix)
        self._weights_in = tuple(weights_in)
        self._recurrent = tuple(recurrent)

    @property
    def config(self) -> DeepESNConfig:
        """生成に使った設定。"""
        return self._config

    @property
    def n_units(self) -> int:
        """**連結後の**ユニット数 N。"""
        return self._config.n_units

    @property
    def n_inputs(self) -> int:
        """入力次元 D_in。"""
        return self._n_inputs

    @property
    def n_layers(self) -> int:
        """層の数 L。"""
        return self._config.n_layers

    def layer_slice(self, layer: int) -> slice[int, int, int]:
        """連結状態から第 ``layer`` 層 (0 始まり) を切り出す ``slice``。

        層ごとの時間スケールを見たいとき (この構成の主張そのもの) に使う。

        Raises:
            IndexError: 層の番号が範囲外の場合。
        """
        if not 0 <= layer < self._config.n_layers:
            raise IndexError(f"層は 0..{self._config.n_layers - 1} です: {layer}")
        start = layer * self._layer_units
        return slice(start, start + self._layer_units)

    def initial_state(self) -> FloatArray:
        """既定の初期状態 (零ベクトル、連結後の長さ)。"""
        return np.zeros(self._config.n_units, dtype=np.float64)

    def _check_state(self, x: FloatArray) -> FloatArray:
        return check_state(x, self._config.n_units)

    def step(
        self,
        x: FloatArray,
        u: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> FloatArray:
        """1ステップ更新して連結状態を返す。

        **浅い層から順に更新し、更新後の値を次の層へ渡す** (同時刻の結合)。
        """
        state = self._check_state(x)
        inputs = np.asarray(u, dtype=np.float64)
        if inputs.shape != (self._n_inputs,):
            raise ValueError(
                f"入力は ({self._n_inputs},) である必要があります: {inputs.shape}"
            )
        updated: FloatArray = np.empty_like(state)
        signal = inputs
        for layer in range(self._config.n_layers):
            span = self.layer_slice(layer)
            block = self._weights_in[layer]
            # 層ごとに入力側の寄与を作り、更新式は 3 モデル共通の ``_kernel``
            # へ渡す。第 l 層の入力は直前の層の**更新後**の状態である。
            updated[span] = leaky_tanh_update(
                state[span],
                block[:, 0] + block[:, 1:] @ signal,
                self._recurrent[layer],
                leak_rate=self._config.leak_rate,
                state_noise=self._config.state_noise,
                rng=rng,
            )
            signal = updated[span]
        return updated

    def run(
        self,
        u: FloatArray,
        x0: FloatArray | None = None,
        rng: np.random.Generator | None = None,
    ) -> FloatArray:
        """入力系列 ``(T, D_in)`` を流して連結状態系列 ``(T, N)`` を返す。"""
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
            state = self.step(state, inputs[index], rng)
            states[index] = state
        return states


__all__ = ["DeepESN", "DeepESNConfig"]

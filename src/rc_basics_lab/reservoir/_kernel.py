"""リザバー実装が共有する更新式と検査 (非公開モジュール).

3 つのモデル (``esn`` / ``ring`` / ``deep``) は**構造だけが違い、状態の
進め方は同じ**である —— 漏れ積分つきの tanh で、``state_noise`` があれば
活性化の内側にガウス雑音を足す。実測で ``_update`` と ``_check_state`` は
3 モデル間で **100% 同一**だった。

**同じコードが3箇所にあると3箇所が独立にドリフトする。** 更新式の不具合を
1つ直すと、残り2つに残る。ここに1つだけ置く。

``diagnostics`` を import しない (``reservoir`` 層の規約)。numpy 以外に依存
しないので、新しいモデルはここを呼ぶだけで更新式を揃えられる。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from rc_basics_lab.types import FloatArray

NOISE_WITHOUT_RNG = (
    "state_noise > 0 のときは rng が必要です "
    "(黙ってノイズ無しにすると設定が効かない実験になる)"
)
"""``state_noise`` があるのに ``rng`` が無いときの文言 (D-36)。

3 モデルで同じ文言にする。片方だけ変えると、同じ間違いに違う説明が出る。
"""


def leaky_tanh_update(
    state: FloatArray,
    drive: FloatArray,
    recurrent: FloatArray,
    *,
    leak_rate: float,
    state_noise: float,
    rng: np.random.Generator | None,
) -> FloatArray:
    """漏れ積分つき tanh の1ステップ (**更新式の唯一の実装**)。

    ``x' = (1 - a) x + a * tanh(drive + W x + noise)``。雑音は活性化の**内側**に
    足す —— 外側に足すと ``|x| <= 1`` が壊れ、診断層が前提にしている有界性が
    失われる。

    演算の順序は変えないこと。``results/`` の成果物はこの順序で作られており、
    数学的に等価な書き換えでも浮動小数の丸めで最終桁が動く (D-74)。

    Args:
        state: 現在の状態 ``(N,)``。
        drive: 入力側の寄与 ``W_in @ [1; u]`` ``(N,)``。
        recurrent: 再帰重み ``(N, N)``。
        leak_rate: 漏れ率 a。
        state_noise: 活性化の内側に足すガウス雑音の標準偏差。
        rng: ``state_noise > 0`` のときに必要な Generator。

    Returns:
        次の状態 ``(N,)``。

    Raises:
        ValueError: ``state_noise > 0`` かつ ``rng is None`` の場合 (D-36)。
    """
    pre_activation = drive + recurrent @ state
    if state_noise > 0.0:
        if rng is None:
            raise ValueError(NOISE_WITHOUT_RNG)
        pre_activation = pre_activation + state_noise * rng.standard_normal(
            state.shape[0]
        )
    activated: FloatArray = np.tanh(pre_activation)
    return (1.0 - leak_rate) * state + leak_rate * activated


def check_state(x: FloatArray, n_units: int) -> FloatArray:
    """状態を ``(N,)`` の ``float64`` に正規化する。

    Raises:
        ValueError: 形が ``(n_units,)`` でない場合。
    """
    state = np.asarray(x, dtype=np.float64)
    if state.shape != (n_units,):
        raise ValueError(f"状態は ({n_units},) である必要があります: {state.shape}")
    return state


def check_input_series(u: FloatArray, n_inputs: int) -> FloatArray:
    """入力系列を ``(T, D_in)`` の ``float64`` に正規化する。

    1次元は受理しない (診断層と同じ規約 —— ``(T,)`` を ``(T, 1)`` と
    黙って解釈すると、多次元入力の取り違えが形の検査をすり抜ける)。

    Raises:
        ValueError: 2次元でない / 入力次元が違う / 空の場合。
    """
    inputs = np.asarray(u, dtype=np.float64)
    if inputs.ndim != 2:
        raise ValueError(f"u は (T, D_in) の2次元配列が必要です: {inputs.shape}")
    if inputs.shape[1] != n_inputs:
        raise ValueError(f"入力次元が一致しません: {inputs.shape[1]} != {n_inputs}")
    if inputs.shape[0] == 0:
        raise ValueError("u が空です")
    return inputs


def run_series(
    u: FloatArray,
    *,
    n_inputs: int,
    n_units: int,
    x0: FloatArray | None,
    advance: Callable[[FloatArray, int], FloatArray],
) -> FloatArray:
    """入力系列を流して状態系列を返す共通ループ。

    検査・初期状態・確保・ループを 1 か所に置く。モデルが渡すのは
    ``advance(state, index) -> next_state`` だけで、**1 ステップの進め方だけが
    モデルごとに違う**。

    ``advance`` が index を受け取るのは、入力側の寄与を前計算するモデル
    (``ESN`` / ``Ring`` は ``inputs @ W_in.T`` を 1 回で作る) がその表を引ける
    ようにするためである。行を渡す形にすると前計算が使えず、逐次の行列積が
    T 回走る。

    Args:
        u: 入力系列 ``(T, D_in)``。
        n_inputs: 期待する入力次元。
        n_units: 状態の次元 (返り値の列数)。
        x0: 初期状態。``None`` なら零ベクトル。
        advance: 1 ステップ進める関数。

    Returns:
        状態系列 ``(T, n_units)``。``X[t]`` は ``u[t]`` を処理した**後**の状態。

    Raises:
        ValueError: 入力の形が合わない / 初期状態の形が合わない場合。
    """
    inputs = check_input_series(u, n_inputs)
    state = (
        np.zeros(n_units, dtype=np.float64) if x0 is None else check_state(x0, n_units)
    )
    states: FloatArray = np.empty((inputs.shape[0], n_units), dtype=np.float64)
    for index in range(inputs.shape[0]):
        state = advance(state, index)
        states[index] = state
    return states


def check_common_config(
    *,
    n_units: int,
    min_units: int,
    leak_rate: float,
    spectral_radius: float,
    scales: dict[str, float],
    state_noise: float,
    n_inputs: int,
) -> None:
    """3 モデルが共通して持つ設定値の範囲を検査する。

    モデル固有のもの (``density`` / ``n_layers`` / ``activation``) は各モデルが
    自分で見る。**共通のものをここに集めるのは、同じ間違いに同じ説明を出す
    ため**である。

    Args:
        n_units: ユニット数。
        min_units: 許す最小値 (リングは閉路なので 2)。
        leak_rate: 漏れ率。
        spectral_radius: スペクトル半径 (リングでは閉路の重み)。
        scales: 名前 -> 値。0 以上であること (``input_scale`` など)。
        state_noise: 状態雑音。
        n_inputs: 入力次元。

    Raises:
        ValueError: いずれかが範囲外の場合。
    """
    if n_units < min_units:
        raise ValueError(f"n_units は {min_units} 以上である必要があります: {n_units}")
    if not 0.0 < leak_rate <= 1.0:
        raise ValueError(f"leak_rate は (0, 1] である必要があります: {leak_rate}")
    if spectral_radius <= 0.0:
        raise ValueError(f"spectral_radius は正である必要があります: {spectral_radius}")
    for name, value in scales.items():
        if value < 0.0:
            raise ValueError(f"{name} は 0 以上である必要があります: {value}")
    if state_noise < 0.0:
        raise ValueError(f"state_noise は 0 以上である必要があります: {state_noise}")
    if n_inputs < 1:
        raise ValueError(f"n_inputs は 1 以上である必要があります: {n_inputs}")


__all__ = [
    "NOISE_WITHOUT_RNG",
    "check_common_config",
    "check_input_series",
    "check_state",
    "leaky_tanh_update",
    "run_series",
]

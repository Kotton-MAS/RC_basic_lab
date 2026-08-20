"""自由走行 (closed-loop) —— 学習済み read-out の出力を次時刻の入力へ戻す実行系.

**このモジュールは ``rc_basics_lab.reservoir`` を import しない** (D-50)。状態を
1ステップ進める写像を ``StateUpdater`` プロトコルで受け取る。

自走は**教師強制で学習した read-out 係数をそのまま使う** (D-44)。返り値の
``coefficients`` は**渡された配列そのもの** (同一オブジェクト) なので、内部で
学習し直していないことを呼び出し側が同一性で検査できる。

**自走を ``ESN.run`` で書いてはならない** (仕様 §5 禁止する構造8)。``run`` は
入力系列が既知の区間専用であり、自走では ``u[t+1]`` が ``y_hat[t]`` に
依存するため、逐次ループ (ESN なら ``ESN.step``) 以外では書けない。
ベクトル化できないことは自走の性質であって最適化の余地ではない (仕様 §10-1)。

**乱数の扱い (D-48 と D-36 の境界)**: 02 の伝播器 (``esn_propagator``) は
**決定的でなければならない** (D-48) —— 条件付き Lyapunov 指数は「同じ軌道の
まわりの摂動」の成長率を測るので、伝播器がノイズを引くと測っている量が
変わってしまう。一方**自走は伝播器ではなく軌道を作る呼び出し**であり、
学習時の状態にノイズを入れた設定 (``ESNConfig.state_noise > 0``) では、自走中も
同じ分布のノイズが乗っていなければ学習時と評価時で系が変わる。したがって
自走の ``StateUpdater`` が ESN のとき、``state_noise > 0`` なら
``ESN.step(x, u, rng=...)`` に **rng を渡すのが正しい** (D-36 の側)。
D-48 を理由に rng を外すと、ノイズ注入の効果 (要件書 設計判断3) を測る
実験がノイズ無しの系を測ることになる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rc_basics_lab.readout.design import (
    FeatureSpec,
    build_design_matrix,
    first_valid_for,
)
from rc_basics_lab.readout.ridge import predict
from rc_basics_lab.types import FloatArray


class StateUpdater(Protocol):
    """状態を1ステップ進める写像。

    契約: ``updater(x, u)`` は「時刻 ``t-1`` の状態 ``x`` に時刻 ``t`` の入力
    ``u`` を与えた後の状態 ``x[t]``」を返す。``ESN.step(x, u, rng)`` と同じ
    向きで、``reservoir`` を import しない構造的型付け (Protocol) なので、
    ESN 以外の状態生成器 (外部シミュレータ・実素子) をそのまま渡せる (D-50)。

    Args:
        x: 現在の状態 ``(N,)``。
        u: 与える入力 ``(D_in,)``。

    Returns:
        次状態 ``(N,)``。
    """

    def __call__(self, x: FloatArray, u: FloatArray) -> FloatArray: ...


@dataclass(frozen=True, slots=True)
class FreeRunResult:
    """自走1本ぶんの結果。

    Attributes:
        inputs: 各ステップで**実際に与えた**入力 ``(n_steps, D_in)``。
            先頭行だけが呼び出し側の与えた ``u0`` で、以降は1つ前の予測。
        states: 各ステップ後の状態 ``(n_steps, N)``。
        predictions: read-out の出力 ``(n_steps, D_out)``。
        coefficients: 使った read-out 係数。**渡された配列そのもの** (D-44 の
            同一性検査の対象)。
        n_completed: 有限値で埋まった行数。``diverged`` なら ``n_steps`` 未満で、
            それ以降の行は ``nan`` のまま残る。
        diverged: 途中で有限でない値が出て打ち切ったか。

    Note:
        発散は自走の**結果の1つ** (4-C の3態分類の1態) であって異常では
        ないので例外にしない。ただし打ち切りを無かったことにもしない ——
        ``n_completed`` と ``diverged`` を必ず残し、残りの行は 0 ではなく
        ``nan`` にする (0 埋めにすると「静かに真値へ近い予測」に化ける)。
    """

    inputs: FloatArray
    states: FloatArray
    predictions: FloatArray
    coefficients: FloatArray
    n_completed: int
    diverged: bool


def _validate_free_run_inputs(
    spec: FeatureSpec,
    coefficients: FloatArray,
    x0: FloatArray,
    u0: FloatArray,
    n_steps: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """形状と、自走が成り立つ前提を検査する。"""
    if n_steps < 1:
        raise ValueError(f"n_steps は 1 以上である必要があります: {n_steps}")
    lag = first_valid_for(spec)
    if lag != 0:
        raise ValueError(
            "自走は u[t] と x[t] だけから特徴を組める仕様にしか対応しません "
            f"(ラグ履歴を要する仕様は閉ループに乗らない): first_valid={lag}"
        )
    weights = np.asarray(coefficients, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError(
            f"coefficients は (F, D_out) の2次元配列が必要です: {weights.shape}"
        )
    state = np.asarray(x0, dtype=np.float64)
    if state.ndim != 1:
        raise ValueError(f"x0 は (N,) の1次元配列が必要です: {state.shape}")
    inputs = np.asarray(u0, dtype=np.float64)
    if inputs.ndim != 1:
        raise ValueError(f"u0 は (D_in,) の1次元配列が必要です: {inputs.shape}")
    if weights.shape[1] != inputs.shape[0]:
        raise ValueError(
            "read-out の出力次元が入力次元と一致しません "
            "(自走は出力をそのまま次時刻の入力へ戻すので一致が必要): "
            f"D_out={weights.shape[1]}, D_in={inputs.shape[0]}"
        )
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(inputs)):
        raise ValueError("自走の初期状態・初期入力に有限でない値があります")
    return weights, state, inputs


def free_run(
    updater: StateUpdater,
    spec: FeatureSpec,
    coefficients: FloatArray,
    x0: FloatArray,
    u0: FloatArray,
    n_steps: int,
) -> FreeRunResult:
    """出力を入力へ戻しながら ``n_steps`` ステップ自走させる (D-44 / D-50)。

    各ステップは::

        x[k] = updater(x[k-1], u[k])
        y[k] = readout(u[k], x[k])
        u[k+1] = y[k]

    で、``u[0]`` (= ``u0``) と ``x[-1]`` (= ``x0``) を呼び出し側が与える。
    ``x0`` は教師強制 (ウォームアップ) の最終状態、``u0`` は切り替え点で
    自走が最初に食う入力である。

    特徴の並びは ``build_design_matrix`` を毎ステップ通して組む。学習時と同じ
    関数を通すので、「教師強制と自走で特徴の並びが違う」(仕様 §5 禁止する構造2)
    が構造上起きない。

    Args:
        updater: 状態を1ステップ進める写像 (``StateUpdater``)。
        spec: 学習に使ったのと**同じ** ``FeatureSpec``。
        coefficients: 教師強制で学習した係数 ``(F, D_out)``。ここでは学習し直さない。
        x0: ウォームアップ終了時の状態 ``(N,)``。
        u0: 自走の最初の入力 ``(D_in,)``。
        n_steps: 自走させるステップ数。

    Returns:
        ``FreeRunResult``。``coefficients`` は渡された配列そのもの (D-44)。

    Raises:
        ValueError: 形状不整合、``D_out != D_in``、``n_steps < 1``、
            ラグ履歴を要する ``spec``、または特徴数と係数の行数が合わない場合。
    """
    weights, state, inputs = _validate_free_run_inputs(
        spec, coefficients, x0, u0, n_steps
    )
    n_units = state.shape[0]
    n_inputs = inputs.shape[0]
    n_outputs = weights.shape[1]

    input_log: FloatArray = np.full((n_steps, n_inputs), np.nan, dtype=np.float64)
    state_log: FloatArray = np.full((n_steps, n_units), np.nan, dtype=np.float64)
    prediction_log: FloatArray = np.full((n_steps, n_outputs), np.nan, dtype=np.float64)

    n_completed = 0
    diverged = False
    for index in range(n_steps):
        state = np.asarray(updater(state, inputs), dtype=np.float64)
        if state.shape != (n_units,):
            raise ValueError(
                "updater の戻り値の形状が状態と一致しません: "
                f"{state.shape} != {(n_units,)}"
            )
        design = build_design_matrix(
            spec, inputs.reshape(1, n_inputs), state.reshape(1, n_units)
        )
        if design.phi.shape[1] != weights.shape[0]:
            raise ValueError(
                "設計行列の特徴数と係数の行数が一致しません "
                "(学習時と違う FeatureSpec を渡していませんか): "
                f"F={design.phi.shape[1]}, coefficients={weights.shape[0]}"
            )
        prediction: FloatArray = predict(design.phi, weights)[0]
        input_log[index] = inputs
        if not (np.all(np.isfinite(state)) and np.all(np.isfinite(prediction))):
            # 発散は結果の1つ。ここで止めるが、無かったことにはしない。
            diverged = True
            break
        state_log[index] = state
        prediction_log[index] = prediction
        n_completed = index + 1
        inputs = prediction

    return FreeRunResult(
        inputs=input_log,
        states=state_log,
        predictions=prediction_log,
        coefficients=coefficients,
        n_completed=n_completed,
        diverged=diverged,
    )


__all__ = ["FreeRunResult", "StateUpdater", "free_run"]

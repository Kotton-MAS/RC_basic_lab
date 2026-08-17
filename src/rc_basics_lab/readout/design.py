"""設計行列の構築 — 3ベースラインを分ける唯一の場所.

受け入れ条件1 (「3ベースラインが同一 API で切り替わる」)の本体。
線形 / 遅延線 / リザバーの違いは **``FeatureSpec`` の差だけ**であり、
``build_design_matrix`` という単一の呼び出し口を通る。手法ごとに別関数を用意すると
「実装差ではなく設計行列の差」という主張が崩れるため、内部でも仕様を
``_Layout`` (どのラグの入力を並べるか / 状態を使うか)へ正規化してから、
組み立て経路は1本に合流させている。

``first_valid`` は「その行以降なら全特徴が定義済み」という境界。遅延線では
``n_lags`` になる。実験ランナーは全手法の ``first_valid`` と washout の最大値を
単一の ``t0`` とし、全手法をまったく同じ行集合で学習・評価する (D-05)。
``first_valid`` より手前の行は **NaN** で埋める。0 埋めにすると、``t0`` の取り違えが
「少しずれた学習結果」として静かに通ってしまうため。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rc_basics_lab.types import FloatArray

BIAS_NAME = "bias"
"""バイアス列の特徴名。``fit_ridge`` の無罰則列の判定に使う (D-03)。"""


@dataclass(frozen=True, slots=True)
class PassthroughSpec:
    """線形ベースライン: ``[1, u[t]]``。"""

    bias: bool = True


@dataclass(frozen=True, slots=True)
class DelayLineSpec:
    """遅延線ベースライン: ``[1, u[t], u[t-1], ..., u[t-n_lags]]``。"""

    n_lags: int
    bias: bool = True


@dataclass(frozen=True, slots=True)
class ReservoirSpec:
    """リザバー読み出し: ``[1, u[t], x[t]]``。"""

    include_input: bool = True
    bias: bool = True


type FeatureSpec = PassthroughSpec | DelayLineSpec | ReservoirSpec
"""3ベースラインを切り替える唯一の軸。"""


@dataclass(frozen=True, slots=True)
class DesignMatrix:
    """構築された設計行列。

    Attributes:
        phi: ``(T, F)``。``first_valid`` より手前の行は NaN。
        first_valid: 全特徴が定義済みになる最初の行 index。
        feature_names: 列名 ``(F,)``。``phi.shape[1]`` と必ず一致する。
    """

    phi: FloatArray
    first_valid: int
    feature_names: tuple[str, ...]

    @property
    def bias_column(self) -> int | None:
        """バイアス列の index (``fit_ridge`` / ``select_alpha`` の無罰則列に渡す)。

        ``bias_column_index(self.feature_names)`` と同じ。呼び出し側が
        ``feature_names`` から自前で index を導出する必要をなくし、
        渡し忘れを型で落とすための入口 (F-1-002)。
        """
        return bias_column_index(self.feature_names)


@dataclass(frozen=True, slots=True)
class _Layout:
    """``FeatureSpec`` を「どの列を並べるか」に正規化した中間表現。"""

    bias: bool
    input_lags: tuple[int, ...]
    use_states: bool


def _layout_of(spec: FeatureSpec) -> _Layout:
    """仕様を ``_Layout`` に正規化する (手法ごとの分岐はここだけ)。"""
    match spec:
        case PassthroughSpec():
            return _Layout(bias=spec.bias, input_lags=(0,), use_states=False)
        case DelayLineSpec():
            if spec.n_lags < 0:
                raise ValueError(f"n_lags は 0 以上である必要があります: {spec.n_lags}")
            lags = tuple(range(spec.n_lags + 1))
            return _Layout(bias=spec.bias, input_lags=lags, use_states=False)
        case ReservoirSpec():
            lags = (0,) if spec.include_input else ()
            return _Layout(bias=spec.bias, input_lags=lags, use_states=True)


def first_valid_for(spec: FeatureSpec) -> int:
    """``build_design_matrix`` が返す ``first_valid`` を、系列を作らずに求める。

    ``first_valid`` は**ラグの構造だけ**で決まり、系列の中身にも長さにも
    依存しない。実験 2-D (washout 感度) は系列を生成する**前に**
    ``t0 = compute_t0(first_valid の列, washout)`` を知る必要がある (D-19: washout
    を増やしても訓練行数が変わらないよう系列長で補償するため)。

    その予測を「遅延線なら ``n_lags``」と別の場所へ書き写すと、手法を足した
    ときに予測と実際の ``t0`` が黙って食い違い、補償が効かなくなる。単一の
    真実を ``_layout_of`` に置いたまま外から引けるようにするのがこの関数。

    Args:
        spec: ``PassthroughSpec`` / ``DelayLineSpec`` / ``ReservoirSpec``。

    Returns:
        全特徴が定義済みになる最初の行 index。

    Raises:
        ValueError: ``DelayLineSpec.n_lags`` が負の場合。
    """
    return max(_layout_of(spec).input_lags, default=0)


def bias_column_index(feature_names: tuple[str, ...]) -> int | None:
    """バイアス列の index。無ければ ``None`` (``fit_ridge`` の罰則行列に渡す)。"""
    return 0 if feature_names[:1] == (BIAS_NAME,) else None


def _validate_inputs(
    spec: FeatureSpec, u: FloatArray, states: FloatArray | None
) -> tuple[FloatArray, FloatArray | None, _Layout, int]:
    """形状検証だけをまとめる (F-1-015: ブロック組み立てと分岐を分離)。

    Args:
        spec: ``FeatureSpec``。
        u: 入力系列 ``(T, D_in)``。
        states: リザバー状態 ``(T, N)`` または ``None``。

    Returns:
        ``(inputs, state_array, layout, first_valid)``。
        ``build_design_matrix`` はこれをそのままブロック組み立てに渡す。

    Raises:
        ValueError: 形状不整合、系列長不足、または ``ReservoirSpec`` に
            ``states=None``。
    """
    inputs = np.asarray(u, dtype=np.float64)
    if inputs.ndim != 2:
        raise ValueError(f"u は (T, D_in) の2次元配列が必要です: {inputs.shape}")
    n_steps, n_inputs = inputs.shape
    if n_steps == 0 or n_inputs == 0:
        raise ValueError(f"u が空です: {inputs.shape}")

    layout = _layout_of(spec)
    # 予測経路 (``first_valid_for``) と実経路が同じ値を返すことを構造で保証する。
    first_valid = first_valid_for(spec)
    if first_valid >= n_steps:
        raise ValueError(
            f"系列長がラグに対して短すぎます: T={n_steps}, first_valid={first_valid}"
        )

    state_array: FloatArray | None = None
    if layout.use_states:
        if states is None:
            raise ValueError("ReservoirSpec には states が必要です")
        state_array = np.asarray(states, dtype=np.float64)
        if state_array.ndim != 2:
            raise ValueError(
                f"states は (T, N) の2次元配列が必要です: {state_array.shape}"
            )
        if state_array.shape[0] != n_steps:
            raise ValueError(
                f"states の行数が u と一致しません: {state_array.shape[0]} != {n_steps}"
            )
    return inputs, state_array, layout, first_valid


def _assemble_blocks(
    layout: _Layout, inputs: FloatArray, state_array: FloatArray | None
) -> tuple[list[FloatArray], list[str]]:
    """検証済みの入力からブロックと列名を組み立てる (分岐なし)。"""
    n_steps, n_inputs = inputs.shape
    blocks: list[FloatArray] = []
    names: list[str] = []
    if layout.bias:
        blocks.append(np.ones((n_steps, 1), dtype=np.float64))
        names.append(BIAS_NAME)
    for lag in layout.input_lags:
        block: FloatArray = np.full((n_steps, n_inputs), np.nan, dtype=np.float64)
        block[lag:] = inputs[: n_steps - lag]
        blocks.append(block)
        names.extend(f"u{dim}_lag{lag}" for dim in range(n_inputs))
    if state_array is not None:
        blocks.append(state_array)
        names.extend(f"x{unit}" for unit in range(state_array.shape[1]))
    if not blocks:
        raise ValueError("特徴が1つもありません (bias=False かつ入力も状態も無し)")
    return blocks, names


def build_design_matrix(
    spec: FeatureSpec,
    u: FloatArray,
    states: FloatArray | None = None,
) -> DesignMatrix:
    """特徴仕様から設計行列を作る (3ベースライン共通の唯一の入口)。

    「検証 (``_validate_inputs``) → レイアウト正規化 (``_layout_of``,
    検証内で実施) → ブロック組み立て (``_assemble_blocks``)」の直線的な流れで、
    入口はこの関数1本のまま (D-05・受け入れ条件1)。

    Args:
        spec: ``PassthroughSpec`` / ``DelayLineSpec`` / ``ReservoirSpec``。
        u: 入力系列 ``(T, D_in)``。1次元は受理しない。
        states: リザバー状態 ``(T, N)``。``ReservoirSpec`` では必須。

    Returns:
        ``DesignMatrix``。列数は順に ``1+D``, ``1+D(k+1)``, ``1+D+N``。

    Raises:
        ValueError: 形状不整合、または ``ReservoirSpec`` に ``states=None``。
    """
    inputs, state_array, layout, first_valid = _validate_inputs(spec, u, states)
    blocks, names = _assemble_blocks(layout, inputs, state_array)
    phi: FloatArray = np.concatenate(blocks, axis=1)
    return DesignMatrix(phi=phi, first_valid=first_valid, feature_names=tuple(names))


__all__ = [
    "BIAS_NAME",
    "DelayLineSpec",
    "DesignMatrix",
    "FeatureSpec",
    "PassthroughSpec",
    "ReservoirSpec",
    "bias_column_index",
    "build_design_matrix",
    "first_valid_for",
]

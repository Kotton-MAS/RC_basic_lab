"""容量測定 (MC / IPC) の共有カーネル (非公開モジュール).

MC (線形メモリ容量) と IPC (情報処理容量) は「同じ状態行列 ``X`` に対して、
多数の目標 ``z_k`` を線形読み出しでどれだけ説明できるか」を測る同一の量であり、
違いは**目標の作り方だけ**である。したがって回帰と容量の計算はここに1本だけ
置き、``memory_capacity`` / ``ipc`` は目標の生成だけを持つ。

このモジュールが守る性質は3つある。

1. **Gram は条件ごとに1回** (D-26)。``CapacityProblem`` が ``Phi = [1, X]`` の
   Gram を構築時に1回だけ作って保持する。目標が何本増えても Gram は作り直さない。
2. **容量は Gram 量だけで閉じる** (D-26)。``capacity_of_targets`` は
   ``rhs = Phi.T @ Z`` を1回だけ作り、残差を
   ``z.T z - 2 w.T rhs + w.T G w`` という **``(F, K)`` と ``(F, F)`` だけの式**で
   評価する。予測 ``Phi @ W`` を実体化しないので、目標数 K を増やしても
   ``O(T * F * K)`` の走査は rhs の1回だけで済む。
3. **基底は宣言された入力分布に対して正規直交** (D-28)。直交していない基底で
   目標を作ると容量が目標間で二重計上され、保存則 (total <= N) が
   「N をわずかに超える」という穏やかな形で破れる。

``rc_basics_lab.readout.ridge.fit_ridge_from_gram`` を使うのは D-23 で明示的に
許可された依存である。バイアス列を正則化しない (D-03) 閉形式解の実装を2箇所に
持つと、片方だけで D-03 が崩れても誰も気づけない。

先頭が ``_`` の非公開モジュールなので、``tests/test_public_api_reexport.py`` の
再エクスポート要求の対象外であり、``diagnostics/__init__.py`` からは公開しない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rc_basics_lab.readout.ridge import fit_ridge_from_gram
from rc_basics_lab.types import FloatArray

UNIFORM = "uniform"
"""入力分布の宣言値: 一様分布 (``U[-a, a]``)。"""

NORMAL = "normal"
"""入力分布の宣言値: 正規分布。"""

LEGENDRE = "legendre"
"""多項式基底の宣言値: Legendre 多項式 (一様分布に対して直交)。"""

HERMITE = "hermite"
"""多項式基底の宣言値: 確率論的 Hermite 多項式 (正規分布に対して直交)。"""

SUPPORTED_BASIS_PAIRS: tuple[tuple[str, str], ...] = (
    (UNIFORM, LEGENDRE),
    (NORMAL, HERMITE),
)
"""``(input_distribution, basis)`` の対応表 (D-28)。

ここに無い組は ``ValueError`` にする。「一様でない入力に Legendre を当てる」を
黙って許すと、目標同士が直交しなくなって容量が二重計上される。
"""

_UNIFORM_HALF_WIDTH_IN_SIGMA = math.sqrt(3.0)
"""一様分布の半値幅と標準偏差の比 (``a = sqrt(3) * sigma_u``、D-17)。"""


@dataclass(frozen=True, slots=True, eq=False)
class CapacityProblem:
    """容量測定の左辺 (設計行列と Gram) を1回だけ作って保持する。

    ``eq=False`` にしてあるのは、numpy 配列を持つ dataclass の ``__eq__`` が
    要素ごとの比較結果 (配列) を真偽値に落とそうとして ``ValueError`` になる
    ため。同一性の比較には使わない。

    Attributes:
        phi: 設計行列 ``[1, X[t0:]]`` ``(T_eff, 1 + N)``。
        gram: ``phi.T @ phi`` ``(F, F)``。
        bias_column: 正則化しない列の index (D-03)。``phi`` の作り方から常に 0。
        t0: ``X`` の何行目から使ったか (D-24 の単一基準点)。
    """

    phi: FloatArray
    gram: FloatArray
    bias_column: int
    t0: int

    @property
    def n_samples(self) -> int:
        """回帰に使う行数 ``T_eff``。全目標がこの行集合を共有する (D-24)。"""
        return int(self.phi.shape[0])

    @property
    def n_features(self) -> int:
        """特徴数 ``F = 1 + N`` (バイアス列を含む)。"""
        return int(self.phi.shape[1])

    @property
    def n_units(self) -> int:
        """状態の次元 ``N``。容量の理論上限 (Dambre 2012) がこの値。"""
        return self.n_features - 1

    @classmethod
    def from_states(cls, X: FloatArray, *, t0: int) -> CapacityProblem:
        """状態系列 ``X`` の ``t0`` 行目以降から ``Phi = [1, X]`` と Gram を作る。

        Args:
            X: 状態系列 ``(T, N)``。
            t0: 使い始める行 (D-24 の単一基準点)。全目標で同じ値を使うこと。

        Raises:
            ValueError: ``X`` が2次元でない / ``t0`` が範囲外 /
                行数が特徴数以下 / 非有限値を含む場合。
        """
        states = np.asarray(X, dtype=np.float64)
        if states.ndim != 2:
            raise ValueError(f"X は (T, N) の2次元配列が必要です: {states.shape}")
        n_steps, n_units = states.shape
        if not 0 <= t0 < n_steps:
            raise ValueError(f"t0 が範囲外です: t0={t0}, T={n_steps}")
        window: FloatArray = states[t0:]
        if not np.all(np.isfinite(window)):
            raise ValueError("X に有限でない値があります")
        n_samples = window.shape[0]
        n_features = n_units + 1
        if n_samples <= n_features:
            raise ValueError(
                "回帰に使える行数が特徴数以下です "
                f"(n_samples={n_samples}, n_features={n_features})。"
                " T を伸ばすか t0 (washout / max_delay) を下げてください"
            )
        ones: FloatArray = np.ones((n_samples, 1), dtype=np.float64)
        phi: FloatArray = np.concatenate((ones, window), axis=1)
        gram: FloatArray = phi.T @ phi
        return cls(phi=phi, gram=gram, bias_column=0, t0=t0)


def capacity_of_targets(
    problem: CapacityProblem,
    Z_chunk: FloatArray,
    alpha: float,
) -> FloatArray:
    """目標チャンク ``Z_chunk`` の各列の容量 ``C_k`` を返す ``(K,)``。

    容量は ``C_k = 1 - ||z_k - Phi w_k||^2 / ||z_k||^2`` (Dambre 2012) だが、
    残差を素直に ``Z - Phi @ W`` で作ると目標数 K に比例して ``(T, K)`` の
    行列積がもう1回増える。ここでは残差を Gram 量だけで展開した

        ``||z - Phi w||^2 = z.T z - 2 w.T (Phi.T z) + w.T (Phi.T Phi) w``

    で評価するため、``Phi`` に触るのは ``rhs = Phi.T @ Z_chunk`` の**1回だけ**、
    ``Z_chunk`` に触るのはその rhs と ``z.T z`` の2回だけで、予測を一度も
    実体化しない (D-26)。``fit_ridge_from_gram`` の呼び出しも1回で、
    ``(F, K)`` の多出力 rhs をそのまま1回の solve に畳む。

    Args:
        problem: ``CapacityProblem`` (Gram は構築済み)。
        Z_chunk: 目標 ``(T_eff, K)``。行は ``problem.phi`` と同じ行集合 (D-24)。
        alpha: 正則化係数 (D-25 により容量測定では微小固定値を使う)。

    Returns:
        容量 ``(K,)``。下限 0 でクリップする —— バイアス列があるので理論上
        ``C_k >= 0`` (平均予測で残差 = 全分散) であり、負値は Gram 展開の
        桁落ちによる数値誤差にすぎない。上限側はクリップしない
        (``C_k > 1`` は基底や行合わせのバグを意味するので隠さない)。

    Raises:
        ValueError: 形状不整合 / 非有限値 / ``z.T z`` が 0 の目標がある場合。
    """
    targets = np.asarray(Z_chunk, dtype=np.float64)
    if targets.ndim != 2:
        raise ValueError(f"Z_chunk は (T, K) の2次元配列が必要です: {targets.shape}")
    if targets.shape[0] != problem.n_samples:
        raise ValueError(
            "Z_chunk の行数が設計行列と一致しません "
            f"(D-24): {targets.shape[0]} != {problem.n_samples}"
        )
    if targets.shape[1] == 0:
        raise ValueError("Z_chunk に目標が1本もありません")
    if not np.all(np.isfinite(targets)):
        raise ValueError("Z_chunk に有限でない値があります")

    # Phi と Z に触るのはこの1行だけ (D-26)。
    rhs: FloatArray = problem.phi.T @ targets
    weights: FloatArray = fit_ridge_from_gram(
        problem.gram, rhs, alpha, bias_column=problem.bias_column
    )
    total: FloatArray = np.einsum("tk,tk->k", targets, targets)
    if np.any(total <= 0.0):
        raise ValueError("二乗和が 0 の目標があるため容量を定義できません")
    cross: FloatArray = np.einsum("fk,fk->k", weights, rhs)
    quadratic: FloatArray = np.einsum("fk,fk->k", weights, problem.gram @ weights)
    residual: FloatArray = total - 2.0 * cross + quadratic
    capacity: FloatArray = 1.0 - residual / total
    return np.clip(capacity, 0.0, None)


def _legendre_normalized(x: FloatArray, degree: int) -> FloatArray:
    """``sqrt(2n+1) P_n(x)`` を漸化式で評価する (``x`` は ``[-1, 1]`` 想定)。

    ``E[P_n(X)^2] = 1/(2n+1)`` (``X ~ U[-1, 1]``) なので ``sqrt(2n+1)`` 倍が
    単位分散になる。
    """
    previous: FloatArray = np.ones_like(x)
    if degree == 0:
        return previous
    current: FloatArray = x
    for order in range(1, degree):
        following: FloatArray = (
            (2 * order + 1) * x * current - order * previous
        ) / float(order + 1)
        previous, current = current, following
    return math.sqrt(2.0 * degree + 1.0) * current


def _hermite_normalized(x: FloatArray, degree: int) -> FloatArray:
    """``He_n(x) / sqrt(n!)`` を漸化式で評価する (``x`` は標準正規想定)。

    確率論的 Hermite は ``E[He_n(X)^2] = n!`` (``X ~ N(0, 1)``) なので
    ``sqrt(n!)`` で割ると単位分散になる。
    """
    previous: FloatArray = np.ones_like(x)
    if degree == 0:
        return previous
    current: FloatArray = x
    for order in range(1, degree):
        following: FloatArray = x * current - order * previous
        previous, current = current, following
    return current / math.sqrt(float(math.factorial(degree)))


def orthonormal_basis(
    u_lagged: FloatArray,
    degree: int,
    distribution: str = UNIFORM,
    *,
    basis: str = LEGENDRE,
) -> FloatArray:
    """次数 ``degree`` の正規直交多項式を ``u_lagged`` の各要素で評価する (D-28)。

    返す列は、宣言された入力分布のもとで **平均 0・分散 1・異なる次数と直交**
    になる。IPC の目標はこの列の積 ``Π_i psi_{n_i}(u[t - k_i])`` で作るので、
    ここが直交していないと容量が目標間で二重計上され、保存則が破れる。

    正規化は入力の**実測**の平均と標準偏差で行う (D-28: 「正規化を振幅に
    追従させることで sigma_u を自由に振れる」)。一様分布では半値幅
    ``a = sqrt(3) * sigma_u`` (D-17) で割ってから Legendre に入れる。

    次数 1 は分布によらず ``(u - mean) / sigma`` に一致する
    (``sqrt(3) P_1(v/sqrt(3)) = v``、``He_1(v)/sqrt(1!) = v``)。MC が
    ``distribution`` を設定項目に持たないのはこのため。

    Args:
        u_lagged: 入力系列、または遅延を並べた配列。形状は問わない
            (平均と標準偏差は配列全体から推定し、同じ形状の配列を返す)。
        degree: 多項式の次数 (0 以上)。0 は定数 1。
        distribution: 宣言された入力分布 (``"uniform"`` / ``"normal"``)。
        basis: 多項式基底 (``"legendre"`` / ``"hermite"``)。

    Returns:
        ``u_lagged`` と同じ形状の配列。

    Raises:
        ValueError: ``degree`` が負 / ``(distribution, basis)`` が
            ``SUPPORTED_BASIS_PAIRS`` に無い / 入力が空・非有限・分散 0 の場合。
    """
    if degree < 0:
        raise ValueError(f"degree は 0 以上である必要があります: {degree}")
    if (distribution, basis) not in SUPPORTED_BASIS_PAIRS:
        raise ValueError(
            "(input_distribution, basis) の組が未対応です (D-28): "
            f"({distribution!r}, {basis!r})。"
            f" 対応する組: {SUPPORTED_BASIS_PAIRS}"
        )
    values = np.asarray(u_lagged, dtype=np.float64)
    if values.size == 0:
        raise ValueError("u_lagged が空です")
    if not np.all(np.isfinite(values)):
        raise ValueError("u_lagged に有限でない値があります")
    sigma = float(np.std(values))
    if sigma <= 0.0:
        raise ValueError("u_lagged が定数のため正規直交基底を定義できません")
    standardized: FloatArray = (values - float(np.mean(values))) / sigma
    if basis == LEGENDRE:
        return _legendre_normalized(standardized / _UNIFORM_HALF_WIDTH_IN_SIGMA, degree)
    return _hermite_normalized(standardized, degree)


def surrogate_threshold(
    problem: CapacityProblem,
    base_targets: FloatArray,
    alpha: float,
    *,
    n_surrogates: int,
    quantile: float,
    chunk_size: int,
    rng: np.random.Generator,
) -> tuple[float, FloatArray]:
    """時間シャッフルサロゲートから容量のしきい値を推定する (D-27)。

    ``base_targets`` の各列を時間方向にシャッフルした系列を ``n_surrogates``
    本ずつ作り、**通常の目標とまったく同じ経路** (``capacity_of_targets``) に
    流して容量を測る。閾値はその分位点。別経路で閾値を計算すると、閾値と容量が
    別実装からずれても誰も気づけない。

    Args:
        problem: 容量問題 (通常の目標と同じもの)。
        base_targets: 実目標から決定的に選んだ代表 ``(T_eff, M)``。
        alpha: 通常の目標と同じ正則化係数。
        n_surrogates: 代表1本あたりのサロゲート本数。
        quantile: 分位点 (0〜1)。
        chunk_size: 1回の solve に畳む列数 (結果には影響しない)。
        rng: ``ctx.seed`` から作った乱数生成器 (D-27: 乱数源はこれだけ)。

    Returns:
        ``(閾値, サロゲート容量 (M * n_surrogates,))``。

    Raises:
        ValueError: ``n_surrogates`` が 1 未満 / ``quantile`` が範囲外 /
            ``chunk_size`` が 1 未満の場合。
    """
    if n_surrogates < 1:
        raise ValueError(f"n_surrogates は 1 以上が必要です: {n_surrogates}")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile は 0〜1 が必要です: {quantile}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size は 1 以上が必要です: {chunk_size}")
    base = np.asarray(base_targets, dtype=np.float64)
    if base.ndim != 2:
        raise ValueError(f"base_targets は (T, M) が必要です: {base.shape}")
    n_samples, n_base = base.shape
    surrogates: FloatArray = np.empty(
        (n_samples, n_base * n_surrogates), dtype=np.float64
    )
    for index in range(n_base * n_surrogates):
        source = base[:, index // n_surrogates]
        surrogates[:, index] = source[rng.permutation(n_samples)]
    capacities = capacity_in_chunks(problem, surrogates, alpha, chunk_size=chunk_size)
    return float(np.quantile(capacities, quantile)), capacities


def capacity_in_chunks(
    problem: CapacityProblem,
    targets: FloatArray,
    alpha: float,
    *,
    chunk_size: int,
) -> FloatArray:
    """``targets`` を ``chunk_size`` 列ずつ ``capacity_of_targets`` に流す。

    ``fit_ridge_from_gram`` の呼び出し回数は ``ceil(K / chunk_size)`` になり、
    目標数 K には比例しない (D-26)。列ごとの容量は互いに独立に決まるので、
    ``chunk_size`` は**結果を変えない性能パラメータ**である
    (``tests/test_diagnostics_memory_capacity.py::test_chunk_size_does_not_change_results``)。

    Raises:
        ValueError: ``chunk_size`` が 1 未満の場合。
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size は 1 以上が必要です: {chunk_size}")
    matrix = np.asarray(targets, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"targets は (T, K) が必要です: {matrix.shape}")
    pieces: list[FloatArray] = [
        capacity_of_targets(problem, matrix[:, start : start + chunk_size], alpha)
        for start in range(0, matrix.shape[1], chunk_size)
    ]
    return np.concatenate(pieces)


__all__ = [
    "HERMITE",
    "LEGENDRE",
    "NORMAL",
    "SUPPORTED_BASIS_PAIRS",
    "UNIFORM",
    "CapacityProblem",
    "capacity_in_chunks",
    "capacity_of_targets",
    "orthonormal_basis",
    "surrogate_threshold",
]

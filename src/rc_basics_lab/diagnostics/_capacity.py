"""容量測定 (MC / IPC) の共有カーネル (非公開モジュール).

MC (線形メモリ容量) と IPC (情報処理容量) は「同じ状態行列 ``X`` に対して、
多数の目標 ``z_k`` を線形読み出しでどれだけ説明できるか」を測る同一の量であり、
違いは**目標の作り方だけ**である。したがって回帰と容量の計算はここに1本だけ
置き、``memory_capacity`` / ``ipc`` は目標の生成だけを持つ。

このモジュールが守る性質は4つある。

1. **Gram は条件ごとに1回、``Phi`` は実体化しない** (D-26)。``CapacityProblem``
   は ``Phi = [1, X]`` の Gram をブロック分解 (
   ``G = [[T_eff, sum(X,0)], [sum(X,0).T, X.T @ X]]``) で構築時に1回だけ
   作って保持し、状態 ``X`` そのもの (バイアス列を足す前のビュー) だけを
   持つ。``T=1e6`` 級の本番設定では ``Phi`` の実体化 (``X`` と同じ大きさの
   コピーがもう1枚) だけでメモリ予算 4GB の 9 割近くを単独消費するため
   (F-03-1-013)、``concatenate`` で ``Phi`` を作る経路を持たない。
2. **容量は Gram 量だけで閉じる** (D-26)。``capacity_of_targets`` は
   ``rhs = [sum(Z,0); X.T @ Z]`` を1回だけ作り (``X`` に触れるのはこの
   ``X.T @ Z`` の**1回だけ**)、残差を
   ``z.T z - 2 w.T rhs + w.T G w`` という **``(F, K)`` と ``(F, F)`` だけの式**で
   評価する。予測 ``Phi @ W`` を実体化しないので、目標数 K を増やしても
   ``O(T * F * K)`` の走査は rhs の1回だけで済む。
3. **サロゲートも通常の目標と同じ経路・同じチャンク化を通る** (D-27)。
   時間シャッフル列を一括確保せず ``chunk_size`` 列ずつ生成しては
   ``capacity_of_targets`` に渡して捨てる (F-03-1-012)。生成順序は
   ``chunk_size`` に依存しないので、結果は chunk_size を変えても1ビットも
   変わらない (``test_chunk_size_does_not_change_results``)。
4. **基底は宣言された入力分布に対して正規直交** (D-28)。直交していない基底で
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
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

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
    """容量測定の左辺 (状態のビューと Gram) を1回だけ作って保持する。

    ``eq=False`` にしてあるのは、numpy 配列を持つ dataclass の ``__eq__`` が
    要素ごとの比較結果 (配列) を真偽値に落とそうとして ``ValueError`` になる
    ため。同一性の比較には使わない。

    設計行列 ``Phi = [1, X]`` はここでは**実体化しない** (F-03-1-013)。
    バイアス列は定数列なので、Gram ``Phi.T @ Phi`` は
    ``[[T_eff, sum(X,0)], [sum(X,0).T, X.T @ X]]`` というブロックに閉じて
    書ける。``x`` に ``X[t0:]`` の**ビュー** (先頭からの基本スライスなので
    コピーが発生しない) を持つだけで済み、``T=1e6`` 級の本番設定で ``Phi``
    のコピー1枚ぶん (状態と同じ大きさ) のメモリを節約できる。

    Attributes:
        x: 状態のビュー ``X[t0:]`` ``(T_eff, N)``。バイアス列は含まない。
        gram: ブロック分解した ``Phi.T @ Phi`` ``(F, F)`` (``F = 1 + N``)。
        bias_column: 正則化しない列の index (D-03)。``gram`` の作り方から常に 0。
        t0: ``X`` の何行目から使ったか (D-24 の単一基準点)。
    """

    x: FloatArray
    gram: FloatArray
    bias_column: int
    t0: int

    @property
    def n_samples(self) -> int:
        """回帰に使う行数 ``T_eff``。全目標がこの行集合を共有する (D-24)。"""
        return int(self.x.shape[0])

    @property
    def n_features(self) -> int:
        """特徴数 ``F = 1 + N`` (バイアス列を含む)。"""
        return int(self.x.shape[1]) + 1

    @property
    def n_units(self) -> int:
        """状態の次元 ``N``。容量の理論上限 (Dambre 2012) がこの値。"""
        return int(self.x.shape[1])

    @classmethod
    def from_states(cls, X: FloatArray, *, t0: int) -> CapacityProblem:
        """状態系列 ``X`` の ``t0`` 行目以降から ``Phi = [1, X]`` の Gram を作る。

        ``Phi`` そのものは作らない。バイアス列の寄与 (行和・列和) は ``X`` の
        縮約 (``sum``) だけで閉じるため、``X`` に触れるのは
        ``X.T @ X`` の**1回**で済む (D-26 と同じ「触れるのは1回」の規律を
        構築時にも適用する)。

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
        # Phi.T @ Phi = [[T_eff, sum(X,0)], [sum(X,0).T, X.T @ X]] (バイアス
        # 列は定数1なので、ones.T @ ones = T_eff、ones.T @ X = sum(X, 0))。
        column_sums: FloatArray = np.sum(window, axis=0)
        gram_xx: FloatArray = window.T @ window
        gram: FloatArray = np.empty((n_features, n_features), dtype=np.float64)
        gram[0, 0] = float(n_samples)
        gram[0, 1:] = column_sums
        gram[1:, 0] = column_sums
        gram[1:, 1:] = gram_xx
        return cls(x=window, gram=gram, bias_column=0, t0=t0)

    def lagged(self, series: FloatArray, delay: int) -> FloatArray:
        """入力系列を D-24 の単一基準点に合わせて遅延 ``delay`` だけずらす。

        ``series[t0 - delay : t0 - delay + n_samples]`` を返す。MC と IPC は
        どちらも「遅延 k の目標は状態と同じ行集合 (``t0`` 始まり) に対応する」
        という同じ規律 (D-24) の実体をこの1本の窓計算に依存する。かつては
        MC (2箇所) と IPC (1箇所) がこの式を共有カーネルの外でそれぞれ書いて
        おり、MC 側だけ複製が値レベルで検査されていなかった (F-03-1-001:
        窓を1ステップずらしても MC のテスト22本が全て緑のまま通った)。ここに
        1本だけ置くことで、以後どの容量診断が増えても複製が生まれない。

        Args:
            series: 1次元の系列 (入力の正規直交多項式など)。
            delay: 遅延 (0 以上)。

        Returns:
            ``(n_samples,)`` のビュー。

        Raises:
            ValueError: ``series`` が1次元でない / ``delay`` が負 /
                窓が ``series`` の範囲外になる場合。
        """
        if delay < 0:
            raise ValueError(f"delay は 0 以上が必要です: {delay}")
        values = np.asarray(series, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"series は1次元配列が必要です: {values.shape}")
        start = self.t0 - delay
        stop = start + self.n_samples
        if start < 0 or stop > values.shape[0]:
            raise ValueError(
                "delay が範囲外です (D-24): "
                f"t0={self.t0}, delay={delay}, series長={values.shape[0]},"
                f" n_samples={self.n_samples}"
            )
        return values[start:stop]


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
        Z_chunk: 目標 ``(T_eff, K)``。行は ``problem.x`` と同じ行集合 (D-24)。
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

    # X に触れるのはこの1行 (X.T @ targets) だけ (D-26 / F-03-1-013)。
    # バイアス行 (rhs[0]) は targets の縮約 (sum) だけで閉じるので、X には
    # 触れない。Phi = [1, X] を実体化した上で phi.T @ targets を1回で計算
    # する経路と比べて、X 自体のコピーを一切作らずに済む。
    bias_rhs: FloatArray = np.sum(targets, axis=0, keepdims=True)
    x_rhs: FloatArray = problem.x.T @ targets
    rhs: FloatArray = np.concatenate((bias_rhs, x_rhs), axis=0)
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


def input_series(u: FloatArray | None, *, diagnostic: str) -> FloatArray:
    """``u`` を1次元の入力系列にして返す。無い / 多変数なら ``ValueError``。

    MC と IPC はどちらも「単一変数の駆動信号から遅延目標を作る」という同じ
    入力検証をほぼ同一に複製していた (F-03-1-020: docstring・分岐構造が
    完全一致で、差分は診断名のみ)。共有カーネルが謳う「他は寄せる」方針
    そのものに反していたため、ここへ1本にまとめる。

    Args:
        u: 入力系列 ``(T, 1)``、または ``None``。
        diagnostic: エラーメッセージに出す診断名 (``"memory_capacity"`` /
            ``"ipc"``)。

    Raises:
        ValueError: ``u`` が無い / 1変数でない / 非有限値を含む場合。
    """
    if u is None:
        raise ValueError(
            f"{diagnostic} は入力系列 u が必須です (遅延・多項式目標を作れません)"
        )
    series = np.asarray(u, dtype=np.float64)
    if series.ndim != 2 or series.shape[1] != 1:
        raise ValueError(
            f"{diagnostic} は1変数入力のみ対応です: u.shape={series.shape}"
        )
    if not np.all(np.isfinite(series)):
        raise ValueError("u に有限でない値があります")
    return series[:, 0]


def _iter_surrogate_chunks(
    base: FloatArray,
    n_surrogates: int,
    chunk_size: int,
    rng: np.random.Generator,
) -> Iterator[FloatArray]:
    """時間シャッフル列を ``chunk_size`` 列ずつ生成しては渡し捨てる (F-03-1-012)。

    生成順序は ``index = 0, 1, ..., n_base * n_surrogates - 1`` の単調増加
    (先頭の代表から順に ``n_surrogates`` 本ずつ) で固定し、``chunk_size`` が
    どこで区切っても ``rng`` の消費順序が変わらないようにする。これにより
    ``chunk_size`` を変えても生成される列そのものは1本残らず同一になり
    (``test_chunk_size_does_not_change_results``)、`` (n_samples, n_base *
    n_surrogates)`` を一括確保していた旧実装 (IPC 既定で peak RSS +3.5GB)
    を、一度に ``(n_samples, chunk_size)`` しか保持しない生成器に置き換える。
    """
    n_samples, n_base = base.shape
    total = n_base * n_surrogates
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        chunk: FloatArray = np.empty((n_samples, stop - start), dtype=np.float64)
        for column, index in enumerate(range(start, stop)):
            source = base[:, index // n_surrogates]
            chunk[:, column] = source[rng.permutation(n_samples)]
        yield chunk


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
    別実装からずれても誰も気づけない。シャッフル列は ``chunk_size`` 列ずつ
    その場で生成しては畳んで捨てる (F-03-1-012)。``(n_samples, M *
    n_surrogates)`` を一括確保しない。

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
    capacities = capacity_of_chunks(
        problem, _iter_surrogate_chunks(base, n_surrogates, chunk_size, rng), alpha
    )
    return float(np.quantile(capacities, quantile)), capacities


def chi2_threshold(*, n_units: int, n_samples: int, quantile: float) -> float:
    """カイ二乗近似のしきい値 ``chi2_N(q) / T_eff`` (Dambre 2012 SupMat 3.2)。

    状態と無相関な目標に対する決定係数は、自由度 ``N`` (バイアス列を除く回帰
    変数の本数) のカイ二乗を標本数で割った分布に漸近する。IPC のしきい値法の
    1つ (``THRESHOLD_CHI2``) が使う。仕様書 (T1 実装時に決めたこと 5) は
    「chi2 は T2 で共有カーネルに足す」と宣言しており、以前は ``ipc.py`` に
    private 関数として置かれ食い違っていた (F-03-1-003)。MC は次数1しか
    評価せず周辺分布が1種類 (サロゲートで足りる) なので現状は呼ばないが、
    将来 MC 側にもカイ二乗近似のしきい値を足す場合はここを import すればよい。
    """
    return float(chi2.ppf(quantile, n_units)) / float(n_samples)


def capacity_of_chunks(
    problem: CapacityProblem,
    chunks: Iterable[FloatArray],
    alpha: float,
) -> FloatArray:
    """目標チャンクの列を順に ``capacity_of_targets`` へ流して連結する。

    引数を **iterable of chunk** にしてあるのは D-26 のため: 呼び出し側は
    ``(T_eff, chunk_size)`` を1枚ずつ作って渡し、畳んだら捨てられる。全目標を
    ``(T_eff, K)`` として実体化する必要が無いので、IPC の T=1e6 x 2395 列
    (19 GB) を持たずに済む。``fit_ridge_from_gram`` の呼び出し回数は
    チャンク数 (``ceil(K / chunk_size)``) であり、目標数 K には比例しない。

    チャンクの切り方は結果を変えない —— 列ごとの容量は互いに独立に決まる
    ため、``chunk_size`` は純粋な性能パラメータである
    (``tests/test_diagnostics_memory_capacity.py::test_chunk_size_does_not_change_results``)。

    Raises:
        ValueError: チャンクが1枚も無い場合。
    """
    pieces: list[FloatArray] = [
        capacity_of_targets(problem, chunk, alpha) for chunk in chunks
    ]
    if not pieces:
        raise ValueError("目標チャンクが1枚もありません")
    return np.concatenate(pieces)


__all__ = [
    "HERMITE",
    "LEGENDRE",
    "NORMAL",
    "SUPPORTED_BASIS_PAIRS",
    "UNIFORM",
    "CapacityProblem",
    "capacity_of_chunks",
    "capacity_of_targets",
    "chi2_threshold",
    "input_series",
    "orthonormal_basis",
    "surrogate_threshold",
]

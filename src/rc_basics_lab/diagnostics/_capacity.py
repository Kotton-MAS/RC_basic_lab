"""容量測定 (MC / IPC) の共有カーネル (非公開モジュール).

MC (線形メモリ容量) と IPC (情報処理容量) は「同じ状態行列 ``X`` に対して、
多数の目標 ``z_k`` を線形読み出しでどれだけ説明できるか」を測る同一の量であり、
違いは**目標の作り方だけ**である。したがって回帰と容量の計算はここに1本だけ
置き、``memory_capacity`` / ``ipc`` は目標の生成だけを持つ。

このモジュールが守る性質は6つある。

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
   「N をわずかに超える」という穏やかな形で破れる。分布と基底は**対でのみ
   意味を持つ**ので ``InputMeasure`` 1値にまとめ、``orthonormal_basis`` は
   これを**既定値なしの第3引数**で受ける (片方だけ渡す呼び方を書けなくする)。
5. **行合わせの担い手は ``RowAlignment`` 1つ** (D-24)。基準点 ``t0`` の算出・
   ``t0 >= T`` の拒否・遅延窓の切り出しを診断ごとに複製しない。状態行列にも
   Gram にも触れないので、行合わせだけを検査するテストがダミーの状態行列を
   作らずに書ける。
6. **チャンク幅は性能軸と確保軸に分ける** (D-33)。``solve_width`` は
   ``cfg.chunk_size`` を上限とする性能軸、``block_width`` は
   ``cfg.chunk_size`` を**読まない**確保軸で、どちらも
   ``bounded_chunk_size`` 1本の純関数へ委譲する。

``rc_basics_lab.readout.ridge.fit_ridge_from_gram`` を使うのは D-23 で明示的に
許可された依存である (D-03 の実装を2箇所に持たないため)。

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


@dataclass(frozen=True, slots=True)
class InputMeasure:
    """入力測度 ``(input_distribution, basis)`` を1つにまとめた値 (D-28)。

    この2つは**対でのみ意味を持つ**。「一様でない入力に Legendre を当てる」
    のような未対応の組は容量を目標間で二重計上させ、保存則が「N をわずかに
    超える」という穏やかな形で破れる (図では正常に見える)。かつては
    ``orthonormal_basis(u, degree, distribution=UNIFORM, *, basis=LEGENDRE)``
    のように**片方だけを既定値つきで**受けており、「片方だけ渡す」呼び方が
    型検査を素通りした (F-03-1-006)。対を1つの値にして
    ``orthonormal_basis`` の**既定値なしの第3引数**として受けることで、
    その呼び方をそもそも書けなくする。

    値域の検証を ``__post_init__`` で行うのは D-09 (設定 dataclass は純データ
    で検証は使う側) に反しない —— ``InputMeasure`` は YAML から構築される
    設定 dataclass ではなく、カーネル内で組み立てる値オブジェクトである。
    設定層 (``IpcConfig``) は YAML と ``meta.json`` の面を変えないため
    **2つの文字列フィールドのまま**保ち、``ipc()`` が入口で1度だけ畳む。

    Attributes:
        distribution: 宣言された入力分布 (``UNIFORM`` / ``NORMAL``)。
        basis: 多項式基底 (``LEGENDRE`` / ``HERMITE``)。

    Raises:
        ValueError: 組が ``SUPPORTED_BASIS_PAIRS`` に無い場合 (**構築時点**)。
    """

    distribution: str
    basis: str

    def __post_init__(self) -> None:
        if (self.distribution, self.basis) not in SUPPORTED_BASIS_PAIRS:
            raise ValueError(
                "(input_distribution, basis) の組が未対応です (D-28): "
                f"({self.distribution!r}, {self.basis!r})。"
                f" 対応する組: {SUPPORTED_BASIS_PAIRS}"
            )


UNIFORM_LEGENDRE = InputMeasure(UNIFORM, LEGENDRE)
"""一様入力 x Legendre 基底 (既定の測度)。"""

NORMAL_HERMITE = InputMeasure(NORMAL, HERMITE)
"""正規入力 x 確率論的 Hermite 基底。"""

SUPPORTED_MEASURES: tuple[InputMeasure, ...] = (UNIFORM_LEGENDRE, NORMAL_HERMITE)
"""対応する測度の一覧 (``SUPPORTED_BASIS_PAIRS`` と1対1)。"""


@dataclass(frozen=True, slots=True)
class RowAlignment:
    """MC / IPC が共有する行合わせ (D-24)。``t0`` と ``n_samples`` **だけ**を持つ。

    行合わせの担い手をこの1つの値にまとめ、(i) 基準点 ``t0 = max(washout,
    その診断の最大遅延)`` の算出、(ii) ``t0 >= T`` の拒否、(iii) 遅延窓の
    切り出しの3つを診断ごとに複製せずここへ置く。F-03-1-001 で潰したのは
    (iii) だけで、(i)(ii) は MC (``memory_capacity``) と IPC (``ipc``) の
    2箇所に複製が残っていた。

    **状態行列にも Gram にも触れない**のがこの型の存在理由である。行合わせ
    だけを検査したいテストは、以前はダミーの状態行列 (特異な Gram ごと)
    を構築する必要があった (``tests`` の ``_dummy_problem``)。
    ``RowAlignment`` を直接構築すればその必要がない
    (``test_row_alignment_needs_no_state_matrix``)。

    チャンク幅の2軸 (D-33) もここに置く。どちらも ``n_samples`` だけの
    関数であり、``bounded_chunk_size`` 1本の純関数へ委譲する
    (128 MiB の予算を2箇所に持たないため)。

    Attributes:
        t0: ``X`` の何行目から使うか (D-24 の単一基準点)。
        n_samples: 回帰に使う行数 ``T_eff``。全目標がこの行集合を共有する。
    """

    t0: int
    n_samples: int

    @classmethod
    def from_series(cls, *, n_steps: int, washout: int, max_delay: int) -> RowAlignment:
        """基準点を算出して行合わせを作る。MC と IPC はこの1本だけを呼ぶ。

        Args:
            n_steps: 状態系列の長さ ``T``。
            washout: ``ctx.washout``。
            max_delay: その診断が使う最大遅延 (MC は ``cfg.max_delay``、
                IPC は ``max(cfg.max_delay_by_degree)``)。

        Raises:
            ValueError: ``t0 >= n_steps`` で回帰に使える行が無い場合
                (黙って切り詰めない)。
        """
        t0 = max(washout, max_delay)
        if t0 >= n_steps:
            raise ValueError(
                "系列が短すぎます (D-24): "
                f"t0=max(washout={washout}, max_delay={max_delay})={t0}"
                f" >= T={n_steps}"
            )
        return cls(t0=t0, n_samples=n_steps - t0)

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

    def solve_width(self, configured: int) -> int:
        """**性能軸** (D-33): 1回の solve に畳む目標の列数。

        運用者が指定した ``cfg.chunk_size`` を上限とし、1チャンクが
        ``_MAX_CHUNK_BYTES`` を超えるときだけ下げる。結果は1ビットも
        変わらない。呼び出し側は ``cfg.chunk_size`` を直接使わずこの
        メソッドを経由し、実効値を ``params['chunk_size_effective']`` に
        記録する (F-03-2-001 / F-03-2-009)。
        """
        return bounded_chunk_size(configured, self.n_samples)

    def block_width(self, n_columns: int) -> int:
        """**確保軸** (D-33): 一度に実体化してよい列数。

        ``cfg.chunk_size`` を**読まない**のがこのメソッドの存在理由である。
        代表目標ブロック (``picked``) の一度の確保量には性能上の意味が無く、
        運用者の性能ノブが確保上限を動かしてはならない。上限は
        ``n_columns`` 自身 (それ以上は必要ない) と 128 MiB 予算だけで決まる。

        Args:
            n_columns: 実体化したい列数 (これを超えて確保する意味は無い)。
        """
        return bounded_chunk_size(n_columns, self.n_samples)


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

    契約 (F-03-2-003): ``x`` は呼び出し側が渡した ``X`` の**ビュー**であり、
    ``gram`` は構築時点の ``X`` から作った**スナップショット**である。
    ``from_states`` の後で元の ``X`` を書き換えると ``x`` はその変更を映すが
    ``gram`` は映さず、両者が独立に (かつ例外もなく) desync する。
    ``from_states`` に渡した後の ``X`` は不変として扱うこと。``x`` 自身は
    ``from_states`` が ``writeable=False`` にして保持するため、``problem.x[...]
    = ...`` は ``ValueError`` になる (F-03-3-006)。ただし元の ``X`` の
    writeable フラグは変えないため、``X`` 自身への書き込みは塞げない
    (呼び出し側が渡す前に読み取り専用にすることが
    ``docs/plans/rc-basics-03.md`` T3 の受け入れ基準に明記されている、
    F-03-4-001)。

    Attributes:
        x: 状態のビュー ``X[t0:]`` ``(T_eff, N)``。バイアス列は含まない。
            ``X`` のビューなので、``from_states`` 後に元の ``X`` を書き換えて
            はならない (``gram`` と desync する。上記の契約を参照)。
        gram: ブロック分解した ``Phi.T @ Phi`` ``(F, F)`` (``F = 1 + N``)。
            構築時点の ``X`` から作ったスナップショット (コピー)。
        bias_column: 正則化しない列の index (D-03)。``gram`` の作り方から常に 0。
        rows: 行合わせ (D-24 の単一基準点と ``T_eff``)。``t0`` を直接持たない
            のは、行合わせの担い手を ``RowAlignment`` 1つにするため。
    """

    x: FloatArray
    gram: FloatArray
    bias_column: int
    rows: RowAlignment

    @property
    def n_samples(self) -> int:
        """回帰に使う行数 ``T_eff``。全目標がこの行集合を共有する (D-24)。"""
        return self.rows.n_samples

    @property
    def n_features(self) -> int:
        """特徴数 ``F = 1 + N`` (バイアス列を含む)。"""
        return int(self.x.shape[1]) + 1

    @property
    def n_units(self) -> int:
        """状態の次元 ``N``。容量の理論上限 (Dambre 2012) がこの値。"""
        return int(self.x.shape[1])

    @classmethod
    def from_states(cls, X: FloatArray, *, rows: RowAlignment) -> CapacityProblem:
        """状態系列 ``X`` の ``t0`` 行目以降から ``Phi = [1, X]`` の Gram を作る。

        ``Phi`` そのものは作らない。バイアス列の寄与 (行和・列和) は ``X`` の
        縮約 (``sum``) だけで閉じるため、``X`` に触れるのは
        ``X.T @ X`` の**1回**で済む (D-26 と同じ「触れるのは1回」の規律を
        構築時にも適用する)。

        Args:
            X: 状態系列 ``(T, N)``。
            rows: 行合わせ (D-24 の単一基準点)。全目標で同じ値を使うこと。

        Raises:
            ValueError: ``X`` が2次元でない / ``rows.t0`` が範囲外 /
                ``rows.n_samples`` が切り出した行数と食い違う /
                行数が特徴数以下 / 非有限値を含む場合。
        """
        states = np.asarray(X, dtype=np.float64)
        if states.ndim != 2:
            raise ValueError(f"X は (T, N) の2次元配列が必要です: {states.shape}")
        n_steps, n_units = states.shape
        if not 0 <= rows.t0 < n_steps:
            raise ValueError(f"t0 が範囲外です: t0={rows.t0}, T={n_steps}")
        window: FloatArray = states[rows.t0 :]
        # 行合わせは RowAlignment が正本なので、状態から実際に切り出した行数と
        # 食い違ったまま進むと「どちらが本物か」が実行時に決まってしまう
        # (目標は rows.n_samples 行、設計行列は window 行で組まれる)。構築時に
        # 一致を要求して、行合わせの単一の担い手 (D-24) を構造で保つ。
        if window.shape[0] != rows.n_samples:
            raise ValueError(
                "RowAlignment の行数が状態から切り出した行数と一致しません "
                f"(D-24): rows.n_samples={rows.n_samples},"
                f" X[t0:] の行数={window.shape[0]} (t0={rows.t0}, T={n_steps})"
            )
        if not np.all(np.isfinite(window)):
            raise ValueError("X に有限でない値があります")
        # F-03-3-006: window は X (または dtype 変換されていなければ X 自身) の
        # ビューであり、書き込み可能なままだと診断内部や呼び出し側が
        # problem.x[...] = ... と書いても素通りし、既に作成済みの gram と
        # 無言で desync する (上記クラス docstring の契約)。ここで held する
        # ビュー自身を読み取り専用にする (元の X の writeable フラグは変えない
        # ので、呼び出し側が X を書き換える経路は 3b の受け入れ条件で塞ぐ)。
        window.flags.writeable = False
        n_samples = rows.n_samples
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
        return cls(x=window, gram=gram, bias_column=0, rows=rows)


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

    で評価する。``Phi`` は実体化しない (F-03-1-013)。``rhs = Phi.T @ Z_chunk``
    はバイアス行 ``sum(Z_chunk, 0)`` と ``X`` 行 ``X.T @ Z_chunk`` の2段に
    ブロック分解して求めるため、``X`` に触るのは ``x_rhs = problem.x.T @
    Z_chunk`` の**1回だけ**、``Z_chunk`` に触るのは ``bias_rhs`` / ``x_rhs`` /
    ``z.T z`` の計算だけで、予測を一度も実体化しない (D-26)。
    ``fit_ridge_from_gram`` の呼び出しも1回で、``(F, K)`` の多出力 rhs を
    そのまま1回の solve に畳む。

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
    measure: InputMeasure,
) -> FloatArray:
    """次数 ``degree`` の正規直交多項式を ``u_lagged`` の各要素で評価する (D-28)。

    返す列は、宣言された入力分布のもとで **平均 0・分散 1・異なる次数と直交**
    になる (D-28)。

    正規化は入力の**実測**の平均と標準偏差で行う (D-28 / D-17)。

    次数 1 は分布によらず ``(u - mean) / sigma`` に一致する
    (``sqrt(3) P_1(v/sqrt(3)) = v``、``He_1(v)/sqrt(1!) = v``)。MC が
    入力分布を設定項目に持たず ``UNIFORM_LEGENDRE`` 固定で足りるのはこのため。

    Args:
        u_lagged: 入力系列、または遅延を並べた配列。形状は問わない
            (平均と標準偏差は配列全体から推定し、同じ形状の配列を返す)。
        degree: 多項式の次数 (0 以上)。0 は定数 1。
        measure: 入力測度 ``InputMeasure(distribution, basis)`` (D-28)。
            **既定値を持たない** —— 分布と基底は対でのみ意味を持つので、
            「片方だけ渡す」呼び方を型検査の時点で書けなくする
            (``test_orthonormal_basis_requires_an_explicit_measure``)。
            未対応の組は ``InputMeasure`` の**構築時点**で落ちるため、
            ここには組の検査を置かない (検査を2箇所に持たない)。

    Returns:
        ``u_lagged`` と同じ形状の配列。

    Raises:
        ValueError: ``degree`` が負 / 入力が空・非有限・分散 0 の場合。
    """
    if degree < 0:
        raise ValueError(f"degree は 0 以上である必要があります: {degree}")
    values = np.asarray(u_lagged, dtype=np.float64)
    if values.size == 0:
        raise ValueError("u_lagged が空です")
    if not np.all(np.isfinite(values)):
        raise ValueError("u_lagged に有限でない値があります")
    sigma = float(np.std(values))
    if sigma <= 0.0:
        raise ValueError("u_lagged が定数のため正規直交基底を定義できません")
    standardized: FloatArray = (values - float(np.mean(values))) / sigma
    if measure.basis == LEGENDRE:
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


_MAX_CHUNK_BYTES = 128 * 1024 * 1024
"""1チャンク分の目標行列に許すおおよその上限バイト数 (128 MiB)。

チャンク生成 (``_iter_surrogate_chunks`` / MC・IPC の ``_iter_*_chunks``) は
「直前のチャンクへの参照がまだ残っているうちに次のチャンクを組み立てる」
という generator + 呼び出し側ループの構造上、**一時的に2チャンク分が同時に
生きる**瞬間がある (呼び出し側のループ変数が古いチャンクを指したまま、
generator 内部で新しいチャンクを ``np.empty`` してから yield するため)。
``chunk_size`` の既定値 256 は T_eff が小さい規模を前提に選ばれており、
T=1e6 級の本番設定では1チャンクが ``1e6 * 256 * 8 byte = 2.05 GB`` に達し、
上記の一時的な2重生存と合わさって peak RSS が単独で 4GB 予算を超える
(F-03-1-012 / F-03-1-013 の BLOCKER 修正後に実測: IPC 既定で 5.0〜6.5GB)。
128 MiB は「2チャンク同時生存 (256 MiB) + 状態行列 X (MC 本番 N=200 で
1.6GB) + 諸々の一時配列」を足しても 4GB に収まるよう安全側に選んだ値。
"""


def bounded_chunk_size(configured: int, n_samples: int) -> int:
    """``configured`` を、1チャンクが ``_MAX_CHUNK_BYTES`` を超えないよう下げる。

    ``chunk_size`` は結果を変えない純粋な性能パラメータ (呼び出し側の
    ``test_chunk_size_does_not_change_results`` が固定) なので、ここで値を
    下げても数値は1ビットも変わらない。``configured`` より**大きくはしない**
    (小さい chunk_size を明示的に指定した呼び出し側の意図は尊重する)。ただし
    逆方向 —— 性能チューニングのために大きい ``configured`` を明示指定した
    意図 —— は保護されない: ``n_samples`` が大きい本番規模では
    ``configured`` の値によらず必ず ``_MAX_CHUNK_BYTES`` 相当まで切り詰める
    (D-33)。この関数は **128 MiB 予算を持つ唯一の場所**であり、性能軸
    (``RowAlignment.solve_width``) と確保軸 (``RowAlignment.block_width``)
    の両方がここへ委譲する (予算を2箇所に持つと片方だけ動かせてしまう)。
    呼び出し側は実際に使われた性能軸の値を ``RowAlignment.solve_width``
    経由で取得し、成果物の ``params`` に記録すること
    (``chunk_size_effective``、F-03-2-001 / F-03-2-009)。

    Args:
        configured: 呼び出し側が指定した ``chunk_size`` (性能パラメータ)。
        n_samples: 回帰に使う行数 ``T_eff``。1チャンクのバイト数はこれと
            列数の積で決まる。

    Returns:
        実際に使う chunk_size。``n_samples <= 0`` ならバイト数を計算できない
        ため ``configured`` をそのまま返す (呼び出し側の値域検証で弾かれる
        前提の防御的分岐)。それ以外は
        ``min(configured, max(1, _MAX_CHUNK_BYTES // (n_samples * 8)))``。
        下限は常に 1 (0 列のチャンクは作らない)。
    """
    if n_samples <= 0:
        return configured
    budget_columns = max(1, _MAX_CHUNK_BYTES // (n_samples * 8))
    return min(configured, budget_columns)


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


def _reject_ndarray_base_blocks(base_blocks: Iterable[FloatArray]) -> None:
    """``base_blocks`` に ``ndarray`` が直接渡された誤用を専用の ``TypeError``
    にする (F-03-4-004)。

    ``base_blocks`` の型は ``FloatArray`` (単一の ``(T, M)`` 配列) から
    ``Iterable[FloatArray]`` (ブロックの列) へ広がったが (F-03-3-002)、
    ``np.ndarray`` はイテレートすると行を返すため ``Iterable[np.ndarray]`` を
    構造的に満たしてしまい、mypy は旧来の呼び方 (2次元配列を直接渡す) を
    素通りする (実測: mypy exit 0)。型で防げない以上、境界で1行落とす。
    """
    if isinstance(base_blocks, np.ndarray):
        raise TypeError(
            "base_blocks に ndarray を直接渡すことはできません "
            f"(F-03-4-004): shape={base_blocks.shape}。"
            " 単一の (T, M) 配列なら [base] のように包んでください"
        )


def _surrogate_capacities(
    problem: CapacityProblem,
    base_blocks: Iterable[FloatArray],
    alpha: float,
    *,
    n_surrogates: int,
    chunk_size: int,
    rng: np.random.Generator,
) -> FloatArray:
    """代表目標の各ブロックをシャッフルサロゲート化し、容量を連結して返す。

    F-03-3-002: round2 (F-03-2-015) で IPC が代表目標 ``base`` を 128MiB 予算
    でブロック化した際、閾値の分位点計算 (``np.quantile``) までも ``ipc.py``
    に複製してしまい、D-27 の rationale (『サロゲートを別経路で計算すると
    閾値と容量が別実装からずれるので同じ関数に流す』) の前提が壊れた。
    ``base_blocks`` を **Iterable of block** にすることで、ブロック化そのもの
    (呼び出し側が 128MiB 予算にどう分けるか) と、しきい値の分位点計算 (この
    共有カーネル1本) を分離する。1ブロックしか無い呼び出し側 (MC) は
    ``[base]`` のように1要素の Iterable を渡せばよい。

    F-03-4-003: モジュール外からの import が0件で、D-27 (『閾値は同じ関数で
    1箇所だけ計算する』) を回避するための材料 (閾値を伴わないサロゲート容量)
    を共有カーネルの公開面にそのまま置いていたため非公開にした
    (``__all__`` からも外した)。公開が必要になったときは D-27 の見直しと
    セットで判断する。

    Args:
        problem: 容量問題 (通常の目標と同じもの)。
        base_blocks: 実目標から決定的に選んだ代表を、呼び出し側が
            (通常は 128MiB 予算で) ブロック化した ``(T_eff, M_i)`` の Iterable。
        alpha: 通常の目標と同じ正則化係数。
        n_surrogates: 代表1本あたりのサロゲート本数。
        chunk_size: 1回の solve に畳む列数 (結果には影響しない、ブロック内の
            サロゲート生成の粒度)。
        rng: ``ctx.seed`` から作った乱数生成器 (D-27: 乱数源はこれだけ)。

    Returns:
        全ブロックのサロゲート容量を連結した ``(Σ M_i * n_surrogates,)``。

    Raises:
        TypeError: ``base_blocks`` に ``ndarray`` が直接渡された場合
            (F-03-4-004)。
        ValueError: ``n_surrogates`` が 1 未満 / ``chunk_size`` が 1 未満 /
            ``base_blocks`` が空 / いずれかのブロックが2次元でない場合。
    """
    _reject_ndarray_base_blocks(base_blocks)
    if n_surrogates < 1:
        raise ValueError(f"n_surrogates は 1 以上が必要です: {n_surrogates}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size は 1 以上が必要です: {chunk_size}")
    pieces: list[FloatArray] = []
    for block in base_blocks:
        base = np.asarray(block, dtype=np.float64)
        if base.ndim != 2:
            raise ValueError(f"base_blocks の要素は (T, M) が必要です: {base.shape}")
        pieces.append(
            capacity_of_chunks(
                problem,
                _iter_surrogate_chunks(base, n_surrogates, chunk_size, rng),
                alpha,
            )
        )
    if not pieces:
        raise ValueError("base_blocks が1つもありません")
    return np.concatenate(pieces)


def surrogate_threshold(
    problem: CapacityProblem,
    base_blocks: Iterable[FloatArray],
    alpha: float,
    *,
    n_surrogates: int,
    quantile: float,
    chunk_size: int,
    rng: np.random.Generator,
) -> tuple[float, FloatArray]:
    """時間シャッフルサロゲートから容量のしきい値を推定する (D-27)。

    ``base_blocks`` の各ブロックの各列を時間方向にシャッフルした系列を
    ``n_surrogates`` 本ずつ作り、**通常の目標とまったく同じ経路**
    (``capacity_of_targets``) に流して容量を測る。閾値はその分位点の**1箇所**
    (この関数) だけで計算する (D-27 / F-03-3-002)。シャッフル列は
    ``chunk_size`` 列ずつその場で生成しては畳んで捨てる (F-03-1-012)。
    ``(n_samples, M * n_surrogates)`` を一括確保しない。

    Args:
        problem: 容量問題 (通常の目標と同じもの)。
        base_blocks: 実目標から決定的に選んだ代表を、呼び出し側がブロック化
            した ``(T_eff, M_i)`` の Iterable。1ブロックしか無い場合は
            ``[base_targets]`` のように1要素の Iterable として渡す。
        alpha: 通常の目標と同じ正則化係数。
        n_surrogates: 代表1本あたりのサロゲート本数。
        quantile: 分位点 (0〜1)。
        chunk_size: 1回の solve に畳む列数 (結果には影響しない)。
        rng: ``ctx.seed`` から作った乱数生成器 (D-27: 乱数源はこれだけ)。

    Returns:
        ``(閾値, 全ブロックのサロゲート容量を連結したもの)``。

    Raises:
        TypeError: ``base_blocks`` に ``ndarray`` が直接渡された場合
            (F-03-4-004。単一の (T, M) 配列は ``[base]`` のように包むこと)。
        ValueError: ``n_surrogates`` が 1 未満 / ``quantile`` が範囲外 /
            ``chunk_size`` が 1 未満 / ``base_blocks`` が空の場合。
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile は 0〜1 が必要です: {quantile}")
    capacities = _surrogate_capacities(
        problem,
        base_blocks,
        alpha,
        n_surrogates=n_surrogates,
        chunk_size=chunk_size,
        rng=rng,
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
    "NORMAL_HERMITE",
    "SUPPORTED_BASIS_PAIRS",
    "SUPPORTED_MEASURES",
    "UNIFORM",
    "UNIFORM_LEGENDRE",
    "CapacityProblem",
    "InputMeasure",
    "RowAlignment",
    "bounded_chunk_size",
    "capacity_of_chunks",
    "capacity_of_targets",
    "chi2_threshold",
    "input_series",
    "orthonormal_basis",
    "surrogate_threshold",
]

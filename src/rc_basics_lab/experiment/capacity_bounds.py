"""実験03 の確保軸の上限と検査 (CWE-789 / CWE-400 / CWE-834).

``experiment/capacity.py`` から**上限と検査だけ**を切り出したモジュール。
行数上限 (D-77) のためだが、切り口としても自然である —— ここは「どれだけ
確保しようとしているか」だけを見ており、容量の測り方も掃引の組み方も知らない。

**確保する前に落とす**のがこの層の役割である。``n_units`` や ``n_steps`` を
大きく書いた設定が、確保に成功してから MemoryError で落ちると何が原因か
分からない。軸ごとに上限を持ち、超えたら値を添えて ``ValueError`` にする。

**04 もここを呼ぶ。** ``freerun.py`` / ``stability.py`` は新しい ``_MAX_*`` を
宣言せず、同じ上限を通す
(``tests/test_config_wiring_chaos.py::test_chaos_config_introduces_no_new_capacity_bound``)。
"""

from __future__ import annotations

_MAX_UNITS = 5_000
"""``CapacityCondition.n_units`` の上書き不能な絶対上限 (F-3b1-1-017, CWE-789)。

``ESN`` の重み生成は ``rng.random((N, N))`` (再帰行列) を確保するため、確保量は
``N**2`` に比例する。3a の D-34 (IPC の確保・組合せ計算量の4段の上限) と同じ
threat model —— 設定 YAML の1行変更 (``conservation.n_units_grid: [100000]``)
だけで防御が無い状態だと数十GB の確保に到達しうる (実測: N=100000 で重み行列
だけで約80GB)。本番設定の最大 ``n_units`` は 200 (3-A) で、``_MAX_UNITS=5000``
は25倍の余裕を残しつつ、重み行列を ``8 * 5000**2`` ≈ 200MB に抑える。
"""


_MAX_STATE_ELEMENTS = 200_000_000
"""``n_units * n_steps`` の上書き不能な絶対上限 (F-3b1-1-017, CWE-400/789)。

状態行列 ``X`` は ``(n_steps, n_units)`` の ``float64`` を確保するため、
確保量は ``n_units * n_steps`` に比例する (D-35 の rationale が言う 4GB 予算と
同じ軸)。本番設定の最大は length_sweep (``n_units=50, n_steps=1_000_000`` =
5e7) で、``_MAX_STATE_ELEMENTS=2e8`` は4倍の余裕を残しつつ状態行列を
``8 * 2e8`` = 1.6GB に抑える。``n_steps`` 単体ではなく積で縛るのは、
``n_units`` が小さければ ``n_steps`` を大きく取れる (length_sweep の実際の
使い方) 一方で、両方を同時に大きくする設定変更は個別の軸の検査をすり抜ける
ため (CWE-789 の threat model は D-34 の rationale と同型)。
"""


_MAX_SEQUENTIAL_RUNS = 2_000
"""逐次で回す ESN シミュレーション本数 (CWE-834) の上書き不能な絶対上限。

ESN の状態更新 (``ESN.run`` / 自走の ``free_run``) は逐次計算でベクトル化
できない (仕様 §10-1) ので、実行時間は「回す本数」に正比例する。04 の
4-A / 4-B は ``base.n_replicates`` を縛る検査がどこにも無く、この値を
YAML の1行変更で任意倍にできた (reviewer-security 実測)。``experiment/
stability.py`` の ``_MAX_CONDITIONS`` (4-C の条件数上限) と**同じ値・同じ
threat model**。``config/chaos04.py`` / ``experiment/freerun.py`` は
新しい ``_MAX_*`` を宣言しない
(``tests/test_config_wiring_chaos.py::test_chaos_config_introduces_no_new_capacity_bound``)
ので、``validate_state_matrix_bounds`` と同じくここに置き、両方の呼び出し元
(``freerun.py`` は4-A/4-B、``stability.py`` は独自の条件数の積を別に持つ) が
再利用する。
"""


def validate_n_units_bound(n_units: int) -> None:
    """``n_units`` 軸だけに絶対上限をかける (D-34)。

    ESN の重み行列は ``n_units**2`` で伸びるので、系列長と無関係にこの軸だけで
    確保が膨らむ。状態行列の軸 (``n_units * n_steps``) と**別の軸**なので、
    片方だけを縛る呼び出し側 (3-C は ``tasks/narma.py`` の ``_validate`` が
    ``length`` と ``length * n_units`` を既に縛っており、欠けていたのは
    ``n_units`` 単体だった) から独立に呼べる形にしてある。

    Args:
        n_units: リザバーのユニット数。

    Raises:
        ValueError: 上限を超える場合 (**確保より前に**落とす)。
    """
    if n_units > _MAX_UNITS:
        raise ValueError(
            f"n_units が上限を超えています: {n_units} > {_MAX_UNITS} "
            "(ESN の重み行列の確保量は n_units**2 に比例するため、"
            "確保する前に検査で落とす)"
        )


def validate_state_matrix_bounds(n_units: int, n_steps: int) -> None:
    """確保軸 ``(n_units, n_steps)`` そのものに絶対上限をかける (D-34)。

    ``CapacityCondition`` を持たない経路 (3-C の ``run_narma10``。状態は 01 の
    ``run_task`` が作る) からも同じ上限を呼べるように、条件オブジェクトではなく
    **軸の値そのもの**を引数に取る。以前は 3-C の ``n_units``
    (``narma.base.esn_mackey_glass.n_units``) を縛るものが1つも無く、
    ``tasks/narma.py`` の ``_validate`` が塞いでいたのは ``length`` 軸だけだった
    (オーケストレータの実測: ``n_units=6000`` で ESN が実際に構築されてから
    無関係な形状エラーで停止していた)。

    Args:
        n_units: リザバーのユニット数。重み行列は ``n_units**2`` で伸びる。
        n_steps: 系列長。状態行列は ``n_units * n_steps`` で伸びる。

    Raises:
        ValueError: いずれかの上限を超える場合 (**確保より前に**落とす)。
    """
    validate_n_units_bound(n_units)
    n_state_elements = n_units * n_steps
    if n_state_elements > _MAX_STATE_ELEMENTS:
        raise ValueError(
            f"n_units * n_steps が上限を超えています: {n_state_elements} > "
            f"{_MAX_STATE_ELEMENTS} (状態行列の確保量は n_units * n_steps に"
            "比例するため、確保する前に検査で落とす)"
        )


def validate_total_step_count(n_total_steps: int) -> None:
    """逐次シミュレーションの総ステップ数 (積の軸) に絶対上限をかける (CWE-834)。

    ``_MAX_STATE_ELEMENTS`` と**同じ値・同じ threat model**を再利用する
    (状態行列の要素数も、逐次シミュレーションの総ステップ数 (例: 4-C の
    「条件数 x stats_steps」) も、どちらも「単位コストが一定の量が多重に
    積み重なる」という同型の脅威モデルである)。個別の軸 (条件数の上限・
    ステップ数の上限) がそれぞれ上限内でも、**積**は両方の軸検査をすり抜けて
    膨らみうる (reviewer-security 実測: 4-C は条件数 <= 2000・stats_steps
    <= 1e6 をどちらも通したまま、積が 1,984,000,000 ステップ = 約13時間
    (予算300秒) に達した)。

    Raises:
        ValueError: 総ステップ数が上限を超える場合。
    """
    if n_total_steps > _MAX_STATE_ELEMENTS:
        raise ValueError(
            "逐次シミュレーションの総ステップ数が上限を超えています: "
            f"{n_total_steps} > {_MAX_STATE_ELEMENTS} "
            "(例: 条件数 x stats_steps。個別の軸がそれぞれ上限内でも、積が"
            "膨らむ経路をここで塞ぐ)"
        )


def validate_sequential_run_count(n_runs: int) -> None:
    """ESN シミュレーションを1本も回す前に、逐次実行の本数を検査する (CWE-834)。

    Raises:
        ValueError: 本数が 1 未満、または ``_MAX_SEQUENTIAL_RUNS`` 超過。
    """
    if n_runs < 1:
        raise ValueError(f"逐次実行の本数が1本もありません: {n_runs}")
    if n_runs > _MAX_SEQUENTIAL_RUNS:
        raise ValueError(
            f"逐次実行の本数が上限を超えています: {n_runs} > {_MAX_SEQUENTIAL_RUNS} "
            "(ESN のシミュレーションは逐次計算なので実行時間はこの本数に比例する)"
        )


__all__ = [
    "validate_n_units_bound",
    "validate_sequential_run_count",
    "validate_state_matrix_bounds",
    "validate_total_step_count",
]

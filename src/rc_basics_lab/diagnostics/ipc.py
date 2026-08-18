"""情報処理容量 (IPC) の診断 (実験 3-B).

Dambre 2012 の情報処理容量。入力の正規直交多項式の積

    ``z[t] = Π_i psi_{n_i}(u[t - k_i])``

を目標に取り、状態 ``X[t]`` からの線形読み出しでどれだけ説明できるかを
全目標について足し上げる。目標が入力測度に対して正規直交である限り、
総和は状態の次元 ``N`` を超えない (保存則)。次数1だけを集めれば MC に一致する。

実装の要点は5つ。

- **目標の列挙は明示的** (仕様 §4 T2-2)。次数 ``d`` の目標は「相異なる遅延
  ``k_1 < ... < k_m`` (``m <= max_variables``)、各次数 ``n_i >= 1``、
  ``Σ n_i = d``、``k_i <= max_delay_by_degree[d-1]``」の全組合せ。数が
  ``max_targets`` を超えたら **黙って切り詰めず** ``ValueError``。
- **行集合は全目標で同一** (D-24)。``t0 = max(ctx.washout, 全次数の最大遅延)``。
  目標ごとに使える行数を変えると、深い遅延・高い次数ほど標本数が減って容量が
  系統的に下がり、それは「記憶が減衰している」という測りたい現象と同じ向きに出る。
- **Gram は1回、solve はチャンク数だけ** (D-26)。目標は ``chunk_size`` 列ずつ
  生成して ``rhs`` に畳んだら捨てる。``fit_ridge_from_gram`` の呼び出し回数は
  ``ceil(K / chunk_size)`` に比例し、目標数 ``K`` には比例しない。
  ``(T, K)`` を実体化しないのはメモリ制約でもある (T=1e6 x 2395 列 = 19 GB)。
- **しきい値は次数ごと** (D-27)。有限標本による容量のかさ上げは目標の周辺分布に
  依存し、それは次数で変わる。サロゲートは**通常の目標とまったく同じ経路**
  (``capacity_of_targets``) に流す。乱数は ``ctx.seed`` のみ。
- **基底は宣言された入力分布に対して正規直交** (D-28)。未対応な
  ``(input_distribution, basis)`` の組は ``ValueError``。

回帰と容量の計算は ``_capacity`` の共有カーネルにあり、MC と1本の実装を共有する。
設定値は ``DiagnosticContext`` ではなく既定値つきキーワード引数 ``cfg`` で渡す (D-15)。
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from rc_basics_lab.diagnostics._capacity import (
    LEGENDRE,
    SUPPORTED_BASIS_PAIRS,
    UNIFORM,
    CapacityProblem,
    capacity_of_chunks,
    orthonormal_basis,
    surrogate_threshold,
)
from rc_basics_lab.diagnostics.base import (
    DiagnosticContext,
    DiagnosticResult,
    resolve_context,
    validate_diagnostic_input,
)
from rc_basics_lab.types import FloatArray

NAME = "ipc"

THRESHOLD_SURROGATE = "surrogate"
"""しきい値法: 時間シャッフルサロゲートの分位点 (既定、D-27)。"""

THRESHOLD_CHI2 = "chi2"
"""しきい値法: カイ二乗近似 (Dambre 2012 SupMat 3.2)。

状態と無相関な目標に対する容量の推定値は、標本数 ``T_eff`` と自由度
``N`` (バイアス列を除く回帰変数の本数) を使って ``chi2_N / T_eff`` に
近似的に従う。その分位点をしきい値にする。次数に依存しないので
``ipc_threshold_degree*`` は全次数で同じ値になる (サロゲート法との差が
そのまま「次数ごとに推定する意味があるか」の一次資料になる)。
"""

THRESHOLD_NONE = "none"
"""しきい値法: しきい値を課さない (生の容量をそのまま足す)。"""

SUPPORTED_THRESHOLD_MODES: tuple[str, ...] = (
    THRESHOLD_SURROGATE,
    THRESHOLD_CHI2,
    THRESHOLD_NONE,
)
"""``IpcConfig.threshold_mode`` が受理する値。

未知の値は ``ValueError`` にする。黙って既定 (サロゲート) にフォールバック
させると、しきい値法の比較 (受け入れ条件3) が「設定したのに効いていない」
状態で通ってしまう。
"""

type TargetSpec = tuple[tuple[int, int], ...]
"""目標1本の仕様 ``((k_1, n_1), ..., (k_m, n_m))``。

``k_i`` は遅延 (1 以上、昇順・相異なる)、``n_i`` はその遅延に掛ける多項式の
次数 (1 以上)。目標の次数は ``Σ n_i``、ヒートマップ上の遅延は ``max k_i``
(仕様 §4 T2-3 の集約規則)。
"""


@dataclass(frozen=True, slots=True)
class IpcConfig:
    """IPC の測定条件。純データ。値域検証は ``ipc`` 側で行う (D-09)。

    Attributes:
        max_delay_by_degree: 次数ごとの遅延の打ち切り。``index d-1`` が次数 ``d``
            に対応し、タプルの長さがそのまま評価する最大次数になる。
        max_variables: 1つの目標に掛け合わせてよい相異なる遅延の本数の上限。
        basis: 多項式基底 (``"legendre"`` / ``"hermite"``)。
        input_distribution: 宣言された入力分布 (``"uniform"`` / ``"normal"``)。
            ``basis`` との組は ``SUPPORTED_BASIS_PAIRS`` に無いと ``ValueError``
            (D-28)。この2つは**対で**意味を持つ。
        alpha: リッジの正則化係数 (D-25 の固定微小値)。
        threshold_mode: しきい値法 (``SUPPORTED_THRESHOLD_MODES``)。
        n_surrogates: 代表目標1本あたりのサロゲート本数。
        n_surrogate_targets: 次数ごとに選ぶ代表目標の本数 (決定的に選ぶ)。
        surrogate_quantile: しきい値に使う分位点。
        chunk_size: 1回の solve に畳む目標の列数。**結果を変えない性能
            パラメータ** であり、他のフィールドとは逆向きの要求を持つ
            (``test_chunk_size_does_not_change_results``)。
        max_targets: 目標数の上限。超えたら黙って切り詰めず ``ValueError``
            (``test_target_enumeration_raises_instead_of_truncating``)。
    """

    max_delay_by_degree: tuple[int, ...] = (60, 20, 10, 6)
    max_variables: int = 3
    basis: str = LEGENDRE
    input_distribution: str = UNIFORM
    alpha: float = 1.0e-9
    threshold_mode: str = THRESHOLD_SURROGATE
    n_surrogates: int = 100
    n_surrogate_targets: int = 4
    surrogate_quantile: float = 0.99
    chunk_size: int = 256
    max_targets: int = 200_000


DEFAULT_IPC = IpcConfig()


def _validate_config(cfg: IpcConfig) -> None:
    """設定の値域を検証する (D-09: 検証は使う側)。"""
    if len(cfg.max_delay_by_degree) < 1:
        raise ValueError("max_delay_by_degree が空です (評価する次数がありません)")
    for degree, max_delay in enumerate(cfg.max_delay_by_degree, start=1):
        if max_delay < 1:
            raise ValueError(
                f"max_delay_by_degree の要素は 1 以上が必要です"
                f" (次数 {degree}: {max_delay})"
            )
    if cfg.max_variables < 1:
        raise ValueError(f"max_variables は 1 以上が必要です: {cfg.max_variables}")
    if (cfg.input_distribution, cfg.basis) not in SUPPORTED_BASIS_PAIRS:
        raise ValueError(
            "(input_distribution, basis) の組が未対応です (D-28): "
            f"({cfg.input_distribution!r}, {cfg.basis!r})。"
            f" 対応する組: {SUPPORTED_BASIS_PAIRS}"
        )
    if cfg.alpha < 0.0:
        raise ValueError(f"alpha は 0 以上が必要です: {cfg.alpha}")
    if cfg.threshold_mode not in SUPPORTED_THRESHOLD_MODES:
        raise ValueError(
            f"未知の threshold_mode です: {cfg.threshold_mode!r}"
            f" (対応: {SUPPORTED_THRESHOLD_MODES})"
        )
    if cfg.chunk_size < 1:
        raise ValueError(f"chunk_size は 1 以上が必要です: {cfg.chunk_size}")
    if cfg.max_targets < 1:
        raise ValueError(f"max_targets は 1 以上が必要です: {cfg.max_targets}")
    if cfg.threshold_mode == THRESHOLD_SURROGATE:
        if cfg.n_surrogates < 1:
            raise ValueError(f"n_surrogates は 1 以上が必要です: {cfg.n_surrogates}")
        if cfg.n_surrogate_targets < 1:
            raise ValueError(
                f"n_surrogate_targets は 1 以上が必要です: {cfg.n_surrogate_targets}"
            )
    if cfg.threshold_mode in (THRESHOLD_SURROGATE, THRESHOLD_CHI2) and not (
        0.0 <= cfg.surrogate_quantile <= 1.0
    ):
        raise ValueError(
            f"surrogate_quantile は 0〜1 が必要です: {cfg.surrogate_quantile}"
        )


def _ordered_compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """``total`` を ``parts`` 個の 1 以上の整数へ分ける**順序つき**の分割。

    順序を区別するのは、遅延 ``(k_1, k_2)`` に次数 ``(1, 2)`` を割り当てた目標と
    ``(2, 1)`` を割り当てた目標が別物だから。``len == C(total-1, parts-1)``。
    """
    if parts == 1:
        yield (total,)
        return
    for head in range(1, total - parts + 2):
        for tail in _ordered_compositions(total - head, parts - 1):
            yield (head, *tail)


def count_targets(cfg: IpcConfig) -> int:
    """列挙する目標の総数を**閉形式で**返す (実体化しない)。

    実際に列挙してから数えると、打ち切りを1桁深くした設定 (例: 次数3 で
    ``max_delay=1000``、``max_variables=3`` なら 5 億通り) で
    ``max_targets`` の検査に到達する前にメモリと時間が尽きる。組合せ数は
    ``Σ_d Σ_m C(K_d, m) * C(d-1, m-1)`` で閉じるので先に数える。
    """
    total = 0
    for degree, max_delay in enumerate(cfg.max_delay_by_degree, start=1):
        for n_vars in range(1, min(cfg.max_variables, degree) + 1):
            total += math.comb(max_delay, n_vars) * math.comb(degree - 1, n_vars - 1)
    return total


def enumerate_targets(cfg: IpcConfig) -> tuple[TargetSpec, ...]:
    """全目標の仕様を次数昇順で列挙する。

    順序は「次数 → 変数本数 → 遅延の組 (辞書順) → 次数の割り当て」で完全に
    決定的。サロゲートの代表目標をこの順序から決定的に選ぶ (D-27) ので、
    順序が変わると閾値も変わる。

    Raises:
        ValueError: 目標数が ``cfg.max_targets`` を超える場合 (**黙って
            切り詰めない**)。
    """
    total = count_targets(cfg)
    if total > cfg.max_targets:
        raise ValueError(
            "目標数が max_targets を超えました "
            f"({total} > {cfg.max_targets})。"
            " 黙って切り詰めないので、max_delay_by_degree / max_variables を"
            " 下げるか max_targets を上げてください"
        )
    if total == 0:
        raise ValueError(
            "目標が1本も列挙されませんでした "
            f"(max_delay_by_degree={cfg.max_delay_by_degree},"
            f" max_variables={cfg.max_variables})"
        )
    specs: list[TargetSpec] = []
    for degree, max_delay in enumerate(cfg.max_delay_by_degree, start=1):
        delays = range(1, max_delay + 1)
        for n_vars in range(1, min(cfg.max_variables, degree) + 1):
            for delay_combo in itertools.combinations(delays, n_vars):
                for orders in _ordered_compositions(degree, n_vars):
                    specs.append(tuple(zip(delay_combo, orders, strict=True)))
    return tuple(specs)


def _target_column(
    problem: CapacityProblem,
    psi_table: Sequence[FloatArray],
    spec: TargetSpec,
) -> FloatArray:
    """目標1本 ``Π_i psi_{n_i}(u[t - k_i])`` を作る。

    ``psi_table[n - 1]`` は次数 ``n`` の正規直交多項式を**系列全体で1回だけ**
    評価したもの。遅延 ``k`` の窓は共有カーネルの ``CapacityProblem.lagged``
    に委譲する (F-03-1-001)。以前はここで ``t0 - delay`` を直接組み立てて
    おり、MC 側の同種の複製 (memory_capacity.py) には値レベルの guard が
    無かった。どの目標も ``problem.t0`` から始まる同一の行集合に対応する (D-24)。
    """
    column: FloatArray | None = None
    for delay, order in spec:
        factor: FloatArray = problem.lagged(psi_table[order - 1], delay)
        column = factor.copy() if column is None else column * factor
    assert column is not None
    return column


def _iter_target_chunks(
    problem: CapacityProblem,
    psi_table: Sequence[FloatArray],
    specs: Sequence[TargetSpec],
    *,
    chunk_size: int,
) -> Iterator[FloatArray]:
    """目標を ``chunk_size`` 列ずつ作って渡し、畳んだら捨てる (D-26)。

    ``(T_eff, K)`` を一度も実体化しないのがこの関数の存在理由。
    """
    n_samples = problem.n_samples
    for start in range(0, len(specs), chunk_size):
        block = specs[start : start + chunk_size]
        chunk: FloatArray = np.empty((n_samples, len(block)), dtype=np.float64)
        for column, spec in enumerate(block):
            chunk[:, column] = _target_column(problem, psi_table, spec)
        yield chunk


def _degree_bounds(
    degree_of: Sequence[int], n_degrees: int
) -> tuple[tuple[int, int], ...]:
    """次数ごとの目標 index の半開区間 ``[start, end)`` を返す。

    ``enumerate_targets`` は次数昇順に並べるので、次数の集合は連続区間になる。
    区間で持てば「次数 d の目標」を毎回走査せずに切り出せる。
    """
    bounds: list[tuple[int, int]] = []
    cursor = 0
    for degree in range(1, n_degrees + 1):
        start = cursor
        while cursor < len(degree_of) and degree_of[cursor] == degree:
            cursor += 1
        bounds.append((start, cursor))
    return tuple(bounds)


def _surrogate_indices(start: int, end: int, n_selected: int) -> tuple[int, ...]:
    """次数内の目標から代表を**決定的に**選ぶ index (等間隔、D-27)。

    先頭から詰めて取らないのは、列挙順の先頭が「変数1本」の目標に偏っており、
    次数 ``d`` の周辺分布の代表として ``psi_d`` 1本だけを見ることになるため。
    等間隔なら変数本数の違う目標がまんべんなく入る。乱数で選ぶと閾値が
    非再現になる (D-27 が禁じているのはまさにこれ)。
    """
    span = end - start
    count = min(n_selected, span)
    if count <= 1:
        return (start,)
    step = (span - 1) / (count - 1)
    return tuple(sorted({start + round(index * step) for index in range(count)}))


def _degree_thresholds(
    problem: CapacityProblem,
    psi_table: Sequence[FloatArray],
    specs: Sequence[TargetSpec],
    bounds: Sequence[tuple[int, int]],
    cfg: IpcConfig,
    *,
    seed: int | None,
) -> tuple[float, ...]:
    """次数ごとのしきい値を返す (D-27)。

    サロゲートは代表目標を時間シャッフルして**通常の目標とまったく同じ経路**
    (``capacity_of_targets``) に流す。別経路で計算すると、閾値と容量が別実装から
    ずれても誰も気づけない。乱数は ``seed`` から作った1本の Generator だけで、
    次数の順に消費する (次数をまたいで再現するのはこの順序があるから)。カイ二乗
    近似のしきい値 (``chi2_threshold``) は共有カーネル側にある (F-03-1-003:
    仕様書は当初から「chi2 は T2 で共有カーネルに足す」としており、以前は
    ``ipc.py`` の private 関数として食い違っていた)。

    Raises:
        ValueError: ``surrogate`` なのに ``seed`` が ``None`` の場合 (D-27)。
    """
    n_degrees = len(bounds)
    if cfg.threshold_mode == THRESHOLD_NONE:
        return (0.0,) * n_degrees
    if cfg.threshold_mode == THRESHOLD_CHI2:
        threshold = chi2_threshold(
            n_units=problem.n_units,
            n_samples=problem.n_samples,
            quantile=cfg.surrogate_quantile,
        )
        return tuple(threshold if end > start else 0.0 for start, end in bounds)
    if seed is None:
        raise ValueError("threshold_mode='surrogate' には ctx.seed が必要です (D-27)")
    rng = np.random.default_rng(seed)
    n_samples = problem.n_samples
    thresholds: list[float] = []
    for start, end in bounds:
        if end <= start:
            thresholds.append(0.0)
            continue
        picked = _surrogate_indices(start, end, cfg.n_surrogate_targets)
        base: FloatArray = np.empty((n_samples, len(picked)), dtype=np.float64)
        for column, index in enumerate(picked):
            base[:, column] = _target_column(problem, psi_table, specs[index])
        threshold, _ = surrogate_threshold(
            problem,
            base,
            cfg.alpha,
            n_surrogates=cfg.n_surrogates,
            quantile=cfg.surrogate_quantile,
            chunk_size=cfg.chunk_size,
            rng=rng,
        )
        thresholds.append(threshold)
    return tuple(thresholds)


def ipc(
    X: FloatArray,
    u: FloatArray | None = None,
    y: FloatArray | None = None,
    *,
    ctx: DiagnosticContext | None = None,
    cfg: IpcConfig = DEFAULT_IPC,
) -> DiagnosticResult:
    """情報処理容量 (IPC) の次数・遅延分解を返す。

    Args:
        X: 状態系列 ``(T, N)``。ESN 由来でなくてよい (受け入れ条件6)。
        u: 入力系列 ``(T, 1)``。**必須**。
        y: 未使用 (プロトコル適合のために受け取る)。
        ctx: ``washout`` と ``seed`` を参照する。``threshold_mode="surrogate"``
            では ``seed`` が必須 (D-27)。
        cfg: 測定条件 (D-15)。

    Returns:
        ``scalars``: ``ipc_total`` (しきい値後の総容量) / ``ipc_total_raw`` /
        ``ipc_linear`` (次数1の合計) / ``ipc_nonlinear`` (次数2以上の合計) /
        ``ipc_threshold_degree{d}`` (次数ごとのしきい値) / ``n_targets`` /
        ``n_targets_kept`` / ``saturation_ratio`` (``ipc_total / N``)。
        ``arrays``: ``ipc_heatmap`` (``(次数, 遅延)`` のセルに ``max k_i`` で
        集約したしきい値後の容量。列 index ``j`` が遅延 ``j+1``) /
        ``ipc_by_degree`` / ``ipc_by_degree_raw``。

    Raises:
        ValueError: ``u`` が無い / 設定が範囲外 / 目標数が ``max_targets``
            超過 / 系列が短すぎる / ``surrogate`` で ``ctx.seed`` が無い場合。
    """
    validate_diagnostic_input(X, u, y, ctx)
    _validate_config(cfg)
    context = resolve_context(ctx)
    series = _input_series(u)

    specs = enumerate_targets(cfg)
    n_steps = int(np.asarray(X).shape[0])
    max_delay = max(cfg.max_delay_by_degree)
    # D-24: 全目標で同一の行集合。基準点は washout と全次数の最大遅延の大きい方。
    t0 = max(context.washout, max_delay)
    if t0 >= n_steps:
        raise ValueError(
            "系列が短すぎます: "
            f"t0=max(washout={context.washout}, max_delay={max_delay})={t0}"
            f" >= T={n_steps}"
        )
    problem = CapacityProblem.from_states(X, t0=t0)
    n_samples = problem.n_samples

    # 正規直交化は系列全体で1回だけ (遅延ごとに標準化し直すと遅延ごとに別の
    # 測度で直交化することになり、保存則が破れる)。
    n_degrees = len(cfg.max_delay_by_degree)
    psi_table: list[FloatArray] = [
        orthonormal_basis(series, degree, cfg.input_distribution, basis=cfg.basis)
        for degree in range(1, n_degrees + 1)
    ]

    degree_of: tuple[int, ...] = tuple(
        sum(order for _, order in spec) for spec in specs
    )
    cell_of: tuple[int, ...] = tuple(max(delay for delay, _ in spec) for spec in specs)
    bounds = _degree_bounds(degree_of, n_degrees)

    # しきい値を先に出すのは fail fast のため (ctx.seed 忘れを、K 本ぶんの
    # 回帰を回し切ってから告げるのではなく着手前に落とす)。乱数は
    # ここでしか使わないので、順序を変えても再現性には影響しない。
    thresholds = _degree_thresholds(
        problem, psi_table, specs, bounds, cfg, t0=t0, seed=context.seed
    )

    capacities = capacity_of_chunks(
        problem,
        _iter_target_chunks(
            psi_table, specs, t0=t0, n_samples=n_samples, chunk_size=cfg.chunk_size
        ),
        cfg.alpha,
    )
    threshold_per_target: FloatArray = np.asarray(
        [thresholds[degree - 1] for degree in degree_of], dtype=np.float64
    )
    kept: FloatArray = np.where(capacities > threshold_per_target, capacities, 0.0)

    # 集約規則 (仕様 §4 T2-3): 目標を (次数 d, max(k_i)) のセルへ足し込む。
    heatmap: FloatArray = np.zeros((n_degrees, max_delay), dtype=np.float64)
    by_degree: FloatArray = np.zeros(n_degrees, dtype=np.float64)
    by_degree_raw: FloatArray = np.zeros(n_degrees, dtype=np.float64)
    for index, (degree, cell) in enumerate(zip(degree_of, cell_of, strict=True)):
        heatmap[degree - 1, cell - 1] += kept[index]
        by_degree[degree - 1] += kept[index]
        by_degree_raw[degree - 1] += capacities[index]

    total = float(np.sum(kept))
    linear = float(by_degree[0])
    scalars: dict[str, float] = {
        "ipc_total": total,
        "ipc_total_raw": float(np.sum(capacities)),
        "ipc_linear": linear,
        "ipc_nonlinear": total - linear,
        "n_targets": float(len(specs)),
        "n_targets_kept": float(np.count_nonzero(kept)),
        "saturation_ratio": total / float(problem.n_units),
    }
    for degree, threshold in enumerate(thresholds, start=1):
        scalars[f"ipc_threshold_degree{degree}"] = threshold

    return DiagnosticResult(
        name=NAME,
        scalars=scalars,
        arrays={
            "ipc_heatmap": heatmap,
            "ipc_by_degree": by_degree,
            "ipc_by_degree_raw": by_degree_raw,
        },
        params={
            "washout": str(context.washout),
            "t0": str(t0),
            "n_samples": str(n_samples),
            "n_units": str(problem.n_units),
            "max_delay_by_degree": repr(cfg.max_delay_by_degree),
            "max_variables": str(cfg.max_variables),
            "basis": cfg.basis,
            "input_distribution": cfg.input_distribution,
            "alpha": repr(cfg.alpha),
            "threshold_mode": cfg.threshold_mode,
            "n_surrogates": str(cfg.n_surrogates),
            "n_surrogate_targets": str(cfg.n_surrogate_targets),
            "surrogate_quantile": repr(cfg.surrogate_quantile),
            "chunk_size": str(cfg.chunk_size),
            "max_targets": str(cfg.max_targets),
            "seed": str(context.seed),
        },
    )


__all__ = [
    "DEFAULT_IPC",
    "NAME",
    "SUPPORTED_THRESHOLD_MODES",
    "THRESHOLD_CHI2",
    "THRESHOLD_NONE",
    "THRESHOLD_SURROGATE",
    "IpcConfig",
    "TargetSpec",
    "count_targets",
    "enumerate_targets",
    "ipc",
]

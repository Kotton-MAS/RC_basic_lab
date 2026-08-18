"""線形メモリ容量 (MC) の診断 (実験 3-A).

Jaeger 2002 の短期記憶容量。遅延 ``k`` の入力 ``u[t-k]`` を状態 ``X[t]`` からの
線形読み出しでどれだけ復元できるかを ``k`` について足し上げた量で、
理論上限は状態の次元 ``N`` (Dambre 2012 の保存則の次数1成分)。

実装の要点は3つ。

- **行集合は全遅延で同一** (D-24)。``t0 = max(ctx.washout, cfg.max_delay)`` を
  単一の基準点にする。遅延ごとに使える行数を変えると、深い遅延ほど標本数が
  減って容量が系統的に下がる —— これは測りたい現象 (記憶の減衰) と**まったく
  同じ向き**に出るため、プロファイルの図を見ても気づけない。
- **正則化は固定の微小 alpha** (D-25)。容量は「線形読み出しで到達可能な最大の
  説明率」という定義そのものなので、検証分割による alpha 選択 (D-04) は行わない。
- **有限標本のかさ上げをサロゲートで差し引く** (D-27)。時間シャッフルした目標を
  通常の目標とまったく同じ経路に流し、その分位点をしきい値にする。

回帰と容量の計算は ``_capacity`` の共有カーネルに置いてあり、IPC (T2) と
1本の実装を共有する。設定値は ``DiagnosticContext`` ではなく既定値つき
キーワード引数 ``cfg`` で渡す (D-15)。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from rc_basics_lab.diagnostics._capacity import (
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

NAME = "memory_capacity"

THRESHOLD_SURROGATE = "surrogate"
"""しきい値法: 時間シャッフルサロゲートの分位点 (既定、D-27)。"""

THRESHOLD_NONE = "none"
"""しきい値法: しきい値を課さない (生の容量をそのまま足す)。"""

SUPPORTED_THRESHOLD_MODES: tuple[str, ...] = (THRESHOLD_SURROGATE, THRESHOLD_NONE)
"""``MemoryCapacityConfig.threshold_mode`` が受理する値。

未知の値は ``ValueError`` にする。黙って既定 (サロゲート) にフォールバック
させると、しきい値法の比較 (受け入れ条件3) が「設定したのに効いていない」
状態で通ってしまう。
"""

_SURROGATE_BASE_DELAY = 1
"""サロゲートの素材にする遅延 (決定的に選ぶ、D-27)。

次数1の目標はどの遅延でも同じ周辺分布 (入力の1次正規直交多項式) を持つので、
時間シャッフル後の容量分布は遅延に依存しない。代表を乱数で選ぶと閾値が
非再現になるため、最も浅い遅延に固定する。
"""


@dataclass(frozen=True, slots=True)
class MemoryCapacityConfig:
    """MC の測定条件。純データ。値域検証は ``memory_capacity`` 側で行う (D-09)。

    Attributes:
        max_delay: 評価する最大遅延 [ステップ]。``t0`` の基準点でもある (D-24)。
        alpha: リッジの正則化係数 (D-25 の固定微小値)。
        threshold_mode: しきい値法 (``SUPPORTED_THRESHOLD_MODES``)。
        n_surrogates: サロゲート本数 (``threshold_mode="surrogate"`` のとき)。
        surrogate_quantile: しきい値に使う分位点。
        chunk_size: 1回の solve に畳む目標の列数。**結果を変えない性能
            パラメータ** であり、他のフィールドとは逆向きの要求を持つ
            (``test_chunk_size_does_not_change_results``)。
    """

    max_delay: int = 400
    alpha: float = 1.0e-9
    threshold_mode: str = THRESHOLD_SURROGATE
    n_surrogates: int = 100
    surrogate_quantile: float = 0.99
    chunk_size: int = 256


DEFAULT_MEMORY_CAPACITY = MemoryCapacityConfig()


def _validate_config(cfg: MemoryCapacityConfig) -> None:
    """設定の値域を検証する (D-09: 検証は使う側)。"""
    if cfg.max_delay < 1:
        raise ValueError(f"max_delay は 1 以上が必要です: {cfg.max_delay}")
    if cfg.alpha < 0.0:
        raise ValueError(f"alpha は 0 以上が必要です: {cfg.alpha}")
    if cfg.threshold_mode not in SUPPORTED_THRESHOLD_MODES:
        raise ValueError(
            f"未知の threshold_mode です: {cfg.threshold_mode!r}"
            f" (対応: {SUPPORTED_THRESHOLD_MODES})"
        )
    if cfg.chunk_size < 1:
        raise ValueError(f"chunk_size は 1 以上が必要です: {cfg.chunk_size}")
    if cfg.threshold_mode == THRESHOLD_SURROGATE:
        if cfg.n_surrogates < 1:
            raise ValueError(f"n_surrogates は 1 以上が必要です: {cfg.n_surrogates}")
        if not 0.0 <= cfg.surrogate_quantile <= 1.0:
            raise ValueError(
                f"surrogate_quantile は 0〜1 が必要です: {cfg.surrogate_quantile}"
            )


def _input_series(u: FloatArray | None) -> FloatArray:
    """``u`` を1次元の入力系列にして返す。無い / 多変数なら ``ValueError``。"""
    if u is None:
        raise ValueError(
            "memory_capacity は入力系列 u が必須です (遅延目標を作れません)"
        )
    series = np.asarray(u, dtype=np.float64)
    if series.shape[1] != 1:
        raise ValueError(
            f"memory_capacity は1変数入力のみ対応です: u.shape={series.shape}"
        )
    if not np.all(np.isfinite(series)):
        raise ValueError("u に有限でない値があります")
    return series[:, 0]


def _iter_delay_chunks(
    psi: FloatArray,
    delays: Sequence[int],
    *,
    t0: int,
    n_samples: int,
    chunk_size: int,
) -> Iterator[FloatArray]:
    """遅延目標を ``chunk_size`` 列ずつ作って渡す (D-26)。

    ``psi`` は入力の1次正規直交多項式を**系列全体で1回だけ**評価したもの。
    遅延 ``k`` の目標は ``psi[t0 - k : T - k]`` という同じ長さのビューであり、
    どの遅延も ``t0`` から始まる同一の行集合に対応する (D-24)。
    """
    for start in range(0, len(delays), chunk_size):
        block = delays[start : start + chunk_size]
        chunk: FloatArray = np.empty((n_samples, len(block)), dtype=np.float64)
        for column, delay in enumerate(block):
            chunk[:, column] = psi[t0 - delay : t0 - delay + n_samples]
        yield chunk


def _effective_delay(profile: FloatArray, delays: FloatArray) -> float:
    """容量で重み付けした平均遅延 (プロファイルの重心)。

    「しきい値を超える最大の遅延」にしないのは、サロゲート閾値が分位点である
    以上、深い遅延にも一定割合の偽陽性が出るためで、その1本が指標を
    ``max_delay`` に張り付かせてしまう。重心はプロファイル全体の形で決まるので
    偽陽性1本では動かない。容量が全く残らない場合は 0 を返す。
    """
    total = float(np.sum(profile))
    if total <= 0.0:
        return 0.0
    return float(np.sum(delays * profile) / total)


def memory_capacity(
    X: FloatArray,
    u: FloatArray | None = None,
    y: FloatArray | None = None,
    *,
    ctx: DiagnosticContext | None = None,
    cfg: MemoryCapacityConfig = DEFAULT_MEMORY_CAPACITY,
) -> DiagnosticResult:
    """線形メモリ容量 (MC) とその遅延プロファイルを返す。

    Args:
        X: 状態系列 ``(T, N)``。ESN 由来でなくてよい (受け入れ条件6)。
        u: 入力系列 ``(T, 1)``。**必須**。
        y: 未使用 (プロトコル適合のために受け取る)。
        ctx: ``washout`` と ``seed`` を参照する。``threshold_mode="surrogate"``
            では ``seed`` が必須 (D-27: 閾値が黙って非再現になるのを防ぐ)。
        cfg: 測定条件 (D-15)。

    Returns:
        ``scalars``: ``mc_total`` (しきい値後の総容量) / ``mc_total_raw``
        (しきい値前) / ``mc_threshold`` / ``mc_effective_delay`` (容量重心) /
        ``mc_ratio`` (``mc_total / N``) / ``n_delays``。
        ``arrays``: ``mc_profile`` (しきい値後の遅延プロファイル ``(max_delay,)``、
        index 0 が遅延1)。

    Raises:
        ValueError: ``u`` が無い / 設定が範囲外 / 系列が短すぎる /
            ``surrogate`` で ``ctx.seed`` が無い場合。
    """
    validate_diagnostic_input(X, u, y, ctx)
    _validate_config(cfg)
    context = resolve_context(ctx)
    series = _input_series(u)

    n_steps = int(np.asarray(X).shape[0])
    # D-24: 全遅延で同一の行集合。基準点は washout と最大遅延の大きい方。
    t0 = max(context.washout, cfg.max_delay)
    if t0 >= n_steps:
        raise ValueError(
            "系列が短すぎます: "
            f"t0=max(washout={context.washout}, max_delay={cfg.max_delay})={t0}"
            f" >= T={n_steps}"
        )
    problem = CapacityProblem.from_states(X, t0=t0)
    n_samples = problem.n_samples

    # 正規直交化は系列全体で1回だけ行う (遅延ごとに標準化し直すと、遅延ごとに
    # 別の測度で直交化することになり保存則が破れる)。次数1は入力分布に
    # よらないので distribution は既定の uniform で足りる。
    psi = orthonormal_basis(series, 1, UNIFORM)
    delays = tuple(range(1, cfg.max_delay + 1))
    profile_raw = capacity_of_chunks(
        problem,
        _iter_delay_chunks(
            psi, delays, t0=t0, n_samples=n_samples, chunk_size=cfg.chunk_size
        ),
        cfg.alpha,
    )

    if cfg.threshold_mode == THRESHOLD_SURROGATE:
        if context.seed is None:
            raise ValueError(
                "threshold_mode='surrogate' には ctx.seed が必要です (D-27)"
            )
        offset = t0 - _SURROGATE_BASE_DELAY
        base: FloatArray = psi[offset : offset + n_samples].reshape(n_samples, 1)
        threshold, _ = surrogate_threshold(
            problem,
            base,
            cfg.alpha,
            n_surrogates=cfg.n_surrogates,
            quantile=cfg.surrogate_quantile,
            chunk_size=cfg.chunk_size,
            rng=np.random.default_rng(context.seed),
        )
        profile: FloatArray = np.where(profile_raw > threshold, profile_raw, 0.0)
    else:
        threshold = 0.0
        profile = profile_raw

    delay_axis: FloatArray = np.asarray(delays, dtype=np.float64)
    total = float(np.sum(profile))
    return DiagnosticResult(
        name=NAME,
        scalars={
            "mc_total": total,
            "mc_total_raw": float(np.sum(profile_raw)),
            "mc_threshold": threshold,
            "mc_effective_delay": _effective_delay(profile, delay_axis),
            "mc_ratio": total / float(problem.n_units),
            "n_delays": float(cfg.max_delay),
        },
        arrays={"mc_profile": profile},
        params={
            "washout": str(context.washout),
            "t0": str(t0),
            "n_samples": str(n_samples),
            "n_units": str(problem.n_units),
            "max_delay": str(cfg.max_delay),
            "alpha": repr(cfg.alpha),
            "threshold_mode": cfg.threshold_mode,
            "n_surrogates": str(cfg.n_surrogates),
            "surrogate_quantile": repr(cfg.surrogate_quantile),
            "chunk_size": str(cfg.chunk_size),
            "seed": str(context.seed),
        },
    )


__all__ = [
    "DEFAULT_MEMORY_CAPACITY",
    "NAME",
    "SUPPORTED_THRESHOLD_MODES",
    "THRESHOLD_NONE",
    "THRESHOLD_SURROGATE",
    "MemoryCapacityConfig",
    "memory_capacity",
]

"""エコー状態性 (ESP) と条件付き Lyapunov 指数の診断 (実験 2-A / 2-B / 2-C).

2本とも「同じ入力を受ける2つの軌道が初期状態の違いを忘れるか」を測る。
``esp_convergence`` は2軌道の距離が実際に潰れるかを直接見る (大域的)。
``conditional_lyapunov`` は参照軌道まわりの微小摂動の成長率を見る (局所的)。
片方だけでは「たまたま選んだ初期状態対では収束した」/「線形化の範囲では
縮むが大域では縮まない」を切り分けられないため、両方を出して符号の整合を
実験層で検査する。

設定値 (閾値・窓・摂動幅) は ``DiagnosticContext`` ではなく既定値つきの
キーワード引数 ``cfg`` で渡す (D-15)。``ctx`` に入れてよいのは系そのものを
表すデータ (第2軌道・伝播器) だけである。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rc_basics_lab.diagnostics.base import (
    DiagnosticContext,
    DiagnosticResult,
    StatePropagator,
    resolve_context,
    validate_diagnostic_input,
)
from rc_basics_lab.types import FloatArray

NAME_ESP = "esp_convergence"
NAME_LYAPUNOV = "conditional_lyapunov"

_PERTURBATION_METHOD = "perturbation"
"""``LyapunovConfig.method`` が受理する唯一の値 (解析 Jacobian 版の差し込み口)。"""

_N_PROPAGATOR_CHECKS = 5
"""伝播器の整合検査を行う時刻の数 (等間隔に取る)。"""

_DIRECTION_SEED = 20240202
"""``ctx.seed`` が ``None`` のときに摂動方向の生成へ使う既定シード。

方向を乱数で引くのは、特定の初期方向 (例: 全成分同符号) が Jacobian の
主固有方向と直交して指数が過小評価される事故を避けるため。既定値を固定して
おくことで ``ctx.seed`` を渡さない呼び出しでも結果は再現する。
"""


@dataclass(frozen=True, slots=True)
class EspConfig:
    """ESP 判定の基準 (D-16)。純データ。値域検証は ``esp_convergence`` が行う。

    Attributes:
        abs_tol: 収束と見なす末尾距離の絶対閾値。
        rel_tol: 初期距離に対する相対閾値。
        window: 末尾距離を測る窓幅 [ステップ]。
        fit_skip: 減衰率の当てはめで washout 直後から追加で捨てるステップ数。
        floor: 対数を取る前の距離の下限。これ以下の点は当てはめから除く。
    """

    abs_tol: float = 1.0e-6
    rel_tol: float = 1.0e-3
    window: int = 200
    fit_skip: int = 50
    floor: float = 1.0e-14


@dataclass(frozen=True, slots=True)
class LyapunovConfig:
    """条件付き Lyapunov 指数の推定条件 (D-18)。純データ。

    Attributes:
        method: 推定法。``"perturbation"`` 以外は ``ValueError``。
        delta: 摂動の大きさ (RMS/ユニット距離)。
        renorm_interval: 再正規化までのステップ数。
        max_growth: 1区間あたりの成長率の上限。超えたら ``ValueError``。
        check_propagator: ``propagator(X[t], t) == X[t+1]`` を実行時に検査するか。
        propagator_tol: その検査の許容 RMS/ユニット距離。
    """

    method: str = _PERTURBATION_METHOD
    delta: float = 1.0e-8
    renorm_interval: int = 1
    max_growth: float = 1.0e3
    check_propagator: bool = True
    propagator_tol: float = 1.0e-10


DEFAULT_ESP = EspConfig()
DEFAULT_LYAPUNOV = LyapunovConfig()


def _rms_distance(a: FloatArray, b: FloatArray) -> FloatArray:
    """RMS/ユニット距離 ``||a - b|| / sqrt(N)`` を時刻ごとに返す (D-16)。

    ``sqrt(N)`` で割るのは閾値をユニット数に依存させないため。N の異なる系
    (メモリスタ配列など) へ同じ閾値のまま移植できる。
    """
    difference = a - b
    n_units = difference.shape[-1]
    norms: FloatArray = np.linalg.norm(difference, axis=-1) / np.sqrt(n_units)
    return norms


def _fit_decay_rate(
    distance: FloatArray, start: int, floor: float
) -> tuple[float, int]:
    """``log distance`` の傾き [1/ステップ] と当てはめ点数を返す。

    ``floor`` 以下の点を落とすのは、収束後に距離が丸め誤差の床 (あるいは厳密な
    0) に張り付いた区間を含めると傾きが 0 側へ引っ張られるため。点が2点未満なら
    ``nan`` を返す (「測れなかった」を 0 と区別する)。
    """
    steps = np.arange(distance.shape[0], dtype=np.float64)
    usable = (steps >= start) & (distance > floor)
    n_points = int(np.count_nonzero(usable))
    if n_points < 2:
        return float("nan"), n_points
    slope, _ = np.polyfit(steps[usable], np.log(distance[usable]), 1)
    return float(slope), n_points


def _validate_esp_config(cfg: EspConfig) -> None:
    if cfg.abs_tol < 0.0:
        raise ValueError(f"abs_tol は 0 以上である必要があります: {cfg.abs_tol!r}")
    if cfg.rel_tol < 0.0:
        raise ValueError(f"rel_tol は 0 以上である必要があります: {cfg.rel_tol!r}")
    if cfg.window < 1:
        raise ValueError(f"window は 1 以上である必要があります: {cfg.window!r}")
    if cfg.fit_skip < 0:
        raise ValueError(f"fit_skip は 0 以上である必要があります: {cfg.fit_skip!r}")
    if cfg.floor <= 0.0:
        raise ValueError(f"floor は正である必要があります: {cfg.floor!r}")


def esp_convergence(
    X: FloatArray,
    u: FloatArray | None = None,
    y: FloatArray | None = None,
    *,
    ctx: DiagnosticContext | None = None,
    cfg: EspConfig = DEFAULT_ESP,
) -> DiagnosticResult:
    """同一入力・異なる初期状態の2軌道が合流するか (ESP) を判定する。

    Args:
        X: 参照軌道の状態系列 ``(T, N)``。
        u: 未使用 (プロトコル適合のために受け取る)。
        y: 未使用 (同上)。
        ctx: ``washout`` と ``companion_states`` を参照する。
            ``companion_states`` は **1本以上必須**。
        cfg: 判定基準 (D-15)。

    Returns:
        ``scalars``: ``d_initial`` / ``d_tail`` / ``converged`` /
        ``decay_rate_per_step`` / ``n_pairs`` / ``n_fit_points``。
        ``arrays``: ``distance`` (最悪ペアの曲線 ``(T,)``) /
        ``distance_all`` (``(n_pairs, T)``)。

    Raises:
        ValueError: ``companion_states`` が空、``T < washout + window``、
            または ``cfg`` の値域外。

    Note:
        ``companion_states`` を必須にしているのは、第2軌道が渡っていないまま
        「ESP 成立」という判定だけが返る事故を殺すため。ESP は2軌道の関係で
        あって単一軌道の性質ではない。
    """
    validate_diagnostic_input(X, u, y, ctx)
    _validate_esp_config(cfg)
    context = resolve_context(ctx)
    states = np.asarray(X, dtype=np.float64)
    n_steps = states.shape[0]
    companions = context.companion_states
    if not companions:
        raise ValueError(
            "esp_convergence には ctx.companion_states が 1 本以上必要です "
            "(ESP は2軌道の合流であり、単一軌道からは判定できない): 0 本"
        )
    if n_steps < context.washout + cfg.window:
        raise ValueError(
            "系列長が washout + window に満たないため末尾窓を取れません: "
            f"T={n_steps}, washout={context.washout}, window={cfg.window}"
        )

    distance_all: FloatArray = np.stack(
        [
            _rms_distance(states, np.asarray(companion, dtype=np.float64))
            for companion in companions
        ]
    )
    tail_all: FloatArray = np.median(distance_all[:, -cfg.window :], axis=1)
    worst = int(np.argmax(tail_all))
    distance: FloatArray = distance_all[worst]

    # 最悪値で判定する (D-16): 末尾距離は最大、初期距離は最小を取ることで
    # 閾値が最も厳しくなる組み合わせを使う。報告する d_initial / d_tail と
    # converged が常に同じ不等式で結ばれるようにするため、ペアごとの判定の
    # AND ではなくこの形にしている。
    d_tail = float(tail_all[worst])
    d_initial = float(np.min(distance_all[:, 0]))
    threshold = max(cfg.abs_tol, cfg.rel_tol * d_initial)
    converged = d_tail <= threshold

    decay_rate, n_fit_points = _fit_decay_rate(
        distance, context.washout + cfg.fit_skip, cfg.floor
    )

    return DiagnosticResult(
        name=NAME_ESP,
        scalars={
            "d_initial": d_initial,
            "d_tail": d_tail,
            "converged": float(converged),
            "decay_rate_per_step": decay_rate,
            "n_pairs": float(len(companions)),
            "n_fit_points": float(n_fit_points),
        },
        arrays={"distance": distance, "distance_all": distance_all},
        params={
            "washout": str(context.washout),
            "abs_tol": str(cfg.abs_tol),
            "rel_tol": str(cfg.rel_tol),
            "window": str(cfg.window),
            "fit_skip": str(cfg.fit_skip),
            "floor": str(cfg.floor),
        },
    )


def _validate_lyapunov_config(cfg: LyapunovConfig) -> None:
    if cfg.method != _PERTURBATION_METHOD:
        raise ValueError(
            "conditional_lyapunov が対応する method は "
            f"{_PERTURBATION_METHOD!r} のみです (解析 Jacobian 版は未実装): "
            f"{cfg.method!r}"
        )
    if cfg.delta <= 0.0:
        raise ValueError(f"delta は正である必要があります: {cfg.delta!r}")
    if cfg.renorm_interval < 1:
        raise ValueError(
            f"renorm_interval は 1 以上である必要があります: {cfg.renorm_interval!r}"
        )
    if cfg.max_growth <= 0.0:
        raise ValueError(f"max_growth は正である必要があります: {cfg.max_growth!r}")
    if cfg.propagator_tol < 0.0:
        raise ValueError(
            f"propagator_tol は 0 以上である必要があります: {cfg.propagator_tol!r}"
        )


def _check_propagator_consistency(
    states: FloatArray,
    propagator: StatePropagator,
    first: int,
    last: int,
    tol: float,
) -> None:
    """``propagator(X[t], t) == X[t+1]`` を数点で検査する (D-18)。

    伝播器が参照軌道と別の入力で回っている (最有力は ``u[t]`` と ``u[t+1]`` の
    取り違え) と、指数は"それらしい値"で出るためレビューでは気づけない。
    ここで実行時に落とす。
    """
    candidates = np.unique(
        np.linspace(first, last, num=_N_PROPAGATOR_CHECKS).astype(int)
    )
    for index in candidates:
        t = int(index)
        predicted = np.asarray(propagator(states[t], t), dtype=np.float64)
        if predicted.shape != states[t].shape:
            raise ValueError(
                "propagator の戻り値の形状が状態と一致しません: "
                f"{predicted.shape} != {states[t].shape}"
            )
        mismatch = float(_rms_distance(predicted, states[t + 1]))
        if not mismatch <= tol:
            raise ValueError(
                "propagator(X[t], t) が X[t+1] と一致しません "
                "(参照軌道と別の入力で伝播している疑い。X[t] は u[t] を処理した "
                "後の状態なので propagator は u[t+1] を使う): "
                f"t={t}, RMS/ユニット距離={mismatch!r} > tol={tol!r}"
            )


def conditional_lyapunov(
    X: FloatArray,
    u: FloatArray | None = None,
    y: FloatArray | None = None,
    *,
    ctx: DiagnosticContext | None = None,
    cfg: LyapunovConfig = DEFAULT_LYAPUNOV,
) -> DiagnosticResult:
    """参照軌道まわりの微小摂動の成長率 (条件付き Lyapunov 指数) を推定する。

    ``t = washout .. T-2`` を ``cfg.renorm_interval`` ごとに区切り、区間頭で
    ``x̃ = X[t] + delta * e`` (``e`` は RMS/ユニットノルムが 1 の方向)、区間末で
    ``g = ||x̃ - X[t_end]|| / sqrt(N) / delta`` を測って ``log g`` を累積し、
    方向を再正規化して次区間へ進む (Benettin 法)。

    Args:
        X: 参照軌道の状態系列 ``(T, N)``。
        u: 未使用 (伝播器が入力を閉じ込めているため)。
        y: 未使用 (プロトコル適合のために受け取る)。
        ctx: ``washout`` / ``dt`` / ``seed`` / ``propagator`` を参照する。
            ``propagator`` は必須。
        cfg: 推定条件 (D-15 / D-18)。

    Returns:
        ``scalars``: ``lyapunov_per_step`` / ``lyapunov_per_time`` /
        ``n_intervals`` / ``max_observed_growth``。

    Raises:
        ValueError: ``ctx.propagator`` が ``None``、``cfg.method`` が
            ``"perturbation"`` 以外、伝播器が参照軌道と不整合、
            成長率が ``cfg.max_growth`` 超、または区間が1つも取れない。
    """
    validate_diagnostic_input(X, u, y, ctx)
    _validate_lyapunov_config(cfg)
    context = resolve_context(ctx)
    propagator = context.propagator
    if propagator is None:
        raise ValueError(
            "conditional_lyapunov には ctx.propagator が必要です "
            "(伝播器なしでは摂動を進められない): None"
        )

    states = np.asarray(X, dtype=np.float64)
    n_steps, n_units = states.shape
    last_start = n_steps - 2
    if context.washout > last_start:
        raise ValueError(
            "washout 後に 1 ステップも進められません: "
            f"T={n_steps}, washout={context.washout}"
        )
    if cfg.check_propagator:
        _check_propagator_consistency(
            states, propagator, context.washout, last_start, cfg.propagator_tol
        )

    scale = float(n_units)
    rng = np.random.default_rng(
        _DIRECTION_SEED if context.seed is None else context.seed
    )
    direction: FloatArray = np.asarray(
        rng.standard_normal(size=n_units), dtype=np.float64
    )
    direction = direction / float(np.linalg.norm(direction))

    log_growth_total = 0.0
    n_measured_steps = 0
    n_intervals = 0
    max_observed_growth = 0.0
    start = context.washout
    while start <= last_start:
        length = min(cfg.renorm_interval, n_steps - 1 - start)
        perturbed: FloatArray = states[start] + (cfg.delta * scale) * direction
        for offset in range(length):
            perturbed = np.asarray(
                propagator(perturbed, start + offset), dtype=np.float64
            )
        separation: FloatArray = perturbed - states[start + length]
        separation_norm = float(np.linalg.norm(separation))
        growth = separation_norm / scale / cfg.delta
        if growth > cfg.max_growth:
            raise ValueError(
                "摂動の成長が max_growth を超えました (線形域を外れており "
                "指数の推定が成り立たない。delta か renorm_interval を見直すこと): "
                f"t={start}, growth={growth!r} > max_growth={cfg.max_growth!r}"
            )
        if growth <= 0.0:
            raise ValueError(
                "摂動が完全に消失したため log を取れません "
                f"(delta を大きくすること): t={start}, growth={growth!r}"
            )
        log_growth_total += float(np.log(growth))
        n_measured_steps += length
        n_intervals += 1
        max_observed_growth = max(max_observed_growth, growth)
        direction = separation / separation_norm
        start += length

    lyapunov_per_step = log_growth_total / n_measured_steps
    return DiagnosticResult(
        name=NAME_LYAPUNOV,
        scalars={
            "lyapunov_per_step": lyapunov_per_step,
            "lyapunov_per_time": lyapunov_per_step / context.dt,
            "n_intervals": float(n_intervals),
            "max_observed_growth": max_observed_growth,
        },
        params={
            "washout": str(context.washout),
            "method": cfg.method,
            "delta": str(cfg.delta),
            "renorm_interval": str(cfg.renorm_interval),
            "max_growth": str(cfg.max_growth),
            "check_propagator": str(cfg.check_propagator),
            "propagator_tol": str(cfg.propagator_tol),
        },
    )


__all__ = [
    "DEFAULT_ESP",
    "DEFAULT_LYAPUNOV",
    "NAME_ESP",
    "NAME_LYAPUNOV",
    "EspConfig",
    "LyapunovConfig",
    "conditional_lyapunov",
    "esp_convergence",
]

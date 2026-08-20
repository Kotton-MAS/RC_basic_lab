"""最大 Lyapunov 指数の数値推定 (実験04、D-42).

自律系 (入力を持たない力学系) の軌道 ``X`` と、その1サンプルぶんの伝播器
``ctx.propagator`` から lambda_max を Benettin 法で推定する。``ctx.dt``
(1サンプルあたりの時間) で割って**時間あたり**の指数を返すので、
Lyapunov 時間 ``1 / lambda_max`` がそのまま有効予測時間の正規化 (D-43) に
使える。

**正本は数値推定である** (D-42)。文献値 (Lorenz なら 0.9056、
``config.chaos04.LORENZ_LYAPUNOV_REFERENCE``) は ``cfg.reference_value`` に
渡したときに**照合として**報告されるだけで、計算には一切使わない。文献値を
正本にすると、積分刻み・サンプリング間隔・burn-in の取り違えが指標に現れず、
図でも有効予測時間でも検出できない。

Benettin 法の反復そのものは ``diagnostics/esp.py`` の ``conditional_lyapunov``
に既にある。**同じ反復を書き直さない**: 自律系は「入力が無い = 伝播器が時刻に
依存しない」場合であり、条件付き Lyapunov 指数の特別な場合そのものである
(``memory_capacity`` と ``ipc`` が ``_capacity`` を共有しているのと同じ形)。
ここが足すのは (a) 自律系向けの名前と ``params``、(b) Lyapunov 時間、
(c) 文献値との照合の3つで、いずれも反復の中身ではない。

命名規約 (D-52): 公開関数名を ``lyapunov`` に**しない**。パッケージ属性
``diagnostics.lyapunov`` がモジュールでなくなると
``import rc_basics_lab.diagnostics.lyapunov as m`` が関数を返し、
``monkeypatch.setattr(m, ...)`` が何も差し替えないまま成功する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rc_basics_lab.diagnostics.base import (
    DiagnosticContext,
    DiagnosticResult,
    resolve_context,
    validate_diagnostic_input,
)
from rc_basics_lab.diagnostics.esp import LyapunovConfig, conditional_lyapunov
from rc_basics_lab.types import FloatArray

NAME_MAX_LYAPUNOV = "max_lyapunov"

_DEFAULT_RENORM_INTERVAL = 10
"""自律系での既定の再正規化間隔 [サンプル]。

``conditional_lyapunov`` の既定 (1 サンプル) より長く取る。駆動リザバーでは
1ステップの成長率が大きいが、連続時間系をサンプリングした軌道では
1サンプルあたりの成長が ``exp(lambda * dt)`` (Lorenz / Delta t=0.02 なら
約 1.018) しかなく、``log`` を取る量の有効桁が落ちる。10 サンプルまとめると
成長率が約 1.20 になり、摂動 (既定 delta=1e-8) を線形域に置いたまま
有効桁を1桁以上稼げる。
"""


def _default_estimator() -> LyapunovConfig:
    """自律系向けの ``LyapunovConfig`` の既定 (再正規化間隔だけを変える)。"""
    return LyapunovConfig(renorm_interval=_DEFAULT_RENORM_INTERVAL)


@dataclass(frozen=True, slots=True)
class MaxLyapunovConfig:
    """lambda_max の推定条件と文献値との照合基準 (D-15 / D-42)。純データ。

    Attributes:
        estimator: Benettin 法の推定条件 (摂動幅・再正規化間隔・伝播器の
            整合検査)。``conditional_lyapunov`` とまったく同じ器を使う。
        reference_value: 照合に使う文献値 [1/時間]。``None`` なら照合しない。
            **推定には使わない** (D-42)。
        reference_rel_tol: 照合の相対許容幅 (``matches_reference`` の基準)。
    """

    estimator: LyapunovConfig = field(default_factory=_default_estimator)
    reference_value: float | None = None
    reference_rel_tol: float = 0.05


DEFAULT_MAX_LYAPUNOV = MaxLyapunovConfig()


def _validate_config(cfg: MaxLyapunovConfig) -> None:
    """照合基準の値域を検証する (推定条件側は委譲先が検証する)。"""
    if cfg.reference_value is not None and cfg.reference_value == 0.0:
        raise ValueError(
            "reference_value に 0 は使えません "
            "(相対誤差が定義できない。照合しないなら None にすること)"
        )
    if cfg.reference_rel_tol < 0.0:
        raise ValueError(
            "reference_rel_tol は 0 以上である必要があります: "
            f"{cfg.reference_rel_tol!r}"
        )


def max_lyapunov(
    X: FloatArray,
    u: FloatArray | None = None,
    y: FloatArray | None = None,
    *,
    ctx: DiagnosticContext | None = None,
    cfg: MaxLyapunovConfig = DEFAULT_MAX_LYAPUNOV,
) -> DiagnosticResult:
    """自律系の最大 Lyapunov 指数を Benettin 法で推定する (D-42)。

    ``t = washout .. T-2`` を ``cfg.estimator.renorm_interval`` ごとに区切り、
    区間頭で微小摂動を与え、区間末で成長率の対数を累積して方向を再正規化する。
    1サンプルあたりの指数を ``ctx.dt`` で割ることで**時間あたり**にする。

    Args:
        X: 軌道 ``(T, D)``。**生の (標準化していない) 状態**を渡すこと ——
            成分ごとに違う倍率で割ると、有限時間の成長率が座標系に依存する。
        u: 未使用 (自律系には入力が無い。プロトコル適合のために受け取る)。
        y: 未使用 (同上)。
        ctx: ``washout`` (burn-in) / ``dt`` (1サンプルあたりの時間) /
            ``seed`` (摂動方向) / ``propagator`` を参照する。
            ``propagator`` は必須で、**1サンプル**進める写像でなければならない
            (``propagator(X[t], t) == X[t+1]`` を委譲先が実行時に検査する)。
        cfg: 推定条件と照合基準 (D-15)。

    Returns:
        ``scalars``: ``lyapunov_per_step`` (1サンプルあたり) /
        ``lyapunov_per_time`` (``ctx.dt`` で正規化) / ``lyapunov_time``
        (``1 / lyapunov_per_time``。指数が正でなければ ``nan``) /
        ``n_intervals`` / ``max_observed_growth``。``cfg.reference_value`` を
        渡した場合は ``reference_value`` / ``reference_rel_error`` /
        ``matches_reference`` が加わる。

    Raises:
        ValueError: ``ctx.propagator`` が ``None``、伝播器が軌道と不整合、
            摂動が線形域を外れた、または ``cfg`` の値域外。
    """
    validate_diagnostic_input(X, u, y, ctx)
    _validate_config(cfg)
    context = resolve_context(ctx)
    estimated = conditional_lyapunov(X, ctx=ctx, cfg=cfg.estimator)

    per_step = estimated.scalars["lyapunov_per_step"]
    per_time = estimated.scalars["lyapunov_per_time"]
    scalars: dict[str, float] = {
        "lyapunov_per_step": per_step,
        "lyapunov_per_time": per_time,
        # Lyapunov 時間。指数が正でないときに 1/lambda を返すと「負の時間」や
        # 発散した値が有効予測時間の分母に入るので nan で「測れなかった」を残す。
        "lyapunov_time": 1.0 / per_time if per_time > 0.0 else float("nan"),
        "n_intervals": estimated.scalars["n_intervals"],
        "max_observed_growth": estimated.scalars["max_observed_growth"],
    }
    params = {**estimated.params, "dt": str(context.dt)}
    if cfg.reference_value is not None:
        relative_error = abs(per_time - cfg.reference_value) / abs(cfg.reference_value)
        scalars["reference_value"] = float(cfg.reference_value)
        scalars["reference_rel_error"] = float(relative_error)
        scalars["matches_reference"] = float(relative_error <= cfg.reference_rel_tol)
        params["reference_rel_tol"] = str(cfg.reference_rel_tol)
    return DiagnosticResult(name=NAME_MAX_LYAPUNOV, scalars=scalars, params=params)


__all__ = [
    "DEFAULT_MAX_LYAPUNOV",
    "NAME_MAX_LYAPUNOV",
    "MaxLyapunovConfig",
    "max_lyapunov",
]

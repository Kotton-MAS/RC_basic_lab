"""最大 Lyapunov 指数の診断の検査 (D-42).

正本は**数値推定**である。文献値 (Viswanath 1998 の 0.9056) は照合にしか
使わないので、ここでは

1. 推定値が文献値の ±5% に入ること (D-42 の guard_test)
2. 文献値を変えても**推定値が1ビットも動かない**こと (照合にしか使っていない
   ことの実測。1. だけだと「文献値をそのまま返す実装」でも緑になる)
3. ``ctx.dt`` が正規化の分母として実際に効いていること (D-43 の分母)

の3本を測る。1. の被験体は Lorenz なので、``tasks/chaotic.py`` の軌道と伝播器を
使う (診断層は ``tasks`` を import しないため、配線はテスト側で行う)。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rc_basics_lab.config import LORENZ_LYAPUNOV_REFERENCE, LorenzConfig
from rc_basics_lab.diagnostics.base import DiagnosticContext, StatePropagator
from rc_basics_lab.diagnostics.esp import LyapunovConfig
from rc_basics_lab.diagnostics.lyapunov import (
    DEFAULT_MAX_LYAPUNOV,
    NAME_MAX_LYAPUNOV,
    MaxLyapunovConfig,
    max_lyapunov,
)
from rc_basics_lab.tasks.chaotic import (
    initial_state,
    integrate_lorenz,
    lorenz_sample_step,
    sampling_interval,
)
from rc_basics_lab.types import FloatArray

LORENZ_TRAJECTORY_SAMPLES = 8000
"""推定に使う軌道の長さ [サンプル]。

本番設定 (``lorenz.length``) と同じ。``Delta t = 0.01`` なので 80 時間単位 =
約 73 Lyapunov 時間ぶんで、再正規化間隔 10 サンプルなら 800 区間になる。
"""


def _lorenz_config() -> LorenzConfig:
    """本番と同じ Lorenz 設定 (推定手順を design.md と一致させるため)。"""
    return LorenzConfig()


def _lorenz_setup(
    cfg: LorenzConfig, seed: int = 20240406
) -> tuple[FloatArray, StatePropagator]:
    """生の Lorenz 軌道と、1サンプル進める伝播器を返す。

    **生の (標準化していない) 軌道**を使う。成分ごとに違う倍率で割ると、
    有限時間の成長率が座標系に依存する。
    """
    trajectory = integrate_lorenz(
        cfg, initial_state(np.random.default_rng(seed)), LORENZ_TRAJECTORY_SAMPLES
    )

    def propagator(x: FloatArray, t: int) -> FloatArray:
        # 自律系なので t に依存しない (それが「条件付き」でないことの実体)。
        return lorenz_sample_step(cfg, x)

    return trajectory, propagator


def test_estimated_lyapunov_matches_the_literature_value() -> None:
    """推定 lambda_max が文献値 0.9056 の ±5% 以内 (D-42)。

    推定手順 (軌道長 8000 サンプル / 再正規化間隔 10 サンプル / burn-in は
    ``lorenz.integration_burn_in`` = 1000 サンプル / 摂動 1e-8) は
    ``docs/design.md`` に記録してある。
    """
    cfg = _lorenz_config()
    trajectory, propagator = _lorenz_setup(cfg)
    ctx = DiagnosticContext(dt=sampling_interval(cfg), propagator=propagator)
    result = max_lyapunov(
        trajectory, ctx=ctx, cfg=MaxLyapunovConfig(reference_value=0.9056)
    )

    estimated = result.scalars["lyapunov_per_time"]
    relative = abs(estimated - 0.9056) / 0.9056
    assert relative <= 0.05, (
        f"推定 lambda_max={estimated!r} が文献値 0.9056 の ±5% を外れました "
        f"(相対差 {relative:.4f})"
    )
    assert result.name == NAME_MAX_LYAPUNOV
    assert result.scalars["matches_reference"] == 1.0
    assert result.scalars["reference_rel_error"] == pytest.approx(relative)
    # Lyapunov 時間 (D-43 の正規化の分母) は指数の逆数。
    assert result.scalars["lyapunov_time"] == pytest.approx(1.0 / estimated)


def test_the_literature_value_is_only_used_for_comparison() -> None:
    """``reference_value`` を変えても推定値が1ビットも動かない (D-42)。

    「文献値は照合にのみ使い、正本は数値推定」の実測。文献値を返すだけの実装
    (あるいは文献値へ寄せる実装) はここで落ちる。
    """
    cfg = _lorenz_config()
    trajectory, propagator = _lorenz_setup(cfg)
    ctx = DiagnosticContext(dt=sampling_interval(cfg), propagator=propagator)

    with_reference = max_lyapunov(
        trajectory, ctx=ctx, cfg=MaxLyapunovConfig(reference_value=0.9056)
    )
    absurd = max_lyapunov(
        trajectory, ctx=ctx, cfg=MaxLyapunovConfig(reference_value=42.0)
    )
    without = max_lyapunov(trajectory, ctx=ctx, cfg=MaxLyapunovConfig())

    estimate = with_reference.scalars["lyapunov_per_time"]
    assert absurd.scalars["lyapunov_per_time"] == estimate
    assert without.scalars["lyapunov_per_time"] == estimate
    # 照合の結果だけが動く。
    assert absurd.scalars["matches_reference"] == 0.0
    assert "reference_value" not in without.scalars
    assert "matches_reference" not in without.scalars


def test_the_reference_value_in_the_config_matches_the_recorded_citation() -> None:
    """``Chaos04Config`` の既定が ``LORENZ_LYAPUNOV_REFERENCE`` と一致する。"""
    from rc_basics_lab.config import Chaos04Config

    assert LORENZ_LYAPUNOV_REFERENCE == 0.9056
    assert Chaos04Config().lyapunov.reference_value == LORENZ_LYAPUNOV_REFERENCE


def test_lyapunov_is_normalized_by_ctx_dt() -> None:
    """``ctx.dt`` が正規化の分母として効いている (D-42 / D-43)。

    ``DiagnosticContext`` に**フィールドを足さず** (D-01)、既に在る ``dt`` を
    使うことの実測でもある。
    """
    cfg = _lorenz_config()
    trajectory, propagator = _lorenz_setup(cfg)
    dt = sampling_interval(cfg)
    result = max_lyapunov(
        trajectory,
        ctx=DiagnosticContext(dt=dt, propagator=propagator),
        cfg=MaxLyapunovConfig(),
    )
    halved = max_lyapunov(
        trajectory,
        ctx=DiagnosticContext(dt=dt / 2.0, propagator=propagator),
        cfg=MaxLyapunovConfig(),
    )
    assert result.scalars["lyapunov_per_step"] == halved.scalars["lyapunov_per_step"]
    assert halved.scalars["lyapunov_per_time"] == pytest.approx(
        2.0 * result.scalars["lyapunov_per_time"]
    )
    assert result.params["dt"] == str(dt)


def test_max_lyapunov_requires_a_propagator() -> None:
    """``ctx.propagator`` なしでは推定できない (委譲先の必須条件)。"""
    rng = np.random.default_rng(0)
    states: FloatArray = rng.standard_normal((50, 3))
    with pytest.raises(ValueError, match="propagator"):
        max_lyapunov(states, ctx=DiagnosticContext(dt=0.01))


def test_lyapunov_time_is_nan_for_a_non_positive_exponent() -> None:
    """指数が正でないときは Lyapunov 時間を ``nan`` にする。

    0 除算も「負の Lyapunov 時間」も出さない。
    縮小写像 (``x -> 0.5 x``) を伝播器にすると指数は負になる。1/lambda を
    そのまま返す実装だと「負の Lyapunov 時間」が有効予測時間の分母に入る。
    """
    decay = 0.5
    n_steps, n_units = 40, 2
    states: FloatArray = np.empty((n_steps, n_units), dtype=np.float64)
    states[0] = np.array([1.0, -1.0])
    for index in range(1, n_steps):
        states[index] = decay * states[index - 1]

    def propagator(x: FloatArray, t: int) -> FloatArray:
        scaled: FloatArray = decay * np.asarray(x, dtype=np.float64)
        return scaled

    result = max_lyapunov(
        states,
        ctx=DiagnosticContext(dt=1.0, propagator=propagator),
        cfg=MaxLyapunovConfig(
            estimator=LyapunovConfig(renorm_interval=1, delta=1.0e-6)
        ),
    )
    assert result.scalars["lyapunov_per_time"] < 0.0
    assert np.isnan(result.scalars["lyapunov_time"])


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        pytest.param(
            MaxLyapunovConfig(reference_value=0.0), "reference_value", id="zero_ref"
        ),
        pytest.param(
            MaxLyapunovConfig(reference_rel_tol=-0.1),
            "reference_rel_tol",
            id="negative_tol",
        ),
    ],
)
def test_config_values_out_of_range_are_rejected(
    cfg: MaxLyapunovConfig, match: str
) -> None:
    """判定基準の値域外は ``ValueError`` (黙って既定へ落ちない、D-09)。"""
    rng = np.random.default_rng(0)
    states: FloatArray = rng.standard_normal((20, 2))

    def propagator(x: FloatArray, t: int) -> FloatArray:
        shifted: FloatArray = states[t + 1] + (x - states[t])
        return shifted

    with pytest.raises(ValueError, match=match):
        max_lyapunov(states, ctx=DiagnosticContext(propagator=propagator), cfg=cfg)


def test_default_config_does_not_carry_a_literature_value() -> None:
    """診断層の既定は系に依存しない (Lorenz の文献値は ``config`` 側が持つ)。

    ``diagnostics`` は ``config`` を import しない (D-12 / D-23) 以上、
    ここに 0.9056 を既定として書くと「診断が特定の系を知っている」ことになる。
    """
    assert DEFAULT_MAX_LYAPUNOV.reference_value is None
    assert DEFAULT_MAX_LYAPUNOV.estimator.renorm_interval == 10


def test_config_is_a_frozen_dataclass_of_pure_data() -> None:
    """設定は純データ (D-15)。``estimator`` は委譲先とまったく同じ器。"""
    assert dataclasses.is_dataclass(MaxLyapunovConfig)
    assert isinstance(DEFAULT_MAX_LYAPUNOV.estimator, LyapunovConfig)
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_MAX_LYAPUNOV.reference_rel_tol = 0.1  # type: ignore[misc]

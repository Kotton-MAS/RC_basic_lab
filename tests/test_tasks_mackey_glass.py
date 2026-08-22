"""Mackey-Glass 課題のテスト.

「設定したのに効いていない」を潰すため、tau / horizon の変更が実際に出力を
変えることを確認する。加えて、生成系列が**周期解に落ちていない**ことを
自己相関と初期値鋭敏性の2方向から確認する (積分の遅延項を取り違えると
リミットサイクルに落ちるが、系列を眺めても気づけない)。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rc_basics_lab.config import MackeyGlassConfig
from rc_basics_lab.tasks import mackey_glass as mg
from rc_basics_lab.tasks.base import TaskGenerator
from rc_basics_lab.tasks.mackey_glass import generate_mackey_glass, integrate
from rc_basics_lab.types import FloatArray

SHORT = MackeyGlassConfig(length=600)

# 署名適合の確認は mypy が行う (make type)。実行時は import できれば十分。
_GENERATOR: TaskGenerator[MackeyGlassConfig] = generate_mackey_glass


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _autocorrelation(series: FloatArray, lag: int) -> float:
    centered = series - series.mean()
    total = float(np.dot(centered, centered))
    return float(np.dot(centered[: len(centered) - lag], centered[lag:]) / total)


def test_shapes_and_finiteness() -> None:
    data = generate_mackey_glass(SHORT, _rng())
    assert data.u.shape == (SHORT.length, 1)
    assert data.y.shape == (SHORT.length, 1)
    assert data.name == "mackey_glass"
    assert data.params["tau"] == str(SHORT.tau)


def test_series_stays_on_the_attractor() -> None:
    """バーンイン後の値が MG(tau=17) の既知の値域に収まる。"""
    series = integrate(SHORT, _rng(), 2000)
    assert 0.2 < float(series.min()) < 0.8
    assert 1.0 < float(series.max()) < 1.6


def test_tau_changes_trajectory() -> None:
    """tau=17 と tau=30 で軌道が異なる。"""
    short_tau = generate_mackey_glass(SHORT, _rng())
    long_tau = generate_mackey_glass(dataclasses.replace(SHORT, tau=30.0), _rng())
    assert not np.allclose(short_tau.u, long_tau.u)
    # 同じ乱数列から出発しているので、差は tau だけに由来する
    assert short_tau.params["tau"] != long_tau.params["tau"]


def test_horizon_changes_target() -> None:
    """horizon 1 -> 5 で目標配列が変わる。"""
    one = generate_mackey_glass(SHORT, _rng())
    five = generate_mackey_glass(dataclasses.replace(SHORT, horizon=5), _rng())
    assert np.array_equal(one.u, five.u)  # 入力は同じ
    assert not np.allclose(one.y, five.y)


def test_target_is_input_shifted_by_horizon() -> None:
    """y[t] == u[t + horizon] (予測課題としての整合)。"""
    horizon = 3
    data = generate_mackey_glass(dataclasses.replace(SHORT, horizon=horizon), _rng())
    assert np.array_equal(data.y[: SHORT.length - horizon], data.u[horizon:])


def test_is_chaotic_not_periodic() -> None:
    """自己相関が長ラグで減衰する (周期解なら包絡線が 1 付近に留まる)。"""
    series = integrate(SHORT, _rng(), 20000)
    assert _autocorrelation(series, 1) > 0.95  # サンプリング間隔 1.0 では滑らか
    long_lag = [abs(_autocorrelation(series, lag)) for lag in range(3000, 4000, 17)]
    assert max(long_lag) < 0.6


def test_nearby_initial_conditions_diverge(monkeypatch: pytest.MonkeyPatch) -> None:
    """初期履歴の 1e-8 の差が育つ (カオスの直接的な確認)。"""
    monkeypatch.setattr(mg, "INITIAL_JITTER", 1e-8)
    cfg = dataclasses.replace(SHORT, integration_burn_in=0)
    first = integrate(cfg, _rng(0), 4000)
    second = integrate(cfg, _rng(1), 4000)
    separation = np.abs(first - second)
    assert float(separation[:10].max()) < 1e-6
    assert float(separation[-100:].max()) > 1e-2


def test_same_seed_is_deterministic() -> None:
    first = generate_mackey_glass(SHORT, _rng(3))
    second = generate_mackey_glass(SHORT, _rng(3))
    assert np.array_equal(first.u, second.u)


def test_task_seed_changes_trajectory() -> None:
    first = generate_mackey_glass(SHORT, _rng(3))
    second = generate_mackey_glass(SHORT, _rng(4))
    assert not np.allclose(first.u, second.u)


def test_non_integer_delay_ratio_raises() -> None:
    """tau / rk4_step が整数でない設定は黙って丸めずに落とす。"""
    with pytest.raises(ValueError, match="整数"):
        generate_mackey_glass(dataclasses.replace(SHORT, tau=17.05), _rng())


@pytest.mark.parametrize(
    "cfg",
    [
        dataclasses.replace(SHORT, rk4_step=0.0),
        dataclasses.replace(SHORT, tau=0.0),
        dataclasses.replace(SHORT, sample_interval=0),
        dataclasses.replace(SHORT, length=0),
        dataclasses.replace(SHORT, horizon=0),
        dataclasses.replace(SHORT, integration_burn_in=-1),
        dataclasses.replace(SHORT, exponent=0),
    ],
    ids=["rk4_step", "tau", "sample_interval", "length", "horizon", "burn_in", "n"],
)
def test_invalid_parameters_raise(cfg: MackeyGlassConfig) -> None:
    with pytest.raises(ValueError):
        generate_mackey_glass(cfg, _rng())

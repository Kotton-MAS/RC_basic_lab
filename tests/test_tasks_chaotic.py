"""カオス課題 (Lorenz と 04 の標準化) の検査 (D-41).

3系統ある。

1. **積分の正しさ** —— ``scipy.integrate.solve_ivp`` による**独立実装**と短時間
   区間で突き合わせる。この検査は ``tasks/chaotic.py`` のループにも定数にも
   触らない (触ると「自分の実装と自分の実装が一致する」同語反復になり、
   係数を全部間違えても緑になる。D-29 の guard の流儀)
2. **標準化係数の出どころ** —— 訓練区間の先頭から推定した1組が全区間へ当たって
   いること。評価区間で推定し直す実装は、図でも有効予測時間でも検出できない
   壊れ方をする (仕様 §10-2) ので、ここで係数そのものを突き合わせる
3. **確保軸** —— 軸1 (積分ステップ数) と軸2 (真の軌道の配列要素数) が
   **確保より前に**、かつ**それぞれ独立に**効くこと
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from rc_basics_lab.config import LorenzConfig, MackeyGlassConfig
from rc_basics_lab.tasks import chaotic
from rc_basics_lab.tasks.chaotic import (
    LORENZ_STATE_DIM,
    TASK_NAME_LORENZ,
    Standardizer,
    generate_lorenz,
    generate_standardized_mackey_glass,
    initial_state,
    integrate_lorenz,
    lorenz_sample_step,
    sampling_interval,
)
from rc_basics_lab.tasks.mackey_glass import TASK_NAME as TASK_NAME_MACKEY_GLASS
from rc_basics_lab.types import FloatArray

# 独立実装の側で書き下す Lorenz のパラメータ。**実装から import しない**
# (実装の定数を参照すると、係数を取り違えても両側が同じだけずれて緑になる)。
REFERENCE_SIGMA = 10.0
REFERENCE_RHO = 28.0
REFERENCE_BETA = 8.0 / 3.0

REFERENCE_X0 = (1.0, 1.0, 1.0)
"""突き合わせに使う初期条件 (乱数を介さず両側へ同じ値を渡す)。"""


def _reference_rhs(t: float, state: FloatArray) -> list[float]:
    """Lorenz の右辺 (独立実装。``tasks/chaotic.py`` を一切参照しない)。"""
    x, y, z = float(state[0]), float(state[1]), float(state[2])
    return [
        REFERENCE_SIGMA * (y - x),
        x * (REFERENCE_RHO - z) - y,
        x * y - REFERENCE_BETA * z,
    ]


def _short_config(**overrides: object) -> LorenzConfig:
    """短時間区間の突き合わせ用の設定 (バーンインなし)。"""
    base = {
        "rk4_step": 0.002,
        "sample_interval": 5,
        "integration_burn_in": 0,
        "length": 200,
        "horizon": 1,
        "standardize_steps": 100,
    }
    return LorenzConfig(**{**base, **overrides})  # type: ignore[arg-type]


def test_lorenz_matches_reference_trajectory() -> None:
    """RK4 の軌道が独立実装 (solve_ivp, rtol/atol 1e-10) と相対 1e-6 以内 (D-41)。

    比較は「差の最大値 / 軌道のスケール」で取る。成分ごとの相対誤差にすると
    x・y がゼロを横切る点で分母が消え、実装の正しさと無関係な位置で落ちる。
    """
    cfg = _short_config()
    n_samples = cfg.length
    dt = sampling_interval(cfg)
    x0 = np.array(REFERENCE_X0, dtype=np.float64)

    mine = integrate_lorenz(cfg, x0, n_samples)
    times = [(index + 1) * dt for index in range(n_samples)]
    solution = solve_ivp(
        _reference_rhs,
        (0.0, times[-1]),
        x0,
        t_eval=times,
        rtol=1.0e-10,
        atol=1.0e-10,
        method="RK45",
    )
    assert solution.success, solution.message
    reference: FloatArray = np.asarray(solution.y, dtype=np.float64).T

    scale = float(np.max(np.abs(reference)))
    relative = float(np.max(np.abs(mine - reference))) / scale
    assert relative < 1.0e-6, f"独立実装との相対差が大きすぎます: {relative:e}"


def test_lorenz_sample_step_reproduces_the_sampled_trajectory() -> None:
    """``lorenz_sample_step`` が軌道の次のサンプルとビット一致する (D-18 の前提)。

    ``diagnostics/lyapunov.py`` はこれを ``ctx.propagator`` として使い、
    委譲先の ``conditional_lyapunov`` が ``propagator(X[t], t) == X[t+1]`` を
    実行時に検査する。ここが近似一致でしかないと、伝播器の整合検査が
    ``propagator_tol`` の設定次第で通ったり落ちたりする。
    """
    cfg = _short_config(length=50, standardize_steps=50)
    trajectory = integrate_lorenz(cfg, np.array(REFERENCE_X0), 50)
    for index in (0, 17, 48):
        stepped = lorenz_sample_step(cfg, trajectory[index])
        assert np.array_equal(stepped, trajectory[index + 1]), (
            f"index={index} で伝播器と軌道が一致しません"
        )


def test_lorenz_parameters_are_not_configurable() -> None:
    """(sigma, rho, beta) が ``LorenzConfig`` のフィールドでない (D-41)。

    設定にすると「カオス域かどうか」も文献値 0.9056 の意味も黙って変わる。
    ``tasks/narma.py`` の係数を設定にしない D-29 と同じ規律。
    """
    field_names = {item.name for item in dataclasses.fields(LorenzConfig)}
    assert not field_names & {"sigma", "rho", "beta"}, (
        f"Lorenz の系パラメータが設定になっています: {sorted(field_names)}"
    )
    assert (chaotic.LORENZ_SIGMA, chaotic.LORENZ_RHO) == (10.0, 28.0)
    assert pytest.approx(8.0 / 3.0) == chaotic.LORENZ_BETA


def test_standardization_uses_the_training_prefix_coefficients_everywhere() -> None:
    """標準化係数が**先頭 ``standardize_steps`` 行**から推定した1組である (D-41)。

    実測は3段で、全部そろって初めて「訓練区間から推定した1組を全区間で使う」に
    なる:

    1. 生の軌道の先頭から自前で推定した係数を当てた結果と**厳密に一致**する
    2. 全区間から推定し直した係数を当てた結果とは**一致しない**
       (1. だけだと、両者が偶然ほぼ同じ値になる系で空虚になる)
    3. ``u`` と ``y`` が**同じ**係数で標準化されている (自走は出力を入力へ
       戻すので、片方だけ別の係数だと単位が食い違う)
    """
    cfg = _short_config(length=400, standardize_steps=100)
    seed = 20240401
    raw = integrate_lorenz(
        cfg, initial_state(np.random.default_rng(seed)), cfg.length + cfg.horizon
    )
    task = generate_lorenz(cfg, np.random.default_rng(seed))

    prefix = raw[: cfg.standardize_steps]
    mean = np.mean(prefix, axis=0)
    scale = np.std(prefix, axis=0)
    expected_u = (raw[: cfg.length] - mean) / scale
    expected_y = (raw[cfg.horizon : cfg.horizon + cfg.length] - mean) / scale
    assert np.array_equal(task.u, expected_u)
    assert np.array_equal(task.y, expected_y)

    whole_mean = np.mean(raw, axis=0)
    whole_scale = np.std(raw, axis=0)
    assert not np.allclose(task.u, (raw[: cfg.length] - whole_mean) / whole_scale), (
        "全区間から推定し直した係数と区別がついていません (この検査は空虚です)"
    )

    # 3. u と y が同じ係数を共有していること: 重なる区間の値が一致する。
    assert np.array_equal(task.u[cfg.horizon :], task.y[: cfg.length - cfg.horizon])


def test_standardized_series_is_not_standardized_again_outside_the_prefix() -> None:
    """推定区間の外では平均0・分散1に**ならない** (D-41 の反対側)。

    区間ごとに標準化し直す実装は、どの区間を切っても平均0・分散1になる。
    そちらは「予測が当たっているように見える」壊れ方 (仕様 §10-2) の温床なので、
    末尾区間が 0/1 からずれていることを積極的に確かめる。
    """
    cfg = _short_config(length=400, standardize_steps=100)
    task = generate_lorenz(cfg, np.random.default_rng(20240402))
    head = task.u[: cfg.standardize_steps]
    tail = task.u[cfg.standardize_steps :]
    assert np.allclose(np.mean(head, axis=0), 0.0, atol=1.0e-12)
    assert np.allclose(np.std(head, axis=0), 1.0, atol=1.0e-12)
    assert not np.allclose(np.mean(tail, axis=0), 0.0, atol=1.0e-3)


def test_standardizer_rejects_a_constant_component() -> None:
    """標準偏差 0 の成分は ``ValueError`` (黙って 0 除算しない)。"""
    series: FloatArray = np.stack(
        [np.arange(10.0), np.full(10, 3.0)],
        axis=1,  # 2列目が定数
    )
    with pytest.raises(ValueError, match="標準偏差が 0"):
        Standardizer.from_training_prefix(series, 10)


@pytest.mark.parametrize("n_steps", [1, 11])
def test_standardizer_rejects_an_out_of_range_prefix(n_steps: int) -> None:
    """推定に使う行数が 2 未満・系列長超のときは ``ValueError``。"""
    series: FloatArray = np.arange(20.0).reshape(10, 2)
    with pytest.raises(ValueError, match="n_steps"):
        Standardizer.from_training_prefix(series, n_steps)


def test_standardizer_round_trips() -> None:
    """``invert(apply(x)) == x`` (自走の予測を物理量へ戻す経路)。"""
    rng = np.random.default_rng(20240403)
    series: FloatArray = rng.standard_normal((50, 3)) * 4.0 + 7.0
    standardizer = Standardizer.from_training_prefix(series, 30)
    assert np.allclose(standardizer.invert(standardizer.apply(series)), series)


def test_lorenz_rejects_a_standardization_window_longer_than_the_series() -> None:
    """``standardize_steps > length`` は生成前に ``ValueError`` (D-41)。"""
    cfg = _short_config(length=100, standardize_steps=101)
    with pytest.raises(ValueError, match="standardize_steps"):
        generate_lorenz(cfg, np.random.default_rng(0))


def test_lorenz_generation_rejects_the_integration_step_axis() -> None:
    """確保軸1 (積分ステップ数) が**確保より前に**落ちる (D-34)。

    ``sample_interval`` だけを大きくして軸を超える。``length`` は本番の
    半分以下なので、軸2 (``length * 3``) はまったく余裕がある —— この設定で
    落ちることが「軸1 が独立に効いている」ことの実測である。
    """
    cfg = _short_config(length=3_000_000, sample_interval=10, integration_burn_in=0)
    assert cfg.length * LORENZ_STATE_DIM < chaotic._MAX_TRAJECTORY_ELEMENTS
    with pytest.raises(ValueError, match="積分ステップ数が上限"):
        integrate_lorenz(cfg, np.array(REFERENCE_X0), cfg.length)


def test_lorenz_generation_rejects_the_trajectory_element_axis() -> None:
    """確保軸2 (真の軌道の配列要素数) が**確保より前に**落ちる (D-34)。

    ``sample_interval=1`` にすると軸1 は通る (積分ステップ数 = length)。
    それでも ``length * 3`` が上限を超えるので落ちることが、
    「軸2 が軸1 と別の軸である」ことの実測である。**軸1 だけを塞いだ実装は
    このテストで落ちる** (3b-2 の「軸が2本あるのに1本だけ塞いだ」事故の再発防止)。
    """
    cfg = _short_config(length=18_000_000, sample_interval=1, integration_burn_in=0)
    n_integration_steps = (cfg.length + cfg.horizon) * cfg.sample_interval
    assert n_integration_steps <= chaotic._MAX_INTEGRATION_STEPS
    with pytest.raises(ValueError, match="真の軌道の配列要素数が上限"):
        integrate_lorenz(cfg, np.array(REFERENCE_X0), cfg.length)


def test_allocation_bounds_cannot_be_raised_from_the_config() -> None:
    """上限は上書き不能なモジュール定数である (設定フィールドにしない、D-34)。"""
    field_names = {item.name for item in dataclasses.fields(LorenzConfig)}
    assert not [name for name in field_names if name.startswith("max_")], (
        f"確保上限が設定フィールドになっています: {sorted(field_names)}"
    )


def test_generate_lorenz_produces_a_horizon_one_prediction_task() -> None:
    """``u[t] = x[t]`` / ``y[t] = x[t + horizon]`` の形と名前・params。"""
    cfg = _short_config(length=120, standardize_steps=60)
    task = generate_lorenz(cfg, np.random.default_rng(20240404))
    assert task.name == TASK_NAME_LORENZ
    assert task.u.shape == (cfg.length, LORENZ_STATE_DIM)
    assert task.y.shape == (cfg.length, LORENZ_STATE_DIM)
    assert task.params["sigma"] == "10.0"
    assert task.params["dt"] == str(sampling_interval(cfg))
    assert task.params["standardize_steps"] == str(cfg.standardize_steps)


def test_mackey_glass_adapter_delegates_instead_of_reimplementing() -> None:
    """04 の MG は 01 の生成器の出力を標準化しただけである (D-41)。

    同じシードで ``generate_mackey_glass`` を呼び、その ``u`` から推定した係数を
    自前で当てた結果と**厳密に一致**することを見る。adapter が独自に積分し直して
    いれば (= 再実装していれば) ここで落ちる。
    """
    from rc_basics_lab.tasks.mackey_glass import generate_mackey_glass

    cfg = MackeyGlassConfig(length=300, integration_burn_in=50)
    steps = 100
    seed = 20240405
    original = generate_mackey_glass(cfg, np.random.default_rng(seed))
    adapted = generate_standardized_mackey_glass(
        cfg, np.random.default_rng(seed), standardize_steps=steps
    )
    standardizer = Standardizer.from_training_prefix(original.u, steps)
    assert np.array_equal(adapted.u, standardizer.apply(original.u))
    assert np.array_equal(adapted.y, standardizer.apply(original.y))
    assert adapted.name == TASK_NAME_MACKEY_GLASS == original.name
    # 生成パラメータは 01 側が単一の真実 (adapter が足すのは標準化だけ)。
    assert dict(original.params).items() <= dict(adapted.params).items()
    assert adapted.params["standardize_steps"] == str(steps)


def test_initial_state_depends_on_the_task_stream() -> None:
    """初期状態は task ストリームの Generator から引く (D-06)。"""
    first = initial_state(np.random.default_rng(1))
    second = initial_state(np.random.default_rng(2))
    assert first.shape == (LORENZ_STATE_DIM,)
    assert not np.allclose(first, second)

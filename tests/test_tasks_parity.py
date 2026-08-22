"""遅延パリティ課題のテスト (受け入れ条件2 の機械的担保).

このファイルには本サイクルの主張そのものが2つ入っている:

1. ``test_target_is_orthogonal_to_lagged_inputs`` — 目標が入力のラグ空間に
   直交すること (線形手法の失敗が偶然ではなく解析的な帰結であることの数値確認)
2. ``test_linear_baselines_fail_and_esn_solves_delay_parity`` — 縮小設定で
   実際に線形・遅延線が失敗し ESN が解けること

閾値は仕様 §3 / T4 受け入れ基準の値をそのまま使う。**通らないときに緩めては
いけない** (緩めた瞬間に「示せた」が意味を失う。仕様 §8 リスク1 に該当し、
止まって相談する)。
"""

from __future__ import annotations

import dataclasses
import statistics

import numpy as np
import pytest

from rc_basics_lab.config import (
    DelayParityConfig,
    ESNConfig,
    ExperimentConfig,
    RidgeConfig,
    SplitConfig,
)
from rc_basics_lab.experiment.runner import ResultRow, build_tasks, run_task
from rc_basics_lab.tasks.base import TaskGenerator
from rc_basics_lab.tasks.delay_parity import generate_delay_parity, lead_in
from rc_basics_lab.types import FloatArray

DEFAULT = DelayParityConfig(n_bits=2, delay=1, length=8000)

# 署名適合の確認は mypy が行う (make type)。
_GENERATOR: TaskGenerator[DelayParityConfig] = generate_delay_parity

MAX_LAG = 10
ORTHOGONALITY_TOLERANCE = 0.05


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_inputs_are_pm_one() -> None:
    data = generate_delay_parity(DEFAULT, _rng())
    assert data.u.shape == (DEFAULT.length, 1)
    assert set(np.unique(data.u).tolist()) == {-1.0, 1.0}
    assert set(np.unique(data.y).tolist()) == {-1.0, 1.0}
    assert data.name == "delay_parity"
    assert data.params == {"n_bits": "2", "delay": "1"}


def test_input_bits_are_balanced() -> None:
    """±1 が i.i.d. で偏っていない (目標の直交性の前提)。"""
    data = generate_delay_parity(DEFAULT, _rng(7))
    assert abs(float(data.u.mean())) < ORTHOGONALITY_TOLERANCE


def test_target_matches_product_of_lagged_bits() -> None:
    """y[t] = u[t-1] * u[t-2] が実際に成立している (先読み分より後の行)。"""
    data = generate_delay_parity(DEFAULT, _rng(1))
    lead = lead_in(DEFAULT)
    u = data.u[:, 0]
    expected = u[lead - 1 : -1] * u[lead - 2 : -2]
    assert np.array_equal(data.y[lead:, 0], expected)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_target_is_orthogonal_to_lagged_inputs(seed: int) -> None:
    """|corr(y, u[t-k])| < 0.05 (k=0..10)。解析的失敗保証の数値確認 (D-07)。"""
    data = generate_delay_parity(DEFAULT, _rng(seed))
    u: FloatArray = data.u[:, 0]
    y: FloatArray = data.y[:, 0]
    assert abs(float(y.mean())) < ORTHOGONALITY_TOLERANCE
    for lag in range(MAX_LAG + 1):
        lagged = u[MAX_LAG - lag : len(u) - lag]
        correlation = float(np.corrcoef(y[MAX_LAG:], lagged)[0, 1])
        assert abs(correlation) < ORTHOGONALITY_TOLERANCE, (
            f"lag={lag} で相関 {correlation:.4f} が閾値を超えました"
        )


def test_n_bits_and_delay_change_target() -> None:
    """n_bits・delay それぞれの変更で目標が変わる。"""
    base = generate_delay_parity(DEFAULT, _rng(2))
    more_bits = generate_delay_parity(dataclasses.replace(DEFAULT, n_bits=3), _rng(2))
    longer_delay = generate_delay_parity(dataclasses.replace(DEFAULT, delay=2), _rng(2))
    assert not np.array_equal(base.y, more_bits.y)
    assert not np.array_equal(base.y, longer_delay.y)
    assert more_bits.params["n_bits"] == "3"
    assert longer_delay.params["delay"] == "2"
    # 入力は同一シードなら不変。変わったのは目標だけであることを固定する。
    assert np.array_equal(base.u, more_bits.u)
    assert np.array_equal(base.u, longer_delay.u)


@pytest.mark.parametrize(
    ("field", "value"),
    [("n_bits", 0), ("delay", -1), ("length", 0)],
)
def test_invalid_parameters_raise(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        generate_delay_parity(dataclasses.replace(DEFAULT, **{field: value}), _rng())


# --- 受け入れ条件2: 線形は解けず ESN は解ける ---------------------------------

REDUCED_N_UNITS = 100
REDUCED_N_REPLICATES = 3
REDUCED_TRAIN_ROWS = 2000


def reduced_config() -> ExperimentConfig:
    """縮小設定 (N=100, 学習 2000 点, 3シード)。10秒以内に終わる規模。"""
    return ExperimentConfig(
        n_replicates=REDUCED_N_REPLICATES,
        split=SplitConfig(washout=100, max_start_offset=50),
        ridge=RidgeConfig(
            alpha_grid=(1e-6, 1e-4, 1e-2, 1.0, 100.0),
            n_lags_grid=(1, 2, 4, 8),
        ),
        # 4150 - 50 (offset) - 100 (t0) = 4000 行を 0.5:0.15:0.35 で分割
        delay_parity=dataclasses.replace(DEFAULT, length=4150),
        esn_delay_parity=ESNConfig(
            n_units=REDUCED_N_UNITS,
            spectral_radius=0.9,
            leak_rate=1.0,
            input_scale=1.0,
            density=0.1,
        ),
    )


def parity_rows(config: ExperimentConfig) -> list[ResultRow]:
    """遅延パリティ課題だけを回して結果行を返す。"""
    entries = [entry for entry in build_tasks(config) if entry.name == "delay_parity"]
    assert len(entries) == 1
    return run_task(config, entries[0])


def _mean_by_method(rows: list[ResultRow], attribute: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.method, []).append(float(getattr(row, attribute)))
    return {method: statistics.fmean(values) for method, values in grouped.items()}


def test_linear_baselines_fail_and_esn_solves_delay_parity() -> None:
    """縮小設定で線形・遅延線が失敗し ESN が解ける (受け入れ条件2)。

    判定はシード平均に対して行う。遅延線の per-seed の符号正解率は有限標本の
    揺らぎで 0.7 付近まで振れるが、平均では 0.5 付近に戻る。
    """
    config = reduced_config()
    rows = parity_rows(config)
    assert len(rows) == REDUCED_N_REPLICATES * 3
    assert {row.n_train for row in rows} == {REDUCED_TRAIN_ROWS}

    nrmse_mean = _mean_by_method(rows, "nrmse")
    sign_mean = _mean_by_method(rows, "sign_accuracy")
    detail = f"nrmse={nrmse_mean}, sign_accuracy={sign_mean}"

    assert nrmse_mean["linear"] >= 0.9, detail
    assert nrmse_mean["delay_line"] >= 0.9, detail
    assert nrmse_mean["esn"] <= 0.6, detail
    assert sign_mean["esn"] >= 0.85, detail
    # 線形手法の符号正解率は当てずっぽう (0.5) に留まる
    assert sign_mean["linear"] <= 0.60, detail
    assert sign_mean["delay_line"] <= 0.60, detail

"""設計行列のテスト (受け入れ条件1 の本体).

3ベースラインが「実装差ではなく設計行列の差」で切り替わることを、
**同一の呼び出し式をループで回す**形で確認する。
"""

from __future__ import annotations

import numpy as np
import pytest

from rc_basics_lab.readout.design import (
    BIAS_NAME,
    DelayLineSpec,
    FeatureSpec,
    PassthroughSpec,
    ReservoirSpec,
    bias_column_index,
    build_design_matrix,
)
from rc_basics_lab.types import FloatArray

N_STEPS = 40
N_INPUTS = 2
N_UNITS = 7


def _inputs(n_steps: int = N_STEPS, n_inputs: int = N_INPUTS) -> FloatArray:
    return np.random.default_rng(0).standard_normal((n_steps, n_inputs))


def _states(n_steps: int = N_STEPS, n_units: int = N_UNITS) -> FloatArray:
    return np.random.default_rng(1).standard_normal((n_steps, n_units))


def test_three_specs_share_one_api() -> None:
    """3手法が同じ ``build_design_matrix`` 呼び出しで処理される。"""
    n_lags = 3
    specs: tuple[FeatureSpec, ...] = (
        PassthroughSpec(),
        DelayLineSpec(n_lags=n_lags),
        ReservoirSpec(),
    )
    expected = (
        1 + N_INPUTS,
        1 + N_INPUTS * (n_lags + 1),
        1 + N_INPUTS + N_UNITS,
    )
    inputs = _inputs()
    states = _states()
    # 呼び出し式は1本。手法ごとの分岐はテスト側にも存在しない。
    designs = [build_design_matrix(spec, inputs, states) for spec in specs]
    assert tuple(design.phi.shape[1] for design in designs) == expected
    for design in designs:
        assert design.phi.shape[0] == N_STEPS
        assert len(design.feature_names) == design.phi.shape[1]
        assert design.feature_names[0] == BIAS_NAME
        assert bias_column_index(design.feature_names) == 0


def test_delay_line_first_valid_equals_n_lags() -> None:
    for n_lags in (0, 1, 5):
        design = build_design_matrix(DelayLineSpec(n_lags=n_lags), _inputs())
        assert design.first_valid == n_lags


def test_other_specs_have_first_valid_zero() -> None:
    assert build_design_matrix(PassthroughSpec(), _inputs()).first_valid == 0
    assert build_design_matrix(ReservoirSpec(), _inputs(), _states()).first_valid == 0


def test_n_lags_changes_column_count() -> None:
    """n_lags を変えると列数と first_valid が変わる (配線の確認)。"""
    small = build_design_matrix(DelayLineSpec(n_lags=1), _inputs())
    large = build_design_matrix(DelayLineSpec(n_lags=8), _inputs())
    assert large.phi.shape[1] - small.phi.shape[1] == N_INPUTS * 7
    assert (small.first_valid, large.first_valid) == (1, 8)
    assert small.feature_names != large.feature_names


def test_delay_line_columns_hold_lagged_inputs() -> None:
    """遅延線の各ブロックが実際に ``u[t-lag]`` になっている。"""
    inputs = _inputs()
    n_lags = 3
    design = build_design_matrix(DelayLineSpec(n_lags=n_lags), inputs)
    for lag in range(n_lags + 1):
        start = 1 + lag * N_INPUTS
        block = design.phi[design.first_valid :, start : start + N_INPUTS]
        expected = inputs[design.first_valid - lag : N_STEPS - lag]
        assert np.array_equal(block, expected)


def test_rows_before_first_valid_are_nan() -> None:
    """t0 の取り違えが静かに通らないよう、無効行は NaN で埋める。"""
    design = build_design_matrix(DelayLineSpec(n_lags=4), _inputs())
    assert np.isnan(design.phi[: design.first_valid]).any()
    assert np.all(np.isfinite(design.phi[design.first_valid :]))


def test_passthrough_columns_are_bias_and_input() -> None:
    inputs = _inputs()
    design = build_design_matrix(PassthroughSpec(), inputs)
    assert np.array_equal(design.phi[:, 0], np.ones(N_STEPS))
    assert np.array_equal(design.phi[:, 1:], inputs)


def test_reservoir_spec_columns() -> None:
    inputs = _inputs()
    states = _states()
    design = build_design_matrix(ReservoirSpec(), inputs, states)
    assert np.array_equal(design.phi[:, 1 : 1 + N_INPUTS], inputs)
    assert np.array_equal(design.phi[:, 1 + N_INPUTS :], states)


def test_include_input_false_drops_input_columns() -> None:
    design = build_design_matrix(
        ReservoirSpec(include_input=False), _inputs(), _states()
    )
    assert design.phi.shape[1] == 1 + N_UNITS


def test_bias_false_drops_bias_column() -> None:
    design = build_design_matrix(PassthroughSpec(bias=False), _inputs())
    assert design.phi.shape[1] == N_INPUTS
    assert bias_column_index(design.feature_names) is None


def test_reservoir_spec_without_states_raises() -> None:
    with pytest.raises(ValueError, match="states"):
        build_design_matrix(ReservoirSpec(), _inputs())


def test_state_row_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="行数"):
        build_design_matrix(ReservoirSpec(), _inputs(), _states(n_steps=N_STEPS - 1))


def test_one_dimensional_input_raises() -> None:
    with pytest.raises(ValueError, match="2次元"):
        build_design_matrix(PassthroughSpec(), np.zeros(N_STEPS))


def test_negative_n_lags_raises() -> None:
    with pytest.raises(ValueError, match="n_lags"):
        build_design_matrix(DelayLineSpec(n_lags=-1), _inputs())


def test_too_short_series_raises() -> None:
    with pytest.raises(ValueError, match="短すぎます"):
        build_design_matrix(DelayLineSpec(n_lags=10), _inputs(n_steps=10))


def test_design_matrix_rejects_the_allocation_axis_before_building_phi() -> None:
    """``phi`` (``n_steps * n_features``) の確保軸が**確保より前に**落ちる。

    実測 (reviewer-security): 04 (``experiments/04_chaotic_freerun``) の
    ``lorenz.length`` と ``base.ridge.n_lags_grid`` はどちらも単独では
    04 の他の確保軸 (積分ステップ数・真の軌道の要素数) を超えない値なのに、
    設計行列の列数 (``n_lags * D_in + 1``) と掛け合わさると上限を超える
    (``length=100_000, n_lags=1_000, D_in=3`` -> ``n_features=3_001``,
    ``n_elements=300_100_000 > 2e8``)。この検査が無いと ``phi`` の実体化
    (``np.full`` / ``np.concatenate``) まで到達し、``n_steps`` を巨大な
    ``u`` を作らずに検証できる (``u`` 自体は 2.4 MB しかない)。
    """
    n_steps = 100_000
    n_lags = 1_000
    n_inputs = 3
    u = np.zeros((n_steps, n_inputs))
    with pytest.raises(ValueError, match="設計行列の要素数が上限"):
        build_design_matrix(DelayLineSpec(n_lags=n_lags), u)


def test_design_matrix_allows_the_axis_at_the_production_scale() -> None:
    """本番規模 (04 の実 config と同じ桁) では確保軸の検査が誤発火しない。

    ``length=8_000, n_lags=16, D_in=3`` は ``n_features=49``,
    ``n_elements=392,000`` で上限 (2e8) の 500 分の1以下。検査を足したことで
    正常系の成果物が変わらないことの実測 (成果物のバイト不変性の根拠の1つ)。
    """
    n_steps = 8_000
    n_lags = 16
    n_inputs = 3
    u = np.zeros((n_steps, n_inputs))
    design = build_design_matrix(DelayLineSpec(n_lags=n_lags), u)
    assert design.phi.shape == (n_steps, 1 + n_inputs * (n_lags + 1))

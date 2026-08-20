"""自走軌道の評価 (純関数層) のテスト —— D-43 / D-45 / D-46 と確保軸4・7.

このファイルが守るのは「判定と指標が**引数と定数だけ**で決まる」ことである。
実験や図を経由せずに測れるので、閾値を動かした / 指標の向きを変えた / 打ち切りを
無かったことにした、という壊れ方が1件ずつ落ちる。

成果物 (``results/04_chaotic_freerun/``) に対する受け入れ条件の実測は
``tests/test_experiment_freerun.py`` と ``tests/test_experiment_stability.py``
にある。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rc_basics_lab.config import LorenzConfig
from rc_basics_lab.experiment import attractor
from rc_basics_lab.experiment.attractor import (
    AMPLITUDE_RATIO_MAX,
    COLLAPSE_STD_RATIO,
    MIN_RETURN_MAP_POINTS,
    MIN_STATS_STEPS,
    REGIME_ATTRACTOR,
    REGIME_DIVERGED,
    REGIME_PERIODIC,
    REGIMES,
    attractor_distance,
    autocorrelation_peak,
    classify_regime,
    first_autocorrelation_zero,
    lyapunov_normalized,
    normalized_error_curve,
    point_set_distance,
    power_spectrum,
    return_map_points,
    shuffled_surrogate,
    spectrum_distance,
    spectrum_length,
    successive_maxima,
    valid_time_from_errors,
    validate_stats_bounds,
)
from rc_basics_lab.tasks.chaotic import initial_state, integrate_lorenz
from rc_basics_lab.types import FloatArray


def lorenz_like(n_steps: int, seed: int = 0) -> FloatArray:
    """真の Lorenz 軌道 ``(n_steps, 3)`` (D-41 の生成をそのまま使う)。

    合成のカオス写像で代用しない —— D-46 の指標が測るのは「滑らかな連続時間の
    軌道が持つ時間構造」であり、離散写像ではスペクトルがほぼ白色になって
    シャッフル代替との差が出ない (実測: 合成写像だと代替とのスペクトル距離が
    0.44 にしかならず、真の Lorenz では 0.93)。
    """
    config = LorenzConfig(
        length=n_steps,
        integration_burn_in=200,
        standardize_steps=min(n_steps, 100),
    )
    series: FloatArray = integrate_lorenz(
        config, initial_state(np.random.default_rng(seed)), n_steps
    )
    return series


# --- 確保軸4 / 確保軸7 ---------------------------------------------------------


def test_validate_stats_bounds_rejects_the_stats_axis() -> None:
    """確保軸4 (``stats_steps``) が上書き不能な定数で塞がれている (D-34)。"""
    validate_stats_bounds(MIN_STATS_STEPS)
    with pytest.raises(ValueError, match="stats_steps は"):
        validate_stats_bounds(MIN_STATS_STEPS - 1)
    with pytest.raises(ValueError, match="上限を超えています"):
        validate_stats_bounds(attractor._MAX_STATS_STEPS + 1)


def test_spectrum_length_follows_the_sample_count() -> None:
    """確保軸7: FFT 長は標本数に**従属**し、独立の設定軸を持たない。"""
    assert spectrum_length(20_000) == 16_384
    assert spectrum_length(1024) == 1024
    assert spectrum_length(1023) == 512
    assert spectrum_length(1) == 0
    frequencies, power = power_spectrum(lorenz_like(2000), dt=0.5)
    assert frequencies.shape == power.shape == (1024 // 2 + 1,)
    assert float(np.sum(power)) == pytest.approx(1.0)


def test_no_public_bin_count_knob_exists() -> None:
    """確保軸7: ビン数 / 分位点数を名乗る公開シンボルが無い。

    「ビン数を独立の軸にしない」は実装の性質であって願望ではない。設定でも
    引数でもビンを渡せないことを、公開名の走査で機械的に固定する。
    """
    offenders = [
        name
        for name in attractor.__all__
        if "bin" in name.lower() or "quantile" in name.lower()
    ]
    assert not offenders, offenders


# --- D-43: 有効予測時間 --------------------------------------------------------


def test_normalized_error_curve_matches_the_nrmse_normalization() -> None:
    """誤差は NRMSE 比 (分母は真の系列の標準偏差、D-02 と同じ)。"""
    truth = lorenz_like(200)
    prediction = truth + 0.1
    curve = normalized_error_curve(truth, prediction)
    assert curve.shape == (200,)
    np.testing.assert_allclose(curve, 0.1 / float(np.std(truth)), rtol=1e-12)


def test_valid_time_stops_at_the_first_crossing() -> None:
    """閾値を初めて超えたステップで止まる (D-43)。"""
    errors = np.array([0.1, 0.2, 0.5, 0.1], dtype=np.float64)
    result = valid_time_from_errors(errors, 0.4)
    assert (result.steps, result.censored) == (2, False)


def test_valid_time_marks_the_run_length_as_censored() -> None:
    """最後まで閾値を超えなければ**打ち切りフラグを立てる** (D-43)。

    打ち切りを無かったことにすると、分布の右端が実際より小さく見える。
    """
    errors = np.full(50, 0.05, dtype=np.float64)
    result = valid_time_from_errors(errors, 0.4)
    assert (result.steps, result.censored) == (50, True)


def test_nan_errors_end_the_valid_time() -> None:
    """打ち切り後の ``nan`` は「閾値を超えた」扱い (自走の破綻を伸ばさない)。"""
    errors = np.array([0.1, 0.1, math.nan, 0.1], dtype=np.float64)
    assert valid_time_from_errors(errors, 0.4).steps == 2


def test_valid_time_threshold_changes_the_result() -> None:
    """閾値を変えると有効予測時間が動く (D-43 の配線の実体)。"""
    errors = np.linspace(0.0, 1.0, 101)
    steps = [valid_time_from_errors(errors, t).steps for t in (0.2, 0.3, 0.4, 0.5)]
    assert steps == sorted(steps) and len(set(steps)) == 4


def test_lyapunov_normalization_divides_by_the_lyapunov_time() -> None:
    """生のステップ数を Lyapunov 時間の単位へ直す (仕様 §5 禁止する構造5)。"""
    assert lyapunov_normalized(200, 0.01, 2.0) == pytest.approx(1.0)


def test_lyapunov_normalization_refuses_a_non_positive_lyapunov_time() -> None:
    """負・0・非有限の Lyapunov 時間で割らない (D-42 の nan を素通ししない)。"""
    for lyapunov_time in (0.0, -1.0, math.nan, math.inf):
        assert math.isnan(lyapunov_normalized(100, 0.01, lyapunov_time))


# --- D-45: 3態分類 ------------------------------------------------------------


def test_classify_regime_catches_a_finite_but_blown_up_trajectory() -> None:
    """**振幅を見る**: ``isfinite`` では捕まらない破綻を発散に分類する (D-45)。

    ``float64`` の範囲内で 1e200 まで伸びる軌道は有限値のままなので、
    ``diverged`` フラグ (非有限で打ち切り) だけを見る分類器はこれを
    「アトラクタ再現」に数える (T4 実装メモ 5 の実測)。
    """
    reference = lorenz_like(2000)
    blown = np.geomspace(1.0, 1.0e200, 500)[:, None] * np.ones((1, 3))
    assert bool(np.all(np.isfinite(blown))), "この検査は有限値の軌道で行う"
    verdict = classify_regime(blown, reference=reference, diverged=False)
    assert verdict.regime == REGIME_DIVERGED
    assert verdict.amplitude_ratio > AMPLITUDE_RATIO_MAX


def test_classify_regime_uses_the_diverged_flag_too() -> None:
    """非有限で打ち切られた自走は、振幅を見るまでもなく発散。"""
    reference = lorenz_like(500)
    verdict = classify_regime(reference[:100], reference=reference, diverged=True)
    assert verdict.regime == REGIME_DIVERGED


def test_classify_regime_calls_a_fixed_point_periodic() -> None:
    """1点へ潰れた軌道は周期軌道 (不動点は周期1)。"""
    reference = lorenz_like(2000)
    collapsed = np.full((1000, 3), 0.3, dtype=np.float64)
    verdict = classify_regime(collapsed, reference=reference, diverged=False)
    assert verdict.regime == REGIME_PERIODIC
    assert verdict.std_ratio < COLLAPSE_STD_RATIO


def test_classify_regime_calls_a_sine_periodic() -> None:
    """振幅は真値と同程度でも、周期的なら周期軌道に分類する。"""
    reference = lorenz_like(4000)
    phase = np.arange(4000) * 0.05
    scale = float(np.std(reference))
    periodic: FloatArray = scale * np.stack(
        [np.sin(phase), np.cos(phase), np.sin(2.0 * phase)], axis=1
    )
    verdict = classify_regime(periodic, reference=reference, diverged=False)
    assert verdict.regime == REGIME_PERIODIC
    assert verdict.autocorr_peak >= 0.95


def test_classify_regime_calls_a_bounded_chaotic_trajectory_attractor() -> None:
    """有界かつ非周期なら ``REGIME_ATTRACTOR`` (真の軌道自身も含む)。"""
    reference = lorenz_like(4000)
    verdict = classify_regime(reference, reference=reference, diverged=False)
    assert verdict.regime == REGIME_ATTRACTOR
    assert verdict.autocorr_peak < 0.95


def test_classify_regime_is_exclusive_and_exhaustive() -> None:
    """どんな入力でも ``REGIMES`` のちょうど1つを返す (排他かつ網羅)。"""
    reference = lorenz_like(2000)
    candidates: tuple[FloatArray, ...] = (
        np.empty((0, 3), dtype=np.float64),
        np.zeros((10, 3), dtype=np.float64),
        reference,
        reference * 1000.0,
        np.full((500, 3), 1.0e-30, dtype=np.float64),
    )
    verdicts = [
        classify_regime(candidate, reference=reference, diverged=False)
        for candidate in candidates
    ]
    assert all(verdict.regime in REGIMES for verdict in verdicts)
    assert len({verdict.regime for verdict in verdicts}) >= 2


def test_autocorrelation_peak_ignores_the_smooth_short_lags() -> None:
    """ラグ1 の高い自己相関 (滑らかにサンプルしただけ) を周期性と読まない。"""
    smooth = np.cumsum(np.random.default_rng(0).normal(0.0, 1.0, 4000))[:, None]
    assert autocorrelation_peak(smooth) < 1.0
    assert first_autocorrelation_zero(smooth) >= 1


# --- D-46: 長時間統計 ----------------------------------------------------------


def test_successive_maxima_finds_the_return_map_coordinate() -> None:
    """極大値の抽出 (リターンマップの座標)。"""
    series = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0], dtype=np.float64)[:, None]
    np.testing.assert_allclose(successive_maxima(series), [1.0, 2.0, 3.0])
    assert return_map_points(series).shape == (2, 2)


def test_point_set_distance_is_zero_for_identical_sets() -> None:
    """同じ点集合の距離は 0、対称である。"""
    points = return_map_points(lorenz_like(4000))
    assert points.shape[0] >= MIN_RETURN_MAP_POINTS
    assert point_set_distance(points, points) == pytest.approx(0.0)
    other = points + 0.5
    assert point_set_distance(points, other) == pytest.approx(
        point_set_distance(other, points)
    )


def test_point_set_distance_is_nan_when_there_are_too_few_points() -> None:
    """点が足りなければ ``nan`` (0 を返すと潰れた軌道ほど近くなる)。"""
    few = np.zeros((MIN_RETURN_MAP_POINTS - 1, 2), dtype=np.float64)
    assert math.isnan(point_set_distance(few, few))


def test_shuffled_surrogate_keeps_the_marginal_and_destroys_the_time_structure() -> (
    None
):
    """シャッフル代替は**分布を変えず時間構造だけ**を壊す (D-46 の対照の要件)。

    分布だけを見る指標では代替と真の軌道を区別できない。この検査があることで、
    指標を「値の分布距離」に差し替えた実装が緑にならない。
    """
    truth = lorenz_like(6000)
    rng = np.random.default_rng(3)
    surrogate = shuffled_surrogate(truth, rng, truth.shape[0])
    np.testing.assert_allclose(np.sort(surrogate, axis=0), np.sort(truth, axis=0))
    distance = attractor_distance(truth, surrogate, dt=1.0)
    self_distance = attractor_distance(truth, truth, dt=1.0)
    assert self_distance.return_map == pytest.approx(0.0)
    assert self_distance.spectrum == pytest.approx(0.0)
    assert distance.return_map > 10.0 * max(self_distance.return_map, 1.0e-6)
    assert distance.spectrum > 0.5


def test_spectrum_distance_refuses_mismatched_lengths() -> None:
    """別の FFT 長で測ったスペクトルは比べない。"""
    with pytest.raises(ValueError, match="スペクトルの長さ"):
        spectrum_distance(np.zeros(4), np.zeros(5))


def test_attractor_distance_reports_the_sample_counts() -> None:
    """距離と一緒に「何点で測ったか」を返す (成果物の列になる)。"""
    truth = lorenz_like(4000)
    distance = attractor_distance(truth, truth[:2000], dt=1.0)
    assert distance.n_return_map_points > 0
    assert distance.n_spectrum_bins == spectrum_length(2000) // 2 + 1

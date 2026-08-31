"""記事04の図5枚のテスト (D-10 / 仕様 §5 禁止する構造7 / 確保軸6).

守るのは3つ。

1. **図は成果物 CSV の行だけを読む**。実験も診断も走らせない —— 図を描く経路で
   実験が動くと、「図を出すたびに結果が変わる」成果物になる。
2. **5枚が1コマンドで出る** (受け入れ条件6)。ここでは関数単位で、
   ``tests/test_experiment_freerun.py`` が成果物単位で確かめる。
3. **ラベルは ``style.label`` を通る** (D-10)。対応表に無い課題名・3態は
   描く前に ``ValueError`` にする (図から静かに消えない)。

行は合成する (実験を回さない)。図の中身の正しさではなく、**行から図までの経路**
を測るテストである。
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from rc_basics_lab.experiment.attractor import (
    REGIME_ATTRACTOR,
    REGIME_DIVERGED,
    REGIMES,
)
from rc_basics_lab.experiment.capacity_rows import (
    CapacityRow,
)
from rc_basics_lab.experiment.freerun import (
    EXPERIMENT_FREERUN,
    KIND_PHASE,
    KIND_RETURN_MAP,
    KIND_SPECTRUM,
    SOURCE_FREERUN,
    SOURCE_TRUTH,
    FreeRunProfileRow,
    FreeRunRow,
)
from rc_basics_lab.experiment.runner import DELAY_LINE, ESN_METHOD, LINEAR, ResultRow
from rc_basics_lab.experiment.stability import EXPERIMENT_STABILITY, StabilityRow
from rc_basics_lab.plotting.figures_freerun import (
    REGIME_LABELS,
    plot_freerun_attractor,
    plot_freerun_stats,
    plot_valid_time,
    profile_points,
)
from rc_basics_lab.plotting.figures_stability import plot_stability_map
from rc_basics_lab.plotting.style import StyleContext

TASKS = ("lorenz", "mackey_glass")
STYLE = StyleContext(cjk_font=None)
"""英語ラベルで描く (CJK フォントの有無で図の生成可否が変わらないこと)。"""


def onestep_rows() -> list[ResultRow]:
    return [
        ResultRow(
            task=task,
            method=method,
            replicate=replicate,
            seed_reservoir=0,
            seed_task=1,
            seed_split=2,
            alpha=1.0e-6,
            n_lags=2,
            rmse=0.1,
            nrmse=0.1 * (index + 1),
            nmse=0.01,
            sign_accuracy=0.9,
            n_train=100,
            n_val=30,
            n_test=70,
            t0=10,
            wall_time_s=0.01,
        )
        for task in TASKS
        for index, method in enumerate((LINEAR, DELAY_LINE, ESN_METHOD))
        for replicate in range(3)
    ]


def freerun_rows(lyapunov_time: float = 1.1) -> list[FreeRunRow]:
    return [
        FreeRunRow(
            experiment=EXPERIMENT_FREERUN,
            task=task,
            method=method,
            replicate=replicate,
            seed_reservoir=0,
            seed_task=1,
            seed_split=2,
            n_units=20,
            rho=0.9,
            leak_rate=0.3,
            state_noise=0.0,
            alpha=1.0e-6,
            val_nrmse=0.1,
            switch_index=100,
            warmup_steps=10,
            free_run_steps=40,
            stats_steps=80,
            dt=0.01,
            lyapunov_per_time=1.0 / lyapunov_time,
            lyapunov_time=lyapunov_time if task == TASKS[0] else math.nan,
            valid_time_threshold=0.4,
            valid_time_steps=10 * (index + 1),
            valid_time=0.1,
            valid_time_lyapunov=(0.5 * (index + 1) if task == TASKS[0] else math.nan),
            valid_time_censored=index == 2 and replicate == 0,
            diverged=False,
            n_completed=80,
            regime=REGIME_ATTRACTOR,
            amplitude_ratio=1.0,
            std_ratio=1.0,
            autocorr_peak=0.2,
            return_map_distance=0.02,
            return_map_distance_surrogate=0.4,
            spectrum_distance=0.3,
            spectrum_distance_surrogate=0.9,
            closer_than_surrogate=True,
            n_stats_samples=80,
            n_return_map_points=12,
            n_spectrum_bins=33,
            wall_time_s=0.01,
        )
        for task in TASKS
        for index, method in enumerate((LINEAR, DELAY_LINE, ESN_METHOD))
        for replicate in range(3)
    ]


def profile_rows() -> list[FreeRunProfileRow]:
    rows: list[FreeRunProfileRow] = []
    for task in TASKS:
        for kind in (KIND_PHASE, KIND_RETURN_MAP, KIND_SPECTRUM):
            for source in (SOURCE_TRUTH, SOURCE_FREERUN):
                rows.extend(
                    FreeRunProfileRow(
                        experiment=EXPERIMENT_FREERUN,
                        task=task,
                        method=ESN_METHOD,
                        replicate=0,
                        kind=kind,
                        source=source,
                        index=index,
                        x=float(index),
                        y=math.sin(0.1 * index),
                    )
                    for index in range(30)
                )
    return rows


def stability_rows() -> list[StabilityRow]:
    return [
        StabilityRow(
            experiment=EXPERIMENT_STABILITY,
            rho=rho,
            leak_rate=leak,
            state_noise=noise,
            replicate=replicate,
            n_units=20,
            alpha=1.0e-6,
            val_nrmse=0.1,
            regime=REGIME_DIVERGED if leak < 0.2 and noise == 0.0 else REGIME_ATTRACTOR,
            amplitude_ratio=1.0,
            std_ratio=1.0,
            autocorr_peak=0.1,
            diverged=False,
            n_completed=80,
            stats_steps=80,
            valid_time_threshold=0.4,
            valid_time_steps=10,
            valid_time_lyapunov=0.5 + rho,
            valid_time_censored=False,
            wall_time_s=0.01,
        )
        for rho in (0.7, 1.1)
        for leak in (0.1, 0.6)
        for noise in (0.0, 1.0e-2)
        for replicate in range(2)
    ]


def capacity_rows() -> list[CapacityRow]:
    return [
        CapacityRow(
            experiment="4D_freerun_capacity",
            replicate=row.replicate,
            seed_reservoir=0,
            seed_drive=1,
            seed_surrogate=4,
            rho=row.rho,
            leak_rate=row.leak_rate,
            input_scale=0.5,
            sigma_u=1.0,
            input_drive_std=1.0,
            n_units=row.n_units,
            density=0.1,
            state_noise=row.state_noise,
            n_steps=600,
            washout=30,
            t0_mc=30,
            n_samples_mc=500,
            mc_total=100.0 + 10.0 * row.rho,
            mc_total_raw=110.0,
            mc_threshold=0.01,
            mc_effective_delay=3.0,
            mc_ratio=0.5,
            n_delays=12,
            t0_ipc=30,
            n_samples_ipc=500,
            ipc_total=200.0,
            ipc_total_raw=210.0,
            ipc_linear=120.0,
            ipc_nonlinear=80.0,
            ipc_saturation_ratio=0.5,
            n_targets=100,
            n_targets_kept=50,
            n_degrees=2,
            chunk_size_mc_effective=64,
            chunk_size_ipc_effective=64,
            wall_time_state_s=0.01,
            wall_time_mc_s=0.01,
            wall_time_ipc_s=0.01,
            wall_time_s=0.03,
        )
        for row in stability_rows()
    ]


def test_all_four_figures_are_written(tmp_path: Path) -> None:
    """4枚が行から描ける (受け入れ条件6 の関数単位の確認)。

    かつては5枚だった。4-A を単独図にするのをやめ、位相図と同じ figure の
    パネルへ移したため1枚減った (FIG-12)。
    """
    paths = (
        # FIG-12: 4-A は単独図をやめ、位相図と同じ figure のパネルになった。
        plot_freerun_attractor(
            profile_rows(),
            tmp_path / "fig_freerun_attractor.png",
            onestep_rows=onestep_rows(),
            style=STYLE,
        ),
        plot_valid_time(freerun_rows(), tmp_path / "fig_valid_time.png", style=STYLE),
        plot_stability_map(
            stability_rows(),
            capacity_rows(),
            tmp_path / "fig_stability_map.png",
            style=STYLE,
        ),
        plot_freerun_stats(
            profile_rows(),
            freerun_rows(),
            tmp_path / "fig_freerun_stats.png",
            style=STYLE,
        ),
    )
    assert len({path.name for path in paths}) == 4
    for path in paths:
        assert path.stat().st_size > 0


def test_figures_never_run_an_experiment_or_a_diagnostic(tmp_path: Path) -> None:
    """**図が診断・実験を走らせない** (仕様 §5 禁止する構造7)。

    実験と診断の入口を「呼ばれたら落ちる」ものに差し替えたまま5枚を描く。
    """
    import rc_basics_lab.diagnostics.ipc as ipc_module
    import rc_basics_lab.diagnostics.memory_capacity as mc_module
    import rc_basics_lab.experiment.freerun as freerun_module
    import rc_basics_lab.experiment.stability as stability_module

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("図が実験・診断を走らせました")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(freerun_module, "run_freerun_experiment", forbidden)
        patch.setattr(freerun_module, "run_free_run", forbidden)
        patch.setattr(stability_module, "run_stability_experiment", forbidden)
        patch.setattr(stability_module, "measure_capacity", forbidden)
        patch.setattr(mc_module, "memory_capacity", forbidden)
        patch.setattr(ipc_module, "ipc", forbidden)
        test_all_four_figures_are_written(tmp_path)


def test_profile_points_restores_the_drawing_order() -> None:
    """長形式の行から ``index`` の昇順で点列を復元する。"""
    rows = list(reversed(profile_rows()))
    points = profile_points(rows, TASKS[0], KIND_PHASE, SOURCE_TRUTH)
    assert points.shape == (30, 2)
    assert list(points[:, 0]) == sorted(points[:, 0])
    assert profile_points(rows, "unknown", KIND_PHASE, SOURCE_TRUTH).shape == (0, 2)


def test_unknown_task_label_fails_before_drawing(tmp_path: Path) -> None:
    """対応表に無い課題名は ``ValueError`` (図から静かに消えない、D-10)。"""
    rows = [
        FreeRunProfileRow(
            experiment=EXPERIMENT_FREERUN,
            task="unknown_system",
            method=ESN_METHOD,
            replicate=0,
            kind=KIND_PHASE,
            source=SOURCE_TRUTH,
            index=index,
            x=float(index),
            y=float(index),
        )
        for index in range(5)
    ]
    with pytest.raises(ValueError, match="ラベルの対応表"):
        plot_freerun_attractor(
            rows, tmp_path / "fig.png", onestep_rows=onestep_rows(), style=STYLE
        )


def test_regime_labels_cover_every_regime() -> None:
    """3態の表示が ``REGIMES`` と過不足なく一致する。"""
    assert set(REGIME_LABELS) == set(REGIMES)


def test_valid_time_figure_needs_a_lyapunov_estimate(tmp_path: Path) -> None:
    """Lyapunov 正規化できる行が1つも無ければ描かずに落ちる (D-42 / D-43)。

    縦軸が Lyapunov 時間の図に ``nan`` の系を並べると、空のパネルが
    「有効予測時間が 0」に見える。
    """
    rows = [
        dataclasses.replace(row, lyapunov_time=math.nan, valid_time_lyapunov=math.nan)
        for row in freerun_rows()
    ]
    with pytest.raises(ValueError, match="lyapunov_time"):
        plot_valid_time(rows, tmp_path / "fig.png", style=STYLE)

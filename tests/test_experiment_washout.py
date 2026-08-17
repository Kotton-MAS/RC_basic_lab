"""実験 2-D (washout 感度) のテスト —— 交絡の除去 (D-19) が本題.

このモジュールが守るのは1点に尽きる: **``washout_sensitivity.csv`` に出る
変動が「washout の効果」であって「訓練データ量の効果」ではないこと**。

``make_split`` は ``n_usable = n_steps - max_start_offset - t0`` で行数を決め、
``t0 = max(washout, 各手法の first_valid)`` なので、washout を素直に振ると
訓練データ量が同時に減る。しかもこの交絡は**滑らかな単調曲線として出る**ため、
図を見ても気づけない (実測: 補償なしだと MG x ESN の NRMSE が washout に対して
単調増加し、いかにも「washout を長く取りすぎると悪化する」という読み方ができる
曲線になる。補償を入れるとその単調性は消える)。

``test_washout_sweep_holds_training_size_constant`` が D-19 の guard であり、
補償ありで行数が一致すること・補償なしでは実際に縮むことの**両方**を実測する。
片側だけだと「補償が効いた」のか「そもそも縮まない設定だった」のかが分からない。
"""

from __future__ import annotations

import dataclasses
import json
import math
import statistics
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import pytest
from conftest import png_dpi

from rc_basics_lab.config import (
    DelayParityConfig,
    ESNConfig,
    Esp02Config,
    EspConfig,
    EspDecayConfig,
    EspMapConfig,
    ExperimentConfig,
    MackeyGlassConfig,
    RidgeConfig,
    SplitConfig,
    TimescaleConfig,
    TimescaleSweepConfig,
    WashoutSweepConfig,
    load_config_as,
)
from rc_basics_lab.config import DriveConfig as EspDriveConfig
from rc_basics_lab.config import ReservoirSweepConfig as EspReservoirConfig
from rc_basics_lab.experiment.esp_pipeline import run_and_report_esp
from rc_basics_lab.experiment.runner import (
    ESN_METHOD,
    TaskEntry,
    build_tasks,
    plan_replicate,
)
from rc_basics_lab.experiment.washout import (
    HEADLINE_METHOD,
    HEADLINE_TASK,
    WashoutRow,
    mean_nrmse_by_washout,
    predicted_t0,
    run_washout_sweep,
    summarize_washout_sensitivity,
    variant_for,
)
from rc_basics_lab.plotting.figures_esp import plot_washout_sensitivity
from rc_basics_lab.plotting.style import setup_style
from rc_basics_lab.tasks.delay_parity import TASK_NAME as DELAY_PARITY

REPO_ROOT = Path(__file__).resolve().parents[1]
ESP_CONFIG_PATH = REPO_ROOT / "experiments" / "02_esp_and_dynamics" / "config.yaml"

REFERENCE_WASHOUT = 200
"""01 の本番値。``test_washout_zero_is_worst_or_equal_for_mackey_glass`` の比較先。"""

RETINA_DPI = 200


def tiny_base(*, washout: int = 40) -> ExperimentConfig:
    """秒未満で1周できる 01 用の縮小設定 (掃引の土台)。"""
    return ExperimentConfig(
        name="washout-test",
        n_replicates=2,
        split=SplitConfig(washout=washout, max_start_offset=40),
        ridge=RidgeConfig(alpha_grid=(1.0e-4, 1.0), n_lags_grid=(1, 4)),
        mackey_glass=MackeyGlassConfig(length=500),
        delay_parity=DelayParityConfig(length=500),
        esn_mackey_glass=ESNConfig(n_units=30, density=0.3),
        esn_delay_parity=ESNConfig(
            n_units=30, density=0.3, leak_rate=1.0, input_scale=1.0
        ),
    )


def tiny_sweep_config(
    *, grid: tuple[int, ...] = (0, 40, 120), pad_series: bool = True
) -> Esp02Config:
    """2-D だけを回すための設定 (``run_washout_sweep`` は washout 節しか読まない)。"""
    return Esp02Config(
        washout=WashoutSweepConfig(
            grid=grid, pad_series=pad_series, base=tiny_base(washout=REFERENCE_WASHOUT)
        )
    )


def tiny_pipeline_config() -> Esp02Config:
    """2-A/2-B/2-C/2-D をすべて縮小した設定 (``meta.json`` の検査用)。"""
    return dataclasses.replace(
        tiny_sweep_config(),
        name="washout-pipeline-test",
        drive=EspDriveConfig(n_steps=300, washout=40, n_pairs=2),
        reservoir=EspReservoirConfig(n_units=15, density=0.3, n_replicates=1),
        decay=EspDecayConfig(rho_grid=(0.6, 1.3)),
        timescale_sweep=TimescaleSweepConfig(leak_rate_grid=(0.3, 1.0)),
        esp_map=EspMapConfig(rho_grid=(0.8, 1.4), sigma_grid=(0.0, 1.0)),
        esp=EspConfig(window=100, fit_skip=5),
        timescale=TimescaleConfig(max_lag=30),
    )


@lru_cache(maxsize=1)
def production_config() -> Esp02Config:
    """本番設定 (``experiments/02_esp_and_dynamics/config.yaml``)。"""
    return load_config_as(ESP_CONFIG_PATH, Esp02Config)


@lru_cache(maxsize=1)
def production_rows() -> tuple[WashoutRow, ...]:
    """本番格子での 2-D の行 (実測 4.4 秒。テスト間で1回だけ回す)。

    記事の主張 (「washout をどう取るかで性能がどれだけ動くか」) は本番格子の
    数値で語るので、縮小設定で代用しない。
    """
    return run_washout_sweep(production_config())


def _sizes_by_washout(
    rows: Sequence[WashoutRow],
) -> dict[int, tuple[int, int, int]]:
    return {row.washout: (row.n_train, row.n_val, row.n_test) for row in rows}


def _t0_by_washout(rows: Sequence[WashoutRow]) -> dict[int, int]:
    return {row.washout: row.t0 for row in rows}


def _headline_entry(config: ExperimentConfig) -> TaskEntry:
    """主役の課題 (Mackey-Glass) の ``TaskEntry``。"""
    return next(item for item in build_tasks(config) if item.name == HEADLINE_TASK)


# --- D-19 guard ------------------------------------------------------------


def test_washout_sweep_holds_training_size_constant() -> None:
    """**D-19 guard**: 補償ありで行数一定・補償なしで ``n_train`` が縮む。

    両方向を1つのテストで見るのは、片側だけだと「補償が効いた」のか
    「そもそも縮まない設定だった」のかを区別できないため。

    補償なし側は**狭義単調減少を要求しない**。``t0 = max(washout, first_valid)``
    なので、washout が遅延線の最大ラグ以下の格子点は同じ ``t0`` になり行数も
    等しくなる (本番格子では washout=0 と 50 がどちらも ``t0=64``)。要求するのは
    「非増加」かつ「``t0`` が増えた区間では実際に減る」こと。
    """
    grid = (0, 40, 120, 240)
    padded = run_washout_sweep(tiny_sweep_config(grid=grid, pad_series=True))
    unpadded = run_washout_sweep(tiny_sweep_config(grid=grid, pad_series=False))

    padded_sizes = _sizes_by_washout(padded)
    assert set(padded_sizes) == set(grid)
    assert len(set(padded_sizes.values())) == 1, (
        f"補償ありなのに行数が格子で揺れています: {padded_sizes}"
    )

    # 補償が「何もしなくても一定だった」わけではないこと (t0 は実際に動く)
    padded_t0 = _t0_by_washout(padded)
    assert len(set(padded_t0.values())) > 1, padded_t0

    unpadded_sizes = _sizes_by_washout(unpadded)
    unpadded_t0 = _t0_by_washout(unpadded)
    trains = [unpadded_sizes[washout][0] for washout in grid]
    assert trains == sorted(trains, reverse=True), (
        f"補償なしで n_train が非増加になっていません: {unpadded_sizes}"
    )
    assert trains[0] > trains[-1], (
        f"補償なしでも n_train が縮んでいません (交絡を再現できていない): {trains}"
    )
    for left, right in zip(grid, grid[1:], strict=True):
        if unpadded_t0[right] > unpadded_t0[left]:
            assert unpadded_sizes[right][0] < unpadded_sizes[left][0], (
                f"t0 が増えたのに n_train が減っていません: {unpadded_sizes}"
            )

    # 補償ありの行数は、格子最小値での補償なしの行数と一致する
    # (伸ばす側にだけ働き、01 の本番設定より短い系列では測らない)
    assert padded_sizes[min(grid)] == unpadded_sizes[min(grid)]


def test_padding_uses_the_same_t0_as_the_runner() -> None:
    """系列を作らずに求めた ``t0`` が、実際に実験を回した ``t0`` と一致する。

    補償量は ``predicted_t0`` (``first_valid_for`` 経由) で決まる。ここが実際の
    ``compute_t0`` とずれると補償が静かに外れ、``training_size_is_constant`` も
    偽になるが、**その前にこのテストが原因を名指しで落とす**。
    """
    config = tiny_sweep_config(grid=(0, 40, 120))
    rows = run_washout_sweep(config)
    actual = _t0_by_washout(rows)
    for washout, t0 in actual.items():
        assert predicted_t0(config.washout.base, washout) == t0


def test_variant_only_changes_the_washout_and_the_series_length() -> None:
    """1格子点ぶんの設定で変わるのは washout と両課題の ``length`` だけ。

    ``run_experiment`` を再利用することで公平性 (D-04/D-05/D-08) が担保される
    のは、**土台の設定を他に触らない**ことが前提である。
    """
    section = tiny_sweep_config(grid=(0, 40, 120)).washout
    base = section.base
    variant = variant_for(section, 120)
    assert variant.split.washout == 120
    assert variant.mackey_glass.length > base.mackey_glass.length
    assert variant.delay_parity.length > base.delay_parity.length
    assert (
        variant.mackey_glass.length - base.mackey_glass.length
        == variant.delay_parity.length - base.delay_parity.length
    )
    # length と washout を戻したら元の設定と完全一致する
    restored = dataclasses.replace(
        variant,
        split=dataclasses.replace(variant.split, washout=base.split.washout),
        mackey_glass=base.mackey_glass,
        delay_parity=base.delay_parity,
    )
    assert restored == base


def test_padding_does_not_disturb_the_rows_that_are_actually_used() -> None:
    """系列を伸ばしても、使う行の中身は伸ばす前と同じである。

    補償は系列の**末尾**を伸ばすだけなので、既存の行は1つも書き換わらない。
    ここが崩れると「行数は同じだが中身が別物」になり、格子点間の比較が
    washout の効果ではなくなる。
    """
    section = tiny_sweep_config(grid=(0, 40, 120)).washout
    base = section.base
    longer = variant_for(section, 120)
    short_plan = plan_replicate(base, _headline_entry(base), 0)
    long_plan = plan_replicate(longer, _headline_entry(longer), 0)
    n_short = short_plan.task.u.shape[0]
    assert long_plan.task.u.shape[0] > n_short
    assert (long_plan.task.u[:n_short] == short_plan.task.u).all()
    assert (long_plan.task.y[:n_short] == short_plan.task.y).all()


# --- 受け入れ条件5: 変動の定量化 -------------------------------------------


def test_washout_sweep_quantifies_performance_variation(tmp_path: Path) -> None:
    """**受け入れ条件5**: NRMSE の (最大/最小) 比が ``meta.json`` に載り 1.0 でない。

    比だけでなく ``replicate_std_max`` / ``exceeds_replicate_noise`` まで載って
    いることを要求する。比が 1.0 でないことは「変動が測れた」ことしか意味せず、
    それだけを載せると「washout に性能が反応した」と読まれてしまう。
    """
    outputs = run_and_report_esp(tiny_pipeline_config(), tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    summary = meta["washout_sensitivity"]

    headline = summary["headline"]
    assert headline["task"] == HEADLINE_TASK
    assert headline["method"] == HEADLINE_METHOD
    assert headline["ratio"] != 1.0
    assert headline["ratio"] >= 1.0
    assert headline["nrmse_max"] > headline["nrmse_min"]
    assert set(headline) >= {
        "ratio",
        "spread",
        "replicate_std_max",
        "exceeds_replicate_noise",
        "nrmse_at_reference",
        "washout_at_min",
        "washout_at_max",
    }
    assert summary["training_size_is_constant"] is True
    assert summary["pad_series"] is True
    assert summary["n_rows"] == len(outputs.washout_rows)
    assert summary["grid"] == list(tiny_pipeline_config().washout.grid)
    # 全 (課題, 手法) の変動幅も残る (主役だけを見て一般化しないため)
    assert len(summary["by_method"]) == len(outputs.sensitivity.by_method)


def test_production_grid_quantifies_the_variation() -> None:
    """本番格子でも変動幅が 1.0 でない (記事に載る数値そのもの)。"""
    sensitivity = summarize_washout_sensitivity(production_config(), production_rows())
    assert sensitivity.training_size_is_constant is True
    assert sensitivity.headline.ratio > 1.0
    assert math.isfinite(sensitivity.headline.ratio)


def test_washout_zero_is_worst_or_equal_for_mackey_glass() -> None:
    """washout=0 の NRMSE が 01 の本番値 (200) 以上である。

    **破れたら記事の主張が変わる**ので、閾値を緩めずに止まって相談すること
    (仕様 §4 T4)。

    実測 (本番格子、補償あり、MG x ESN のレプリケート平均):
    washout=0 -> 7.094e-4 / washout=200 -> 7.077e-4。差は 0.24% しかなく、
    同じ格子点のレプリケート間標準偏差 (約 9e-5 = 平均の 13%) の**内側**である。
    つまり順序としては成り立つが、「washout を短く取ると悪化する」と言える
    ほどの効果は無い。この温度感は ``meta.json`` の
    ``washout_sensitivity.headline.exceeds_replicate_noise`` に残る。
    """
    means = mean_nrmse_by_washout(production_rows(), HEADLINE_TASK, ESN_METHOD)
    assert 0 in means, means
    assert REFERENCE_WASHOUT in means, means
    assert means[0] >= means[REFERENCE_WASHOUT], (
        "washout=0 が 01 の本番値より良くなっています。"
        f"washout=0 -> {means[0]:.6g} / washout={REFERENCE_WASHOUT} -> "
        f"{means[REFERENCE_WASHOUT]:.6g}"
    )


def test_both_tasks_and_all_three_methods_are_swept() -> None:
    """MG と遅延パリティの両方 x 3手法が回る (パリティは対照)。"""
    rows = run_washout_sweep(tiny_sweep_config())
    assert {row.task for row in rows} == {HEADLINE_TASK, DELAY_PARITY}
    assert {row.method for row in rows} == {"linear", "delay_line", ESN_METHOD}


def test_nrmse_std_is_the_replicate_spread() -> None:
    """``nrmse_std`` が同じ (課題, 手法, washout) のレプリケート間標準偏差。"""
    rows = run_washout_sweep(tiny_sweep_config())
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        grouped.setdefault((row.task, row.method, row.washout), []).append(row.nrmse)
    for row in rows:
        values = grouped[row.task, row.method, row.washout]
        expected = statistics.stdev(values) if len(values) > 1 else 0.0
        assert row.nrmse_std == pytest.approx(expected)


def test_rows_carry_the_design_they_were_measured_under() -> None:
    """``pad_series`` が行に載る (成果物だけで交絡の有無を判定できる)。"""
    padded = run_washout_sweep(tiny_sweep_config(pad_series=True))
    unpadded = run_washout_sweep(tiny_sweep_config(pad_series=False))
    assert all(row.pad_series for row in padded)
    assert not any(row.pad_series for row in unpadded)


@pytest.mark.parametrize(
    ("grid", "match"),
    [
        pytest.param((), "空", id="empty"),
        pytest.param((0, -1), "0 以上", id="negative"),
    ],
)
def test_invalid_grid_raises(grid: tuple[int, ...], match: str) -> None:
    """格子が空 / 負の washout を含む場合は ``ValueError``。"""
    with pytest.raises(ValueError, match=match):
        run_washout_sweep(tiny_sweep_config(grid=grid))


def test_summary_requires_the_headline_pair() -> None:
    """主役の組が掃引に無ければ要約は作れない (黙って別の組で代用しない)。"""
    rows = run_washout_sweep(tiny_sweep_config())
    without_esn = tuple(row for row in rows if row.method != ESN_METHOD)
    with pytest.raises(ValueError, match=HEADLINE_METHOD):
        summarize_washout_sensitivity(tiny_sweep_config(), without_esn)


def test_summary_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="行がありません"):
        summarize_washout_sensitivity(tiny_sweep_config(), ())


# --- 図 --------------------------------------------------------------------


def test_figure_is_written_at_retina_resolution(tmp_path: Path) -> None:
    """2-D の図が 200 dpi 以上で出る (受け入れ条件7)。"""
    config = tiny_sweep_config()
    rows = run_washout_sweep(config)
    sensitivity = summarize_washout_sensitivity(config, rows)
    path = plot_washout_sensitivity(
        rows,
        tmp_path / "fig_washout_sensitivity.png",
        style=setup_style(),
        sensitivity=sensitivity,
    )
    assert png_dpi(path) >= RETINA_DPI


def test_figure_rejects_empty_rows(tmp_path: Path) -> None:
    config = tiny_sweep_config()
    sensitivity = summarize_washout_sensitivity(config, run_washout_sweep(config))
    with pytest.raises(ValueError, match="rows"):
        plot_washout_sensitivity(
            (), tmp_path / "empty.png", style=setup_style(), sensitivity=sensitivity
        )

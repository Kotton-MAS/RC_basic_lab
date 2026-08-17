"""実験ランナーの公平性テスト (D-04 / D-05 / D-08).

比較実験がひとりでに無効になる経路は3つある:

1. 手法ごとに alpha 格子が違う → 結論が逆転しうる (D-04)
2. 手法ごとに評価行がずれる → そもそも同じ問題を解いていない (D-05)
3. ESN だけ構造ハイパーパラメータを検証で選ぶ → 探索予算の非対称 (D-08)

いずれも出力の見た目は正常なので、ここが最終防衛線になる。
``select_alpha`` の呼び出しを差し替えて「実際に何が渡されたか」を記録し、
主張を実測に落とす。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from rc_basics_lab.config import (
    ConfigError,
    DelayParityConfig,
    ESNConfig,
    ExperimentConfig,
    RidgeConfig,
    SeedConfig,
    SplitConfig,
    load_config,
)
from rc_basics_lab.experiment import runner
from rc_basics_lab.experiment.runner import (
    DELAY_LINE,
    ESN_METHOD,
    LINEAR,
    ResultRow,
    build_methods,
    build_tasks,
    plan_replicate,
    run_task,
)
from rc_basics_lab.experiment.split import compute_t0, make_split
from rc_basics_lab.readout.design import DelayLineSpec
from rc_basics_lab.readout.ridge import AlphaSelection
from rc_basics_lab.readout.ridge import select_alpha as real_select_alpha
from rc_basics_lab.types import FloatArray

TINY_ALPHA_GRID = (1e-4, 1e-2, 1.0)
TINY_N_LAGS_GRID = (1, 3)


def tiny_config() -> ExperimentConfig:
    """秒未満で回る最小構成 (公平性の検査に必要な構造だけを残す)。"""
    return ExperimentConfig(
        n_replicates=2,
        seeds=SeedConfig(reservoir=0, task=1, split=2),
        split=SplitConfig(washout=50, max_start_offset=20),
        ridge=RidgeConfig(alpha_grid=TINY_ALPHA_GRID, n_lags_grid=TINY_N_LAGS_GRID),
        delay_parity=DelayParityConfig(n_bits=2, delay=1, length=800),
        esn_delay_parity=ESNConfig(n_units=30, leak_rate=1.0, input_scale=1.0),
    )


def parity_entry(config: ExperimentConfig) -> runner.TaskEntry:
    entries = [entry for entry in build_tasks(config) if entry.name == "delay_parity"]
    assert len(entries) == 1
    return entries[0]


class _SelectAlphaSpy:
    """``select_alpha`` の呼び出し引数を記録しつつ本物に委譲する。"""

    def __init__(self) -> None:
        self.grids: list[tuple[float, ...]] = []
        self.train_targets: list[bytes] = []
        self.val_targets: list[bytes] = []
        self.train_rows: list[int] = []

    def __call__(
        self,
        phi_tr: FloatArray,
        y_tr: FloatArray,
        phi_val: FloatArray,
        y_val: FloatArray,
        alphas: Sequence[float],
        *,
        bias_column: int | None = 0,
    ) -> AlphaSelection:
        self.grids.append(tuple(float(alpha) for alpha in alphas))
        self.train_targets.append(y_tr.tobytes())
        self.val_targets.append(y_val.tobytes())
        self.train_rows.append(int(phi_tr.shape[0]))
        return real_select_alpha(
            phi_tr, y_tr, phi_val, y_val, alphas, bias_column=bias_column
        )


def _spy_run(
    monkeypatch: pytest.MonkeyPatch, config: ExperimentConfig
) -> tuple[_SelectAlphaSpy, list[ResultRow]]:
    spy = _SelectAlphaSpy()
    monkeypatch.setattr(runner, "select_alpha", spy)
    rows = run_task(config, parity_entry(config))
    return spy, rows


def test_alpha_grid_is_shared_across_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    """全 (手法, 候補) が config.ridge.alpha_grid をそのまま受け取る (D-04)。"""
    config = tiny_config()
    spy, rows = _spy_run(monkeypatch, config)
    # 手法3種 x レプリケート2 のうち、遅延線は候補ぶん呼ばれる
    expected_calls = config.n_replicates * (1 + len(TINY_N_LAGS_GRID) + 1)
    assert len(spy.grids) == expected_calls
    assert set(spy.grids) == {tuple(config.ridge.alpha_grid)}
    # 実際に格子の中から選ばれている (格子が使われずに既定値が効いていない)
    assert {row.alpha for row in rows} <= set(config.ridge.alpha_grid)


def test_per_method_alpha_grid_key_is_config_error(tmp_path: Path) -> None:
    """手法別 alpha 格子キーは YAML に置けない (D-04 の入口の防衛)。"""
    path = tmp_path / "config.yaml"
    path.write_text("ridge:\n  alpha_grid_esn: [1.0]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="alpha_grid_esn"):
        load_config(path)


def test_all_methods_share_identical_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """1レプリケート内で全手法の train/val/test 行集合が完全一致する (D-05)。"""
    config = dataclasses.replace(tiny_config(), n_replicates=1)
    spy, rows = _spy_run(monkeypatch, config)
    # 目標行がバイト一致 = 同じ行 index を見ている (目標は手法によらず1本)
    assert len(set(spy.train_targets)) == 1
    assert len(set(spy.val_targets)) == 1
    assert len(set(spy.train_rows)) == 1
    # テスト区間と基準点も全手法で共通
    assert {(row.n_train, row.n_val, row.n_test, row.t0) for row in rows} == {
        (spy.train_rows[0], rows[0].n_val, rows[0].n_test, rows[0].t0)
    }
    assert {row.method for row in rows} == {LINEAR, DELAY_LINE, ESN_METHOD}


def test_t0_covers_every_candidate_first_valid() -> None:
    """t0 が全手法・全候補の first_valid と washout を覆う (D-05 の基準点)。"""
    config = tiny_config()
    plan = plan_replicate(config, parity_entry(config), replicate=0)
    first_valids = [
        design.first_valid for group in plan.designs.values() for design in group
    ]
    assert plan.t0 == max([config.split.washout, *first_valids])
    assert plan.t0 >= max(TINY_N_LAGS_GRID)
    # 分割はすべて t0 以降にある = どの手法でも NaN 行を踏まない
    assert plan.split.start >= plan.t0
    for group in plan.designs.values():
        for design in group:
            block = design.phi[plan.split.start : plan.split.test.stop]
            assert np.all(np.isfinite(block))


def test_split_seed_changes_boundaries() -> None:
    """seeds.split を変えると開始オフセットが変わり、seeds.reservoir では変わらない。"""
    config = tiny_config()
    n_steps = config.delay_parity.length
    t0 = config.split.washout

    def offset_for(seeds: SeedConfig) -> int:
        from rc_basics_lab.seeds import SeedStream, make_rng

        rng = make_rng(seeds, SeedStream.SPLIT, 0)
        return make_split(config.split, n_steps, t0, rng).offset

    base = config.seeds
    other_split = dataclasses.replace(base, split=base.split + 100)
    other_reservoir = dataclasses.replace(base, reservoir=base.reservoir + 100)
    assert offset_for(base) != offset_for(other_split)
    assert offset_for(base) == offset_for(other_reservoir)


def test_split_window_size_does_not_depend_on_offset() -> None:
    """オフセットが動いても行数は不変 (レプリケート間で n_train が揺れない)。"""
    config = tiny_config()
    sizes = set()
    offsets = set()
    for seed in range(6):
        split = make_split(
            config.split,
            config.delay_parity.length,
            config.split.washout,
            np.random.default_rng(seed),
        )
        sizes.add(split.sizes)
        offsets.add(split.offset)
    assert len(sizes) == 1
    assert len(offsets) > 1


def test_esn_hyperparameters_are_not_validation_selected() -> None:
    """ESN の構造ハイパーパラメータは検証で選ばれない (D-08)。

    3方向から固定する: (1) 検証候補は特徴仕様だけで ESN 側は候補1つ、
    (2) 設定値がそのままリザバー生成に渡る、(3) 設定を変えると結果が変わる
    (検証で選んでいるなら、悪い設定を与えても同じ良い値に収束してしまう)。
    """
    config = tiny_config()
    methods = {method.name: method for method in build_methods(config)}
    assert len(methods[ESN_METHOD].candidates) == 1
    assert len(methods[LINEAR].candidates) == 1
    # 検証で選ぶ余地があるのは遅延線の n_lags だけ
    assert len(methods[DELAY_LINE].candidates) == len(TINY_N_LAGS_GRID)
    assert all(
        isinstance(candidate, DelayLineSpec)
        for candidate in methods[DELAY_LINE].candidates
    )

    entry = parity_entry(config)
    assert entry.esn == config.esn_delay_parity

    weak = dataclasses.replace(
        config,
        n_replicates=1,
        esn_delay_parity=dataclasses.replace(
            config.esn_delay_parity, spectral_radius=0.05, input_scale=0.05
        ),
    )
    strong = dataclasses.replace(config, n_replicates=1)

    def esn_nrmse(cfg: ExperimentConfig) -> float:
        rows = run_task(cfg, parity_entry(cfg))
        return next(row.nrmse for row in rows if row.method == ESN_METHOD)

    assert esn_nrmse(weak) != pytest.approx(esn_nrmse(strong), rel=1e-3)


def test_compute_t0_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="washout"):
        compute_t0([0, 1], -1)
    with pytest.raises(ValueError, match="first_valid"):
        compute_t0([-1], 0)


def test_run_task_produces_one_row_per_method_and_replicate() -> None:
    config = tiny_config()
    rows = run_task(config, parity_entry(config))
    assert len(rows) == config.n_replicates * 3
    assert {(row.task, row.method, row.replicate) for row in rows} == {
        ("delay_parity", method, replicate)
        for method in (LINEAR, DELAY_LINE, ESN_METHOD)
        for replicate in range(config.n_replicates)
    }
    for row in rows:
        assert row.seed_reservoir == config.seeds.reservoir
        assert row.seed_task == config.seeds.task
        assert row.seed_split == config.seeds.split
        assert row.wall_time_s > 0.0
        assert row.nmse == pytest.approx(row.nrmse**2)


def test_delay_line_reports_selected_n_lags() -> None:
    """遅延線の n_lags 列が候補格子の値になる (選択結果が記録されている)。"""
    config = tiny_config()
    rows = run_task(config, parity_entry(config))
    for row in rows:
        expected = set(TINY_N_LAGS_GRID) if row.method == DELAY_LINE else {0}
        assert row.n_lags in expected

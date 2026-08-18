"""実験 3-C (NARMA10) の配線テスト (D-04 / D-05 / D-31 / D-39).

3-C の主張は「**探索予算をそろえた**対照で NARMA10 を測り直す」ことなので、
公平性の機構 (01 の ``run_task``) を本当に通っているかが結論そのものを決める。
ここでは4つを実測する:

1. 01 の ``run_task`` を通り、全手法が同一の行 index で学習・評価する
   (D-05 / D-31)。
2. alpha 格子が単一キーから全手法へそのまま渡り、本番 YAML では 01 と同一の
   格子である (D-04)。
3. 容量を測った状態行列が ``run_task`` が使った状態行列と**同一オブジェクト**
   である (再生成していない)。
4. 3-C の ESN が宣言どおり N=50 である (D-39)。

加えて、01 の成果物 (``results/comparison.csv``) が1バイトも動いていない
ことを実際に再生成して確かめる (D-31 の「``build_tasks`` に足さない」の帰結)。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from rc_basics_lab.config import (
    Capacity03Config,
    CapacityDriveConfig,
    ESNConfig,
    ExperimentConfig,
    IpcConfig,
    MemoryCapacityConfig,
    Narma10Config,
    RidgeConfig,
    SeedConfig,
    SplitConfig,
    load_config,
    load_config_as,
)
from rc_basics_lab.diagnostics.base import DiagnosticContext
from rc_basics_lab.experiment import narma as narma_module
from rc_basics_lab.experiment import runner
from rc_basics_lab.experiment.capacity import EXPERIMENT_NARMA10, CapacityMeasurement
from rc_basics_lab.experiment.narma import (
    NARMA10_ESN_SECTION,
    NARMA10_REFERENCE_NMSE,
    NARMA10_REFERENCE_NOTE,
    Narma10Results,
    narma_esn_config,
    narma_task_entry,
    run_narma10,
    summarize_narma10,
)
from rc_basics_lab.experiment.report import COMPARISON_CSV, write_comparison_csv
from rc_basics_lab.experiment.runner import (
    DELAY_LINE,
    ESN_METHOD,
    LINEAR,
    ReplicatePlan,
    ResultRow,
    build_tasks,
    run_experiment,
)
from rc_basics_lab.readout.ridge import AlphaSelection
from rc_basics_lab.readout.ridge import select_alpha as real_select_alpha
from rc_basics_lab.tasks.narma import NARMA10_INPUT_STD, TASK_NAME
from rc_basics_lab.types import FloatArray

ROOT = Path(__file__).resolve().parents[1]
CONFIG_03 = ROOT / "experiments" / "03_capacity" / "config.yaml"
CONFIG_01 = ROOT / "experiments" / "01_what_is_rc" / "config.yaml"
RESULTS_01 = ROOT / "results"

TINY_ALPHA_GRID = (1e-4, 1e-2, 1.0)
TINY_N_LAGS_GRID = (2, 6)


def tiny_config() -> Capacity03Config:
    """秒未満で 3-C を回せる縮小設定 (**構造は本番と同じ**)。"""
    return Capacity03Config(
        name="narma-tiny",
        drive=CapacityDriveConfig(distribution="uniform", washout=40),
        mc=MemoryCapacityConfig(max_delay=20, n_surrogates=5),
        ipc=IpcConfig(
            max_delay_by_degree=(8, 4), n_surrogates=5, n_surrogate_targets=2
        ),
        narma=Narma10Config(
            length=700,
            base=ExperimentConfig(
                name="narma-tiny-base",
                n_replicates=3,
                seeds=SeedConfig(reservoir=0, task=1, split=2),
                split=SplitConfig(washout=50, max_start_offset=20),
                ridge=RidgeConfig(
                    alpha_grid=TINY_ALPHA_GRID, n_lags_grid=TINY_N_LAGS_GRID
                ),
                esn_mackey_glass=ESNConfig(
                    n_units=12, leak_rate=1.0, input_scale=1.0, density=0.5
                ),
            ),
        ),
    )


class _SelectAlphaSpy:
    """``select_alpha`` の呼び出し引数を記録しつつ本物に委譲する (01 と同じ形)。"""

    def __init__(self) -> None:
        self.grids: list[tuple[float, ...]] = []
        self.train_targets: list[bytes] = []
        self.train_rows: list[int] = []

    def __call__(
        self,
        phi_tr: FloatArray,
        y_tr: FloatArray,
        phi_val: FloatArray,
        y_val: FloatArray,
        alphas: Sequence[float],
        *,
        bias_column: int | None,
    ) -> AlphaSelection:
        self.grids.append(tuple(float(alpha) for alpha in alphas))
        self.train_targets.append(y_tr.tobytes())
        self.train_rows.append(int(phi_tr.shape[0]))
        return real_select_alpha(
            phi_tr, y_tr, phi_val, y_val, alphas, bias_column=bias_column
        )


# --- D-05 / D-31: 01 の run_task をそのまま通る -------------------------------


def test_narma10_reuses_run_task_and_shares_rows_across_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3-C は 01 の ``run_task`` を通り、全手法が同一の行で評価される。

    ``run_task`` を経由していることを**呼び出しの実測**で固定し (差し替えた
    スパイが呼ばれる)、その中で D-05 (1レプリケート内で全手法が同一の行
    index) が効いていることを行の値で確かめる。``build_tasks`` に NARMA10 が
    混入していないこと (D-31) も同時に見る —— 混入すると 01 の
    ``comparison.csv`` に行が増える。
    """
    config = tiny_config()
    calls: list[tuple[ExperimentConfig, runner.TaskEntry, ReplicatePlan | None]] = []
    real_run_task = narma_module.run_task

    def spy(
        cfg: ExperimentConfig,
        task_entry: runner.TaskEntry,
        *,
        plan0: ReplicatePlan | None = None,
    ) -> list[ResultRow]:
        calls.append((cfg, task_entry, plan0))
        return real_run_task(cfg, task_entry, plan0=plan0)

    monkeypatch.setattr(narma_module, "run_task", spy)
    results = run_narma10(config)

    # 01 の run_task をちょうど1回、plan0 つきで呼んでいる
    assert len(calls) == 1
    cfg, entry, plan0 = calls[0]
    assert cfg is config.narma.base
    assert entry.name == TASK_NAME
    assert plan0 is results.plan0

    base = config.narma.base
    assert len(results.rows) == 3 * base.n_replicates
    assert {row.method for row in results.rows} == {LINEAR, DELAY_LINE, ESN_METHOD}
    assert {row.task for row in results.rows} == {TASK_NAME}
    for replicate in range(base.n_replicates):
        group = [row for row in results.rows if row.replicate == replicate]
        assert len(group) == 3
        # D-05: 同一レプリケート内では基準点も分割サイズも全手法で共通
        assert len({(row.t0, row.n_train, row.n_val, row.n_test) for row in group}) == 1

    # D-31: 01 の課題列挙に NARMA10 は混ざっていない
    assert {item.name for item in build_tasks(load_config(CONFIG_01))} == {
        "mackey_glass",
        "delay_parity",
    }
    assert TASK_NAME not in {item.name for item in build_tasks(config.narma.base)}
    # ExperimentConfig (01 専用) にも NARMA10 のフィールドは無い (D-13 / D-31)
    assert not hasattr(ExperimentConfig(), "narma")


# --- D-04: alpha 格子は単一キーから全手法へ ----------------------------------


def test_narma10_alpha_grid_is_shared_across_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全 (手法, 候補) が ``narma.base.ridge.alpha_grid`` をそのまま受け取る。

    本番 YAML では**その格子が 01 と同一**であることも突き合わせる。3-C の
    主張は「探索予算をそろえた比較」なので、01 と違う格子で回すと 01 の結果と
    並べて読めなくなる。
    """
    config = tiny_config()
    spy = _SelectAlphaSpy()
    monkeypatch.setattr(runner, "select_alpha", spy)
    results = run_narma10(config)

    base = config.narma.base
    # 線形1 + 遅延線 len(n_lags_grid) + ESN1 を全レプリケートぶん
    expected_calls = base.n_replicates * (1 + len(TINY_N_LAGS_GRID) + 1)
    assert len(spy.grids) == expected_calls
    assert set(spy.grids) == {tuple(base.ridge.alpha_grid)}
    assert {row.alpha for row in results.rows} <= set(base.ridge.alpha_grid)
    # 遅延線が選ぶ k は格子の中 (探索予算の非対称は遅延線の側に大きい、D-08)
    assert {row.n_lags for row in results.rows if row.method == DELAY_LINE} <= set(
        base.ridge.n_lags_grid
    )

    production = load_config_as(CONFIG_03, Capacity03Config)
    assert (
        production.narma.base.ridge.alpha_grid
        == load_config(CONFIG_01).ridge.alpha_grid
    ), "3-C の alpha 格子が 01 と食い違っています (D-04)"
    assert production.narma.base.ridge.n_lags_grid == (10, 15, 20, 25, 30), (
        "要件書のタップ数 k=10〜30"
    )


# --- 容量測定と ESN が同じ状態行列を見る --------------------------------------


def test_narma10_capacity_uses_the_same_states_as_the_esn_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """容量行の元になった状態が ``plan0.states`` と**同一オブジェクト**である。

    「同じ値の別配列」では足りない。値が一致するのはシードが同じだからで
    あって、片方の経路の ``n_steps`` やシードを変えれば黙って別物になる。
    ``run_task`` がレプリケート0 を作り直していないこと (``plan0=`` で渡した
    ぶんの ``plan_replicate`` が省かれていること) も呼び出し回数で固定する。
    """
    config = tiny_config()
    plans: list[ReplicatePlan] = []
    inner_calls: list[int] = []
    seen_states: list[FloatArray] = []
    seen_inputs: list[FloatArray] = []

    real_plan_replicate = narma_module.plan_replicate
    real_measure = narma_module.measure_capacity

    def outer_plan(
        cfg: ExperimentConfig, task_entry: runner.TaskEntry, replicate: int
    ) -> ReplicatePlan:
        plan = real_plan_replicate(cfg, task_entry, replicate)
        plans.append(plan)
        return plan

    def inner_plan(
        cfg: ExperimentConfig, task_entry: runner.TaskEntry, replicate: int
    ) -> ReplicatePlan:
        inner_calls.append(replicate)
        return real_plan_replicate(cfg, task_entry, replicate)

    def measure_spy(
        states: FloatArray,
        u: FloatArray,
        *,
        ctx: DiagnosticContext,
        mc_cfg: MemoryCapacityConfig,
        ipc_cfg: IpcConfig,
    ) -> CapacityMeasurement:
        seen_states.append(states)
        seen_inputs.append(u)
        return real_measure(states, u, ctx=ctx, mc_cfg=mc_cfg, ipc_cfg=ipc_cfg)

    monkeypatch.setattr(narma_module, "plan_replicate", outer_plan)
    monkeypatch.setattr(runner, "plan_replicate", inner_plan)
    monkeypatch.setattr(narma_module, "measure_capacity", measure_spy)
    results = run_narma10(config)

    assert len(plans) == 1, "3-C 側が plan_replicate を2回以上呼んでいます"
    assert len(seen_states) == 1
    # **同一オブジェクト** (再生成でも複製でもない)
    assert seen_states[0] is plans[0].states
    assert seen_states[0] is results.plan0.states
    assert seen_inputs[0] is plans[0].task.u
    # run_task はレプリケート0 を作り直さない (plan0= の効き)
    assert inner_calls == list(range(1, config.narma.base.n_replicates))
    # D-35: 外部生成の X でも measure_capacity が読み取り専用にする
    assert results.plan0.states.flags.writeable is False

    row = results.capacity.row
    assert row.experiment == EXPERIMENT_NARMA10
    assert row.n_steps == results.plan0.states.shape[0] == config.narma.length
    assert row.n_units == results.plan0.states.shape[1]
    # 駆動強度は「設定値 = 宣言した分布の閉形式」/「実測値」で列を分ける
    assert row.sigma_u == pytest.approx(NARMA10_INPUT_STD)
    assert row.input_drive_std == pytest.approx(
        float(np.std(results.plan0.task.u)), rel=1e-12
    )
    assert row.sigma_u != row.input_drive_std


# --- D-39: 3-C の ESN 規模 ----------------------------------------------------


def test_narma10_esn_size_matches_the_declared_choice() -> None:
    """本番の 3-C は N=50 で回る (D-39)。**実際に読む設定**を見る。

    セクション名 (``NARMA10_ESN_SECTION``) の宣言と、``narma_task_entry`` が
    実際に ``TaskEntry`` へ載せる ESN 設定が同じ1本であることも固定する ——
    宣言だけを見ると「50 に直したセクションを誰も読んでいない」を通す。
    N=50 は 3-B (IPC 掃引) と同じ規模であり、そのまま突き合わせられる。
    """
    config = load_config_as(CONFIG_03, Capacity03Config)
    base = config.narma.base
    entry = narma_task_entry(config)

    assert entry.esn is narma_esn_config(base)
    assert entry.esn is getattr(base, NARMA10_ESN_SECTION)
    assert entry.esn.n_units == 50
    assert entry.esn.n_units == config.ipc_sweep.n_units, (
        "D-39: 3-C の ESN は 3-B (IPC 掃引) と同じ規模にする"
    )
    # 参照値が N = 50 規模の報告であることと対応している (原典は未特定)
    assert set(NARMA10_REFERENCE_NMSE) == {"linear_ceiling", "nonlinear_rc"}
    assert "未特定" in NARMA10_REFERENCE_NOTE


# --- 01 の成果物が動いていない -----------------------------------------------


def _without_wall_time(csv_text: str) -> list[list[str]]:
    rows = [line.split(",") for line in csv_text.strip().splitlines()]
    dropped = rows[0].index("wall_time_s")
    return [
        [field for index, field in enumerate(row) if index != dropped] for row in rows
    ]


def test_01_artifacts_are_unchanged(tmp_path: Path) -> None:
    """01 の ``comparison.csv`` が (``wall_time_s`` を除いて) 一致する。

    3-C は 01 の ``run_task`` を**再利用**するので、``build_tasks`` や
    ``ExperimentConfig`` を触ると 01 の成果物が動く (D-31)。宣言
    (``build_tasks`` の中身) だけでなく、本番設定で実際に再生成した CSV を
    コミット済みの成果物と突き合わせる。
    """
    committed = (RESULTS_01 / COMPARISON_CSV).read_text(encoding="utf-8")
    rows = run_experiment(load_config(CONFIG_01))
    regenerated = write_comparison_csv(rows, tmp_path / COMPARISON_CSV).read_text(
        encoding="utf-8"
    )
    assert _without_wall_time(regenerated) == _without_wall_time(committed)


# --- 勝敗の要約 (向きは問わない) ---------------------------------------------


def test_verdict_records_either_direction() -> None:
    """``narma10_verdict`` はどちらが勝っても同じ形で残る (仕様 §4 T4)。"""
    results = run_narma10(tiny_config())
    verdict = results.verdict
    assert set(verdict.nmse_mean) == {LINEAR, DELAY_LINE, ESN_METHOD}
    assert verdict.best_method in verdict.nmse_mean
    assert verdict.nmse_mean[verdict.best_method] == min(verdict.nmse_mean.values())
    assert verdict.delay_line_beats_esn == (
        verdict.nmse_mean[DELAY_LINE] < verdict.nmse_mean[ESN_METHOD]
    )
    summary = verdict.to_summary()
    assert summary["reference_nmse"] == dict(NARMA10_REFERENCE_NMSE)
    assert "未特定" in str(summary["reference_note"])

    # 向きが逆でも同じ形 (人工の行で両方向を通す)
    def row(method: str, nmse: float, n_lags: int = 0) -> ResultRow:
        return ResultRow(
            task=TASK_NAME,
            method=method,
            replicate=0,
            seed_reservoir=0,
            seed_task=1,
            seed_split=2,
            alpha=1.0,
            n_lags=n_lags,
            rmse=nmse,
            nrmse=nmse,
            nmse=nmse,
            sign_accuracy=0.5,
            n_train=10,
            n_val=3,
            n_test=7,
            t0=1,
            wall_time_s=0.0,
        )

    esn_wins = summarize_narma10(
        [row(LINEAR, 1.0), row(DELAY_LINE, 0.4, n_lags=10), row(ESN_METHOD, 0.2)]
    )
    assert esn_wins.best_method == ESN_METHOD
    assert esn_wins.delay_line_beats_esn is False
    assert esn_wins.selected_n_lags == (10,)

    delay_wins = summarize_narma10(
        [row(LINEAR, 1.0), row(DELAY_LINE, 0.1, n_lags=30), row(ESN_METHOD, 0.2)]
    )
    assert delay_wins.best_method == DELAY_LINE
    assert delay_wins.delay_line_beats_esn is True

    with pytest.raises(ValueError, match="rows"):
        summarize_narma10([])


def test_capacity_row_shares_the_surrogate_seed_with_the_sweeps() -> None:
    """3-C の容量も ``seeds.surrogate`` を共有する (D-37 の対象に入る)。"""
    config = tiny_config()
    results: Narma10Results = run_narma10(config)
    row = results.capacity.row
    assert row.seed_surrogate == config.seeds.surrogate
    assert row.washout == config.drive.washout
    assert row.seed_reservoir == config.narma.base.seeds.reservoir
    # 3-C のリザバーを駆動するのは課題の入力なので、駆動側は task ストリーム
    assert row.seed_drive == config.narma.base.seeds.task

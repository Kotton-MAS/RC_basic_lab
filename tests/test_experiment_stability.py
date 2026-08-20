"""実験 4-C (3態マップ) と 4-D (同じ状態行列への容量) のテスト.

このファイルが守るのは4つ。

1. **3態が ``state_noise`` で変わる** (受け入れ条件4 / D-45) —— 成果物
   ``results/04_chaotic_freerun/stability.csv`` の行から判定する。図は見ない。
2. **確保軸5 (条件数) が条件を1つも作る前に効く** (D-34)。
3. **1条件につきリザバーは1つ、状態行列は1本** (仕様 §5 禁止する構造4)。
4. **02 の ESP 判定経路を使わない** (D-47 / ADR 0001 §3.4)。
"""

from __future__ import annotations

import csv
import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import rc_basics_lab.experiment.stability as stability_module
from rc_basics_lab.config import (
    Chaos04Config,
    ESNConfig,
    ExperimentConfig,
    FreeRunConfig,
    IpcConfig,
    LorenzConfig,
    MackeyGlassConfig,
    MemoryCapacityConfig,
    RidgeConfig,
    SplitConfig,
    StabilityConfig,
)
from rc_basics_lab.experiment.attractor import REGIMES
from rc_basics_lab.experiment.freerun import estimate_lorenz_lyapunov
from rc_basics_lab.experiment.stability import (
    CAPACITY_CSV,
    EXPERIMENT_FREERUN_CAPACITY,
    EXPERIMENT_STABILITY,
    STABILITY_CSV,
    STABILITY_CSV_COLUMNS,
    StabilityCondition,
    StabilityRow,
    condition_esn_config,
    condition_task_entry,
    regime_map,
    run_stability_experiment,
    stability_conditions,
    validate_condition_count,
    write_stability_csv,
)

if TYPE_CHECKING:  # pragma: no cover - 型検査時のみ必要
    from collections.abc import Mapping

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "04_chaotic_freerun"


def read_rows(path: Path) -> list[dict[str, str]]:
    """成果物 CSV を辞書の並びとして読む (図も実験も走らせない)。"""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def small_config() -> Chaos04Config:
    """4-C / 4-D を秒未満で回せる縮小設定 (**構造は本番と同じ**)。"""
    return Chaos04Config(
        name="stability-test",
        base=ExperimentConfig(
            n_replicates=1,
            split=SplitConfig(washout=30, max_start_offset=10),
            ridge=RidgeConfig(alpha_grid=(1.0e-6, 1.0e-3), n_lags_grid=(2,)),
            mackey_glass=MackeyGlassConfig(length=700, integration_burn_in=100),
            esn_mackey_glass=ESNConfig(
                n_units=15, leak_rate=0.5, input_scale=0.5, density=0.5
            ),
        ),
        lorenz=LorenzConfig(integration_burn_in=100, length=600, standardize_steps=150),
        freerun=FreeRunConfig(warmup_steps=10, free_run_steps=40, stats_steps=80),
        stability=StabilityConfig(
            spectral_radius_grid=(0.9,),
            leak_rate_grid=(0.5,),
            state_noise_grid=(0.0,),
            n_replicates=1,
        ),
        mc=MemoryCapacityConfig(max_delay=12, n_surrogates=8),
        ipc=IpcConfig(
            max_delay_by_degree=(8, 4), n_surrogates=8, n_surrogate_targets=2
        ),
    )


# --- 確保軸5 -------------------------------------------------------------------


def test_validate_condition_count_rejects_the_condition_axis() -> None:
    """確保軸5 が上書き不能な定数で塞がれている (D-34)。"""
    validate_condition_count(1)
    with pytest.raises(ValueError, match="条件が1つもありません"):
        validate_condition_count(0)
    with pytest.raises(ValueError, match="条件数が上限"):
        validate_condition_count(stability_module._MAX_CONDITIONS + 1)


def test_condition_count_is_checked_before_any_condition_is_built() -> None:
    """確保軸5 は**条件を1つも作る前に**効く (D-34 の規律)。

    ``StabilityCondition`` の構築を「呼ばれたら落ちる」ものに差し替えたうえで
    上限超えの格子を渡す。上限の検査が後ろにあると、構築側の
    ``AssertionError`` が先に出る。
    """
    config = dataclasses.replace(
        small_config(),
        stability=StabilityConfig(
            spectral_radius_grid=tuple(float(index) for index in range(60)),
            leak_rate_grid=tuple(float(index) for index in range(60)),
            state_noise_grid=(0.0,),
            n_replicates=1,
        ),
    )

    def forbidden(*args: object, **kwargs: object) -> StabilityCondition:
        raise AssertionError("上限検査より先に条件を作っています")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(stability_module, "StabilityCondition", forbidden)
        with pytest.raises(ValueError, match="条件数が上限"):
            stability_conditions(config)


def test_condition_count_is_the_product_of_the_grids_and_replicates() -> None:
    """条件数は格子の積 x レプリケート (片方だけを縛る検査はすり抜ける)。"""
    config = dataclasses.replace(
        small_config(),
        stability=StabilityConfig(
            spectral_radius_grid=(0.7, 0.9),
            leak_rate_grid=(0.1, 0.3, 0.6),
            state_noise_grid=(0.0, 1.0e-3),
            n_replicates=2,
        ),
    )
    assert len(stability_conditions(config)) == 2 * 3 * 2 * 2


# --- 掃引の組み立て -------------------------------------------------------------


def test_condition_esn_config_moves_only_the_three_axes() -> None:
    """掃引で動くのは rho / リーク率 / 状態ノイズだけ (D-08)。"""
    config = small_config()
    condition = StabilityCondition(
        rho=1.2, leak_rate=0.8, state_noise=1.0e-3, replicate=0
    )
    changed = condition_esn_config(config, condition)
    base = config.base.esn_mackey_glass
    moved = {
        item.name
        for item in dataclasses.fields(ESNConfig)
        if getattr(base, item.name) != getattr(changed, item.name)
    }
    assert moved == {"spectral_radius", "leak_rate", "state_noise"}
    assert condition_task_entry(config, condition).esn == changed


def test_stability_never_uses_the_esp_condition_path() -> None:
    """4-C は 02 の比較軌道経路を1回も呼ばない (D-47 / ADR 0001 §3.4)。

    状態ノイズを掃引軸に持つのに ``simulate_condition`` を通ると、
    D-14 の3ストリーム分離の外から4本目の変動が混ざる。呼びたくなったら
    設計の逸脱である、という決定を実測で固定する。
    """
    import rc_basics_lab.experiment.esp as esp_module

    config = dataclasses.replace(
        small_config(),
        stability=StabilityConfig(
            spectral_radius_grid=(0.9,),
            leak_rate_grid=(0.5,),
            state_noise_grid=(1.0e-3,),
            n_replicates=1,
        ),
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("4-C が 02 の ESP 判定経路を呼びました (D-47)")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(esp_module, "simulate_condition", forbidden)
        patch.setattr(stability_module, "run_free_run", stability_module.run_free_run)
        results = run_stability_experiment(config, estimate_lorenz_lyapunov(config))
    assert len(results.outcomes) == 1


def test_one_state_matrix_per_condition_is_shared_by_4c_and_4d() -> None:
    """1条件につき状態行列は1本 (仕様 §5 禁止する構造4)。

    ``plan_replicate`` の呼び出し回数が条件数と一致し、``measure_capacity`` へ
    渡る配列が**その ``plan.states`` そのもの**であることを実測する。値の一致
    ではなく同一性で測るのは、「同じシードで作り直した」実装が値では通って
    しまうためである。
    """
    import rc_basics_lab.experiment.freerun as freerun_module

    config = small_config()
    plans: list[object] = []
    original_plan = freerun_module.plan_replicate
    original_measure = stability_module.measure_capacity
    measured: list[object] = []

    def spy_plan(*args: object, **kwargs: object) -> object:
        plan = original_plan(*args, **kwargs)  # type: ignore[arg-type]
        plans.append(plan)
        return plan

    def spy_measure(states: object, u: object, **kwargs: object) -> object:
        measured.append(states)
        return original_measure(states, u, **kwargs)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(freerun_module, "plan_replicate", spy_plan)
        patch.setattr(stability_module, "measure_capacity", spy_measure)
        results = run_stability_experiment(config, estimate_lorenz_lyapunov(config))

    assert len(plans) == len(results.outcomes)
    assert len(measured) == len(results.outcomes)
    assert measured[0] is plans[0].states  # type: ignore[attr-defined]


def test_capacity_rows_go_through_the_03_seam() -> None:
    """4-D は 03 の接ぎ目 (``capacity_row_from``) をそのまま使う。

    行の組み立てを 04 側で複製すると、``CapacityRow`` に列を1本足したときに
    04 だけ置き去りになる (型検査では落ちない)。
    """
    config = small_config()
    calls: list[str] = []
    original = stability_module.capacity_row_from

    def spy(*args: object, **kwargs: object) -> object:
        calls.append(str(kwargs.get("experiment")))
        return original(*args, **kwargs)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(stability_module, "capacity_row_from", spy)
        results = run_stability_experiment(config, estimate_lorenz_lyapunov(config))
    assert calls == [EXPERIMENT_FREERUN_CAPACITY] * len(results.outcomes)
    assert all(
        row.experiment == EXPERIMENT_FREERUN_CAPACITY for row in results.capacity_rows
    )


def test_regime_map_breaks_ties_toward_the_worse_regime() -> None:
    """多数決の同数は ``REGIMES`` の並び (発散 -> 周期 -> 再現) で決める。"""

    def row(regime: str, replicate: int) -> StabilityRow:
        return StabilityRow(
            experiment=EXPERIMENT_STABILITY,
            rho=0.9,
            leak_rate=0.3,
            state_noise=0.0,
            replicate=replicate,
            n_units=10,
            alpha=1.0,
            val_nrmse=0.5,
            regime=regime,
            amplitude_ratio=1.0,
            std_ratio=1.0,
            autocorr_peak=0.1,
            diverged=False,
            n_completed=10,
            stats_steps=10,
            valid_time_threshold=0.4,
            valid_time_steps=1,
            valid_time_lyapunov=0.1,
            valid_time_censored=False,
            wall_time_s=0.0,
        )

    rows = [row(REGIMES[2], 0), row(REGIMES[0], 1)]
    assert regime_map(rows, 0.0)[(0.9, 0.3)] == REGIMES[0]
    assert regime_map(rows, 1.0) == {}


def test_write_stability_csv_uses_the_declared_column_order(tmp_path: Path) -> None:
    """列順は ``StabilityRow`` の宣言順が単一の真実。"""
    config = small_config()
    results = run_stability_experiment(config, estimate_lorenz_lyapunov(config))
    path = write_stability_csv(results.rows, tmp_path / STABILITY_CSV)
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert tuple(header) == STABILITY_CSV_COLUMNS


# --- 成果物に対する受け入れ条件 (図は見ない) ------------------------------------


def committed_stability_rows() -> list[dict[str, str]]:
    return read_rows(RESULTS / STABILITY_CSV)


def committed_capacity_rows() -> list[dict[str, str]]:
    return read_rows(RESULTS / CAPACITY_CSV)


def _map_of(rows: list[dict[str, str]], noise: str) -> dict[tuple[str, str], str]:
    """成果物の行から (rho, leak) -> 多数決の3態 を作る (``regime_map`` と同じ規律)。"""
    typed = [
        StabilityRow(
            experiment=row["experiment"],
            rho=float(row["rho"]),
            leak_rate=float(row["leak_rate"]),
            state_noise=float(row["state_noise"]),
            replicate=int(row["replicate"]),
            n_units=int(row["n_units"]),
            alpha=float(row["alpha"]),
            val_nrmse=float(row["val_nrmse"]),
            regime=row["regime"],
            amplitude_ratio=float(row["amplitude_ratio"]),
            std_ratio=float(row["std_ratio"]),
            autocorr_peak=float(row["autocorr_peak"]),
            diverged=row["diverged"] == "True",
            n_completed=int(row["n_completed"]),
            stats_steps=int(row["stats_steps"]),
            valid_time_threshold=float(row["valid_time_threshold"]),
            valid_time_steps=int(row["valid_time_steps"]),
            valid_time_lyapunov=float(row["valid_time_lyapunov"]),
            valid_time_censored=row["valid_time_censored"] == "True",
            wall_time_s=float(row["wall_time_s"]),
        )
        for row in rows
    ]
    return {
        (f"{key[0]:g}", f"{key[1]:g}"): regime
        for key, regime in regime_map(typed, float(noise)).items()
    }


def test_noise_changes_the_regime_map() -> None:
    """**受け入れ条件4**: ``state_noise`` の注入で3態の領域が変わる (D-45)。

    成果物の行だけから判定する (図も目視も使わない、仕様 §5 禁止する構造6)。
    状態ノイズが最小のマップと最大のマップを格子点ごとに比べ、少なくとも
    1点で3態が変わることを要求する。加えて「変わった向き」も assert する ——
    状態ノイズは自走を安定させる (発散していた点が発散しなくなる) 側に効く、
    というのが要件書 設計判断3 の主張であり、逆向きに効いているなら結論を
    書き換えなければならない。
    """
    rows = committed_stability_rows()
    noises = sorted({row["state_noise"] for row in rows}, key=float)
    assert len(noises) >= 2, noises
    lowest = _map_of(rows, noises[0])
    highest = _map_of(rows, noises[-1])
    assert set(lowest) == set(highest), "格子が state_noise ごとに違います"
    changed = {key for key in lowest if lowest[key] != highest[key]}
    assert changed, "状態ノイズを変えても3態マップが 1 点も変わりません"
    stabilized = sum(
        1 for key in changed if lowest[key] == REGIMES[0] and highest[key] != REGIMES[0]
    )
    assert stabilized >= 1, (
        "状態ノイズで発散が減った点が 1 つもありません "
        f"(変化した点={sorted(changed)} / 低={lowest} / 高={highest})"
    )


def test_regimes_in_the_artifact_are_from_the_declared_set() -> None:
    """成果物の ``regime`` 列は3態のいずれか (排他かつ網羅)。"""
    regimes = {row["regime"] for row in committed_stability_rows()}
    assert regimes <= set(REGIMES)
    assert REGIMES[2] in regimes, "アトラクタ再現の条件が1つもありません"


def test_stability_and_capacity_rows_join_on_the_condition_keys() -> None:
    """4-C と 4-D の行が条件キーで1対1に対応する (同じ状態行列の2つの見方)。"""

    def key(row: Mapping[str, str]) -> tuple[str, ...]:
        return (row["rho"], row["leak_rate"], row["state_noise"], row["replicate"])

    stability_keys = [key(row) for row in committed_stability_rows()]
    capacity_keys = [key(row) for row in committed_capacity_rows()]
    assert stability_keys == capacity_keys
    assert len(set(stability_keys)) == len(stability_keys)


def test_committed_stability_csv_matches_the_production_grid() -> None:
    """成果物の格子が本番設定の格子と一致する (掃引を黙って縮めていない)。"""
    from rc_basics_lab.config import load_config_as

    config = load_config_as(
        ROOT / "experiments" / "04_chaotic_freerun" / "config.yaml", Chaos04Config
    )
    rows = committed_stability_rows()
    assert len(rows) == len(stability_conditions(config))
    for column, grid in (
        ("rho", config.stability.spectral_radius_grid),
        ("leak_rate", config.stability.leak_rate_grid),
        ("state_noise", config.stability.state_noise_grid),
    ):
        assert {float(row[column]) for row in rows} == set(grid)


def _iter_columns(rows: list[dict[str, str]], column: str) -> Iterator[float]:
    for row in rows:
        yield float(row[column])


def test_capacity_note_records_the_measurement_caveat() -> None:
    """4-D の但し書きが ``meta.json`` に残っている (数字だけを孤立させない)。

    Lorenz の駆動は i.i.d. ではないので容量は保存則 (<= N) を超える。
    これを書かずに数字だけ残すと、後から「保存則が破れている」とだけ読まれる。
    """
    import json

    from rc_basics_lab.experiment.freerun_pipeline import CAPACITY_NOTE

    meta = json.loads((RESULTS / "meta.json").read_text(encoding="utf-8"))
    assert meta["capacity_note"] == CAPACITY_NOTE
    rows = committed_capacity_rows()
    n_units = {int(row["n_units"]) for row in rows}
    assert len(n_units) == 1
    assert max(_iter_columns(rows, "mc_total")) > float(next(iter(n_units))), (
        "但し書きが説明している現象 (容量が N を超える) が実際には起きていません"
    )

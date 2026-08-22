"""実験1-B: 入力空間と状態空間の PCA 比較 (受け入れ条件4).

要件書の受け入れ条件4 は「リザバー状態の PCA が**入力空間**より高次元に広がる」。
入力空間には2つの読み方があるため、両方を数値として記録する:

- ``raw_input``: 課題が与える生の入力 (本連載ではどちらも 1 次元)
- ``delay_embedded_input``: 遅延線ベースラインが実際に使う特徴空間 (k+1 次元)

前者に対しては ``n_components_95`` の不等号が成り立つ (ここでテストする)。
**後者に対しては成り立たない**というのが本サイクルの実測結果であり、
``docs/design.md`` §7 に数値付きで記録してある。ここでは「両方の数値が必ず
記録される」ことだけを固定し、成り立たない不等号をテストで主張しない。
"""

from __future__ import annotations

from rc_basics_lab.config import (
    DelayParityConfig,
    ESNConfig,
    ExperimentConfig,
    MackeyGlassConfig,
    RidgeConfig,
    SplitConfig,
)
from rc_basics_lab.experiment.runner import build_tasks
from rc_basics_lab.experiment.state_space import (
    DELAY_EMBEDDED_INPUT,
    RAW_INPUT,
    RESERVOIR_STATE,
    analyze_task,
    collect_state_space,
    summarize,
)
from rc_basics_lab.seeds import SeedConfig

N_LAGS_GRID = (1, 4)


def tiny_config() -> ExperimentConfig:
    """秒未満で回る縮小設定 (構造は本番と同じ)。"""
    return ExperimentConfig(
        n_replicates=1,
        seeds=SeedConfig(reservoir=0, task=1, split=2),
        split=SplitConfig(
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            washout=20,
            max_start_offset=10,
        ),
        ridge=RidgeConfig(alpha_grid=(1e-4, 1e-1), n_lags_grid=N_LAGS_GRID),
        mackey_glass=MackeyGlassConfig(length=400, integration_burn_in=50),
        delay_parity=DelayParityConfig(length=400),
        esn_mackey_glass=ESNConfig(n_units=30, density=0.3),
        esn_delay_parity=ESNConfig(
            n_units=30, density=0.3, leak_rate=1.0, input_scale=1.0
        ),
    )


def test_reservoir_state_spans_more_components_than_the_raw_input() -> None:
    """状態空間が生の入力空間より高次元に広がる (受け入れ条件4 の literal な形)。"""
    config = tiny_config()
    for report in collect_state_space(config):
        state = report.space(RESERVOIR_STATE)
        raw = report.space(RAW_INPUT)
        assert raw.n_features == 1
        assert state.n_features == config.esn_mackey_glass.n_units
        assert state.n_components_95 > raw.n_components_95, report.task


def test_delay_embedded_comparison_is_always_recorded() -> None:
    """遅延埋め込みとの比較値が (不等号の向きに関わらず) 必ず残る。"""
    for report in collect_state_space(tiny_config()):
        embedded = report.space(DELAY_EMBEDDED_INPUT)
        assert embedded.n_features == max(N_LAGS_GRID) + 1
        assert 1 <= embedded.n_components_95 <= embedded.n_features
        state = report.space(RESERVOIR_STATE)
        assert 1 <= state.n_components_95 <= state.n_features
        assert state.participation_ratio > 0.0


def test_analysis_uses_the_same_rows_as_the_experiment() -> None:
    """PCA を取る行が実験の評価窓と一致する (別の区間を見て語らない)。"""
    config = tiny_config()
    entry = build_tasks(config)[0]
    report = analyze_task(config, entry)
    n_usable = (
        config.mackey_glass.length
        - config.split.max_start_offset
        - max(config.split.washout, max(N_LAGS_GRID))
    )
    assert report.n_rows == n_usable
    assert report.n_lags == max(N_LAGS_GRID)


def test_summary_is_json_friendly() -> None:
    """``meta.json`` に載せる要約が配列を含まないプレーンな値だけになる。"""
    summaries = summarize(collect_state_space(tiny_config()))
    assert summaries
    for summary in summaries:
        spaces = summary["spaces"]
        assert isinstance(spaces, list)
        for space in spaces:
            assert set(space) == {
                "space",
                "n_features",
                "n_components_95",
                "participation_ratio",
            }

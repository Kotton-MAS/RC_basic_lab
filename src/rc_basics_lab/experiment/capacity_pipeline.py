"""1コマンドで 03 の成果物をそろえる経路 (受け入れ条件7).

``capacity.csv`` / ``capacity_profile.csv`` / 図4枚 / ``meta.json`` をここで
一括生成する。CLI (``main.py --experiment 03`` と
``experiments/03_capacity/run_03.py``) はこの関数を呼ぶだけの薄い層にして、
「どのコマンドから走らせても同じ成果物が出る」を構造で保証する
(01 の ``pipeline.py`` / 02 の ``esp_pipeline.py`` と同じ規律)。

``meta.json`` には ``wall_time_breakdown`` (区間ごとの実測時間の内訳) を載せる。
仕様 §5 が「**状態生成の合計** < 60 秒 / 内訳を ``meta.json`` に出す」を性能
予算として要求しているためで、予算を割ったときに「診断が重いのか状態生成
(素の tanh ESN、O(T*N^2) の Python ループ) が重いのか」を成果物だけで
切り分けられるようにする。
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rc_basics_lab.config import Capacity03Config
from rc_basics_lab.experiment.capacity import (
    CAPACITY_CSV_COLUMNS,
    CAPACITY_PROFILE_CSV_COLUMNS,
    FIGURE_EXPERIMENTS,
    CapacityOutcome,
    CapacityProfileRow,
    CapacityResults,
    CapacityRow,
    profile_rows,
    run_capacity_experiment,
    run_length_sweep,
)
from rc_basics_lab.experiment.report import META_JSON, write_meta_for

logger = logging.getLogger(__name__)

CAPACITY_CSV = "capacity.csv"
CAPACITY_PROFILE_CSV = "capacity_profile.csv"
CAPACITY_LENGTH_CSV = "capacity_length.csv"
FIG_MC_SWEEP = "fig_mc_sweep.png"
FIG_IPC_PROFILE = "fig_ipc_profile.png"
FIG_MEMORY_NONLINEARITY = "fig_memory_nonlinearity.png"
FIG_IPC_CONSERVATION = "fig_ipc_conservation.png"

CAPACITY_ARTIFACTS: tuple[str, ...] = (
    CAPACITY_CSV,
    CAPACITY_PROFILE_CSV,
    FIG_MC_SWEEP,
    FIG_IPC_PROFILE,
    FIG_MEMORY_NONLINEARITY,
    FIG_IPC_CONSERVATION,
    META_JSON,
)
"""1コマンド (``make figures-03``) で必ず出る 03 の成果物 (CSV2枚 + 図4枚 + meta)。

並びは 02 の ``ESP_ARTIFACTS`` と同じく「CSV -> 図 -> meta.json」で、
``run_and_report_capacity`` が返す ``paths`` の順序と一致する。宣言と実体が
食い違ったら落ちるテスト (``test_artifacts_are_regenerated_in_one_command_within_the_budget``)
がこの並びを見る。

``capacity_length.csv`` (系列長 T の掃引) は**この並びに入れない**。記事に
載る成果物ではなく「容量が足りないのか T が足りないのか」を切り分ける補助
実験であり、T=1e6 まで回すので単独で 900 秒予算を食い潰す。
``--length-sweep`` (= ``make saturation-03``) で明示的に再生成する
(02 の ``esp_threshold_sensitivity.csv`` / ``threshold-02`` と同じ規律)。
"""


@dataclass(frozen=True, slots=True)
class SectionTiming:
    """1実験ぶんの実測時間の内訳 (``meta.json`` の ``wall_time_breakdown``)。

    Attributes:
        experiment: 実験ラベル。
        n_conditions: 回した条件数。
        wall_time_state_s: 状態生成 (``ESN.run``) の合計 [秒]。
        wall_time_mc_s: MC 診断の合計 [秒]。
        wall_time_ipc_s: IPC 診断の合計 [秒]。
        wall_time_s: 条件の合計 [秒] (上記3つ + 行の組み立て)。
    """

    experiment: str
    n_conditions: int
    wall_time_state_s: float
    wall_time_mc_s: float
    wall_time_ipc_s: float
    wall_time_s: float

    def to_summary(self) -> dict[str, float | int | str]:
        """``meta.json`` に載せるプレーンな dict。"""
        return dataclasses.asdict(self)


def summarize_timing(
    experiment: str, outcomes: Sequence[CapacityOutcome]
) -> SectionTiming:
    """条件ごとの実測時間を実験単位に足し上げる (仕様 §5 の内訳)。"""
    return SectionTiming(
        experiment=experiment,
        n_conditions=len(outcomes),
        wall_time_state_s=sum(item.row.wall_time_state_s for item in outcomes),
        wall_time_mc_s=sum(item.row.wall_time_mc_s for item in outcomes),
        wall_time_ipc_s=sum(item.row.wall_time_ipc_s for item in outcomes),
        wall_time_s=sum(item.row.wall_time_s for item in outcomes),
    )


@dataclass(frozen=True, slots=True)
class CapacityOutputs:
    """``run_and_report_capacity`` の成果物。

    Attributes:
        results: 3実験ぶんの条件別の結果 (行 + 図が使う配列)。
        timings: 実験ごとの実測時間の内訳 (``FIGURE_EXPERIMENTS`` と同じ並び)。
        paths: 生成したファイル (``CAPACITY_ARTIFACTS`` と同じ並び)。
        wall_time_s: 計算部分の実測 wall time (書き出しは含まない)。
    """

    results: CapacityResults
    timings: tuple[SectionTiming, ...]
    paths: tuple[Path, ...]
    wall_time_s: float

    @property
    def rows(self) -> tuple[CapacityRow, ...]:
        """``capacity.csv`` と同じ行。"""
        return self.results.rows

    @property
    def profile_rows(self) -> tuple[CapacityProfileRow, ...]:
        """``capacity_profile.csv`` と同じ行。"""
        return self.results.profile_rows


def write_capacity_csv(rows: Sequence[CapacityRow], path: Path) -> Path:
    """条件ごとの容量を CSV に書く (列順は ``CapacityRow`` の宣言順)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CAPACITY_CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))
    return path


def write_capacity_profile_csv(rows: Sequence[CapacityProfileRow], path: Path) -> Path:
    """遅延・次数ごとの容量を長形式で CSV に書く (D-38)。

    列順は ``CapacityProfileRow`` の宣言順で、**cfg に依らず一定**である。
    行は「しきい値後の容量が厳密に正のセル」だけで、絞り込みそのものは
    ``capacity.profile_rows`` が行う (書き出し側で条件を書くと、CSV と
    ``n_targets_kept`` の規準が別々にドリフトする)。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CAPACITY_PROFILE_CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))
    return path


def _log_timings(timings: Sequence[SectionTiming]) -> None:
    """区間ごとの実測時間を**数値として**ログに残す (仕様 §5 の性能予算)。"""
    for timing in timings:
        logger.info(
            "experiment=%s 条件数=%d wall_time=%.2fs "
            "(状態生成 %.2fs / MC %.2fs / IPC %.2fs)",
            timing.experiment,
            timing.n_conditions,
            timing.wall_time_s,
            timing.wall_time_state_s,
            timing.wall_time_mc_s,
            timing.wall_time_ipc_s,
        )
    logger.info(
        "状態生成の合計=%.2fs / 診断の合計=%.2fs",
        sum(timing.wall_time_state_s for timing in timings),
        sum(timing.wall_time_mc_s + timing.wall_time_ipc_s for timing in timings),
    )


def _rows_of(outcomes: Sequence[CapacityOutcome]) -> tuple[CapacityRow, ...]:
    """1実験ぶんの ``capacity.csv`` の行 (図はこの並びを読む)。"""
    return tuple(outcome.row for outcome in outcomes)


def _profile_of(
    outcomes: Sequence[CapacityOutcome],
) -> tuple[CapacityProfileRow, ...]:
    """1実験ぶんの長形式の行 (``capacity_profile.csv`` と同じ、D-38)。

    図は配列 (``CapacityOutcome.mc_profile`` / ``ipc_heatmap``) ではなく
    **書き出したのと同じ長形式の行**を読む。成果物と図の食い違い (CSV には
    正値セルしか無いのに図は配列の全セルを見ている、など) を構造で防ぐ。
    診断はここでも図でも一切走らせない。
    """
    return tuple(row for outcome in outcomes for row in profile_rows(outcome))


def run_and_report_capacity(config: Capacity03Config, out_dir: Path) -> CapacityOutputs:
    """実験 3-A / 3-B / 3-B' を実行し、CSV2枚・図4枚・meta.json を書き出す。

    Args:
        config: 03 の実験設定。
        out_dir: 出力ディレクトリ (無ければ作る)。

    Returns:
        生成した結果・区間ごとの実測時間・ファイルパス・実測 wall time。
    """
    started = time.perf_counter()
    results = run_capacity_experiment(config)
    wall_time_s = time.perf_counter() - started
    rows = results.rows
    profile = results.profile_rows
    timings = tuple(
        summarize_timing(experiment, outcomes)
        for experiment, outcomes in zip(
            FIGURE_EXPERIMENTS,
            (results.mc_sweep, results.ipc_sweep, results.conservation),
            strict=True,
        )
    )
    _log_timings(timings)

    style = setup_style()
    paths = (
        write_capacity_csv(rows, out_dir / CAPACITY_CSV),
        write_capacity_profile_csv(profile, out_dir / CAPACITY_PROFILE_CSV),
        plot_mc_sweep(
            _rows_of(results.mc_sweep),
            _profile_of(results.mc_sweep),
            out_dir / FIG_MC_SWEEP,
            style=style,
        ),
        plot_ipc_profile(
            _rows_of(results.ipc_sweep),
            _profile_of(results.ipc_sweep),
            out_dir / FIG_IPC_PROFILE,
            style=style,
        ),
        plot_memory_nonlinearity(
            _rows_of(results.ipc_sweep),
            out_dir / FIG_MEMORY_NONLINEARITY,
            style=style,
        ),
        plot_ipc_conservation(
            _rows_of(results.conservation),
            out_dir / FIG_IPC_CONSERVATION,
            style=style,
        ),
        write_meta_for(
            config,
            config.seeds,
            wall_time_s,
            # n_rows は capacity.csv の行数。capacity_profile.csv は列が違う
            # 別 CSV なので足し込まず profile_rows に分けて残す (足すと
            # 「どちらの CSV の行数か」が meta.json から読めなくなる)。
            len(rows),
            out_dir / META_JSON,
            extra={
                "n_profile_rows": len(profile),
                "wall_time_breakdown": [timing.to_summary() for timing in timings],
                # 図のラベル言語を決めた要因 (02 の meta.json と同じ形)。
                # 英語ラベルの図が出たときに「フォントが無い環境で生成した」と
                # 成果物だけで判別できる。
                "cjk_font": style.cjk_font,
            },
        ),
    )
    logger.info(
        "完了: %d 行 (capacity.csv) + %d 行 (capacity_profile.csv) / "
        "wall_time=%.2fs / 出力=%s",
        len(rows),
        len(profile),
        wall_time_s,
        ", ".join(str(path) for path in paths),
    )
    return CapacityOutputs(
        results=results, timings=timings, paths=paths, wall_time_s=wall_time_s
    )


def run_and_report_length_sweep(config: Capacity03Config, out_dir: Path) -> Path:
    """系列長 T の掃引を回し ``capacity_length.csv`` に書く。

    本体の成果物とは独立に走る (``CAPACITY_ARTIFACTS`` に含めない理由は
    モジュール docstring)。T=1e6 まで回すので ``make figures-03`` の予算
    (900 秒) の外で ``make saturation-03`` として手動実行する。
    """
    started = time.perf_counter()
    outcomes = run_length_sweep(config)
    rows = tuple(outcome.row for outcome in outcomes)
    path = write_capacity_csv(rows, out_dir / CAPACITY_LENGTH_CSV)
    logger.info(
        "系列長掃引: %d 行 / 状態生成 %.2fs / wall_time=%.2fs / 出力=%s",
        len(rows),
        sum(row.wall_time_state_s for row in rows),
        time.perf_counter() - started,
        path,
    )
    return path


__all__ = [
    "CAPACITY_ARTIFACTS",
    "CAPACITY_CSV",
    "CAPACITY_LENGTH_CSV",
    "CAPACITY_PROFILE_CSV",
    "FIG_IPC_CONSERVATION",
    "FIG_IPC_PROFILE",
    "FIG_MC_SWEEP",
    "FIG_MEMORY_NONLINEARITY",
    "CapacityOutputs",
    "SectionTiming",
    "run_and_report_capacity",
    "run_and_report_length_sweep",
    "summarize_timing",
    "write_capacity_csv",
    "write_capacity_profile_csv",
]

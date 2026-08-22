"""1コマンドで 03 の成果物をそろえる経路 (受け入れ条件7).

``capacity.csv`` / ``capacity_profile.csv`` / ``narma10.csv`` / 図5枚 /
``meta.json`` をここで一括生成する。CLI (``main.py --experiment 03`` と
``experiments/03_capacity/run_03.py``) はこの関数を呼ぶだけの薄い層にして、
「どのコマンドから走らせても同じ成果物が出る」を構造で保証する
(01 の ``pipeline.py`` / 02 の ``esp_pipeline.py`` と同じ規律)。

``meta.json`` には ``threshold_comparison`` (受け入れ条件3: しきい値処理の
有無で総容量がどれだけ変わるか) と ``wall_time_breakdown`` (区間ごとの実測
時間の内訳) を載せる。
仕様 §5 が「**状態生成の合計** < 60 秒 / 内訳を ``meta.json`` に出す」を性能
予算として要求しているためで、予算を割ったときに「診断が重いのか状態生成
(素の tanh ESN、O(T*N^2) の Python ループ) が重いのか」を成果物だけで
切り分けられるようにする。
"""

from __future__ import annotations

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
    EXPERIMENT_CONSERVATION,
    EXPERIMENT_IPC_SWEEP,
    EXPERIMENT_MC_SWEEP,
    EXPERIMENT_NARMA10,
    FIGURE_EXPERIMENTS,
    CapacityOutcome,
    CapacityProfileRow,
    CapacityResults,
    CapacityRow,
    profile_rows,
    run_capacity_experiment,
    run_length_sweep,
)
from rc_basics_lab.experiment.capacity_threshold import (
    ThresholdComparison,
    run_threshold_comparison,
)
from rc_basics_lab.experiment.narma import Narma10Results, run_narma10
from rc_basics_lab.experiment.narma_taps import (
    CSV_COLUMNS as NARMA10_TAPS_CSV_COLUMNS,
)
from rc_basics_lab.experiment.narma_taps import (
    run_narma10_tap_sweep,
    summarize_tap_sweep,
)
from rc_basics_lab.experiment.report import (
    META_JSON,
    DataclassSummaryMixin,
    write_comparison_csv,
    write_meta_for,
    write_rows_csv,
)
from rc_basics_lab.experiment.runner import ResultRow
from rc_basics_lab.experiment.waveform_data import waveform_predictions

logger = logging.getLogger(__name__)

CAPACITY_CSV = "capacity.csv"
CAPACITY_PROFILE_CSV = "capacity_profile.csv"
CAPACITY_LENGTH_CSV = "capacity_length.csv"
NARMA10_CSV = "narma10.csv"
NARMA10_TAPS_CSV = "narma10_taps.csv"
FIG_MC_SWEEP = "fig_mc_sweep.png"
FIG_IPC_PROFILE = "fig_ipc_profile.png"
FIG_MEMORY_NONLINEARITY = "fig_memory_nonlinearity.png"
FIG_IPC_CONSERVATION = "fig_ipc_conservation.png"
FIG_NARMA10_CONTROL = "fig_narma10_control.png"
FIG_NARMA10_TAPS = "fig_narma10_taps.png"
FIG_NARMA10_WAVEFORM = "fig_narma10_waveform.png"

CAPACITY_ARTIFACTS: tuple[str, ...] = (
    CAPACITY_CSV,
    CAPACITY_PROFILE_CSV,
    NARMA10_CSV,
    NARMA10_TAPS_CSV,
    FIG_MC_SWEEP,
    FIG_IPC_PROFILE,
    FIG_MEMORY_NONLINEARITY,
    FIG_IPC_CONSERVATION,
    FIG_NARMA10_CONTROL,
    FIG_NARMA10_TAPS,
    FIG_NARMA10_WAVEFORM,
    META_JSON,
)
"""1コマンド (``make figures-03``) で必ず出る 03 の成果物 (CSV4枚 + 図6枚 + meta)。

並びは 02 の ``ESP_ARTIFACTS`` と同じく「CSV -> 図 -> meta.json」で、
``run_and_report_capacity`` が返す ``paths`` の順序と一致する。宣言と実体が
食い違ったら落ちるテスト
(``test_artifacts_are_regenerated_in_one_command_within_the_budget``)
がこの並びを見る。

``capacity_length.csv`` (系列長 T の掃引) は**この並びに入れない**。記事に
載る成果物ではなく「容量が足りないのか T が足りないのか」を切り分ける補助
実験であり、T=1e6 まで回すので単独で 900 秒予算を食い潰す。
``--length-sweep`` (= ``make saturation-03``) で明示的に再生成する
(02 の ``esp_threshold_sensitivity.csv`` / ``threshold-02`` と同じ規律)。
"""


@dataclass(frozen=True, slots=True)
class SectionTiming(DataclassSummaryMixin):
    """1実験ぶんの実測時間の内訳 (``meta.json`` の ``wall_time_breakdown``)。

    **``wall_time_s`` は3-Cだけ意味が違う** (F-3b2-1-004/M4)。他の実験
    (``CapacityRow.wall_time_s`` の合計 = 状態生成 + MC + IPC) と違い、3-C は
    ``_narma_timing`` が ``narma.wall_time_s`` (``run_task``、3手法 x
    全レプリケートを含む3-C全体) に差し替える。仕様 §5 の3-C予算 (< 120秒) が
    成績の計算まで含めた区間に対する数字であり、容量測定ぶんだけでは予算判断に
    使えないため (実測: 3-C の残差は 0.149秒 = 行の45%、他の実験は
    0.002〜0.004秒の丸め)。``capacity.csv`` の行の ``wall_time_s`` (常に容量
    測定のみ、``CapacityRow.wall_time_s`` の docstring参照) とは3-Cだけ値が
    食い違うので、``meta.json`` の ``wall_time_breakdown`` を読む側は両者を
    同一視しないこと。フィールドを足す (``wall_time_scope`` 等) 選択もあったが、
    ``meta.json`` のキーが増えると成果物の再生成と `docs/design.md` §11.5 の
    実行時間表の機械照合を同時に更新する必要があるため、この docstring の訂正
    だけで塞ぐことにした。

    Attributes:
        experiment: 実験ラベル。
        n_conditions: 回した条件数。
        wall_time_state_s: 状態生成 (``ESN.run``) の合計 [秒]。
        wall_time_mc_s: MC 診断の合計 [秒]。
        wall_time_ipc_s: IPC 診断の合計 [秒]。
        wall_time_s: 条件の合計 [秒] (上記3つ + 行の組み立て)。**3-C だけ例外**
            (上記参照)。
    """

    experiment: str
    n_conditions: int
    wall_time_state_s: float
    wall_time_mc_s: float
    wall_time_ipc_s: float
    wall_time_s: float


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
        results: 掃引3本ぶんの条件別の結果 (行 + 図が使う配列)。
        narma: 実験 3-C の結果 (``narma10.csv`` の行 + 容量1条件)。
        threshold: しきい値法の比較 (受け入れ条件3。``meta.json`` の
            ``threshold_comparison``)。
        timings: 実験ごとの実測時間の内訳 (``FIGURE_EXPERIMENTS`` と同じ並び)。
        paths: 生成したファイル (``CAPACITY_ARTIFACTS`` と同じ並び)。
        wall_time_s: 計算部分の実測 wall time (書き出しは含まない)。
    """

    results: CapacityResults
    narma: Narma10Results
    threshold: ThresholdComparison
    timings: tuple[SectionTiming, ...]
    paths: tuple[Path, ...]
    wall_time_s: float

    @property
    def rows(self) -> tuple[CapacityRow, ...]:
        """``capacity.csv`` と同じ行 (掃引3本 + 3-C の1行)。"""
        return (*self.results.rows, self.narma.capacity.row)

    @property
    def profile_rows(self) -> tuple[CapacityProfileRow, ...]:
        """``capacity_profile.csv`` と同じ行 (掃引3本 + 3-C)。"""
        return (*self.results.profile_rows, *profile_rows(self.narma.capacity))

    @property
    def narma_rows(self) -> tuple[ResultRow, ...]:
        """``narma10.csv`` と同じ行 (01 の ``ResultRow``)。"""
        return self.narma.rows


def write_capacity_csv(rows: Sequence[CapacityRow], path: Path) -> Path:
    """条件ごとの容量を CSV に書く (列順は ``CapacityRow`` の宣言順)。"""
    return write_rows_csv(rows, path, CAPACITY_CSV_COLUMNS)


def write_capacity_profile_csv(rows: Sequence[CapacityProfileRow], path: Path) -> Path:
    """遅延・次数ごとの容量を長形式で CSV に書く (D-38)。

    列順は ``CapacityProfileRow`` の宣言順で、**cfg に依らず一定**である。
    行は「しきい値後の容量が厳密に正のセル」だけで、絞り込みそのものは
    ``capacity.profile_rows`` が行う (書き出し側で条件を書くと、CSV と
    ``n_targets_kept`` の規準が別々にドリフトする)。
    """
    return write_rows_csv(rows, path, CAPACITY_PROFILE_CSV_COLUMNS)


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


def _narma_timing(narma: Narma10Results) -> SectionTiming:
    """3-C の実測時間 (**容量測定の内訳 + 3-C 全体の wall time**)。

    ``wall_time_state_s`` / ``wall_time_mc_s`` / ``wall_time_ipc_s`` は掃引と
    同じ意味 (状態行列1本の生成と2診断) だが、``wall_time_s`` だけは
    ``run_task`` (3手法 x 全レプリケート) を含む 3-C 全体である。3-C の予算
    (仕様 §5: < 120 秒) は成績の計算まで含めた区間に対する数字なので、
    容量測定ぶんだけを載せると予算の判断に使えない内訳になる。
    """
    return dataclasses.replace(
        summarize_timing(EXPERIMENT_NARMA10, (narma.capacity,)),
        wall_time_s=narma.wall_time_s,
    )


def run_and_report_capacity(config: Capacity03Config, out_dir: Path) -> CapacityOutputs:
    """実験 3-A / 3-B / 3-B' / 3-C を実行し、CSV3枚・図5枚・meta.json を書き出す。

    3-C (NARMA10) の**成績**は 01 の ``ResultRow`` のまま ``narma10.csv`` へ、
    **容量** (MC / IPC) は ``experiment="3C_narma10"`` の1行として
    ``capacity.csv`` / ``capacity_profile.csv`` へ合流する。2枚の CSV を条件
    キーで join すると「NARMA10 の成績が容量のどの成分と相関するか」を見られる
    (要件書 実験3-C)。

    Args:
        config: 03 の実験設定。
        out_dir: 出力ディレクトリ (無ければ作る)。

    Returns:
        生成した結果・区間ごとの実測時間・ファイルパス・実測 wall time。
    """
    # 作図層の import を関数本体に置くのは D-53 (循環 import の解消)。
    # 先頭へ戻すと tests/test_layer_boundaries.py の AST guard と
    # subprocess guard の両方が落ちる。
    from rc_basics_lab.meta import git_commit
    from rc_basics_lab.plotting.figures_capacity import (
        plot_ipc_conservation,
        plot_ipc_profile,
        plot_mc_sweep,
        plot_memory_nonlinearity,
        plot_narma10_control,
    )
    from rc_basics_lab.plotting.figures_narma_taps import (
        plot_narma10_taps,
    )
    from rc_basics_lab.plotting.style import setup_style
    from rc_basics_lab.plotting.waveforms import (
        plot_prediction_waveform,
    )

    started = time.perf_counter()
    results = run_capacity_experiment(config)
    narma = run_narma10(config)
    # 3-C' (D-95): タップ数の軸。3-C 本体の行は1つも触らない。
    # 空の掃引をここで落とすのは、成果物 (narma10_taps.csv / 図) が静かに
    # 欠けるのを防ぐため —— 03 の成果物一覧はこの2つを含む。
    if not config.narma.n_lags_sweep:
        raise ValueError(
            "narma.n_lags_sweep が空です (3-C' の成果物を作れません)。"
            "掃引を止めたい場合も、意図を config に書いて格子を与えてください。"
        )
    taps = run_narma10_tap_sweep(config)
    # 受け入れ条件3 の一次資料。掃引とは別の1条件を回すので
    # wall_time_breakdown (実験ごとの内訳) には入らず、自分の wall_time_s を
    # threshold_comparison の中に持つ (全体の wall_time_s には含まれる)。
    threshold = run_threshold_comparison(config)
    wall_time_s = time.perf_counter() - started
    rows = (*results.rows, narma.capacity.row)
    profile = (*results.profile_rows, *profile_rows(narma.capacity))
    timings = (
        summarize_timing(EXPERIMENT_MC_SWEEP, results.mc_sweep),
        summarize_timing(EXPERIMENT_IPC_SWEEP, results.ipc_sweep),
        summarize_timing(EXPERIMENT_CONSERVATION, results.conservation),
        _narma_timing(narma),
    )
    # 内訳の並びは FIGURE_EXPERIMENTS が単一の真実 (実験を1本足したときに
    # meta.json の内訳から静かに落ちるのを防ぐ)。
    if tuple(timing.experiment for timing in timings) != FIGURE_EXPERIMENTS:
        raise ValueError(
            "wall_time_breakdown の並びが FIGURE_EXPERIMENTS と違います: "
            f"{[timing.experiment for timing in timings]}"
        )
    _log_timings(timings)

    # commit は meta.json と図の footnote (FIG-6 / D-87) で同じ値を使う。
    style = setup_style(commit=git_commit())
    paths = (
        write_capacity_csv(rows, out_dir / CAPACITY_CSV),
        write_capacity_profile_csv(profile, out_dir / CAPACITY_PROFILE_CSV),
        # 列順は 01 の CSV_COLUMNS (= ResultRow の宣言順) をそのまま使う。
        # 3-C 専用の書き出しを作ると列順の単一の真実が2つになる (D-31)。
        write_comparison_csv(narma.rows, out_dir / NARMA10_CSV),
        write_rows_csv(taps, out_dir / NARMA10_TAPS_CSV, NARMA10_TAPS_CSV_COLUMNS),
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
            # 打ち切りの外を「未計算」として 0 と別の色にするために渡す
            # (FIG-7 / D-88)。設定にしか無い情報なので、行からは復元できない。
            max_delay_by_degree=dict(
                enumerate(config.ipc.max_delay_by_degree, start=1)
            ),
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
        plot_narma10_control(
            narma.rows,
            out_dir / FIG_NARMA10_CONTROL,
            style=style,
        ),
        plot_narma10_taps(taps, out_dir / FIG_NARMA10_TAPS, style=style),
        # FIG-11 追加図3 (D-107)。「NMSE 0.15 と 0.27 の違い」を目で見せる。
        plot_prediction_waveform(
            *waveform_predictions(config.narma.base, narma.plan0),
            out_dir / FIG_NARMA10_WAVEFORM,
            task_label=("NARMA10", "NARMA10"),
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
                # 3-C は向きを問わない (仕様 §4 T4)。遅延線が ESN を上回った
                # 場合もここに同じ形で残る。参照値 (0.16 / 0.107) とその
                # **原典未特定**である旨も一緒に書く。
                "narma10_verdict": narma.verdict.to_summary(),
                "n_narma10_rows": len(narma.rows),
                "n_narma10_taps_rows": len(taps),
                "narma10_tap_sweep": summarize_tap_sweep(taps),
                # 受け入れ条件3: しきい値処理の有無で総容量がどれだけ変わるか
                # (docs/design.md §11.2 の表の一次資料)。既定 (D-27) が
                # どの行かは default_mc_mode / default_ipc_mode が名乗る。
                "threshold_comparison": threshold.to_summary(),
                # 図のラベル言語を決めた要因 (02 の meta.json と同じ形)。
                # 英語ラベルの図が出たときに「フォントが無い環境で生成した」と
                # 成果物だけで判別できる。
                "cjk_font": style.cjk_font,
            },
        ),
    )
    logger.info(
        "完了: %d 行 (capacity.csv) + %d 行 (capacity_profile.csv) + "
        "%d 行 (narma10.csv) / wall_time=%.2fs / 出力=%s",
        len(rows),
        len(profile),
        len(narma.rows),
        wall_time_s,
        ", ".join(str(path) for path in paths),
    )
    return CapacityOutputs(
        results=results,
        narma=narma,
        threshold=threshold,
        timings=timings,
        paths=paths,
        wall_time_s=wall_time_s,
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
    "FIG_NARMA10_CONTROL",
    "FIG_NARMA10_TAPS",
    "FIG_NARMA10_WAVEFORM",
    "NARMA10_CSV",
    "NARMA10_TAPS_CSV",
    "CapacityOutputs",
    "SectionTiming",
    "run_and_report_capacity",
    "run_and_report_length_sweep",
    "summarize_timing",
    "write_capacity_csv",
    "write_capacity_profile_csv",
]

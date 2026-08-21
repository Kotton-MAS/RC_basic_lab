"""1コマンドで 05 の成果物をそろえる経路 (仕様 §4 T5).

``anomaly.csv`` / ``anomaly_threshold.csv`` / ``anomaly_timeline.csv`` /
``anomaly_protocol.csv`` / ``anomaly_size.csv`` / 図5枚 / ``meta.json`` を
ここで一括生成する。CLI (``main.py --experiment 05`` と
``experiments/05_anomaly_detection/run_05.py``) はこの関数を呼ぶだけの薄い層に
して、「どのコマンドから走らせても同じ成果物が出る」を構造で保証する
(01 の ``pipeline.py`` / 02 の ``esp_pipeline.py`` / 03 の
``capacity_pipeline.py`` / 04 の ``freerun_pipeline.py`` と同じ規律)。

**系列源は1回だけ組み立てる** (``build_sources``)。5-A / 5-C / 5-D の3つへ
同じ辞書を値で配るので、掃引だけ別の源を掴む経路が存在しない。

**作図層の import は関数本体の中で行う** (D-53)。``experiment`` 配下が
``plotting`` を module-level で import すると循環が復活し、
``tests/test_layer_boundaries.py`` の AST guard と subprocess guard の両方が
落ちる。

``meta.json`` には仕様 §5 の受け入れ条件を**成果物だけで**判定できる材料を
載せる: 区間ごとの実測時間 (``wall_time_breakdown``)、前処理と基準行の一意性
(条件1)、``f1_test_optimal - f1_calibrated`` の分布 (条件3)、5-C の順位入替の
集計 (条件4)、5-D の劣化点 (条件5)。
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rc_basics_lab.config import Anomaly05Config
from rc_basics_lab.experiment.anomaly import (
    AnomalyCondition,
    AnomalyResults,
    run_anomaly_headline,
    run_anomaly_replicate,
)
from rc_basics_lab.experiment.anomaly_ranking import (
    CONTROL_SIGN_TEST_ALPHA,
    aggregate_methods,
)
from rc_basics_lab.experiment.anomaly_rows import (
    ANOMALY_CSV,
    ANOMALY_PROTOCOL_CSV,
    ANOMALY_PROTOCOL_CSV_COLUMNS,
    ANOMALY_SIZE_CSV,
    ANOMALY_SIZE_CSV_COLUMNS,
    ANOMALY_THRESHOLD_CSV,
    ANOMALY_THRESHOLD_CSV_COLUMNS,
    ANOMALY_TIMELINE_CSV,
    ANOMALY_TIMELINE_CSV_COLUMNS,
    AnomalyRow,
    ProtocolSweepRow,
    SizeSweepRow,
    ThresholdSweepRow,
    TimelineRow,
    anomaly_csv_columns,
    rows_as_dicts,
)
from rc_basics_lab.experiment.anomaly_score import ANOMALY_METHODS
from rc_basics_lab.experiment.anomaly_sources import build_sources
from rc_basics_lab.experiment.anomaly_sweep import (
    run_protocol_sweep,
    run_size_sweep,
    summarize_protocol_sweep,
    summarize_size_sweep,
)
from rc_basics_lab.experiment.anomaly_threshold import calibrate_threshold
from rc_basics_lab.experiment.report import (
    META_JSON,
    DataclassSummaryMixin,
    write_meta_for,
    write_rows_csv,
)
from rc_basics_lab.tasks.anomaly import SeriesSource
from rc_basics_lab.types import BoolArray, FloatArray

logger = logging.getLogger(__name__)

FIG_PR_CURVES = "fig_pr_curves.png"
FIG_SCORE_TIMELINE = "fig_score_timeline.png"
FIG_THRESHOLD_TRADEOFF = "fig_threshold_tradeoff.png"
FIG_PROTOCOL_SENSITIVITY = "fig_protocol_sensitivity.png"
FIG_SIZE_VS_PERFORMANCE = "fig_size_vs_performance.png"

ANOMALY_ARTIFACTS: tuple[str, ...] = (
    ANOMALY_CSV,
    ANOMALY_THRESHOLD_CSV,
    ANOMALY_TIMELINE_CSV,
    ANOMALY_PROTOCOL_CSV,
    ANOMALY_SIZE_CSV,
    FIG_PR_CURVES,
    FIG_SCORE_TIMELINE,
    FIG_THRESHOLD_TRADEOFF,
    FIG_PROTOCOL_SENSITIVITY,
    FIG_SIZE_VS_PERFORMANCE,
    META_JSON,
)
"""1コマンド (``make figures-05``) で必ず出る 05 の成果物。

並びは 02〜04 と同じく「CSV -> 図 -> meta.json」で、
``run_and_report_anomaly`` が返す ``paths`` の順序と一致する。宣言と実体が
食い違ったら落ちるテストがこの並びを見る。

CSV が5枚あるのは、**図が成果物 CSV の行だけを読む** (仕様 §5 禁止する構造7)
ためである。5-A / 5-B / 5-C / 5-D の4枚に加えて、時系列そのものを描く
``fig_score_timeline`` の入力として ``anomaly_timeline.csv`` が要る
(04 の ``freerun_profile.csv`` と同じ役割)。
"""

TIMELINE_MAX_POINTS = 3000
"""``anomaly_timeline.csv`` に残すテスト区間の点数の上限 (系統あたり)。

**設定の葉にしない** —— 図の見た目を決めるだけで結論を1ビットも動かさない
(``SPLIT_OFFSET_DIVISOR`` / ``DEGRADATION_FRACTION`` と同じ判断)。間引きは
等間隔で行い、**元の行 index を保つ** ので、図の横軸は系列上の位置のまま
である。
"""


@dataclass(frozen=True, slots=True)
class SectionTiming(DataclassSummaryMixin):
    """区間ごとの実測時間 (``meta.json`` の ``wall_time_breakdown``)。

    仕様 §5 が区間ごとに予算を切っているので、成果物だけで「どの区間が予算を
    割ったか」を判定できる形にする。

    **5-A と 5-B は分けて測れない**: ``run_anomaly_headline`` が1回の評価の中で
    ``anomaly.csv`` の行と 5-B の掃引行の両方を作るためで、分けて測るには
    実験層の内側に計測を仕込むことになる。合算して ``headline_s`` に入れ、
    予算も合算 (400 + 60 = 460 秒) で見る。

    Attributes:
        headline_s: 5-A + 5-B (予算 < 460 秒)。
        timeline_s: 図に出す1例の再実行 (予算 < 30 秒)。
        protocol_s: 5-C (予算 < 250 秒)。
        size_s: 5-D (予算 < 150 秒)。
        figures_s: 図5枚 + CSV5枚の書き出し (予算 < 20 秒)。
    """

    headline_s: float
    timeline_s: float
    protocol_s: float
    size_s: float
    figures_s: float


SECTION_BUDGETS_S: Mapping[str, float] = {
    "headline_s": 460.0,
    "timeline_s": 30.0,
    "protocol_s": 250.0,
    "size_s": 150.0,
    "figures_s": 20.0,
}
"""区間ごとの予算 [秒] (仕様 §5)。``meta.json`` に載せて成果物だけで判定できる
ようにする。合計は ``TOTAL_BUDGET_S`` と別に持たない (下の総和が正本)。"""

TOTAL_BUDGET_S = 900.0
"""``make figures-05`` 全体の予算 [秒] (仕様 §3 のハード制約)。"""


@dataclass(frozen=True, slots=True)
class AnomalyOutputs:
    """``run_and_report_anomaly`` の成果物。

    Attributes:
        headline: 5-A / 5-B の結果 (``anomaly.csv`` / ``anomaly_threshold.csv``)。
        timeline_rows: 図に出す1例 (``anomaly_timeline.csv``)。
        protocol_rows: 5-C の行 (``anomaly_protocol.csv``)。
        size_rows: 5-D の行 (``anomaly_size.csv``)。
        timing: 区間ごとの実測時間。
        paths: 生成したファイル (``ANOMALY_ARTIFACTS`` と同じ並び)。
        wall_time_s: 実測 wall time (図と書き出しを含む)。
    """

    headline: AnomalyResults
    timeline_rows: tuple[TimelineRow, ...]
    protocol_rows: tuple[ProtocolSweepRow, ...]
    size_rows: tuple[SizeSweepRow, ...]
    timing: SectionTiming
    paths: tuple[Path, ...]
    wall_time_s: float

    @property
    def rows(self) -> tuple[AnomalyRow, ...]:
        """``anomaly.csv`` と同じ行。"""
        return self.headline.rows

    @property
    def threshold_rows(self) -> tuple[ThresholdSweepRow, ...]:
        """``anomaly_threshold.csv`` と同じ行。"""
        return self.headline.threshold_rows


def _thinning_stride(n_points: int) -> int:
    """テスト区間を ``TIMELINE_MAX_POINTS`` 点以下に落とす等間隔の間引き幅。"""
    return max(1, math.ceil(n_points / TIMELINE_MAX_POINTS))


def build_timeline_rows(
    config: Anomaly05Config, sources: Mapping[str, SeriesSource]
) -> tuple[TimelineRow, ...]:
    """図に出す1例のテスト区間を行にする (D-82)。

    条件は **5-A の (先頭の使える系列, レプリケート 0)** そのものである。
    シードは ``AnomalyCondition.draw`` だけで決まるので、ここで作る系列・
    前処理・スコアは ``anomaly.csv`` の対応行とビット単位で同じものになる ——
    図のためだけの別条件を作らないための構造 (``threshold`` 列が一致すること
    を ``tests/test_anomaly_pipeline.py`` が実測する)。

    Args:
        config: 実験設定。
        sources: 系列名 -> 系列源 (5-A と**同じ辞書**を渡すこと)。

    Returns:
        6系統 x 間引き後のテスト区間の行 (``ANOMALY_METHODS`` の順)。

    Raises:
        ValueError: 使える源が1本も無い場合、または間引いた結果として異常点が
            1点も残らない場合 (正解ラベルを重ね描きできない図になる)。
    """
    available = [
        (name, source) for name, source in sources.items() if source.is_available()
    ]
    if not available:
        raise ValueError(
            "使える系列源が1本もありません (make data-05 を実行しましたか)"
        )
    name, source = available[0]
    outcome = run_anomaly_replicate(
        config,
        source,
        AnomalyCondition(
            series=name,
            series_index=0,
            replicate=0,
            n_replicates=config.reservoir.n_replicates,
        ),
    )
    plan = outcome.plan
    test = plan.split.test
    stride = _thinning_stride(len(test))
    indices = range(test.start, test.stop, stride)
    labels: BoolArray = plan.series.labels
    ignore: BoolArray = plan.series.ignore
    if not any(bool(labels[index]) for index in indices):
        raise ValueError(
            "間引いたテスト区間に異常点が1つも残りません "
            f"(系列 {name}、テスト {len(test)} 点、上限 {TIMELINE_MAX_POINTS} 点)"
        )
    # **閾値は 5-A の行が持っている値をそのまま使う** (D-82)。ここで
    # calibrate_threshold を呼び直すと、較正区間の切り出し規則が実験層と
    # 作図用で2実装に割れる (D-57 と同じ壊れ方)。
    thresholds = {row.method: row.threshold for row in outcome.rows}
    rows: list[TimelineRow] = []
    for method in ANOMALY_METHODS:
        scores: FloatArray = plan.scores[method].values
        rows.extend(
            TimelineRow(
                dataset=config.dataset.source,
                series=name,
                method=method,
                replicate=0,
                index=index,
                score=float(scores[index]),
                is_anomaly=bool(labels[index]),
                is_ignored=bool(ignore[index]),
                threshold=thresholds[method],
            )
            for index in indices
        )
    logger.info(
        "timeline: series=%s replicate=0 test=[%d, %d) stride=%d rows=%d",
        name,
        test.start,
        test.stop,
        stride,
        len(rows),
    )
    return tuple(rows)


def write_anomaly_csv(
    config: Anomaly05Config, rows: Sequence[AnomalyRow], path: Path
) -> Path:
    """``anomaly.csv`` を書く (**列が設定で増減する**ので専用の書き出し)。

    ``report.write_rows_csv`` は ``dataclasses.asdict`` で固定列の行を書く
    ヘルパーなので、``f1_test_optimal`` と PA%K の列が増減する 05 の主 CSV は
    ここで書く。列順の単一の真実は ``anomaly_csv_columns`` のままで、値の
    取り出しも ``anomaly_row_as_dict`` に閉じている (D-55)。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(anomaly_csv_columns(config)))
        writer.writeheader()
        writer.writerows(rows_as_dicts(rows))
    return path


def _log_timing(timing: SectionTiming, wall_time_s: float) -> None:
    """区間ごとの実測時間を**数値として**ログに残す (仕様 §5 の性能予算)。"""
    logger.info(
        "05 の区間別 wall time: 5-A+5-B %.1fs / timeline %.1fs / 5-C %.1fs / "
        "5-D %.1fs / 図 %.1fs / 合計 %.1fs (予算 %.0fs)",
        timing.headline_s,
        timing.timeline_s,
        timing.protocol_s,
        timing.size_s,
        timing.figures_s,
        wall_time_s,
        TOTAL_BUDGET_S,
    )


def run_and_report_anomaly(config: Anomaly05Config, out_dir: Path) -> AnomalyOutputs:
    """実験 5-A / 5-B / 5-C / 5-D を実行し、CSV5枚・図5枚・meta.json を書く。

    Args:
        config: 05 の実験設定。
        out_dir: 出力ディレクトリ (無ければ作る)。

    Returns:
        生成した行・区間ごとの実測時間・ファイルパス・実測 wall time。

    Raises:
        ValueError: 使える系列源が無い、格子が基準条件を含まない (D-79)、
            5-D の学習量が揃っていない (D-80) などの場合。
    """
    # 作図層の import を関数本体に置くのは D-53 (循環 import の解消)。
    # 先頭へ戻すと tests/test_layer_boundaries.py の AST guard が落ちる。
    from rc_basics_lab.plotting.figures_anomaly import (
        plot_pr_curves,
        plot_protocol_sensitivity,
        plot_score_timeline,
        plot_size_vs_performance,
        plot_threshold_tradeoff,
    )
    from rc_basics_lab.plotting.style import setup_style

    started = time.perf_counter()
    # 源は1回だけ組み立て、5-A / 5-C / 5-D へ同じ辞書を配る。
    sources = build_sources(config)

    headline_started = time.perf_counter()
    headline = run_anomaly_headline(config, sources)
    headline_s = time.perf_counter() - headline_started

    timeline_started = time.perf_counter()
    timeline_rows = build_timeline_rows(config, sources)
    timeline_s = time.perf_counter() - timeline_started

    protocol_started = time.perf_counter()
    protocol_rows = run_protocol_sweep(config, sources)
    protocol_s = time.perf_counter() - protocol_started

    size_started = time.perf_counter()
    size_rows = run_size_sweep(config, sources)
    size_s = time.perf_counter() - size_started

    figures_started = time.perf_counter()
    style = setup_style()
    paths = (
        write_rows_csv(
            rows_as_dicts(headline.rows),  # type: ignore[arg-type]
            out_dir / ANOMALY_CSV,
            anomaly_csv_columns(config),
        ),
        write_rows_csv(
            headline.threshold_rows,
            out_dir / ANOMALY_THRESHOLD_CSV,
            ANOMALY_THRESHOLD_CSV_COLUMNS,
        ),
        write_rows_csv(
            timeline_rows, out_dir / ANOMALY_TIMELINE_CSV, ANOMALY_TIMELINE_CSV_COLUMNS
        ),
        write_rows_csv(
            protocol_rows, out_dir / ANOMALY_PROTOCOL_CSV, ANOMALY_PROTOCOL_CSV_COLUMNS
        ),
        write_rows_csv(size_rows, out_dir / ANOMALY_SIZE_CSV, ANOMALY_SIZE_CSV_COLUMNS),
        plot_pr_curves(
            headline.rows,
            headline.threshold_rows,
            out_dir / FIG_PR_CURVES,
            style=style,
        ),
        plot_score_timeline(timeline_rows, out_dir / FIG_SCORE_TIMELINE, style=style),
        plot_threshold_tradeoff(
            headline.rows,
            headline.threshold_rows,
            out_dir / FIG_THRESHOLD_TRADEOFF,
            style=style,
        ),
        plot_protocol_sensitivity(
            protocol_rows, out_dir / FIG_PROTOCOL_SENSITIVITY, style=style
        ),
        plot_size_vs_performance(
            size_rows, out_dir / FIG_SIZE_VS_PERFORMANCE, style=style
        ),
    )
    figures_s = time.perf_counter() - figures_started

    timing = SectionTiming(
        headline_s=headline_s,
        timeline_s=timeline_s,
        protocol_s=protocol_s,
        size_s=size_s,
        figures_s=figures_s,
    )
    wall_time_s = time.perf_counter() - started
    _log_timing(timing, wall_time_s)
    meta_path = write_meta_for(
        config,
        config.seeds,
        wall_time_s,
        len(headline.rows),
        out_dir / META_JSON,
        extra=_meta_extra(
            headline, timeline_rows, protocol_rows, size_rows, timing, style.cjk_font
        ),
    )
    logger.info(
        "05 の成果物を書きました: %s (5-A %d 行 / 5-B %d 行 / timeline %d 行 / "
        "5-C %d 行 / 5-D %d 行, wall_time=%.1fs)",
        [path.name for path in (*paths, meta_path)],
        len(headline.rows),
        len(headline.threshold_rows),
        len(timeline_rows),
        len(protocol_rows),
        len(size_rows),
        wall_time_s,
    )
    return AnomalyOutputs(
        headline=headline,
        timeline_rows=timeline_rows,
        protocol_rows=protocol_rows,
        size_rows=size_rows,
        timing=timing,
        paths=(*paths, meta_path),
        wall_time_s=wall_time_s,
    )


def preprocessor_uniqueness(rows: Sequence[AnomalyRow]) -> dict[str, int]:
    """(系列, レプリケート) 内の ``preprocessor_id`` / ``t0`` の異なり数の最大。

    仕様 §5 の受け入れ条件1 (「同一前処理・同一行で比較している」) を
    **成果物だけで**判定するための量。6系統が同じ前処理を通っていれば
    どちらも 1 になる (D-05 / D-57)。
    """
    grouped: dict[tuple[str, int], tuple[set[str], set[int]]] = {}
    for row in rows:
        ids, offsets = grouped.setdefault((row.series, row.replicate), (set(), set()))
        ids.add(row.preprocessor_id)
        offsets.add(row.t0)
    return {
        "max_distinct_preprocessor_ids": max(len(ids) for ids, _ in grouped.values()),
        "max_distinct_t0": max(len(offsets) for _, offsets in grouped.values()),
        "n_groups": len(grouped),
    }


def f1_gap_summary(rows: Sequence[AnomalyRow]) -> dict[str, float]:
    """``f1_test_optimal - f1_calibrated`` の分布 (仕様 §5 の受け入れ条件3)。

    「テスト側で閾値を選ぶとどれだけ良く見えるか」の一次資料。報告しない設定
    (``threshold.report_test_optimal=False``) では ``nan`` が並ぶので、
    件数 0 の要約を返す。
    """
    gaps = [
        row.f1_test_optimal - row.f1_calibrated
        for row in rows
        if not math.isnan(row.f1_test_optimal)
    ]
    if not gaps:
        return {"n": 0.0}
    values = np.asarray(gaps, dtype=np.float64)
    return {
        "n": float(values.size),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n_negative": float(np.count_nonzero(values < 0.0)),
    }


def _meta_extra(
    headline: AnomalyResults,
    timeline_rows: Sequence[TimelineRow],
    protocol_rows: Sequence[ProtocolSweepRow],
    size_rows: Sequence[SizeSweepRow],
    timing: SectionTiming,
    cjk_font: str | None,
) -> dict[str, object]:
    """``meta.json`` の追加項目 (**受け入れ条件の一次資料をここに集める**)。"""
    aggregates = aggregate_methods(headline.rows)
    return {
        "wall_time_breakdown": timing.to_summary(),
        "wall_time_budget_s": dict(SECTION_BUDGETS_S),
        "total_budget_s": TOTAL_BUDGET_S,
        "n_threshold_rows": len(headline.threshold_rows),
        "n_timeline_rows": len(timeline_rows),
        "n_protocol_rows": len(protocol_rows),
        "n_size_rows": len(size_rows),
        # 受け入れ条件1 / D-05 / D-57: 同一前処理・同一行で比較していること。
        "preprocessor_uniqueness": preprocessor_uniqueness(headline.rows),
        # 受け入れ条件3 / D-56: テスト側最適化との差の分布。
        "f1_gap": f1_gap_summary(headline.rows),
        # 5-A の集計 (README の数値表の出どころ。印の根拠列も含む、D-78)。
        "headline_auprc": {
            method: aggregates[method].to_summary() for method in ANOMALY_METHODS
        },
        "anomaly_rate_mean": float(
            np.mean([row.anomaly_rate for row in headline.rows])
        ),
        "control_sign_test_alpha": CONTROL_SIGN_TEST_ALPHA,
        # 受け入れ条件4 / D-78: 順位入替の集計 (行から計算する)。
        "protocol_summary": summarize_protocol_sweep(protocol_rows).to_summary(),
        # 受け入れ条件5 / D-80: 劣化点と、格子の端が選ばれたかの区別。
        "size_summary": summarize_size_sweep(size_rows).to_summary(),
        # 図のラベル言語を決めた要因 (02〜04 の meta.json と同じ形)。
        "cjk_font": cjk_font,
    }


__all__ = [
    "ANOMALY_ARTIFACTS",
    "FIG_PROTOCOL_SENSITIVITY",
    "FIG_PR_CURVES",
    "FIG_SCORE_TIMELINE",
    "FIG_SIZE_VS_PERFORMANCE",
    "FIG_THRESHOLD_TRADEOFF",
    "SECTION_BUDGETS_S",
    "TIMELINE_MAX_POINTS",
    "TOTAL_BUDGET_S",
    "AnomalyOutputs",
    "SectionTiming",
    "build_timeline_rows",
    "f1_gap_summary",
    "preprocessor_uniqueness",
    "preprocessor_uniqueness",
    "run_and_report_anomaly",
]

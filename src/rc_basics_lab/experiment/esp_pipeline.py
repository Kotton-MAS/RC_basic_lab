"""1コマンドで 02 の7成果物を作る経路 (受け入れ条件7).

``esp_diagnostics.csv`` / ``washout_sensitivity.csv`` / ``fig_esp_decay.png`` /
``fig_leak_timescale.png`` / ``fig_esp_map.png`` /
``fig_washout_sensitivity.png`` / ``meta.json`` をここで一括生成する。CLI
(``main.py --experiment 02`` と ``experiments/02_esp_and_dynamics/run.py``) は
この関数を呼ぶだけの薄い層にして、「どのコマンドから走らせても同じ成果物が
出る」を構造で保証する (01 の ``pipeline.py`` と同じ規律)。

``meta.json`` には ``esp_defaults`` (コード側にしか無い固定値) と
``verdict_lyapunov_agreement`` (λ の符号と ESP 判定の整合の内訳) と
``washout_sensitivity`` (2-D の変動幅) を載せる。
``verdict_lyapunov_agreement`` は「λ<0 なのに非収束」がどこで起きたかを
sigma_u と rho の分布まで残すので、記事で多安定性を説明するときの一次資料になる。
``washout_sensitivity`` は受け入れ条件5 (「washout 長の性能変動が定量化されて
いる」) の一次資料で、変動幅そのものに加えて**行数が格子全体で一定だったか**
(D-19 の補償が効いた実行か) も残す。
"""

from __future__ import annotations

import csv
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rc_basics_lab.config import Esp02Config
from rc_basics_lab.experiment.esp import (
    ESP_CSV_COLUMNS,
    EspResults,
    EspRow,
    VerdictAgreement,
    esp_defaults,
    run_esp_experiment,
    summarize_verdict_agreement,
)
from rc_basics_lab.experiment.report import META_JSON, write_meta_for, write_rows_csv
from rc_basics_lab.experiment.threshold import (
    ThresholdRow,
    run_threshold_sweep,
    threshold_csv_columns,
    threshold_row_as_dict,
)
from rc_basics_lab.experiment.washout import (
    WASHOUT_CSV_COLUMNS,
    WashoutRow,
    WashoutSensitivity,
    run_washout_sweep,
    summarize_washout_sensitivity,
)

logger = logging.getLogger(__name__)

ESP_DIAGNOSTICS_CSV = "esp_diagnostics.csv"
WASHOUT_SENSITIVITY_CSV = "washout_sensitivity.csv"
FIG_ESP_DECAY = "fig_esp_decay.png"
FIG_LEAK_TIMESCALE = "fig_leak_timescale.png"
FIG_ESP_MAP = "fig_esp_map.png"
FIG_WASHOUT_SENSITIVITY = "fig_washout_sensitivity.png"
ESP_THRESHOLD_SENSITIVITY_CSV = "esp_threshold_sensitivity.csv"

ESP_ARTIFACTS: tuple[str, ...] = (
    ESP_DIAGNOSTICS_CSV,
    WASHOUT_SENSITIVITY_CSV,
    FIG_ESP_DECAY,
    FIG_LEAK_TIMESCALE,
    FIG_ESP_MAP,
    FIG_WASHOUT_SENSITIVITY,
    META_JSON,
)
"""1コマンドで必ず出る 02 の成果物 (図4枚 + CSV2枚 + meta.json)。

2-D の行は 2-A/2-B/2-C と列が異なるため CSV を分ける (仕様 §3 ソフト制約:
要件書の6成果物 +1)。1枚にまとめると、どちらかの列が空欄だらけになるか、
``EspRow`` と ``WashoutRow`` のどちらかの宣言順が CSV 列順の単一の真実で
なくなる。

``esp_threshold_sensitivity.csv`` (D-16 の閾値感度) は**この並びに入れない**。
記事に載る成果物ではなく「既定値が結論を作っていないことの根拠」であり、
2-C の格子をもう一度回すぶん実行時間が倍近くになる。``--threshold-sweep``
(= ``make threshold-02``) で明示的に再生成する。
"""


@dataclass(frozen=True, slots=True)
class EspOutputs:
    """``run_and_report_esp`` の成果物。

    Attributes:
        results: 3実験ぶんの条件別の結果 (行 + 図が使う曲線)。
        agreement: λ の符号と ESP 判定の整合の要約。
        washout_rows: 2-D の長形式の行 (``washout_sensitivity.csv`` と同じ)。
        sensitivity: 2-D の変動幅の要約。
        paths: 生成したファイル (``ESP_ARTIFACTS`` と同じ並び)。
        wall_time_s: 計算部分の実測 wall time (図の書き出しは含まない)。
    """

    results: EspResults
    agreement: VerdictAgreement
    washout_rows: tuple[WashoutRow, ...]
    sensitivity: WashoutSensitivity
    paths: tuple[Path, ...]
    wall_time_s: float

    @property
    def rows(self) -> tuple[EspRow, ...]:
        """``esp_diagnostics.csv`` と同じ行 (2-A/2-B/2-C)。"""
        return self.results.rows


def write_esp_csv(rows: Sequence[EspRow], path: Path) -> Path:
    """条件ごとの診断結果を CSV に書く (列順は ``EspRow`` の宣言順)。"""
    return write_rows_csv(rows, path, ESP_CSV_COLUMNS)


def write_washout_csv(rows: Sequence[WashoutRow], path: Path) -> Path:
    """2-D の結果を CSV に書く (列順は ``WashoutRow`` の宣言順)。"""
    return write_rows_csv(rows, path, WASHOUT_CSV_COLUMNS)


def write_threshold_csv(
    rows: Sequence[ThresholdRow], sigma_grid: Sequence[float], path: Path
) -> Path:
    """閾値感度の結果を CSV に書く (列順は ``threshold_csv_columns``)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(threshold_csv_columns(sigma_grid))
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(threshold_row_as_dict(row))
    return path


def run_and_report_threshold_sweep(config: Esp02Config, out_dir: Path) -> Path:
    """D-16 の閾値感度を測り ``esp_threshold_sensitivity.csv`` に書く。

    本体の7成果物とは独立に走る (``ESP_ARTIFACTS`` に含めない理由は上の
    docstring)。``docs/design.md`` §9 の感度表はこの CSV と行数が一致する
    ことを ``tests/test_design_doc.py`` が固定している。
    """
    started = time.perf_counter()
    rows = run_threshold_sweep(config)
    path = write_threshold_csv(
        rows, config.esp_map.sigma_grid, out_dir / ESP_THRESHOLD_SENSITIVITY_CSV
    )
    logger.info(
        "閾値感度: %d 行 / 基準からずれた臨界 rho は %d 件 / wall_time=%.2fs / 出力=%s",
        len(rows),
        sum(row.n_sigma_shifted for row in rows),
        time.perf_counter() - started,
        path,
    )
    return path


def _log_agreement(agreement: VerdictAgreement) -> None:
    """λ の符号と ESP 判定の整合を**数値として**ログに残す (受け入れ条件3)。"""
    logger.info(
        "λ と ESP 判定の整合: 比較対象=%d 件 (境界近傍 |λ|<=%.3g を %d 件除外) / "
        "偽の ESP (λ>0 かつ収束)=%d 件 / 局所安定だが非収束 (λ<0)=%d 件 / "
        "強駆動 (sigma_u>=%.2g) では %d 件中 %d 件の不一致",
        agreement.n_compared,
        agreement.boundary_lambda,
        agreement.n_near_boundary,
        agreement.n_false_esp,
        agreement.n_local_but_not_global,
        agreement.strong_drive_sigma,
        agreement.n_compared_strong_drive,
        agreement.n_disagreement_strong_drive,
    )


def _log_sensitivity(sensitivity: WashoutSensitivity) -> None:
    """2-D の変動幅を**数値として**ログに残す (受け入れ条件5)。"""
    headline = sensitivity.headline
    logger.info(
        "2-D washout 感度 (%s x %s): 変動幅 (最大/最小) = %.4f 倍 "
        "(%.4g @ washout=%d .. %.4g @ washout=%d) / "
        "レプリケート間 s.d. 最大 %.4g -> %s / "
        "pad_series=%s (行数一定=%s)",
        headline.task,
        headline.method,
        headline.ratio,
        headline.nrmse_min,
        headline.washout_at_min,
        headline.nrmse_max,
        headline.washout_at_max,
        headline.replicate_std_max,
        "ばらつきを超える"
        if headline.exceeds_replicate_noise
        else "ばらつき以下 (washout に反応したとは言えない)",
        sensitivity.pad_series,
        sensitivity.training_size_is_constant,
    )


def run_and_report_esp(config: Esp02Config, out_dir: Path) -> EspOutputs:
    """実験 2-A / 2-B / 2-C / 2-D を実行し、CSV2枚・図4枚・meta.json を書き出す。

    Args:
        config: 02 の実験設定。
        out_dir: 出力ディレクトリ (無ければ作る)。

    Returns:
        生成した結果・整合の要約・2-D の行と要約・ファイルパス・実測 wall time。
    """
    # 作図層の import を関数本体に置くのは D-53 (循環 import の解消)。
    # 先頭へ戻すと tests/test_layer_boundaries.py の AST guard と
    # subprocess guard の両方が落ちる。
    from rc_basics_lab.meta import git_commit
    from rc_basics_lab.plotting.figures_esp import (
        plot_esp_decay,
        plot_esp_map,
    )
    from rc_basics_lab.plotting.figures_leak import plot_leak_timescale
    from rc_basics_lab.plotting.figures_washout import plot_washout_sensitivity
    from rc_basics_lab.plotting.style import setup_style

    started = time.perf_counter()
    results = run_esp_experiment(config)
    washout_rows = run_washout_sweep(config)
    wall_time_s = time.perf_counter() - started
    rows = results.rows
    agreement = summarize_verdict_agreement(rows)
    sensitivity = summarize_washout_sensitivity(config, washout_rows)
    _log_agreement(agreement)
    _log_sensitivity(sensitivity)

    # commit は meta.json と図の footnote (FIG-6 / D-87) で同じ値を使う。
    style = setup_style(commit=git_commit())
    paths = (
        write_esp_csv(rows, out_dir / ESP_DIAGNOSTICS_CSV),
        write_washout_csv(washout_rows, out_dir / WASHOUT_SENSITIVITY_CSV),
        plot_esp_decay(results.decay, out_dir / FIG_ESP_DECAY, style=style),
        plot_leak_timescale(
            results.timescale, out_dir / FIG_LEAK_TIMESCALE, style=style
        ),
        plot_esp_map(
            tuple(outcome.row for outcome in results.esp_map),
            out_dir / FIG_ESP_MAP,
            style=style,
        ),
        plot_washout_sensitivity(
            washout_rows,
            out_dir / FIG_WASHOUT_SENSITIVITY,
            style=style,
            sensitivity=sensitivity,
        ),
        write_meta_for(
            config,
            config.seeds,
            wall_time_s,
            # n_rows は esp_diagnostics.csv の行数。2-D は列が違う別 CSV なので
            # 足し込まず washout_sensitivity.n_rows に分けて残す (足すと
            # 「どちらの CSV の行数か」が meta.json から読めなくなる)。
            len(rows),
            out_dir / META_JSON,
            extra={
                "esp_defaults": esp_defaults(config),
                "verdict_lyapunov_agreement": agreement.to_summary(),
                "washout_sensitivity": sensitivity.to_summary(),
                "cjk_font": style.cjk_font,
            },
        ),
    )
    logger.info(
        "完了: %d 行 (2-A/2-B/2-C) + %d 行 (2-D) / wall_time=%.2fs / 出力=%s",
        len(rows),
        len(washout_rows),
        wall_time_s,
        ", ".join(str(path) for path in paths),
    )
    return EspOutputs(
        results=results,
        agreement=agreement,
        washout_rows=washout_rows,
        sensitivity=sensitivity,
        paths=paths,
        wall_time_s=wall_time_s,
    )


__all__ = [
    "ESP_ARTIFACTS",
    "ESP_DIAGNOSTICS_CSV",
    "ESP_THRESHOLD_SENSITIVITY_CSV",
    "FIG_ESP_DECAY",
    "FIG_ESP_MAP",
    "FIG_LEAK_TIMESCALE",
    "FIG_WASHOUT_SENSITIVITY",
    "WASHOUT_SENSITIVITY_CSV",
    "EspOutputs",
    "run_and_report_esp",
    "run_and_report_threshold_sweep",
    "write_esp_csv",
    "write_threshold_csv",
    "write_washout_csv",
]

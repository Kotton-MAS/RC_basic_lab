"""1コマンドで 02 の5成果物を作る経路 (受け入れ条件7 の T3 ぶん).

``esp_diagnostics.csv`` / ``fig_esp_decay.png`` / ``fig_leak_timescale.png`` /
``fig_esp_map.png`` / ``meta.json`` をここで一括生成する。CLI
(``main.py --experiment 02`` と ``experiments/02_esp_and_dynamics/run.py``) は
この関数を呼ぶだけの薄い層にして、「どのコマンドから走らせても同じ成果物が
出る」を構造で保証する (01 の ``pipeline.py`` と同じ規律)。

``meta.json`` には ``esp_defaults`` (コード側にしか無い固定値) と
``verdict_lyapunov_agreement`` (λ の符号と ESP 判定の整合の内訳) を載せる。
後者は「λ<0 なのに非収束」がどこで起きたかを σ_u と rho の分布まで残すので、
記事で多安定性を説明するときの一次資料になる。
"""

from __future__ import annotations

import csv
import dataclasses
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
from rc_basics_lab.experiment.report import META_JSON, write_meta_for
from rc_basics_lab.plotting.figures_esp import (
    plot_esp_decay,
    plot_esp_map,
    plot_leak_timescale,
)
from rc_basics_lab.plotting.style import setup_style

logger = logging.getLogger(__name__)

ESP_DIAGNOSTICS_CSV = "esp_diagnostics.csv"
FIG_ESP_DECAY = "fig_esp_decay.png"
FIG_LEAK_TIMESCALE = "fig_leak_timescale.png"
FIG_ESP_MAP = "fig_esp_map.png"

ESP_ARTIFACTS: tuple[str, ...] = (
    ESP_DIAGNOSTICS_CSV,
    FIG_ESP_DECAY,
    FIG_LEAK_TIMESCALE,
    FIG_ESP_MAP,
    META_JSON,
)
"""1コマンドで必ず出る 02 の成果物 (T3 ぶん)。

2-D (``fig_washout_sensitivity.png`` / ``washout_sensitivity.csv``) は T4 で
この並びに加わる。
"""


@dataclass(frozen=True, slots=True)
class EspOutputs:
    """``run_and_report_esp`` の成果物。

    Attributes:
        results: 3実験ぶんの条件別の結果 (行 + 図が使う曲線)。
        agreement: λ の符号と ESP 判定の整合の要約。
        paths: 生成したファイル (``ESP_ARTIFACTS`` と同じ並び)。
        wall_time_s: 計算部分の実測 wall time (図の書き出しは含まない)。
    """

    results: EspResults
    agreement: VerdictAgreement
    paths: tuple[Path, ...]
    wall_time_s: float

    @property
    def rows(self) -> tuple[EspRow, ...]:
        """``esp_diagnostics.csv`` と同じ行。"""
        return self.results.rows


def write_esp_csv(rows: Sequence[EspRow], path: Path) -> Path:
    """条件ごとの診断結果を CSV に書く (列順は ``EspRow`` の宣言順)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ESP_CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))
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


def run_and_report_esp(config: Esp02Config, out_dir: Path) -> EspOutputs:
    """実験 2-A / 2-B / 2-C を実行し、CSV1枚・図3枚・meta.json を書き出す。

    Args:
        config: 02 の実験設定。
        out_dir: 出力ディレクトリ (無ければ作る)。

    Returns:
        生成した結果・整合の要約・ファイルパス・実測 wall time。
    """
    started = time.perf_counter()
    results = run_esp_experiment(config)
    wall_time_s = time.perf_counter() - started
    rows = results.rows
    agreement = summarize_verdict_agreement(rows)
    _log_agreement(agreement)

    style = setup_style()
    paths = (
        write_esp_csv(rows, out_dir / ESP_DIAGNOSTICS_CSV),
        plot_esp_decay(results.decay, out_dir / FIG_ESP_DECAY, style=style),
        plot_leak_timescale(
            results.timescale, out_dir / FIG_LEAK_TIMESCALE, style=style
        ),
        plot_esp_map(
            tuple(outcome.row for outcome in results.esp_map),
            out_dir / FIG_ESP_MAP,
            style=style,
        ),
        write_meta_for(
            config,
            config.seeds,
            wall_time_s,
            len(rows),
            out_dir / META_JSON,
            extra={
                "esp_defaults": esp_defaults(config),
                "verdict_lyapunov_agreement": agreement.to_summary(),
                "cjk_font": style.cjk_font,
            },
        ),
    )
    logger.info(
        "完了: %d 行 / wall_time=%.2fs / 出力=%s",
        len(rows),
        wall_time_s,
        ", ".join(str(path) for path in paths),
    )
    return EspOutputs(
        results=results, agreement=agreement, paths=paths, wall_time_s=wall_time_s
    )


__all__ = [
    "ESP_ARTIFACTS",
    "ESP_DIAGNOSTICS_CSV",
    "FIG_ESP_DECAY",
    "FIG_ESP_MAP",
    "FIG_LEAK_TIMESCALE",
    "EspOutputs",
    "run_and_report_esp",
    "write_esp_csv",
]

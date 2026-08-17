"""結果の書き出し — ``comparison.csv`` / ``comparison_summary.csv`` / ``meta.json``.

CSV の列順は ``runner.CSV_COLUMNS`` (= ``ResultRow`` の宣言順) を単一の真実とする。
``meta.json`` は ``meta.collect_meta`` の内容に実測 wall time と行数を足したもの。

``comparison_summary.csv`` は (課題, 手法) ごとの NRMSE 平均±標準偏差と符号正解率
平均を持つ生成物 (F-1-005)。README の手書きの表はこの値であることを
``tests/test_readme_summary.py`` が固定する。
"""

from __future__ import annotations

import csv
import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from rc_basics_lab.config import ExperimentConfig
from rc_basics_lab.experiment.runner import CSV_COLUMNS, ResultRow
from rc_basics_lab.experiment.summary import Aggregate
from rc_basics_lab.meta import collect_meta

COMPARISON_CSV = "comparison.csv"
COMPARISON_SUMMARY_CSV = "comparison_summary.csv"
META_JSON = "meta.json"

SUMMARY_CSV_COLUMNS: tuple[str, ...] = (
    "task",
    "method",
    "n",
    "nrmse_mean",
    "nrmse_std",
    "sign_accuracy_mean",
)
"""``comparison_summary.csv`` の列順。"""


def write_comparison_csv(rows: Sequence[ResultRow], path: Path) -> Path:
    """長形式の結果を CSV に書く。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))
    return path


def write_comparison_summary_csv(
    stats: Mapping[tuple[str, str], Aggregate], path: Path
) -> Path:
    """(課題, 手法) ごとの集計値を CSV に書く (F-1-003 / F-1-005)。

    ``comparison.csv`` (長形式30行) と対になる集計版。集計ロジックそのものは
    ``experiment.summary.aggregate_nrmse`` にある。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_CSV_COLUMNS))
        writer.writeheader()
        for (task, method), aggregate in stats.items():
            writer.writerow(
                {
                    "task": task,
                    "method": method,
                    "n": aggregate.n,
                    "nrmse_mean": aggregate.mean,
                    "nrmse_std": aggregate.std,
                    "sign_accuracy_mean": aggregate.sign_accuracy_mean,
                }
            )
    return path


def write_meta(
    config: ExperimentConfig,
    wall_time_s: float,
    n_rows: int,
    path: Path,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """実行メタ情報を JSON に書く (実測 wall time は性能受け入れ基準の根拠)。

    Args:
        config: 実験設定。
        wall_time_s: 実測 wall time。
        n_rows: ``comparison.csv`` の行数。
        path: 出力先。
        extra: 追加で載せる項目 (実験1-B の ``state_space`` など)。既存キーと
            衝突したら ``ValueError`` (静かな上書きで情報が消えるのを防ぐ)。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, object] = collect_meta(config)
    meta["wall_time_s"] = wall_time_s
    meta["n_rows"] = n_rows
    for key, value in (extra or {}).items():
        if key in meta:
            raise ValueError(f"meta のキーが衝突しています: {key}")
        meta[key] = value
    path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = ["COMPARISON_CSV", "META_JSON", "write_comparison_csv", "write_meta"]

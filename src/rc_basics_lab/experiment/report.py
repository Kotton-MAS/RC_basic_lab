"""結果の書き出し — ``comparison.csv`` と ``meta.json``.

CSV の列順は ``runner.CSV_COLUMNS`` (= ``ResultRow`` の宣言順) を単一の真実とする。
``meta.json`` は ``meta.collect_meta`` の内容に実測 wall time と行数を足したもの。
"""

from __future__ import annotations

import csv
import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from rc_basics_lab.config import ExperimentConfig
from rc_basics_lab.experiment.runner import CSV_COLUMNS, ResultRow
from rc_basics_lab.meta import collect_meta

COMPARISON_CSV = "comparison.csv"
META_JSON = "meta.json"


def write_comparison_csv(rows: Sequence[ResultRow], path: Path) -> Path:
    """長形式の結果を CSV に書く。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))
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

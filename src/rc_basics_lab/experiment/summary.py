"""(課題, 手法) ごとの NRMSE 集計 (受け入れ条件3).

集計ロジックはここに置く。``plotting/figures.py`` は表示のためにこれを import
するだけなので、matplotlib を import しなくても集計できる (F-1-003)。

集計値は ``experiment.report.write_comparison_summary_csv`` で
``results/comparison_summary.csv`` として書き出す。README の手書きの表は
この生成物から引いた値であることを ``tests/test_readme_summary.py`` が固定する
(F-1-005)。
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from rc_basics_lab.experiment.runner import ResultRow


@dataclass(frozen=True, slots=True)
class Aggregate:
    """1 (課題, 手法) の集計値。

    Attributes:
        mean: NRMSE の標本平均。
        std: NRMSE の標本標準偏差 (ddof=1)。レプリケートが1本のときは 0。
        n: レプリケート数。
        sign_accuracy_mean: 符号正解率の標本平均。
    """

    mean: float
    std: float
    n: int
    sign_accuracy_mean: float


def aggregate_nrmse(rows: Sequence[ResultRow]) -> dict[tuple[str, str], Aggregate]:
    """(課題, 手法) ごとの NRMSE の平均・標準偏差・符号正解率平均 (受け入れ条件3)。"""
    grouped: dict[tuple[str, str], list[ResultRow]] = {}
    for row in rows:
        grouped.setdefault((row.task, row.method), []).append(row)
    return {
        key: Aggregate(
            mean=statistics.fmean(row.nrmse for row in group),
            std=statistics.stdev([row.nrmse for row in group])
            if len(group) > 1
            else 0.0,
            n=len(group),
            sign_accuracy_mean=statistics.fmean(row.sign_accuracy for row in group),
        )
        for key, group in grouped.items()
    }


__all__ = ["Aggregate", "aggregate_nrmse"]

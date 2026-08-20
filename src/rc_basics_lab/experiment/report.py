"""結果の書き出し — ``comparison.csv`` / ``comparison_summary.csv`` / ``meta.json``.

CSV の列順は ``runner.CSV_COLUMNS`` (= ``ResultRow`` の宣言順) を単一の真実とする。
``meta.json`` は ``meta.collect_meta`` の内容に実測 wall time と行数を足したもの。

``comparison_summary.csv`` は (課題, 手法) ごとの NRMSE 平均±標準偏差と符号正解率
平均を持つ生成物 (F-1-005)。README の手書きの表はこの値であることを
``tests/test_readme_summary.py`` が固定する。

``write_rows_csv`` は ``mkdir`` -> ``DictWriter`` -> ``asdict`` -> ``writerow`` の
手続きを 9 箇所 (05 削減候補 #4) で共通化した薄いヘルパー。列順・ファイル名・
``encoding`` / ``newline`` の指定はいずれも呼び出し側 (``XXX_CSV_COLUMNS`` 定数)
が単一の真実のまま —— この関数は書き出しの手続きだけをまとめる。

``DataclassSummaryMixin`` は ``to_summary`` が単に ``dataclasses.asdict(self)`` を
返すだけの 6 箇所 (05 削減候補 #9) を共通化する。選別や単位変換を伴う
``to_summary`` (``ThresholdComparison`` など) は対象外で、各クラスに個別実装が
残る。
"""

from __future__ import annotations

import csv
import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TypeVar

from rc_basics_lab.config import ExperimentConfig
from rc_basics_lab.experiment.runner import CSV_COLUMNS, ResultRow
from rc_basics_lab.experiment.summary import Aggregate
from rc_basics_lab.meta import collect_meta_for

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

_RowT = TypeVar("_RowT", bound="DataclassInstance")

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


def write_rows_csv(rows: Sequence[_RowT], path: Path, columns: Sequence[str]) -> Path:
    """dataclass の行列を CSV に書く (05 削減候補 #4 の共通ヘルパー)。

    9 箇所 (``report`` / ``esp_pipeline`` / ``capacity_pipeline`` / ``freerun`` /
    ``stability``) が個別実装していた「``mkdir`` -> ``DictWriter`` -> ``asdict`` ->
    ``writerow``」を1本にまとめる。9箇所とも ``encoding="utf-8"`` /
    ``newline=""`` / ``mkdir(parents=True, exist_ok=True)`` が同一であることを
    実装前に実測で確認済み (異なる箇所があれば呼び出し側に残す)。

    Args:
        rows: 書き出す行 (dataclass インスタンス)。
        path: 出力先。
        columns: 列順 (``XXX_CSV_COLUMNS`` 定数。単一の真実は呼び出し側のまま)。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))
    return path


class DataclassSummaryMixin:
    """``to_summary`` が ``dataclasses.asdict(self)`` を返すだけの実装の共通化.

    05 削減候補 #9。frozen かつ ``slots=True`` な dataclass に混ぜても
    ``__slots__ = ()`` により追加の ``__dict__`` を生やさない。
    ``__dataclass_fields__`` は継承先の ``@dataclass`` が実行時に必ず埋めるので
    ここでは型注釈だけを宣言し、``dataclasses.asdict`` が要求する
    ``DataclassInstance`` プロトコルを ``self`` に満たさせる。
    """

    __slots__: tuple[str, ...] = ()

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[object]]]

    def to_summary(self) -> dict[str, object]:
        """``meta.json`` に載せるプレーンな dict。"""
        return dataclasses.asdict(self)


def write_comparison_csv(rows: Sequence[ResultRow], path: Path) -> Path:
    """長形式の結果を CSV に書く。"""
    return write_rows_csv(rows, path, CSV_COLUMNS)


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


def write_meta_for(
    config: object,
    seeds: object,
    wall_time_s: float,
    n_rows: int,
    path: Path,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """任意の実験設定について実行メタ情報を JSON に書く。

    実験ごとに設定クラスは分かれる (D-13) が、``meta.json`` の書き出し規律
    (実測 wall time と行数を必ず載せる / ``extra`` のキー衝突は ``ValueError``)
    はここ1か所に置く。

    Args:
        config: 実験設定 dataclass。
        seeds: シード設定 dataclass。
        wall_time_s: 実測 wall time。
        n_rows: 出力した CSV の行数。
        path: 出力先。
        extra: 追加で載せる項目 (01 の ``state_space``、02 の
            ``verdict_lyapunov_agreement`` など)。既存キーと衝突したら
            ``ValueError`` (静かな上書きで情報が消えるのを防ぐ)。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, object] = collect_meta_for(config, seeds)
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


def write_meta(
    config: ExperimentConfig,
    wall_time_s: float,
    n_rows: int,
    path: Path,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """実行メタ情報を JSON に書く (実測 wall time は性能受け入れ基準の根拠)。

    ``write_meta_for(config, config.seeds, ...)`` への委譲。既存の呼び出しを
    壊さないため署名はそのまま残す。
    """
    return write_meta_for(config, config.seeds, wall_time_s, n_rows, path, extra)


__all__ = [
    "COMPARISON_CSV",
    "COMPARISON_SUMMARY_CSV",
    "META_JSON",
    "SUMMARY_CSV_COLUMNS",
    "DataclassSummaryMixin",
    "write_comparison_csv",
    "write_comparison_summary_csv",
    "write_meta",
    "write_meta_for",
    "write_rows_csv",
]

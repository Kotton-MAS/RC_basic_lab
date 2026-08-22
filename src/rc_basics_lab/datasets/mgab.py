"""MGAB (Mackey-Glass Anomaly Benchmark) の取得と読み取り.

出典・ライセンスの単一の真実は ``manifests/mgab.csv`` (CC0-1.0)。
ファイル本体はリポジトリに入れない (D-58) ので、``fetch`` が
``data/05_anomaly/mgab/`` へ取ってきて SHA256 で照合する。

CSV の列は ``index,value,is_anomaly,is_ignored`` (先頭列は無名のインデックス
列)。``is_ignored`` は**異常の前後ではなく系列の先頭 257 点**に立っている
(実測、10系列すべて同じ) —— 過渡区間を評価から外すためのマスクで、
「ignore は異常の近傍にしか付かない」と決め打つと読み違える。

``train_end`` は CSV に無いので**最初の異常が始まる index** を使う
(``AnomalySeries.train_end`` の定義 = 異常が1点も無いことが保証された前半の
終端)。実測では系列ごとに 29,068〜44,944 の範囲にある。
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rc_basics_lab.datasets.fetch import (
    DEFAULT_DATA_DIR,
    Opener,
    RemoteFile,
    ensure_file,
    is_cached,
)
from rc_basics_lab.datasets.manifest import MANIFEST_DIR, Manifest, read_manifest
from rc_basics_lab.tasks.anomaly import AnomalySeries
from rc_basics_lab.types import BoolArray, FloatArray

DATASET_NAME = "mgab"

MANIFEST_PATH = MANIFEST_DIR / "mgab.csv"

LICENSE = "CC0-1.0"
"""``manifests/mgab.csv`` の ``# license:`` および README と**同一の文字列**。"""

URL_TEMPLATE = "https://raw.githubusercontent.com/MarkusThill/MGAB/master/{series}.csv"

SERIES = tuple(str(number) for number in range(1, 11))
"""公開されている系列名 (1〜10)。"""


def manifest() -> Manifest:
    """``manifests/mgab.csv`` を読む。"""
    return read_manifest(MANIFEST_PATH)


def remote_file(series: str) -> RemoteFile:
    """系列1本ぶんの取得元・期待ハッシュ・保存先。"""
    row = manifest().row("series", series)
    return RemoteFile(
        url=URL_TEMPLATE.format(series=series),
        sha256=row["sha256"],
        relative_path=row["relative_path"],
    )


def remote_files(series: Sequence[str] = SERIES) -> tuple[RemoteFile, ...]:
    """複数系列ぶんの ``RemoteFile``。"""
    return tuple(remote_file(name) for name in series)


def fetch(
    series: Sequence[str] = SERIES,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    opener: Opener | None = None,
) -> tuple[Path, ...]:
    """未取得の系列をダウンロードし、SHA256 で照合する (D-58)。"""
    return tuple(
        ensure_file(remote, data_dir=data_dir, opener=opener)
        for remote in remote_files(series)
    )


def series_path(series: str, *, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """キャッシュ上の位置 (存在するとは限らない)。"""
    return data_dir / manifest().row("series", series)["relative_path"]


def is_available(series: str, *, data_dir: Path = DEFAULT_DATA_DIR) -> bool:
    """キャッシュが在り SHA256 も一致するか (**ネットワークに触れない**、D-60)。"""
    return is_cached(remote_file(series), data_dir=data_dir)


def _read_columns(path: Path) -> tuple[FloatArray, BoolArray, BoolArray]:
    """``value`` / ``is_anomaly`` / ``is_ignored`` の3列を読む。"""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"MGAB の CSV が空です: {path}")
    values = np.array([float(row["value"]) for row in rows], dtype=np.float64)
    labels = np.array([row["is_anomaly"] == "1" for row in rows], dtype=np.bool_)
    ignore = np.array([row["is_ignored"] == "1" for row in rows], dtype=np.bool_)
    return values, labels, ignore


def load_series(series: str, *, data_dir: Path = DEFAULT_DATA_DIR) -> AnomalySeries:
    """キャッシュ上の CSV を ``AnomalySeries`` にする (取得はしない)。

    Raises:
        FileNotFoundError: キャッシュが無い (``fetch`` か ``make data-05`` を
            先に実行する)。
    """
    path = series_path(series, data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"MGAB の系列 {series} がキャッシュにありません: {path} "
            "(make data-05 で取得してください)"
        )
    values, labels, ignore = _read_columns(path)
    train_end = int(np.argmax(labels))
    params: Mapping[str, str] = {
        "source": DATASET_NAME,
        "license": LICENSE,
        "url": URL_TEMPLATE.format(series=series),
    }
    return AnomalySeries(
        values=values.reshape(-1, 1),
        labels=labels,
        ignore=ignore,
        train_end=train_end,
        name=f"{DATASET_NAME}_{series}",
        params=params,
    )


@dataclass(frozen=True, slots=True)
class MgabSeriesSource:
    """MGAB の系列1本を ``SeriesSource`` (D-71) として渡すための束縛。

    系列名とキャッシュ先を構築時に受け取り、``__call__`` は ``load_series``
    へそのまま委譲する。実験層はこの型を ``SeriesSource`` としてしか見ないので、
    「MGAB のときだけキャッシュを確認する」分岐を持たずに済む。

    Attributes:
        series: 系列名 (``SERIES`` のいずれか)。
        data_dir: キャッシュ先 (既定は ``DEFAULT_DATA_DIR``)。
    """

    series: str
    data_dir: Path = DEFAULT_DATA_DIR

    def is_available(self) -> bool:
        """キャッシュが在り SHA256 も一致するか (**ネットワークに触れない**)。

        呼ぶのは同名のモジュール関数 ``mgab.is_available`` (メソッド自身では
        ない —— メソッドの本体からはモジュールのグローバルが見える)。
        """
        return is_available(self.series, data_dir=self.data_dir)

    def __call__(self, rng: np.random.Generator) -> AnomalySeries:
        """キャッシュ上の CSV を ``AnomalySeries`` にする (取得はしない)。

        実データなので ``rng`` は使わない —— ``SeriesSource`` の呼び出し口を
        源によらず1つに保つために受け取るだけである。
        """
        del rng
        return load_series(self.series, data_dir=self.data_dir)


__all__ = [
    "DATASET_NAME",
    "LICENSE",
    "MANIFEST_PATH",
    "SERIES",
    "URL_TEMPLATE",
    "MgabSeriesSource",
    "fetch",
    "is_available",
    "load_series",
    "manifest",
    "remote_file",
    "remote_files",
    "series_path",
]

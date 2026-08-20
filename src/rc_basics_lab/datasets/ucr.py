"""UCR Time Series Anomaly Archive (2021) の取得と読み取り.

**ライセンスは未指定**である。公式ページにライセンス表記が一切なく引用の
お願いだけがあるため、再配布可否が法的に不明で、本リポジトリはデータ本体を
一切含めない (D-58)。マニフェスト (``manifests/ucr.csv``) が持つのは
ファイル名・SHA256・系列メタだけ。

ZIP は実測 184,066,400 byte あるので、``fetch`` は**採用サブセット8系列だけ**を
展開する。展開は ``zipfile.ZipFile.extractall`` ではなく ``fetch.extract_members``
(member 名検査つき) を通す。

ラベルの復元
------------
ファイル名 ``NNN_UCR_Anomaly_{name}_{train_end}_{start}_{end}.txt`` が唯一の
ラベル源で、異常区間は1系列につきちょうど1本。**0 始まりか 1 始まりかは
UCR 公式に明記が無い**ので、解釈をこのモジュールの ``anomaly_slice`` 1箇所に
閉じ、``tests/test_datasets_anomaly.py::
test_ucr_filename_index_convention_is_pinned`` で固定する。

採った解釈は **1-indexed・``end`` は排他** (0-based では
``labels[start-1 : end-1]``、異常長 = ``end - start``)。根拠は
``.claude/tmp/dataset-manifest-source.md`` の実測表で、そこに記録された8系列の
異常長・異常率がすべて ``end - start`` で計算されているため、マニフェストと
読み取り実装が同じ数を指すのはこの解釈だけである。もう一方の読み (両端とも
閉区間、異常長 = ``end - start + 1``) との差は**1点**なので、AUPRC には
実質的な影響が無いが、**どちらか一方に固定しておかないと後から静かに揺れる**。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rc_basics_lab.datasets.fetch import (
    DEFAULT_DATA_DIR,
    Opener,
    RemoteFile,
    ensure_file,
    extract_members,
    is_cached,
    sha256_of,
)
from rc_basics_lab.datasets.manifest import MANIFEST_DIR, Manifest, read_manifest
from rc_basics_lab.tasks.anomaly import AnomalySeries
from rc_basics_lab.types import FloatArray

DATASET_NAME = "ucr"

MANIFEST_PATH = MANIFEST_DIR / "ucr.csv"

LICENSE = "未指定 (再配布可否不明・データ本体は同梱しない)"
"""``manifests/ucr.csv`` の ``# license:`` および README と**同一の文字列**。"""

ARCHIVE_URL = (
    "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"
    "UCR_TimeSeriesAnomalyDatasets2021.zip"
)

ARCHIVE_RELATIVE_PATH = "UCR_TimeSeriesAnomalyDatasets2021.zip"

FILENAME_PATTERN = re.compile(
    r"^(?P<number>\d+)_UCR_Anomaly_(?P<name>.+?)"
    r"_(?P<train_end>\d+)_(?P<start>\d+)_(?P<end>\d+)\.txt$"
)
"""``NNN_UCR_Anomaly_{name}_{train_end}_{start}_{end}.txt``。"""


@dataclass(frozen=True, slots=True)
class UcrFileSpec:
    """ファイル名から復元した系列のメタ情報。

    Attributes:
        number: 通し番号 (1〜250)。
        name: 系列名 (領域を表す文字列)。
        train_end: 学習に使ってよい区間の終端 (1-indexed の点番号)。
        anomaly_start: 異常区間の開始 (1-indexed)。
        anomaly_end: 異常区間の終端 (1-indexed、**排他**)。
        filename: 元のファイル名。
    """

    number: int
    name: str
    train_end: int
    anomaly_start: int
    anomaly_end: int
    filename: str


def parse_filename(filename: str) -> UcrFileSpec:
    """ファイル名から ``UcrFileSpec`` を復元する (規則に合わなければ例外)。"""
    match = FILENAME_PATTERN.match(filename)
    if match is None:
        raise ValueError(
            f"UCR の命名規則に一致しません: {filename!r} "
            "(NNN_UCR_Anomaly_{name}_{train_end}_{start}_{end}.txt)"
        )
    return UcrFileSpec(
        number=int(match["number"]),
        name=match["name"],
        train_end=int(match["train_end"]),
        anomaly_start=int(match["start"]),
        anomaly_end=int(match["end"]),
        filename=filename,
    )


def anomaly_slice(spec: UcrFileSpec) -> slice[int, int, int]:
    """0-based の異常区間 (**この1関数が index 解釈の単一の真実**)。

    1-indexed・``end`` 排他と読む。``labels[anomaly_slice(spec)] = True`` で
    異常点数は ``anomaly_end - anomaly_start`` になる (モジュール docstring)。
    """
    return slice(spec.anomaly_start - 1, spec.anomaly_end - 1)


def train_end_index(spec: UcrFileSpec) -> int:
    """0-based の ``train_end`` (排他 index)。1-indexed の閉端と同じ値になる。"""
    return spec.train_end


def manifest() -> Manifest:
    """``manifests/ucr.csv`` を読む。"""
    return read_manifest(MANIFEST_PATH)


def subset() -> tuple[str, ...]:
    """採用サブセットのファイル名 (マニフェストの行順)。"""
    return tuple(row["filename"] for row in manifest().rows)


def archive_remote() -> RemoteFile:
    """ZIP 本体の取得元・期待ハッシュ・保存先。"""
    return RemoteFile(
        url=ARCHIVE_URL,
        sha256=manifest().header["archive_sha256"],
        relative_path=ARCHIVE_RELATIVE_PATH,
    )


def member_name(filename: str) -> str:
    """ZIP 内の member パス (先頭の階層はマニフェストが持つ)。"""
    return f"{manifest().header['archive_member_prefix']}{filename}"


def series_path(filename: str, *, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """展開先 (存在するとは限らない)。"""
    return data_dir / manifest().row("filename", filename)["relative_path"]


def is_available(filename: str, *, data_dir: Path = DEFAULT_DATA_DIR) -> bool:
    """展開済みで SHA256 も一致するか (**ネットワークに触れない**、D-60)。"""
    row = manifest().row("filename", filename)
    return is_cached(
        RemoteFile(
            url=ARCHIVE_URL, sha256=row["sha256"], relative_path=row["relative_path"]
        ),
        data_dir=data_dir,
    )


def fetch(
    filenames: Sequence[str] | None = None,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    opener: Opener | None = None,
    keep_archive: bool = True,
) -> tuple[Path, ...]:
    """ZIP を取得し、採用サブセットだけを展開して SHA256 で照合する。

    展開済みで照合に通るファイルは触らない。``keep_archive=False`` を渡すと
    展開後に 184 MB の ZIP を消す (CI では使わない。D-60 によりテストは
    ネットワークに触れない)。
    """
    names = tuple(filenames) if filenames is not None else subset()
    if all(is_available(name, data_dir=data_dir) for name in names):
        return tuple(series_path(name, data_dir=data_dir) for name in names)
    archive = ensure_file(archive_remote(), data_dir=data_dir, opener=opener)
    destination = data_dir / "ucr"
    extract_members(
        archive,
        [member_name(name) for name in names],
        destination,
        data_dir=data_dir,
    )
    written: list[Path] = []
    for name in names:
        row = manifest().row("filename", name)
        path = data_dir / row["relative_path"]
        actual = sha256_of(path)
        if actual != row["sha256"]:
            path.unlink(missing_ok=True)
            raise ValueError(
                "展開したファイルの SHA256 が一致しません "
                f"(キャッシュには残しません): {name} "
                f"expected={row['sha256']} actual={actual}"
            )
        written.append(path)
    if not keep_archive:
        archive.unlink(missing_ok=True)
    return tuple(written)


def _read_values(path: Path) -> FloatArray:
    """1行1値のプレーンテキストを読む。"""
    with path.open(encoding="utf-8") as handle:
        values = [float(line) for line in handle if line.strip()]
    if not values:
        raise ValueError(f"UCR の系列が空です: {path}")
    return np.array(values, dtype=np.float64)


def load_series(filename: str, *, data_dir: Path = DEFAULT_DATA_DIR) -> AnomalySeries:
    """展開済みのファイルを ``AnomalySeries`` にする (取得はしない)。

    Raises:
        FileNotFoundError: キャッシュが無い (``fetch`` か ``make data-05``)。
    """
    path = series_path(filename, data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"UCR の系列がキャッシュにありません: {path} "
            "(make data-05 で取得してください)"
        )
    spec = parse_filename(filename)
    values = _read_values(path)
    labels = np.zeros(values.size, dtype=np.bool_)
    labels[anomaly_slice(spec)] = True
    ignore = np.zeros(values.size, dtype=np.bool_)
    params: Mapping[str, str] = {
        "source": DATASET_NAME,
        "license": LICENSE,
        "url": ARCHIVE_URL,
        "anomaly_start": str(spec.anomaly_start),
        "anomaly_end": str(spec.anomaly_end),
    }
    return AnomalySeries(
        values=values.reshape(-1, 1),
        labels=labels,
        ignore=ignore,
        train_end=train_end_index(spec),
        name=f"{DATASET_NAME}_{spec.number:03d}_{spec.name}",
        params=params,
    )


@dataclass(frozen=True, slots=True)
class UcrSeriesSource:
    """UCR の系列1本を ``SeriesSource`` (D-71) として渡すための束縛。

    ``MgabSeriesSource`` と同じ形 —— ファイル名と展開先を構築時に受け取り、
    ``__call__`` は ``load_series`` へ委譲する。

    Attributes:
        filename: 採用サブセットのファイル名 (``subset()`` のいずれか)。
        data_dir: 展開先 (既定は ``DEFAULT_DATA_DIR``)。
    """

    filename: str
    data_dir: Path = DEFAULT_DATA_DIR

    def is_available(self) -> bool:
        """展開済みで SHA256 も一致するか (**ネットワークに触れない**)。

        呼ぶのは同名のモジュール関数 ``ucr.is_available`` (メソッドの本体からは
        モジュールのグローバルが見える)。
        """
        return is_available(self.filename, data_dir=self.data_dir)

    def __call__(self, rng: np.random.Generator) -> AnomalySeries:
        """展開済みのファイルを ``AnomalySeries`` にする (取得はしない)。

        実データなので ``rng`` は使わない (呼び出し口を源によらず1つに保つ)。
        """
        del rng
        return load_series(self.filename, data_dir=self.data_dir)


__all__ = [
    "ARCHIVE_RELATIVE_PATH",
    "ARCHIVE_URL",
    "DATASET_NAME",
    "FILENAME_PATTERN",
    "LICENSE",
    "MANIFEST_PATH",
    "UcrFileSpec",
    "UcrSeriesSource",
    "anomaly_slice",
    "archive_remote",
    "fetch",
    "is_available",
    "load_series",
    "manifest",
    "member_name",
    "parse_filename",
    "series_path",
    "subset",
    "train_end_index",
]

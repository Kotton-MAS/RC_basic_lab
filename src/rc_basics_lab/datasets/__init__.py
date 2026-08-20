"""データ層 —— 外部データセットの取得・照合・読み取り (D-58 / D-59 / D-60).

**ネットワークとファイル I/O を持つ唯一のパッケージ**である。課題層
(``tasks/``) と指標層 (``metrics_detection.py``) は純関数のまま保ち、
依存の向きは ``datasets -> tasks`` の一方向にする (D-59)。逆向きの辺を1本でも
引くと、課題層がステートフルな I/O 層に化けて移植性 (D-12 が守る性質) が
失われる。

- ``fetch``: HTTPS 限定のダウンロード・SHA256 照合・キャッシュ・ZIP 展開
- ``manifest``: ``manifests/*.csv`` (出典 URL / ライセンス / SHA256) の読み取り
- ``mgab``: MGAB (CC0-1.0)。既定の実データ源
- ``ucr``: UCR Anomaly Archive (**ライセンス未指定**。本体は同梱しない)
- ``cli``: ``make data-05`` の実体

データ本体はリポジトリに入れない (D-58)。``data/`` は ``.gitignore`` 済みで、
**pytest はキャッシュが無ければ skip する** (D-60)。
"""

from rc_basics_lab.datasets import cli, fetch, manifest, mgab, ucr
from rc_basics_lab.datasets.fetch import (
    DEFAULT_DATA_DIR,
    ChecksumMismatchError,
    DatasetError,
    DownloadTooLargeError,
    RemoteFile,
    UnsafeArchiveMemberError,
    download,
    ensure_file,
    extract_members,
    is_cached,
    sha256_of,
)
from rc_basics_lab.datasets.manifest import MANIFEST_DIR, Manifest, read_manifest

__all__ = [
    "DEFAULT_DATA_DIR",
    "MANIFEST_DIR",
    "ChecksumMismatchError",
    "DatasetError",
    "DownloadTooLargeError",
    "Manifest",
    "RemoteFile",
    "UnsafeArchiveMemberError",
    "cli",
    "download",
    "ensure_file",
    "extract_members",
    "fetch",
    "is_cached",
    "manifest",
    "mgab",
    "read_manifest",
    "sha256_of",
    "ucr",
]

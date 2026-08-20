"""外部データセットの取得・照合・キャッシュ (D-58 / D-59).

**このパッケージがネットワークとファイル I/O を持つ唯一の場所**である
(D-59)。``tasks/`` と ``metrics_detection.py`` は純関数層に保ち、
``tests/test_layer_boundaries.py::test_tasks_and_metrics_never_perform_io``
が AST で機械検査する。依存の向きは ``datasets -> tasks`` の一方向。

取得側で守るもの (仕様 §5 安全性観点):

- **HTTPS のみ**。リダイレクト先も1つずつ検査する
- **リダイレクト追随の上限** (``MAX_REDIRECTS``)
- **サイズ上限** (``MAX_DOWNLOAD_BYTES`` = 200 MB。UCR の ZIP が実測 184 MB)
- **タイムアウト** (``DEFAULT_TIMEOUT_S``)
- **SHA256 不一致は例外にし、キャッシュにも残さない** (D-58)。書き込みは
  ``.part`` の一時ファイルへ行い、照合に通ったものだけを ``os.replace`` で
  確定させる —— 途中で落ちた半端なファイルが「キャッシュ済み」に見える経路を
  作らない
- **``data_dir`` の外に書かない**。相対パスは ``resolve_under`` が解決し、
  ``..`` や絶対パスは例外
- ZIP は **member 名を検査**してから展開する (パストラバーサル・シンボリック
  リンク・展開後サイズ)

HTTP を開く部分は ``Opener`` として差し替えられる。pytest は
**ネットワークに一切触れない** (D-60) ので、照合の検査はローカル fixture の
opener で行う。
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from io import IOBase
from pathlib import Path, PurePosixPath
from typing import IO, Protocol
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, OpenerDirector, Request, build_opener

DEFAULT_DATA_DIR = Path("data/05_anomaly")
"""キャッシュ先 (``.gitignore`` 済み)。

**YAML の設定フィールドにしない** —— 値を変えても実験結果が1ビットも変わらない
死んだフィールドになる (仕様 §3)。変えたいときは CLI の ``--data-dir``。
"""

MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024
"""1ファイルの上限 [byte]。UCR の ZIP が実測 184,066,400 byte。"""

MAX_REDIRECTS = 3
"""追随してよいリダイレクトの回数。標準の 10 より厳しくしてある。"""

DEFAULT_TIMEOUT_S = 60.0
"""ソケットのタイムアウト [秒]。"""

MAX_ARCHIVE_MEMBER_BYTES = 64 * 1024 * 1024
"""ZIP から取り出す 1 member の展開後サイズ上限 [byte] (zip bomb 対策)。"""

_CHUNK_BYTES = 1 << 20
_USER_AGENT = "rc-basics-lab/0.1 (+https://github.com/)"


class DatasetError(RuntimeError):
    """データセットの取得・検証の失敗 (このパッケージの基底例外)。"""


class ChecksumMismatchError(DatasetError):
    """SHA256 が期待値と一致しない (D-58)。"""


class DownloadTooLargeError(DatasetError):
    """サイズ上限を超えた。"""


class UnsafeArchiveMemberError(DatasetError):
    """ZIP の member 名が安全でない (パストラバーサル等)。"""


class HttpResponse(Protocol):
    """``urlopen`` の戻り値のうち、この層が使う部分だけ。"""

    def read(self, amt: int = ...) -> bytes: ...

    def close(self) -> None: ...


type Opener = Callable[[str, float], HttpResponse]
"""``(url, timeout) -> レスポンス``。テストが差し替える唯一の穴 (D-60)。"""


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """取得する1ファイル。

    Attributes:
        url: 取得元 (HTTPS のみ)。
        sha256: 期待する SHA256 (小文字16進64桁)。
        relative_path: ``data_dir`` からの相対パス (``..`` 不可)。
    """

    url: str
    sha256: str
    relative_path: str


def sha256_of(path: Path) -> str:
    """ファイルの SHA256 (小文字16進)。``shasum -a 256`` と同じ値。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_under(root: Path, relative: str) -> Path:
    """``root`` の内側に解決される絶対パスを返す (外に出る指定は例外)。"""
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise UnsafeArchiveMemberError(f"data_dir の外を指す相対パスです: {relative!r}")
    resolved = (root / Path(*candidate.parts)).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise UnsafeArchiveMemberError(
            f"data_dir の外に書こうとしています: {resolved} (root={root})"
        )
    return resolved


def require_https(url: str) -> None:
    """``https://`` 以外を拒む。"""
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise DatasetError(f"HTTPS 以外の URL は取得しません: {url!r}")
    if not parts.netloc:
        raise DatasetError(f"ホスト名がありません: {url!r}")


class _HttpsOnlyRedirectHandler(HTTPRedirectHandler):
    """リダイレクト先も HTTPS に限り、回数を ``MAX_REDIRECTS`` に絞る。"""

    max_redirections = MAX_REDIRECTS

    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> Request | None:
        require_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)  # type: ignore[arg-type]


def _build_opener() -> OpenerDirector:
    return build_opener(_HttpsOnlyRedirectHandler)


def open_https(url: str, timeout: float) -> HttpResponse:
    """既定の ``Opener``。HTTPS のみ・リダイレクト上限つきで開く。"""
    require_https(url)
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    response: HttpResponse = _build_opener().open(request, timeout=timeout)
    return response


def _stream_to_file(
    response: HttpResponse, destination: Path, max_bytes: int
) -> tuple[str, int]:
    """レスポンスを ``destination`` へ書きながら SHA256 と byte 数を測る。"""
    digest = hashlib.sha256()
    total = 0
    with destination.open("wb") as handle:
        while True:
            chunk = response.read(_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise DownloadTooLargeError(
                    f"サイズ上限を超えました: {total} > {max_bytes} byte"
                )
            digest.update(chunk)
            handle.write(chunk)
    return digest.hexdigest(), total


def _make_partial_path(target: Path) -> Path:
    """``target`` と同じディレクトリに、予測不能な名前の一時ファイルを作る。

    (F-1-019) 固定名 (``f"{target.name}.part"``) は書き込み中に横から
    差し替えられる TOCTOU の的になる —— 同じ ``data_dir`` に書ける別プロセス・
    別ユーザーが、正規のストリームからダイジェストが計算されている間にその
    パスだけを別ファイルへ差し替えると、ダイジェストは正規のストリームから
    計算されたまま、確定するファイルだけが攻撃者のものになる。
    ``tempfile.mkstemp`` で名前を予測不能にし、攻撃者がまず「どのパスを
    差し替えればよいか」を知ることそのものを難しくする。
    """
    descriptor, name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".part"
    )
    os.close(descriptor)
    return Path(name)


def _replace_after_reverifying(
    partial: Path,
    target: Path,
    expected_sha256: str,
    *,
    error_cls: type[DatasetError],
) -> None:
    """``os.replace`` の直前にディスク上の実バイト列を再照合してから確定させる。

    (F-1-019) ここまでに計算した digest は「自分が書いたはずのバイト列」から
    計算した値であり、``partial`` が確定させる実際のバイト列と同一である保証
    にはならない。名前を予測不能にしても、書き込み完了から ``os.replace`` まで
    の間に同じパスが差し替えられる余地をゼロにはできないため、確定させる
    **直前**にディスクを読み直して同じ値かどうかを測る
    (``download()`` と ``extract_members()`` の両方が通る、この層の最終防衛線)。
    """
    on_disk_sha256 = sha256_of(partial)
    if on_disk_sha256 != expected_sha256:
        partial.unlink(missing_ok=True)
        raise error_cls(
            "SHA256 が一致しません (確定直前にディスク上のバイト列が"
            "差し替えられた可能性があります。キャッシュには残しません): "
            f"expected={expected_sha256} actual={on_disk_sha256}"
        )
    os.replace(partial, target)


def download(
    remote: RemoteFile,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    opener: Opener | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> Path:
    """1ファイルを取得し、SHA256 が一致したときだけキャッシュに残す (D-58)。

    Args:
        remote: 取得元・期待ハッシュ・保存先。
        data_dir: キャッシュのルート。ここより外へは書かない。
        opener: HTTP を開く関数 (既定 ``open_https``)。テストが差し替える。
        timeout: ソケットのタイムアウト [秒]。
        max_bytes: 受け取ってよい最大 byte 数。

    Returns:
        保存したファイルの絶対パス。

    Raises:
        ChecksumMismatchError: SHA256 が期待値と違う。**一時ファイルも保存先も
            残さない** —— 残すと次回の実行が「キャッシュ済み」として拾う。
        DownloadTooLargeError: サイズ上限を超えた。
        DatasetError: HTTPS 以外、または取得そのものに失敗した。
    """
    require_https(remote.url)
    target = resolve_under(data_dir, remote.relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    open_url = opener if opener is not None else open_https
    try:
        response = open_url(remote.url, timeout)
    except HTTPError as error:  # pragma: no cover - ネットワーク経路
        raise DatasetError(f"取得に失敗しました: {remote.url} ({error})") from error
    partial = _make_partial_path(target)
    try:
        digest, total = _stream_to_file(response, partial, max_bytes)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        _close(response)
    if digest != remote.sha256:
        partial.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            "SHA256 が一致しません (取得先が差し替わった可能性があります。"
            "キャッシュには残しません): "
            f"url={remote.url} expected={remote.sha256} actual={digest} "
            f"size={total}"
        )
    _replace_after_reverifying(
        partial, target, remote.sha256, error_cls=ChecksumMismatchError
    )
    return target


def ensure_file(
    remote: RemoteFile,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    opener: Opener | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> Path:
    """キャッシュがあり SHA256 が一致すればそれを返し、無ければ取得する。

    キャッシュのハッシュが合わない場合は**そのファイルを消してから**取り直す
    (壊れたキャッシュを黙って使わない。D-58)。
    """
    target = resolve_under(data_dir, remote.relative_path)
    if target.exists():
        if sha256_of(target) == remote.sha256:
            return target
        target.unlink()
    return download(
        remote,
        data_dir=data_dir,
        opener=opener,
        timeout=timeout,
        max_bytes=max_bytes,
    )


def is_cached(remote: RemoteFile, *, data_dir: Path = DEFAULT_DATA_DIR) -> bool:
    """キャッシュが在り、SHA256 も一致するか (ネットワークに触れない)。"""
    target = resolve_under(data_dir, remote.relative_path)
    return target.exists() and sha256_of(target) == remote.sha256


def check_member_name(member: str) -> str:
    """ZIP の member 名を検査し、取り出すファイル名 (basename) を返す。

    絶対パス・``..``・Windows のドライブ指定・バックスラッシュを拒む。展開先は
    **basename へ平坦化**する —— 元の階層を再現しないので、たとえ検査を1つ
    見落としても ``dest_dir`` の外には出ない。
    """
    if not member or member.endswith("/"):
        raise UnsafeArchiveMemberError(f"ファイルではない member です: {member!r}")
    if "\\" in member or ":" in member:
        raise UnsafeArchiveMemberError(f"安全でない member 名です: {member!r}")
    parts = PurePosixPath(member)
    if parts.is_absolute() or ".." in parts.parts:
        raise UnsafeArchiveMemberError(f"安全でない member 名です: {member!r}")
    return parts.name


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return (info.external_attr >> 16) & 0o170000 == 0o120000


def _extract_member(
    bundle: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path
) -> None:
    """1 member を安全に取り出す (``download()`` と同型の TOCTOU 対策、F-1-019)。

    展開したファイルも「ディスクに確定するバイト列が検証されていない」という
    ``download()`` と同じ不変条件の対象である。予測可能な basename へ直接
    書かず、``target.parent`` 内の予測不能な一時ファイルへ書いてから、
    書き込み中に計算した digest でディスク上の実バイト列を再照合し、
    ``os.replace`` で確定させる。
    """
    partial = _make_partial_path(target)
    digest = hashlib.sha256()
    try:
        with bundle.open(info) as source, partial.open("wb") as sink:
            for chunk in iter(lambda: source.read(_CHUNK_BYTES), b""):
                digest.update(chunk)
                sink.write(chunk)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    _replace_after_reverifying(
        partial, target, digest.hexdigest(), error_cls=UnsafeArchiveMemberError
    )


def extract_members(
    archive: Path,
    members: Sequence[str],
    destination: Path,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    max_member_bytes: int = MAX_ARCHIVE_MEMBER_BYTES,
) -> tuple[Path, ...]:
    """ZIP から指定 member だけを ``destination`` へ平坦に取り出す。

    ``zipfile.ZipFile.extractall`` を使わない —— member 名を検査せずに展開すると
    ``../`` を含むエントリが展開先の外へ書ける (CVE-2007-4559 系)。

    Raises:
        UnsafeArchiveMemberError: member 名が安全でない、シンボリックリンク、
            展開後サイズが上限を超える、または確定直前のディスク再照合で
            バイト列の差し替えを検出した場合 (F-1-019)。
    """
    if not destination.resolve().is_relative_to(data_dir.resolve()):
        raise UnsafeArchiveMemberError(
            f"data_dir の外に展開しようとしています: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with zipfile.ZipFile(archive) as bundle:
        for member in members:
            name = check_member_name(member)
            info = bundle.getinfo(member)
            if _is_symlink(info):
                raise UnsafeArchiveMemberError(
                    f"シンボリックリンクは展開しません: {member!r}"
                )
            if info.file_size > max_member_bytes:
                raise UnsafeArchiveMemberError(
                    "展開後サイズが上限を超えます: "
                    f"{member!r} {info.file_size} > {max_member_bytes} byte"
                )
            target = resolve_under(destination, name)
            _extract_member(bundle, info, target)
            written.append(target)
    return tuple(written)


def _close(response: HttpResponse) -> None:
    """``Opener`` が返したものを閉じる (閉じ方を持たない差し替えも許す)。"""
    if isinstance(response, IOBase) or hasattr(response, "close"):
        response.close()


def missing(
    remotes: Iterable[RemoteFile], *, data_dir: Path = DEFAULT_DATA_DIR
) -> tuple[RemoteFile, ...]:
    """キャッシュに無い (またはハッシュが合わない) ものだけを返す。"""
    return tuple(
        remote for remote in remotes if not is_cached(remote, data_dir=data_dir)
    )


__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_TIMEOUT_S",
    "MAX_ARCHIVE_MEMBER_BYTES",
    "MAX_DOWNLOAD_BYTES",
    "MAX_REDIRECTS",
    "ChecksumMismatchError",
    "DatasetError",
    "DownloadTooLargeError",
    "HttpResponse",
    "Opener",
    "RemoteFile",
    "UnsafeArchiveMemberError",
    "check_member_name",
    "download",
    "ensure_file",
    "extract_members",
    "is_cached",
    "missing",
    "open_https",
    "require_https",
    "resolve_under",
    "sha256_of",
]

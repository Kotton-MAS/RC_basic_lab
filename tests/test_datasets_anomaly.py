"""データ層 (``datasets/``) の検査 (D-58 / D-59 / D-60).

**このファイルはネットワークに1バイトも触れない** (D-60)。HTTP を開く部分は
``fetch.Opener`` として差し替えられるので、取得・照合の検査はローカルの
バイト列を返す fixture で行う。実データ源のテストはキャッシュが無ければ
``skip`` する —— CI がネットワーク可用性に依存すると、UCR の URL が死んだ日に
リポジトリ全体が赤になり、実装の正しさと外部の可用性が区別できなくなる。
"""

from __future__ import annotations

import ast
import hashlib
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from rc_basics_lab.config import SyntheticAnomalyConfig
from rc_basics_lab.datasets import fetch, mgab, ucr
from rc_basics_lab.datasets.fetch import (
    ChecksumMismatchError,
    DatasetError,
    DownloadTooLargeError,
    Opener,
    RemoteFile,
    UnsafeArchiveMemberError,
)
from rc_basics_lab.datasets.manifest import read_manifest
from rc_basics_lab.tasks.anomaly import generate_synthetic_anomalies

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

PAYLOAD = b"value,is_anomaly\n0.5,0\n" * 16
"""取得のふりをして流すローカルのバイト列 (ネットワーク不使用)。"""


class _LocalResponse:
    """``fetch.HttpResponse`` のふりをする、メモリ上のバイト列。"""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0
        self.closed = False

    def read(self, amt: int = -1) -> bytes:
        if amt < 0:
            amt = len(self._payload) - self._offset
        chunk = self._payload[self._offset : self._offset + amt]
        self._offset += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


def _local_opener(payload: bytes) -> Opener:
    """ローカルのバイト列を返す ``Opener`` (**ネットワークを使わない**)。"""

    def opener(url: str, timeout: float) -> fetch.HttpResponse:
        assert url.startswith("https://"), "HTTPS 以外が渡ってきました"
        assert timeout > 0.0
        return _LocalResponse(payload)

    return opener


def _sha256_of_bytes(payload: bytes, tmp_path: Path) -> str:
    """``fetch.sha256_of`` (= ``shasum -a 256``) でバイト列のハッシュを測る。"""
    scratch = tmp_path / "scratch.bin"
    scratch.write_bytes(payload)
    digest = fetch.sha256_of(scratch)
    scratch.unlink()
    return digest


# --- SHA256 照合 (D-58) ------------------------------------------------------


def test_download_is_rejected_when_the_sha256_does_not_match(tmp_path: Path) -> None:
    """ハッシュが違うファイルを掴ませると例外になり、**キャッシュに残らない** (D-58)。

    URL 先が差し替わったとき、照合が無いと「違うデータで実験して同じ数値が
    出ない」という形でしか気づけない。半端な ``.part`` が残ると次の実行が
    「キャッシュ済み」として拾うので、**両方消えている**ことまで測る。
    """
    remote = RemoteFile(
        url="https://example.invalid/mgab/1.csv",
        sha256="0" * 64,
        relative_path="mgab/1.csv",
    )
    with pytest.raises(ChecksumMismatchError, match="SHA256"):
        fetch.download(
            remote, data_dir=tmp_path, opener=_local_opener(PAYLOAD), timeout=1.0
        )
    assert not (tmp_path / "mgab" / "1.csv").exists(), "不一致のファイルが残りました"
    assert not (tmp_path / "mgab" / "1.csv.part").exists(), "一時ファイルが残りました"
    assert list((tmp_path / "mgab").iterdir()) == []


def test_download_keeps_the_file_when_the_sha256_matches(tmp_path: Path) -> None:
    """一致したときだけキャッシュに残る (照合の肯定側)。"""
    digest = _sha256_of_bytes(PAYLOAD, tmp_path)
    remote = RemoteFile(
        url="https://example.invalid/mgab/1.csv",
        sha256=digest,
        relative_path="mgab/1.csv",
    )
    path = fetch.download(
        remote, data_dir=tmp_path, opener=_local_opener(PAYLOAD), timeout=1.0
    )
    assert path.read_bytes() == PAYLOAD
    assert fetch.is_cached(remote, data_dir=tmp_path)
    assert not path.with_name(f"{path.name}.part").exists()


def test_ensure_file_replaces_a_corrupted_cache(tmp_path: Path) -> None:
    """壊れたキャッシュは黙って使わず、取り直す (D-58)。"""
    digest = _sha256_of_bytes(PAYLOAD, tmp_path)
    remote = RemoteFile(
        url="https://example.invalid/mgab/1.csv",
        sha256=digest,
        relative_path="mgab/1.csv",
    )
    target = tmp_path / "mgab" / "1.csv"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupted")
    path = fetch.ensure_file(
        remote, data_dir=tmp_path, opener=_local_opener(PAYLOAD), timeout=1.0
    )
    assert path.read_bytes() == PAYLOAD


def test_ensure_file_does_not_open_anything_when_the_cache_is_valid(
    tmp_path: Path,
) -> None:
    """キャッシュが有効なら ``Opener`` を1回も呼ばない (2回目以降はオフライン)。"""
    digest = _sha256_of_bytes(PAYLOAD, tmp_path)
    remote = RemoteFile(
        url="https://example.invalid/mgab/1.csv",
        sha256=digest,
        relative_path="mgab/1.csv",
    )
    target = tmp_path / "mgab" / "1.csv"
    target.parent.mkdir(parents=True)
    target.write_bytes(PAYLOAD)

    def forbidden(url: str, timeout: float) -> fetch.HttpResponse:
        raise AssertionError(f"キャッシュがあるのに取得しようとしました: {url}")

    assert fetch.ensure_file(remote, data_dir=tmp_path, opener=forbidden) == target


# --- TOCTOU (reviewer-security 指摘) -----------------------------------------


def test_partial_path_is_not_predictable(tmp_path: Path) -> None:
    """``.part`` の名前は固定 (``f"{name}.part"``) ではなく予測不能である。

    固定名は、同じ ``data_dir`` に書ける別プロセス・別ユーザーが「どのパスを
    差し替えればよいか」を書き込み前から知っている TOCTOU の的になる
    (reviewer-security 指摘)。少なくとも旧来の固定名パターンとは一致せず、2回呼んでも
    毎回違う名前になることを固定する。
    """
    target = tmp_path / "probe.bin"
    first = fetch._make_partial_path(target)
    first.unlink()
    second = fetch._make_partial_path(target)
    second.unlink()
    assert first != second
    assert first.name != f"{target.name}.part"
    assert second.name != f"{target.name}.part"


def test_replace_after_reverifying_rejects_bytes_swapped_before_replace(
    tmp_path: Path,
) -> None:
    """``download()`` と ``extract_members()`` が共有する最終防衛線。

    書き込みが完了した直後に ``.part`` の中身が差し替えられても、
    ``os.replace`` 直前の再照合が検出し、一時ファイルも確定先も残さない。
    """
    partial = tmp_path / ".probe.bin.deadbeef.part"
    target = tmp_path / "probe.bin"
    partial.write_bytes(b"attacker-controlled-bytes")
    expected = hashlib.sha256(b"legitimate-bytes").hexdigest()
    with pytest.raises(fetch.ChecksumMismatchError, match="差し替え"):
        fetch._replace_after_reverifying(
            partial, target, expected, error_cls=fetch.ChecksumMismatchError
        )
    assert not partial.exists()
    assert not target.exists()


def test_download_and_extract_members_both_route_through_the_replace_guard() -> None:
    """``download()`` と ``_extract_member()`` の両方が最終防衛線を通る。

    片方だけ直して他方を「一貫性のため」据え置く事故を機械的に防ぐ
    (reviewer-security 指摘: 同じクラスの欠陥は全経路で潰す)。
    """
    tree = ast.parse(
        Path(fetch.__file__).read_text(encoding="utf-8"), filename=fetch.__file__
    )
    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    for name in ("download", "_extract_member"):
        called = {
            call.func.id
            for call in ast.walk(functions[name])
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }
        assert "_replace_after_reverifying" in called, (
            f"{name} が最終防衛線 (_replace_after_reverifying) を通っていません"
        )


def test_download_is_rejected_when_the_part_file_is_swapped_mid_write(
    tmp_path: Path,
) -> None:
    """予測不能な一時名でも、ディレクトリを監視 (glob) できる攻撃者は検出する。

    TOCTOU の再現 (``.claude/tmp/repro_toctou.py``) が前提にしていた「固定名を
    知っている」より強い攻撃者モデル —— 名前の予測不能性だけに頼らず、確定
    直前にディスク上の実バイト列を再照合する経路がここで効くことを確認する。
    """
    legit = b"LEGIT-" * 200_000
    evil = b"EVIL-" * 200_000
    expected = hashlib.sha256(legit).hexdigest()
    swapped = tmp_path / ".attacker-payload.bin"

    class _RacingResponse:
        def __init__(self) -> None:
            self._offset = 0

        def read(self, amt: int = -1) -> bytes:
            if self._offset >= len(legit):
                return b""
            chunk = legit[self._offset : self._offset + 4096]
            self._offset += len(chunk)
            if self._offset == len(chunk):  # 最初の読み出し直後に差し替える
                # ``os.replace`` で inode ごと差し替える (``Path.write_bytes`` の
                # ような同一 inode への上書きでは、元の記述子がその後も同じ
                # inode へ書き続けるため攻撃が成立しない。実際の攻撃者は
                # 別ファイルを作ってからパスへ rename する)。
                for part in tmp_path.rglob("*.part"):
                    swapped.write_bytes(evil)
                    os.replace(swapped, part)
            return chunk

        def close(self) -> None:
            return None

    remote = RemoteFile(
        url="https://example.invalid/probe.bin",
        sha256=expected,
        relative_path="probe.bin",
    )
    with pytest.raises(ChecksumMismatchError, match="差し替え"):
        fetch.download(
            remote, data_dir=tmp_path, opener=lambda url, timeout: _RacingResponse()
        )
    assert not (tmp_path / "probe.bin").exists()
    assert list(tmp_path.rglob("*.part")) == []


# --- 取得の安全性 (仕様 §5 安全性観点) ---------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://example.invalid/a.csv",
        "ftp://example.invalid/a.csv",
        "file:///etc/passwd",
    ],
)
def test_download_refuses_anything_but_https(url: str, tmp_path: Path) -> None:
    """HTTPS 以外は取得しない。"""
    remote = RemoteFile(url=url, sha256="0" * 64, relative_path="a.csv")
    with pytest.raises(DatasetError, match="HTTPS"):
        fetch.download(remote, data_dir=tmp_path, opener=_local_opener(PAYLOAD))


def test_download_stops_at_the_size_limit(tmp_path: Path) -> None:
    """サイズ上限を超えたら止める (ZIP 200 MB が既定)。"""
    remote = RemoteFile(
        url="https://example.invalid/big.bin",
        sha256="0" * 64,
        relative_path="big.bin",
    )
    with pytest.raises(DownloadTooLargeError, match="サイズ上限"):
        fetch.download(
            remote,
            data_dir=tmp_path,
            opener=_local_opener(b"x" * 4096),
            max_bytes=16,
        )
    assert not (tmp_path / "big.bin").exists()
    assert not (tmp_path / "big.bin.part").exists()


def test_the_size_limit_covers_the_ucr_archive() -> None:
    """上限が UCR の ZIP (実測 184,066,400 byte) を通し、青天井でもない。"""
    assert fetch.MAX_DOWNLOAD_BYTES == 200 * 1024 * 1024
    assert 184_066_400 < fetch.MAX_DOWNLOAD_BYTES < 300 * 1024 * 1024


def test_redirects_are_bounded_and_stay_on_https() -> None:
    """リダイレクトは回数上限つきで、追随先も HTTPS に限る。"""
    assert fetch.MAX_REDIRECTS == 3
    handler = fetch._HttpsOnlyRedirectHandler()
    assert handler.max_redirections == fetch.MAX_REDIRECTS
    with pytest.raises(DatasetError, match="HTTPS"):
        fetch.require_https("http://example.invalid/redirected")


@pytest.mark.parametrize(
    "relative", ["../escape.csv", "/etc/passwd", "a/../../escape.csv"]
)
def test_download_never_writes_outside_the_data_dir(
    relative: str, tmp_path: Path
) -> None:
    """``data_dir`` の外を指す相対パスは受け付けない。"""
    with pytest.raises(UnsafeArchiveMemberError):
        fetch.resolve_under(tmp_path, relative)


def _make_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)


def test_extract_members_rejects_a_path_traversal_member(tmp_path: Path) -> None:
    """``../`` を含む member は展開しない (CVE-2007-4559 系)。

    ``extractall`` を使うと同じ ZIP で展開先の外へ書ける。
    """
    archive = tmp_path / "evil.zip"
    _make_zip(archive, {"../escaped.txt": b"pwned", "ok.txt": b"fine"})
    destination = tmp_path / "out"
    with pytest.raises(UnsafeArchiveMemberError):
        fetch.extract_members(
            archive, ["../escaped.txt"], destination, data_dir=tmp_path
        )
    assert not (tmp_path.parent / "escaped.txt").exists()


def test_extract_members_takes_only_the_requested_files(tmp_path: Path) -> None:
    """指定した member だけを平坦に取り出す (250 個の ZIP から8個だけ)。"""
    archive = tmp_path / "bundle.zip"
    _make_zip(
        archive,
        {
            "prefix/a.txt": b"aaa",
            "prefix/b.txt": b"bbb",
            "prefix/c.txt": b"ccc",
        },
    )
    destination = tmp_path / "out"
    written = fetch.extract_members(
        archive, ["prefix/a.txt", "prefix/c.txt"], destination, data_dir=tmp_path
    )
    assert [path.name for path in written] == ["a.txt", "c.txt"]
    assert sorted(item.name for item in destination.iterdir()) == ["a.txt", "c.txt"]


def test_extract_members_refuses_to_write_outside_the_data_dir(tmp_path: Path) -> None:
    """展開先が ``data_dir`` の外なら例外。"""
    archive = tmp_path / "bundle.zip"
    _make_zip(archive, {"a.txt": b"aaa"})
    with pytest.raises(UnsafeArchiveMemberError, match="data_dir"):
        fetch.extract_members(
            archive, ["a.txt"], tmp_path.parent / "elsewhere", data_dir=tmp_path
        )


def test_extract_members_rejects_an_oversized_member(tmp_path: Path) -> None:
    """展開後サイズの上限 (zip bomb 対策)。"""
    archive = tmp_path / "bomb.zip"
    _make_zip(archive, {"big.txt": b"0" * 4096})
    with pytest.raises(UnsafeArchiveMemberError, match="展開後サイズ"):
        fetch.extract_members(
            archive,
            ["big.txt"],
            tmp_path / "out",
            data_dir=tmp_path,
            max_member_bytes=16,
        )


def _make_zip_with_symlink(path: Path, name: str, link_target: bytes) -> None:
    """``name`` をシンボリックリンクとしてマークした ZIP を作る (reviewer-test 指摘)。

    ``external_attr`` の上位16bitに Unix のファイルモードが入る。
    ``S_IFLNK (0o120000)`` を立てると、実 OS 上のシンボリックリンクを
    アーカイブしていなくても ``_is_symlink`` が拾う対象を再現できる。
    """
    info = zipfile.ZipInfo(name)
    info.external_attr = (0o120777 << 16) | 0x08  # S_IFLNK + Unix 属性フラグ
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(info, link_target)


def test_extract_members_rejects_a_symlink_member(tmp_path: Path) -> None:
    """シンボリックリンクとしてマークされた member は展開しない (reviewer-test 指摘)。

    ``docstring``/``Raises`` に明記された「シンボリックリンク」を通る guard が
    無かった (measured: fetch.py 147 stmts / 91% cover, missing に 326 行の
    raise を含む)。``_is_symlink`` のビット演算 (``0o170000``/``0o120000``) を
    間違えても検知できるように、実際に ``UnsafeArchiveMemberError`` を要求する。
    """
    archive = tmp_path / "symlink.zip"
    _make_zip_with_symlink(archive, "evil-link.txt", b"/etc/passwd")
    with pytest.raises(UnsafeArchiveMemberError, match="シンボリックリンク"):
        fetch.extract_members(
            archive, ["evil-link.txt"], tmp_path / "out", data_dir=tmp_path
        )


# --- ネットワークに触れない (D-60) -------------------------------------------


def test_default_source_needs_no_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """既定データ源は合成で、テストはネットワークに一切触れない (D-60)。

    ``urllib`` の口を塞いだ状態で、

    - 合成源が系列を返す (これが既定)
    - マニフェストが読める (出典・ライセンス・SHA256 はリポジトリの中)
    - キャッシュの有無の問い合わせが通る (無ければ ``False``)

    ことを確かめる。CI がネットワーク可用性に依存すると、UCR の URL が死んだ日に
    リポジトリ全体が赤になり、実装の正しさと外部の可用性が区別できなくなる。
    """

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("テストがネットワークに触れようとしました (D-60)")

    monkeypatch.setattr(fetch, "open_https", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)

    cfg = SyntheticAnomalyConfig(length=3000, n_anomalies=2, segment_length=40)
    series = generate_synthetic_anomalies(cfg, np.random.default_rng(0))
    assert series.n_steps == cfg.length
    assert series.n_anomalies == cfg.n_anomalies

    assert mgab.manifest().license == mgab.LICENSE
    assert ucr.manifest().license == ucr.LICENSE
    assert mgab.is_available("1", data_dir=tmp_path) is False
    assert ucr.is_available(ucr.subset()[0], data_dir=tmp_path) is False

    with pytest.raises(FileNotFoundError):
        mgab.load_series("1", data_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        ucr.load_series(ucr.subset()[0], data_dir=tmp_path)


# --- fetch() のオーケストレーション (reviewer-test 指摘) ----------------------
#
# ``mgab.fetch`` / ``mgab.remote_files`` / ``ucr.fetch`` は `make data-05` が
# 実際に呼ぶ最上位の関数だが、実データのマニフェストが本物の URL を指すため
# 0% カバレッジだった。マニフェストを monkeypatch でローカル CSV に差し替え、
# ``_local_opener`` (ネットワーク不使用) だけで正常系・異常系を駆動する。


def _write_mgab_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    lines = ["# license: CC0-1.0 (テスト用)\n", "series,relative_path,sha256\n"]
    lines += [
        f"{row['series']},{row['relative_path']},{row['sha256']}\n" for row in rows
    ]
    path.write_text("".join(lines), encoding="utf-8")


def test_mgab_fetch_downloads_a_missing_series_via_the_local_opener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``mgab.fetch`` / ``remote_files`` の正常系 (reviewer-test 指摘)。"""
    payload = b"index,value,is_anomaly,is_ignored\n0,0.1,0,0\n1,0.2,1,0\n"
    digest = _sha256_of_bytes(payload, tmp_path)
    manifest_csv = tmp_path / "mgab_manifest.csv"
    _write_mgab_manifest(
        manifest_csv, [{"series": "a", "relative_path": "mgab/a.csv", "sha256": digest}]
    )
    monkeypatch.setattr(mgab, "MANIFEST_PATH", manifest_csv)

    remotes = mgab.remote_files(["a"])
    assert remotes[0].sha256 == digest

    written = mgab.fetch(["a"], data_dir=tmp_path, opener=_local_opener(payload))
    assert written == (mgab.series_path("a", data_dir=tmp_path),)
    assert written[0].read_bytes() == payload
    assert mgab.is_available("a", data_dir=tmp_path)


def test_mgab_fetch_raises_and_removes_the_file_when_the_sha256_does_not_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``mgab.fetch`` の異常系: マニフェストと違う SHA256 は残さない (D-58)。"""
    payload = b"index,value,is_anomaly,is_ignored\n0,0.1,0,0\n"
    manifest_csv = tmp_path / "mgab_manifest.csv"
    _write_mgab_manifest(
        manifest_csv,
        [{"series": "a", "relative_path": "mgab/a.csv", "sha256": "0" * 64}],
    )
    monkeypatch.setattr(mgab, "MANIFEST_PATH", manifest_csv)

    with pytest.raises(ChecksumMismatchError, match="SHA256"):
        mgab.fetch(["a"], data_dir=tmp_path, opener=_local_opener(payload))
    assert not (tmp_path / "mgab" / "a.csv").exists()


def _write_ucr_manifest(
    path: Path, *, archive_sha256: str, member_prefix: str, rows: list[dict[str, str]]
) -> None:
    lines = [
        "# license: 未指定 (テスト用)\n",
        f"# archive_sha256: {archive_sha256}\n",
        f"# archive_member_prefix: {member_prefix}\n",
        "filename,relative_path,sha256\n",
    ]
    lines += [
        f"{row['filename']},{row['relative_path']},{row['sha256']}\n" for row in rows
    ]
    path.write_text("".join(lines), encoding="utf-8")


def test_ucr_fetch_downloads_and_extracts_via_the_local_opener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ucr.fetch`` の正常系 (ZIP 取得 -> extract_members -> 展開後の再照合)。

    reviewer-test 指摘: 部品単体 (``fetch.download`` / ``fetch.extract_members``) の
    テストはあっても、それらをつなぐ最上位の経路が0%カバレッジだった。
    """
    content = b"1.0\n2.0\n3.0\n"
    content_sha256 = _sha256_of_bytes(content, tmp_path)
    archive_path = tmp_path / "archive.zip"
    _make_zip(archive_path, {"series_a.txt": content})
    archive_sha256 = fetch.sha256_of(archive_path)

    manifest_csv = tmp_path / "ucr_manifest.csv"
    _write_ucr_manifest(
        manifest_csv,
        archive_sha256=archive_sha256,
        member_prefix="",
        rows=[
            {
                "filename": "series_a.txt",
                "relative_path": "ucr/series_a.txt",
                "sha256": content_sha256,
            }
        ],
    )
    monkeypatch.setattr(ucr, "MANIFEST_PATH", manifest_csv)

    archive_bytes = archive_path.read_bytes()
    written = ucr.fetch(
        ["series_a.txt"], data_dir=tmp_path, opener=_local_opener(archive_bytes)
    )
    assert written == (tmp_path / "ucr" / "series_a.txt",)
    assert written[0].read_bytes() == content
    assert ucr.is_available("series_a.txt", data_dir=tmp_path)


def test_ucr_fetch_rejects_and_removes_a_series_whose_extracted_sha256_is_wrong(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ucr.fetch`` の異常系: 展開後の再照合 (ucr.py 2箇所目の防衛線)。

    ZIP 全体の SHA256 (``archive_sha256``) は一致していても、展開した個別
    ファイルの SHA256 がマニフェストと食い違えば ``ValueError`` にし、
    展開したファイルを消す。
    """
    content = b"1.0\n2.0\n3.0\n"
    archive_path = tmp_path / "archive.zip"
    _make_zip(archive_path, {"series_a.txt": content})
    archive_sha256 = fetch.sha256_of(archive_path)

    manifest_csv = tmp_path / "ucr_manifest.csv"
    _write_ucr_manifest(
        manifest_csv,
        archive_sha256=archive_sha256,
        member_prefix="",
        rows=[
            {
                "filename": "series_a.txt",
                "relative_path": "ucr/series_a.txt",
                "sha256": "0" * 64,  # 意図的に一致しない
            }
        ],
    )
    monkeypatch.setattr(ucr, "MANIFEST_PATH", manifest_csv)

    archive_bytes = archive_path.read_bytes()
    with pytest.raises(ValueError, match="SHA256"):
        ucr.fetch(
            ["series_a.txt"], data_dir=tmp_path, opener=_local_opener(archive_bytes)
        )
    assert not (tmp_path / "ucr" / "series_a.txt").exists()


# --- マニフェストとライセンス表記 --------------------------------------------


def test_manifest_licenses_match_the_module_constants() -> None:
    """コード上のライセンス文字列とマニフェストが一致する。"""
    assert read_manifest(mgab.MANIFEST_PATH).license == "CC0-1.0"
    assert mgab.LICENSE == "CC0-1.0"
    assert read_manifest(ucr.MANIFEST_PATH).license == ucr.LICENSE


def test_readme_license_matches_the_manifests() -> None:
    """README の「データセットのライセンスと取得手順」と CSV が同じ文字列を持つ。

    仕様 §4 T2 受け入れ基準6。文書とマニフェストが別々に育つと、記事に
    「CC0 のデータを使いました」と書いたまま実体が入れ替わる事故が起きる。
    """
    text = README.read_text(encoding="utf-8")
    assert "## データセットのライセンスと取得手順" in text
    section = text.split("## データセットのライセンスと取得手順", 1)[1]
    section = section.split("\n## ", 1)[0]
    assert mgab.LICENSE in section, "README に MGAB のライセンス文字列がありません"
    assert ucr.LICENSE in section, "README に UCR のライセンス文字列がありません"
    assert "https://github.com/MarkusThill/MGAB" in section
    assert "10.5281/zenodo.3760086" in section
    assert "make data-05" in section


def test_readme_states_that_the_ucr_data_is_not_redistributed() -> None:
    """UCR は「ライセンス未指定・再配布可否不明・本体は同梱しない」と書く (D-58)。

    リスク1 (仕様 §7): README の記述が「再配布していない」以上の主張をしそうに
    なったら止める、という約束の機械的な下限。
    """
    text = README.read_text(encoding="utf-8")
    section = text.split("## データセットのライセンスと取得手順", 1)[1]
    for phrase in ("未指定", "再配布", "同梱しない"):
        assert phrase in section, f"README に「{phrase}」の記述がありません"


def test_no_dataset_payload_is_committed_to_the_repository() -> None:
    """``datasets/`` にデータ本体が紛れ込んでいない (D-58)。

    置いてよいのは Python と ``manifests/*.csv`` だけ。
    """
    package = ROOT / "src" / "rc_basics_lab" / "datasets"
    unexpected = [
        path.relative_to(ROOT).as_posix()
        for path in package.rglob("*")
        if path.is_file()
        and path.suffix not in {".py", ".typed", ".pyc"}
        and "__pycache__" not in path.parts
        and path.parent.name != "manifests"
    ]
    assert not unexpected, f"データ本体らしきファイルがあります: {unexpected}"
    for path in (package / "manifests").glob("*"):
        assert path.suffix == ".csv"
        assert path.stat().st_size < 32 * 1024, f"マニフェストが大きすぎます: {path}"


# --- UCR のファイル名からのラベル復元 ----------------------------------------


def test_ucr_filename_index_convention_is_pinned() -> None:
    """index の解釈を1箇所 (``ucr.anomaly_slice``) に閉じ、値で固定する。

    UCR 公式は 0 始まりか 1 始まりかを明記していない。採ったのは
    **1-indexed・``end`` 排他** (0-based で ``labels[start-1 : end-1]``、
    異常長 = ``end - start``) で、``.claude/tmp/dataset-manifest-source.md`` の
    実測表 (8系列の異常長・異常率) と一致するのはこの読み方だけである。
    もう一方の読み (両端とも閉区間) との差は1点。
    """
    spec = ucr.parse_filename("119_UCR_Anomaly_ECG1_10000_11800_12100.txt")
    assert spec.number == 119
    assert spec.name == "ECG1"
    assert spec.train_end == 10000
    assert (spec.anomaly_start, spec.anomaly_end) == (11800, 12100)
    assert ucr.anomaly_slice(spec) == slice(11799, 12099)
    assert ucr.train_end_index(spec) == 10000

    labels = np.zeros(30000, dtype=np.bool_)
    labels[ucr.anomaly_slice(spec)] = True
    assert int(np.count_nonzero(labels)) == 300
    assert bool(labels[11799]) and bool(labels[12098])
    assert not bool(labels[11798]) and not bool(labels[12099])


def test_ucr_filename_that_breaks_the_naming_rule_is_rejected() -> None:
    """規則に合わない名前は黙って通さない。"""
    with pytest.raises(ValueError, match="命名規則"):
        ucr.parse_filename("119_UCR_Anomaly_ECG1_10000_11800.txt")


def test_ucr_subset_matches_the_manifest_rows() -> None:
    """採用サブセット8系列がマニフェストの行と一致し、名前も規則に合う。"""
    names = ucr.subset()
    assert len(names) == 8
    for name in names:
        spec = ucr.parse_filename(name)
        row = ucr.manifest().row("filename", name)
        assert spec.train_end == int(row["train_end"])
        assert spec.anomaly_start == int(row["anomaly_start"])
        assert spec.anomaly_end == int(row["anomaly_end"])
        assert spec.anomaly_end - spec.anomaly_start == int(row["anomaly_length"]), (
            "マニフェストの異常長と index 解釈が食い違っています"
        )
        assert "DISTORTED" not in name and "NOISE" not in name


def test_mgab_manifest_covers_all_ten_series() -> None:
    """MGAB のマニフェストが 10 系列ぶんの SHA256 と実測メタを持つ。"""
    manifest = mgab.manifest()
    assert len(manifest.rows) == 10
    assert set(mgab.SERIES) == {row["series"] for row in manifest.rows}
    for row in manifest.rows:
        assert len(row["sha256"]) == 64
        assert int(row["n_points"]) == 100000
        assert int(row["n_anomaly_points"]) == 4010
        assert int(row["n_anomaly_segments"]) == 10
        assert float(row["anomaly_rate"]) == pytest.approx(0.0401)


def test_ucr_manifest_records_the_archive_hash_and_size() -> None:
    """ZIP 全体の SHA256 とサイズも併記する (仕様 T2 実装メモ)。"""
    header = ucr.manifest().header
    assert len(header["archive_sha256"]) == 64
    assert int(header["archive_size_bytes"]) == 184066400
    assert header["source"] == ucr.ARCHIVE_URL


# --- キャッシュがあるときだけ走る検査 (D-60) ---------------------------------

_MGAB_CACHED = mgab.series_path("1").exists()
_UCR_CACHED = all(ucr.series_path(name).exists() for name in ucr.subset())


@pytest.mark.skipif(not _MGAB_CACHED, reason="MGAB のキャッシュが無い (D-60)")
def test_cached_mgab_series_matches_the_manifest() -> None:
    """キャッシュ済みの MGAB がマニフェストの実測値と一致する。"""
    series = mgab.load_series("1")
    row = mgab.manifest().row("series", "1")
    assert fetch.sha256_of(mgab.series_path("1")) == row["sha256"]
    assert series.n_steps == int(row["n_points"])
    assert int(np.count_nonzero(series.labels)) == int(row["n_anomaly_points"])
    assert series.n_anomalies == int(row["n_anomaly_segments"])
    assert int(np.count_nonzero(series.ignore)) == int(row["n_ignored"])
    assert series.anomaly_rate == pytest.approx(float(row["anomaly_rate"]))
    assert series.train_end == int(np.argmax(np.asarray(series.labels)))
    assert bool(np.asarray(series.ignore)[0]), (
        "MGAB の is_ignored は系列の先頭 (過渡区間) に立つ —— "
        "異常の近傍にしか付かないと決め打つと読み違える"
    )


@pytest.mark.skipif(not _UCR_CACHED, reason="UCR のキャッシュが無い (D-60)")
@pytest.mark.parametrize("filename", ucr.subset())
def test_cached_ucr_series_matches_the_manifest(filename: str) -> None:
    """キャッシュ済みの UCR がマニフェストの実測値と一致する (index 解釈込み)。"""
    row = ucr.manifest().row("filename", filename)
    assert fetch.sha256_of(ucr.series_path(filename)) == row["sha256"]
    series = ucr.load_series(filename)
    assert series.n_steps == int(row["n_points"])
    assert int(np.count_nonzero(series.labels)) == int(row["anomaly_length"])
    assert series.n_anomalies == 1
    assert series.train_end == int(row["train_end"])
    assert series.anomaly_rate == pytest.approx(float(row["anomaly_rate"]), abs=5e-6)

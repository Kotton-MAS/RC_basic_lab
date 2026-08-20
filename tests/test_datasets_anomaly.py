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
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import IO

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


def test_staged_write_uses_an_unpredictable_partial_name(tmp_path: Path) -> None:
    """``.part`` の名前は固定 (``f"{name}.part"``) ではなく予測不能である。

    固定名は、同じ ``data_dir`` に書ける別プロセス・別ユーザーが「どのパスを
    差し替えればよいか」を書き込み前から知っている TOCTOU の的になる
    (reviewer-security 指摘)。少なくとも旧来の固定名パターンとは一致せず、2回呼んでも
    毎回違う名前になることを固定する。
    """
    target = tmp_path / "probe.bin"
    with fetch._staged_write(target) as first:
        first_name = first.partial.name
    with fetch._staged_write(target) as second:
        second_name = second.partial.name
    assert first_name != second_name
    assert first_name != f"{target.name}.part"
    assert second_name != f"{target.name}.part"


def test_staged_write_commit_rejects_bytes_swapped_before_replace(
    tmp_path: Path,
) -> None:
    """``download()`` と ``extract_members()`` が共有する最終防衛線。

    (``_StagedSink.commit``)

    確定直前に一時ファイルの実体そのものが (パス越しに) 別の実体へ差し替え
    られても、fd から再照合するだけでは検出できない自己整合性の穴を
    fstat/stat の実体一致検査が塞ぎ、一時ファイルも確定先も残さない。
    """
    target = tmp_path / "probe.bin"
    swapped = tmp_path / "attacker-controlled.bin"
    swapped.write_bytes(b"attacker-controlled-bytes")
    expected = hashlib.sha256(b"legitimate-bytes").hexdigest()
    with (
        pytest.raises(fetch.ChecksumMismatchError, match="差し替え"),
        fetch._staged_write(target) as sink,
    ):
        sink.write(b"legitimate-bytes")
        os.replace(swapped, sink.partial)
        sink.commit(expected, error_cls=fetch.ChecksumMismatchError)
    assert not target.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_staged_write_operations_stay_pinned_to_the_directory_fd_after_the_path_is_replaced(
    tmp_path: Path,
) -> None:
    """D-67: 一時ファイルの作成・再照合・確定は ``target.parent`` を都度パス
    として開き直すのではなく、``_staged_write`` の冒頭で1回だけ取得した
    ``dir_fd`` に固定される。

    (reviewer-architecture 指摘、F-4-001) ``with`` に入った**後**で
    ``target.parent`` という名前を「元のディレクトリを改名して逃がし、その
    名前へ別ディレクトリへの symlink を差し込む」形で丸ごと差し替えても、
    書き込み・再照合・確定 (``os.replace``) は最初に取得した ``dir_fd`` が
    指す実ディレクトリ (差し替え後は改名先からしか辿れない) でだけ行われ、
    symlink の指す先には1バイトも書かれない。``target.parent`` を都度パス
    文字列として開き直す実装 (round3 以前) に戻すと、``os.stat``/``os.replace``
    が symlink の指す先を辿ってしまい、このテストは赤くなる (fixer が
    tempfile 上に複製した変異版で実測済み。.claude/decisions.yaml D-67 参照)。
    """
    parent = tmp_path / "sub"
    parent.mkdir()
    target = parent / "probe.bin"
    payload = b"pinned-to-dir-fd"
    expected = hashlib.sha256(payload).hexdigest()

    moved_original = tmp_path / "sub-moved-out-of-the-way"
    outside = tmp_path / "outside"
    outside.mkdir()

    with fetch._staged_write(target) as sink:
        # dir_fd はここまでで確定済み。この後で「target.parent」という名前
        # そのものを丸ごと差し替える (実ディレクトリは moved_original から
        # 引き続き辿れる。symlink 越しの outside は無関係の別ディレクトリ)。
        os.rename(parent, moved_original)
        os.symlink(outside, parent)
        sink.write(payload)
        sink.commit(expected, error_cls=fetch.DatasetError)

    assert (moved_original / target.name).read_bytes() == payload
    assert list(outside.iterdir()) == [], "symlink の指す先 (data_dir 外) に着弾した"
    assert not target.exists(), "sub は今 symlink であり、その先には何も無いはず"


# --- guard: ファイルへ書く経路の在り処 -------------------------------------
#
# round3 で reviewer-architecture / reviewer-security / reviewer-test の
# 3者から独立に指摘された穴: 旧2テストは「``os.replace`` の在り処」を
# ``ast.FunctionDef`` (async 不可) だけを名前一致で除外して測っていたため、
# (a) staging を使わず ``target.open("wb")`` で直接書く関数、
# (b) モジュールレベルの ``os.replace``、
# (c) ``from os import replace`` / ``import os as X`` 経由、
# (d) ``_StagedSink`` 以外のクラスの ``commit`` という名のメソッド、
# (e) ``async def`` 内の ``os.replace``
# のいずれもすり抜けた。ここでは「``os.replace`` の在り処」ではなく
# 「fetch.py 内でファイルへ書く経路の在り処」を全称で固定し、除外は名前一致
# ではなく ``_StagedSink`` クラス直下への所属で行う。


_WRITE_CAPABLE_MODULE_ATTRS = {
    ("os", "open"),
    ("os", "write"),
    ("os", "replace"),
    ("os", "rename"),
    ("os", "fdopen"),
    ("shutil", "move"),
    ("numpy", "save"),
}
"""``<module>.<attr>(...)`` の形でファイルへ書ける危険な組。

(round4 reviewer-test 指摘、F-4-016) ``getattr(os, "replace")(a, b)`` のような
**動的な属性解決**は追わない —— 静的 AST 解析は実行時にしか決まらない属性名を
原理的に解決できず、これを追いかけようとすると終わりがない。この集合は
「``<モジュール>.<属性名>`` の形でソースへ直接書かれているもの」に限る。
"""

_WRITE_CAPABLE_BARE_ATTRS = {"write_bytes", "write_text"}
"""受け手を問わず危険な属性呼び出し (``anything.write_bytes(...)`` 等)。

(round4 reviewer-test 指摘、F-4-016) ``.write(`` はここに含めない ——
``_StagedSink.write`` への正規の呼び出し (``sink.write(chunk)``) まで拾って
しまうため、``_write_capable_calls`` 側で ``_StagedSink`` を指す名前かどうかを
個別に判定する。
"""

_ALLOWED_WRITE_FUNCTION_NAMES = {"_staged_write", "_open_unique_temp_file"}
"""``_StagedSink`` のメソッド以外で、ファイル書き込み系呼び出しを許す関数名。

``_staged_write`` 自身 (``dir_fd`` の取得) と、その内部からしか呼ばれない
``_open_unique_temp_file`` (``os.open`` での一時ファイル作成) の2つだけ。
"""


def _is_write_mode_arg(value: ast.expr | None) -> bool:
    """``"wb"`` のような書き込み系モード文字列か (``"rb"`` は False)。"""
    if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
        return False
    return bool(set(value.value) & {"w", "a", "x", "+"})


_TRACKED_MODULES = {"os", "shutil", "io", "numpy"}
"""``.open``/``.save`` 等の危険な呼び出しを追跡するモジュール群。"""


def _module_import_aliases(tree: ast.Module) -> dict[str, str]:
    """``import os as o`` / ``import numpy as np`` の局所名 -> 実モジュール名。"""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _TRACKED_MODULES:
                    aliases[alias.asname or alias.name] = alias.name
    return aliases


def _from_import_targets(tree: ast.Module) -> dict[str, tuple[str, str]]:
    """``from os import replace as r`` の局所名 -> (モジュール名, 元の属性名)。"""
    targets: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _TRACKED_MODULES:
            for alias in node.names:
                targets[alias.asname or alias.name] = (node.module, alias.name)
    return targets


def _staged_sink_names(tree: ast.Module) -> set[str]:
    """``_StagedSink`` を指す名前の集合 (モジュール全体から集める)。

    (round4 reviewer-test 指摘、F-4-016) 名前一致 (例: 変数名が ``sink``)
    ではなく **構造** —— ``with _staged_write(...) as X`` の ``X``、または
    型注釈が ``_StagedSink`` である関数引数 —— で判定する。そうしないと
    ``sink`` という名前の別物 (``tempfile.NamedTemporaryFile()`` の戻り値等)
    まで正規の書き込みとして見逃してしまう一方、素朴な名前一致は
    ``_stream_to_file(response, sink: _StagedSink, ...)`` のように
    ``with`` を経由せず引数として受け取る正規の呼び出しを拾えない。
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                call = item.context_expr
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "_staged_write"
                    and isinstance(item.optional_vars, ast.Name)
                ):
                    names.add(item.optional_vars.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in (*node.args.args, *node.args.kwonlyargs):
                if (
                    isinstance(arg.annotation, ast.Name)
                    and arg.annotation.id == "_StagedSink"
                ):
                    names.add(arg.arg)
    return names


def _write_capable_calls(
    node: ast.AST,
    module_aliases: dict[str, str],
    from_targets: dict[str, tuple[str, str]],
    sink_names: set[str],
) -> list[str]:
    """``node`` 自身の中 (ネストした関数定義の内部も含む) にある、ファイルへ
    書き込める可能性のある呼び出しの説明を返す (空なら安全)。

    (round4 reviewer-test 指摘、F-4-016) 組み込み ``open`` の裸呼び出し・
    ``io.open`` (mode がレシーバに応じて第1/第2引数のいずれかに来る)・
    ``os.fdopen``・``*.write_text``・一時オブジェクトへの ``.write`` (例:
    ``tempfile.NamedTemporaryFile(...).write(...)``)・``numpy.save`` を追加で
    検出する。動的な ``getattr(os, "replace")`` は検出しない (原理的な限界。
    上記 ``_WRITE_CAPABLE_MODULE_ATTRS`` の docstring 参照)。
    """
    found: list[str] = []
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Attribute):
            if func.attr == "write":
                if isinstance(func.value, ast.Name) and func.value.id in sink_names:
                    continue  # _StagedSink.write() への正規の呼び出し
                found.append(".write(")
                continue
            if func.attr in _WRITE_CAPABLE_BARE_ATTRS:
                found.append(f".{func.attr}(")
                continue
            if func.attr == "open":
                receiver = None
                if isinstance(func.value, ast.Name):
                    receiver = module_aliases.get(func.value.id, func.value.id)
                # ``io.open(file, mode)`` は mode が第2引数、``Path.open(mode)``
                # 規約 (mode が第1引数) はそれ以外のレシーバに適用する。
                mode_index = 1 if receiver == "io" else 0
                mode_arg = (
                    call.args[mode_index] if len(call.args) > mode_index else None
                )
                for keyword in call.keywords:
                    if keyword.arg == "mode":
                        mode_arg = keyword.value
                if _is_write_mode_arg(mode_arg):
                    found.append(".open(書き込みモード)")
                continue
            if isinstance(func.value, ast.Name):
                receiver = module_aliases.get(func.value.id, func.value.id)
                if (receiver, func.attr) in _WRITE_CAPABLE_MODULE_ATTRS:
                    found.append(f"{receiver}.{func.attr}(")
        elif isinstance(func, ast.Name):
            if func.id == "open":
                # 組み込み open(file, mode=...) は ast.Name であり、
                # モジュール属性の追跡では原理的に捕まらない。
                mode_arg = call.args[1] if len(call.args) > 1 else None
                for keyword in call.keywords:
                    if keyword.arg == "mode":
                        mode_arg = keyword.value
                if _is_write_mode_arg(mode_arg):
                    found.append("open(書き込みモード) (builtin)")
                continue
            target = from_targets.get(func.id)
            if target is not None and target in _WRITE_CAPABLE_MODULE_ATTRS:
                found.append(f"{target[0]}.{target[1]}( (direct import)")
    return found


def _iter_functions_with_allowance(
    tree: ast.Module,
) -> Iterator[tuple[str, ast.AST, bool]]:
    """(関数名, ノード, 除外してよいか) を全 ``def``/``async def`` について返す。

    除外は名前一致ではなく **``_StagedSink`` クラス直下への所属**、または
    ``_ALLOWED_WRITE_FUNCTION_NAMES`` にある専用ヘルパー関数かどうかで決める
    (名前一致による除外は同名の別関数・別クラスで回避できる)。
    """

    def walk(
        node: ast.AST, in_staged_sink: bool
    ) -> Iterator[tuple[str, ast.AST, bool]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                nested = in_staged_sink or child.name == "_StagedSink"
                yield from walk(child, nested)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                allowed = in_staged_sink or child.name in _ALLOWED_WRITE_FUNCTION_NAMES
                yield (child.name, child, allowed)
                yield from walk(child, in_staged_sink)
            else:
                yield from walk(child, in_staged_sink)

    yield from walk(tree, False)


def _module_level_statements(tree: ast.Module) -> list[ast.stmt]:
    """どの関数・クラスにも属さないモジュールレベルの文 (import 文を除く)。"""
    return [
        stmt
        for stmt in tree.body
        if not isinstance(
            stmt,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Import,
                ast.ImportFrom,
            ),
        )
    ]


def _offending_write_paths(tree: ast.Module) -> list[str]:
    """``tree`` 内で、許可されていない場所にあるファイル書き込み系呼び出しの
    在り処 (関数名。モジュールレベルは ``"<module>"``) を返す。
    """
    module_aliases = _module_import_aliases(tree)
    from_targets = _from_import_targets(tree)
    sink_names = _staged_sink_names(tree)
    offenders: list[str] = []
    for stmt in _module_level_statements(tree):
        if _write_capable_calls(stmt, module_aliases, from_targets, sink_names):
            offenders.append("<module>")
    for name, node, allowed in _iter_functions_with_allowance(tree):
        if allowed:
            continue
        if _write_capable_calls(node, module_aliases, from_targets, sink_names):
            offenders.append(name)
    return offenders


def test_every_write_capable_path_in_fetch_goes_through_staged_write() -> None:
    """fetch.py 内でファイルへ書く経路の在り処を全称で固定する。

    (reviewer-architecture  / reviewer-security  / reviewer-test
     指摘) 「``os.replace`` の在り処」ではなく「ファイルへ書く経路の
    在り処」を測ることで、``target.open("wb")`` のような staging を経由しない
    直接書き込み・モジュールレベルの呼び出し・``from os import replace`` /
    ``import os as X`` 経由・``async def`` 内・``_StagedSink`` 以外のクラスの
    同名メソッド、のいずれが増えても機械的に落ちる
    (``test_the_write_path_guard_detects_known_bypasses`` がこの主張自体を
    固定する)。
    """
    tree = ast.parse(
        Path(fetch.__file__).read_text(encoding="utf-8"), filename=fetch.__file__
    )
    offenders = _offending_write_paths(tree)
    assert offenders == [], (
        "_StagedSink のメソッド (write/commit) と _staged_write/"
        f"_open_unique_temp_file 以外でファイルへ書く経路があります: {offenders}"
    )


_WRITE_PATH_BYPASS_SOURCES = {
    "direct_open_write_mode": (
        "def sneaky(path):\n"
        '    with path.open("wb") as handle:\n'
        '        handle.write(b"data")\n'
    ),
    "write_bytes": ('def sneaky(path):\n    path.write_bytes(b"data")\n'),
    "module_level_replace": ('import os\nos.replace("a", "b")\n'),
    "aliased_import_module": (
        "import os as o\ndef sneaky():\n    o.replace('a', 'b')\n"
    ),
    "from_import_direct": (
        "from os import replace\ndef sneaky():\n    replace('a', 'b')\n"
    ),
    "method_named_commit_outside_staged_sink": (
        "class NotStagedSink:\n    def commit(self):\n        os.replace('a', 'b')\n"
    ),
    "async_function": ("async def sneaky():\n    os.replace('a', 'b')\n"),
    "shutil_move": ("def sneaky():\n    shutil.move('a', 'b')\n"),
    "os_rename": ("def sneaky():\n    os.rename('a', 'b')\n"),
    "builtin_open_write_mode": (
        'def sneaky(path):\n    handle = open(path, "wb")\n    handle.close()\n'
    ),
    "io_open_write_mode": (
        'import io\ndef sneaky(path):\n    handle = io.open(path, "wb")\n'
        "    handle.close()\n"
    ),
    "os_fdopen": (
        'def sneaky(fd):\n    handle = os.fdopen(fd, "wb")\n    handle.close()\n'
    ),
    "path_write_text": ('def sneaky(path):\n    path.write_text("data")\n'),
    "namedtemporaryfile_write": (
        "import tempfile\n"
        "def sneaky():\n"
        '    tempfile.NamedTemporaryFile().write(b"data")\n'
    ),
    "numpy_save": ("import numpy\ndef sneaky(path, arr):\n    numpy.save(path, arr)\n"),
}
"""round3/round4 で reviewer が実測した回避クラス (M1/M3/M4/M5/M8 系、および
round4 F-4-016 の7パターン) をソース文字列として与え、``_offending_write_paths``
がそれぞれを実際に検出できることを固定する (guard の guard)。

(round4 reviewer-test 指摘、F-4-016) ``getattr(os, "replace")(a, b)`` は
このカタログに**含めない** —— 動的な属性解決は静的 AST 解析の原理的な限界で
あり検出しない、という決定そのものなので、検出できないことを前提に除外して
ある (含めると本テストが必ず落ちる)。
"""


@pytest.mark.parametrize("label", sorted(_WRITE_PATH_BYPASS_SOURCES))
def test_the_write_path_guard_detects_known_bypasses(label: str) -> None:
    """この guard 自体が回避形を検出できることを固定する (guard の guard)。

    guard の述語をそのまま合成ソースへ適用して検出できることを確認しない
    限り、「全称」を名乗る docstring は主張だけで裏付けが無い
    (guard 自体が回避形を検出できることを実測で示す必要がある、という指摘そのもの)。
    """
    tree = ast.parse(_WRITE_PATH_BYPASS_SOURCES[label])
    offenders = _offending_write_paths(tree)
    assert offenders != [], f"{label} の回避形が検出されませんでした"


def test_every_function_that_stages_a_write_also_commits_it() -> None:
    """``_staged_write`` を呼ぶ関数は、その ``with ... as X`` の ``X`` に対して
    必ず ``.commit(`` を呼ぶ (全称)。

    (reviewer-test  指摘) 旧版は「本文中のどこかに ``.commit(`` が
    1つでもあれば真」という判定だったため、``_staged_write`` の結果とは無関係
    な decoy オブジェクトへの ``.commit()`` 呼び出しを1つ足すだけですり抜け
    られた。``with _staged_write(...) as X:`` の ``X`` と、``.commit(`` の
    呼び出し元 (``Call.func.value``) が同一識別子であることまで確認する。
    """
    tree = ast.parse(
        Path(fetch.__file__).read_text(encoding="utf-8"), filename=fetch.__file__
    )
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        staged_names: set[str] = set()
        for sub in ast.walk(node):
            if not isinstance(sub, ast.With):
                continue
            for item in sub.items:
                call = item.context_expr
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "_staged_write"
                    and isinstance(item.optional_vars, ast.Name)
                ):
                    staged_names.add(item.optional_vars.id)
        if not staged_names:
            continue
        committed = any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "commit"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id in staged_names
            for call in ast.walk(node)
        )
        if not committed:
            offenders.append(node.name)
    assert offenders == [], (
        "_staged_write を呼ぶが、その sink に対して commit() を呼ばない関数: "
        f"{offenders}"
    )


def test_decoy_commit_call_does_not_satisfy_the_staged_write_guard() -> None:
    """``_staged_write`` の sink とは無関係な ``.commit()`` を decoy として
    持つ関数は、``test_every_function_that_stages_a_write_also_commits_it``
    と同じ述語で検出されることを固定する (reviewer-test が実測したシナリオそのもの)。
    """
    source = (
        "def sneaky(target, other):\n"
        "    with _staged_write(target) as sink:\n"
        "        sink.write(b'data')\n"
        "        other.commit()\n"  # decoy: sink ではなく other への呼び出し
        "        os.replace(sink.partial, target)\n"
    )
    tree = ast.parse(source)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        staged_names: set[str] = set()
        for sub in ast.walk(node):
            if not isinstance(sub, ast.With):
                continue
            for item in sub.items:
                call = item.context_expr
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "_staged_write"
                    and isinstance(item.optional_vars, ast.Name)
                ):
                    staged_names.add(item.optional_vars.id)
        if not staged_names:
            continue
        committed = any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "commit"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id in staged_names
            for call in ast.walk(node)
        )
        if not committed:
            offenders.append(node.name)
    assert offenders == ["sneaky"], (
        f"decoy の .commit() 呼び出しで真の commit 無しがすり抜けています: {offenders}"
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


def test_extract_members_is_rejected_when_the_part_file_is_swapped_mid_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``extract_members`` も、書き込み完了直後の ``.part`` 差し替えを検出する。

    ``download()`` 側にしか存在しなかった振る舞いテストの対 (reviewer-test 指摘)。
    AST の存在確認テストだけでは、確定直前の再照合を無力化
    する変異を ``_extract_member`` に注入しても全テストが green のまま通る
    ことが実測されている —— この振る舞いテストがその回帰検知の穴を塞ぐ。
    """
    legit = b"LEGIT-" * 200_000
    evil = b"EVIL-" * 200_000
    archive = tmp_path / "bundle.zip"
    _make_zip(archive, {"series.bin": legit})
    swapped = tmp_path / ".attacker-payload.bin"
    real_open = zipfile.ZipFile.open

    class _RacingMemberFile:
        def __init__(self, inner: IO[bytes]) -> None:
            self._inner = inner
            self._swapped = False

        def read(self, amt: int = -1) -> bytes:
            chunk = self._inner.read(amt)
            if chunk and not self._swapped:
                self._swapped = True
                # ``download`` 側の攻撃 (os.replace で inode ごと差し替え) と同型。
                for part in tmp_path.rglob("*.part"):
                    swapped.write_bytes(evil)
                    os.replace(swapped, part)
            return chunk

        def __enter__(self) -> _RacingMemberFile:
            return self

        def __exit__(self, *exc_info: object) -> None:
            self._inner.close()

    def racing_open(
        self: zipfile.ZipFile, name: str | zipfile.ZipInfo
    ) -> _RacingMemberFile:
        return _RacingMemberFile(real_open(self, name))

    monkeypatch.setattr(zipfile.ZipFile, "open", racing_open)

    with pytest.raises(UnsafeArchiveMemberError, match="差し替え"):
        fetch.extract_members(
            archive, ["series.bin"], tmp_path / "out", data_dir=tmp_path
        )
    assert not (tmp_path / "out" / "series.bin").exists()
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


def test_resolve_under_rejects_a_relative_path_deeper_than_one_directory(
    tmp_path: Path,
) -> None:
    """``relative_path`` は1階層 (ディレクトリ1つ + ファイル名) までしか受け
    付けない (reviewer-security 指摘、F-4-011)。

    ``_staged_write`` の ``os.O_NOFOLLOW`` は ``target.parent`` という**パス
    の最終成分だけ**を守るため、2階層以上になると中間成分の symlink 差し替え
    を検出できない。manifest が実際にこの形しか使わないことを不変条件として
    ここで固定する。
    """
    with pytest.raises(UnsafeArchiveMemberError, match="ディレクトリ1つ"):
        fetch.resolve_under(tmp_path, "a/b/escape.csv")


@pytest.mark.parametrize("relative", ["archive.zip", "mgab/1.csv"])
def test_resolve_under_accepts_paths_at_most_one_directory_deep(
    relative: str, tmp_path: Path
) -> None:
    """1階層 (``archive.zip``) と「ディレクトリ1つ + ファイル名」
    (``mgab/1.csv``) はどちらも受理する (manifest の実際の形と一致)。"""
    resolved = fetch.resolve_under(tmp_path, relative)
    assert resolved.is_relative_to(tmp_path.resolve())


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


def test_manifest_relative_paths_are_at_most_one_directory_deep() -> None:
    """全 manifest 行の ``relative_path`` が1階層 (ディレクトリ1つ + ファイル名)
    以下であることを不変条件として固定する (reviewer-security 指摘、F-4-011)。

    ``resolve_under`` がこれを拒むようになった (2階層以上は
    ``UnsafeArchiveMemberError``) 以上、実データの側もこの形しか使わない
    ことをここで確認しておく —— manifest 側が2階層以上になった場合、
    ``download()``/``ensure_file()`` が壊れるのではなく最初から拒否される
    ことの裏付け。
    """
    for row in mgab.manifest().rows:
        assert len(PurePosixPath(row["relative_path"]).parts) <= 2
    for row in ucr.manifest().rows:
        assert len(PurePosixPath(row["relative_path"]).parts) <= 2
    assert len(PurePosixPath(ucr.ARCHIVE_RELATIVE_PATH).parts) <= 2


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

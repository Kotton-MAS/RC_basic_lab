"""成果物 (``results/``) のバイト単位の指紋を読み書きする (D-74)。

リファクタリングの合否判定に使う。「振る舞いを変えずに削れたか」は
**成果物が1バイトも変わっていないこと**でしか言えないが、この判定は
これまで手作業で行われており、実際に ``shasum`` (SHA-1) と ``sha256`` の
取り違えが4サイクル続いた記録がある
(docs/process/agent-operations-retrospective.md 2 節)。

さらに悪いことに、SubagentStop の自動コミットが作業ツリーを丸ごと拾うため、
成果物が変わってもコミットされて ``git status`` は綺麗なままになる。
基準 ref からの差分を見るには基準 ref を覚えている必要があり、
**気づかないまま通る経路**が残っていた。

そこで指紋をリポジトリにコミットし、``make ci`` で毎回照合する。
成果物を意図的に変えたときだけ、明示的に指紋を書き直す:

    make artifacts-manifest

再生成はレビューで必ず目に入る (差分が出る) ので、
「なぜ成果物が変わったか」を説明せずに通すことができない。

## 指紋を2本持つ理由

バイト指紋 (``sha256``) だけだと、**再生成のたびに全行が変わる**。実測で
23 枚の CSV のうち 15 枚と JSON 5 枚が実行時間を含むためで、
``results/03_capacity/capacity.csv`` は 39 列中 4 列が ``wall_time_*`` である。
数値がまったく同じでも指紋は必ず動くので、レビューで差分を見たときに
**本物の変化が時間のノイズに埋もれる**。

そこで内容指紋 (``content_sha256``) を別に持つ。時間と時刻の列を除いてから
ハッシュするので、**これが動いたときだけ実質的な変化がある**。
バイト指紋は残す —— そちらは「誰かが成果物を手で編集していないこと」を
測っており、時間列の手編集も検出できる必要がある。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
MANIFEST_PATH = Path(__file__).with_name("artifact_manifest.csv")
MANIFEST_COLUMNS = ("path", "sha256", "bytes", "content_sha256")

TIMING_COLUMN_PREFIX = "wall_time"
"""CSV の時間列の接頭辞。内容指紋から除く。"""

VOLATILE_JSON_KEYS = frozenset({"wall_time_s", "timestamp_utc", "commit"})
"""``meta.json`` の**実行ごとに必ず変わる**キー。**深さを問わず**除く (D-141)。

``commit`` も除く。当初は「どのコミットで作られたかは内容の一部」として
残したが、実測すると再生成のたびに ``meta.json`` が「内容が変わった」側に
出てしまい、**分類の意味が無くなった** (commit は必ず動くため)。

除いても弱くならない。commit がそろっていることは ``test_cycle_hygiene`` が
``meta.json`` を直接読んで見ており、内容指紋を経由していない。

**深さを問わず落とす** (D-141)。03 の ``threshold_comparison.wall_time_s``
のように、同じ名前が入れ子の中にも出るためである。
"""

VOLATILE_JSON_SUBTREES = frozenset({"wall_time_breakdown"})
"""中身が丸ごと実測値である節。**部分木ごと**除く (D-141)。

``wall_time_breakdown`` の下は ``capacity_s`` / ``stability_s`` のように
節ごとに名前が違う。名前の規則 (``*_s`` で終わる) で落とす案は**採れない**
—— 05 の ``total_budget_s`` と ``wall_time_budget_s.*`` は**設定値**であり、
落とすと予算を変えたことが内容指紋に出なくなる。測っている値と設定した値が
同じ接尾辞を共有している以上、名前ではなく**どの節にあるか**で決める。

これを入れる前は 04 / 05 の ``meta.json`` が再生成のたびに「内容が変わった」
側に出ていた。実行時間しか動いていないのに毎回説明を要する行が2件出ると、
**本当に説明を要する行を見落とす**。
"""


def artifact_paths() -> tuple[Path, ...]:
    """``results/`` 配下の全ファイル (リポジトリ相対で安定した順)。"""
    return tuple(sorted(path for path in RESULTS_DIR.rglob("*") if path.is_file()))


def fingerprint(path: Path) -> tuple[str, str, int, str]:
    """1ファイルの (相対パス, SHA256, バイト数, 内容 SHA256)。

    **SHA-256 である**こと。``shasum`` の既定は SHA-1 で、過去に取り違えが
    4サイクル続いている。
    """
    payload = path.read_bytes()
    relative = path.relative_to(REPO_ROOT).as_posix()
    return (
        relative,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        content_digest(path, payload),
    )


def content_digest(path: Path, payload: bytes) -> str:
    """実行のたびに変わる値を除いた指紋。

    CSV は ``wall_time*`` の列を落とし、``meta.json`` は実行時間・時刻・commit
    を落としてからハッシュする。それ以外 (PNG など) はバイトそのまま。

    **PNG は正規化しない。** footnote に commit を焼き込んでいる (FIG-6 / D-87)
    ので、画素から commit だけを抜くことはできない。再生成して一致するかは
    ``tests/test_golden.py`` が commit を固定して確かめている。

    Args:
        path: 対象のファイル (拡張子で扱いを決める)。
        payload: 読み込み済みのバイト列 (二度読みしない)。

    Returns:
        16 進の SHA-256。
    """
    if path.suffix == ".csv":
        return hashlib.sha256(_csv_without_timing(payload)).hexdigest()
    if path.suffix == ".json":
        return hashlib.sha256(_json_without_timing(payload)).hexdigest()
    return hashlib.sha256(payload).hexdigest()


def _csv_without_timing(payload: bytes) -> bytes:
    """``wall_time*`` 列を落とした CSV を返す (列が無ければそのまま)。"""
    text = payload.decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return payload
    keep = [
        index
        for index, name in enumerate(header)
        if not name.startswith(TIMING_COLUMN_PREFIX)
    ]
    if len(keep) == len(header):
        return payload
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([header[index] for index in keep])
    writer.writerows([row[index] for index in keep] for row in reader)
    return buffer.getvalue().encode("utf-8")


def _json_without_timing(payload: bytes) -> bytes:
    """実行ごとに変わるキーを**深さを問わず**落とした JSON を返す (D-141)。"""
    parsed: object = json.loads(payload.decode("utf-8"))
    if not isinstance(parsed, dict):
        return payload
    stripped = _without_volatile(parsed)
    return json.dumps(stripped, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _without_volatile(node: object) -> object:
    """実測値のキーと節を落とした複製を返す (辞書と配列を再帰的に歩く)。"""
    if isinstance(node, dict):
        mapping: dict[str, object] = node
        return {
            key: _without_volatile(value)
            for key, value in mapping.items()
            if key not in VOLATILE_JSON_KEYS and key not in VOLATILE_JSON_SUBTREES
        }
    if isinstance(node, list):
        elements: list[object] = node
        return [_without_volatile(element) for element in elements]
    return node


def volatile_json_paths(payload: bytes) -> tuple[str, ...]:
    """その JSON で内容指紋から落ちるパス (**落ちすぎを検査するため**)。

    落とす規則を広げると、設定を変えたのに内容指紋が動かない状態を静かに
    作れてしまう。何が落ちているかを列挙できるようにしておき、
    ``test_artifact_content_digest`` が実際の成果物で全件を突き合わせる。
    """
    parsed: object = json.loads(payload.decode("utf-8"))
    return tuple(sorted(_volatile_paths(parsed, "")))


def _volatile_paths(node: object, prefix: str) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        mapping: dict[str, object] = node
        for key, value in mapping.items():
            path = f"{prefix}{key}"
            if key in VOLATILE_JSON_KEYS or key in VOLATILE_JSON_SUBTREES:
                found.add(path)
            else:
                found |= _volatile_paths(value, f"{path}.")
    elif isinstance(node, list):
        elements: list[object] = node
        for index, element in enumerate(elements):
            found |= _volatile_paths(element, f"{prefix}[{index}].")
    return found


def read_manifest() -> dict[str, tuple[str, int, str]]:
    """コミットされた指紋 (相対パス -> (SHA256, バイト数, 内容 SHA256))。"""
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            row["path"]: (row["sha256"], int(row["bytes"]), row["content_sha256"])
            for row in reader
        }


def write_manifest() -> int:
    """現在の成果物から指紋を書き直す。書いた行数を返す。"""
    rows = [fingerprint(path) for path in artifact_paths()]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    written = write_manifest()
    sys.stdout.write(
        f"{MANIFEST_PATH.relative_to(REPO_ROOT)} に {written} 件を書きました\n"
    )

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
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
MANIFEST_PATH = Path(__file__).with_name("artifact_manifest.csv")
MANIFEST_COLUMNS = ("path", "sha256", "bytes")


def artifact_paths() -> tuple[Path, ...]:
    """``results/`` 配下の全ファイル (リポジトリ相対で安定した順)。"""
    return tuple(sorted(path for path in RESULTS_DIR.rglob("*") if path.is_file()))


def fingerprint(path: Path) -> tuple[str, str, int]:
    """1ファイルの (相対パス, SHA256, バイト数)。

    **SHA-256 である**こと。``shasum`` の既定は SHA-1 で、過去に取り違えが
    4サイクル続いている。
    """
    payload = path.read_bytes()
    relative = path.relative_to(REPO_ROOT).as_posix()
    return relative, hashlib.sha256(payload).hexdigest(), len(payload)


def read_manifest() -> dict[str, tuple[str, int]]:
    """コミットされた指紋 (相対パス -> (SHA256, バイト数))。"""
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["path"]: (row["sha256"], int(row["bytes"])) for row in reader}


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

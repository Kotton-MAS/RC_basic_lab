"""``make data-05`` の実体 —— 実データ源をキャッシュへ取ってくる薄い CLI.

使い方::

    uv run python -m rc_basics_lab.datasets.cli --dataset mgab
    uv run python -m rc_basics_lab.datasets.cli --dataset all --data-dir data/05_anomaly

進捗は ``print`` ではなく ``logging`` で出す (ruff T20)。実験スクリプト
(``experiments/04_chaotic_freerun/run_04.py``) と同じ薄さに保ち、取得・照合の
本体は ``datasets/fetch.py`` にある。

**pytest はこの経路を呼ばない** (D-60)。テストはネットワークに一切触れない。
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rc_basics_lab.datasets import mgab, ucr
from rc_basics_lab.datasets.fetch import DEFAULT_DATA_DIR

logger = logging.getLogger("rc_basics_lab.datasets")

DATASET_CHOICES = ("mgab", "ucr", "all")


@dataclass(frozen=True, slots=True)
class Args:
    """コマンドライン引数。"""

    dataset: str
    data_dir: Path


def parse_args(argv: Sequence[str] | None = None) -> Args:
    """引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="実験05 の外部データセットを取得し SHA256 で照合する (D-58)"
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="mgab",
        help="取得するデータセット (既定: mgab)",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"キャッシュ先 (既定: {DEFAULT_DATA_DIR})",
    )
    namespace = parser.parse_args(argv)
    return Args(dataset=str(namespace.dataset), data_dir=Path(str(namespace.data_dir)))


def main(argv: Sequence[str] | None = None) -> int:
    """データセットを取得する。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    if args.dataset in ("mgab", "all"):
        logger.info("MGAB を取得します (%s, %s)", mgab.LICENSE, args.data_dir)
        paths = mgab.fetch(data_dir=args.data_dir)
        logger.info("MGAB: %d 系列を照合しました", len(paths))
    if args.dataset in ("ucr", "all"):
        logger.info(
            "UCR を取得します (ライセンス %s / ZIP 184 MB, %s)",
            ucr.LICENSE,
            args.data_dir,
        )
        paths = ucr.fetch(data_dir=args.data_dir)
        logger.info("UCR: %d 系列を照合しました", len(paths))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 経路
    raise SystemExit(main())


__all__ = ["DATASET_CHOICES", "Args", "main", "parse_args"]

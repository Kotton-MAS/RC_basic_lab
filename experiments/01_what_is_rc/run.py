"""実験1-A: 3ベースライン比較を1コマンドで再生成する.

使い方::

    uv run python experiments/01_what_is_rc/run.py \
        --config experiments/01_what_is_rc/config.yaml

``--out`` (既定 ``results``) に ``comparison.csv`` と ``meta.json`` を書く。
実測 wall time は ``meta.json`` の ``wall_time_s`` に記録する (性能受け入れ基準)。
進捗は ``print`` ではなく ``logging`` で出す (ruff T20)。
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rc_basics_lab.config import load_config
from rc_basics_lab.experiment.report import (
    COMPARISON_CSV,
    META_JSON,
    write_comparison_csv,
    write_meta,
)
from rc_basics_lab.experiment.runner import run_experiment

logger = logging.getLogger("rc_basics_lab.experiments.01_what_is_rc")

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_OUT = Path("results")


@dataclass(frozen=True, slots=True)
class Args:
    """コマンドライン引数。"""

    config: Path
    out: Path


def parse_args(argv: Sequence[str] | None = None) -> Args:
    """引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="実験1-A: 線形 / 遅延線 / ESN の比較を実行し CSV を出力する"
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"実験設定 YAML (既定: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"出力ディレクトリ (既定: {DEFAULT_OUT})",
    )
    namespace = parser.parse_args(argv)
    return Args(config=Path(str(namespace.config)), out=Path(str(namespace.out)))


def main(argv: Sequence[str] | None = None) -> int:
    """実験を実行し、``comparison.csv`` と ``meta.json`` を書く。"""
    args = parse_args(argv)
    config = load_config(args.config)
    logger.info(
        "設定を読み込みました: %s (n_replicates=%d)", args.config, config.n_replicates
    )

    started = time.perf_counter()
    rows = run_experiment(config)
    wall_time_s = time.perf_counter() - started

    csv_path = write_comparison_csv(rows, args.out / COMPARISON_CSV)
    meta_path = write_meta(config, wall_time_s, len(rows), args.out / META_JSON)
    logger.info(
        "完了: %d 行を %s に、メタ情報を %s に書き出しました (wall_time=%.1fs)",
        len(rows),
        csv_path,
        meta_path,
        wall_time_s,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())

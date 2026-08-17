"""実験1-A/1-B: 図まで含めて1コマンドで再生成する.

使い方::

    uv run python experiments/01_what_is_rc/run.py \
        --config experiments/01_what_is_rc/config.yaml

``--out`` (既定 ``results``) に ``comparison.csv`` / ``comparison_summary.csv`` /
``fig_comparison.png`` / ``fig_state_space.png`` / ``meta.json`` の5点を書く
(受け入れ条件5)。
実測 wall time は ``meta.json`` の ``wall_time_s`` に記録する (性能受け入れ基準)。
進捗は ``print`` ではなく ``logging`` で出す (ruff T20)。

計算と書き出しの本体は ``rc_basics_lab.experiment.pipeline.run_and_report`` にあり、
ここは引数解析だけの薄い層である (``main.py --experiment 01`` と同じ経路を通す)。
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rc_basics_lab.config import load_config
from rc_basics_lab.experiment.pipeline import run_and_report

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
        description="実験1: 線形 / 遅延線 / ESN の比較と PCA 図を1コマンドで作る"
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
    """実験を実行し、CSV2枚・図2枚・meta.json を書く。"""
    args = parse_args(argv)
    config = load_config(args.config)
    logger.info(
        "設定を読み込みました: %s (n_replicates=%d)", args.config, config.n_replicates
    )
    run_and_report(config, args.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())

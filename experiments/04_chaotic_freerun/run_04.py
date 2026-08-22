"""実験04 (カオス時系列の自由走行予測) の成果物を1コマンドで再生成する.

使い方::

    uv run python experiments/04_chaotic_freerun/run_04.py \
        --config experiments/04_chaotic_freerun/config.yaml

``--out`` (既定 ``results/04_chaotic_freerun``) に CSV5枚 (``onestep.csv`` /
``freerun.csv`` / ``freerun_profile.csv`` / ``stability.csv`` /
``capacity.csv``)・図5枚・``meta.json`` を書く。成果物の一覧の単一の真実は
``experiment/freerun_pipeline.py`` の ``FREERUN_ARTIFACTS``。実測 wall time は
``meta.json`` の ``wall_time_s`` と ``wall_time_breakdown`` に区間ごとに記録
する (性能受け入れ基準)。進捗は ``print`` ではなく ``logging`` で出す
(ruff T20)。

計算と書き出しの本体は
``rc_basics_lab.experiment.freerun_pipeline.run_and_report_freerun`` にあり、
ここは引数解析だけの薄い層である (``main.py --experiment 04`` と同じ経路)。
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rc_basics_lab.config import Chaos04Config, load_config_as
from rc_basics_lab.experiment.freerun_pipeline import run_and_report_freerun

logger = logging.getLogger("rc_basics_lab.experiments.04_chaotic_freerun")

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_OUT = Path("results/04_chaotic_freerun")


@dataclass(frozen=True, slots=True)
class Args:
    """コマンドライン引数。"""

    config: Path
    out: Path


def parse_args(argv: Sequence[str] | None = None) -> Args:
    """引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="実験04: カオス時系列の自由走行予測 (4-A / 4-B / 4-C / 4-D)"
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
    """実験を実行し、CSV5枚・図5枚・``meta.json`` を書く。"""
    args = parse_args(argv)
    config = load_config_as(args.config, Chaos04Config)
    logger.info(
        "設定を読み込みました: %s (Lorenz T=%d sample_interval=%d / "
        "MG T=%d / n_replicates=%d)",
        args.config,
        config.lorenz.length,
        config.lorenz.sample_interval,
        config.base.mackey_glass.length,
        config.base.n_replicates,
    )
    run_and_report_freerun(config, args.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())

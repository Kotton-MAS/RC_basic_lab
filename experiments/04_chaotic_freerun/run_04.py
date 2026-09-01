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

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rc_basics_lab.cli import (
    ExperimentArgs,
    build_parser,
    default_out_for,
    parse_experiment_args,
)
from rc_basics_lab.config import (
    Chaos04Config,
    MackeyGlassTask,
    load_config_as,
    require_task,
)
from rc_basics_lab.experiment.freerun_pipeline import run_and_report_freerun

logger = logging.getLogger("rc_basics_lab.experiments.04_chaotic_freerun")

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_OUT = default_out_for(DEFAULT_CONFIG)
"""``--out`` 未指定時の出力先。``main.py`` と同じ関数から導く。"""


@dataclass(frozen=True, slots=True)
class Args:
    """コマンドライン引数 (共通分は ``ExperimentArgs``)。"""

    common: ExperimentArgs


def parse_args(argv: Sequence[str] | None = None) -> Args:
    """引数を解析する。共通フラグは ``rc_basics_lab.cli`` が持つ。"""
    parser = build_parser("実験4: カオス時系列の自由走行予測", DEFAULT_CONFIG)
    common, _ = parse_experiment_args(parser, argv)
    return Args(common=common)


def main(argv: Sequence[str] | None = None) -> int:
    """実験を実行し、CSV5枚・図5枚・``meta.json`` を書く。"""
    args = parse_args(argv)
    config = load_config_as(
        args.common.config,
        Chaos04Config,
        preset=args.common.preset,
        overrides=args.common.overrides,
    )
    logger.info(
        "設定を読み込みました: %s (Lorenz T=%d sample_interval=%d / "
        "MG T=%d / n_replicates=%d)",
        args.common.config,
        config.lorenz.length,
        config.lorenz.sample_interval,
        require_task(config.base, MackeyGlassTask, "04 の実行ログ").params.length,
        config.base.n_replicates,
    )
    run_and_report_freerun(config, args.common.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())

"""実験1-A/1-B: 図まで含めて1コマンドで再生成する.

使い方::

    # 成果物を再生成する (results/ へ書くのはこの経路だけ)
    make figures-01

    # 手元で試す (既定の出力先は scratch/01_what_is_rc)
    uv run python experiments/01_what_is_rc/run.py --preset quick
    uv run python experiments/01_what_is_rc/run.py \\
        --set tasks.mackey_glass.reservoir.n_units=50

``--out`` (既定 ``scratch/01_what_is_rc``) に ``comparison.csv`` /
``comparison_summary.csv`` / ``fig_comparison.png`` / ``fig_state_space.png`` /
``meta.json`` の5点を書く
(受け入れ条件5)。
実測 wall time は ``meta.json`` の ``wall_time_s`` に記録する (性能受け入れ基準)。
進捗は ``print`` ではなく ``logging`` で出す (ruff T20)。

計算と書き出しの本体は ``rc_basics_lab.experiment.pipeline.run_and_report`` にあり、
ここは引数解析だけの薄い層である (``main.py --experiment 01`` と同じ経路を通す)。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from rc_basics_lab.cli import build_parser, default_out_for, parse_experiment_args
from rc_basics_lab.config import load_config
from rc_basics_lab.experiment.pipeline import run_and_report

logger = logging.getLogger("rc_basics_lab.experiments.01_what_is_rc")

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_OUT = default_out_for(DEFAULT_CONFIG)
"""``--out`` 未指定時の出力先。``main.py`` と同じ関数から導く。"""


def main(argv: Sequence[str] | None = None) -> int:
    """実験を実行し、CSV2枚・図2枚・meta.json を書く。"""
    parser = build_parser(
        "実験1: 線形 / 遅延線 / ESN の比較と PCA 図を1コマンドで作る", DEFAULT_CONFIG
    )
    args, _ = parse_experiment_args(parser, argv)
    config = load_config(args.config, preset=args.preset, overrides=args.overrides)
    logger.info(
        "設定を読み込みました: %s (n_replicates=%d)%s%s",
        args.config,
        config.n_replicates,
        f" preset={args.preset.name}" if args.preset else "",
        f" set={list(args.overrides)}" if args.overrides else "",
    )
    run_and_report(config, args.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())

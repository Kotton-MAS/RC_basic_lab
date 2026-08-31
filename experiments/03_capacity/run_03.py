"""実験3-A/3-B/3-B': 成果物を1コマンドで再生成する.

使い方::

    uv run python experiments/03_capacity/run_03.py \
        --config experiments/03_capacity/config.yaml

ファイル名が 01 の ``run.py`` と揃っていないのは、**mypy がリポジトリ配下の
同名トップレベルモジュールを重複と見なして解析を止める**ため
(``Duplicate module named "run"``)。02 以降は ``run_<番号>.py`` を使う。

``--out`` (既定 ``results/03_capacity``) に ``capacity.csv`` /
``capacity_profile.csv`` / ``meta.json`` を書く (図4枚は 3b-1 の T3 が足す)。
``--length-sweep`` を付けたときは代わりに ``capacity_length.csv``
(系列長 T の掃引) だけを書く。実測 wall time は ``meta.json`` の
``wall_time_s`` と ``wall_time_breakdown`` に記録する (性能受け入れ基準)。
進捗は ``print`` ではなく ``logging`` で出す (ruff T20)。

計算と書き出しの本体は
``rc_basics_lab.experiment.capacity_pipeline.run_and_report_capacity`` にあり、
ここは引数解析だけの薄い層である (``main.py --experiment 03`` と同じ経路)。
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
from rc_basics_lab.config import Capacity03Config, load_config_as
from rc_basics_lab.experiment.capacity_pipeline import (
    run_and_report_capacity,
    run_and_report_length_sweep,
    run_and_report_symmetry_sweep,
)

logger = logging.getLogger("rc_basics_lab.experiments.03_capacity")

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_OUT = default_out_for(DEFAULT_CONFIG)
"""``--out`` 未指定時の出力先。``main.py`` と同じ関数から導く。"""


@dataclass(frozen=True, slots=True)
class Args:
    """コマンドライン引数 (共通分は ``ExperimentArgs``)。"""

    common: ExperimentArgs
    length_sweep: bool
    symmetry_sweep: bool


def parse_args(argv: Sequence[str] | None = None) -> Args:
    """引数を解析する。共通フラグは ``rc_basics_lab.cli`` が持つ。"""
    parser = build_parser(
        "実験3: 線形メモリ容量 (MC) と情報処理容量 (IPC) を測る", DEFAULT_CONFIG
    )
    parser.add_argument(
        "--length-sweep",
        action="store_true",
        help=(
            "本体の成果物は作らず、系列長 T の掃引 (capacity_length.csv) だけを"
            "再生成する (make saturation-03)"
        ),
    )
    parser.add_argument(
        "--symmetry-sweep",
        action="store_true",
        help=("駆動入力の対称性の掃引 (capacity_symmetry.csv) だけを再生成する"),
    )
    common, namespace = parse_experiment_args(parser, argv)
    return Args(
        common=common,
        length_sweep=bool(namespace.length_sweep),
        symmetry_sweep=bool(namespace.symmetry_sweep),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """実験を実行し、CSV2枚と meta.json を書く。

    ``--symmetry-sweep`` のときは 3-S の CSV だけを、``--length-sweep`` のときは
    系列長掃引の CSV だけを書く。**どちらも書いたら本体は走らせない**
    (どちらの分岐も return する)。本体と分けるのは、
    T=1e6 まで回すので単独で ``make figures-03`` の予算 (900 秒) を食い潰す
    うえ、記事に載る成果物ではなく「容量が足りないのか T が足りないのか」を
    切り分ける補助実験だからである。
    """
    args = parse_args(argv)
    config = load_config_as(
        args.common.config,
        Capacity03Config,
        preset=args.common.preset,
        overrides=args.common.overrides,
    )
    logger.info(
        "設定を読み込みました: %s (3-A N=%d T=%d / 3-B N=%d T=%d / n_replicates=%d)",
        args.common.config,
        config.mc_sweep.n_units,
        config.mc_sweep.n_steps,
        config.ipc_sweep.n_units,
        config.ipc_sweep.n_steps,
        config.reservoir.n_replicates,
    )
    if args.symmetry_sweep:
        run_and_report_symmetry_sweep(config, args.common.out)
        return 0
    if args.length_sweep:
        run_and_report_length_sweep(config, args.common.out)
        return 0
    run_and_report_capacity(config, args.common.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())

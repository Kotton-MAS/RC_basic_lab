"""実験2-A/2-B/2-C: 図まで含めて1コマンドで再生成する.

使い方::

    uv run python experiments/02_esp_and_dynamics/run_02.py \
        --config experiments/02_esp_and_dynamics/config.yaml

ファイル名が 01 の ``run.py`` と揃っていないのは、**mypy がリポジトリ配下の
同名トップレベルモジュールを重複と見なして解析を止める**ため
(``Duplicate module named "run"``)。実験ディレクトリ名は数字始まりで
パッケージにできず、``[tool.mypy]`` の設定は変更しない制約があるので、
02 以降は ``run_<番号>.py`` を使う。01 の ``run.py`` は既存の公開コマンドなので
そのまま残す。

``--out`` (既定 ``results/02_esp_and_dynamics``) に ``esp_diagnostics.csv`` /
``washout_sensitivity.csv`` / ``fig_esp_decay.png`` / ``fig_leak_timescale.png`` /
``fig_esp_map.png`` / ``fig_washout_sensitivity.png`` / ``meta.json`` の7点を書く。
``--threshold-sweep`` を付けたときは代わりに
``esp_threshold_sensitivity.csv`` (D-16 の閾値感度) だけを書く。
実測 wall time は ``meta.json`` の ``wall_time_s`` に記録する (性能受け入れ基準)。
進捗は ``print`` ではなく ``logging`` で出す (ruff T20)。

計算と書き出しの本体は
``rc_basics_lab.experiment.esp_pipeline.run_and_report_esp`` にあり、
ここは引数解析だけの薄い層である (``main.py --experiment 02`` と同じ経路)。
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
from rc_basics_lab.config import Esp02Config, load_config_as
from rc_basics_lab.experiment.esp_pipeline import (
    run_and_report_esp,
    run_and_report_threshold_sweep,
)

logger = logging.getLogger("rc_basics_lab.experiments.02_esp_and_dynamics")

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_OUT = default_out_for(DEFAULT_CONFIG)
"""``--out`` 未指定時の出力先。``main.py`` と同じ関数から導く。"""


@dataclass(frozen=True, slots=True)
class Args:
    """コマンドライン引数 (共通分は ``ExperimentArgs``)。"""

    common: ExperimentArgs
    threshold_sweep: bool


def parse_args(argv: Sequence[str] | None = None) -> Args:
    """引数を解析する。共通フラグは ``rc_basics_lab.cli`` が持つ。"""
    parser = build_parser(
        "実験2: ESP 判定・条件付き Lyapunov 指数・実効時定数を測る", DEFAULT_CONFIG
    )
    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        help=(
            "本体の7成果物は作らず、ESP 判定の閾値感度 "
            "(esp_threshold_sensitivity.csv) だけを再生成する (D-16 / design.md §9)"
        ),
    )
    common, namespace = parse_experiment_args(parser, argv)
    return Args(
        common=common,
        threshold_sweep=bool(namespace.threshold_sweep),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """実験を実行し、CSV2枚・図4枚・meta.json を書く。

    ``--threshold-sweep`` のときは閾値感度 CSV だけを書く。本体と分けるのは、
    2-C の格子をもう一度回すので実行時間が倍近くになるうえ、記事に載る成果物
    ではなく「既定値が結論を作っていない」ことの根拠だからである。
    """
    args = parse_args(argv)
    config = load_config_as(
        args.common.config,
        Esp02Config,
        preset=args.common.preset,
        overrides=args.common.overrides,
    )
    logger.info(
        "設定を読み込みました: %s (n_units=%d, n_steps=%d, n_replicates=%d)",
        args.common.config,
        config.reservoir.n_units,
        config.drive.n_steps,
        config.reservoir.n_replicates,
    )
    if args.threshold_sweep:
        run_and_report_threshold_sweep(config, args.common.out)
        return 0
    run_and_report_esp(config, args.common.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())

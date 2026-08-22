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

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rc_basics_lab.config import Capacity03Config, load_config_as
from rc_basics_lab.experiment.capacity_pipeline import (
    run_and_report_capacity,
    run_and_report_length_sweep,
)

logger = logging.getLogger("rc_basics_lab.experiments.03_capacity")

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_OUT = Path("results/03_capacity")


@dataclass(frozen=True, slots=True)
class Args:
    """コマンドライン引数。"""

    config: Path
    out: Path
    length_sweep: bool


def parse_args(argv: Sequence[str] | None = None) -> Args:
    """引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="実験3: 線形メモリ容量 (MC) と情報処理容量 (IPC) を測る"
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
    parser.add_argument(
        "--length-sweep",
        action="store_true",
        help=(
            "本体の成果物は作らず、系列長 T の掃引 (capacity_length.csv) だけを"
            "再生成する (make saturation-03)"
        ),
    )
    namespace = parser.parse_args(argv)
    return Args(
        config=Path(str(namespace.config)),
        out=Path(str(namespace.out)),
        length_sweep=bool(namespace.length_sweep),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """実験を実行し、CSV2枚と meta.json を書く。

    ``--length-sweep`` のときは系列長掃引の CSV だけを書く。本体と分けるのは、
    T=1e6 まで回すので単独で ``make figures-03`` の予算 (900 秒) を食い潰す
    うえ、記事に載る成果物ではなく「容量が足りないのか T が足りないのか」を
    切り分ける補助実験だからである。
    """
    args = parse_args(argv)
    config = load_config_as(args.config, Capacity03Config)
    logger.info(
        "設定を読み込みました: %s (3-A N=%d T=%d / 3-B N=%d T=%d / n_replicates=%d)",
        args.config,
        config.mc_sweep.n_units,
        config.mc_sweep.n_steps,
        config.ipc_sweep.n_units,
        config.ipc_sweep.n_steps,
        config.reservoir.n_replicates,
    )
    if args.length_sweep:
        run_and_report_length_sweep(config, args.out)
        return 0
    run_and_report_capacity(config, args.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())

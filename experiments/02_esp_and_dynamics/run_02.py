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

``--out`` (既定 ``results``) に ``esp_diagnostics.csv`` / ``fig_esp_decay.png`` /
``fig_leak_timescale.png`` / ``fig_esp_map.png`` / ``meta.json`` の5点を書く。
実測 wall time は ``meta.json`` の ``wall_time_s`` に記録する (性能受け入れ基準)。
進捗は ``print`` ではなく ``logging`` で出す (ruff T20)。

計算と書き出しの本体は
``rc_basics_lab.experiment.esp_pipeline.run_and_report_esp`` にあり、
ここは引数解析だけの薄い層である (``main.py --experiment 02`` と同じ経路)。
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rc_basics_lab.config import Esp02Config, load_config_as
from rc_basics_lab.experiment.esp_pipeline import run_and_report_esp

logger = logging.getLogger("rc_basics_lab.experiments.02_esp_and_dynamics")

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
        description="実験2: ESP 判定・条件付き Lyapunov 指数・実効時定数を測る"
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
    """実験を実行し、CSV1枚・図3枚・meta.json を書く。"""
    args = parse_args(argv)
    config = load_config_as(args.config, Esp02Config)
    logger.info(
        "設定を読み込みました: %s (n_units=%d, n_steps=%d, n_replicates=%d)",
        args.config,
        config.reservoir.n_units,
        config.drive.n_steps,
        config.reservoir.n_replicates,
    )
    run_and_report_esp(config, args.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())

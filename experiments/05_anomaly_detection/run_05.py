"""実験05 (センサー時系列の異常検知) の成果物を1コマンドで再生成する.

使い方::

    uv run python experiments/05_anomaly_detection/run_05.py \
        --config experiments/05_anomaly_detection/config.yaml

``--out`` (既定 ``results/05_anomaly_detection``) に CSV5枚 (``anomaly.csv`` /
``anomaly_threshold.csv`` / ``anomaly_timeline.csv`` / ``anomaly_protocol.csv``
/ ``anomaly_size.csv``)・図5枚・``meta.json`` を書く。成果物の一覧の単一の
真実は ``experiment/anomaly_pipeline.py`` の ``ANOMALY_ARTIFACTS``。実測
wall time は ``meta.json`` の ``wall_time_s`` と ``wall_time_breakdown`` に
区間ごとに記録する (性能受け入れ基準)。進捗は ``print`` ではなく ``logging``
で出す (ruff T20)。

**既定の設定 (`config.yaml`) は実データ源 MGAB を使う**ので、先に
``make data-05`` でキャッシュを作ること (D-58: データ本体はリポジトリに
含めない)。キャッシュが無い源は静かに落ちるのではなく ``ValueError`` になる。
``Anomaly05Config()`` の**コード上の既定は合成源**のままである (D-60: pytest
はネットワークに一切触れない)。

計算と書き出しの本体は
``rc_basics_lab.experiment.anomaly_pipeline.run_and_report_anomaly`` にあり、
ここは引数解析だけの薄い層である (``main.py --experiment 05`` と同じ経路)。
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rc_basics_lab.config import Anomaly05Config, load_config_as
from rc_basics_lab.experiment.anomaly_pipeline import run_and_report_anomaly

logger = logging.getLogger("rc_basics_lab.experiments.05_anomaly_detection")

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_OUT = Path("results/05_anomaly_detection")


@dataclass(frozen=True, slots=True)
class Args:
    """コマンドライン引数。"""

    config: Path
    out: Path


def parse_args(argv: Sequence[str] | None = None) -> Args:
    """引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="実験05: センサー時系列の異常検知 (5-A / 5-B / 5-C / 5-D)"
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
    config = load_config_as(args.config, Anomaly05Config)
    logger.info(
        "設定を読み込みました: %s (source=%s / 系列 %d 本 / max_length=%d / "
        "N=%d / n_replicates=%d)",
        args.config,
        config.dataset.source,
        len(config.dataset.series),
        config.dataset.max_length,
        config.reservoir.n_units,
        config.reservoir.n_replicates,
    )
    run_and_report_anomaly(config, args.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())

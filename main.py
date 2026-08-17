"""実験ランナーの薄い CLI エントリ.

使い方::

    uv run python main.py --experiment 01

``--experiment`` は ``experiments/`` 配下の実験番号。設定 YAML の場所を知っている
だけの層であり、計算・書き出しは
``rc_basics_lab.experiment.pipeline.run_and_report`` が行う
(``experiments/01_what_is_rc/run.py`` と完全に同じ経路)。
別の設定で走らせたいときは ``run.py --config <path>`` を使う。
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rc_basics_lab.config import load_config
from rc_basics_lab.experiment.pipeline import run_and_report

logger = logging.getLogger("rc_basics_lab.main")

ROOT = Path(__file__).resolve().parent

EXPERIMENTS: dict[str, Path] = {
    "01": ROOT / "experiments" / "01_what_is_rc" / "config.yaml",
}
"""実験番号 -> 設定 YAML。

**「ここに1行足すだけ」では 02〜05 は動かない**:
- YAML が ``ExperimentConfig`` に無いキーを持つ場合 D-09 により ``ConfigError``
  で落ちる (01 専用の設定クラスしか無いため)。
- ``run_and_report`` は ``run_experiment`` / ``collect_state_space`` /
  ``plot_comparison`` / ``plot_state_space`` という 01 専用の5成果物に
  ハードコードされている。02 の成果物 (``fig_esp_map.png`` /
  ``esp_diagnostics.csv`` 等) はこの経路を通らない。

02 着手時は、実験ごとの設定クラスとパイプライン関数の組をどう選択させるか
(``EXPERIMENTS`` の値を YAML パスから ``(設定ローダ, パイプライン関数, YAML パス)``
の組へ変える、など) をこの構造ごと見直すこと。
"""

DEFAULT_OUT = Path("results")


@dataclass(frozen=True, slots=True)
class Args:
    """コマンドライン引数。"""

    experiment: str
    out: Path


def parse_args(argv: Sequence[str] | None = None) -> Args:
    """引数を解析する。未知の実験番号は argparse が弾く。"""
    parser = argparse.ArgumentParser(description="rc-basics-lab の実験ランナー")
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default="01",
        help="実行する実験番号 (既定: 01)",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"出力ディレクトリ (既定: {DEFAULT_OUT})",
    )
    namespace = parser.parse_args(argv)
    return Args(experiment=str(namespace.experiment), out=Path(str(namespace.out)))


def main(argv: Sequence[str] | None = None) -> int:
    """指定した実験を実行し、成果物を ``--out`` に書き出す。"""
    args = parse_args(argv)
    config_path = EXPERIMENTS[args.experiment]
    config = load_config(config_path)
    logger.info(
        "実験 %s を実行します: %s (n_replicates=%d)",
        args.experiment,
        config_path,
        config.n_replicates,
    )
    run_and_report(config, args.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())

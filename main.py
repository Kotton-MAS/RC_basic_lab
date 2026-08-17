"""実験ランナーの薄い CLI エントリ.

使い方::

    uv run python main.py --experiment 01
    uv run python main.py --experiment 02

``--experiment`` は ``experiments/`` 配下の実験番号。この層が知っているのは
「どの設定 YAML を、どのローダで読み、どのパイプラインに渡すか」だけで、
計算・書き出しは ``rc_basics_lab.experiment.*_pipeline`` が行う
(``experiments/<番号>_*/run.py`` と完全に同じ経路)。
別の設定で走らせたいときは ``run.py --config <path>`` を使う。
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from rc_basics_lab.config import Esp02Config, load_config, load_config_as
from rc_basics_lab.experiment.esp_pipeline import run_and_report_esp
from rc_basics_lab.experiment.pipeline import run_and_report

logger = logging.getLogger("rc_basics_lab.main")

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """実験1本ぶんの「設定 YAML と、それを走らせる関数」。

    仕様 §4 T3 は ``(ローダ, パイプライン, YAML パス)`` の組と書いていたが、
    ローダの戻り値型は実験ごとに違う (``ExperimentConfig`` /
    ``Esp02Config``) ため、組のままでは ``Any`` を使わずに型を付けられない。
    「設定を読んでパイプラインへ渡す」までを1つの ``run`` に閉じることで、
    実験ごとの型が関数の内側に収まり、レジストリは単一の型で書ける。

    Attributes:
        config_path: 既定の設定 YAML。
        run: ``(設定 YAML, 出力ディレクトリ)`` を受けて成果物を書く関数。
    """

    config_path: Path
    run: Callable[[Path, Path], None]


def _run_01(config_path: Path, out_dir: Path) -> None:
    """実験01 (3ベースラインの比較 + 状態空間 PCA)。"""
    config = load_config(config_path)
    logger.info(
        "実験01 を実行します: %s (n_replicates=%d)", config_path, config.n_replicates
    )
    run_and_report(config, out_dir)


def _run_02(config_path: Path, out_dir: Path) -> None:
    """実験02 (ESP・スペクトル半径・リーク率)。"""
    config = load_config_as(config_path, Esp02Config)
    logger.info(
        "実験02 を実行します: %s (n_units=%d, n_steps=%d, n_replicates=%d)",
        config_path,
        config.reservoir.n_units,
        config.drive.n_steps,
        config.reservoir.n_replicates,
    )
    run_and_report_esp(config, out_dir)


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "01": ExperimentSpec(
        config_path=ROOT / "experiments" / "01_what_is_rc" / "config.yaml",
        run=_run_01,
    ),
    "02": ExperimentSpec(
        config_path=ROOT / "experiments" / "02_esp_and_dynamics" / "config.yaml",
        run=_run_02,
    ),
}
"""実験番号 -> ``ExperimentSpec``。

03〜05 を足すときは、その実験の設定クラスとパイプラインを呼ぶ ``_run_XX`` を
書いてここに1行足す。**設定クラスを 01 の ``ExperimentConfig`` に相乗りさせ
ない** (D-13)。相乗りさせると YAML の未知キー検査 (D-09) と配線テストの
被覆が同時に壊れる。
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
    spec = EXPERIMENTS[args.experiment]
    spec.run(spec.config_path, args.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())

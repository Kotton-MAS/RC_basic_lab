"""実験ランナーの薄い CLI エントリ.

使い方::

    uv run python main.py --experiment 01
    uv run python main.py --experiment 02
    uv run python main.py --experiment 03
    uv run python main.py --experiment 04

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

from rc_basics_lab.config import (
    Capacity03Config,
    Chaos04Config,
    Esp02Config,
    load_config,
    load_config_as,
)
from rc_basics_lab.experiment.capacity_pipeline import run_and_report_capacity
from rc_basics_lab.experiment.esp_pipeline import run_and_report_esp
from rc_basics_lab.experiment.freerun import run_and_report_onestep
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
        out_dir: ``--out`` 未指定時に使う既定の出力ディレクトリ。実験ごとに
            異ならなければならない (成果物が衝突すると黙って上書きされる)。
    """

    config_path: Path
    run: Callable[[Path, Path], None]
    out_dir: Path


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


def _run_03(config_path: Path, out_dir: Path) -> None:
    """実験03 (メモリ容量・情報処理容量)。"""
    config = load_config_as(config_path, Capacity03Config)
    logger.info(
        "実験03 を実行します: %s (3-A N=%d / 3-B N=%d / n_replicates=%d)",
        config_path,
        config.mc_sweep.n_units,
        config.ipc_sweep.n_units,
        config.reservoir.n_replicates,
    )
    run_and_report_capacity(config, out_dir)


def _run_04(config_path: Path, out_dir: Path) -> None:
    """実験04 (カオス時系列の自由走行予測。T4 時点では 4-A のみ)。"""
    config = load_config_as(config_path, Chaos04Config)
    logger.info(
        "実験04 を実行します: %s (Lorenz T=%d dt=%g / n_replicates=%d)",
        config_path,
        config.lorenz.length,
        config.lorenz.rk4_step * config.lorenz.sample_interval,
        config.base.n_replicates,
    )
    run_and_report_onestep(config, out_dir)


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "01": ExperimentSpec(
        config_path=ROOT / "experiments" / "01_what_is_rc" / "config.yaml",
        run=_run_01,
        out_dir=Path("results"),
    ),
    "02": ExperimentSpec(
        config_path=ROOT / "experiments" / "02_esp_and_dynamics" / "config.yaml",
        run=_run_02,
        out_dir=Path("results/02_esp_and_dynamics"),
    ),
    "03": ExperimentSpec(
        config_path=ROOT / "experiments" / "03_capacity" / "config.yaml",
        run=_run_03,
        out_dir=Path("results/03_capacity"),
    ),
    "04": ExperimentSpec(
        config_path=ROOT / "experiments" / "04_chaotic_freerun" / "config.yaml",
        run=_run_04,
        out_dir=Path("results/04_chaotic_freerun"),
    ),
}
"""実験番号 -> ``ExperimentSpec``。

03〜05 を足すときは、その実験の設定クラスとパイプラインを呼ぶ ``_run_XX`` を
書いてここに1行足す。**設定クラスを 01 の ``ExperimentConfig`` に相乗りさせ
ない** (D-13)。相乗りさせると YAML の未知キー検査 (D-09) と配線テストの
被覆が同時に壊れる。**``out_dir`` は実験ごとに異なる値にする** —— 揃えると
``--out`` 未指定の実行が別実験の成果物 (``meta.json`` など) を黙って上書きする
(``test_experiment_registry_has_unique_default_out_dirs`` が機械的に守る)。
"""


@dataclass(frozen=True, slots=True)
class Args:
    """コマンドライン引数。"""

    experiment: str
    out: Path | None


def parse_args(argv: Sequence[str] | None = None) -> Args:
    """引数を解析する。未知の実験番号は argparse が弾く。

    ``--out`` を省略した場合は ``None`` を返す。既定の出力先は実験ごとに
    異なる (``EXPERIMENTS[experiment].out_dir``) ため、この時点 (実験番号と
    独立に引数を解析する段階) では確定できない。実際の既定値解決は
    ``main()`` が ``args.experiment`` を見てから行う。
    """
    parser = argparse.ArgumentParser(description="rc-basics-lab の実験ランナー")
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default="01",
        help="実行する実験番号 (既定: 01)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="出力ディレクトリ (既定: 実験ごとに異なる。EXPERIMENTS[番号].out_dir)",
    )
    namespace = parser.parse_args(argv)
    out = None if namespace.out is None else Path(str(namespace.out))
    return Args(experiment=str(namespace.experiment), out=out)


def main(argv: Sequence[str] | None = None) -> int:
    """指定した実験を実行し、成果物を ``--out`` に書き出す。

    ``--out`` 未指定時は ``EXPERIMENTS[experiment].out_dir`` (実験ごとに異なる
    既定値) を使う。かつては全実験が同じ ``DEFAULT_OUT = Path("results")`` を
    共有しており、``--out`` を付けずに ``--experiment 02`` を実行すると 01 の
    ``results/meta.json`` を黙って上書きしていた。
    """
    args = parse_args(argv)
    spec = EXPERIMENTS[args.experiment]
    out_dir = spec.out_dir if args.out is None else args.out
    spec.run(spec.config_path, out_dir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())

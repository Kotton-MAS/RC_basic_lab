"""実験ランナーの薄い CLI エントリ.

使い方::

    uv run python main.py --experiment 01              # 手元 (scratch/ へ)
    uv run python main.py --experiment 03 --results    # 成果物 (results/ へ)
    uv run python main.py --experiment 03 --variant length
    uv run python main.py --experiment 01 --preset quick --set n_replicates=1

**この層が持つ知識はゼロである。** 何を走らせるかは
``rc_basics_lab.experiment.catalog`` の ``CATALOG`` が宣言し、ここは引数を
それに渡すだけ (D-125)。かつては ``main.py`` の ``EXPERIMENTS`` 辞書・
``experiments/0N_*/run_0N.py`` 5本・``Makefile`` の ``figures-0N`` 5ターゲットが
**同じ事実を3箇所に書いて**おり、食い違いをテストが事後に照合していた。

別の設定で走らせたいときは ``--config <path>``。
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rc_basics_lab.experiment.catalog import (
    BY_NUMBER,
    MAIN,
    RunRequest,
    spec_for,
)

logger = logging.getLogger("rc_basics_lab.main")


@dataclass(frozen=True, slots=True)
class Args:
    """コマンドライン引数。"""

    experiment: str
    variant: str
    out: Path | None
    to_results: bool
    config: Path | None
    preset: Path | None
    overrides: tuple[str, ...]


def parse_args(argv: Sequence[str] | None = None) -> Args:
    """引数を解析する。未知の実験番号は argparse が弾く。

    ``--out`` を省いた場合は ``None`` を返す。既定の書き出し先は実験ごとに
    違う (``ExperimentSpec.scratch_dir`` / ``results_dir``) ので、実験番号と
    独立に解析するこの時点では確定できない。
    """
    parser = argparse.ArgumentParser(description="rc-basics-lab の実験ランナー")
    parser.add_argument(
        "--experiment",
        choices=sorted(BY_NUMBER),
        default="01",
        help="実行する実験番号 (既定: 01)",
    )
    parser.add_argument(
        "--variant",
        default=MAIN,
        help=(
            "走らせ方 (既定: main)。実験ごとの候補は catalog.CATALOG の variants を参照"
        ),
    )
    parser.add_argument("--out", default=None, help="出力ディレクトリ")
    parser.add_argument(
        "--results",
        action="store_true",
        help="成果物のディレクトリ (results/...) へ書く。make figures-0N が使う",
    )
    parser.add_argument("--config", default=None, help="設定 YAML (既定: 実験ごと)")
    parser.add_argument("--preset", default=None, help="かぶせる YAML")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="設定を1つ上書きする (例: --set tasks.mackey_glass.reservoir.n_units=50)",
    )
    namespace = parser.parse_args(argv)
    return Args(
        experiment=str(namespace.experiment),
        variant=str(namespace.variant),
        out=None if namespace.out is None else Path(str(namespace.out)),
        to_results=bool(namespace.results),
        config=None if namespace.config is None else Path(str(namespace.config)),
        preset=None if namespace.preset is None else Path(str(namespace.preset)),
        overrides=tuple(str(item) for item in namespace.overrides),
    )


def resolve_out(args: Args) -> Path:
    """書き出し先を決める (**決め方はここ1か所**)。

    優先順は ``--out`` > ``--results`` > 手元 (``scratch/``)。既定を
    ``scratch/`` にしてあるのは、``--out`` を忘れた実行が ``results/`` の
    成果物を黙って上書きしないようにするためである。
    """
    spec = spec_for(args.experiment)
    if args.out is not None:
        return args.out
    return spec.results_dir if args.to_results else spec.scratch_dir


def main(argv: Sequence[str] | None = None) -> int:
    """指定した実験の指定した variant を実行する。

    Raises:
        ValueError: 実験番号または variant が無い場合 (候補を並べて落とす)。
    """
    args = parse_args(argv)
    spec = spec_for(args.experiment)
    run = spec.variant(args.variant)
    run(
        RunRequest(
            config=spec.config_path if args.config is None else args.config,
            out=resolve_out(args),
            preset=args.preset,
            overrides=args.overrides,
        )
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())

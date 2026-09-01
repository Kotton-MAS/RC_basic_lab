"""実験の run スクリプトが共有する引数解析.

5本の run スクリプトはどれも「設定を読む / 出力先を決める」だけの薄層で、
そこに ``--set`` と ``--preset`` を足すと**同じ argparse が5回写経される**。
写経すると、片方だけフラグが増えたり既定値がずれたりする (実測: 既に
``--config`` の help 文言が実験ごとに違っていた)。

**既定の出力先は ``scratch/`` である。** ``results/`` は指紋が固定された成果物で
(``tests/artifact_manifest.csv``)、手元で条件を振るたびにそこへ書くと
``make ci`` が赤くなる。試行と成果物が同じ場所を奪い合わないよう、
``results/`` へ書くのは ``make figures-0N`` が ``--out`` を明示する経路だけに
限る。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SCRATCH_DIR = Path("scratch")
"""既定の出力先。``.gitignore`` 済みで、指紋の検査の対象外。"""

PRESETS_DIRNAME = "presets"
"""``--preset <名前>`` が探すディレクトリ (実験ディレクトリの直下)。"""


def default_out_for(config: Path) -> Path:
    """設定ファイルの位置から既定の出力先を決める (**唯一の決め方**)。

    ``main.py`` の ``EXPERIMENTS[番号].out_dir`` と各 run スクリプトの
    ``DEFAULT_OUT`` は、以前どちらも手書きの ``results/...`` で、一致を
    ``tests/test_main.py`` が事後に照合していた。**両方をこの関数から導けば
    食い違いようがない** (照合のテストは残す。関数を経由しない経路が生えたら
    そこで気づく)。

    Args:
        config: 実験の ``config.yaml`` のパス。

    Returns:
        ``scratch/<実験ディレクトリ名>``。
    """
    return SCRATCH_DIR / config.parent.name


@dataclass(frozen=True, slots=True)
class ExperimentArgs:
    """run スクリプト共通の引数。

    Attributes:
        config: 本体の設定 YAML。
        out: 出力ディレクトリ。
        preset: かぶせる YAML (``--preset`` を解決した結果)。無ければ ``None``。
        overrides: ``--set`` の ``key.path=value`` の並び。
    """

    config: Path
    out: Path
    preset: Path | None
    overrides: tuple[str, ...]


def build_parser(
    description: str, default_config: Path, default_out: Path | None = None
) -> argparse.ArgumentParser:
    """共通フラグを備えたパーサを返す。

    実験固有のフラグ (``--variant threshold`` など) は呼び出し側が
    ``parser.add_argument`` で足す。

    Args:
        description: ``--help`` の1行説明。
        default_config: 既定の設定 YAML。
        default_out: 既定の出力先。``None`` なら ``scratch/<実験ディレクトリ名>``。
    """
    resolved_out = default_out or default_out_for(default_config)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=default_config,
        help=f"実験設定 YAML (既定: {default_config})",
    )
    parser.add_argument(
        "--out",
        default=resolved_out,
        help=(
            f"出力ディレクトリ (既定: {resolved_out})。"
            "results/ は成果物なので、書くのは make figures-0N だけにする"
        ),
    )
    parser.add_argument(
        "--preset",
        default=None,
        help=(
            f"{PRESETS_DIRNAME}/<名前>.yaml をかぶせる (例: quick)。"
            "差分だけを書いた YAML で、--set より先に適用する"
        ),
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help=(
            "設定を1つ上書きする (例: --set esn_mackey_glass.n_units=50)。"
            "値は YAML として解釈する。複数指定でき、左から順に適用する"
        ),
    )
    return parser


def resolve_preset(config: Path, name: str | None) -> Path | None:
    """``--preset quick`` を ``<config の隣>/presets/quick.yaml`` に解決する。

    パス区切りを含む指定はそのままパスとして扱う (リポジトリ外のプリセットも
    使えるようにするため)。

    Args:
        config: 本体の設定 YAML のパス。
        name: ``--preset`` の値。``None`` なら ``None`` を返す。

    Raises:
        FileNotFoundError: 解決したファイルが無い場合。**黙って無視しない** ——
            プリセット名のタイプミスが「効いていない実験」になるため。
    """
    if name is None:
        return None
    candidate = (
        Path(name)
        if "/" in name or name.endswith((".yaml", ".yml"))
        else config.parent / PRESETS_DIRNAME / f"{name}.yaml"
    )
    if not candidate.is_file():
        raise FileNotFoundError(f"プリセットが見つかりません: {candidate}")
    return candidate


def parse_experiment_args(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None = None
) -> tuple[ExperimentArgs, argparse.Namespace]:
    """共通引数を取り出す。

    Returns:
        ``(共通引数, 解析結果そのもの)``。2つ目は実験固有のフラグを読むため。
    """
    namespace = parser.parse_args(argv)
    config = Path(str(namespace.config))
    return (
        ExperimentArgs(
            config=config,
            out=Path(str(namespace.out)),
            preset=resolve_preset(config, namespace.preset),
            overrides=tuple(str(item) for item in namespace.overrides),
        ),
        namespace,
    )


__all__ = [
    "PRESETS_DIRNAME",
    "SCRATCH_DIR",
    "ExperimentArgs",
    "build_parser",
    "default_out_for",
    "parse_experiment_args",
    "resolve_preset",
]

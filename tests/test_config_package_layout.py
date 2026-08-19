"""``config`` package の形を固定する (D-49).

サイクル04 T1 で ``src/rc_basics_lab/config.py`` (非空 615 行) を
``src/rc_basics_lab/config/`` package へ**移動だけ**で割った。移動だけである
ことの証明は2本立てで、片方はここに、もう片方は成果物のバイト一致
(``make figures-01`` / ``figures-02`` / ``figures-03`` の再生成) にある。

ここが固定するのは3つ:

1. **公開シンボルの経路が変わらない** —— 分割前の ``rc_basics_lab.config`` から
   取った公開名のスナップショットと突き合わせる。``__all__`` は差分0を要求し、
   ``dir()`` にしか出ない名前 (実装の都合で入っていた import) は**消える側も
   増える側も全部書き出して固定する**。「``__all__`` さえ合っていればよい」に
   すると、package 化のついでに実装 import が公開名として増えても気づけない
2. **1モジュールあたりの行数の上限** —— 分割の目的そのもの
3. **package 内の依存が非循環で、許可した辺しか無い** —— 依存が双方向になると
   01 の設定を読むためだけに 02・03 の設定まで引き込まれる

いずれも ``config.py`` を書き戻したり、``__init__.py`` にサイクル間の import を
足したりすると落ちる。
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

import rc_basics_lab.config as config_pkg

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "src" / "rc_basics_lab" / "config"
PACKAGE_NAME = "rc_basics_lab.config"

MAX_NONEMPTY_LINES_PER_MODULE = 300
"""1モジュールあたりの非空行数の上限 (仕様 docs/plans/rc-basics-04.md §4 T1)。

分割前の ``config.py`` は非空 615 行で、サイクル03 が置いた着手条件
(非空 600 行) に到達していた。上限を 300 にしておくと、次に到達した時点で
「もう1段割る」判断を機械が要求する。
"""

PRE_SPLIT_ALL = (
    "DEFAULT_ALPHA_GRID",
    "DEFAULT_ESP_MAP_RHO_GRID",
    "DEFAULT_ESP_MAP_SIGMA_GRID",
    "TASK_LENGTH_FIELDS",
    "Capacity03Config",
    "CapacityDriveConfig",
    "CapacityReservoirConfig",
    "CapacitySeedConfig",
    "ConfigError",
    "ConservationConfig",
    "DelayParityConfig",
    "DriveConfig",
    "ESNConfig",
    "Esp02Config",
    "EspConfig",
    "EspDecayConfig",
    "EspMapConfig",
    "EspSeedConfig",
    "ExperimentConfig",
    "IpcConfig",
    "IpcSweepConfig",
    "LengthSweepConfig",
    "LyapunovConfig",
    "MackeyGlassConfig",
    "McSweepConfig",
    "MemoryCapacityConfig",
    "Narma10Config",
    "ReservoirSweepConfig",
    "RidgeConfig",
    "SplitConfig",
    "TimescaleConfig",
    "TimescaleSweepConfig",
    "WashoutSweepConfig",
    "esp_stream_seed",
    "load_config",
    "load_config_as",
)
"""分割**前**の ``config.py`` の ``__all__`` (実測スナップショット、36 名)。

取得コマンド (base-ref 8810d4e 時点):
``uv run python -c "import rc_basics_lab.config as c; print(sorted(c.__all__))"``

このタプルはコード側から生成せず**リテラルで持つ**。``config.__all__`` から
組み立てると「``__all__`` が ``__all__`` と一致する」という同語反復になり、
package 化で名前を落としても落とせなくなる。
"""

PRE_SPLIT_DIR_ONLY = (
    "Mapping",
    "Path",
    "Protocol",
    "SeedConfig",
    "SeedStream",
    "Sequence",
    "UnionType",
    "annotations",
    "cast",
    "dataclass",
    "dataclasses",
    "field",
    "get_args",
    "get_origin",
    "get_type_hints",
    "np",
    "yaml",
)
"""分割前の ``dir(config)`` に在って ``__all__`` に無かった公開名 (17 名)。

すべて ``config.py`` が実装のために書いた import 文の副作用であり、API として
公開したものではない (``__all__`` に入っていないので ``from ... import *`` にも
乗らない)。``test_no_module_imported_the_dir_only_names_from_config`` が
「リポジトリの誰もこの経路で import していない」ことを実測する。
"""

SURVIVING_DIR_ONLY = ("annotations",)
"""分割後も ``dir(config)`` に残る ``PRE_SPLIT_DIR_ONLY`` の名前。

``from __future__ import annotations`` は package の ``__init__.py`` でも
書くので残る (このリポジトリの全モジュールが書いている慣習)。
"""

EXPECTED_SUBMODULES = ("capacity03", "esp02", "experiment01")
"""分割で ``dir(config)`` に**増える**公開名 (実験サイクル単位のサブモジュール)。

``_common`` は ``_`` 始まりなので公開名には出ない。ここに書いていない名前が
増えたら、``__init__.py`` が実装 import を公開名に漏らしている。
"""

ALLOWED_INTERNAL_EDGES = frozenset(
    {
        ("__init__", "_common"),
        ("__init__", "experiment01"),
        ("__init__", "esp02"),
        ("__init__", "capacity03"),
        ("experiment01", "_common"),
        ("esp02", "experiment01"),
        ("capacity03", "experiment01"),
    }
)
"""``config`` package 内で許可する import の辺 (``from`` -> ``to``)。

- ``experiment01 -> _common``: 01 向けの別名 ``load_config`` が
  ``load_config_as`` に委譲する。``_common`` は package 内の**葉**で、
  どのサイクルモジュールも import しない
- ``esp02 -> experiment01``: ``WashoutSweepConfig.base: ExperimentConfig``
  (2-D が 01 の ``run_experiment`` を再利用するための内包、D-19)
- ``capacity03 -> experiment01``: ``Narma10Config.base: ExperimentConfig``
  (3-C が 01 の ``run_task`` を再利用するための内包、D-31)

``esp02`` と ``capacity03`` は**互いを import しない**。この2本はどちらも
01 を内包する同じ形の辺で、向きが逆になることは無い (01 が 02・03 の設定を
知ることは D-13 が禁じている)。
"""


def _module_paths() -> list[Path]:
    return sorted(PACKAGE_DIR.glob("*.py"))


def _nonempty_line_count(path: Path) -> int:
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def _internal_edges() -> set[tuple[str, str]]:
    """``config/*.py`` の import 文から package 内の辺を集める。

    実行時の属性ではなくソースの AST を見るのは、循環 import は「import 文が
    どう書かれているか」の問題であり、実行して通ってしまえば検出できない
    (Python は部分初期化されたモジュールを返すことがある) ため。
    """
    edges: set[tuple[str, str]] = set()
    for path in _module_paths():
        source = path.stem
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.ImportFrom):
                if node.level == 1 and node.module is not None:
                    targets.append(node.module.split(".")[0])
                elif node.level == 1:
                    targets.extend(alias.name for alias in node.names)
                elif node.module is not None and node.module.startswith(
                    f"{PACKAGE_NAME}."
                ):
                    targets.append(node.module[len(PACKAGE_NAME) + 1 :].split(".")[0])
            elif isinstance(node, ast.Import):
                targets.extend(
                    alias.name[len(PACKAGE_NAME) + 1 :].split(".")[0]
                    for alias in node.names
                    if alias.name.startswith(f"{PACKAGE_NAME}.")
                )
            edges.update((source, target) for target in targets if target != source)
    return edges


def test_public_symbols_are_importable_from_the_package_root() -> None:
    """分割前の公開シンボルが ``rc_basics_lab.config`` から同じ名前で引ける (D-49)。

    ``__all__`` は分割前のスナップショットと**差分0**であること、かつ列挙された
    名前が実際に属性として解決できることの両方を要求する。``__all__`` に名前を
    残したまま import 文を落とすと後者で落ちる。
    """
    assert tuple(config_pkg.__all__) == PRE_SPLIT_ALL, (
        "config.__all__ が分割前と一致しません "
        f"(不足={sorted(set(PRE_SPLIT_ALL) - set(config_pkg.__all__))}, "
        f"余剰={sorted(set(config_pkg.__all__) - set(PRE_SPLIT_ALL))})"
    )
    missing = [name for name in PRE_SPLIT_ALL if not hasattr(config_pkg, name)]
    assert not missing, f"__all__ に在るが解決できない名前: {missing}"


@pytest.mark.parametrize("name", PRE_SPLIT_ALL)
def test_each_public_symbol_resolves_through_the_import_statement(name: str) -> None:
    """``from rc_basics_lab.config import <名前>`` が1つずつ通る (D-49)。

    ``hasattr`` だけだと、``__init__.py`` が実行時に属性を差し込む形
    (``__getattr__`` 等) でも通ってしまう。import 文の形で1名ずつ確かめる。
    """
    module = importlib.import_module(PACKAGE_NAME)
    obj = getattr(module, name, None)
    assert obj is not None, f"from {PACKAGE_NAME} import {name} が解決できません"


def test_dir_only_names_changed_exactly_as_recorded() -> None:
    """``dir()`` の公開名の差分が、記録した集合と**厳密に**一致する (D-49)。

    分割前の公開名 (``__all__`` 36 + ``dir()`` のみ 17 = 53) に対して、
    消えるのは ``PRE_SPLIT_DIR_ONLY`` から ``SURVIVING_DIR_ONLY`` を除いたもの、
    増えるのは実験サイクル単位のサブモジュール3つだけ。両側を固定するので、
    ``__init__.py`` が numpy や yaml を公開名に漏らしても落ちる。
    """
    actual = {name for name in dir(config_pkg) if not name.startswith("_")}
    before = set(PRE_SPLIT_ALL) | set(PRE_SPLIT_DIR_ONLY)
    assert len(before) == 53, "スナップショットの名前数が実測 (53) と違います"

    expected_removed = set(PRE_SPLIT_DIR_ONLY) - set(SURVIVING_DIR_ONLY)
    assert before - actual == expected_removed
    assert actual - before == set(EXPECTED_SUBMODULES)


def test_no_module_imported_the_dir_only_names_from_config() -> None:
    """消える ``dir()`` 名を ``config`` 経由で import している箇所が無い (D-49)。

    ``PRE_SPLIT_DIR_ONLY`` は ``config.py`` の実装 import が公開名として
    見えていただけで API ではない —— という主張の実測。リポジトリ全体の
    ``from rc_basics_lab.config import ...`` を AST で拾い、``__all__`` の外の
    名前が1つも使われていないことを確かめる。使われていれば「移動だけ」では
    済まないので、そのときは ``__all__`` を増やす判断が要る。
    """
    dir_only = set(PRE_SPLIT_DIR_ONLY)
    offenders: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        if ".venv" in path.parts or PACKAGE_DIR in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module != PACKAGE_NAME:
                continue
            offenders.extend(
                f"{path.relative_to(ROOT)}: {alias.name}"
                for alias in node.names
                if alias.name in dir_only
            )
    assert not offenders, (
        f"__all__ の外の名前を config から import しています: {offenders}"
    )


@pytest.mark.parametrize("path", _module_paths(), ids=lambda p: p.name)
def test_each_config_module_stays_under_the_line_budget(path: Path) -> None:
    """``config/`` の各モジュールが非空 300 行以下 (仕様 §4 T1 の受け入れ基準)。

    分割前の ``config.py`` は非空 615 行だった。実測値そのものの照合は
    ``tests/test_design_doc.py`` (design.md §11.5 の表) が行う。
    """
    nonempty = _nonempty_line_count(path)
    assert nonempty <= MAX_NONEMPTY_LINES_PER_MODULE, (
        f"{path.name} が非空 {nonempty} 行で上限 "
        f"{MAX_NONEMPTY_LINES_PER_MODULE} 行を超えています"
    )


def test_config_package_has_exactly_the_expected_modules() -> None:
    """``config/`` の中身が ``_common`` + 実験サイクル3本 + ``__init__`` だけ。

    04 T4 が ``chaos04.py`` を足すときはここが赤くなる (置き場所は T1 で決めて
    あるが、**T1 では作らない**)。
    """
    submodules = {
        info.name
        for info in pkgutil.iter_modules([str(PACKAGE_DIR)])
    }
    assert submodules == {"_common", *EXPECTED_SUBMODULES}


def test_config_package_internal_dependencies_are_one_way() -> None:
    """package 内の import が許可した辺だけで、循環が無い (D-49)。

    ``_common`` は葉であること (package 内の誰も import しない先を持たない) と、
    ``esp02`` / ``capacity03`` が互いを import しないことを同時に固定する。
    """
    edges = _internal_edges()
    assert edges <= ALLOWED_INTERNAL_EDGES, (
        f"許可していない package 内 import: {sorted(edges - ALLOWED_INTERNAL_EDGES)}"
    )
    assert not [edge for edge in edges if edge[0] == "_common"], (
        "_common は config package 内の葉でなければなりません"
    )
    for pair in (("esp02", "capacity03"), ("capacity03", "esp02")):
        assert pair not in edges, f"実験サイクル間の import があります: {pair}"

    # 許可した辺の集合だけで循環が作れないことを、到達可能性で確かめる
    # (許可リストを書き換えて循環を足すと、ここで落ちる)。
    reachable: dict[str, set[str]] = {}
    for source, target in ALLOWED_INTERNAL_EDGES:
        reachable.setdefault(source, set()).add(target)
    for source in list(reachable):
        seen: set[str] = set()
        stack = list(reachable[source])
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(reachable.get(node, ()))
        assert source not in seen, f"config package 内に循環があります: {source}"

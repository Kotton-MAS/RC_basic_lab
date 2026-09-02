"""汎用基盤と連載の境界を機械で守る (D-126).

## なぜ要るのか

このリポジトリは**連載記事のための実験リポジトリ**として始まったが、
中身は2つに分かれている:

- **汎用基盤**: リザバー・課題・読み出し・診断・指標。どの実験からでも使える
- **連載**: 記事5本のパイプライン・図・設定・成果物

汎用側だけを別リポジトリへ持ち出せる形にしておくと、「基盤」と「連載」を
分ける日が来たときに ``git`` の操作だけで切れる。**そのためには汎用側が
連載側を import していないこと**が要る。

境界は散文では守れない。ここでデータとして持ち、import を実測する。

## この検査が測らないこと

「連載側が汎用側を使ってよいか」は測らない (使ってよい。それが依存の向き)。
測るのは**逆向きが 0 件であること**だけである。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "rc_basics_lab"
PACKAGE = "rc_basics_lab"

CORE_PACKAGES: frozenset[str] = frozenset(
    {"tasks", "reservoir", "readout", "diagnostics", "datasets"}
)
"""汎用側のサブパッケージ。**連載を1行も知らない。**"""

CORE_MODULES: frozenset[str] = frozenset(
    {
        "types",
        "seeds",
        "metrics",
        "metrics_detection",
        "metrics_significance",
        "cli",
        "overrides",
        "split",
    }
)
"""汎用側のトップレベルモジュール。"""

CORE_EXPERIMENT_MODULES: frozenset[str] = frozenset(
    {
        "rows_csv",
        "diagnostics_rows",
        "capacity_bounds",
    }
)
"""``experiment/`` のうち、実験の中身を知らない共通ヘルパ。

``experiment/`` は記事番号での縦割りなので大半が連載側だが、CSV の書き出し
(``rows_csv``) は「どの実験か」を知らない。ここだけは汎用側として持ち出せる。

分割 (``split``) は**トップレベルへ出した** —— ``config`` が ``SplitConfig``
を再エクスポートするので、``experiment/`` に置いたままだと
``config`` -> ``experiment/__init__`` -> ``config`` の循環になる。

**``report`` / ``summary`` は連載側である。** どちらも 01 の ``ResultRow`` と
``ExperimentConfig`` を知っているので、持ち出しても単体では動かない。
汎用のヘルパ (``write_rows_csv``) だけを ``rows_csv`` へ切り出した。
"""

SERIES_MODULES: frozenset[str] = frozenset({"meta", ""})
"""連載側のトップレベルモジュール。

``meta`` は 01 の ``ExperimentConfig`` を知っているので連載側である
(``meta.json`` の骨格そのものは汎用だが、集める中身が 01 の設定に縛られている)。
空文字はパッケージ直下の ``__init__`` (バージョンだけ)。
"""

SERIES_PACKAGES: frozenset[str] = frozenset({"plotting", "config"})
"""連載側のサブパッケージ。

``config/`` は実験ごとの設定 dataclass (D-13) なので連載側である
(ローダ本体 ``_common`` / ``_dump`` は汎用だが、パッケージとしては
``config.experiment01`` などと同居しているので丸ごと連載側に置く ——
**境界は粗くてよい。細かくするほど守られなくなる**)。
"""


def _module_name(path: Path) -> str:
    """ファイルパス -> ``rc_basics_lab`` からの相対モジュール名。"""
    rel = path.relative_to(PACKAGE_ROOT).with_suffix("")
    parts = [part for part in rel.parts if part != "__init__"]
    return ".".join(parts)


def _is_core(module: str) -> bool:
    head, _, rest = module.partition(".")
    if head in CORE_PACKAGES:
        return True
    if head == "experiment":
        return rest.split(".")[0] in CORE_EXPERIMENT_MODULES
    return module in CORE_MODULES


def _is_series(module: str) -> bool:
    head, _, rest = module.partition(".")
    if head in SERIES_PACKAGES:
        return True
    if head == "experiment":
        return rest == "" or rest.split(".")[0] not in CORE_EXPERIMENT_MODULES
    return module in SERIES_MODULES


def _imported_modules(path: Path) -> set[str]:
    """``rc_basics_lab.*`` の import 先をモジュール名で返す (関数内も含む)。"""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(f"{PACKAGE}."):
                found.add(node.module[len(PACKAGE) + 1 :])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(f"{PACKAGE}."):
                    found.add(alias.name[len(PACKAGE) + 1 :])
    return found


def _source_files() -> list[Path]:
    return sorted(
        path for path in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in path.parts
    )


def _core_files() -> list[Path]:
    return [path for path in _source_files() if _is_core(_module_name(path))]


def test_the_core_never_imports_the_series() -> None:
    """汎用側が連載側を import していない (**これが持ち出せる条件**)。"""
    offenders: dict[str, list[str]] = {}
    for path in _core_files():
        series = sorted(
            imported for imported in _imported_modules(path) if _is_series(imported)
        )
        if series:
            offenders[_module_name(path)] = series
    assert not offenders, (
        "汎用側が連載側を import しています (境界が壊れました):\n"
        + "\n".join(
            f"  {name} -> {', '.join(imports)}" for name, imports in offenders.items()
        )
        + "\n汎用側から使いたい型は汎用側へ移してください (D-126)。"
    )


def test_every_module_is_classified() -> None:
    """全モジュールが汎用か連載のどちらかに分類されている。

    分類漏れを許すと、**新しいモジュールが黙って検査の外に出る** ——
    境界が「守られている」ように見えて実際には測っていない状態になる。
    """
    unclassified = sorted(
        name
        for name in (_module_name(path) for path in _source_files())
        if name and not _is_core(name) and not _is_series(name)
    )
    assert not unclassified, (
        f"汎用とも連載とも分類されていないモジュール: {unclassified}\n"
        "tests/test_core_series_boundary.py の集合へ足してください。"
    )


def test_the_core_is_not_empty_and_covers_the_reusable_layers() -> None:
    """分類そのものが壊れていないこと (集合を空にして緑にする経路を塞ぐ)。"""
    core = {_module_name(path) for path in _core_files()}
    assert len(core) >= 40, f"汎用側が {len(core)} 本しかありません"
    for expected in (
        "reservoir.esn",
        "reservoir.topology",
        "tasks.mackey_glass",
        "readout.ridge",
        "diagnostics.topology",
    ):
        assert expected in core, f"{expected} が汎用側に分類されていません"


@pytest.mark.parametrize(
    "module", sorted(CORE_EXPERIMENT_MODULES | {"tasks", "reservoir"})
)
def test_the_declared_core_modules_exist(module: str) -> None:
    """宣言した汎用モジュールが実在する (改名・削除でリストが腐らない)。"""
    candidates = (
        PACKAGE_ROOT / f"{module}.py",
        PACKAGE_ROOT / module / "__init__.py",
        PACKAGE_ROOT / "experiment" / f"{module}.py",
    )
    assert any(path.exists() for path in candidates), f"{module} が見つかりません"

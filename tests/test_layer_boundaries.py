"""レイヤ境界の検査 —— 合成層 → 作図層の辺と、単独 import の完結性 (D-53).

``experiment`` (合成層) が ``plotting`` (作図層) を module-level で import して
いたため、``import rc_basics_lab.plotting`` を最初に行うと循環 import で
``ImportError`` になっていた (仕様 §2.4-4)。循環の実体は

    plotting/__init__ -> plotting.figures -> experiment.runner
        -> experiment/__init__ -> experiment.pipeline -> plotting.figures (部分初期化)

であり、辺の性質が2種類ある:

- ``plotting -> experiment`` は**静的**な依存 (行 dataclass の型・集計関数・
  記事メタの文言)。3b-2 で「記事メタを単一の真実にする」目的で引いた辺であり
  **残す**
- ``experiment -> plotting`` は**動的**な依存 (図を描く関数を呼ぶだけ)。型注釈にも
  定数にも現れない

D-53 は後者だけを関数本体の中へ落とす (ADR 0001 §5)。

**このファイルの検査は仕様 §4 T2 の受け入れ基準より強い**。
``plotting/__init__`` の import を遅延化する案 (却下案A) は
``test_plotting_can_be_imported_first`` を緑にするが、``ImportError`` は
属性アクセス1回ぶん先へ移動するだけで直っていない。
``test_every_package_resolves_all_of_its_public_names_when_imported_first`` が
その逃げ道を塞ぐ (ADR 0001 §5.4)。
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest
from test_public_api_reexport import PACKAGE_NAMES

import rc_basics_lab

SRC = Path(rc_basics_lab.__file__).parent

PLOTTING_ROOT = "rc_basics_lab.plotting"
EXPERIMENT_ROOT = "rc_basics_lab.experiment"


def _import_in_a_fresh_interpreter(script: str) -> subprocess.CompletedProcess[str]:
    """新しいインタプリタで ``script`` を走らせる.

    同一プロセスで測ると ``sys.modules`` に他のテストが import 済みの
    ``rc_basics_lab.experiment`` が残っており、循環が起きない状態で
    「単独 import できた」ことになってしまう。import 順の検査は
    **プロセスを分けないと空虚になる**。
    """
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )


def test_plotting_can_be_imported_first() -> None:
    """``import rc_basics_lab.plotting`` 単独が通る (仕様 §4 T2 の指定名).

    着手前は循環 import で ``ImportError: cannot import name 'plot_comparison'
    from partially initialized module 'rc_basics_lab.plotting.figures'`` になる
    ことを実測してから直した。
    """
    result = _import_in_a_fresh_interpreter(f"import {PLOTTING_ROOT}")
    assert result.returncode == 0, (
        f"{PLOTTING_ROOT} を最初に import できません (循環 import):\n{result.stderr}"
    )


@pytest.mark.parametrize("package_name", PACKAGE_NAMES)
def test_every_package_resolves_all_of_its_public_names_when_imported_first(
    package_name: str,
) -> None:
    """単独 import した直後に ``__all__`` の**全名前**が解決する (ADR 0001 §5.4).

    ``__init__`` の import を遅延化しただけだと、import 文そのものは通るが
    最初の属性アクセスで同じ ``ImportError`` が出る。「テストは緑だが直って
    いない」を作らないために、import の成功ではなく**公開名の解決**まで測る。
    """
    module = f"rc_basics_lab.{package_name}"
    script = (
        f"import {module} as m\n"
        "missing = [name for name in m.__all__ if not hasattr(m, name)]\n"
        "assert not missing, f'解決できない公開名: {missing}'\n"
    )
    result = _import_in_a_fresh_interpreter(script)
    assert result.returncode == 0, (
        f"{module} を単独 import した後に __all__ を解決できません:\n{result.stderr}"
    )


def _module_level_imported_roots(path: Path) -> set[str]:
    """``path`` が**関数の外**で import しているトップレベル名を集める。

    ``if TYPE_CHECKING:`` の中も module-level として数える —— 実行時には
    走らないが、そこに置くと循環の解消が型検査の設定に依存する形になり、
    D-53 の「関数本体の中で import する」という規律が読めなくなる。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()

    def visit(node: ast.AST, inside_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            is_function = isinstance(
                child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
            )
            if not inside_function:
                if isinstance(child, ast.Import):
                    roots.update(alias.name for alias in child.names)
                elif isinstance(child, ast.ImportFrom) and child.module is not None:
                    roots.add(child.module)
            visit(child, inside_function or is_function)

    visit(tree, False)
    return roots


def _package_modules(package_name: str) -> list[Path]:
    package = importlib.import_module(f"rc_basics_lab.{package_name}")
    return sorted(Path(package.__file__ or "").parent.glob("*.py"))


def _modules_importing(package_name: str, target_root: str) -> dict[str, set[str]]:
    """``package_name`` 配下で ``target_root`` を module-level import する一覧。"""
    found: dict[str, set[str]] = {}
    for path in _package_modules(package_name):
        hits = {
            root
            for root in _module_level_imported_roots(path)
            if root == target_root or root.startswith(f"{target_root}.")
        }
        if hits:
            found[path.name] = hits
    return found


def test_experiment_never_imports_plotting_at_module_level() -> None:
    """``experiment`` 配下に module-level の ``plotting`` import が0件 (D-53).

    AST 走査なので、04b-2 が ``freerun_pipeline.py`` を足しても
    **一覧への追記なしで自動的に被覆される**。
    """
    offenders = _modules_importing("experiment", PLOTTING_ROOT)
    assert not offenders, (
        "experiment 配下が plotting を module-level で import しています "
        f"(関数本体の中へ移してください。D-53): {offenders}"
    )


def test_plotting_may_import_experiment_at_module_level() -> None:
    """逆向きの辺は**許可されている** (D-53)。

    暗黙にしておくと、次の fixer が「一貫性のため」両方向を消しに来る。
    許可の理由は ``plotting`` が実験層から取っているのが行 dataclass の型
    だけでなく**集計関数** (``aggregate_nrmse``) と記事メタでもあるためで、
    これは記事メタを ``article/`` へ移す案 (ADR 0001 §5.4 案D) が循環の
    解決策にならない理由そのものである。その事実をここで固定する。
    """
    allowed = _modules_importing("plotting", EXPERIMENT_ROOT)
    assert allowed, (
        "plotting -> experiment の辺が1本も残っていません。D-53 が許可している "
        "辺なので、消すのであれば決定の側を先に改訂してください"
    )
    figures = SRC / "plotting" / "figures.py"
    tree = ast.parse(figures.read_text(encoding="utf-8"), filename=str(figures))
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith(EXPERIMENT_ROOT)
        for alias in node.names
    }
    assert "aggregate_nrmse" in imported_names, (
        "plotting/figures.py が実験層の**関数**を import しなくなりました。"
        "行 dataclass だけの依存になったのなら ADR 0001 §5.5 の見直し条件に "
        "該当します (decisions.yaml の D-53 を先に改訂してください)"
    )

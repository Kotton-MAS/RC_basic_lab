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
    """``path`` が**関数の外**で import しているトップレベル名 (D-53)。"""
    return imported_roots(path, include_function_bodies=False)


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


# --- 純関数層の境界 (D-59) ---------------------------------------------------

IO_IMPORT_ROOTS = frozenset(
    {
        "urllib",
        "requests",
        "socket",
        "http",
        "pathlib",
        "os",
        "io",
        "shutil",
        "zipfile",
        "tarfile",
        "tempfile",
        "subprocess",
        "sqlite3",
        "csv",
        "json",
    }
)
"""純関数層 (``tasks`` / ``metrics_detection``) が import してはいけない根。

仕様 §4 T2 の受け入れ基準2 が名指しするのは ``urllib`` / ``requests`` /
``socket`` / ``open`` / ``pathlib`` の5つだが、``os`` / ``io`` / ``zipfile`` /
``csv`` などは**同じことを別の名前でやる**ので一緒に塞ぐ。名指しの5つだけを
禁じると「``pathlib`` は使っていない (``os.path`` を使った)」が通ってしまう。
"""

IO_BUILTIN_CALLS = frozenset({"open", "eval", "exec", "compile", "__import__"})
"""同じく、呼んではいけない組み込み。"""

IO_NUMPY_FUNCTIONS = frozenset(
    {
        "load",
        "loadtxt",
        "genfromtxt",
        "fromfile",
        "save",
        "savez",
        "savetxt",
        "tofile",
        "memmap",
    }
)
"""numpy 経由のファイル I/O。

``import numpy as np`` は純関数層でも当然許すので、**根の名前だけを見る検査には
この穴が残る**。``np.loadtxt`` は ``open`` も ``pathlib`` も使わずにファイルを
読む。
"""


def _pure_layer_modules() -> list[Path]:
    """純関数層のソース一覧 (``tasks/*.py`` + ``metrics_detection.py``)。"""
    return [*_package_modules("tasks"), SRC / "metrics_detection.py"]


def _io_roots(path: Path) -> set[str]:
    """``path`` が (関数の中も含めて) import している I/O 系の根。"""
    roots = {
        root.split(".")[0]
        for root in imported_roots(path, include_function_bodies=True)
    }
    return roots & IO_IMPORT_ROOTS


@pytest.mark.parametrize("path", _pure_layer_modules(), ids=lambda p: p.name)
def test_tasks_and_metrics_never_perform_io(path: Path) -> None:
    """課題層と検知指標層が I/O を1行も持たない (D-59)。

    **module-level だけでなく関数本体の中も見る** —— 「関数の中で import すれば
    層の境界を越えていない」という抜け道を塞ぐためで、D-53 (``experiment ->
    plotting`` を関数内へ落とす) とはここが逆向きになる。あちらは循環 import
    の話で、こちらは**層が何をする層なのか**の話である。

    課題層に HTTP とキャッシュが入ると純関数層がステートフルな I/O 層に化け、
    memristor-rc-lab への移植性 (D-12 が守る性質) が失われる。取得・展開・
    ライセンス確認は ``datasets/`` だけが持つ。
    """
    offenders = _io_roots(path)
    assert not offenders, (
        f"{path.name} が I/O 系モジュールを import しています "
        f"(取得・読み書きは datasets/ に置いてください。D-59): {sorted(offenders)}"
    )

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called_names & IO_BUILTIN_CALLS, (
        f"{path.name} が I/O 系の組み込みを呼んでいます: "
        f"{sorted(called_names & IO_BUILTIN_CALLS)}"
    )

    attribute_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not attribute_calls & IO_NUMPY_FUNCTIONS, (
        f"{path.name} が numpy 経由でファイルを読み書きしています: "
        f"{sorted(attribute_calls & IO_NUMPY_FUNCTIONS)}"
    )


def test_the_io_guard_would_catch_an_import_inside_a_function(tmp_path: Path) -> None:
    """上の検査が**関数本体の中の import** を実際に捕まえる (変異注入)。

    ``imported_roots`` を module-level 版のままにすると
    ``test_tasks_and_metrics_never_perform_io`` は緑のまま素通りする。
    「関数の中に書けば通る」guard は guard ではないので、そのことをここで
    実測して固定する。
    """
    probe = tmp_path / "probe.py"
    probe.write_text(
        "def load(path):\n    import urllib.request\n    return urllib.request\n",
        encoding="utf-8",
    )
    module_level = {
        root.split(".")[0]
        for root in imported_roots(probe, include_function_bodies=False)
    }
    assert not module_level & IO_IMPORT_ROOTS
    assert _io_roots(probe) == {"urllib"}


def test_datasets_is_the_only_package_that_may_perform_io() -> None:
    """I/O の側は ``datasets`` に**在る**ことを固定する (D-59 の裏返し)。

    禁止側だけを測ると、``datasets`` が空になっても緑のままになる
    (「どこにも I/O が無い」は要件を満たしていない)。取得層が実際に
    ``urllib`` を持っていることをここで要求しておく。
    """
    roots: set[str] = set()
    for path in _package_modules("datasets"):
        roots |= {
            root.split(".")[0]
            for root in imported_roots(path, include_function_bodies=True)
        }
    assert "urllib" in roots, (
        "datasets/ が urllib を持っていません。取得層はここにしか置けません (D-59)"
    )

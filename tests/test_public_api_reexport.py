"""パッケージの ``__init__.py`` が全サブモジュールを再エクスポートしていることの検査.

conventions.md は「既存パッケージは全サブモジュールを再エクスポートしている」と
慣習として記録しているが、これまで機械的に固定するテストが無かった (F-2-012)。
02 で ``diagnostics/echo_state.py`` 等の新規モジュールを追加する際、
``__init__.py`` への配線 (import 文 / ``__all__`` 追加) を忘れると、公開 API から
そのモジュールが静かに欠落する。D-12 の推移閉包 guard
(``tests/test_diagnostics_base.py::test_diagnostics_package_does_not_transitively_import_reservoir_or_config``)
は ``pkgutil.iter_modules`` でサブモジュールを直接列挙するため配線漏れの影響を
受けないが、公開 API の一貫性はこの guard の対象外であり、本テストが別途守る。

**D-52 (04a T2)**: ``test_package_init_reexports_all_public_submodules`` は
``hasattr`` しか見ないため、``__init__`` が同名の**関数**でサブモジュールを
隠していても緑で通していた —— この慣習テスト自身が隠蔽の共犯だった。
``test_package_attributes_are_modules_not_shadowed`` が「属性が
``ModuleType`` であること」まで測ってその穴を塞ぐ。
"""

from __future__ import annotations

import ast
import importlib
import json
import pkgutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

import rc_basics_lab

PACKAGE_NAMES = (
    "config",
    "datasets",
    "diagnostics",
    "reservoir",
    "readout",
    "tasks",
    "experiment",
    "plotting",
)
"""慣習の対象パッケージ (conventions.md §関連ファイルマップ)。

``config`` は 04 T1 で ``config.py`` から package 化した7つ目 (D-49)。
``datasets`` は 05 T2 が足した8つ目で、**ネットワークとファイル I/O を持つ
唯一のパッケージ**である (D-59)。
公開シンボルは ``config/__init__.py`` が明示的に再エクスポートするので、
サブモジュール (``experiment01`` / ``esp02`` / ``capacity03``) は
``__init__`` の import 文の副作用として親の属性になる。

``_`` 始まりの private モジュールは意図的に公開したくない内部実装として
許容し、対象から除外する (公開したくないモジュールは名前を ``_`` で始めれば
このテストの対象から外せる、という抜け道を明示的な設計として残す)。
"""


NOT_ON_THE_FACADE: dict[str, tuple[str, ...]] = {"datasets": ("cli",)}
"""``__init__`` に**載せない**と決めた公開サブモジュール (D-72)。

慣習は「全サブモジュールを再エクスポートする」だが、``datasets.cli`` は例外
である —— ``cli.py`` は ``from rc_basics_lab.datasets import mgab, ucr`` で
パッケージへ戻るので、``__init__`` が ``cli`` を import すると
``__init__ -> cli -> __init__`` の辺ができる。現状は submodule import 機構の
おかげで動いているだけで、``cli`` が ``__init__`` の**再エクスポートを1つでも
使い始めた瞬間に ``ImportError``** になる (T5 の CLI 配線はまさに再エクスポート
を使いたくなる作業で、そこで初めて壊れると原因が CLI 変更に見えて循環に
見えない)。CLI は公開 API ではなく ``make data-05`` の入口なので、facade から
外す方を選んだ。

**この辞書に名前を足すのは、循環を構造で断つときだけ**である。単に
``__init__`` への追記を忘れたモジュールを逃がす穴として使うと、この慣習テスト
自体が意味を失う (D-52 が塞いだのと同型の抜け道)。
"""


def _public_submodule_names(package: ModuleType) -> set[str]:
    exempt = NOT_ON_THE_FACADE.get(package.__name__.rsplit(".", 1)[-1], ())
    return {
        info.name
        for info in pkgutil.iter_modules(package.__path__)
        if not info.name.startswith("_") and info.name not in exempt
    }


def test_package_names_matches_automatic_enumeration() -> None:
    """``PACKAGE_NAMES`` が実際のトップレベルパッケージ集合と一致する (F-02-1-020)。

    以前は手書きの固定タプルで、実際の ``rc_basics_lab`` 配下のパッケージ集合と
    機械的に突き合わせる完全性チェックが無かった。7つ目のトップレベル
    パッケージが増えても ``PACKAGE_NAMES`` への追記を忘れると黙って検査対象
    から漏れる。``pkgutil.iter_modules`` の ``ispkg`` でサブパッケージ
    (``seeds.py`` 等の単一モジュールは除く) だけを列挙して突き合わせる。
    """
    actual = {
        info.name
        for info in pkgutil.iter_modules(rc_basics_lab.__path__)
        if info.ispkg and not info.name.startswith("_")
    }
    assert set(PACKAGE_NAMES) == actual, (
        "PACKAGE_NAMES が実際のトップレベルパッケージ集合と一致しません "
        f"(不足={sorted(actual - set(PACKAGE_NAMES))}, "
        f"余剰={sorted(set(PACKAGE_NAMES) - actual)})"
    )


@pytest.mark.parametrize("package_name", PACKAGE_NAMES)
def test_package_init_reexports_all_public_submodules(package_name: str) -> None:
    """非 private サブモジュールが ``__init__.py`` から再エクスポートされる.

    ``from rc_basics_lab.<pkg>.<submodule> import ...`` が実行されると、
    Python の import 機構が ``<submodule>`` を親パッケージオブジェクトの
    属性として自動的にセットする。これを ``hasattr`` で確認することで、
    ``__init__.py`` の文面を解析せずに「再エクスポートされているか」を
    実行時に検証できる。登録漏れがあれば
    ``from rc_basics_lab.<pkg> import <submodule 内の関数>`` が解決できず、
    公開 API から静かに欠落する。
    """
    package = importlib.import_module(f"rc_basics_lab.{package_name}")
    expected = _public_submodule_names(package)
    assert expected, f"{package_name} 配下に非private モジュールが見つかりません"

    missing = sorted(name for name in expected if not hasattr(package, name))
    assert not missing, (
        f"rc_basics_lab.{package_name}.__init__ が次のサブモジュールを "
        f"re-export していません: {missing}"
    )


DIAGNOSTICS_ALL = (
    "DEFAULT_ESP",
    "DEFAULT_IPC",
    "DEFAULT_LYAPUNOV",
    "DEFAULT_MAX_LYAPUNOV",
    "DEFAULT_MEMORY_CAPACITY",
    "DEFAULT_TIMESCALE",
    "Diagnostic",
    "DiagnosticContext",
    "DiagnosticResult",
    "EspConfig",
    "IpcConfig",
    "LyapunovConfig",
    "MaxLyapunovConfig",
    "MemoryCapacityConfig",
    "StatePropagator",
    "TimescaleConfig",
    "autocorrelation_time",
    "conditional_lyapunov",
    "esp_convergence",
    "max_lyapunov",
    "state_mean_norm",
    "state_pca",
    "validate_diagnostic_input",
)
"""``rc_basics_lab.diagnostics.__all__`` のスナップショット (D-52)。

04b-1 で ``lyapunov`` モジュールの3名 (``DEFAULT_MAX_LYAPUNOV`` /
``MaxLyapunovConfig`` / ``max_lyapunov``) を足した。**関数名は
``max_lyapunov``** で、モジュール名 ``lyapunov`` と衝突しないので
``__all__`` に載せてよい (D-52)。

04a T2 で**関数** ``ipc`` / ``memory_capacity`` の2名を外した。増減の
**両側**を固定するのは、「2名が消えたこと」だけを見ると他の名前を巻き添えで
落としても緑になり、「他が動いていないこと」だけを見ると2名が戻っても
気づけないため。
"""

DIAGNOSTIC_FUNCTIONS_BY_MODULE = {
    "dummy": ("state_mean_norm",),
    "esp": ("conditional_lyapunov", "esp_convergence"),
    "ipc": ("ipc",),
    "lyapunov": ("max_lyapunov",),
    "memory_capacity": ("memory_capacity",),
    "state_space": ("state_pca",),
    "timescale": ("autocorrelation_time",),
}
"""診断モジュール -> そのモジュールが定義する診断関数 (D-52 の「旧経路」)。

D-52 は ``from rc_basics_lab.diagnostics import ipc`` が**関数**を返すことを
やめただけで、**フルパスは正規の入手経路として残す**。本番3ファイルが現に
使っている経路なので、ここで固定する。
"""


@pytest.mark.parametrize("package_name", PACKAGE_NAMES)
def test_package_attributes_are_modules_not_shadowed(package_name: str) -> None:
    """公開サブモジュール名と同名の公開シンボルを再エクスポートしない (D-52)。

    ``from rc_basics_lab.diagnostics.ipc import ipc`` を ``__init__`` に書くと、
    パッケージ属性 ``diagnostics.ipc`` (=モジュール) が**関数**で上書きされる。
    ``import a.b.c as m`` は ``sys.modules`` より先に親の属性を見るため ``m`` は
    関数になり、``monkeypatch.setattr(m, "...")`` が関数オブジェクトへの属性
    設定として**成功**したまま何も差し替わらない (3a のレビューで実際に踏み、
    変異試験が偽の緑になった)。

    全7パッケージ x 全公開サブモジュールを回すので、04b-1 が
    ``diagnostics/lyapunov.py`` を足したときに公開関数名を ``lyapunov`` に
    すると**一覧への追記なしで自動的に赤くなる**。
    """
    package = importlib.import_module(f"rc_basics_lab.{package_name}")
    shadowed = {
        name: type(getattr(package, name)).__name__
        for name in _public_submodule_names(package)
        if not isinstance(getattr(package, name, None), ModuleType)
    }
    assert not shadowed, (
        f"rc_basics_lab.{package_name} のサブモジュール名が同名の公開シンボルで "
        f"隠されています (D-52。__init__ の再エクスポートを外してください): "
        f"{shadowed}"
    )


def test_diagnostics_all_matches_the_recorded_snapshot() -> None:
    """``diagnostics.__all__`` を増減の**両側**で固定する (D-52)。"""
    package = importlib.import_module("rc_basics_lab.diagnostics")
    actual = tuple(package.__all__)
    assert actual == DIAGNOSTICS_ALL, (
        "diagnostics.__all__ が記録と一致しません "
        f"(増={sorted(set(actual) - set(DIAGNOSTICS_ALL))}, "
        f"減={sorted(set(DIAGNOSTICS_ALL) - set(actual))})"
    )
    assert "ipc" not in actual and "memory_capacity" not in actual, (
        "モジュール名と同名の関数が __all__ へ戻っています (D-52)"
    )


@pytest.mark.parametrize(
    ("module_name", "function_names"),
    sorted(DIAGNOSTIC_FUNCTIONS_BY_MODULE.items()),
)
def test_diagnostic_functions_are_importable_from_their_modules(
    module_name: str, function_names: tuple[str, ...]
) -> None:
    """関数の正規の入手経路 (フルパス) が全診断で通る (D-52 の「残す」側)。"""
    module = importlib.import_module(f"rc_basics_lab.diagnostics.{module_name}")
    for name in function_names:
        assert callable(getattr(module, name)), (
            f"rc_basics_lab.diagnostics.{module_name}.{name} が呼べません"
        )


DATASETS_INIT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "rc_basics_lab"
    / "datasets"
    / "__init__.py"
)


def test_the_datasets_facade_does_not_import_the_cli() -> None:
    """``datasets/__init__`` が ``cli`` を import しない (D-72)。

    ``__init__ -> cli -> __init__`` の辺を**構造で**断つ。3方向から測る:

    1. ``__init__.py`` の import 文 (AST) に ``cli`` が現れない
    2. ``__all__`` に ``"cli"`` が無い
    3. まっさらな interpreter で ``import rc_basics_lab.datasets`` しても
       ``sys.modules`` に ``rc_basics_lab.datasets.cli`` が入らない

    3 を別プロセスで測るのは、pytest の1プロセス内では他のテストが
    ``datasets.cli`` を import した副作用で ``sys.modules`` にも親の属性にも
    ``cli`` が現れ、1 と 2 を消しても緑のまま通ってしまうため
    (``tests/test_diagnostics_base.py`` の推移的 import 検査と同じ事情)。
    """
    source = DATASETS_INIT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(DATASETS_INIT))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
            if node.module is not None:
                imported.add(node.module.rsplit(".", 1)[-1])
        elif isinstance(node, ast.Import):
            imported.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
    assert "cli" not in imported, (
        "datasets/__init__.py が cli を import しています (D-72)。"
        "cli.py はパッケージへ戻る辺を持つので、循環になります"
    )

    package = importlib.import_module("rc_basics_lab.datasets")
    assert "cli" not in package.__all__, "datasets.__all__ に cli が戻っています (D-72)"

    probe = textwrap.dedent("""
        import json
        import sys

        import rc_basics_lab.datasets  # noqa: F401

        print("LOADED=" + json.dumps(sorted(
            name for name in sys.modules if name.startswith("rc_basics_lab.datasets")
        )))
        """)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    line = next(
        row for row in completed.stdout.splitlines() if row.startswith("LOADED=")
    )
    loaded = json.loads(line[len("LOADED=") :])
    assert "rc_basics_lab.datasets.cli" not in loaded, (
        f"import rc_basics_lab.datasets が cli を引き込んでいます (D-72): {loaded}"
    )
    assert "rc_basics_lab.datasets.mgab" in loaded, (
        "facade が mgab を再エクスポートしていません (探索条件が壊れています)"
    )

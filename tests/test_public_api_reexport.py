"""パッケージの ``__init__.py`` が全サブモジュールを再エクスポートしていることの検査.

conventions.md は「既存6パッケージは全サブモジュールを再エクスポートしている」と
慣習として記録しているが、これまで機械的に固定するテストが無かった (F-2-012)。
02 で ``diagnostics/echo_state.py`` 等の新規モジュールを追加する際、
``__init__.py`` への配線 (import 文 / ``__all__`` 追加) を忘れると、公開 API から
そのモジュールが静かに欠落する。D-12 の推移閉包 guard
(``tests/test_diagnostics_base.py::test_diagnostics_package_does_not_transitively_import_reservoir``)
は ``pkgutil.iter_modules`` でサブモジュールを直接列挙するため配線漏れの影響を
受けないが、公開 API の一貫性はこの guard の対象外であり、本テストが別途守る。
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

import pytest

import rc_basics_lab

PACKAGE_NAMES = (
    "diagnostics",
    "reservoir",
    "readout",
    "tasks",
    "experiment",
    "plotting",
)
"""慣習の対象6パッケージ (conventions.md §関連ファイルマップ)。

``_`` 始まりの private モジュールは意図的に公開したくない内部実装として
許容し、対象から除外する (公開したくないモジュールは名前を ``_`` で始めれば
このテストの対象から外せる、という抜け道を明示的な設計として残す)。
"""


def _public_submodule_names(package: ModuleType) -> set[str]:
    return {
        info.name
        for info in pkgutil.iter_modules(package.__path__)
        if not info.name.startswith("_")
    }


def test_package_names_matches_automatic_enumeration() -> None:
    """``PACKAGE_NAMES`` が実際のトップレベルパッケージ集合と一致する (F-02-1-020)。

    以前は手書きの固定タプルで、実際の ``rc_basics_lab`` 配下のパッケージ集合と
    機械的に突き合わせる完全性チェックが無かった。7つ目のトップレベル
    パッケージが増えても ``PACKAGE_NAMES`` への追記を忘れると黙って検査対象
    から漏れる。``pkgutil.iter_modules`` の ``ispkg`` でサブパッケージ
    (``config.py`` 等の単一モジュールは除く) だけを列挙して突き合わせる。
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

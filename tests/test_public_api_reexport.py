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
from pathlib import Path

import pytest

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


def _public_submodule_names(package_dir: Path) -> set[str]:
    return {
        info.name
        for info in pkgutil.iter_modules([str(package_dir)])
        if not info.name.startswith("_")
    }


@pytest.mark.parametrize("package_name", PACKAGE_NAMES)
def test_package_init_reexports_all_public_submodules(package_name: str) -> None:
    """各パッケージの非 private サブモジュールが ``__init__.py`` から再エクスポートされている。

    ``from rc_basics_lab.<pkg>.<submodule> import ...`` が実行されると、
    Python の import 機構が ``<submodule>`` を親パッケージオブジェクトの
    属性として自動的にセットする。これを ``hasattr`` で確認することで、
    ``__init__.py`` の文面を解析せずに「再エクスポートされているか」を
    実行時に検証できる。登録漏れがあれば
    ``from rc_basics_lab.<pkg> import <submodule 内の関数>`` が解決できず、
    公開 API から静かに欠落する。
    """
    package = importlib.import_module(f"rc_basics_lab.{package_name}")
    package_dir = Path(package.__file__).parent
    expected = _public_submodule_names(package_dir)
    assert expected, f"{package_name} 配下に非private モジュールが見つかりません"

    missing = sorted(name for name in expected if not hasattr(package, name))
    assert not missing, (
        f"rc_basics_lab.{package_name}.__init__ が次のサブモジュールを "
        f"re-export していません: {missing}"
    )

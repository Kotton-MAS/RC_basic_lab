"""``docs/guide/モジュール地図.md`` が現在のコードと一致することの検査.

**地図が嘘をつくと、地図が無いより悪い** —— 読者は嘘を確かめる手間を余分に
払う。手書きの表は、次にモジュールを足した人が写経を忘れた時点で嘘になる。
生成し直して差分が無いことを見る (`make ci` が赤くなるので、足した時点で気づく)。

直し方: ``uv run python scripts/module_map.py``
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "module_map.py"
OUTPUT = ROOT / "docs" / "guide" / "モジュール地図.md"


def _module_map() -> object:
    """``scripts/module_map.py`` を読み込む (``scripts`` は package ではない)。"""
    spec = importlib.util.spec_from_file_location("module_map", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["module_map"] = module
    spec.loader.exec_module(module)
    return module


def test_the_map_is_up_to_date() -> None:
    """生成し直しても内容が変わらない。

    モジュールを足したり、冒頭 docstring の1行目を直したりしたら落ちる。
    **要約の正本はコード側**なので、直すのは地図ではなくコードでよい。
    """
    module = _module_map()
    rendered: str = module.render()  # type: ignore[attr-defined]
    assert OUTPUT.is_file(), f"{OUTPUT} がありません"
    assert OUTPUT.read_text(encoding="utf-8") == rendered, (
        "モジュール地図が古くなっています。\n"
        "uv run python scripts/module_map.py を実行してください。"
    )


def test_every_module_appears_in_the_map() -> None:
    """``src/rc_basics_lab/`` の全モジュールが地図に出る。

    層の登録漏れ (``LAYERS`` に無いディレクトリ) があると表から落ちるので、
    件数で突き合わせる。
    """
    module = _module_map()
    src = ROOT / "src" / "rc_basics_lab"
    expected = {
        path.relative_to(src).as_posix()
        for path in src.rglob("*.py")
        if path.name != "__init__.py" and "__pycache__" not in path.parts
    }
    text = OUTPUT.read_text(encoding="utf-8")
    missing = sorted(name for name in expected if f"`{name}`" not in text)
    assert not missing, f"地図に出ていないモジュール: {missing}"
    del module


def test_the_layer_diagram_marks_the_deferred_import() -> None:
    """``experiment -> plotting`` が**破線**で描かれる (D-53)。

    module-level import を禁じている辺なので、実線で描くと規約違反があるように
    見える。ここが実線になったら、実際に module-level import が生えたか、
    区別のロジックが壊れたかのどちらかである。
    """
    module = _module_map()
    edges = module.package_edges()  # type: ignore[attr-defined]
    deferred = [
        (source, target)
        for source, target, at_module_level in edges
        if not at_module_level
    ]
    assert ("experiment", "plotting") in deferred, (
        "experiment -> plotting が module-level import になっています (D-53)"
    )


@pytest.mark.parametrize(
    ("source", "target"),
    [("plotting", "config"), ("tasks", "experiment"), ("readout", "experiment")],
)
def test_the_diagram_has_no_backward_edges(source: str, target: str) -> None:
    """逆流の辺が地図に出ない (出たら依存の向きが壊れている)。"""
    module = _module_map()
    edges = {(a, b) for a, b, _ in module.package_edges()}  # type: ignore[attr-defined]
    assert (source, target) not in edges, f"逆流しています: {source} -> {target}"

"""汎用側だけを切り出して**実際に動く**ことを測る (D-127).

## なぜ AST の検査だけでは足りないのか

``tests/test_core_series_boundary.py`` は import 文を読んで逆流が無いことを
見る。これは静的な検査なので、**関数の中で import する**経路や、コード以外の
同梱物 (``datasets/manifests/*.csv``) の取りこぼしは見えない。

ここは実際に汎用側だけを一時ディレクトリへ複製し、連載側が1バイトも無い状態で
import して回す。**切り出しの段取り (docs/guide/切り出す.md) が嘘になっていない
ことの担保**である。

## 速さ

複製は 50 本ほどのファイルで、走らせるのは N=30 / T=400 の1本だけ。実測で
1秒未満なので ``make ci`` に入れてよい。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from test_core_series_boundary import PACKAGE_ROOT, _core_files, _module_name

CORE_DATA_PACKAGES: frozenset[str] = frozenset(
    {"tasks", "reservoir", "readout", "diagnostics", "datasets"}
)
"""``.py`` 以外の同梱物も運ぶサブパッケージ (``datasets/manifests/*.csv`` など)。"""

SMOKE = """
import numpy as np
from rc_basics_lab.diagnostics.state_space import state_pca
from rc_basics_lab.diagnostics.topology import degree_distribution
from rc_basics_lab.reservoir.esn import ESNConfig
from rc_basics_lab.reservoir.registry import build_reservoir, require_graph
from rc_basics_lab.tasks.mackey_glass import MackeyGlassConfig
from rc_basics_lab.tasks.registry import build_task

rng = np.random.default_rng(0)
task = build_task(MackeyGlassConfig(length=400), rng)
reservoir = build_reservoir(ESNConfig(n_units=30), rng, n_inputs=task.u.shape[1])
states = reservoir.run(task.u)
degrees = degree_distribution(require_graph(reservoir, used_by="切り出し検査"))
print(states.shape[0], states.shape[1], degrees.scalars["n_units"])
"""
"""切り出した汎用側だけで「課題を作る -> 流す -> 測る」が通ることの最小確認。"""


def _copy_core(destination: Path) -> list[str]:
    """汎用側のファイルを ``destination`` へ複製し、モジュール名を返す。"""
    package = destination / "rc_basics_lab"
    for path in _core_files():
        target = package / path.relative_to(PACKAGE_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, target)
    for path in PACKAGE_ROOT.rglob("*"):
        if not path.is_file() or path.suffix == ".py" or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(PACKAGE_ROOT)
        if relative.parts[0] in CORE_DATA_PACKAGES:
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, target)
    for directory in {path.parent for path in package.rglob("*.py")} | {package}:
        init = directory / "__init__.py"
        if not init.exists():
            init.write_text('"""汎用基盤。"""\n', encoding="utf-8")
    (package / "__init__.py").write_text('__version__ = "0.1.0"\n', encoding="utf-8")
    return sorted(name for name in (_module_name(p) for p in _core_files()) if name)


def _run(code: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_core_imports_without_the_series_on_disk() -> None:
    """連載側が**存在しない**状態で汎用側を全部 import できる。

    関数内 import で連載側に触っていると、ここで初めて落ちる。
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        modules = _copy_core(root)
        assert len(modules) >= 40, f"汎用側が {len(modules)} 本しかありません"
        result = _run(
            "\n".join(f"import rc_basics_lab.{name}" for name in modules), root
        )
        assert result.returncode == 0, (
            "汎用側だけでは import できません (切り出せない状態です):\n"
            + result.stderr.strip()
        )


def test_the_extracted_core_can_run_an_experiment_end_to_end() -> None:
    """切り出した汎用側だけで「課題を作る -> 流す -> 測る」が通る。

    import が通るだけでは「動く」とは言えない (同梱データの取りこぼしは
    実行時にしか出ない)。
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _copy_core(root)
        result = _run(SMOKE, root)
        assert result.returncode == 0, (
            "切り出した汎用側で実験を回せません:\n" + result.stderr.strip()
        )
        assert result.stdout.split() == ["400", "30", "30.0"], result.stdout

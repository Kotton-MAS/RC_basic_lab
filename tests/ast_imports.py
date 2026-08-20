"""ソースの import 文からモジュール名/シンボル名を集める共通機構.

「このモジュールが何を import しているか」を AST で調べる判定器が、layer
境界 (D-53/D-59)・レイヤ独立性 (D-23/D-50)・循環 import の禁止 (D-72) など
複数の決定の guard として個別に書かれ、事実上同じ走査ロジックが繰り返されて
いた。走査そのもの (module-level だけを見るか関数本体も見るか、``Import`` /
``ImportFrom`` をどう畳み込むか) をここへ集約し、**正規化 (根だけにする・
末尾だけにする、等) は呼び出し側の責務**として残す —— 呼び出し箇所ごとに
必要な正規化が違う (`.split(".")[0]` で根だけを見たい判定と、``rsplit`` で
末尾のサブモジュール名だけを見たい判定が両方ある) ため、ここで正規化を
固定すると一部の判定の検出力が落ちる。
"""

from __future__ import annotations

import ast
from pathlib import Path


def imported_roots(path: Path, *, include_function_bodies: bool) -> set[str]:
    """``path`` の import 文が指すモジュール名 (未正規化) を集める。

    ``include_function_bodies=False`` なら**関数の外**だけを見る (D-53 の検査。
    ``if TYPE_CHECKING:`` の中も module-level として数える —— 実行時には
    走らないが、そこに置くと循環の解消が型検査の設定に依存する形になり、
    「関数本体の中で import する」という規律が読めなくなる)。

    ``include_function_bodies=True`` なら関数本体の中も数える。D-59 の
    「``tasks`` と ``metrics_detection`` は I/O を持たない」は**関数の中に
    書いても破れてしまう**ので、そちらの検査はこの版を使う。

    返す値は ``import a.b.c`` の ``"a.b.c"`` や ``from a.b import c`` の
    ``"a.b"`` のように**未正規化のまま**である。根だけが要る呼び出し側は
    ``.split(".")[0]`` を、末尾のサブモジュール名だけが要る呼び出し側は
    ``.rsplit(".", 1)[-1]`` を自分で適用すること。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()

    def visit(node: ast.AST, inside_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            is_function = isinstance(
                child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
            )
            if include_function_bodies or not inside_function:
                if isinstance(child, ast.Import):
                    roots.update(alias.name for alias in child.names)
                elif isinstance(child, ast.ImportFrom) and child.module is not None:
                    roots.add(child.module)
            visit(child, inside_function or is_function)

    visit(tree, False)
    return roots


def imported_symbol_names(path: Path) -> set[str]:
    """``from a.b import c`` の ``c`` (元の属性名) を集める。

    ``imported_roots`` は「どこから」import したかを見るのに対し、こちらは
    「何を」import したかを見る (特定の関数・シンボルを実際に import して
    いるかの判定に使う)。module-level・関数本体の両方を対象にする —— 現時点の
    呼び出し箇所は module-level 限定の版を必要としていない。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    return names

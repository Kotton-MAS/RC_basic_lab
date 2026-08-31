"""``docs/モジュール地図.md`` を生成する.

**手書きしない。** 102 本のモジュールを手で並べた表は、次にモジュールを足した
人が写経を忘れた時点で嘘になる。地図が嘘をつくと、地図が無いより悪い
(読者は嘘を確かめる手間を余分に払う)。

各モジュールの1行要約は**既に冒頭 docstring の1行目にある** (実測: 102 本中
101 本が 60 字以内)。それを集めるだけなので、要約の正本はコード側のままである。

``uv run python scripts/module_map.py`` で書き出し、
``tests/test_module_map.py`` が「生成し直しても同じか」を見る。

**行数の列は持たない。** 以前は各モジュールの行数を載せていたが、これが
**マージ衝突の唯一の原因**になった (実測: 3 本連続で PR がこのファイルだけで
衝突した)。行数は無関係なブランチのリファクタでも動くので、並行して作業すると
必ず同じ行を書き換え合う。しかも行数は ``tests/test_module_line_budget.py``
が既に守っており、ここに置くのは二重管理である。

この表が変わるのは**モジュールが増減・改名されたときと、1行要約が変わったとき
だけ**にする。そこは「地図が古くなった」と言うべき変化そのものなので、
衝突しても手で直す価値がある。
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "rc_basics_lab"
OUTPUT = ROOT / "docs" / "モジュール地図.md"

LAYERS: tuple[tuple[str, str], ...] = (
    ("core", "パッケージ直下の単体モジュール (型・指標・入口の道具)"),
    ("config", "実験ごとの設定 dataclass と読み込み規律"),
    ("tasks", "課題 (データ生成)。**純関数層** —— I/O もリザバーも知らない"),
    ("datasets", "外部データの取得・キャッシュ・SHA256。**I/O を持つ唯一の場所**"),
    ("reservoir", "リザバー本体と、モデルを足すための接合面"),
    ("readout", "特徴設計 (FeatureSpec) とリッジ回帰"),
    ("diagnostics", "ESP / 容量 / IPC / リアプノフ。``X`` だけを見る"),
    ("experiment", "合成層。実験の骨格と成果物の書き出し"),
    ("plotting", "作図層"),
)
"""層の並び (依存の向きの順)。**ここに無いディレクトリが増えたら地図に出ない**
ので、``tests/test_module_map.py`` がモジュール数の一致で気づく。"""

ENTRY_POINTS: tuple[tuple[str, str], ...] = (
    ("実験を1本読む", "`experiment/pipeline.py` -> `experiment/runner.py`"),
    (
        "手法の違いがどこにあるか",
        "`readout/design.py` の `_layout_of` (分岐はここだけ)",
    ),
    ("リザバーを足す", "`reservoir/protocol.py` と `reservoir/registry.py`"),
    ("図を1枚直す", "`plotting/style.py` の `new_figure` / `save_png` から辿る"),
    ("設定を1つ足す", "`config/0N.py` の dataclass -> 値を変えたら出力が変わるテスト"),
)
"""**最初に開く場所**。目的から入口へ引く索引で、モジュール一覧の前に置く。"""


def summary_of(path: Path) -> str:
    """冒頭 docstring の1行目 (末尾の句点は落とす)。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    doc = ast.get_docstring(tree, clean=False)
    if not doc:
        return "(要約なし)"
    return doc.splitlines()[0].strip().rstrip("。.")


def modules_by_layer() -> dict[str, list[tuple[str, str]]]:
    """層 -> ``(モジュール名, 要約)`` の並び。**行数は持たない** (上の注を参照)。"""
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "__init__.py" or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(SRC)
        layer = relative.parts[0] if len(relative.parts) > 1 else "core"
        grouped[layer].append((relative.as_posix(), summary_of(path)))
    return grouped


def package_edges() -> list[tuple[str, str, bool]]:
    """層と層のあいだの import の辺を**実際のコードから**数える。

    ``LAYERS`` の並びから直線の鎖を描くと嘘になる (実測: 依存は直線ではなく、
    ``experiment`` は ``tasks`` / ``readout`` / ``reservoir`` / ``diagnostics``
    を直に読む)。**嘘の図は図が無いより悪い** —— 読者は嘘を確かめる手間を
    余分に払う。

    **module-level の import と関数内の import を区別する。** ``experiment`` は
    ``plotting`` を module-level import しない (D-53) ので、そこを実線で描くと
    規約違反があるように見える。関数内だけの辺は破線にする。

    Returns:
        ``(呼ぶ側, 呼ばれる側, module-level か)`` の並び (``LAYERS`` の順)。
    """
    order = [name for name, _ in LAYERS]
    found: set[tuple[str, str, bool]] = set()
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(SRC)
        source = relative.parts[0] if len(relative.parts) > 1 else "core"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_level = {id(node) for node in tree.body}
        for node in ast.walk(tree):
            target = _imported_layer(node)
            if target is not None and target != source:
                found.add((source, target, id(node) in module_level))
    # 同じ辺に module-level と関数内の両方があれば module-level を採る
    strongest: dict[tuple[str, str], bool] = {}
    for source, target, at_module_level in found:
        key = (source, target)
        strongest[key] = strongest.get(key, False) or at_module_level
    return [
        (source, target, strongest[(source, target)])
        for source, target in sorted(
            strongest, key=lambda e: (order.index(e[0]), order.index(e[1]))
        )
        if source in order and target in order
    ]


def _imported_layer(node: ast.AST) -> str | None:
    """``rc_basics_lab.<層>`` を読む import なら層の名前を返す。"""
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
    elif isinstance(node, ast.Import):
        module = node.names[0].name
    else:
        return None
    if not module.startswith("rc_basics_lab"):
        return None
    parts = module.split(".")
    if len(parts) < 2:
        return None
    child = parts[1]
    return child if child in {name for name, _ in LAYERS} else "core"


def render() -> str:
    """Markdown 本文を組み立てる。"""
    grouped = modules_by_layer()
    total = sum(len(items) for items in grouped.values())
    out: list[str] = [
        "# モジュール地図",
        "",
        f"`src/rc_basics_lab/` の **{total} モジュール**を層ごとに並べる。",
        "",
        "**このファイルは生成物である。** 手で直さず "
        "`uv run python scripts/module_map.py` を実行する "
        "(`tests/test_module_map.py` が生成し直した結果と照合する)。",
        "各行の要約は各モジュールの冒頭 docstring の1行目で、正本はコード側にある。",
        "",
        "## 目的から入口へ",
        "",
        "| やりたいこと | まず開く |",
        "|---|---|",
    ]
    out += [f"| {goal} | {entry} |" for goal, entry in ENTRY_POINTS]
    out += [
        "",
        "## 依存の向き",
        "",
        "**実際の import から起こしている**",
        "",
        "```mermaid",
        "graph LR",
    ]
    edges = package_edges()
    drawn = {node for source, target, _ in edges for node in (source, target)}
    out += [f"  {name}[{name}]" for name, _ in LAYERS if name in drawn]
    out += [
        f"  {source} {'-->' if at_module_level else '-.->'} {target}"
        for source, target, at_module_level in edges
    ]
    out += [
        "```",
        "",
        "実線が module-level の import、**破線が関数内の import** である。",
        "",
        "**逆流させない。** `experiment` は `plotting` を module-level import しない",
        "(D-53、作図は関数内 import)。`tasks` と `metrics*` は純関数層で",
        "I/O を持たない (D-59)。外部 I/O は `datasets` だけ。",
        "",
        "## 層ごとのモジュール",
        "",
    ]
    for layer, description in LAYERS:
        items = grouped.get(layer, [])
        out += [f"### `{layer}/` — {description}", ""]
        if not items:
            out += ["(モジュールなし)", ""]
            continue
        out += ["| モジュール | 何をするか |", "|---|---|"]
        out += [f"| `{name}` | {summary} |" for name, summary in items]
        out += [""]
    unknown = sorted(set(grouped) - {layer for layer, _ in LAYERS})
    if unknown:
        out += [
            "### 層に登録されていないモジュール",
            "",
            "`scripts/module_map.py` の `LAYERS` に追記してください。",
            "",
        ]
        for layer in unknown:
            out += [f"| `{name}` | {summary} |" for name, summary in grouped[layer]]
        out += [""]
    return "\n".join(out)


def main() -> int:
    """地図を書き出す。"""
    OUTPUT.write_text(render(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

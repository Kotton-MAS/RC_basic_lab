"""図の外枠 — 空チェック / rcParams の一時適用 / ``Figure`` の生成 / PNG の書き出し.

作図モジュール4本 (``figures`` / ``figures_esp`` / ``figures_capacity`` /
``figures_freerun``) が持っていた共通の外枠をここ1か所に集める。各モジュールが
持つのは「何を描くか」だけになる。

**なぜ1か所か**: 以前は ``_new_figure`` / ``_save`` が4モジュールに同一のコピーで
置かれ、``matplotlib.rc_context(...)  # type: ignore[arg-type]`` が16か所に散って
いた。下の型スタブの折り合いは本質的に1か所の問題であり、16回書けば16回ぶん
ドリフトする余地が生まれる。

**保存を ``figure_canvas`` の内側に置く理由**: ``savefig.dpi`` (D-10 の 200 dpi) と
``savefig.bbox`` は rcParams なので、``savefig`` が ``rc_context`` の**内側**で
呼ばれないと効かない。以前はこれが16か所の書き方の規律だった。文脈マネージャが
退出時に保存すれば、順序を間違える書き方そのものが存在しなくなる。

pyplot を使わず ``Figure`` + ``FigureCanvasAgg`` を直接組むのは、CI にディスプレイが
無いため既定バックエンドに依存させないため。描画設定はプロセス全体の
``matplotlib.rcParams`` を書き換えず、描画中だけ一時適用する (F-1-008)。
"""

from __future__ import annotations

from collections.abc import Iterator, Sized
from contextlib import contextmanager
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from rc_basics_lab.plotting.style import StyleContext, rc_params_for


def require_non_empty(values: Sized, name: str) -> None:
    """空の入力を ``ValueError`` にする。

    空のまま描くと軸だけの図が黙って生成され、成果物としては正常に見える。

    Raises:
        ValueError: ``values`` が空の場合。
    """
    if len(values) == 0:
        raise ValueError(f"{name} が空です")


def new_figure(width: float, height: float) -> Figure:
    """constrained layout の ``Figure`` を作る (軸ラベルとタイトルの重なりを防ぐ)。"""
    figure = Figure(figsize=(width, height))
    figure.set_layout_engine("constrained")
    return figure


def save_figure(figure: Figure, path: Path) -> Path:
    """Agg キャンバスで PNG を書く (ディスプレイに依存しない)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)
    figure.savefig(path, format="png")
    return path


@contextmanager
def figure_canvas(
    path: Path, *, style: StyleContext, width: float, height: float
) -> Iterator[Figure]:
    """描画設定を一時適用した ``Figure`` を渡し、退出時に ``path`` へ保存する。

    本体が例外を投げた場合は保存しない (壊れた PNG を成果物として残さない)。

    Args:
        path: 出力先 PNG。
        style: ``setup_style()`` の戻り値 (ラベル言語と rcParams を決める)。
        width: 図の幅 (インチ)。
        height: 図の高さ (インチ)。
    """
    # matplotlib の rc_context 型スタブは rcParams キーの閉じた Literal 集合を
    # 要求するが、rc_params_for は動的に決まる部分集合の dict[str, object] を
    # 返す。キー自体は既定の rcParams から取った有効なものなので実行時は安全。
    # プロジェクト全体で `# type: ignore` はこの1か所だけである。
    with matplotlib.rc_context(rc_params_for(style)):  # type: ignore[arg-type]
        figure = new_figure(width, height)
        yield figure
        save_figure(figure, path)


__all__ = ["figure_canvas", "new_figure", "require_non_empty", "save_figure"]

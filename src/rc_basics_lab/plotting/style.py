"""図のスタイル設定と CJK フォント探索 (D-10).

新規依存 (``japanize-matplotlib`` 等) を足さず、実行時に
``matplotlib.font_manager`` が知っているフォント名を調べる。見つかれば日本語
ラベル、見つからなければ英語ラベルへ切り替える (``labels.label``)。

記事用の図は CJK フォントのあるローカルで生成し、CI は「生成できること」だけを
検証する。

``setup_style()`` は CJK フォントの探索と ``StyleContext`` の生成だけを行い、
``matplotlib.rcParams`` をプロセス全体に書き換えない (F-1-008)。実際の描画設定は
``rc_params_for`` が返す辞書を呼び出し側が ``matplotlib.rc_context`` に渡して
描画中だけ一時適用する (``plotting/figures.py`` が行う)。

**配色と footnote の単一の真実もここに置く** (図の設計方針 FIG-5 / FIG-6、
D-85 / D-86 / D-87)。各 ``figures_*.py`` が自前で色を選ぶと、記事をまたいで
同じ対照群が違う色で出る (実測: 3-A viridis / 3-B tab:blue+tab:orange /
3-C 単色)。読者は5記事を続けて読むので、色の意味は連載通しで固定する。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from rc_basics_lab.plotting.labels import label

logger = logging.getLogger(__name__)

CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "Hiragino Sans",
    "Noto Sans CJK JP",
    "IPAexGothic",
    "Yu Gothic",
)
"""探索する CJK フォント名 (先頭から順に採用する)。"""

SAVEFIG_DPI = 200
"""保存時の dpi。retina 相当 (仕様 §3: 図は 200 dpi)。"""

FIGURE_DPI = 100
"""画面表示時の dpi。"""


@dataclass(frozen=True, slots=True)
class StyleContext:
    """``setup_style`` が返す描画コンテキスト。

    Attributes:
        cjk_font: 採用した CJK フォント名。見つからなければ ``None``。
    """

    cjk_font: str | None = None

    @property
    def cjk_available(self) -> bool:
        """CJK フォントが利用可能か。"""
        return self.cjk_font is not None

    def label(self, ja: str, en: str) -> str:
        """ラベル文字列を選ぶ (``labels.label`` への薄い委譲)。"""
        return label(ja, en, cjk=self.cjk_available)


def _available_font_names() -> frozenset[str]:
    """matplotlib が認識しているフォント名の集合。

    テストはここを差し替えて「CJK フォントが1つも無い環境」を作る
    (``tests/test_plotting_style.py::test_labels_fall_back_to_english_without_cjk_font``)。
    """
    return frozenset(entry.name for entry in font_manager.fontManager.ttflist)


def find_cjk_font(candidates: Sequence[str] = CJK_FONT_CANDIDATES) -> str | None:
    """利用可能な CJK フォント名を1つ返す。無ければ ``None``。"""
    available = _available_font_names()
    for name in candidates:
        if name in available:
            return name
    return None


def setup_style() -> StyleContext:
    """CJK フォントの有無を判定し ``StyleContext`` を返す。

    ``matplotlib.rcParams`` は書き換えない (F-1-008)。CJK フォントが見つから
    ない場合は ``logger.warning`` を出し、``cjk_available=False`` のコンテキスト
    を返す (呼び出し側は英語ラベルで描く)。実際の描画設定は ``rc_params_for``
    を経由して ``matplotlib.rc_context`` で一時適用する。
    """
    cjk_font = find_cjk_font()
    if cjk_font is None:
        logger.warning(
            "CJK フォントが見つかりません (候補: %s)。図のラベルを英語で描きます。",
            ", ".join(CJK_FONT_CANDIDATES),
        )
        return StyleContext(cjk_font=None)
    logger.info("CJK フォントを使用します: %s", cjk_font)
    return StyleContext(cjk_font=cjk_font)


def rc_params_for(style: StyleContext) -> dict[str, object]:
    """``style`` に応じた rcParams の差分 (``matplotlib.rc_context`` に渡す辞書)。

    プロセス全体の rcParams は書き換えず、描画のたびにこの辞書を
    ``with matplotlib.rc_context(rc_params_for(style)):`` で一時適用する
    (F-1-008)。``savefig.dpi`` は ``SAVEFIG_DPI`` (>= 200) に固定する。
    """
    params: dict[str, object] = {
        "savefig.dpi": SAVEFIG_DPI,
        "figure.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "figure.autolayout": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "legend.frameon": False,
    }
    if style.cjk_font is not None:
        sans_serif = list(matplotlib.rcParams["font.sans-serif"])
        params["font.family"] = "sans-serif"
        params["font.sans-serif"] = [
            style.cjk_font,
            *(name for name in sans_serif if name != style.cjk_font),
        ]
        # CJK フォントは U+2212 (MINUS SIGN) を持たないことがあり、負の目盛が
        # 豆腐になる。ASCII のハイフンに落とす。
        params["axes.unicode_minus"] = False
    return params


@contextmanager
def rc_context_for(style: StyleContext) -> Iterator[None]:
    """``rc_params_for(style)`` を描画中だけ一時適用する.

    ``matplotlib.rc_context`` の型スタブは rcParams キーの閉じた Literal 集合を
    要求するが、``rc_params_for`` は動的に決まる部分集合の ``dict[str, object]``
    を返す。キー自体は既定の rcParams から取った有効なものなので実行時は安全
    —— この ``# type: ignore[arg-type]`` は、以前は4本の作図モジュールの
    呼び出し側16箇所に同一のコメントごと複製されていた。ここへ1箇所へ集約する。
    """
    with matplotlib.rc_context(rc_params_for(style)):  # type: ignore[arg-type]
        yield


def require_rows(rows: Sequence[object]) -> None:
    """``rows`` が空なら ``ValueError`` を送出する.

    作図関数の入口で個別に書かれていた
    ``if not rows: raise ValueError("rows が空です")`` の重複 (作図層12箇所) を
    まとめる。
    """
    if not rows:
        raise ValueError("rows が空です")


def new_figure(width: float, height: float) -> Figure:
    """constrained layout の ``Figure`` を作る (4本の作図モジュール共通の規律)。

    ``constrained`` layout engine が軸ラベルとタイトルの重なりを防ぐ。
    """
    figure = Figure(figsize=(width, height))
    figure.set_layout_engine("constrained")
    return figure


def save_png(figure: Figure, path: Path) -> Path:
    """Agg キャンバスで PNG を書く (ディスプレイに依存しない)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)
    figure.savefig(path, format="png")
    return path


def unique_sorted(values: Sequence[float]) -> tuple[float, ...]:
    """重複を除いて昇順に並べる (格子の軸を行から復元する)。"""
    return tuple(sorted(set(values)))


__all__ = [
    "CJK_FONT_CANDIDATES",
    "FIGURE_DPI",
    "SAVEFIG_DPI",
    "StyleContext",
    "find_cjk_font",
    "new_figure",
    "rc_context_for",
    "rc_params_for",
    "require_rows",
    "save_png",
    "setup_style",
    "unique_sorted",
]

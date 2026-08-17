"""図のスタイル設定と CJK フォント探索 (D-10).

新規依存 (``japanize-matplotlib`` 等) を足さず、実行時に
``matplotlib.font_manager`` が知っているフォント名を調べる。見つかれば日本語
ラベル、見つからなければ英語ラベルへ切り替える (``labels.label``)。

記事用の図は CJK フォントのあるローカルで生成し、CI は「生成できること」だけを
検証する。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib
from matplotlib import font_manager

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
    """rcParams を設定し、CJK フォントの有無を返す。

    ``savefig.dpi`` は ``SAVEFIG_DPI`` (>= 200) に固定する。CJK フォントが
    見つからない場合は ``logger.warning`` を出し、``cjk_available=False`` の
    コンテキストを返す (呼び出し側は英語ラベルで描く)。
    """
    matplotlib.rcParams.update(
        {
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
    )
    cjk_font = find_cjk_font()
    if cjk_font is None:
        logger.warning(
            "CJK フォントが見つかりません (候補: %s)。図のラベルを英語で描きます。",
            ", ".join(CJK_FONT_CANDIDATES),
        )
        return StyleContext(cjk_font=None)
    sans_serif = list(matplotlib.rcParams["font.sans-serif"])
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        cjk_font,
        *(name for name in sans_serif if name != cjk_font),
    ]
    # CJK フォントは U+2212 (MINUS SIGN) を持たないことがあり、負の目盛が
    # 豆腐になる。ASCII のハイフンに落とす。
    matplotlib.rcParams["axes.unicode_minus"] = False
    logger.info("CJK フォントを使用します: %s", cjk_font)
    return StyleContext(cjk_font=cjk_font)


__all__ = [
    "CJK_FONT_CANDIDATES",
    "FIGURE_DPI",
    "SAVEFIG_DPI",
    "StyleContext",
    "find_cjk_font",
    "setup_style",
]

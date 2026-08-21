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
from typing import TypedDict

import matplotlib
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from rc_basics_lab.plotting.labels import label
from rc_basics_lab.types import FloatArray

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

LINEAR_METHOD = "linear"
DELAY_LINE_METHOD = "delay_line"
DELAY_LINE_OLS_METHOD = "delay_line_ols"
ESN_METHOD = "esn"
"""手法の識別子 (``experiment.runner`` の手法名と同じ綴り)。

``delay_line_ols`` は正則化なし最小二乗の遅延線 (Goudarzi 2014 の対照) 用に
**先に色だけ確保してある**。実験として足すかどうかは別の判断だが、色を後から
足すと既存4記事の配色がずれる。
"""

METHOD_COLORS: dict[str, str] = {
    LINEAR_METHOD: "#8073ac",
    DELAY_LINE_METHOD: "#2166ac",
    DELAY_LINE_OLS_METHOD: "#67a9cf",
    ESN_METHOD: "#1a9850",
}
"""手法 (カテゴリ) の固定色 —— **連載5記事で共通** (FIG-5、D-85)。

遅延線の2水準は同系の青にして「同じ手法の正則化違い」であることを色で示す。
ESN の緑と 05 の ``figures_anomaly.METHOD_COLORS[ESN_RESIDUAL]`` は同じ値で、
記事をまたいでも ESN は緑である。
"""

SEQUENTIAL_CMAP = "viridis"
"""連続量 (rho / リーク率 / N の掃引) の配色 —— **連載5記事で共通** (FIG-5)。"""

_SEQUENTIAL_SPAN = (0.0, 0.9)
"""``SEQUENTIAL_CMAP`` から取る区間 (上端は明るすぎて白背景で読めない)。"""

REFERENCE_COLOR = "#000000"
"""参照線・理論上限の色。**データ系列と絶対に同色にしない** (FIG-5、D-86)。

黒を選ぶのは、手法4色・05 の系統6色・04 の3態色のいずれとも重ならず、かつ
「測ったものではなく引いた線」であることが体裁だけで読めるため。
"""

REFERENCE_LINEWIDTH = 1.2
"""参照線の線幅。"""

REFERENCE_DASHES: tuple[tuple[float, float], ...] = ((6.0, 3.0), (2.0, 2.0))
"""参照線の破線パターン。

色は1つに固定するので、同じ図に参照線が2本以上あるときは破線の刻みで
区別する (3-C の NMSE = 0.16 / 0.107)。
"""

FOOTNOTE_FONTSIZE = 7
"""再現条件の footnote の文字サイズ (FIG-6)。"""

FOOTNOTE_COLOR = "#555555"
"""footnote の色 (データではないので本文より薄くする)。"""

COMMIT_LENGTH = 7
"""footnote に載せるコミットハッシュの桁数。"""


@dataclass(frozen=True, slots=True)
class StyleContext:
    """``setup_style`` が返す描画コンテキスト。

    Attributes:
        cjk_font: 採用した CJK フォント名。見つからなければ ``None``。
        commit: 図を生成したときの HEAD (footnote に焼き込む。FIG-6)。
            与えられなければ footnote から ``commit=`` の項が落ちる。
    """

    cjk_font: str | None = None
    commit: str | None = None

    @property
    def cjk_available(self) -> bool:
        """CJK フォントが利用可能か。"""
        return self.cjk_font is not None

    def label(self, ja: str, en: str) -> str:
        """ラベル文字列を選ぶ (``labels.label`` への薄い委譲)。"""
        return label(ja, en, cjk=self.cjk_available)


def method_color(method: str) -> str:
    """手法名から固定色を引く。**未知の手法は描く前に落とす** (FIG-5、D-85)。

    既定色にフォールバックすると、手法を1つ足したときに「なぜか他と同じ色の
    系列が2本ある」図が静かに出る。色の対応表は連載の約束なので、足すなら
    ``METHOD_COLORS`` に足す。
    """
    color = METHOD_COLORS.get(method)
    if color is None:
        raise ValueError(
            f"手法の色が決まっていません: {method!r} "
            f"(style.METHOD_COLORS に足してください: {sorted(METHOD_COLORS)})"
        )
    return color


def sequential_colors(count: int) -> FloatArray:
    """連続量の掃引に使う ``count`` 色 (FIG-5: viridis 系で統一)。

    ``count`` が 0 以下でも空配列を返さず1色ぶん確保する (格子が1点しかない
    縮退ケースで ``colors[0]`` が落ちないようにする)。
    """
    low, high = _SEQUENTIAL_SPAN
    ramp: FloatArray = np.linspace(low, high, max(count, 1))
    colors: FloatArray = matplotlib.colormaps[SEQUENTIAL_CMAP](ramp)
    return colors


class ReferenceLineStyle(TypedDict):
    """参照線に渡す描画引数 (``axhline`` / ``plot`` の keyword)。

    ``dict[str, object]`` にすると ``**`` 展開が mypy strict で通らない
    (``axhline`` の位置引数と衝突する)。キーを型で固定する。
    """

    color: str
    linestyle: tuple[float, tuple[float, float]]
    linewidth: float


def reference_line_kwargs(index: int = 0) -> ReferenceLineStyle:
    """参照線の体裁 (色・破線・線幅) を返す (FIG-5、D-86)。

    ``axis.axhline(value, **reference_line_kwargs(), label=...)`` の形で使う。
    ``index`` は同じ図に複数本の参照線があるときの破線パターンの選択で、
    **色は常に ``REFERENCE_COLOR``** である。
    """
    dashes = REFERENCE_DASHES[index % len(REFERENCE_DASHES)]
    return ReferenceLineStyle(
        color=REFERENCE_COLOR,
        linestyle=(0.0, dashes),
        linewidth=REFERENCE_LINEWIDTH,
    )


def replicates_field(replicates: Sequence[int]) -> str:
    """footnote の「何本の繰り返しか」の項を作る (FIG-6)。

    載せるのは**レプリケート番号**であってシード値ではない。行が持つ
    ``seed_*`` 列は基底シードで、全レプリケートで同じ値 (実測: 03 の
    ``capacity.csv`` は 117 行すべて ``seed_reservoir=0``) であり、
    実際に振られているのはレプリケート番号のほうである。

    Raises:
        ValueError: ``replicates`` が空の場合。
    """
    unique = sorted(set(replicates))
    if not unique:
        raise ValueError("replicates が空です")
    span = f"{unique[0]}" if len(unique) == 1 else f"{unique[0]}-{unique[-1]}"
    return f"{len(unique)} rep (replicate {span})"


def footnote_text(conditions: str, *, commit: str | None) -> str:
    """再現条件の1行を組み立てる (FIG-6)。

    **ASCII だけで組む**。CJK フォントの有無で内容が変わると、英語版の図と
    日本語版の図で「焼き込まれた条件」が食い違う (D-10 の切り替えは
    ラベル文字列の話であって、再現条件の話ではない)。
    """
    if not conditions:
        raise ValueError("conditions が空です")
    if commit is None:
        return conditions
    return f"{conditions}, commit={commit[:COMMIT_LENGTH]}"


def add_footnote(figure: Figure, conditions: str, *, style: StyleContext) -> None:
    """図の右下に再現条件を焼き込む (FIG-6、D-87)。

    ``meta.json`` にある情報は、図が単体で流通した時点で失われる。タイトルへ
    条件を書く方式は図ごとに項目が揃わないので、**全図共通の footnote へ機械
    生成で移す**。
    """
    figure.text(
        1.0,
        0.0,
        footnote_text(conditions, commit=style.commit),
        ha="right",
        va="bottom",
        fontsize=FOOTNOTE_FONTSIZE,
        color=FOOTNOTE_COLOR,
    )


def add_provenance(
    figure: Figure,
    conditions: str,
    replicates: Sequence[int],
    *,
    style: StyleContext,
) -> None:
    """再現条件 + レプリケート数の footnote を書く (FIG-6 / D-87)。

    ``figures_*.py`` が各自で組み立てると項目の並びが図ごとにずれるので、
    **組み立てはここ1か所**にする。呼び出し側が渡すのは図ごとに違う条件文字列
    (``N = 200, sigma_u = 0.1``) だけである。
    """
    add_footnote(figure, f"{conditions}, {replicates_field(replicates)}", style=style)


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


def setup_style(*, commit: str | None = None) -> StyleContext:
    """CJK フォントの有無を判定し ``StyleContext`` を返す。

    ``matplotlib.rcParams`` は書き換えない (F-1-008)。CJK フォントが見つから
    ない場合は ``logger.warning`` を出し、``cjk_available=False`` のコンテキスト
    を返す (呼び出し側は英語ラベルで描く)。実際の描画設定は ``rc_params_for``
    を経由して ``matplotlib.rc_context`` で一時適用する。

    Args:
        commit: footnote に焼き込む HEAD (FIG-6)。**git はここでは呼ばない**
            —— 作図層が subprocess を持つと、テストが図を描くたびに git が
            起動する。実験層 (``meta.git_commit()`` を1回呼ぶ側) から渡す。
    """
    cjk_font = find_cjk_font()
    if cjk_font is None:
        logger.warning(
            "CJK フォントが見つかりません (候補: %s)。図のラベルを英語で描きます。",
            ", ".join(CJK_FONT_CANDIDATES),
        )
        return StyleContext(cjk_font=None, commit=commit)
    logger.info("CJK フォントを使用します: %s", cjk_font)
    return StyleContext(cjk_font=cjk_font, commit=commit)


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
    "COMMIT_LENGTH",
    "DELAY_LINE_METHOD",
    "DELAY_LINE_OLS_METHOD",
    "ESN_METHOD",
    "FIGURE_DPI",
    "FOOTNOTE_COLOR",
    "FOOTNOTE_FONTSIZE",
    "LINEAR_METHOD",
    "METHOD_COLORS",
    "REFERENCE_COLOR",
    "REFERENCE_DASHES",
    "REFERENCE_LINEWIDTH",
    "SAVEFIG_DPI",
    "SEQUENTIAL_CMAP",
    "ReferenceLineStyle",
    "StyleContext",
    "add_footnote",
    "add_provenance",
    "find_cjk_font",
    "footnote_text",
    "method_color",
    "new_figure",
    "rc_context_for",
    "rc_params_for",
    "reference_line_kwargs",
    "replicates_field",
    "require_rows",
    "save_png",
    "sequential_colors",
    "setup_style",
    "unique_sorted",
]

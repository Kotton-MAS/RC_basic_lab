"""記事03の図が読む「行 → 格子」の復元 (D-38 の長形式から).

``figures_capacity`` から**描かない部分だけ**を切り出したモジュール。
``capacity_profile.csv`` の長形式の行を (rho x 遅延) / (次数 x 遅延) の配列へ
戻す処理と、上限線の座標計算が入る。描画 (matplotlib への呼び出し) は1つも
含まない —— ここはテストから配列として検算できる層である。

切り出した直接の理由は D-77 の行数ラチェットで、``figures_capacity.py`` は
凍結値 861 行から1行も増やせない。図の設計方針 (FIG-1〜FIG-7) が要求する
出典・footnote・打ち切りの明示を足すぶんの場所を、**描画でない部分を外へ
出して**作った。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from rc_basics_lab.experiment.capacity import (
    DIAGNOSTIC_IPC,
    DIAGNOSTIC_MC,
    CapacityProfileRow,
    CapacityRow,
)
from rc_basics_lab.plotting.style import require_rows, unique_sorted
from rc_basics_lab.types import FloatArray

BOUND_MARGIN = 0.1
"""上限線 y=N (対角線) を格子の外へ伸ばす割合。

``n_units`` が1点しかない縮退ケース (縮小設定) でも**線**として描けるように
両端へ余白を取る。1点だけを結ぶと長さ 0 の線分になり、図の主張である
「傾き1の対角線」が消える。
"""


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """レプリケート平均と標準偏差 (母標準偏差、``ddof=0``)。

    レプリケートが1本しかない縮退ケースでも落ちないよう ``np.std`` を使う
    (``statistics.stdev`` は n=1 で例外になる)。
    """
    if not values:
        return float("nan"), float("nan")
    array: FloatArray = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array))


def n_replicates(rows: Sequence[CapacityRow]) -> int:
    """行に現れるレプリケートの本数。

    長形式の行は**正値セルだけ**なので、容量が 0 のレプリケートは1行も
    書かれていない。平均を「在る行の数」で割ると、0 のレプリケートを無視した
    過大評価になる。分母はここ (``capacity.csv`` 側の行) から取る。
    """
    return max(len({row.replicate for row in rows}), 1)


def _for_rows(
    profile: Sequence[CapacityProfileRow], rows: Sequence[CapacityRow]
) -> tuple[CapacityProfileRow, ...]:
    """``rows`` と同じ実験ラベルの長形式の行だけに絞る。

    3-A と 3-B は (rho, leak_rate) の格子が一部重なるため、実験ラベルで
    絞らずに読むと別の実験のセルが図に混ざりうる。呼び出し側の絞り込み漏れを
    ここで塞ぐ (絞り込み済みの並びを渡しても結果は変わらない)。
    """
    experiments = {row.experiment for row in rows}
    return tuple(row for row in profile if row.experiment in experiments)


def representative_leak_rate(
    rows: Sequence[CapacityRow], value_of: Callable[[CapacityRow], float]
) -> float:
    """パネルに出す代表リーク率を選ぶ (``value_of`` の平均が最大のもの)。

    3-A の右パネルと 3-B のヒートマップは1つのリーク率しか描けない。**総容量が
    最も大きい動作点**を代表にするのは、その図が見せたい構造 (プロファイルの
    伸び / 次数の配分) が最も読み取れる点だからである。同点のときは小さい方を
    採り、選択が行の並び順に依存しないようにする。

    ``key: str`` + ``getattr`` (列名を文字列指定) ではなく
    ``Callable[[CapacityRow], float]`` にしてあるのは (F-3b1-1-007)、
    ``getattr`` の戻り値は mypy が ``Any`` とみなすため列名のタイプミスや
    ``CapacityRow`` のリネームが図の生成時 (``AttributeError``) まで検出
    されなかったため。呼び出し側は ``lambda row: row.mc_total`` のように渡す。

    Args:
        rows: 1実験ぶんの行。
        value_of: 比較に使う値を1行から取り出す関数
            (``lambda row: row.mc_total`` / ``lambda row: row.ipc_total``)。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    require_rows(rows)
    totals: dict[float, list[float]] = {}
    for row in rows:
        totals.setdefault(row.leak_rate, []).append(float(value_of(row)))
    return min(totals, key=lambda leak: (-float(np.mean(totals[leak])), leak))


def mc_profile_means(
    rows: Sequence[CapacityRow], profile: Sequence[CapacityProfileRow], leak_rate: float
) -> dict[float, FloatArray]:
    """MC の遅延プロファイルを rho ごとにレプリケート平均する (D-38 の長形式から)。

    長形式には正値セルしか無いので、``(n_delays,)`` の 0 配列に足し込んでから
    レプリケート数で割る。``n_delays`` は ``capacity.csv`` 側の列
    (``mc.max_delay``) を使う —— 長形式の最大遅延から決めると、条件によって
    横軸の長さが変わって rho 間の比較ができなくなる。
    """
    n_delays = max((row.n_delays for row in rows), default=0)
    divisor = float(n_replicates(rows))
    means = {
        rho: np.zeros(n_delays, dtype=np.float64)
        for rho in unique_sorted([row.rho for row in rows])
    }
    for row in _for_rows(profile, rows):
        if row.diagnostic != DIAGNOSTIC_MC or row.leak_rate != leak_rate:
            continue
        cells = means.get(row.rho)
        if cells is None or not 1 <= row.delay <= n_delays:
            continue
        cells[row.delay - 1] += row.capacity / divisor
    return means


def ipc_heatmap_means(
    rows: Sequence[CapacityRow], profile: Sequence[CapacityProfileRow], leak_rate: float
) -> dict[float, FloatArray]:
    """IPC の (次数 x 遅延) 容量を rho ごとにレプリケート平均する (D-38 の長形式から)。

    形は全パネルで共通の ``(max(n_degrees), 最大遅延)`` にそろえる。遅延の
    打ち切りは次数ごとに違う (本番は 60/20/10/6) ので、打ち切りの外は 0 のまま
    残る。**その 0 は「容量が無い」ではなく「測っていない」**なので、描く側は
    打ち切りをマスクして別の色にする (FIG-7 / D-88、``plotting/heatmap.py``)。
    """
    n_degrees = max((row.n_degrees for row in rows), default=1)
    cells = _for_rows(profile, rows)
    n_delays = max(
        (row.delay for row in cells if row.diagnostic == DIAGNOSTIC_IPC), default=1
    )
    divisor = float(n_replicates(rows))
    means = {
        rho: np.zeros((n_degrees, n_delays), dtype=np.float64)
        for rho in unique_sorted([row.rho for row in rows])
    }
    for row in cells:
        if row.diagnostic != DIAGNOSTIC_IPC or row.leak_rate != leak_rate:
            continue
        grid = means.get(row.rho)
        if grid is None or not 1 <= row.degree <= n_degrees:
            continue
        grid[row.degree - 1, row.delay - 1] += row.capacity / divisor
    return means


def conservation_bound(units: Sequence[int]) -> tuple[FloatArray, FloatArray]:
    """上限線 ``y = N`` (傾き1の対角線) の端点を返す (受け入れ条件2)。

    描画とテストが**同じ1か所**からこの座標を取る。図の主張は
    「``ipc_total`` はこの対角線を超えない」なので、線が消えたことを
    ``test_conservation_figure_draws_the_bound_line`` が検出できる必要がある。

    Raises:
        ValueError: ``units`` が空の場合。
    """
    if not units:
        raise ValueError("units が空です")
    low = min(units) * (1.0 - BOUND_MARGIN)
    high = max(units) * (1.0 + BOUND_MARGIN)
    edge: FloatArray = np.array([low, high], dtype=np.float64)
    return edge, edge


__all__ = [
    "BOUND_MARGIN",
    "conservation_bound",
    "even_degree_share",
    "ipc_heatmap_means",
    "mc_profile_means",
    "mean_std",
    "n_replicates",
    "representative_leak_rate",
    "sweep_conditions",
]


def even_degree_share(profile: Sequence[CapacityProfileRow]) -> float:
    """偶数次の容量が全体に占める割合 (D-94)。

    ``fig_ipc_profile`` では次数2と4のセルが共通の色スケールで 0 と
    見分けられない。**理由が図に無いと、読者は最初にそこで引っかかる**
    (FIG-7 で「未計算」を 0 と区別できるようにした以上、「0 である理由」も
    要る)。割合を数え直して注に埋めるので、掃引の設定が変われば注も変わる。
    """
    total = sum(row.capacity for row in profile)
    if total <= 0.0:
        return 0.0
    even = sum(row.capacity for row in profile if row.degree % 2 == 0)
    return even / total


def even_degree_note(profile: Sequence[CapacityProfileRow]) -> tuple[str, str]:
    """偶数次が空である理由の注 (日本語, 英語) を割合つきで返す (D-94)。"""
    share = even_degree_share(profile)
    return (
        "注: 次数2・4 がほぼ空なのは、駆動入力が対称 (平均0の一様) で tanh が"
        " 奇関数のため、偶数次の項が打ち消し合うからである"
        f" (残る偶数次は全容量の {share:.1%}。厳密に 0 でないのはバイアス項が"
        " 対称性をわずかに破るため)。",
        "Note: degrees 2 and 4 are nearly empty because the drive is symmetric"
        " (zero-mean uniform) and tanh is odd, so even-order terms cancel"
        f" ({share:.1%} of the total capacity remains; the bias term breaks the"
        " symmetry slightly).",
    )


def sweep_conditions(row: CapacityRow, leak_rate: float | None = None) -> str:
    """footnote に載せる掃引条件の1行 (D-87)。

    3-A / 3-B / 3-B' が同じ文字列を手書きで組んでいた。手書きだと図ごとに
    載せる項目がずれる (実測: 条件を書いている図は5枚あり、項目は
    ばらばらだった)。
    """
    text = f"N = {row.n_units}, sigma_u = {row.sigma_u:g}"
    return text if leak_rate is None else f"{text}, a = {leak_rate:g}"

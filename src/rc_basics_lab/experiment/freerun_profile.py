"""自走の「配列」を長形式の行へ畳む層 (D-128).

``freerun.py`` から切り出した (1モジュール 600 行の上限、D-63 / D-77)。

**図は成果物 CSV の行だけを読む。** 位相図・リターンマップ・スペクトルのように
「行」ではなく「配列」で表現される量も、書き出したのと同じ長形式の行として図へ
渡す —— 03 の ``capacity_profile.csv`` と同じ役割で、成果物と図が食い違う経路
(CSV に無いものを図が描く) を構造で塞ぐ。
"""

from __future__ import annotations

import numpy as np

from rc_basics_lab.experiment.attractor import (
    first_autocorrelation_zero,
    power_spectrum,
    return_map_points,
)
from rc_basics_lab.experiment.freerun_rows import (
    FreeRunEvaluation,
    FreeRunProfileRow,
    FreeRunRow,
)
from rc_basics_lab.types import FloatArray

KIND_PHASE = "phase"
"""``FreeRunProfileRow.kind``: 位相図 (Lorenz は (x, z)、1変数系は遅延座標)。"""

KIND_RETURN_MAP = "return_map"
"""``FreeRunProfileRow.kind``: リターンマップ ``(z_n, z_(n+1))`` (D-46 の1本目)。"""

KIND_SPECTRUM = "spectrum"
"""``FreeRunProfileRow.kind``: 正規化パワースペクトル (D-46 の2本目)。"""

SOURCE_TRUTH = "truth"
"""``FreeRunProfileRow.source``: 真の軌道。"""

SOURCE_FREERUN = "freerun"
"""``FreeRunProfileRow.source``: 自走の軌道。"""

PROFILE_MAX_POINTS = 4000
"""確保軸6: 位相図に載せる点数の上限 (間引きの上限)。

PNG のサイズと描画時間はここに比例する。**上書き不能な定数**で、
``freerun_profile_rows`` が ``stats_steps`` に関係なくこの本数まで間引く
(``stats_steps`` を伸ばすと図の点数が黙って増える、を塞ぐ)。
"""


def _phase_points(series: FloatArray, lag: int) -> FloatArray:
    """位相図の2次元投影。多変数なら (第0成分, 最終成分)、1変数なら遅延座標。

    ``lag`` は**真の軌道から**決めた1個を自走側にも使う (別々に決めると同じ
    座標系で重ね描きできない)。
    """
    if series.shape[1] >= 2:
        projected: FloatArray = np.stack([series[:, 0], series[:, -1]], axis=1)
        return projected
    if series.shape[0] <= lag:
        return np.empty((0, 2), dtype=np.float64)
    embedded: FloatArray = np.stack(
        [series[lag:, 0], series[: series.shape[0] - lag, 0]], axis=1
    )
    return embedded


def _thinned(points: FloatArray) -> FloatArray:
    """確保軸6: 図に載せる点数を ``PROFILE_MAX_POINTS`` まで間引く。"""
    if points.shape[0] <= PROFILE_MAX_POINTS:
        return points
    stride = int(np.ceil(points.shape[0] / PROFILE_MAX_POINTS))
    thinned: FloatArray = points[::stride][:PROFILE_MAX_POINTS]
    return thinned


def _profile_block(
    row: FreeRunRow, kind: str, source: str, points: FloatArray
) -> list[FreeRunProfileRow]:
    return [
        FreeRunProfileRow(
            experiment=row.experiment,
            task=row.task,
            method=row.method,
            replicate=row.replicate,
            kind=kind,
            source=source,
            index=index,
            x=float(point[0]),
            y=float(point[1]),
        )
        for index, point in enumerate(points)
    ]


def freerun_profile_rows(
    evaluation: FreeRunEvaluation, dt: float
) -> tuple[FreeRunProfileRow, ...]:
    """図が読む長形式の行を組む (**診断も実験もここでは走らせない**)。

    位相図・リターンマップ・スペクトルの3種類を、真の軌道と自走の両方について
    出す。点数は ``PROFILE_MAX_POINTS`` (確保軸6) で間引く。

    Args:
        evaluation: ``evaluate_free_run`` の結果。
        dt: サンプリング間隔 [時間] (スペクトルの周波数軸)。

    Returns:
        長形式の行。
    """
    row = evaluation.row
    truth = evaluation.truth_series
    trajectory = evaluation.trajectory
    lag = first_autocorrelation_zero(truth)
    rows: list[FreeRunProfileRow] = []
    rows += _profile_block(
        row, KIND_PHASE, SOURCE_TRUTH, _thinned(_phase_points(truth, lag))
    )
    rows += _profile_block(
        row, KIND_RETURN_MAP, SOURCE_TRUTH, _thinned(return_map_points(truth))
    )
    if trajectory.shape[0] >= 3:
        rows += _profile_block(
            row, KIND_PHASE, SOURCE_FREERUN, _thinned(_phase_points(trajectory, lag))
        )
        rows += _profile_block(
            row,
            KIND_RETURN_MAP,
            SOURCE_FREERUN,
            _thinned(return_map_points(trajectory)),
        )
    n_common = min(truth.shape[0], trajectory.shape[0])
    if n_common >= 8:
        for source, series in (
            (SOURCE_TRUTH, truth[:n_common]),
            (SOURCE_FREERUN, trajectory[:n_common]),
        ):
            frequencies, power = power_spectrum(series, dt)
            rows += _profile_block(
                row,
                KIND_SPECTRUM,
                source,
                _thinned(np.stack([frequencies, power], axis=1)),
            )
    return tuple(rows)


__all__ = [
    "KIND_PHASE",
    "KIND_RETURN_MAP",
    "KIND_SPECTRUM",
    "PROFILE_MAX_POINTS",
    "SOURCE_FREERUN",
    "SOURCE_TRUTH",
    "freerun_profile_rows",
]

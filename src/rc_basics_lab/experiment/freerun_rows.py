"""実験04 の成果物の行 dataclass と、行からの要約.

``experiment/freerun.py`` から**行の形と要約だけ**を切り出したモジュール。
``experiment/anomaly_rows.py`` (05) / ``experiment/capacity_rows.py`` (03) と
同じ切り口である。

ここが持つのは「CSV の1行がどんな列を持つか」と「並んだ行をどう畳むか」
だけで、**自走のさせ方も教師強制も知らない**。要約 (有効予測時間の感度、
アトラクタ再現の判定) は行だけを入力に取るので、成果物から独立に検算できる。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rc_basics_lab.experiment.report import DataclassSummaryMixin
from rc_basics_lab.experiment.valid_time import VALID_TIME_THRESHOLD_GRID
from rc_basics_lab.metrics_significance import sign_test_p_value
from rc_basics_lab.types import FloatArray


@dataclass(frozen=True, slots=True)
class FreeRunRow:
    """``freerun.csv`` の1行 = 自走1本 (課題 x 手法 x レプリケート)。

    列順はこの宣言順が単一の真実である (§2.2-1)。

    Attributes:
        experiment: ``EXPERIMENT_FREERUN``。
        task: 課題名。
        method: 手法名 (対照も自走させる)。
        replicate: レプリケート番号。
        seed_reservoir / seed_task / seed_split: 基底シード (D-06)。
        n_units / rho / leak_rate / state_noise: ESN の条件。
        alpha: 教師強制で選ばれた正則化係数。
        val_nrmse: 選択時の検証 NRMSE。
        switch_index: 教師強制から自走へ切り替えた行 index。
        warmup_steps / free_run_steps / stats_steps: 自走の設定。
        dt: サンプリング間隔 [時間]。
        lyapunov_per_time: 数値推定した lambda_max [1/時間] (D-42)。
        lyapunov_time: ``1 / lambda_max`` [時間] (正規化の分母)。
        valid_time_threshold: 有効予測時間の閾値 (NRMSE 比、D-43)。
        valid_time_steps: 有効予測時間 [ステップ]。**生の値だけで報告しない**
            (仕様 §5 禁止する構造5) ので ``valid_time_lyapunov`` を必ず併記する。
        valid_time: 同 [時間]。
        valid_time_lyapunov: 同 [Lyapunov 時間] —— **報告の主指標**。
        valid_time_censored: 自走長の最後まで閾値を超えなかったか (右側打ち切り)。
        diverged / n_completed: 自走の打ち切り (T4)。
        regime: 3態分類 (D-45)。
        amplitude_ratio / std_ratio / autocorr_peak: 分類の根拠になった数値。
        return_map_distance: リターンマップの点集合距離 (D-46 の1本目)。
        return_map_distance_surrogate: 同じ指標を**真の軌道のシャッフル代替**に
            対して測った値。自走の方が小さくなければ「再現した」と言わない。
        spectrum_distance / spectrum_distance_surrogate: パワースペクトルの
            全変動距離と、その代替に対する値 (D-46 の2本目)。
        closer_than_surrogate: **2指標とも**代替より小さいか。
        n_stats_samples / n_return_map_points / n_spectrum_bins: 統計に使った量。
        wall_time_s: 実測 wall time [秒]。
    """

    experiment: str
    task: str
    method: str
    replicate: int
    seed_reservoir: int
    seed_task: int
    seed_split: int
    n_units: int
    rho: float
    leak_rate: float
    state_noise: float
    alpha: float
    val_nrmse: float
    switch_index: int
    warmup_steps: int
    free_run_steps: int
    stats_steps: int
    dt: float
    lyapunov_per_time: float
    lyapunov_time: float
    valid_time_threshold: float
    valid_time_steps: int
    valid_time: float
    valid_time_lyapunov: float
    valid_time_censored: bool
    diverged: bool
    n_completed: int
    regime: str
    amplitude_ratio: float
    std_ratio: float
    autocorr_peak: float
    return_map_distance: float
    return_map_distance_surrogate: float
    spectrum_distance: float
    spectrum_distance_surrogate: float
    closer_than_surrogate: bool
    n_stats_samples: int
    n_return_map_points: int
    n_spectrum_bins: int
    wall_time_s: float


@dataclass(frozen=True, slots=True)
class FreeRunProfileRow:
    """``freerun_profile.csv`` の1行 (図が読む長形式の点)。

    Attributes:
        experiment / task / method / replicate: どの自走の点か。
        kind: ``KIND_PHASE`` / ``KIND_RETURN_MAP`` / ``KIND_SPECTRUM``。
        source: ``SOURCE_TRUTH`` / ``SOURCE_FREERUN``。
        index: 系列内の通し番号 (描画順)。
        x / y: 点の座標 (スペクトルは (周波数, 正規化パワー))。
    """

    experiment: str
    task: str
    method: str
    replicate: int
    kind: str
    source: str
    index: int
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class FreeRunEvaluation:
    """1本の自走に対する評価 (行 + 感度 + 図の材料)。

    Attributes:
        row: ``freerun.csv`` の1行。
        valid_time_by_threshold: ``VALID_TIME_THRESHOLD_GRID`` と同じ並びの
            有効予測時間 [Lyapunov 時間] (閾値感度表の一次資料)。
        censored_by_threshold: 同じ並びの打ち切りフラグ。
        trajectory: 自走の有限行 ``(n_completed, D)``。
        truth_series: 真の系列 ``(T, D)`` (位相図が全体を要る)。
        truth_aligned: **自走区間だけ**の真値 ``(n_completed, D)``。
            時間軸の図はこちらを使う —— ``truth_series`` は系列全体なので
            長さが合わない (実測: truth 8000 / predicted 20000 で落ちた)。
    """

    row: FreeRunRow
    valid_time_by_threshold: tuple[float, ...]
    censored_by_threshold: tuple[bool, ...]
    trajectory: FloatArray
    truth_series: FloatArray
    truth_aligned: FloatArray


@dataclass(frozen=True, slots=True)
class ValidTimeSensitivity(DataclassSummaryMixin):
    """閾値感度表の1行 (``meta.json`` の ``valid_time_sensitivity``)。

    ``docs/design.md`` §12 の感度表はここから機械照合する。閾値を1点だけ
    報告すると「その閾値だから出た結論」を否定できない。

    Attributes:
        task / method: どの群か。
        threshold: NRMSE 比の閾値。
        median_lyapunov: 有効予測時間の中央値 [Lyapunov 時間]。
        min_lyapunov / max_lyapunov: 同じ群の最小・最大。
        n_rows: 群の行数 (= シード数)。
        n_censored: 打ち切られた行数 (**無かったことにしない**)。
    """

    task: str
    method: str
    threshold: float
    median_lyapunov: float
    min_lyapunov: float
    max_lyapunov: float
    n_rows: int
    n_censored: int


def summarize_valid_time(
    evaluations: Sequence[FreeRunEvaluation],
) -> tuple[ValidTimeSensitivity, ...]:
    """閾値 x (課題, 手法) ごとに有効予測時間を要約する (D-43 の感度表)。"""
    groups: dict[tuple[str, str], list[FreeRunEvaluation]] = {}
    for evaluation in evaluations:
        groups.setdefault((evaluation.row.task, evaluation.row.method), []).append(
            evaluation
        )
    summary: list[ValidTimeSensitivity] = []
    for (task, method), items in sorted(groups.items()):
        for position, threshold in enumerate(VALID_TIME_THRESHOLD_GRID):
            values = [item.valid_time_by_threshold[position] for item in items]
            summary.append(
                ValidTimeSensitivity(
                    task=task,
                    method=method,
                    threshold=threshold,
                    median_lyapunov=float(np.median(values)),
                    min_lyapunov=float(np.min(values)),
                    max_lyapunov=float(np.max(values)),
                    n_rows=len(values),
                    n_censored=sum(
                        1 for item in items if item.censored_by_threshold[position]
                    ),
                )
            )
    return tuple(summary)


@dataclass(frozen=True, slots=True)
class AttractorVerdict(DataclassSummaryMixin):
    """アトラクタ再現の判定 (D-46)。**視覚評価は結論に使わない**。

    「シャッフル代替より近い」が (課題, 手法) 群の**全行で**成り立つかを
    符号検定で数える。1行だけ見せると「その回はたまたま」を否定できず、
    中央値だけ見せると分布の片側の外れを隠せる。

    Attributes:
        task / method: 群。
        n_rows: 群の行数 (= シード数)。
        n_closer: 2指標**とも**代替より小さかった行数。
        sign_test_p: 片側符号検定の p 値 (帰無仮説「代替より近くない」)。
        median_return_map / median_return_map_surrogate: 距離の中央値。
        median_spectrum / median_spectrum_surrogate: 同上。
    """

    task: str
    method: str
    n_rows: int
    n_closer: int
    sign_test_p: float
    median_return_map: float
    median_return_map_surrogate: float
    median_spectrum: float
    median_spectrum_surrogate: float


def _median_of(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return float(np.median(finite))


def summarize_attractor(
    evaluations: Sequence[FreeRunEvaluation],
) -> tuple[AttractorVerdict, ...]:
    """(課題, 手法) ごとにアトラクタ再現を要約する (D-46)。"""
    groups: dict[tuple[str, str], list[FreeRunRow]] = {}
    for evaluation in evaluations:
        groups.setdefault((evaluation.row.task, evaluation.row.method), []).append(
            evaluation.row
        )
    verdicts: list[AttractorVerdict] = []
    for (task, method), rows in sorted(groups.items()):
        n_closer = sum(1 for row in rows if row.closer_than_surrogate)
        verdicts.append(
            AttractorVerdict(
                task=task,
                method=method,
                n_rows=len(rows),
                n_closer=n_closer,
                sign_test_p=sign_test_p_value(len(rows), n_closer),
                median_return_map=_median_of([row.return_map_distance for row in rows]),
                median_return_map_surrogate=_median_of(
                    [row.return_map_distance_surrogate for row in rows]
                ),
                median_spectrum=_median_of([row.spectrum_distance for row in rows]),
                median_spectrum_surrogate=_median_of(
                    [row.spectrum_distance_surrogate for row in rows]
                ),
            )
        )
    return tuple(verdicts)


__all__ = [
    "AttractorVerdict",
    "FreeRunEvaluation",
    "FreeRunProfileRow",
    "FreeRunRow",
    "ValidTimeSensitivity",
    "summarize_attractor",
    "summarize_valid_time",
]

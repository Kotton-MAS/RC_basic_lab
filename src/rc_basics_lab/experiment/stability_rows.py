"""04 の行の層 — ``stability.csv`` の行と、診断の長形式への畳み込み.

02 の ``esp_rows.py`` / 03 の ``capacity_rows.py`` と同じ役割である。
``stability.py`` が 600 行の上限 (D-63 / D-77) を超えて凍結されているため、
**行の組み立てはこちらへ置く**。上限のほうを緩めない。

``StabilityRow`` の宣言順が ``stability.csv`` の列順の単一の真実である。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

from rc_basics_lab.diagnostics.base import DiagnosticResult
from rc_basics_lab.experiment.capacity_rows import CapacityRow
from rc_basics_lab.experiment.diagnostics_rows import (
    DiagnosticScalarRow,
    condition_key,
    scalar_rows,
)


@dataclass(frozen=True, slots=True)
class StabilityRow:
    """``stability.csv`` の1行 (4-C の1条件)。列順はこの宣言順が単一の真実。

    容量 (MC / IPC) の列は**ここに複製しない**。4-D の行は同じ条件キー
    (``rho`` / ``leak_rate`` / ``state_noise`` / ``replicate``) を持つ
    ``capacity.csv`` (04) 側にあり、2枚を join すれば「自走が上手くいく領域が
    容量指標で説明できるか」を見られる (03 の ``narma10.csv`` と
    ``capacity.csv`` の関係と同じ)。約35列ある ``CapacityRow`` をここへ写すと
    列の単一の真実が2つになる。

    Attributes:
        experiment: ``EXPERIMENT_STABILITY``。
        rho / leak_rate / state_noise / replicate: 条件。
        n_units: リザバーのユニット数 (掃引では動かさない)。
        alpha / val_nrmse: 教師強制で選ばれた読み出し。
        regime: 3態分類 (D-45)。**純関数 + 数値基準**で決まる。
        amplitude_ratio / std_ratio / autocorr_peak: 分類の根拠になった数値。
        diverged / n_completed: 自走の打ち切り。
        stats_steps: 自走させたステップ数 (4-B と同じ窓で測る)。
        valid_time_threshold / valid_time_steps / valid_time_lyapunov /
        valid_time_censored: 有効予測時間 (D-43。4-B と同じ定義)。
        wall_time_s: 条件の実測 wall time [秒] (状態生成 + 学習 + 自走)。
    """

    experiment: str
    rho: float
    leak_rate: float
    state_noise: float
    replicate: int
    n_units: int
    alpha: float
    val_nrmse: float
    regime: str
    amplitude_ratio: float
    std_ratio: float
    autocorr_peak: float
    diverged: bool
    n_completed: int
    stats_steps: int
    valid_time_threshold: float
    valid_time_steps: int
    valid_time_lyapunov: float
    valid_time_censored: bool
    wall_time_s: float


STABILITY_CSV_COLUMNS: tuple[str, ...] = tuple(
    item.name for item in fields(StabilityRow)
)
"""``stability.csv`` の列順 (``StabilityRow`` の宣言順)。"""


def stability_diagnostic_rows(
    capacity: CapacityRow, results: Sequence[DiagnosticResult]
) -> tuple[DiagnosticScalarRow, ...]:
    """1条件ぶんの診断結果を長形式の行に畳む (D-118)。

    **主表 (``stability.csv`` / ``capacity.csv``) の列は1つも動かさない。**
    軸は 4-C / 4-D が実際に振っている4つだけを入れる。

    Args:
        capacity: その条件の ``capacity.csv`` の行 (軸の値をここから取る)。
        results: その条件で走った診断の結果。

    Returns:
        長形式の行。
    """
    return scalar_rows(
        results,
        experiment=capacity.experiment,
        condition_id=condition_key(
            {
                "rho": capacity.rho,
                "leak_rate": capacity.leak_rate,
                "n_units": capacity.n_units,
                "state_noise": capacity.state_noise,
            }
        ),
        replicate=capacity.replicate,
    )


__all__ = [
    "STABILITY_CSV_COLUMNS",
    "StabilityRow",
    "stability_diagnostic_rows",
]

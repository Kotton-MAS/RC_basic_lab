"""02 の行の層 — ``esp_diagnostics.csv`` の行と、診断の長形式への畳み込み.

03 の ``capacity_rows.py`` と同じ役割である。``esp.py`` が 600 行の上限
(D-63 / D-77) を超えて凍結されているため、**行の組み立てはこちらへ置く**。
上限のほうを緩めない。

``EspRow`` の宣言順が ``esp_diagnostics.csv`` の列順の単一の真実である。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

from rc_basics_lab.diagnostics.base import DiagnosticResult
from rc_basics_lab.experiment.diagnostics_rows import (
    DiagnosticScalarRow,
    condition_key,
    scalar_rows,
)


@dataclass(frozen=True, slots=True)
class EspRow:
    """``esp_diagnostics.csv`` の1行。**宣言順が CSV の列順の単一の真実**。

    ``input_scale`` / ``n_units`` / ``density`` は ``Esp02Config.reservoir``
    (``ReservoirSweepConfig``) 由来で、セクション固有の YAML キーではない
    (F-02-1-004)。``washout`` は λ と自己相関に効く値であり、ESP の距離当て
    はめには ``ESP_DISTANCE_WASHOUT`` (=0) が使われる点に注意。
    """

    experiment: str
    replicate: int
    seed_reservoir: int
    seed_drive: int
    seed_probe: int
    rho: float
    leak_rate: float
    input_scale: float
    sigma_u: float
    input_amplitude: float
    input_drive_std: float
    n_units: int
    density: float
    n_steps: int
    washout: int
    window: int
    n_pairs: int
    d_initial: float
    d_tail: float
    converged: int
    decay_rate_per_step: float
    lyapunov_per_step: float
    lyapunov_per_time: float
    tau_1e: float
    tau_censored: float
    tau_integrated: float
    wall_time_s: float


ESP_CSV_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(EspRow))
"""``esp_diagnostics.csv`` の列順 (``EspRow`` の宣言順が単一の真実)。"""


def esp_diagnostic_rows(
    row: EspRow, results: Sequence[DiagnosticResult]
) -> tuple[DiagnosticScalarRow, ...]:
    """1条件ぶんの診断結果を長形式の行に畳む (D-118)。

    **主表 (``esp_diagnostics.csv``) の列は1つも動かさない。** 軸は 02 が
    実際に振っている4つ (``rho`` / ``leak_rate`` / ``sigma_u`` / ``n_units``)
    だけを入れる —— 振っていない軸を入れると、条件を1つに畳んだときに
    区別できない行が生まれる。

    Args:
        row: その条件の主表の行 (軸の値をここから取る)。
        results: その条件で走った診断の結果。

    Returns:
        長形式の行。
    """
    return scalar_rows(
        results,
        experiment=row.experiment,
        condition_id=condition_key(
            {
                "rho": row.rho,
                "leak_rate": row.leak_rate,
                "sigma_u": row.sigma_u,
                "n_units": row.n_units,
            }
        ),
        replicate=row.replicate,
    )


__all__ = ["ESP_CSV_COLUMNS", "EspRow", "esp_diagnostic_rows"]

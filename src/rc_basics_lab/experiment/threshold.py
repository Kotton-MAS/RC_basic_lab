"""ESP 判定の閾値感度 —— 既定値 (D-16) が結論を作っていないことの実測.

D-16 は ``abs_tol=1e-6`` / ``window=200`` を既定に選んでいるが、この2つは
**判定基準そのもの**なので、値によって 2-C の結論 (「入力を強くすると ESP が
成立する rho の上限が上がる」) が動くなら、記事の主張は現象ではなく閾値の
選び方を語っていることになる。ここでは ``abs_tol`` 3点 x ``window`` 3点の
9通りで **sigma_u 別の臨界 rho** を測り直し、境界がどれだけ動くかを CSV に
落とす (``docs/design.md`` §9 の感度表の一次資料)。

``esp_convergence`` は ``(states, companions, cfg)`` の純関数なので、判定基準を
変えるだけなら**軌道を作り直す必要が無い**。1条件につき ``simulate_condition``
を1回だけ呼び、9通りの ``cfg`` で判定だけをやり直す (素直に9回掃引すると
実行時間も9倍になる)。

臨界 rho の定義は「rho の昇順で、収束率がレプリケートの過半数を割る最初の
rho」。格子内に境界が無い (全 rho で収束する) 場合は ``nan`` を返す ——
「境界が格子の外にある」を格子上端と混同させないため。
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields

from rc_basics_lab.config import Esp02Config
from rc_basics_lab.diagnostics.base import DiagnosticContext
from rc_basics_lab.diagnostics.esp import esp_convergence
from rc_basics_lab.experiment.esp import ESP_DISTANCE_WASHOUT, simulate_condition

logger = logging.getLogger(__name__)

ABS_TOL_GRID: tuple[float, ...] = (1.0e-4, 1.0e-6, 1.0e-8)
"""感度を見る絶対閾値。既定 (1e-6) を中心に上下2桁ずつ振る。"""

WINDOW_GRID: tuple[int, ...] = (100, 200, 400)
"""感度を見る判定窓 [ステップ]。既定 (200) を中心に半分と倍。"""

REFERENCE_ABS_TOL = 1.0e-6
REFERENCE_WINDOW = 200
"""比較の基準にする既定値 (D-16)。``max_abs_shift`` はここからのずれ。"""

MAJORITY = 0.5
"""臨界 rho の判定に使う収束率のしきい (これを下回った最初の rho が境界)。"""

CRITICAL_RHO_PREFIX = "critical_rho_sigma_"
"""CSV で sigma_u 別の臨界 rho を並べる列の接頭辞。"""


@dataclass(frozen=True, slots=True)
class ThresholdRow:
    """閾値感度 CSV の1行 (= 1つの ``(abs_tol, window)`` の組)。

    ``critical_rho_by_sigma`` 以外の**宣言順が CSV の列順**であり、末尾に
    ``critical_rho_by_sigma`` を sigma_u ごとの列
    (``critical_rho_sigma_<値>``) へ展開する。sigma_u 格子は設定値なので
    列名を dataclass のフィールドとして固定できず、ここだけ「宣言順が単一の
    真実」を1フィールドの展開という形で緩めている
    (展開規則は ``threshold_csv_columns`` / ``threshold_row_as_dict`` の対で
    1か所に閉じてあり、``test_threshold_csv_header_matches_rows`` が固定する)。

    Attributes:
        abs_tol: 判定の絶対閾値。
        window: 末尾距離を測る窓幅 [ステップ]。
        n_conditions: 判定した 2-C の条件数 (rho x sigma_u x レプリケート)。
        n_converged: そのうち ESP 成立と判定された条件数。
        converged_fraction: ``n_converged / n_conditions``。
        n_sigma_with_boundary: 臨界 rho が格子内に見つかった sigma_u の数。
        n_sigma_shifted: 基準 (D-16 の既定値) と臨界 rho が違う sigma_u の数。
        max_abs_shift: 基準からの臨界 rho のずれの最大値 (両方有限な
            sigma_u のみで取る。1つも無ければ 0.0)。
        critical_rho_by_sigma: ``(sigma_u, 臨界 rho)`` を sigma_u の昇順で。
    """

    abs_tol: float
    window: int
    n_conditions: int
    n_converged: int
    converged_fraction: float
    n_sigma_with_boundary: int
    n_sigma_shifted: int
    max_abs_shift: float
    critical_rho_by_sigma: tuple[tuple[float, float], ...]


_EXPANDED_FIELD = "critical_rho_by_sigma"

THRESHOLD_SCALAR_COLUMNS: tuple[str, ...] = tuple(
    f.name for f in fields(ThresholdRow) if f.name != _EXPANDED_FIELD
)
"""展開しない列 (``ThresholdRow`` の宣言順)。"""


def sigma_column(sigma: float) -> str:
    """sigma_u 別の臨界 rho の列名。"""
    return f"{CRITICAL_RHO_PREFIX}{sigma:g}"


def threshold_csv_columns(sigma_grid: Sequence[float]) -> tuple[str, ...]:
    """``esp_threshold_sensitivity.csv`` の列順。"""
    return THRESHOLD_SCALAR_COLUMNS + tuple(sigma_column(sigma) for sigma in sigma_grid)


def threshold_row_as_dict(row: ThresholdRow) -> dict[str, object]:
    """1行を CSV の列名 -> 値の dict にする (列順は上の関数と同じ規則)。"""
    values: dict[str, object] = {
        name: getattr(row, name) for name in THRESHOLD_SCALAR_COLUMNS
    }
    for sigma, critical in row.critical_rho_by_sigma:
        values[sigma_column(sigma)] = critical
    return values


def critical_rho(converged_fraction_by_rho: Mapping[float, float]) -> float:
    """収束率が過半数を割る最初の rho (= ESP 成立境界)。無ければ ``nan``。

    非単調な収束率でも「最初に割った点」を境界と呼ぶ。実測では 2-C の収束率は
    rho に対してきれいな階段になるが、定義を最小値に固定しておかないと格子を
    変えたときに境界の意味が変わる。
    """
    for rho in sorted(converged_fraction_by_rho):
        if converged_fraction_by_rho[rho] < MAJORITY:
            return rho
    return math.nan


def _shift(case: float, reference: float) -> tuple[bool, float]:
    """基準からのずれ。``(ずれたか, |差|)``。片方だけ nan ならずれた扱い。"""
    if math.isnan(case) and math.isnan(reference):
        return False, 0.0
    if math.isnan(case) or math.isnan(reference):
        return True, math.nan
    return case != reference, abs(case - reference)


def run_threshold_sweep(
    config: Esp02Config,
    *,
    abs_tol_grid: Sequence[float] = ABS_TOL_GRID,
    window_grid: Sequence[int] = WINDOW_GRID,
) -> tuple[ThresholdRow, ...]:
    """2-C の格子を1回だけ回し、``abs_tol`` x ``window`` の全組で判定し直す。

    Args:
        config: 02 の実験設定 (2-C の格子 ``config.esp_map`` を使う)。
        abs_tol_grid: 絶対閾値の候補。
        window_grid: 判定窓の候補。

    Returns:
        ``(abs_tol, window)`` の組ごとの1行。行数は
        ``len(abs_tol_grid) * len(window_grid)``。

    Raises:
        ValueError: 格子が空、または基準の組
            (``REFERENCE_ABS_TOL`` / ``REFERENCE_WINDOW``) が格子に無い場合。
    """
    if not abs_tol_grid or not window_grid:
        raise ValueError("abs_tol_grid / window_grid は1点以上必要です")
    if REFERENCE_ABS_TOL not in abs_tol_grid or REFERENCE_WINDOW not in window_grid:
        raise ValueError(
            "基準となる既定値 (D-16) が格子に含まれていません: "
            f"abs_tol={REFERENCE_ABS_TOL} window={REFERENCE_WINDOW} / "
            f"格子={tuple(abs_tol_grid)} x {tuple(window_grid)}"
        )
    section = config.esp_map
    cases = tuple(
        (abs_tol, window) for abs_tol in abs_tol_grid for window in window_grid
    )

    # (abs_tol, window) x sigma_u x rho -> レプリケートごとの converged
    hits: dict[tuple[float, int], dict[float, dict[float, list[int]]]] = {
        case: {
            sigma: {rho: [] for rho in section.rho_grid} for sigma in section.sigma_grid
        }
        for case in cases
    }
    for replicate in range(config.reservoir.n_replicates):
        for rho in section.rho_grid:
            for sigma_u in section.sigma_grid:
                trajectories = simulate_condition(
                    config,
                    rho=rho,
                    leak_rate=section.leak_rate,
                    sigma_u=sigma_u,
                    replicate=replicate,
                )
                ctx = DiagnosticContext(
                    washout=ESP_DISTANCE_WASHOUT,
                    companion_states=trajectories.companions,
                )
                for abs_tol, window in cases:
                    cfg = dataclasses.replace(
                        config.esp, abs_tol=abs_tol, window=window
                    )
                    result = esp_convergence(trajectories.states, ctx=ctx, cfg=cfg)
                    hits[(abs_tol, window)][sigma_u][rho].append(
                        int(result.scalars["converged"])
                    )

    reference = _critical_by_sigma(hits[(REFERENCE_ABS_TOL, REFERENCE_WINDOW)])
    rows = tuple(_build_row(case, hits[case], reference) for case in cases)
    logger.info(
        "閾値感度: %d 通り (abs_tol %d x window %d) / 基準からずれた "
        "(sigma_u, 臨界 rho) は %d 件",
        len(rows),
        len(abs_tol_grid),
        len(window_grid),
        sum(row.n_sigma_shifted for row in rows),
    )
    return rows


def _critical_by_sigma(
    verdicts: Mapping[float, Mapping[float, Sequence[int]]],
) -> dict[float, float]:
    """sigma_u ごとの臨界 rho。"""
    return {
        sigma_u: critical_rho(
            {rho: sum(flags) / len(flags) for rho, flags in by_rho.items()}
        )
        for sigma_u, by_rho in verdicts.items()
    }


def _build_row(
    case: tuple[float, int],
    verdicts: Mapping[float, Mapping[float, Sequence[int]]],
    reference: Mapping[float, float],
) -> ThresholdRow:
    abs_tol, window = case
    critical = _critical_by_sigma(verdicts)
    flags = [
        flag
        for by_rho in verdicts.values()
        for cell in by_rho.values()
        for flag in cell
    ]
    shifts = [_shift(critical[sigma], reference[sigma]) for sigma in sorted(critical)]
    finite = [size for shifted, size in shifts if not math.isnan(size)]
    return ThresholdRow(
        abs_tol=abs_tol,
        window=window,
        n_conditions=len(flags),
        n_converged=sum(flags),
        converged_fraction=sum(flags) / len(flags) if flags else 0.0,
        n_sigma_with_boundary=sum(
            1 for value in critical.values() if not math.isnan(value)
        ),
        n_sigma_shifted=sum(1 for shifted, _ in shifts if shifted),
        max_abs_shift=max(finite) if finite else 0.0,
        critical_rho_by_sigma=tuple(
            (sigma, critical[sigma]) for sigma in sorted(critical)
        ),
    )


__all__ = [
    "ABS_TOL_GRID",
    "CRITICAL_RHO_PREFIX",
    "REFERENCE_ABS_TOL",
    "REFERENCE_WINDOW",
    "THRESHOLD_SCALAR_COLUMNS",
    "WINDOW_GRID",
    "ThresholdRow",
    "critical_rho",
    "run_threshold_sweep",
    "sigma_column",
    "threshold_csv_columns",
    "threshold_row_as_dict",
]

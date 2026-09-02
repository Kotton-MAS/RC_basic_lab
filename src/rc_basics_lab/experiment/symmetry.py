"""実験 3-S: 駆動入力の対称性と IPC の偶数次 (D-116).

記事03 §2.1 は「次数2と4のセルがほぼ空」と述べるが、その理由は**未検証**の
まま書けなかった。仮説は「駆動入力がゼロ対称 (一様・平均0) で tanh が奇関数
なので、状態が入力の奇関数になり、偶数次の目標と相関しない」である。

ここは仮説を**行の値で**判定できる形にする配線だけを持つ。振るのは
**駆動入力の平均のずれだけ**で、分布の形 (一様) も標準偏差 (``sigma_u``) も
変えない:

- 一様のままなので ``orthonormal_basis`` が実測の平均・標準偏差で標準化した
  あとも Legendre 基底は厳密に正規直交であり、D-28 を満たしたまま測れる。
  **分布の形を歪める設計は採らない** —— 目標同士が直交しなくなって容量が
  二重計上され、「偶数次が増えた」のか「二重計上が増えた」のかを分離できない
- 平均をずらすとリザバーの動作点が 0 から外れる。tanh が奇関数なのは 0 の
  まわりだけなので、対称性の破れはここに入る

``n_units`` / ``n_steps`` が本番の 3-B と同規模で、``figures-03`` の予算 (900 秒)
の外で ``make symmetry-03`` として手動実行する (``length_sweep`` と同じ扱い)。

**この実験は仮説を棄却しうる。** 平均をずらしても偶数次が現れなければ、
理由は対称性ではなく基底の構成か打ち切りの側にある。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, fields

import numpy as np

from rc_basics_lab.config import Capacity03Config
from rc_basics_lab.diagnostics.ipc import ipc
from rc_basics_lab.experiment.capacity import (
    EXPERIMENT_SYMMETRY,
    capacity_context,
    ipc_config_for,
    simulate_condition_trajectory,
)
from rc_basics_lab.experiment.capacity_rows import CapacityCondition

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SymmetryRow:
    """``capacity_symmetry.csv`` の1行 (**長形式**: 1 (条件, 次数) = 1 行)。

    宣言順が CSV の列順。次数を**行の値**に落とすので、偶数次の容量の有無は
    図を見ずに CSV から判定できる (T3 の受け入れ条件)。

    Attributes:
        experiment: 常に ``EXPERIMENT_SYMMETRY``。
        replicate: レプリケート番号 (0 始まり)。
        offset_ratio: 駆動入力に加えた定数 / ``sigma_u`` (0.0 がゼロ対称)。
        drive_mean: 駆動入力の**実測**平均。設定値どおりに効いたかを行に残す。
        drive_std: 駆動入力の**実測**標準偏差。オフセットで変わらないこと
            (振っているのは平均だけ) がこの列で確かめられる。
        rho: スペクトル半径。
        leak_rate: リーク率。
        n_units: リザバーのユニット数 N。
        sigma_u: 駆動信号の標準偏差の設定値 (D-17)。
        n_steps: 系列長 [ステップ]。
        degree: 次数 (1 始まり)。
        capacity: その次数のしきい値後の容量。
        capacity_raw: しきい値前の容量。
        ipc_total: その条件の全次数合計 (行をまたいで同じ値)。
        wall_time_s: その条件の測定時間 [秒] (行をまたいで同じ値)。
    """

    experiment: str
    replicate: int
    offset_ratio: float
    drive_mean: float
    drive_std: float
    rho: float
    leak_rate: float
    n_units: int
    sigma_u: float
    n_steps: int
    degree: int
    capacity: float
    capacity_raw: float
    ipc_total: float
    wall_time_s: float


SYMMETRY_CSV_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(SymmetryRow))
"""``capacity_symmetry.csv`` の列順 (``SymmetryRow`` の宣言順が単一の真実)。"""


def _condition(config: Capacity03Config, replicate: int) -> CapacityCondition:
    """掃引の1条件 (オフセット以外は全点で共通)。"""
    section = config.symmetry_sweep
    return CapacityCondition(
        experiment=EXPERIMENT_SYMMETRY,
        rho=section.rho,
        leak_rate=section.leak_rate,
        n_units=section.n_units,
        state_noise=0.0,
        sigma_u=section.sigma_u,
        n_steps=section.n_steps,
        replicate=replicate,
    )


def run_symmetry_sweep(config: Capacity03Config) -> tuple[SymmetryRow, ...]:
    """駆動入力の平均を振って IPC の次数分解を測る (3-S)。

    ``ctx`` は全条件で1個を共有する (D-37 の共通乱数法)。しきい値の推定
    ノイズが条件間の差に独立に乗ると、偶数次の増減がノイズか実体かを
    分離できなくなる。

    Args:
        config: 03 の設定 (``symmetry_sweep`` セクションを読む)。

    Returns:
        ``(条件数 x 次数)`` 本の行。

    Raises:
        ValueError: 確保軸を超える設定、または診断側の値域違反。
    """
    section = config.symmetry_sweep
    ctx = capacity_context(config)
    rows: list[SymmetryRow] = []
    for replicate in range(section.n_replicates):
        condition = _condition(config, replicate)
        cfg_ipc = ipc_config_for(config, condition.experiment)
        for offset_ratio in section.offset_ratio_grid:
            started = time.perf_counter()
            trajectory = simulate_condition_trajectory(
                config, condition, drive_offset=offset_ratio * section.sigma_u
            )
            states = trajectory.states
            states.flags.writeable = False
            result = ipc(states, trajectory.drive, ctx=ctx, cfg=cfg_ipc)
            wall_time_s = time.perf_counter() - started

            by_degree = result.arrays["ipc_by_degree"]
            by_degree_raw = result.arrays["ipc_by_degree_raw"]
            drive = np.asarray(trajectory.drive, dtype=np.float64)
            for index, (capacity, capacity_raw) in enumerate(
                zip(by_degree, by_degree_raw, strict=True)
            ):
                rows.append(
                    SymmetryRow(
                        experiment=EXPERIMENT_SYMMETRY,
                        replicate=replicate,
                        offset_ratio=offset_ratio,
                        drive_mean=float(np.mean(drive)),
                        drive_std=float(np.std(drive)),
                        rho=section.rho,
                        leak_rate=section.leak_rate,
                        n_units=section.n_units,
                        sigma_u=section.sigma_u,
                        n_steps=section.n_steps,
                        degree=index + 1,
                        capacity=float(capacity),
                        capacity_raw=float(capacity_raw),
                        ipc_total=float(result.scalars["ipc_total"]),
                        wall_time_s=wall_time_s,
                    )
                )
            logger.info(
                "3-S replicate=%d offset=%.2f x sigma_u (実測平均=%.4f): "
                "次数別=%s 合計=%.3f (%.1fs)",
                replicate,
                offset_ratio,
                float(np.mean(drive)),
                [f"{float(value):.3f}" for value in by_degree],
                float(result.scalars["ipc_total"]),
                wall_time_s,
            )
    return tuple(rows)


def even_degree_share_at_offset(
    rows: tuple[SymmetryRow, ...], offset_ratio: float
) -> float:
    """あるオフセットでの「偶数次が全容量に占める割合」(レプリケート平均)。

    仮説の判定に使う要約。``rows`` から機械的に出るので、図の目視に依存しない。

    ``plotting.capacity_grids.even_degree_share`` と比を計算する式は同じだが、
    あちらは ``CapacityProfileRow`` 全体から図の注 (D-94) を作るもので、
    こちらは 3-S の行を**オフセットで絞ってから**測る。作図層は実験層を
    import してよいが逆は許されない (D-53) ので、共有はしない。

    Raises:
        ValueError: そのオフセットの行が無い場合。
    """
    selected = [row for row in rows if row.offset_ratio == offset_ratio]
    if not selected:
        raise ValueError(f"offset_ratio={offset_ratio} の行がありません")
    even = sum(row.capacity for row in selected if row.degree % 2 == 0)
    total = sum(row.capacity for row in selected)
    if total <= 0.0:
        raise ValueError(f"offset_ratio={offset_ratio} の総容量が 0 です")
    return even / total


__all__ = [
    "SYMMETRY_CSV_COLUMNS",
    "SymmetryRow",
    "even_degree_share_at_offset",
    "run_symmetry_sweep",
]

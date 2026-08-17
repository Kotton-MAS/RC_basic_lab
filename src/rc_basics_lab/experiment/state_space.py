"""実験1-B — 入力空間とリザバー状態空間の広がりを数値で比べる.

比較する空間は3つ:

1. ``raw_input``: 生の入力 ``u`` (本連載ではどちらの課題も 1 次元)
2. ``delay_embedded_input``: 遅延線ベースラインが実際に使う特徴空間
   ``[u[t], ..., u[t-k]]`` (``k = max(ridge.n_lags_grid)``)
3. ``reservoir_state``: ESN の状態 ``x[t]`` (N 次元)

行の範囲は実験と同じ分割窓 (``split.start`` 〜 ``split.test.stop``) に揃える。
washout も t0 も既にこの窓の外なので、``DiagnosticContext`` の washout は 0 でよい。

PCA そのものは ``diagnostics.state_space.state_pca`` (``reservoir`` に依存しない
移植可能な診断) を呼ぶだけで、ここは「どの行列を渡すか」の配線だけを持つ。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rc_basics_lab.config import ExperimentConfig
from rc_basics_lab.diagnostics.base import DiagnosticResult
from rc_basics_lab.diagnostics.state_space import state_pca
from rc_basics_lab.experiment.runner import (
    ReplicatePlan,
    TaskEntry,
    build_tasks,
    plan_replicate,
)
from rc_basics_lab.readout.design import DelayLineSpec, build_design_matrix
from rc_basics_lab.types import FloatArray

logger = logging.getLogger(__name__)

RAW_INPUT = "raw_input"
DELAY_EMBEDDED_INPUT = "delay_embedded_input"
RESERVOIR_STATE = "reservoir_state"


@dataclass(frozen=True, slots=True)
class SpaceSummary:
    """1つの特徴空間の PCA 要約。

    Attributes:
        space: 空間の名前 (``RAW_INPUT`` / ``DELAY_EMBEDDED_INPUT`` /
            ``RESERVOIR_STATE``)。
        n_features: 元の次元数 (PCA 前の列数)。
        n_components_95: 累積寄与率 95% に要する主成分数。
        participation_ratio: ``(Σλ)² / Σλ²``。
        cumulative_ratio: 累積寄与率曲線。
        pc_scores: 先頭2成分のスコア (1 次元空間では1列だけ)。
    """

    space: str
    n_features: int
    n_components_95: int
    participation_ratio: float
    cumulative_ratio: FloatArray
    pc_scores: FloatArray

    @classmethod
    def from_result(cls, space: str, result: DiagnosticResult) -> SpaceSummary:
        """``state_pca`` の結果から要約を作る。"""
        return cls(
            space=space,
            n_features=int(result.scalars["n_features"]),
            n_components_95=int(result.scalars["n_components_95"]),
            participation_ratio=result.scalars["participation_ratio"],
            cumulative_ratio=result.arrays["cumulative_ratio"],
            pc_scores=result.arrays["pc_scores"],
        )

    def to_summary(self) -> dict[str, object]:
        """``meta.json`` に載せるプレーンな dict (配列は含めない)。"""
        return {
            "space": self.space,
            "n_features": self.n_features,
            "n_components_95": self.n_components_95,
            "participation_ratio": self.participation_ratio,
        }


@dataclass(frozen=True, slots=True)
class StateSpaceReport:
    """1課題ぶんの空間比較。"""

    task: str
    replicate: int
    n_lags: int
    n_rows: int
    spaces: tuple[SpaceSummary, ...]

    def space(self, name: str) -> SpaceSummary:
        """名前で要約を引く。"""
        for summary in self.spaces:
            if summary.space == name:
                return summary
        raise KeyError(f"未知の空間名です: {name}")

    def to_summary(self) -> dict[str, object]:
        """``meta.json`` に載せるプレーンな dict。"""
        return {
            "task": self.task,
            "replicate": self.replicate,
            "n_lags": self.n_lags,
            "n_rows": self.n_rows,
            "spaces": [summary.to_summary() for summary in self.spaces],
        }


def analyze_task(
    config: ExperimentConfig,
    task_entry: TaskEntry,
    replicate: int = 0,
    *,
    plan: ReplicatePlan | None = None,
) -> StateSpaceReport:
    """1課題について3つの空間の PCA を取る。

    Args:
        plan: 呼び出し側が既に持っている ``ReplicatePlan`` (``replicate`` に
            対応するもの) を渡すと ``plan_replicate`` の呼び直しを省く
            (F-1-009)。省略時はこれまでどおり内部で作る。
    """
    resolved_plan = (
        plan if plan is not None else plan_replicate(config, task_entry, replicate)
    )
    start, stop = resolved_plan.split.start, resolved_plan.split.test.stop
    n_lags = max(config.ridge.n_lags_grid)
    embedded = build_design_matrix(
        DelayLineSpec(n_lags=n_lags, bias=False), resolved_plan.task.u
    ).phi
    matrices: tuple[tuple[str, FloatArray], ...] = (
        (RAW_INPUT, resolved_plan.task.u[start:stop]),
        (DELAY_EMBEDDED_INPUT, embedded[start:stop]),
        (RESERVOIR_STATE, resolved_plan.states[start:stop]),
    )
    spaces = tuple(
        SpaceSummary.from_result(name, state_pca(matrix)) for name, matrix in matrices
    )
    report = StateSpaceReport(
        task=task_entry.name,
        replicate=replicate,
        n_lags=n_lags,
        n_rows=stop - start,
        spaces=spaces,
    )
    for summary in spaces:
        logger.info(
            "task=%s space=%s n_features=%d n_components_95=%d "
            "participation_ratio=%.2f",
            report.task,
            summary.space,
            summary.n_features,
            summary.n_components_95,
            summary.participation_ratio,
        )
    return report


def collect_state_space(
    config: ExperimentConfig,
    replicate: int = 0,
    *,
    plans: Mapping[str, ReplicatePlan] | None = None,
) -> tuple[StateSpaceReport, ...]:
    """全課題について空間比較を取る (図と meta.json の材料)。

    Args:
        plans: タスク名 -> ``replicate`` に対応する ``ReplicatePlan``。渡すと
            ``run_experiment`` 側で作った plan を再利用し、``plan_replicate``
            の呼び直しを省く (F-1-009)。
    """
    return tuple(
        analyze_task(
            config,
            entry,
            replicate,
            plan=None if plans is None else plans.get(entry.name),
        )
        for entry in build_tasks(config)
    )


def summarize(reports: Sequence[StateSpaceReport]) -> list[dict[str, object]]:
    """``meta.json`` の ``state_space`` フィールドに入れる形にする。"""
    return [report.to_summary() for report in reports]


__all__ = [
    "DELAY_EMBEDDED_INPUT",
    "RAW_INPUT",
    "RESERVOIR_STATE",
    "SpaceSummary",
    "StateSpaceReport",
    "analyze_task",
    "collect_state_space",
    "summarize",
]

"""教師強制で read-out を学習する層 (D-128).

``freerun.py`` から切り出した (1モジュール 600 行の上限、D-63 / D-77)。

自走 (closed loop) を始める前に、**教師強制 (open loop) で1ステップ先を
学習する** (D-44)。ここは「学習まで」で、切り替えと自走は ``freerun.py`` にある。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rc_basics_lab.config import Chaos04Config
from rc_basics_lab.experiment.capacity_bounds import (
    validate_n_units_bound,
)
from rc_basics_lab.experiment.freerun_tasks import (
    standardize_steps_for,
    validate_standardization_window,
)
from rc_basics_lab.experiment.runner import (
    ESN_METHOD,
    ReplicatePlan,
    TaskEntry,
    build_methods,
    plan_replicate,
)
from rc_basics_lab.readout.design import (
    DesignMatrix,
    FeatureSpec,
    ReservoirSpec,
)
from rc_basics_lab.readout.ridge import fit_ridge, select_alpha
from rc_basics_lab.types import FloatArray

FREE_RUN_SPEC = ReservoirSpec()
"""自走に使う特徴仕様 ``[1, u[t], x[t]]``。

多項式読み出しは v0.1 では入れない (仕様 §3.2)。

01 の ``build_methods`` が ESN 手法に与える候補と**同一の値**であることを
``test_free_run_spec_matches_the_one_step_esn_candidate`` が固定する。ここが
ずれると「教師強制と自走で別の特徴を使う」(仕様 §5 禁止する構造2) になる。
"""


@dataclass(frozen=True, slots=True)
class TeacherForcedReadout:
    """教師強制で学習した read-out (自走はこれをそのまま使う、D-44)。

    Attributes:
        plan: 01 の ``plan_replicate`` が作った課題・状態・設計行列・分割。
        design: 選ばれた候補の設計行列。
        alpha: 検証分割で選ばれた正則化係数 (D-04 の格子から)。
        coefficients: リッジ解 ``(F, D_out)``。**このオブジェクトを自走へ渡す**。
        val_nrmse: 選択時の検証 NRMSE。
        method: 手法名 (``LINEAR`` / ``DELAY_LINE`` / ``ESN_METHOD``)。
            対照 (線形・遅延線) も自走させるのは、受け入れ条件3 の後半
            「自走では対照が成立しない」を**数値で**示すためである。
            「原理的に不利」を主張だけで済ませると、読者は「回してみたら
            動いたかもしれない」を否定できない。
        spec: 学習に使った特徴仕様 (遅延線は選ばれた ``n_lags`` を持つ)。
            **閉ループで使う仕様とは表現が違うことがある** ——
            ``closed_loop_setup`` を参照。
    """

    plan: ReplicatePlan
    design: DesignMatrix
    alpha: float
    coefficients: FloatArray
    val_nrmse: float
    method: str = ESN_METHOD
    spec: FeatureSpec = FREE_RUN_SPEC


def _rows(array: FloatArray, selection: range) -> FloatArray:
    block: FloatArray = array[selection.start : selection.stop]
    return block


def method_candidates(config: Chaos04Config, method: str) -> tuple[FeatureSpec, ...]:
    """手法名 -> 候補の特徴仕様。

    **手法の列挙は 01 の ``build_methods`` が単一の真実**である。

    04 側で ``PassthroughSpec()`` / ``DelayLineSpec(...)`` を書き写さないのは、
    01 が候補を1本足したときに 04 の自走だけ古い候補を使い続ける事故を防ぐため
    である (``plan.designs[method]`` の並びとここが必ず一致する)。

    Raises:
        ValueError: 04 が自走させない手法名の場合。
    """
    for item in build_methods(config.base):
        if item.name == method:
            return item.candidates
    raise ValueError(f"01 の手法ではありません: {method!r}")


def fit_teacher_forced(
    config: Chaos04Config,
    task_entry: TaskEntry,
    replicate: int,
    *,
    method: str = ESN_METHOD,
    plan: ReplicatePlan | None = None,
) -> TeacherForcedReadout:
    """教師強制で読み出しを学習する (01 の ``_evaluate`` と同じ2段)。

    ``select_alpha`` (検証分割で候補と alpha を選ぶ) -> ``fit_ridge`` (訓練区間で
    係数を解く) の順序も、渡す行集合 (``plan.split``) も、同点のときに先に
    評価した候補を残す規律も 01 と同一である。**同じ (候補, alpha) が選ばれる
    こと**は ``test_free_run_readout_matches_the_one_step_selection`` が 4-A の
    行と突き合わせて実測する (経路が2本あることを主張だけで済ませない)。

    Args:
        config: 04 の設定。
        task_entry: 課題 (ESN 設定を含む)。
        replicate: レプリケート番号。
        method: 学習する手法 (既定は ESN)。対照も同じ経路で学習する。
        plan: すでに作ってある ``ReplicatePlan``。``None`` なら作る。
            **同じ (課題, レプリケート) で3手法を回すときは1個を共有する** ——
            01 の ``run_task(..., plan0=...)`` (3-C) と同じ形で、状態行列が
            1本しか存在しないことを構造で保証する。

    Raises:
        ValueError: 標準化の推定区間が訓練区間を越える場合 (D-41)、ESN 手法の
            候補が1本でない場合 (D-08)、または課題・分割側の値域違反。
    """
    base = config.base
    validate_n_units_bound(task_entry.reservoir.n_units)
    if plan is None:
        plan = plan_replicate(base, task_entry, replicate)
    validate_standardization_window(
        standardize_steps_for(config, task_entry.name), plan.split
    )
    specs = method_candidates(config, method)
    designs = plan.designs[method]
    if method == ESN_METHOD and len(designs) != 1:
        raise ValueError(f"ESN 手法の候補は1本のはずです (D-08): {len(designs)} 本")
    if len(designs) != len(specs):
        raise ValueError(
            f"候補数が一致しません: designs={len(designs)} specs={len(specs)}"
        )
    split = plan.split
    y_train = _rows(plan.task.y, split.train)
    y_val = _rows(plan.task.y, split.val)
    best_index = -1
    best_alpha = math.nan
    best_val_nrmse = math.inf
    for index, design in enumerate(designs):
        selection = select_alpha(
            _rows(design.phi, split.train),
            y_train,
            _rows(design.phi, split.val),
            y_val,
            base.ridge.alpha_grid,
            bias_column=design.bias_column,
        )
        if selection.val_nrmse < best_val_nrmse:
            best_index = index
            best_alpha = selection.alpha
            best_val_nrmse = selection.val_nrmse
    if best_index < 0:  # pragma: no cover - 候補が空なら build_methods が落ちる
        raise ValueError(f"候補の選択に失敗しました: {method!r}")
    design = designs[best_index]
    coefficients = fit_ridge(
        _rows(design.phi, split.train),
        y_train,
        best_alpha,
        bias_column=design.bias_column,
    )
    return TeacherForcedReadout(
        plan=plan,
        design=design,
        alpha=best_alpha,
        coefficients=coefficients,
        val_nrmse=best_val_nrmse,
        method=method,
        spec=specs[best_index],
    )


__all__ = [
    "FREE_RUN_SPEC",
    "TeacherForcedReadout",
    "fit_teacher_forced",
    "method_candidates",
]

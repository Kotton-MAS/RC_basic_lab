"""alpha 格子の下端に関する実測の固定 (D-11).

Mackey-Glass の検証 NRMSE は alpha に対して**単調増加**であり、内点解が存在しない
(最適は alpha -> 0、すなわち OLS 側にある)。したがって「格子の端が選ばれている =
探索予算が足りない」という一般則をこの実験に機械的に当てはめると誤診になる。
格子下端が選ばれるのは探索の失敗ではなく、期待される挙動である。

``experiments/01_what_is_rc/config.yaml`` の下端 ``1.0e-10`` は数値条件数による
限界であって、恣意的な打ち切りではない。実測では ``alpha <= 1.0e-11`` で
``fit_ridge`` の ``scipy.linalg.solve(assume_a="pos")`` が特異行列として落ちるが、
**その失敗の出方は BLAS/LAPACK 実装に依存する**ため、テストでは主張しない。
代わりに、下端付近で Gram 行列の条件数が倍精度の限界域に入っていることを測る
(環境差で偽陽性を出さず、かつ「下端は数値限界」の裏付けになる形)。
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from rc_basics_lab.config import MackeyGlassConfig, load_config
from rc_basics_lab.readout.design import DelayLineSpec, build_design_matrix
from rc_basics_lab.readout.ridge import select_alpha
from rc_basics_lab.tasks.mackey_glass import generate_mackey_glass
from rc_basics_lab.types import FloatArray

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG_PATH = REPO_ROOT / "experiments" / "01_what_is_rc" / "config.yaml"

GRID_LOWER_BOUND = 1.0e-10
"""``config.yaml`` の alpha 格子下端 (数値限界。これ以上は下げられない)。"""

ILL_CONDITIONED = 1.0e12
"""この条件数を超えると倍精度 (eps ~ 2.2e-16) では有効桁が残らない領域に入る。"""

SHORT = MackeyGlassConfig(length=1500)
"""縮小設定。本番 (length=8200) と同じ生成器・同じ遅延線特徴で単調性だけを見る。"""

TASK_SEED = 1
WASHOUT = 100
TRAIN_FRACTION = 0.6


def _experiment_alpha_grid() -> tuple[float, ...]:
    """本番の実験設定から alpha 格子をそのまま読む (D-04 の単一キー)。"""
    return load_config(EXPERIMENT_CONFIG_PATH).ridge.alpha_grid


def _experiment_max_n_lags() -> int:
    """遅延線が実際に使う最大ラグ数 (条件数が最も悪くなる候補)。"""
    return max(load_config(EXPERIMENT_CONFIG_PATH).ridge.n_lags_grid)


def _train_val_blocks(
    n_lags: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """MG の遅延線設計行列を train / val に連続分割して返す。"""
    data = generate_mackey_glass(SHORT, np.random.default_rng(TASK_SEED))
    design = build_design_matrix(DelayLineSpec(n_lags=n_lags), data.u)
    start = max(design.first_valid, WASHOUT)
    n_train = int((SHORT.length - start) * TRAIN_FRACTION)
    train = slice(start, start + n_train)
    val = slice(start + n_train, SHORT.length)
    return design.phi[train], data.y[train], design.phi[val], data.y[val]


def test_mackey_glass_validation_nrmse_is_monotone_in_alpha() -> None:
    """MG の検証 NRMSE は alpha に対して単調増加で、最良は格子下端 (D-11).

    内点解が現れた瞬間 (= 「格子端が選ばれている」を探索の失敗と読んでよく
    なった瞬間) にこのテストが落ちる。
    """
    grid = _experiment_alpha_grid()
    phi_tr, y_tr, phi_val, y_val = _train_val_blocks(_experiment_max_n_lags())
    selection = select_alpha(phi_tr, y_tr, phi_val, y_val, grid)

    assert tuple(alpha for alpha, _ in selection.curve) == tuple(sorted(grid))
    scores = [score for _, score in selection.curve]
    assert all(later > earlier for earlier, later in pairwise(scores))
    # 下端が最良 = 内点解が無い。選択が格子の端に張り付くのは期待される挙動
    assert selection.alpha == pytest.approx(min(grid))
    assert selection.val_nrmse == pytest.approx(scores[0])
    # 初版の下端 1e-8 まででは、この単調性のぶんだけ悪い解を選んでいた
    old_lower_bound_score = dict(selection.curve)[1.0e-8]
    assert scores[0] < old_lower_bound_score


def test_alpha_grid_lower_bound_is_a_numerical_limit() -> None:
    """下端 1e-10 は恣意的な打ち切りではなく、条件数の限界である (D-11 の根拠).

    ``alpha <= 1e-11`` で Cholesky が落ちること自体は環境依存なので主張しない。
    ここでは「下端付近の正則化量が Gram 行列の数値誤差と同じ桁にある」ことを
    条件数で測る。
    """
    grid = _experiment_alpha_grid()
    assert min(grid) == pytest.approx(GRID_LOWER_BOUND)

    phi_tr, _, _, _ = _train_val_blocks(_experiment_max_n_lags())
    gram: FloatArray = phi_tr.T @ phi_tr
    condition_number = float(np.linalg.cond(gram))
    assert condition_number > ILL_CONDITIONED
    # 下端の alpha は Gram 行列の対角スケールに対して無視できる大きさ
    # (= これ以上下げても OLS に漸近するだけで、数値的にしか変わらない)
    assert min(grid) < float(np.max(np.diag(gram))) * np.finfo(np.float64).eps

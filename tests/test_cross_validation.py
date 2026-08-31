"""時系列の交差検証の検査.

**測るのは「漏れが無いこと」が第一である。** 交差検証は「たくさん試したから
良い値が選べた」ように見えるので、漏れていても数字は良くなる。良くなる方向の
壊れ方は気づけないので、構造の側を固定する。
"""

from __future__ import annotations

import numpy as np
import pytest

from rc_basics_lab.readout.cross_validation import (
    Fold,
    FoldScheme,
    folds_never_look_ahead,
    make_folds,
    select_alpha_cv,
)
from rc_basics_lab.readout.ridge import select_alpha
from rc_basics_lab.types import FloatArray

SPAN = range(0, 1000)


# --- 漏れの検査 --------------------------------------------------------------


@pytest.mark.parametrize("n_folds", [2, 3, 4, 8])
@pytest.mark.parametrize("embargo", [0, 10, 50])
def test_rolling_folds_never_train_on_the_future(n_folds: int, embargo: int) -> None:
    """``rolling`` は検証区間より後ろの行で学習しない。

    **これがこの折り方の存在理由**である。破れたら、予測しようとしている時刻
    より後の情報が係数に入る。
    """
    folds = make_folds(SPAN, n_folds, scheme=FoldScheme.ROLLING, embargo=embargo)
    for fold in folds:
        assert fold.train.stop <= fold.val.start, (
            f"訓練区間が検証区間に食い込んでいます: {fold}"
        )


@pytest.mark.parametrize("scheme", list(FoldScheme))
@pytest.mark.parametrize("embargo", [0, 25, 100])
def test_the_embargo_gap_is_respected(scheme: FoldScheme, embargo: int) -> None:
    """訓練区間と検証区間のあいだに ``embargo`` 行の隙間がある。

    設計行列の1行は過去 ``first_valid`` 行の入力を含むので、隙間が足りないと
    **検証行が訓練行と同じ入力を持つ** (遅延線で顕著)。
    """
    folds = make_folds(SPAN, 4, scheme=scheme, embargo=embargo)
    assert folds_never_look_ahead(folds, embargo), (
        f"{scheme} で禁足区間が守られていません: {folds}"
    )


def test_the_folds_never_touch_rows_outside_the_span() -> None:
    """折りが与えた範囲の外へ出ない (テスト区間を触らないことの実体)。"""
    for scheme in FoldScheme:
        for fold in make_folds(SPAN, 4, scheme=scheme, embargo=10):
            assert SPAN.start <= fold.train.start and fold.train.stop <= SPAN.stop
            assert SPAN.start <= fold.val.start and fold.val.stop <= SPAN.stop


def test_the_validation_blocks_do_not_overlap_each_other() -> None:
    """検証ブロックが重ならない (同じ行を2回採点しない)。"""
    for scheme in FoldScheme:
        seen: set[int] = set()
        for fold in make_folds(SPAN, 4, scheme=scheme, embargo=10):
            rows = set(fold.val)
            assert not (rows & seen), f"{scheme} で検証区間が重なっています"
            seen |= rows


# --- 設定の検査 --------------------------------------------------------------


@pytest.mark.parametrize("n_folds", [-1, 0, 1])
def test_too_few_folds_are_rejected(n_folds: int) -> None:
    """折りが2つ未満なら落ちる (交差になっていない)。"""
    with pytest.raises(ValueError, match="n_folds"):
        make_folds(SPAN, n_folds, scheme=FoldScheme.ROLLING, embargo=0)


def test_a_negative_embargo_is_rejected() -> None:
    """禁足区間が負なら落ちる (訓練と検証が重なる)。"""
    with pytest.raises(ValueError, match="embargo"):
        make_folds(SPAN, 3, scheme=FoldScheme.ROLLING, embargo=-1)


def test_an_embargo_that_eats_the_block_is_rejected() -> None:
    """禁足区間がブロックより大きければ落ちる。

    黙って折りを減らすと、**指定したより少ない折りで選んだ**ことに気づけない。
    """
    with pytest.raises(ValueError, match="折りが作れません"):
        make_folds(range(0, 100), 4, scheme=FoldScheme.ROLLING, embargo=50)


# --- 選択の検査 --------------------------------------------------------------


def _linear_problem(
    n_steps: int = 600, noise: float = 0.05
) -> tuple[FloatArray, FloatArray]:
    """``y = 2 x + 1`` にノイズを乗せた素直な問題 (バイアス列つき)。"""
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, (n_steps, 1))
    phi: FloatArray = np.hstack([np.ones((n_steps, 1)), x])
    y: FloatArray = 2.0 * x + 1.0 + noise * rng.standard_normal((n_steps, 1))
    return phi, y


def test_the_selection_returns_an_alpha_from_the_grid() -> None:
    """選ばれる alpha は格子の中の値である。"""
    phi, y = _linear_problem()
    folds = make_folds(range(0, 600), 4, scheme=FoldScheme.ROLLING, embargo=5)
    grid = [1.0e-8, 1.0e-4, 1.0, 100.0]
    selection = select_alpha_cv(phi, y, folds, grid, bias_column=0)
    assert selection.alpha in grid
    assert len(selection.curve) == len(grid)


def test_the_curve_is_the_mean_over_folds() -> None:
    """``curve`` の各点が折りの平均である (単一分割の値をそのまま返さない)。"""
    phi, y = _linear_problem()
    folds = make_folds(range(0, 600), 3, scheme=FoldScheme.ROLLING, embargo=5)
    grid = [1.0e-6, 1.0]
    selection = select_alpha_cv(phi, y, folds, grid, bias_column=0)
    for alpha, mean_score in selection.curve:
        per_fold = [
            select_alpha(
                phi[f.train.start : f.train.stop],
                y[f.train.start : f.train.stop],
                phi[f.val.start : f.val.stop],
                y[f.val.start : f.val.stop],
                [alpha],
                bias_column=0,
            ).val_nrmse
            for f in folds
        ]
        assert mean_score == pytest.approx(sum(per_fold) / len(per_fold))


def test_a_single_fold_matches_the_plain_selection() -> None:
    """折りが1つなら単一分割の選択と一致する (**交差検証の側が別物でない**)。

    ここがずれると、交差検証を有効にした瞬間に「折り方以外の何か」も変わって
    いることになる。
    """
    phi, y = _linear_problem()
    fold = Fold(train=range(0, 400), val=range(410, 600))
    grid = [1.0e-8, 1.0e-4, 1.0, 100.0]
    by_cv = select_alpha_cv(phi, y, [fold], grid, bias_column=0)
    by_split = select_alpha(
        phi[0:400], y[0:400], phi[410:600], y[410:600], grid, bias_column=0
    )
    assert by_cv.alpha == by_split.alpha
    assert by_cv.val_nrmse == pytest.approx(by_split.val_nrmse)


def test_an_empty_grid_is_rejected() -> None:
    """格子が空なら落ちる。"""
    phi, y = _linear_problem()
    folds = make_folds(range(0, 600), 3, scheme=FoldScheme.ROLLING, embargo=5)
    with pytest.raises(ValueError, match="alpha 格子が空"):
        select_alpha_cv(phi, y, folds, [], bias_column=0)


def test_empty_folds_are_rejected() -> None:
    """折りが空なら落ちる (黙って何も採点しない経路を作らない)。"""
    phi, y = _linear_problem()
    with pytest.raises(ValueError, match="折りが空"):
        select_alpha_cv(phi, y, [], [1.0], bias_column=0)


# --- 既定の挙動を変えないこと ------------------------------------------------


def test_cross_validation_is_off_by_default() -> None:
    """既定では交差検証を使わない。

    ``results/`` は単一の検証区間で作られている。既定を変えると記事の数値が
    黙って変わるので、**既定が 0 であること自体**を固定する。
    """
    from rc_basics_lab.config import CrossValidationConfig, ExperimentConfig

    assert CrossValidationConfig().n_folds == 0
    assert ExperimentConfig().ridge.cv.n_folds == 0


def test_the_shipped_config_keeps_it_off() -> None:
    """同梱の 01 の YAML でも無効である。"""
    from pathlib import Path

    from rc_basics_lab.config import load_config

    config = load_config(Path("experiments/01_what_is_rc/config.yaml"))
    assert config.ridge.cv.n_folds == 0


def test_enabling_it_changes_the_selection_path() -> None:
    """有効にすると選択が実際に変わりうる (D-13: 効かない設定は飾りである)。

    **値が変わることではなく、経路が変わることを測る。** 交差検証は選択を
    安定させる方向に働くので、たまたま同じ alpha を選ぶ場合がある。ここでは
    折りを実際に作って採点した結果が返ることを見る。
    """
    phi, y = _linear_problem()
    folds = make_folds(range(0, 600), 4, scheme=FoldScheme.ROLLING, embargo=5)
    assert len(folds) == 4
    selection = select_alpha_cv(phi, y, folds, [1.0e-6, 1.0], bias_column=0)
    # 4 折の平均なので、どの単一の折りとも一致しない値になりうる
    assert 0.0 < selection.val_nrmse < 10.0

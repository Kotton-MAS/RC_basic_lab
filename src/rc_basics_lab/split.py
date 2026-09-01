"""時系列の連続分割 (D-05 / D-06).

時系列なのでシャッフルせず連続区間で切る。1レプリケート内では**全手法が
まったく同じ行 index** で学習・評価する。基準点は

    t0 = max(全手法の first_valid, washout)

の1つだけであり (``compute_t0``)、ここが手法ごとにずれた瞬間に全ベースライン
比較が無効になる。

split シードは「系列内のどこから使い始めるか」(``offset``) に実配線する。
未使用パラメータを作らないための配線であり、``seeds.split`` を変えると分割境界が
動き、``seeds.reservoir`` を変えても動かない
(``tests/test_experiment_fairness.py::test_split_seed_changes_boundaries``)。

分割窓の長さは ``offset`` に依存しない (``n_steps - max_start_offset - t0``)。
オフセットで行数まで変わると、レプリケート間で n_train が揺れて平均±標準偏差の
意味が濁るため。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """時系列を連続区間で切る分割設定 (シャッフルしない)。"""

    train_ratio: float = 0.5
    val_ratio: float = 0.15
    test_ratio: float = 0.35
    washout: int = 200
    max_start_offset: int = 200


_RATIO_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class Split:
    """1レプリケートぶんの分割。

    Attributes:
        t0: 全手法共通の基準行 (これより手前の行はどの手法でも使わない)。
        offset: split シードが選んだ ``t0`` からの追加オフセット。
        train: 学習行の range。
        val: 検証行の range (alpha と n_lags の選択に使う)。
        test: テスト行の range。
    """

    t0: int
    offset: int
    train: range
    val: range
    test: range

    @property
    def start(self) -> int:
        """分割窓の開始行 ``t0 + offset``。"""
        return self.train.start

    @property
    def stop(self) -> int:
        """分割窓の終了行 (排他)。"""
        return self.test.stop

    @property
    def sizes(self) -> tuple[int, int, int]:
        """``(n_train, n_val, n_test)``。"""
        return len(self.train), len(self.val), len(self.test)


def compute_t0(first_valids: Iterable[int], washout: int) -> int:
    """全手法共通の基準行を返す (D-05)。

    Args:
        first_valids: 各手法・各候補の設計行列の ``first_valid``。
        washout: リザバー状態の初期過渡を捨てる行数。

    Raises:
        ValueError: ``washout`` または ``first_valid`` が負の場合。
    """
    if washout < 0:
        raise ValueError(f"washout は 0 以上である必要があります: {washout}")
    values = list(first_valids)
    if any(value < 0 for value in values):
        raise ValueError(f"first_valid が負です: {values}")
    return max([washout, *values])


def make_split(
    cfg: SplitConfig, n_steps: int, t0: int, rng: np.random.Generator
) -> Split:
    """連続分割を作る。

    Args:
        cfg: 分割比・オフセット上限。
        n_steps: 系列長 T。
        t0: ``compute_t0`` の戻り値。
        rng: split ストリームの Generator (D-06)。

    Raises:
        ValueError: 比が正でない / 合計が 1 でない / 系列が短すぎる場合。
    """
    ratios = (cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)
    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError(f"分割比は正である必要があります: {ratios}")
    if abs(sum(ratios) - 1.0) > _RATIO_TOLERANCE:
        raise ValueError(f"分割比の合計が 1 ではありません: {sum(ratios)} {ratios}")
    if cfg.max_start_offset < 0:
        raise ValueError(
            f"max_start_offset は 0 以上である必要があります: {cfg.max_start_offset}"
        )
    if t0 < 0:
        raise ValueError(f"t0 は 0 以上である必要があります: {t0}")

    n_usable = n_steps - cfg.max_start_offset - t0
    if n_usable < 3:
        raise ValueError(
            "分割に使える行がありません: "
            f"T={n_steps}, t0={t0}, max_start_offset={cfg.max_start_offset}"
        )
    n_train = int(n_usable * cfg.train_ratio)
    n_val = int(n_usable * cfg.val_ratio)
    n_test = n_usable - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(
            "各分割に 1 行以上必要です: "
            f"{(n_train, n_val, n_test)} (n_usable={n_usable})"
        )

    offset = int(rng.integers(0, cfg.max_start_offset + 1))
    start = t0 + offset
    train = range(start, start + n_train)
    val = range(train.stop, train.stop + n_val)
    test = range(val.stop, val.stop + n_test)
    return Split(t0=t0, offset=offset, train=train, val=val, test=test)


__all__ = ["Split", "compute_t0", "make_split"]

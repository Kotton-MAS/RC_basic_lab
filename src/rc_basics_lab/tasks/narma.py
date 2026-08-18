"""NARMA10 系列の生成 (D-29 / D-30).

採用する漸化式 (**記事にそのまま載せる形**)::

    y[t+1] = 0.3 y[t] + 0.05 y[t] sum_{i=0}^{9} y[t-i] + 1.5 u[t-9] u[t] + 0.1
    u ~ U[0, 0.5] i.i.d.

**係数と入力分布はモジュール定数であり設定フィールドにしない** (D-29)。
文献の添字が2系統 (``1.5 u(n-9) u(n)`` 系と ``1.5 u(k-1) u(k-10)`` 系) に
割れているため、どちらを使ったかで数値が変わる。設定にすると『YAML を1行
変えると別の系になる』経路ができ、記事に載せた式と成果物の対応が黙って切れる。
``Narma10Config`` が持つのは ``length`` (と 01 側の土台 ``base``) だけである。

**発散 (``|y| > 1e3`` または非有限) は ``ValueError``** (D-30)。クリップも
自動再抽選もしない —— 黙ってクリップすると「クリップの飽和特性」という別の
非線形性を課題に混ぜることになり、遅延線対照が不当に有利/不利になる。
再抽選も、どのシードを捨てたかが記録されないと 3-C の公平性の外に穴が開く。

先頭 ``NARMA10_ORDER`` (=10) 行の目標は 0 で初期化する。この区間の ``y`` は
入力から観測できない初期条件だが、実験ランナーは ``t0 >= washout`` 行目から
評価するので使われない (``tasks/delay_parity.py`` の先読み区間と同じ扱い)。
"""

from __future__ import annotations

import math

import numpy as np

from rc_basics_lab.config import Narma10Config
from rc_basics_lab.tasks.base import TaskData
from rc_basics_lab.types import FloatArray

TASK_NAME = "narma10"

NARMA10_ORDER = 10
"""漸化式が参照する過去の本数 (``sum_{i=0}^{9}`` と ``u[t-9]`` の 10)。"""

NARMA10_LEAK = 0.3
"""``0.3 y[t]`` の係数。"""

NARMA10_QUADRATIC = 0.05
"""``0.05 y[t] sum_{i=0}^{9} y[t-i]`` の係数。"""

NARMA10_INPUT_PRODUCT = 1.5
"""``1.5 u[t-9] u[t]`` の係数。"""

NARMA10_OFFSET = 0.1
"""定数項。"""

NARMA10_INPUT_LOW = 0.0
NARMA10_INPUT_HIGH = 0.5
"""入力の一様分布 ``U[NARMA10_INPUT_LOW, NARMA10_INPUT_HIGH]`` (D-29)。"""

NARMA10_INPUT_STD = (NARMA10_INPUT_HIGH - NARMA10_INPUT_LOW) / math.sqrt(12.0)
"""宣言した入力分布の標準偏差 (一様分布の閉形式 ``(b - a) / sqrt(12)``)。

02・03 の掃引が持つ ``sigma_u`` (**駆動信号の標準偏差の設定値**、D-17) と
同じ意味の量である。3-C には駆動強度の設定フィールドが無いので、
``capacity.csv`` の ``sigma_u`` 列にはこの理論値を書く (実測値は
``input_drive_std`` 列に別途出る)。
"""

DIVERGENCE_LIMIT = 1.0e3
"""発散と見なす ``|y|`` の閾値 (D-30)。"""


def _validate(cfg: Narma10Config) -> None:
    if cfg.length < 1:
        raise ValueError(f"length は 1 以上である必要があります: {cfg.length}")


def narma10_series(u: FloatArray) -> FloatArray:
    """入力系列 ``u`` (1次元) から NARMA10 の出力系列を作る (D-29 の漸化式)。

    ``y[0] .. y[NARMA10_ORDER - 1]`` は 0 で初期化し、``t = 9 .. T-2`` について
    ``y[t+1]`` を漸化式で埋める。返る配列は ``u`` と同じ長さ。

    Args:
        u: 入力系列 ``(T,)``。

    Returns:
        出力系列 ``(T,)``。

    Raises:
        ValueError: ``u`` が1次元でない / 空 / 有限でない値を含む場合、および
            ``|y| > DIVERGENCE_LIMIT`` か非有限になった場合 (D-30)。
    """
    if u.ndim != 1:
        raise ValueError(f"u は1次元配列が必要です: {u.shape}")
    if u.shape[0] == 0:
        raise ValueError("u が空です")
    if not np.all(np.isfinite(u)):
        raise ValueError("u に有限でない値があります")
    n_steps = int(u.shape[0])
    y: FloatArray = np.zeros(n_steps, dtype=np.float64)
    for t in range(NARMA10_ORDER - 1, n_steps - 1):
        window = float(np.sum(y[t - NARMA10_ORDER + 1 : t + 1]))
        value = (
            NARMA10_LEAK * float(y[t])
            + NARMA10_QUADRATIC * float(y[t]) * window
            + NARMA10_INPUT_PRODUCT * float(u[t - NARMA10_ORDER + 1]) * float(u[t])
            + NARMA10_OFFSET
        )
        # D-30: クリップも再抽選もせず、その場で落とす。
        if not math.isfinite(value) or abs(value) > DIVERGENCE_LIMIT:
            raise ValueError(
                f"NARMA10 が発散しました (t={t + 1}, y={value!r}, "
                f"上限={DIVERGENCE_LIMIT})"
            )
        y[t + 1] = value
    return y


def generate_narma10(cfg: Narma10Config, rng: np.random.Generator) -> TaskData:
    """``u ~ U[0, 0.5]`` を引いて NARMA10 の入出力系列を作る。返す行数は ``cfg.length``。

    Args:
        cfg: 系列長 (係数と入力分布は設定にしない、D-29)。
        rng: **task ストリーム**の Generator (D-06)。

    Returns:
        ``u[t]`` と ``y[t]`` を同じ行 index に並べた ``TaskData``。

    Raises:
        ValueError: ``length`` が 1 未満、または系列が発散した場合 (D-30)。
    """
    _validate(cfg)
    drive: FloatArray = rng.uniform(NARMA10_INPUT_LOW, NARMA10_INPUT_HIGH, cfg.length)
    target = narma10_series(drive)
    params = {
        "order": str(NARMA10_ORDER),
        "leak": str(NARMA10_LEAK),
        "quadratic": str(NARMA10_QUADRATIC),
        "input_product": str(NARMA10_INPUT_PRODUCT),
        "offset": str(NARMA10_OFFSET),
        "input_low": str(NARMA10_INPUT_LOW),
        "input_high": str(NARMA10_INPUT_HIGH),
    }
    return TaskData(
        u=drive.reshape(-1, 1), y=target.reshape(-1, 1), name=TASK_NAME, params=params
    )


__all__ = [
    "DIVERGENCE_LIMIT",
    "NARMA10_INPUT_HIGH",
    "NARMA10_INPUT_LOW",
    "NARMA10_INPUT_PRODUCT",
    "NARMA10_INPUT_STD",
    "NARMA10_LEAK",
    "NARMA10_OFFSET",
    "NARMA10_ORDER",
    "NARMA10_QUADRATIC",
    "TASK_NAME",
    "generate_narma10",
    "narma10_series",
]

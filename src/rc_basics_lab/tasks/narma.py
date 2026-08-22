"""NARMA10 系列の生成 (D-29 / D-30).

採用する漸化式 (**記事にそのまま載せる形**)::

    y[t+1] = 0.3 y[t] + 0.05 y[t] sum_{i=0}^{9} y[t-i] + 1.5 u[t-9] u[t] + 0.1
    u ~ U[0, 0.5] i.i.d.

**係数と入力分布はモジュール定数であり設定フィールドにしない** (D-29)。
``Narma10Config`` が持つのは ``length`` (と 01 側の土台 ``base``) だけである。

**発散 (``|y| > 1e3`` または非有限) は ``ValueError``** (D-30)。クリップも
自動再抽選もしない。

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

_MAX_LENGTH = 200_000_000
"""``length`` 単体の上書き不能な絶対上限 (F-3b2-1-001/HIGH-1, CWE-400/789)。

3-C (実験 ``experiment/narma.py`` の ``run_narma10``) は ``CapacityCondition``
を持たず ``experiment/capacity.py`` の ``_validate_condition_bounds``
(確保より前に ``n_units`` / ``n_units * n_steps`` を落とす、D-34) を通らない
ため、課題層 (ここ) 単体で呼ばれても塞がるようにする。``generate_narma10``
は ``u`` / ``y`` を ``length`` 要素の ``float64`` で確保するので、
``length=10**12`` のような値を検査なしで通すと確保だけで数TBに達する。
``tasks/`` は ``experiment/`` に依存しない (レイヤ順序が逆) ので、
``experiment/capacity.py`` の ``_MAX_STATE_ELEMENTS`` と同じ値
(``2e8``、状態行列 ``8 * 2e8`` ≈ 1.6GB) をここに独立して定義する。
"""

_MAX_STATE_ELEMENTS = 200_000_000
"""``length * base.esn_mackey_glass.n_units`` の絶対上限 (F-3b2-1-001/HIGH-1)。

3-C のリザバー状態行列は ``(length, n_units)`` の ``float64`` を確保する
(``ESN.run``)。``length`` 単体は ``_MAX_LENGTH`` の25倍まで許すが、
``n_units`` (既定 200) を掛けると同じ確保が ``_MAX_LENGTH`` よりずっと手前で
危険域に入る (実測: ``length=1e8`` x ``n_units=50`` で状態行列だけ 5e9 要素
= 40GB、``_MAX_STATE_ELEMENTS=2e8`` の25倍)。``experiment/capacity.py`` の
``_MAX_STATE_ELEMENTS`` と同じ値・同じ threat model (CWE-400/789) だが、
レイヤ順序 (``tasks`` は ``experiment`` に依存しない) のため独立して定義する。
"""


def _validate(cfg: Narma10Config) -> None:
    if cfg.length < 1:
        raise ValueError(f"length は 1 以上である必要があります: {cfg.length}")
    if cfg.length > _MAX_LENGTH:
        raise ValueError(
            f"length が上限を超えています: {cfg.length} > {_MAX_LENGTH} "
            "(u / y の確保量は length に比例するため、確保する前に検査で落とす)"
        )
    n_units = cfg.base.esn_mackey_glass.n_units
    n_state_elements = cfg.length * n_units
    if n_state_elements > _MAX_STATE_ELEMENTS:
        raise ValueError(
            "length * base.esn_mackey_glass.n_units が上限を超えています: "
            f"{n_state_elements} > {_MAX_STATE_ELEMENTS} "
            "(3-C の状態行列の確保量は length * n_units に比例するため、"
            "確保する前に検査で落とす)"
        )


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
    """``u ~ U[0, 0.5]`` を引いて入出力系列を作る。返す行数は ``cfg.length``。

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

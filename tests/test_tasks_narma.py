"""NARMA10 課題の検査 (D-29 の漸化式 / D-30 の発散処理).

D-29 で最も壊れやすいのは**添字**である。文献の実装は ``1.5 u(n-9) u(n)`` 系と
``1.5 u(k-1) u(k-10)`` 系に割れており、どちらでも系列は「それらしく」見えるが
数値は変わる。``test_matches_reference_recurrence`` は先頭5ステップを
**漸化式から独立に、1ステップずつ書き下した式**で照合するので、添字が1つ
ずれるとその場で落ちる。
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from rc_basics_lab.config import Narma10Config
from rc_basics_lab.seeds import SeedConfig, SeedStream, make_rng
from rc_basics_lab.tasks.narma import (
    DIVERGENCE_LIMIT,
    NARMA10_INPUT_HIGH,
    NARMA10_INPUT_LOW,
    NARMA10_INPUT_STD,
    NARMA10_ORDER,
    TASK_NAME,
    generate_narma10,
    narma10_series,
)
from rc_basics_lab.types import FloatArray

REFERENCE_INPUT: tuple[Fraction, ...] = (
    Fraction(1, 10),
    Fraction(2, 10),
    Fraction(3, 10),
    Fraction(4, 10),
    Fraction(5, 10),
    Fraction(5, 100),
    Fraction(15, 100),
    Fraction(25, 100),
    Fraction(35, 100),
    Fraction(45, 100),
    Fraction(4, 10),
    Fraction(3, 10),
    Fraction(2, 10),
    Fraction(1, 10),
    Fraction(5, 100),
    Fraction(1, 10),
    Fraction(2, 10),
    Fraction(3, 10),
    Fraction(4, 10),
    Fraction(45, 100),
    Fraction(35, 100),
    Fraction(25, 100),
)
"""手計算に使う入力列 (22 ステップ)。値はすべて有理数で表せるものを選ぶ。

先頭15ステップ (~ ``y[14]``) だけでは足りない (F-3b2-1-002/HIGH-2)。
``y[0..9]`` が 0 初期化のため、窓を ``sum_{i=0}^{9}`` から
``sum_{i=0}^{10}`` に広げる変異は、11項目 (``y[t-10]``) が非0の値を
拾い始める ``t - 10 >= 10`` (= ``y[21]`` を計算する時点) まで検出できない。
それより手前では11項目が常に ``y[<10] == 0`` を拾うだけなので、10項の
正しい実装と偶然一致してしまう。22ステップまで延ばすのはこの点に届かせるため。
"""

RECURRENCE_TOLERANCE = 1.0e-12
"""手計算との許容差 (仕様 §4 T4 の受け入れ基準)。"""


def _hand_computed_first_five_steps() -> tuple[Fraction, ...]:
    """``y[10] .. y[14]`` を**1ステップずつ書き下して**有理数で計算する。

    実装 (``narma10_series``) はループで畳んでいるが、ここでは 5 ステップを
    展開して書く。係数もモジュール定数を import せず数値で直書きする ——
    実装と同じ定数・同じループを参照すると、添字や係数を取り違えたまま
    両方が同じ値を出す「同語反復のテスト」になる。

    ``y[0] .. y[9]`` は 0 なので、``sum_{i=0}^{9} y[t-i]`` の窓は
    ステップごとに 1本 -> 2本 -> 3本 -> 4本 と増える (窓の下端が
    ``y[1]``, ``y[2]``, ... と上がるが、そこは 0 のまま)。
    """
    u = REFERENCE_INPUT
    leak = Fraction(3, 10)
    quadratic = Fraction(1, 20)
    product = Fraction(3, 2)
    offset = Fraction(1, 10)
    zero = Fraction(0)

    # t=9: y[10] = 0.3 y[9] + 0.05 y[9] (y[9]+...+y[0]) + 1.5 u[0] u[9] + 0.1
    y10 = leak * zero + quadratic * zero * zero + product * u[0] * u[9] + offset
    # t=10: 窓は y[10]+y[9]+...+y[1] = y[10]
    y11 = leak * y10 + quadratic * y10 * y10 + product * u[1] * u[10] + offset
    # t=11: 窓は y[11]+y[10]
    y12 = leak * y11 + quadratic * y11 * (y11 + y10) + product * u[2] * u[11] + offset
    # t=12: 窓は y[12]+y[11]+y[10]
    y13 = (
        leak * y12
        + quadratic * y12 * (y12 + y11 + y10)
        + product * u[3] * u[12]
        + offset
    )
    # t=13: 窓は y[13]+y[12]+y[11]+y[10]
    y14 = (
        leak * y13
        + quadratic * y13 * (y13 + y12 + y11 + y10)
        + product * u[4] * u[13]
        + offset
    )
    return (y10, y11, y12, y13, y14)


def _hand_computed_window_reference(
    u: tuple[Fraction, ...], upto: int, *, window_width: int = NARMA10_ORDER
) -> tuple[Fraction, ...]:
    """独立実装 (Fraction) で ``y[NARMA10_ORDER] .. y[upto]`` を計算する。

    ``narma10_series`` のコードは一切参照しない (係数はリテラルの Fraction、
    ループもここに独立して書く)。``_hand_computed_first_five_steps`` と同じ
    数値をこのループでも再現することは ``test_matches_reference_recurrence``
    で確認済み。``window_width`` を変えると窓の項数を実装と違えられる ——
    HIGH-2 (D-29 のセルフテスト、F-3b2-1-002) が使う。
    """
    leak = Fraction(3, 10)
    quadratic = Fraction(1, 20)
    product = Fraction(3, 2)
    offset = Fraction(1, 10)
    y: list[Fraction] = [Fraction(0)] * NARMA10_ORDER
    for t in range(NARMA10_ORDER - 1, upto):
        lo = max(t - window_width + 1, 0)
        window = sum(y[lo : t + 1], Fraction(0))
        y.append(
            leak * y[t]
            + quadratic * y[t] * window
            + product * u[t - NARMA10_ORDER + 1] * u[t]
            + offset
        )
    return tuple(y[NARMA10_ORDER : upto + 1])


def test_matches_reference_recurrence() -> None:
    """手計算した先頭5ステップと ``1e-12`` 以内で一致する (D-29)。

    添字が1つずれた実装 (``1.5 u[t-1] u[t-10]`` 系) は、この入力列では
    ``y[10]`` の時点で 0.1675 ではなく別の値になる。
    """
    u = np.array([float(value) for value in REFERENCE_INPUT], dtype=np.float64)
    y = narma10_series(u)

    assert y.shape == u.shape
    # 初期条件の 10 行は 0 (定義できない過去には触れない)
    assert np.array_equal(y[:NARMA10_ORDER], np.zeros(NARMA10_ORDER))

    expected = _hand_computed_first_five_steps()
    actual = y[NARMA10_ORDER : NARMA10_ORDER + len(expected)]
    assert len(actual) == 5
    for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        assert abs(float(value) - float(reference)) <= RECURRENCE_TOLERANCE, (
            f"step {NARMA10_ORDER + index}: {value!r} != {float(reference)!r}"
        )
    # 手計算の値そのもの (定数を1つ書き換えたら気づけるように literal で置く)
    assert float(expected[0]) == 0.1675


def test_shifted_index_would_not_match() -> None:
    """添字を1つずらすと手計算と一致しない (上のテストが空虚でないことの確認)。

    ``1.5 u[t-9] u[t]`` を ``1.5 u[t-1] u[t-10]`` 系に取り違えた実装を
    その場で書き、同じ入力列で**異なる**値になることを示す。
    """
    u = [float(value) for value in REFERENCE_INPUT]
    shifted = [0.0] * len(u)
    for t in range(NARMA10_ORDER - 1, len(u) - 1):
        window = sum(shifted[t - NARMA10_ORDER + 1 : t + 1])
        shifted[t + 1] = (
            0.3 * shifted[t]
            + 0.05 * shifted[t] * window
            # 取り違えた版: u[t] u[t-9] ではなく u[t-1] u[t-10]
            + 1.5 * u[t - 1] * u[max(t - NARMA10_ORDER, 0)]
            + 0.1
        )
    expected = _hand_computed_first_five_steps()
    assert abs(shifted[NARMA10_ORDER] - float(expected[0])) > RECURRENCE_TOLERANCE


def test_matches_reference_recurrence_through_extended_window() -> None:
    """先頭5ステップだけでなく ``y[21]`` まで手計算と ``1e-12`` 以内で一致する.

    HIGH-2 (F-3b2-1-002): 窓を ``sum_{i=0}^{9}`` から ``sum_{i=0}^{10}``
    (11項) に広げる変異は、``y[14]`` までしか照合しない従来の
    ``test_matches_reference_recurrence`` では検出できなかった —— ``y[0..9]``
    が 0 初期化のため、11項目 (``y[t-10]``) が非0の値を拾い始めるのは
    ``t - 10 >= 10`` (= ``y[21]`` を計算する時点) からである。ここまで区間を
    延ばして検出する (空虚でないことは ``test_wider_window_would_not_match``
    が確認する)。
    """
    u = np.array([float(value) for value in REFERENCE_INPUT], dtype=np.float64)
    y = narma10_series(u)

    upto = len(REFERENCE_INPUT) - 1  # 21
    expected = _hand_computed_window_reference(REFERENCE_INPUT, upto)
    actual = y[NARMA10_ORDER : upto + 1]
    assert len(actual) == len(expected)
    for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        assert abs(float(value) - float(reference)) <= RECURRENCE_TOLERANCE, (
            f"step {NARMA10_ORDER + index}: {value!r} != {float(reference)!r}"
        )
    # 手計算の値そのもの (定数を1つ書き換えたら気づけるように literal で置く)
    assert abs(float(expected[-1]) - 0.5398125327224791) < RECURRENCE_TOLERANCE


def test_wider_window_would_not_match() -> None:
    """窓を11項に広げると ``y[21]`` で初めて食い違う (上のテストの空虚さ確認).

    HIGH-2 (F-3b2-1-002)、``test_shifted_index_would_not_match`` と対。
    ``y[0..9]`` が 0 初期化のため、``y[20]`` までは10項の正しい窓と
    11項の変異が偶然一致する —— 短い区間だけを照合する guard_test では
    この変異を検出できないことを、ここで実測して示す。
    """
    upto = len(REFERENCE_INPUT) - 1  # 21
    correct = _hand_computed_window_reference(REFERENCE_INPUT, upto)
    wide = _hand_computed_window_reference(
        REFERENCE_INPUT, upto, window_width=NARMA10_ORDER + 1
    )
    assert len(correct) == len(wide)
    # y[21] (最後の要素) より手前は偶然一致する。
    for index in range(len(correct) - 1):
        assert correct[index] == wide[index]
    # y[21] で初めて食い違う。
    assert abs(float(correct[-1]) - float(wide[-1])) > RECURRENCE_TOLERANCE


def test_divergence_raises_instead_of_clipping() -> None:
    """発散は ``ValueError``。クリップも自動再抽選もしない (D-30)。

    宣言した入力分布 ``U[0, 0.5]`` の**内側**でも発散は起こる (シード 0〜199 の
    うち 6 本が 8000 ステップ以内に発散する)。黙ってクリップすると
    「クリップの飽和特性」という別の非線形性が課題に混ざり、遅延線対照が
    不当に有利/不利になる。
    """
    with pytest.raises(ValueError, match="発散"):
        generate_narma10(Narma10Config(length=8000), np.random.default_rng(75))

    # 定数入力でも同じ (境界 0.5 を張り付かせると発散する)
    with pytest.raises(ValueError, match="発散"):
        narma10_series(np.full(200, 0.5, dtype=np.float64))

    # 上限を超えた値が「丸められて」返っていないこと (返り値経路が無い)
    diverging = np.full(200, 2.0, dtype=np.float64)
    with pytest.raises(ValueError) as error:
        narma10_series(diverging)
    assert str(DIVERGENCE_LIMIT) in str(error.value)


def test_production_replicates_do_not_diverge() -> None:
    """本番の 5 レプリケートは発散しない (D-30 が 3-C を止めないことの実測)。

    task ストリーム (D-06) から引いた入力で確認する。**レプリケート7 は
    発散する**ので、``narma.base.n_replicates`` を 7 以上に増やすときは
    D-30 で落ちることを承知の上で行うこと。
    """
    seeds = SeedConfig(reservoir=0, task=1, split=2)
    cfg = Narma10Config(length=8000)
    for replicate in range(5):
        data = generate_narma10(cfg, make_rng(seeds, SeedStream.TASK, replicate))
        assert np.all(np.isfinite(data.y))
        assert float(np.max(np.abs(data.y))) < DIVERGENCE_LIMIT
    with pytest.raises(ValueError, match="発散"):
        generate_narma10(cfg, make_rng(seeds, SeedStream.TASK, 7))


def test_input_is_uniform_on_the_declared_range() -> None:
    """入力は ``U[0, 0.5]`` i.i.d. で、宣言した標準偏差は閉形式と一致する (D-29)。"""
    data = generate_narma10(Narma10Config(length=20000), np.random.default_rng(3))
    u = data.u[:, 0]
    assert data.name == TASK_NAME
    assert float(np.min(u)) >= NARMA10_INPUT_LOW
    assert float(np.max(u)) <= NARMA10_INPUT_HIGH
    assert pytest.approx(0.5 / np.sqrt(12.0)) == NARMA10_INPUT_STD
    # 実測の標準偏差は理論値の近くにあるが**一致はしない** (だから列を分ける)
    assert float(np.std(u)) == pytest.approx(NARMA10_INPUT_STD, rel=0.02)
    assert float(np.std(u)) != NARMA10_INPUT_STD


def test_shapes_and_params() -> None:
    """``(T, 1)`` の2次元で返り、採用式のパラメータが params に残る。"""
    data = generate_narma10(Narma10Config(length=500), np.random.default_rng(1))
    assert data.u.shape == (500, 1)
    assert data.y.shape == (500, 1)
    assert data.n_inputs == 1
    assert data.params["order"] == "10"
    assert data.params["input_high"] == "0.5"


@pytest.mark.parametrize(
    ("length", "message"),
    [pytest.param(0, "length", id="zero"), pytest.param(-1, "length", id="negative")],
)
def test_invalid_length_raises(length: int, message: str) -> None:
    """系列長が 1 未満なら ``ValueError`` (他の課題と同じ規律)。"""
    with pytest.raises(ValueError, match=message):
        generate_narma10(Narma10Config(length=length), np.random.default_rng(0))


@pytest.mark.parametrize(
    "u",
    [
        pytest.param(np.zeros((3, 1)), id="2d"),
        pytest.param(np.zeros(0), id="empty"),
        pytest.param(np.array([0.1, np.nan, 0.2]), id="nan"),
    ],
)
def test_series_rejects_malformed_input(u: FloatArray) -> None:
    """``narma10_series`` は形の壊れた入力を黙って受理しない。"""
    with pytest.raises(ValueError):
        narma10_series(u.astype(np.float64))

"""``reservoir/registry.py`` と ``reservoir/protocol.py`` の検査.

モデルを足せる形になっていることを測る。**「足せるはず」ではなく、
足したものが本番経路を通ることを測る** —— Protocol を定義しただけでは、
実験層が具体型を名指ししたままでも緑になる。
"""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest

from rc_basics_lab.config._common import _kind_of
from rc_basics_lab.reservoir.deep import DeepESNConfig
from rc_basics_lab.reservoir.esn import ESN, ESNConfig
from rc_basics_lab.reservoir.protocol import Reservoir, ReservoirConfig
from rc_basics_lab.reservoir.registry import build_reservoir
from rc_basics_lab.reservoir.ring import RingConfig
from rc_basics_lab.types import FloatArray


def test_the_registry_builds_an_esn_by_default() -> None:
    """既定の設定は ESN を返す (既存 YAML と成果物を変えないため)。"""
    reservoir = build_reservoir(ESNConfig(n_units=32), np.random.default_rng(0))
    assert isinstance(reservoir, ESN)
    assert reservoir.n_units == 32


def test_the_registry_draws_the_same_weights_as_a_direct_construction() -> None:
    """生成口を通しても**乱数の引き方が変わらない** (D-74)。

    ここがずれると成果物のバイト不変が壊れる。指紋の検査 (実験を丸ごと回す)
    より手前で、同じことを 1 ミリ秒で捕まえる。
    """
    config = ESNConfig(n_units=48)
    direct = ESN(config, np.random.default_rng(7))
    built = build_reservoir(config, np.random.default_rng(7))
    assert isinstance(built, ESN)
    assert built.W.tobytes() == direct.W.tobytes()
    assert built.W_in.tobytes() == direct.W_in.tobytes()


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (ESNConfig(n_units=16), "ESN"),
        (DeepESNConfig(n_units=16, n_layers=2, density=0.3), "DeepESN"),
        (RingConfig(n_units=16), "RingReservoir"),
    ],
)
def test_every_registered_kind_builds_its_model(
    config: ReservoirConfig, expected: str
) -> None:
    """登録した全モデルが生成口から出る。

    ``ReservoirConfig`` に足して ``registry`` の ``case`` を書き忘れたら
    mypy が落とすが、**逆 (case はあるが union に無い)** は型では捕まらない。
    ここが全モデルを回す。
    """
    built = build_reservoir(config, np.random.default_rng(0))
    assert type(built).__name__ == expected
    assert built.n_units == 16


def test_every_union_member_is_reachable_from_the_registry() -> None:
    """``ReservoirConfig`` の全要素が生成口で作れる (取りこぼしが無い)。

    union に足したのに ``case`` を書き忘れる方向は mypy が見るが、
    **既定値だけでは作れない設定**があると実行時に初めて分かる。
    """
    members = get_args(ReservoirConfig.__value__)
    assert len(members) >= 3, f"union の要素が減っています: {members}"
    for member in members:
        assert _kind_of(member) is not None, f"{member} が KIND を名乗っていません"


def test_the_esn_satisfies_the_reservoir_protocol() -> None:
    """``ESN`` が接合面を満たす。

    ``runtime_checkable`` の ``isinstance`` はメソッドの**有無**しか見ない
    ので、署名まで見る静的検査 (mypy) と対にして初めて意味を持つ。ここは
    「面から属性が消えていないか」だけを見る。
    """
    reservoir = build_reservoir(ESNConfig(n_units=16), np.random.default_rng(1))
    assert isinstance(reservoir, Reservoir)
    for name in ("run", "step", "config", "n_units", "n_inputs"):
        assert hasattr(reservoir, name), f"接合面から {name} が消えています"


class _ConstantReservoir:
    """状態が常に同じ値になるだけのリザバー (接合面の検査用)。

    ``ESN`` を1行も参照せずに ``Reservoir`` を満たせることを示す。満たせない
    なら、面に ESN 固有のものが混ざっている。
    """

    def __init__(self, n_units: int, n_inputs: int, value: float) -> None:
        self._n_units = n_units
        self._n_inputs = n_inputs
        self._value = value

    @property
    def config(self) -> ESNConfig:
        return ESNConfig(n_units=self._n_units)

    @property
    def n_units(self) -> int:
        return self._n_units

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    def step(
        self,
        x: FloatArray,
        u: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> FloatArray:
        del x, u, rng
        filled: FloatArray = np.full(self._n_units, self._value, dtype=np.float64)
        return filled

    def run(
        self,
        u: FloatArray,
        x0: FloatArray | None = None,
        rng: np.random.Generator | None = None,
    ) -> FloatArray:
        del x0, rng
        rows = int(np.asarray(u).shape[0])
        states: FloatArray = np.full(
            (rows, self._n_units), self._value, dtype=np.float64
        )
        return states


def test_a_reservoir_that_never_mentions_the_esn_satisfies_the_protocol() -> None:
    """ESN を参照しない実装が接合面を満たす (**これが拡張点の意味**)。

    満たせなくなったら、面に ESN 固有のもの (``W`` など) が混ざっている。
    """
    other: Reservoir = _ConstantReservoir(n_units=5, n_inputs=1, value=0.25)
    states = other.run(np.zeros((10, 1), dtype=np.float64))
    assert states.shape == (10, 5)
    assert other.step(states[0], np.zeros(1, dtype=np.float64)).shape == (5,)


def test_the_registry_rejects_an_invalid_configuration() -> None:
    """設定値の検査は各モデルの ``__init__`` が持つ (生成口では重複させない)。"""
    with pytest.raises(ValueError):
        build_reservoir(ESNConfig(n_units=0), np.random.default_rng(0))

"""系列源の共通型 ``SeriesSource`` の検査 (D-71).

T3 は合成源・MGAB・UCR の3つを**同じループ**に流す。共通型が無いと、そこで
実験層に源ごとの ``if`` 分岐が3本入り、``datasets -> tasks`` の一方向依存
(D-59) が実験層で崩れる (実験モジュールが両層の具象を捌く形になる)。

ここが固定するのは2つ:

1. **3実装が Protocol を満たす** —— 属性の有無だけでなく ``__call__`` の
   引数名・種別まで突き合わせる。``isinstance`` (``runtime_checkable``) は
   メソッドの**存在**しか見ないので、引数を変えても素通りする
2. **1つの ``Mapping[str, SeriesSource]`` で回せる** —— 源の具象名で分岐せずに
   「使えるものだけ回す」ループが書けること

実データが無い環境でも走る (D-60): ``is_available`` はネットワークに触れず、
キャッシュが無ければ ``False`` を返すだけである。
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import get_type_hints

import numpy as np
import pytest

from rc_basics_lab.config import SyntheticAnomalyConfig, SyntheticMackeyGlassConfig
from rc_basics_lab.datasets import mgab, ucr
from rc_basics_lab.tasks.anomaly import (
    AnomalySeries,
    SeriesSource,
    SyntheticSeriesSource,
)

SMALL = SyntheticAnomalyConfig(
    length=1500,
    n_anomalies=2,
    segment_length=40,
    ignore_margin=10,
    mackey_glass=SyntheticMackeyGlassConfig(),
)
"""合成源の小さい設定 (1本の生成が実測 0.011 秒)。"""


def _sources() -> dict[str, SeriesSource]:
    """3源を1つの辞書にまとめる (T3 の実験ループが受け取る形そのもの)。

    **この注釈が要点**である。``Mapping[str, SeriesSource]`` として書ける
    ことを mypy strict が構造的に検査するので、どれかの実装が Protocol から
    外れれば ``make type`` が赤くなる (実行時の検査は下のテストが受け持つ)。
    """
    return {
        "synthetic": SyntheticSeriesSource(cfg=SMALL),
        "mgab_1": mgab.MgabSeriesSource(series="1"),
        f"ucr_{ucr.subset()[0]}": ucr.UcrSeriesSource(filename=ucr.subset()[0]),
    }


def _signature_of(owner: type, method: str) -> list[tuple[str, str]]:
    """``owner.method`` の ``(引数名, 引数の種別)`` の並び (``self`` は除く)。

    クラスとメソッド名で受けるのは、関数オブジェクトを引数の型として書くと
    ``Callable[..., object]`` になり、``disallow_any_explicit`` (CLAUDE.md の
    「``Any`` 禁止」) に触れるため。
    """
    return [
        (name, parameter.kind.name)
        for name, parameter in inspect.signature(
            getattr(owner, method)
        ).parameters.items()
        if name != "self"
    ]


@pytest.mark.parametrize("key", sorted(_sources()))
def test_each_source_satisfies_the_series_source_protocol(key: str) -> None:
    """3実装が ``SeriesSource`` を満たす —— 署名まで突き合わせる (D-71)。

    ``isinstance(source, SeriesSource)`` はメソッドの存在しか見ないため、
    ``__call__(self, rng)`` を ``__call__(self, generator, *, data_dir)`` に
    変えても通ってしまう。``inspect.signature`` で Protocol 側の宣言と
    引数名・種別・戻り値の型まで一致させることで、「1つの辞書で回せる」を
    実行時にも固定する。
    """
    source = _sources()[key]
    assert isinstance(source, SeriesSource)

    assert _signature_of(type(source), "__call__") == _signature_of(
        SeriesSource, "__call__"
    ), f"{key} の __call__ の引数が SeriesSource と違います (D-71)"
    assert _signature_of(type(source), "is_available") == _signature_of(
        SeriesSource, "is_available"
    ), f"{key} の is_available の引数が SeriesSource と違います (D-71)"

    hints = get_type_hints(type(source).__call__)
    assert hints["return"] is AnomalySeries
    assert get_type_hints(type(source).is_available)["return"] is bool


def test_availability_is_decidable_without_touching_the_network() -> None:
    """``is_available`` が3源すべてで**呼べる** (D-60)。

    合成源は常に ``True``。実データ源はキャッシュの有無で決まるので、値そのもの
    ではなく「ネットワーク無しに bool が返る」ことを固定する。
    """
    availability = {key: source.is_available() for key, source in _sources().items()}
    assert availability["synthetic"] is True
    assert all(isinstance(value, bool) for value in availability.values())


def test_one_loop_runs_every_available_source_without_naming_it() -> None:
    """源の具象名で分岐せずに「使えるものだけ回す」ループが書ける (D-71)。

    T3 の実験ループの最小形。``sources`` の鍵も値の型も見ずに、
    ``is_available`` と ``__call__`` だけで回している —— ここに ``if key ==
    "mgab"`` が要るなら Protocol が用を成していない。
    """
    sources: Mapping[str, SeriesSource] = _sources()
    rng = np.random.default_rng(0)
    loaded: dict[str, AnomalySeries] = {
        key: source(rng) for key, source in sources.items() if source.is_available()
    }
    assert "synthetic" in loaded, "合成源は常に使えます (D-60)"
    for key, series in loaded.items():
        assert isinstance(series, AnomalySeries)
        assert series.n_steps > 0, key


def test_the_synthetic_binding_does_not_reimplement_the_generator() -> None:
    """``SyntheticSeriesSource`` は ``generate_synthetic_anomalies`` と同一の出力。

    束縛が「実験層から呼ぶと少し違う合成源」になっていないことの実測。
    """
    from rc_basics_lab.tasks.anomaly import generate_synthetic_anomalies

    direct = generate_synthetic_anomalies(SMALL, np.random.default_rng(7))
    through = SyntheticSeriesSource(cfg=SMALL)(np.random.default_rng(7))
    assert np.array_equal(np.asarray(direct.values), np.asarray(through.values))
    assert np.array_equal(np.asarray(direct.labels), np.asarray(through.labels))
    assert direct.train_end == through.train_end
    assert direct.params == through.params


def test_the_protocol_lives_in_the_pure_task_layer() -> None:
    """``SeriesSource`` の在り処が ``tasks/anomaly.py`` である (D-59 / D-71)。

    ``datasets`` 側に置くと、合成源しか使わない実験まで I/O 層を import する
    ことになり、依存の向き ``datasets -> tasks`` が実験層で反転する。
    """
    assert SeriesSource.__module__ == "rc_basics_lab.tasks.anomaly"
    assert mgab.MgabSeriesSource.__module__ == "rc_basics_lab.datasets.mgab"
    assert ucr.UcrSeriesSource.__module__ == "rc_basics_lab.datasets.ucr"

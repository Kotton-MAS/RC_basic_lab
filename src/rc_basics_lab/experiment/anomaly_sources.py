"""``dataset.source`` から系列源の辞書を作る唯一の場所 (D-71 / D-59).

``experiment/`` の中で ``datasets`` (I/O を持つ唯一のパッケージ) を import
するのは**このモジュールだけ**である。実験本体 (``experiment/anomaly.py``) は
``tasks.anomaly.SeriesSource`` という Protocol にしか触れないので、
合成源でも MGAB でも UCR でも同じコードを通る。

源の具象名で分岐してよいのも ``build_sources`` 1箇所だけである。分岐が
2箇所目に生えた瞬間に「実験モジュールが ``datasets`` と ``tasks`` の両方の
具象を if で捌く」形になり、``datasets -> tasks`` の一方向依存 (D-59) が
実験層で崩れる。
"""

from __future__ import annotations

from collections.abc import Mapping

from rc_basics_lab.config import Anomaly05Config
from rc_basics_lab.datasets import mgab, ucr
from rc_basics_lab.tasks.anomaly import SeriesSource, SyntheticSeriesSource

SYNTHETIC_SOURCE = "synthetic"
MGAB_SOURCE = "mgab"
UCR_SOURCE = "ucr"

ANOMALY_SOURCES: tuple[str, ...] = (SYNTHETIC_SOURCE, MGAB_SOURCE, UCR_SOURCE)
"""``dataset.source`` に書ける値。

既定は合成 (D-60: pytest はネットワークに触れない)。
"""


def build_sources(config: Anomaly05Config) -> Mapping[str, SeriesSource]:
    """``dataset.source`` から系列名 -> 系列源の辞書を作る。

    Args:
        config: 実験設定。``dataset.series`` の各要素が鍵になる。

    Raises:
        ValueError: ``dataset.source`` が未対応、または ``dataset.series`` が空。
    """
    names = config.dataset.series
    if not names:
        raise ValueError("dataset.series が空です")
    match config.dataset.source:
        case "synthetic":
            # 合成源は系列名によらず同一の設定を使う (系列の違いは task
            # ストリームの draw だけで作る)。1インスタンスを配ることで
            # 「系列ごとに少し違う合成源」が生まれる経路も消える。
            return dict.fromkeys(names, SyntheticSeriesSource(cfg=config.synthetic))
        case "mgab":
            return {name: mgab.MgabSeriesSource(series=name) for name in names}
        case "ucr":
            return {name: ucr.UcrSeriesSource(filename=name) for name in names}
        case _:
            raise ValueError(
                f"dataset.source は {ANOMALY_SOURCES} のいずれかです: "
                f"{config.dataset.source!r}"
            )


__all__ = [
    "ANOMALY_SOURCES",
    "MGAB_SOURCE",
    "SYNTHETIC_SOURCE",
    "UCR_SOURCE",
    "build_sources",
]

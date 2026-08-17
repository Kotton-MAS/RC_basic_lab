"""乱数ストリームの分離 (D-06).

リザバー生成 / タスク生成 / 分割 の3ストリームを ``SeedSequence`` の
``spawn_key`` で分離する。1ストリームのシードを変えても他ストリームが生成する
乱数列はバイト単位で不変であり、「リザバーだけ変えたときの分散」を測れる。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class SeedStream(Enum):
    """独立に扱う乱数ストリームの種別。"""

    RESERVOIR = "reservoir"
    TASK = "task"
    SPLIT = "split"


# spawn_key の第1要素。値は永続化される (変えると既存結果が再現しなくなる)。
_STREAM_INDEX: dict[SeedStream, int] = {
    SeedStream.RESERVOIR: 0,
    SeedStream.TASK: 1,
    SeedStream.SPLIT: 2,
}


@dataclass(frozen=True, slots=True)
class SeedConfig:
    """3ストリームそれぞれの基底シード。"""

    reservoir: int = 0
    task: int = 1
    split: int = 2


def _base_seed(config: SeedConfig, stream: SeedStream) -> int:
    """ストリームに対応する基底シードを返す。

    他ストリームのシードを一切参照しないことが独立性の根拠なので、
    ``getattr`` ではなく明示的な分岐で書く。
    """
    match stream:
        case SeedStream.RESERVOIR:
            return config.reservoir
        case SeedStream.TASK:
            return config.task
        case SeedStream.SPLIT:
            return config.split


def make_rng(
    config: SeedConfig, stream: SeedStream, replicate: int
) -> np.random.Generator:
    """``(stream, replicate)`` に対応する Generator を返す。

    Args:
        config: 3ストリームの基底シード。
        stream: 取り出すストリーム。
        replicate: レプリケート番号 (0 始まり)。

    Raises:
        ValueError: ``replicate`` が負の場合。
    """
    if replicate < 0:
        raise ValueError(f"replicate は 0 以上である必要があります: {replicate}")
    sequence = np.random.SeedSequence(
        entropy=_base_seed(config, stream),
        spawn_key=(_STREAM_INDEX[stream], replicate),
    )
    return np.random.default_rng(sequence)


__all__ = ["SeedConfig", "SeedStream", "make_rng"]

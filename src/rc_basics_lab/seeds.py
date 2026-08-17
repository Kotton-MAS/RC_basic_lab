"""乱数ストリームの分離 (D-06).

リザバー生成 / タスク生成 / 分割 / 初期状態プローブ の4ストリームを
``SeedSequence`` の ``spawn_key`` で分離する。1ストリームのシードを変えても
他ストリームが生成する乱数列はバイト単位で不変であり、「リザバーだけ変えた
ときの分散」を測れる。

``PROBE`` は 02 の ESP 判定が使う初期状態対の生成専用のストリームで、
**01 の ``SeedConfig`` には対応するフィールドを持たない** (D-14)。
01 の設定にフィールドを足すと ``tests/test_config_wiring.py`` の被覆
(全フィールドが 01 のパイプライン出力を変える) が破れるため、02 側は
``EspSeedConfig`` に基底シードを持ち ``make_rng_for`` で直接渡す。
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
    PROBE = "probe"


# spawn_key の第1要素。値は永続化される (変えると既存結果が再現しなくなる)。
# PROBE には既存3本と重ならない 3 を割り当てる (0〜2 を動かすと 01 の成果物が
# 再現しなくなるため、追加は必ず末尾に足す)。
_STREAM_INDEX: dict[SeedStream, int] = {
    SeedStream.RESERVOIR: 0,
    SeedStream.TASK: 1,
    SeedStream.SPLIT: 2,
    SeedStream.PROBE: 3,
}


@dataclass(frozen=True, slots=True)
class SeedConfig:
    """01 が使う3ストリームそれぞれの基底シード。

    ``PROBE`` は意図的に含めない (D-14)。02 の初期状態対は
    ``EspSeedConfig`` から ``make_rng_for`` 経由で引く。
    """

    reservoir: int = 0
    task: int = 1
    split: int = 2


def _base_seed(config: SeedConfig, stream: SeedStream) -> int:
    """ストリームに対応する基底シードを返す。

    他ストリームのシードを一切参照しないことが独立性の根拠なので、
    ``getattr`` ではなく明示的な分岐で書く。

    Raises:
        ValueError: ``SeedConfig`` が基底シードを持たないストリームの場合。
    """
    match stream:
        case SeedStream.RESERVOIR:
            return config.reservoir
        case SeedStream.TASK:
            return config.task
        case SeedStream.SPLIT:
            return config.split
        case SeedStream.PROBE:
            raise ValueError(
                "SeedConfig は PROBE の基底シードを持ちません (D-14)。"
                " make_rng_for(base_seed, SeedStream.PROBE, replicate) を使ってください"
            )


def make_rng_for(
    base_seed: int, stream: SeedStream, replicate: int
) -> np.random.Generator:
    """基底シードを直接指定して ``(stream, replicate)`` の Generator を返す。

    ``make_rng`` の下位互換な一般形。設定 dataclass を経由せずに引けるので、
    ``SeedConfig`` に無いストリーム (``PROBE``) をここから使える (D-14)。

    Args:
        base_seed: そのストリームの基底シード。
        stream: 取り出すストリーム。
        replicate: レプリケート番号 (0 始まり)。

    Raises:
        ValueError: ``replicate`` が負の場合。
    """
    if replicate < 0:
        raise ValueError(f"replicate は 0 以上である必要があります: {replicate}")
    sequence = np.random.SeedSequence(
        entropy=base_seed,
        spawn_key=(_STREAM_INDEX[stream], replicate),
    )
    return np.random.default_rng(sequence)


def make_rng(
    config: SeedConfig, stream: SeedStream, replicate: int
) -> np.random.Generator:
    """``(stream, replicate)`` に対応する Generator を返す。

    Args:
        config: 3ストリームの基底シード。
        stream: 取り出すストリーム。
        replicate: レプリケート番号 (0 始まり)。

    Raises:
        ValueError: ``replicate`` が負の場合、または ``config`` が基底シードを
            持たないストリーム (``PROBE``) を要求された場合。
    """
    return make_rng_for(_base_seed(config, stream), stream, replicate)


__all__ = ["SeedConfig", "SeedStream", "make_rng", "make_rng_for"]

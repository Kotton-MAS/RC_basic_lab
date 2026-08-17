"""乱数ストリーム分離のテスト (D-06)。"""

from __future__ import annotations

import dataclasses

import pytest

from rc_basics_lab.seeds import SeedConfig, SeedStream, make_rng

_N_BYTES = 64


def _fingerprint(config: SeedConfig, stream: SeedStream, replicate: int = 0) -> bytes:
    """ストリームが生成する乱数列の先頭バイト列。"""
    return make_rng(config, stream, replicate).bytes(_N_BYTES)


def test_streams_are_independent() -> None:
    """1ストリームのシードだけ変えると、そのストリームだけが変わる (D-06)。

    3ストリームすべてについて対称に検証する。
    """
    base = SeedConfig()
    for changed in SeedStream:
        field_name = changed.value
        current: int = getattr(base, field_name)
        modified = dataclasses.replace(base, **{field_name: current + 1000})
        for stream in SeedStream:
            before = _fingerprint(base, stream)
            after = _fingerprint(modified, stream)
            if stream is changed:
                assert before != after, f"{field_name} を変えたのに {stream} が不変"
            else:
                assert before == after, (
                    f"{field_name} を変えたら無関係の {stream} が変化した"
                )


def test_streams_differ_from_each_other_under_identical_seeds() -> None:
    """3ストリームに同じ整数を与えても、乱数列は互いに異なる (spawn_key 分離)。"""
    config = SeedConfig(reservoir=7, task=7, split=7)
    fingerprints = {stream: _fingerprint(config, stream) for stream in SeedStream}
    assert len(set(fingerprints.values())) == len(SeedStream)


def test_replicates_are_distinct() -> None:
    config = SeedConfig()
    fingerprints = {
        replicate: _fingerprint(config, SeedStream.RESERVOIR, replicate)
        for replicate in range(5)
    }
    assert len(set(fingerprints.values())) == 5


def test_is_reproducible() -> None:
    config = SeedConfig()
    assert _fingerprint(config, SeedStream.TASK, 3) == _fingerprint(
        config, SeedStream.TASK, 3
    )


def test_negative_replicate_raises() -> None:
    with pytest.raises(ValueError, match="replicate"):
        make_rng(SeedConfig(), SeedStream.TASK, -1)

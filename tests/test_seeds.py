"""乱数ストリーム分離のテスト (D-06 / D-14)。"""

from __future__ import annotations

import dataclasses

import pytest

from rc_basics_lab.seeds import SeedConfig, SeedStream, make_rng, make_rng_for

_N_BYTES = 64


def _fingerprint(config: SeedConfig, stream: SeedStream, replicate: int = 0) -> bytes:
    """ストリームが生成する乱数列の先頭バイト列。"""
    return make_rng(config, stream, replicate).bytes(_N_BYTES)


def _seed_config_streams() -> tuple[tuple[str, SeedStream], ...]:
    """``SeedConfig`` のフィールドと対応する ``SeedStream`` の対を列挙する。

    ``SeedStream`` 全体ではなく ``SeedConfig`` のフィールドを列挙の起点にする。
    ``SeedConfig`` が基底シードを持たないストリーム (``PROBE``, D-14) は
    ``getattr(config, ...)`` が定義できないためこの検査の対象にならない
    (``PROBE`` の独立性は
    ``test_probe_stream_is_independent_of_seed_config_streams`` が担保する)。

    フィールド名と ``SeedStream`` の値は1対1に対応させる規約なので、対応の無い
    フィールドを足すとここで ``KeyError`` になる (結合を緩めない)。
    """
    by_value = {stream.value: stream for stream in SeedStream}
    return tuple(
        (item.name, by_value[item.name]) for item in dataclasses.fields(SeedConfig)
    )


def test_seed_config_fields_map_one_to_one_onto_streams() -> None:
    """``SeedConfig`` の全フィールドが ``SeedStream`` の値と対応する。

    ``test_streams_are_independent`` の列挙は ``SeedConfig`` のフィールドを
    起点にしているため、対応が崩れると独立性の検査が静かに縮む。
    """
    pairs = _seed_config_streams()
    assert len(pairs) == len(dataclasses.fields(SeedConfig))
    assert {name for name, _ in pairs} == {stream.value for _, stream in pairs}


def test_streams_are_independent() -> None:
    """1ストリームのシードだけ変えると、そのストリームだけが変わる (D-06)。

    ``SeedConfig`` が持つ全ストリームについて対称に検証する。
    """
    base = SeedConfig()
    pairs = _seed_config_streams()
    for field_name, changed in pairs:
        current: int = getattr(base, field_name)
        modified = dataclasses.replace(base, **{field_name: current + 1000})
        for _, stream in pairs:
            before = _fingerprint(base, stream)
            after = _fingerprint(modified, stream)
            if stream is changed:
                assert before != after, f"{field_name} を変えたのに {stream} が不変"
            else:
                assert before == after, (
                    f"{field_name} を変えたら無関係の {stream} が変化した"
                )


def test_probe_stream_is_independent_of_seed_config_streams() -> None:
    """``PROBE`` は 01 の3ストリームと独立で、``SeedConfig`` には現れない (D-14)。

    ESP 判定の頑健性は「同じ重み・同じ入力で初期状態対だけ振ったときに判定が
    変わらないか」で測るため、初期状態は reservoir / drive と独立に振れる必要が
    ある。一方 ``SeedConfig`` に ``probe`` を足すと 01 の配線テストの被覆が
    破れる (D-13 と同じ理由)。この2つを同時に満たしていることを固定する。
    """
    config = SeedConfig()
    pairs = _seed_config_streams()

    # 1. SeedConfig は probe を持たない (持たせた瞬間に 01 の被覆が破れる)
    assert "probe" not in {item.name for item in dataclasses.fields(SeedConfig)}
    with pytest.raises(ValueError, match="PROBE"):
        make_rng(config, SeedStream.PROBE, 0)

    # 2. probe の基底シードを振ると probe の乱数列だけが変わる
    probe_before = make_rng_for(3, SeedStream.PROBE, 0).bytes(_N_BYTES)
    probe_after = make_rng_for(1003, SeedStream.PROBE, 0).bytes(_N_BYTES)
    assert probe_before != probe_after
    # 01 側は probe をそもそも参照しないので、値に関わらずバイト単位で不変
    for _, stream in pairs:
        assert _fingerprint(config, stream) == make_rng_for(
            getattr(config, stream.value), stream, 0
        ).bytes(_N_BYTES)

    # 3. 同じ基底シードでも probe は 01 の各ストリームと別の列を出す
    #    (spawn_key の分離。ここが崩れると「初期状態を振ったら重みも変わる」)
    for name, stream in pairs:
        base_seed: int = getattr(config, name)
        assert make_rng_for(base_seed, SeedStream.PROBE, 0).bytes(
            _N_BYTES
        ) != _fingerprint(config, stream)


def test_streams_differ_from_each_other_under_identical_seeds() -> None:
    """4ストリームに同じ整数を与えても、乱数列は互いに異なる (spawn_key 分離)。"""
    fingerprints = {
        stream: make_rng_for(7, stream, 0).bytes(_N_BYTES) for stream in SeedStream
    }
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


def test_make_rng_for_matches_make_rng() -> None:
    """``make_rng`` は ``make_rng_for`` への委譲であり、乱数列は不変 (01 の再現性)。"""
    config = SeedConfig(reservoir=5, task=6, split=7)
    for name, stream in _seed_config_streams():
        for replicate in range(3):
            assert _fingerprint(config, stream, replicate) == make_rng_for(
                getattr(config, name), stream, replicate
            ).bytes(_N_BYTES)


def test_negative_replicate_raises() -> None:
    with pytest.raises(ValueError, match="replicate"):
        make_rng(SeedConfig(), SeedStream.TASK, -1)


def test_negative_replicate_raises_for_make_rng_for() -> None:
    with pytest.raises(ValueError, match="replicate"):
        make_rng_for(0, SeedStream.PROBE, -1)

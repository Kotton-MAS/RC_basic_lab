"""``kind`` によるリザバーモデルの選択の検査.

**「足せるはず」ではなく、足したものが設定から選ばれることを測る。**
union が1要素のうちは選択の余地が無いので、**テスト側で2つ目を定義して**
判別の経路そのものを動かす。ここを ``ESNConfig`` だけで済ませると、
2つ目を足した日に初めて壊れていたことが分かる。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from rc_basics_lab.config import load_config
from rc_basics_lab.config._common import ConfigError, _build, _coerce
from rc_basics_lab.overrides import KIND_KEY
from rc_basics_lab.reservoir.esn import ESNConfig

CONFIG = Path("experiments/01_what_is_rc/config.yaml")


@dataclass(frozen=True, slots=True)
class _RingConfig:
    """テスト用の2つ目のモデル設定 (本番には無い)。"""

    KIND: ClassVar[str] = "ring"

    n_units: int = 8
    gain: float = 0.5


type _TestReservoirConfig = ESNConfig | _RingConfig


# --- union が1要素のうちの挙動 (本番の 01) ----------------------------------


def test_the_kind_may_be_omitted() -> None:
    """``kind`` を書かなければ既定 (ESN) になる。既存の YAML はそのまま通る。"""
    config = load_config(CONFIG)
    assert isinstance(config.tasks[0].reservoir, ESNConfig)


def test_the_default_kind_can_be_written_explicitly() -> None:
    """``kind: esn`` と明示できる。

    **2つ目を足した日に YAML を書き換えずに済む**ようにするための経路で、
    union が1要素のうちから通しておく。
    """
    config = load_config(CONFIG, overrides=["tasks.mackey_glass.reservoir.kind=esn"])
    assert isinstance(config.tasks[0].reservoir, ESNConfig)
    assert config.tasks[0].reservoir.n_units == 200


def test_an_unknown_kind_is_rejected_with_the_known_names() -> None:
    """未知の ``kind`` は既知の名前を添えて落とす (黙って既定に落ちない)。"""
    with pytest.raises(ConfigError, match="未知の kind です"):
        load_config(
            CONFIG, overrides=["tasks.mackey_glass.reservoir.kind=no_such_model"]
        )


@pytest.mark.parametrize(("preset", "kind"), [("ring", "ring"), ("deep", "deep_esn")])
def test_every_registered_kind_is_selectable_from_a_preset(
    preset: str, kind: str
) -> None:
    """登録した全モデルがプリセットから選べる。

    ここが落ちたら「registry には在るが設定から選べない」状態である。
    """
    config = load_config(CONFIG, preset=CONFIG.parent / "presets" / f"{preset}.yaml")
    assert kind == type(config.tasks[0].reservoir).KIND
    assert kind == type(config.tasks[1].reservoir).KIND


def test_changing_the_kind_replaces_the_whole_section() -> None:
    """``kind`` が変わるセクションは**丸ごと置き換わる**。

    混ぜると前のモデル固有のキー (ESN の ``density`` / ``activation``) が
    残って必ず未知キーになり、「``kind`` を書いたのにモデルを替えられない」
    状態になる (実測でそうなった)。
    """
    config = load_config(CONFIG, preset=CONFIG.parent / "presets" / "ring.yaml")
    assert not hasattr(config.tasks[0].reservoir, "density"), (
        "ESN 固有のキーが残っています (セクションが混ざりました)"
    )


def test_changing_the_kind_with_set_alone_explains_why_it_fails() -> None:
    """``--set`` で ``kind`` だけ替えたら、**なぜ駄目か**を言って落とす。

    セクション単位の変更なので1つの値では替えられない。黙って未知キーと
    だけ言うと、利用者は自分の綴りを疑うことになる。
    """
    with pytest.raises(ConfigError, match="セクションを丸ごと"):
        load_config(CONFIG, overrides=["tasks.mackey_glass.reservoir.kind=ring"])


def test_the_kind_never_reaches_the_artifacts() -> None:
    """``KIND`` は ``dataclasses.asdict`` に現れない。

    フィールドにすると ``meta.json`` が変わり、既存の成果物の指紋が壊れる。
    判別子は「どの型を作るか」の情報であって、その型の設定値ではない。
    """
    dumped = dataclasses.asdict(ESNConfig())
    assert KIND_KEY not in dumped
    assert "KIND" not in dumped
    assert "KIND" not in {field.name for field in dataclasses.fields(ESNConfig)}


# --- union が2要素になったときの挙動 (テスト側で先に確かめる) -----------------


def test_a_second_model_is_selected_by_its_kind() -> None:
    """2つ目のモデルが ``kind`` で選ばれる (**これが拡張点の意味**)。"""
    built = _coerce(
        {"kind": "ring", "n_units": 5, "gain": 0.25},
        _TestReservoirConfig,
        "test",
    )
    assert isinstance(built, _RingConfig)
    assert (built.n_units, built.gain) == (5, 0.25)


def test_the_first_member_is_the_default_in_a_union() -> None:
    """``kind`` を省いたら union の**先頭**を作る。

    既定を並び順で表すので、``ReservoirConfig`` に2つ目を足しても既存の
    YAML の意味は変わらない (先頭が ``ESNConfig`` である限り)。
    """
    built = _coerce({"n_units": 32}, _TestReservoirConfig, "test")
    assert isinstance(built, ESNConfig)
    assert built.n_units == 32


def test_an_unknown_kind_in_a_union_lists_every_known_name() -> None:
    """union でも既知の名前を全部添えて落とす。"""
    with pytest.raises(ConfigError, match="esn, ring"):
        _coerce({"kind": "nope"}, _TestReservoirConfig, "test")


def test_a_member_without_a_kind_is_rejected() -> None:
    """``KIND`` を名乗らない要素が union にあったら落とす。

    名乗らない要素があると、``kind`` で選べない型が黙って混ざる。
    """

    @dataclass(frozen=True, slots=True)
    class _Nameless:
        n_units: int = 1

    with pytest.raises(ConfigError, match="KIND を名乗らない"):
        _coerce({"kind": "esn"}, ESNConfig | _Nameless, "test")


def test_the_unknown_key_check_still_applies_to_the_selected_model() -> None:
    """選ばれたモデルの未知キーは従来どおり落ちる (D-09)。

    ``kind`` を取り除く処理が、他のキーの検査を素通しさせていないこと。
    """
    with pytest.raises(ConfigError, match="未知のキーです: gian"):
        _build(_RingConfig, {"kind": "ring", "gian": 1.0}, "test")

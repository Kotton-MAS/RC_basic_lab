"""``--set`` と ``--preset`` の検査 (試行環境).

**「上書きできる」だけでなく「間違いが黙って通らない」ことを測る。**
設定を振る仕組みの価値は、振ったことが確実に効くことにあるので、
効かない経路が1つでもあると仕組みごと信用できなくなる (D-09 と同じ理屈)。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rc_basics_lab.config import ExperimentConfig, load_config
from rc_basics_lab.config._common import ConfigError
from rc_basics_lab.overrides import (
    OverrideError,
    apply_overrides,
    parse_override,
)

CONFIG = Path("experiments/01_what_is_rc/config.yaml")
QUICK = Path("experiments/01_what_is_rc/presets/quick.yaml")


@pytest.mark.parametrize(
    ("text", "expected_path", "expected_value"),
    [
        ("a=1", ("a",), 1),
        ("a.b=1", ("a", "b"), 1),
        ("a.b=0.5", ("a", "b"), 0.5),
        ("a=tanh", ("a",), "tanh"),
        ("a=true", ("a",), True),
        ("a=[1, 2]", ("a",), [1, 2]),
        ("a=1.0e-8", ("a",), 1e-8),
    ],
)
def test_the_value_is_read_as_yaml(
    text: str, expected_path: tuple[str, ...], expected_value: object
) -> None:
    """値は YAML として解釈する。

    独自の変換規則を作らないので、「ファイルに書いたとき」と「CLI で振った
    とき」で同じ文字列が同じ値になる。ここがずれると、YAML では通る設定が
    ``--set`` では別の型になる。
    """
    path, value = parse_override(text)
    assert path == expected_path
    assert value == expected_value


@pytest.mark.parametrize("text", ["novalue", "=1", "a..b=1", "a.=1"])
def test_a_malformed_override_is_rejected(text: str) -> None:
    """書式が壊れていたら落とす (黙って無視しない)。"""
    with pytest.raises(OverrideError):
        parse_override(text)


@pytest.mark.parametrize(
    ("text", "suggested"),
    [("a=1e-8", "1.0e-8"), ("a=1E-8", "1.0E-8"), ("a=-2e3", "-2.0e3")],
)
def test_yaml_hostile_scientific_notation_is_rejected_with_the_fix(
    text: str, suggested: str
) -> None:
    """``1e-8`` は**書き方まで示して**落とす。

    YAML 1.1 は仮数に小数点が無い指数表記を数値と認めず、文字列にする (実測)。
    そのまま通すと後段で「数値が必要です: '1e-8'」という、原因の分からない
    エラーになる。利用者は数値のつもりで打っているので、直し方を出す。
    """
    with pytest.raises(OverrideError, match=suggested):
        parse_override(text)


@pytest.mark.parametrize("text", ["a=tanh", "a=zscore", "a=01_what_is_rc"])
def test_ordinary_strings_are_not_mistaken_for_numbers(text: str) -> None:
    """普通の文字列は指数表記の検出に当たらない。"""
    _, value = parse_override(text)
    assert isinstance(value, str)


def test_an_override_does_not_mutate_the_original_mapping() -> None:
    """元のマッピングを書き換えない (同じ設定を2回読む経路があるため)。"""
    raw: dict[str, object] = {"a": {"b": 1}}
    result = apply_overrides(raw, ["a.b=2"])
    assert result["a"] == {"b": 2}
    assert raw["a"] == {"b": 1}, "元のマッピングが書き換わっています"


def test_a_missing_parent_path_is_rejected() -> None:
    """**親のパスが無ければ落とす。**

    黙って新しいキーを作ると、``esn_mackey_glas.n_units`` (親のタイプミス) が
    「設定したのに効いていない」実験になる。葉のタイプミスは ``_build`` の
    未知キー検査が捕まえるが、親は作られてしまうのでここで塞ぐ。
    """
    with pytest.raises(OverrideError, match="という設定はありません"):
        apply_overrides({"a": {"b": 1}}, ["z.b=2"])


def test_a_leaf_typo_is_caught_by_the_unknown_key_check() -> None:
    """葉のタイプミスは既存の未知キー検査 (D-09) が捕まえる。

    上書き専用の検査を別に書かないので、YAML 側と CLI 側で厳しさが割れない。
    """
    with pytest.raises(ConfigError, match="未知のキーです: n_unit"):
        load_config(CONFIG, overrides=["tasks.mackey_glass.reservoir.n_unit=50"])


def test_an_override_reaches_the_built_config() -> None:
    """``--set`` が実際に設定へ届く (D-13: 効かないフィールドは飾りである)。"""
    base = load_config(CONFIG)
    changed = load_config(CONFIG, overrides=["tasks.mackey_glass.reservoir.n_units=17"])
    assert base.tasks[0].reservoir.n_units != 17
    assert changed.tasks[0].reservoir.n_units == 17


def test_overrides_apply_left_to_right() -> None:
    """同じキーを2回指定したら右が勝つ。"""
    config = load_config(
        CONFIG,
        overrides=[
            "tasks.mackey_glass.reservoir.n_units=17",
            "tasks.mackey_glass.reservoir.n_units=23",
        ],
    )
    assert config.tasks[0].reservoir.n_units == 23


def test_a_preset_is_merged_under_the_overrides() -> None:
    """適用の順は 本体 -> プリセット -> ``--set``。右が勝つ。"""
    with_preset = load_config(CONFIG, preset=QUICK)
    both = load_config(CONFIG, preset=QUICK, overrides=["n_replicates=9"])
    assert with_preset.n_replicates == 2, "プリセットが効いていません"
    assert both.n_replicates == 9, "--set がプリセットに負けています"


def test_a_preset_only_changes_what_it_names() -> None:
    """プリセットは差分だけを書く (深くかぶせる)。

    ``esn_mackey_glass`` の ``n_units`` だけを書いたプリセットが、同じ
    セクションの ``spectral_radius`` を消してはいけない。
    """
    base = load_config(CONFIG)
    quick = load_config(CONFIG, preset=QUICK)
    assert quick.tasks[0].reservoir.n_units != base.tasks[0].reservoir.n_units
    assert (
        quick.tasks[0].reservoir.spectral_radius
        == base.tasks[0].reservoir.spectral_radius
    ), "プリセットが書いていないフィールドが消えています"


def test_every_experiment_has_a_quick_preset() -> None:
    """5実験すべてに ``quick`` がある。

    1つでも欠けると「速く回す」導線が実験ごとに割れる。
    """
    missing = [
        directory.name
        for directory in sorted(Path("experiments").iterdir())
        if directory.is_dir() and any(directory.glob("run*.py"))
        if not (directory / "presets" / "quick.yaml").is_file()
    ]
    assert not missing, f"quick プリセットが無い実験があります: {missing}"


def test_the_quick_preset_builds_a_valid_config() -> None:
    """``quick`` が実際に設定として組み立つ (未知キーが無い)。

    プリセットは本体と同じ未知キー検査を通るので、本体のキー名が変わったら
    ここが落ちる —— プリセットだけ古いまま残る事故を塞ぐ。
    """
    config = load_config(CONFIG, preset=QUICK)
    assert isinstance(config, ExperimentConfig)
    assert config.n_replicates == 2

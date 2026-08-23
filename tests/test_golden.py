"""成果物のバイト不変検査 (ゴールデン).

リファクタリングの合否判定はここ 1 本で行う。**コードを読まず、出力だけを見る**
のが役割で、行の値を検証する他のテスト群 (構造を動かすと一緒に書き換わる) とは
目的が違う。

機構と除外規則は ``tests/golden_support.py`` の docstring にある。基準値の更新は
``make golden-update``。

数値 (CSV / meta.json) と図 (PNG) を別のテストに分けてあるのは、期待してよい
一致の条件が違うため。数値は BLAS 実装まで、図はさらに matplotlib と CJK
フォントまで一致していないと同じにならない。片方だけ黙って飛ばすと「全部
照合できた」と読めてしまうので、飛ばすときは飛ばした側が skip として出る。
"""

from __future__ import annotations

import pytest
from golden_support import (
    CASES,
    CONFIG_DIR,
    FIGURE_FINGERPRINT_KEYS,
    NUMERIC_FINGERPRINT_KEYS,
    GoldenCase,
    environment_fingerprint,
    load_manifest,
    manifest_digests,
    manifest_fingerprint,
    mismatched_keys,
    run_case,
)

_CASE_IDS = [case.name for case in CASES]

FIGURE_SUFFIX = ".png"


def _skip_unless_fingerprint_matches(keys: tuple[str, ...]) -> None:
    """基準値を取った環境と食い違っていたら skip する。"""
    mismatched = mismatched_keys(
        manifest_fingerprint(load_manifest()), environment_fingerprint(), keys
    )
    if mismatched:
        pytest.skip(
            "基準値を取った環境と異なるため照合しません: "
            + ", ".join(sorted(mismatched))
        )


@pytest.fixture(scope="module")
def digests_by_case(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, dict[str, str]]:
    """全ケースを 1 度だけ実行し、``ケース名 -> (相対パス -> ダイジェスト)`` を返す。

    数値側と図側の 2 テストが同じ出力を見るため、実行はモジュール内で 1 回に
    まとめる (4 実験を 2 度回すと所要が倍になる)。
    """
    root = tmp_path_factory.mktemp("golden")
    return {case.name: run_case(case, root / case.name) for case in CASES}


def test_every_case_has_a_config() -> None:
    """ケースの設定ファイルが実在する。"""
    for case in CASES:
        assert (CONFIG_DIR / case.config).is_file(), case.config


def test_manifest_lists_exactly_the_registered_cases() -> None:
    """マニフェストとケース定義がずれていない。

    ケースを足して基準値を取り忘れると落ちる。
    """
    cases = load_manifest()["cases"]
    assert isinstance(cases, dict)
    assert sorted(cases) == sorted(_CASE_IDS)


@pytest.mark.parametrize("case", CASES, ids=_CASE_IDS)
def test_artifact_set_is_unchanged(
    case: GoldenCase, digests_by_case: dict[str, dict[str, str]]
) -> None:
    """生成されるファイルの顔ぶれが変わっていない。

    中身の照合と違い、環境に依存しないので指紋によらず必ず走る。
    """
    expected = manifest_digests(load_manifest(), case.name)
    assert sorted(digests_by_case[case.name]) == sorted(expected)


@pytest.mark.parametrize("case", CASES, ids=_CASE_IDS)
def test_data_artifacts_are_byte_identical(
    case: GoldenCase, digests_by_case: dict[str, dict[str, str]]
) -> None:
    """CSV と meta.json が基準値とバイト単位で一致する (実測 wall time を除く)。"""
    _skip_unless_fingerprint_matches(NUMERIC_FINGERPRINT_KEYS)
    expected = manifest_digests(load_manifest(), case.name)
    actual = digests_by_case[case.name]
    changed = sorted(
        name
        for name, value in actual.items()
        if not name.endswith(FIGURE_SUFFIX) and expected.get(name) != value
    )
    assert not changed, f"成果物が変化しました: {', '.join(changed)}"


@pytest.mark.parametrize("case", CASES, ids=_CASE_IDS)
def test_figures_are_byte_identical(
    case: GoldenCase, digests_by_case: dict[str, dict[str, str]]
) -> None:
    """PNG が基準値とバイト単位で一致する。"""
    _skip_unless_fingerprint_matches(FIGURE_FINGERPRINT_KEYS)
    expected = manifest_digests(load_manifest(), case.name)
    actual = digests_by_case[case.name]
    changed = sorted(
        name
        for name, value in actual.items()
        if name.endswith(FIGURE_SUFFIX) and expected.get(name) != value
    )
    assert not changed, f"図が変化しました: {', '.join(changed)}"

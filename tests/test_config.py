"""設定層のテスト (D-09: 未知キーは黙って無視しない)。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from rc_basics_lab.config import (
    DEFAULT_ALPHA_GRID,
    ConfigError,
    ExperimentConfig,
    MackeyGlassTask,
    load_config,
    require_task,
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_unknown_key_raises(tmp_path: Path) -> None:
    """YAML に未定義キーを混ぜると ConfigError (D-09 の guard test)。"""
    path = _write(tmp_path, "n_replicates: 3\nn_replicate: 3\n")
    with pytest.raises(ConfigError, match="n_replicate"):
        load_config(path)


def test_unknown_nested_key_raises(tmp_path: Path) -> None:
    """ネストしたセクション内のタイプミスも検出する。"""
    path = _write(
        tmp_path, "tasks:\n  - kind: mackey_glass\n    params:\n      taus: 17.0\n"
    )
    with pytest.raises(ConfigError, match="mackey_glass"):
        load_config(path)


def test_per_method_alpha_grid_key_is_rejected(tmp_path: Path) -> None:
    """手法別 alpha 格子キーは未知キーとして弾かれる (D-04 の前提)。"""
    path = _write(tmp_path, "ridge:\n  alpha_grid_esn: [1.0]\n")
    with pytest.raises(ConfigError, match="alpha_grid_esn"):
        load_config(path)


def test_empty_yaml_gives_defaults(tmp_path: Path) -> None:
    path = _write(tmp_path, "")
    assert load_config(path) == ExperimentConfig()


def test_values_override_defaults(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "n_replicates: 3\nseeds:\n  reservoir: 11\n"
        "tasks:\n  - kind: mackey_glass\n    params:\n      tau: 30.0\n",
    )
    config = load_config(path)
    assert config.n_replicates == 3
    assert config.seeds.reservoir == 11
    task = require_task(config, MackeyGlassTask, "テスト")
    assert task.params.tau == pytest.approx(30.0)
    # 未指定のフィールドは既定値のまま
    assert config.seeds.task == ExperimentConfig().seeds.task


def test_alpha_grid_is_parsed_as_float_tuple(tmp_path: Path) -> None:
    path = _write(tmp_path, "ridge:\n  alpha_grid: [1, 0.1]\n")
    grid = load_config(path).ridge.alpha_grid
    assert grid == (1.0, 0.1)
    assert all(isinstance(value, float) for value in grid)


def test_default_alpha_grid_is_exactly_eleven_decade_literals() -> None:
    """既定の alpha 格子が 10 のべき乗の literal と**厳密に**一致する (D-114)。

    ``pytest.approx`` ではなく厳密比較にする。かつては
    ``np.logspace(-8, 2, 11)`` で計算しており、``power(10.0, -5.0)`` の最下位
    ビットが libm 実装に依存した (macOS arm64: ``1e-05`` / Linux x86_64:
    ``9.999999999999999e-06``)。本番 YAML は literal を書くので、Linux でだけ
    既定と YAML が食い違い ``test_production_config_matches_the_committed_yaml``
    が CI で落ちていた。
    """
    assert DEFAULT_ALPHA_GRID == (
        1e-08,
        1e-07,
        1e-06,
        1e-05,
        1e-04,
        1e-03,
        1e-02,
        1e-01,
        1.0,
        10.0,
        100.0,
    )


def test_config_defaults_are_not_computed_with_floating_point_powers() -> None:
    """設定の既定値を ``np.logspace`` で作らない (D-114)。

    値の厳密比較 (上のテスト) はプラットフォーム依存の再発を**そのプラット
    フォームでしか**捕まえない —— CI は macOS で回る (D-113) ので、Linux で
    だけ壊れる書き方に戻されても緑のまま通ってしまう。ここは書き方そのものを
    見るので、どのプラットフォームでも落ちる。
    """
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "rc_basics_lab"
        / "config"
        / "experiment01.py"
    ).read_text(encoding="utf-8")
    offending = [
        line
        for line in source.splitlines()
        if "np.logspace" in line and not line.lstrip().startswith(("#", "*", '"'))
    ]
    assert not offending, (
        "設定の既定値を np.logspace で計算しています (D-114)。"
        f" literal で書いてください: {offending}"
    )


def test_type_mismatch_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "n_replicates: 3.5\n")
    with pytest.raises(ConfigError, match="整数"):
        load_config(path)


def test_scalar_where_section_expected_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "seeds: 3\n")
    with pytest.raises(ConfigError, match="マッピング"):
        load_config(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="見つかりません"):
        load_config(tmp_path / "absent.yaml")


def test_broken_yaml_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "n_replicates: [1, 2\n")
    with pytest.raises(ConfigError, match="YAML"):
        load_config(path)


@pytest.mark.parametrize(
    ("yaml_text", "match"),
    [
        # bool -> int (D-09 隣接: _coerce_scalar が isinstance(value, bool) を
        # 明示的に弾く分岐)。YAML の true/false は Python では int のサブクラス。
        pytest.param("n_replicates: true\n", "整数が必要です", id="bool_to_int_top"),
        pytest.param(
            "tasks:\n  - kind: mackey_glass\n    params:\n      exponent: false\n",
            "整数が必要です",
            id="bool_to_int_nested",
        ),
        # bool -> float
        pytest.param(
            "tasks:\n  - kind: mackey_glass\n    params:\n      tau: true\n",
            "数値が必要です",
            id="bool_to_float",
        ),
        pytest.param(
            "split:\n  train_ratio: false\n",
            "数値が必要です",
            id="bool_to_float_nested",
        ),
        # str -> int / str -> float (数値らしい文字列でも変換しない)
        pytest.param('n_replicates: "3"\n', "整数が必要です", id="str_to_int"),
        pytest.param(
            'tasks:\n  - kind: mackey_glass\n    params:\n      tau: "17.0"\n',
            "数値が必要です",
            id="str_to_float",
        ),
        # 非文字列 -> str
        pytest.param("name: 5\n", "文字列が必要です", id="int_to_str"),
        pytest.param("name: 5.0\n", "文字列が必要です", id="float_to_str"),
        pytest.param("name: true\n", "文字列が必要です", id="bool_to_str"),
        # tuple[float, ...] / tuple[int, ...] フィールドにスカラや文字列を渡す
        pytest.param(
            "ridge:\n  alpha_grid: 5\n", "リストが必要です", id="scalar_for_tuple"
        ),
        pytest.param(
            'ridge:\n  alpha_grid: "abc"\n',
            "リストが必要です",
            id="string_for_tuple",
        ),
        # tuple の要素型違反 (bool/float が int 要素・str 要素に混入)
        pytest.param(
            "ridge:\n  alpha_grid: [true, 1.0]\n",
            "数値が必要です",
            id="bool_element_in_float_tuple",
        ),
        pytest.param(
            "ridge:\n  n_lags_grid: [1.5, 2]\n",
            "整数が必要です",
            id="float_element_in_int_tuple",
        ),
        pytest.param(
            "ridge:\n  n_lags_grid: [true, 2]\n",
            "整数が必要です",
            id="bool_element_in_int_tuple",
        ),
    ],
)
def test_coerce_scalar_rejects_loose_conversions(
    tmp_path: Path, yaml_text: str, match: str
) -> None:
    """_coerce_scalar / _coerce_tuple が拒否する緩い変換を網羅する。

    bool は Python では int のサブクラスなので、明示的に isinstance(bool) で
    弾かないと ``n_replicates: true`` のような YAML が黙って通ってしまう。
    D-09 (未知キーは ConfigError で即座に失敗させる) と隣接する規律だが、
    このガード自体はこれまでテストが無く、退行しても何も落ちなかった。
    """
    path = _write(tmp_path, yaml_text)
    with pytest.raises(ConfigError, match=match):
        load_config(path)


def test_config_is_frozen() -> None:
    config = ExperimentConfig()
    # 静的にも代入不可なので、実行時の凍結確認には setattr を使う
    with pytest.raises(FrozenInstanceError):
        setattr(config, "n_replicates", 7)  # noqa: B010  # frozen 検証のため意図的

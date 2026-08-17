"""設定層のテスト (D-09: 未知キーは黙って無視しない)。"""

from __future__ import annotations

from pathlib import Path

import pytest

from rc_basics_lab.config import (
    DEFAULT_ALPHA_GRID,
    ConfigError,
    ExperimentConfig,
    load_config,
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
    path = _write(tmp_path, "mackey_glass:\n  taus: 17.0\n")
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
        "n_replicates: 3\nseeds:\n  reservoir: 11\nmackey_glass:\n  tau: 30.0\n",
    )
    config = load_config(path)
    assert config.n_replicates == 3
    assert config.seeds.reservoir == 11
    assert config.mackey_glass.tau == pytest.approx(30.0)
    # 未指定のフィールドは既定値のまま
    assert config.seeds.task == ExperimentConfig().seeds.task


def test_alpha_grid_is_parsed_as_float_tuple(tmp_path: Path) -> None:
    path = _write(tmp_path, "ridge:\n  alpha_grid: [1, 0.1]\n")
    grid = load_config(path).ridge.alpha_grid
    assert grid == (1.0, 0.1)
    assert all(isinstance(value, float) for value in grid)


def test_default_alpha_grid_is_logspace_of_eleven_points() -> None:
    assert len(DEFAULT_ALPHA_GRID) == 11
    assert DEFAULT_ALPHA_GRID[0] == pytest.approx(1e-8)
    assert DEFAULT_ALPHA_GRID[-1] == pytest.approx(1e2)


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


def test_config_is_frozen() -> None:
    config = ExperimentConfig()
    with pytest.raises(AttributeError):
        config.n_replicates = 7  # type: ignore[misc]  # frozen 検証のため意図的

"""実行メタ情報のテスト。"""

from __future__ import annotations

import json
import subprocess

import pytest

from rc_basics_lab import meta
from rc_basics_lab.config import ExperimentConfig


def test_meta_is_json_serializable() -> None:
    """meta.json にそのまま書ける形で返り、commit キーを含む。"""
    payload = meta.collect_meta(ExperimentConfig())
    assert "commit" in payload
    dumped = json.dumps(payload, ensure_ascii=False)
    assert json.loads(dumped)["commit"] == payload["commit"]


def test_meta_contains_versions_and_config() -> None:
    config = ExperimentConfig(n_replicates=3)
    payload = meta.collect_meta(config)
    for key in ("numpy_version", "scipy_version", "timestamp_utc", "seeds", "config"):
        assert key in payload
    config_dump = payload["config"]
    assert isinstance(config_dump, dict)
    assert config_dump["n_replicates"] == 3


def test_commit_falls_back_to_unknown_when_git_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """git 取得に失敗しても例外を投げず "unknown" になる。"""

    def _raise(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert meta.collect_meta(ExperimentConfig())["commit"] == meta.UNKNOWN


def test_collect_meta_for_rejects_a_non_dataclass_config() -> None:
    """config が dataclass インスタンスでなければ TypeError (docstring の Raises)。"""
    with pytest.raises(TypeError, match="config"):
        meta.collect_meta_for(object(), ExperimentConfig().seeds)


def test_collect_meta_for_rejects_a_non_dataclass_seeds() -> None:
    """seeds が dataclass インスタンスでなければ TypeError (docstring の Raises)。"""
    with pytest.raises(TypeError, match="seeds"):
        meta.collect_meta_for(ExperimentConfig(), object())


def test_collect_meta_for_rejects_a_dataclass_type_instead_of_an_instance() -> None:
    """dataclass 型そのもの (インスタンスでない) を渡すのも誤用として TypeError。"""
    with pytest.raises(TypeError, match="config"):
        meta.collect_meta_for(ExperimentConfig, ExperimentConfig().seeds)

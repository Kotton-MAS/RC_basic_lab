"""実行メタ情報の収集.

``results/meta.json`` に落とすための情報を1か所で組み立てる。再現に必要な
「いつ・どのコミットで・どのライブラリ版で・どの設定で」を JSON 化可能な
プレーンな値だけで返す。
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import platform
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import scipy

from rc_basics_lab import __version__
from rc_basics_lab.config import ExperimentConfig

UNKNOWN = "unknown"
"""git 情報を取得できなかったときのフォールバック値。"""

_GIT_TIMEOUT_S = 5.0


def _git_commit() -> str:
    """HEAD のコミットハッシュ。取得できなければ ``"unknown"`` を返す。

    git が無い / リポジトリ外 / タイムアウトのいずれでも例外を投げない。
    メタ情報の収集で実験本体を落とさないため。
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    if completed.returncode != 0:
        return UNKNOWN
    commit = completed.stdout.strip()
    return commit or UNKNOWN


def _as_dict(value: object, label: str) -> dict[str, object]:
    """dataclass インスタンスをプレーンな dict にする (``Any`` を書かずに)。"""
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        raise TypeError(f"{label} は dataclass のインスタンスが必要です: {value!r}")
    return dataclasses.asdict(value)


def collect_meta_for(config: object, seeds: object) -> dict[str, object]:
    """任意の実験設定 dataclass から実行メタ情報を組み立てる。

    実験ごとに設定クラスは分かれる (D-13) が、「いつ・どのコミットで・どの
    ライブラリ版で・どの設定で」の集め方は1か所に置く。実験ごとに写経すると、
    片方だけ項目が欠けても何も落ちない。

    Args:
        config: 実験設定 dataclass (``ExperimentConfig`` / ``Esp02Config`` など)。
        seeds: シード設定 dataclass。``config`` の中にあるが、フィールド名が
            実験ごとに違う (01 は ``seeds``、02 も ``seeds`` だが型が別) ため
            明示的に受け取る。

    Raises:
        TypeError: ``config`` / ``seeds`` が dataclass インスタンスでない場合。
    """
    return {
        "commit": _git_commit(),
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "package_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "seeds": _as_dict(seeds, "seeds"),
        "config": _as_dict(config, "config"),
    }


def collect_meta(config: ExperimentConfig) -> dict[str, object]:
    """実行メタ情報を JSON 化可能な dict で返す (01 の ``ExperimentConfig`` 用)。

    ``collect_meta_for(config, config.seeds)`` への委譲。既存の呼び出しを
    壊さないため署名はそのまま残す。
    """
    return collect_meta_for(config, config.seeds)


__all__ = ["UNKNOWN", "collect_meta", "collect_meta_for"]

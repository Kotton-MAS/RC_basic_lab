"""GitHub Actions のワークフローが満たすべき条件 (D-113).

このリポジトリのガードは「成果物が1バイトも変わらないこと」で合否を出す
(D-74 / ``tests/test_artifact_invariance.py``) が、bit-exact なゴールデンは
原理的に**プラットフォームに固定される**。ゴールデンは macOS arm64 で採って
あり、``ubuntu-latest`` では BLAS / libm の実装差で下位桁が動く。

実測 (run 33303926712、``ubuntu-latest``):

- ``test_01_artifacts_are_unchanged``: ``0.03284851573645807`` が
  ``0.032848516130530726`` になる (9桁目以降)
- ``test_synthetic_source_values_match_a_known_seed_golden_case``: 先頭5点は
  ``atol=0`` で一致する (RNG は同一) のに全量の sha256 が一致しない

CI を ubuntu へ戻すと、この2件が**また静かに赤くなる**。赤が常態化した CI は
誰も見なくなるので、実際に 4 run すべてが失敗したまま 8 日間放置されていた。
ここで理由ごと固定する。
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "python-ci.yaml"
)

MACOS_PREFIX = "macos"
"""``runs-on`` に要求する接頭辞 (``macos-latest`` / ``macos-15`` など)。"""


def _ci_job() -> dict[str, object]:
    """ワークフローの ``ci`` ジョブ。"""
    loaded: object = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), WORKFLOW
    jobs = loaded.get("jobs")
    assert isinstance(jobs, dict), "jobs がありません"
    job = jobs.get("ci")
    assert isinstance(job, dict), "ci ジョブがありません"
    return cast("dict[str, object]", job)


def test_ci_runs_on_macos_because_the_goldens_are_platform_locked() -> None:
    """``runs-on`` が macOS であること (D-113)。

    ubuntu へ戻すなら、先に ``.claude/decisions.yaml`` の D-113 を見直すこと。
    「CI が赤いので許容誤差へ緩める」は、``test_artifact_invariance`` (D-74) が
    測っている価値そのものを消す操作である。
    """
    runs_on = _ci_job().get("runs-on")
    assert isinstance(runs_on, str), (
        f"runs-on が単一の文字列ではありません: {runs_on!r}"
    )
    assert runs_on.startswith(MACOS_PREFIX), (
        f"CI の runs-on が {runs_on!r} です。bit-exact なゴールデンは macOS arm64 で"
        " 採ってあり、他のプラットフォームでは BLAS / libm の差で下位桁が動きます"
        " (D-113)。戻すなら決定の見直しが先です"
    )


def test_ci_has_enough_time_for_the_full_suite() -> None:
    """``timeout-minutes`` が実測に対して余裕を持つこと。

    ubuntu では ``test`` だけで 184 秒だった。既定の 5 分では実験を含む
    スイートが途中で切られる。
    """
    timeout = _ci_job().get("timeout-minutes")
    assert isinstance(timeout, int), (
        f"timeout-minutes が整数ではありません: {timeout!r}"
    )
    assert timeout >= 10, f"timeout-minutes={timeout} は短すぎます"


def test_ci_delegates_the_checks_to_make_ci() -> None:
    """検証ロジックが ``make ci`` に一元化されていること。

    ローカル / Stop hook / CI が同じコマンドを呼ぶことで、検証内容の乖離を防ぐ。
    """
    steps = _ci_job().get("steps")
    assert isinstance(steps, list)
    commands = [
        step.get("run") for step in steps if isinstance(step, dict) and "run" in step
    ]
    assert "make ci" in commands, f"make ci を呼ぶステップがありません: {commands}"

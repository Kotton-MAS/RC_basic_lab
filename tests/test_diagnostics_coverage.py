"""どの実験が ``diagnostics.csv`` を出すかを固定する (D-133).

D-118 の rule は「診断が返すスカラは ``diagnostics.csv`` へ」と全称で書いてある。
実際に出しているのは 01 / 02 / 03 / 04 で、**05 だけ出していない** ——
05 は ``DiagnosticResult`` を返す診断を1本も呼んでいないからである
(ESN は回すが、測っているのは検知の成績であって内部状態ではない)。

「意図的に外した」のか「まだ配線していない」のかが記録から読めないと、次に見た人が
同じ調査をやり直す。ここで**理由ごと固定する**: 05 に診断を足したら、この検査が
「配線も足してください」と言って赤くなる。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from rc_basics_lab.experiment.catalog import CATALOG, ExperimentSpec
from rc_basics_lab.experiment.diagnostics_rows import DIAGNOSTICS_CSV

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "rc_basics_lab"

WITHOUT_DIAGNOSTICS: frozenset[str] = frozenset({"05"})
"""``diagnostics.csv`` を出さない実験。

**05 は診断を1本も呼んでいない。** 検知の成績 (PR-AUC など) は
``metrics_detection`` が返す値で、``DiagnosticResult`` ではない。出す行が無いのに
空の CSV を1枚増やすと、指紋とゴールデンの維持コストだけが増えて情報は増えない。
"""

ANOMALY_MODULES = ("experiment/anomaly.py", "experiment/anomaly_score.py")


@pytest.mark.parametrize("spec", CATALOG, ids=lambda s: s.number)
def test_the_declared_artifacts_match_the_diagnostics_policy(
    spec: ExperimentSpec,
) -> None:
    """``diagnostics.csv`` を出す実験と出さない実験が宣言どおりであること。"""
    declared = DIAGNOSTICS_CSV in spec.artifacts
    assert declared is (spec.number not in WITHOUT_DIAGNOSTICS), (
        f"実験 {spec.number} の {DIAGNOSTICS_CSV} の扱いが宣言と違います"
    )


@pytest.mark.parametrize("spec", CATALOG, ids=lambda s: s.number)
def test_the_committed_artifacts_match_the_policy(spec: ExperimentSpec) -> None:
    """コミット済みの成果物も同じであること (宣言だけ直しても通らない)。"""
    path = spec.results_dir / DIAGNOSTICS_CSV
    if not spec.results_dir.is_dir():
        pytest.skip(f"{spec.results_dir} がまだありません")
    assert path.is_file() is (spec.number not in WITHOUT_DIAGNOSTICS), (
        f"{path} の有無が宣言と違います (make figures-{spec.number})"
    )


def test_experiment_05_still_calls_no_diagnostic() -> None:
    """05 が診断を呼び始めたら**配線も足すよう**に言う (D-133)。

    ``WITHOUT_DIAGNOSTICS`` に 05 が残っている理由そのものを測る。理由が消えたら
    リストからも外れるべきで、放置すると「出せるのに出していない」状態になる。
    """
    importers = [
        name
        for name in ANOMALY_MODULES
        if any(
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("rc_basics_lab.diagnostics")
            for node in ast.walk(ast.parse((PACKAGE_ROOT / name).read_text("utf-8")))
        )
    ]
    assert not importers, (
        f"05 が診断を呼び始めています: {importers}\n"
        f"{DIAGNOSTICS_CSV} への配線を足し、WITHOUT_DIAGNOSTICS から 05 を"
        "外してください (D-133)"
    )

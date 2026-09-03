"""コミット済みの成果物の実行時間が、宣言した予算に収まっていることの検査.

``catalog.ExperimentSpec.budget_s`` は「超えたら設計の見直しを要求する目安」
だが、**05 以外は誰も見ていなかった**。実際に起きた: D-146 で 3-T が本番に
入って ``figures-03`` が 327 秒から 1,761 秒になったとき、``docs/design.md``
は「予算を 1,800 秒へ引き上げた」と書き直されたのに ``catalog.py`` の
``budget_s`` は 900.0 のまま残り、**文書と宣言が食い違っても両方緑**だった。

枚数 (FIG-12) は機械が見ていて、パネル (FIG-15) と時間は見ていなかった。
**見ている予算だけが守られる**ので、1つを守るために他が破れる。ここは時間の
側の機械である (D-147)。

一次資料は ``results/<実験>/meta.json`` の ``wall_time_s`` で、成果物と同じ
再生成で更新される。**新しく測り直す必要は無い** —— 予算に対する比を、
コミットされている値で見る。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rc_basics_lab.experiment.catalog import CATALOG, ExperimentSpec

ROOT = Path(__file__).resolve().parent.parent

WARN_RATIO = 0.9
"""この比を超えたら「次の追加が入らない」と警告する水準。"""


def _committed_wall_time(name: str) -> float | None:
    """``results/<実験>/meta.json`` の ``wall_time_s`` (無ければ None)。"""
    path = ROOT / "results" / name / "meta.json"
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    value = loaded.get("wall_time_s")
    return float(value) if isinstance(value, int | float) else None


@pytest.mark.parametrize("spec", CATALOG, ids=lambda spec: spec.number)
def test_the_committed_wall_time_is_within_its_budget(spec: ExperimentSpec) -> None:
    """実測が宣言した予算を超えていない。

    変異注入: ``catalog.py`` の 03 の ``budget_s`` を 900.0 に戻すと赤くなる。
    """
    budget = float(spec.budget_s)
    actual = _committed_wall_time(spec.name)
    if actual is None:
        pytest.skip(f"{spec.name} の meta.json がありません (results/ 直下の実験)")
    assert actual <= budget, (
        f"実験 {spec.number} の実測 {actual:.1f} 秒が予算 {budget:.1f} 秒を超えました。"
        "**予算を上げて通すのは、そのぶん次の追加を無条件に許すこと**です —— "
        "掃引点や図を減らせないか先に確かめ、上げるなら docs/design.md に"
        "理由を書いてください (D-147)。"
    )


def test_no_experiment_is_about_to_outgrow_its_budget() -> None:
    """予算の 90% を超えている実験を**名指しで報告する**。

    落とさずに報告するのは、超過そのものではなく「次の追加が入らない」という
    先の話だからである。ここが赤くならない限り、03 が 97% であることは誰の
    目にも入らなかった (実測、2026-09-03 の棚卸し)。
    """
    tight = {
        spec.name: (actual, spec.budget_s)
        for spec in CATALOG
        if (actual := _committed_wall_time(spec.name)) is not None
        and actual > spec.budget_s * WARN_RATIO
    }
    assert not tight, (
        "予算の 90% を超えている実験があります (次の追加が入りません):\n"
        + "\n".join(
            f"  {name}: {actual:.1f} / {budget:.1f} 秒 ({actual / budget:.0%})"
            for name, (actual, budget) in sorted(tight.items())
        )
        + "\n減らす当てが無いなら、その実験は分割の時期です。"
    )

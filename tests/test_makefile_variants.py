"""``make help`` と ``Makefile`` の target が CATALOG の variant を網羅する検査.

実験の行は ``CATALOG`` から生成しているのに (D-125)、**variant の行は手書きの
まま**だった。結果、``ladder-03`` / ``ladder-threshold-03`` / ``operating-03``
の3本が ``make help`` に一度も出ないまま増えた (実測、2026-09-03 の棚卸し F)。

target 名は variant 名から機械的に導けない (``length`` -> ``saturation-03``)
ので、生成ではなく**照合**で守る。生成しようとすると名前の対応表をどこかに
持つことになり、その表が今度は写経の対象になる。
"""

from __future__ import annotations

import re
from pathlib import Path

from rc_basics_lab.experiment.catalog import CATALOG, MAIN

ROOT = Path(__file__).resolve().parent.parent
MAKEFILE = ROOT / "Makefile"

VARIANT_INVOCATION = re.compile(r"--variant\s+([\w-]+)")
"""Makefile のレシピが呼ぶ variant。"""

TARGET_LINE = re.compile(r"^([\w.-]+):")
"""target の宣言行 (``saturation-03:``)。"""


def _targets_by_variant() -> dict[str, str]:
    """variant 名 -> それを呼ぶ Makefile の target 名。"""
    found: dict[str, str] = {}
    current = ""
    for line in MAKEFILE.read_text(encoding="utf-8").splitlines():
        matched = TARGET_LINE.match(line)
        if matched:
            current = matched.group(1)
            continue
        for variant in VARIANT_INVOCATION.findall(line):
            found.setdefault(variant, current)
    return found


def _help_body() -> str:
    """``help:`` レシピの本文 (次の target 宣言まで)。"""
    lines = MAKEFILE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("help:"))
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if TARGET_LINE.match(lines[i]) and not lines[i].startswith("\t")
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def test_every_variant_has_a_makefile_target() -> None:
    """CATALOG の全 variant に Makefile の target がある。

    変異注入: ``Makefile`` の ``ladder-03`` を消すと赤くなる。
    """
    targets = _targets_by_variant()
    missing = sorted(
        f"{spec.number}:{name}"
        for spec in CATALOG
        for name in spec.variants
        if name != MAIN and name not in targets
    )
    assert not missing, (
        f"Makefile の target が無い variant があります: {missing}\n"
        "手で回せない variant は、成果物の作り方が誰にも見えません。"
    )


def test_every_variant_target_appears_in_help() -> None:
    """その target が ``make help`` に出る。

    実験の行は CATALOG から生成しているが (D-125)、variant の行は手書きな
    ので、足したときに help へ書き足し忘れる。**忘れたらここが赤くなる。**
    """
    body = _help_body()
    targets = _targets_by_variant()
    missing = sorted(
        {
            targets[name]
            for spec in CATALOG
            for name in spec.variants
            if name != MAIN and name in targets and targets[name] not in body
        }
    )
    assert not missing, (
        f"make help に出ていない variant の target があります: {missing}\n"
        "Makefile の help レシピに1行足してください。"
    )

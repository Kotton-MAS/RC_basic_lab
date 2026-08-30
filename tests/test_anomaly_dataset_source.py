"""記事05 の本番設定が MGAB 単独であること (D-117).

記事05 の本文は当初「UCR Anomaly Archive と MGAB を使った」と書いていたが、
`results/` を見ると **UCR は1点も使われていなかった** (``anomaly.csv`` は
90 行すべて ``dataset=mgab``)。文章と成果物が食い違ったまま公開される寸前だった。

食い違いの直し方は2つあり (UCR を回す / 記事を MGAB 単独に直す)、後者を採った
(D-117 の rationale)。**採ったほうを設定側にも固定する**のがこの検査である。

`src/rc_basics_lab/datasets/ucr.py` は残す —— 取得と読み取りの経路まで消すと、
「なぜ UCR を回さなかったのか」の判断材料 (`docs/plans/checkpoint-05b-t3.md` の
8系列の実測) を再現できなくなる。**在ることと、本番で回すことは別である。**
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "05_anomaly_detection" / "config.yaml"
RESULTS = ROOT / "results" / "05_anomaly_detection"

EXPECTED_SOURCE = "mgab"
"""記事05 の本番設定が使う唯一のデータ源 (D-117)。"""


def test_the_production_config_declares_mgab_as_the_only_source() -> None:
    """``config.yaml`` の ``dataset.source`` が ``mgab``。"""
    loaded: object = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    dataset = loaded.get("dataset")
    assert isinstance(dataset, dict), "dataset セクションがありません"
    assert dataset.get("source") == EXPECTED_SOURCE, (
        f"記事05 の本番設定は {EXPECTED_SOURCE} 単独です (D-117)。"
        f" 変えるなら決定の見直しが先です: {dataset.get('source')!r}"
    )


def test_the_committed_meta_records_the_same_source() -> None:
    """``meta.json`` の ``dataset.source`` も ``mgab``。

    設定だけを見ると「設定したのに効いていない」を見逃す。**実際に回した記録**
    の側も固定する。
    """
    loaded: object = json.loads((RESULTS / "meta.json").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    config = loaded.get("config")
    assert isinstance(config, dict)
    dataset = config.get("dataset")
    assert isinstance(dataset, dict)
    assert dataset.get("source") == EXPECTED_SOURCE


def test_no_committed_row_comes_from_another_dataset() -> None:
    """``anomaly.csv`` の ``dataset`` 列が ``mgab`` だけ (D-117)。

    **記事が引用するのは行であって設定ではない。** ここが本命の検査で、
    UCR の行が1つでも混ざれば落ちる。
    """
    with (RESULTS / "anomaly.csv").open(encoding="utf-8", newline="") as handle:
        sources = {row["dataset"] for row in csv.DictReader(handle)}
    assert sources == {EXPECTED_SOURCE}, (
        f"本番の成果物に {EXPECTED_SOURCE} 以外のデータ源が混ざっています: "
        f"{sorted(sources)} (D-117)"
    )


def test_the_readme_says_the_repository_has_no_ucr_results() -> None:
    """README を読んだ人が「UCR の結果はこのリポジトリに無い」と分かる。

    取得コードだけが在ると、読者は結果も在ると読む。実際に記事本文がそう書いて
    公開寸前になった。
    """
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "UCR の行は1つも" in text, (
        "README に「UCR の結果は無い」旨の記述がありません (D-117)"
    )
    assert "D-117" in text, "README から決定 D-117 を辿れません"

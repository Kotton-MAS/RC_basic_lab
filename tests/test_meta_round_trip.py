"""``meta.json`` の設定ブロックが**読み直せる**ことを見る (D-129).

## なぜ要るのか

``results/`` は読者に届く唯一のもので (CLAUDE.md)、``meta.json`` はその素性の
記録である。``dataclasses.asdict`` は ``KIND`` が ``ClassVar`` なので落として
しまい、記録から「どのモデル・どの課題・どのトポロジだったか」が復元できない:

===============================  ==============================
書いたもの                       ``asdict`` の結果
===============================  ==============================
``RingTopologyConfig()``         ``{}`` (フィールド0個。情報ゼロ)
``RingConfig()`` (SCR モデル)    ``kind`` 無し。ESN と区別できない
``BarabasiAlbertConfig(m=2)``    ``{"m": 2}`` から**推測**するしかない
===============================  ==============================

スケールフリーの実験を何本か回して ``results/`` をアーカイブしたとき、
どれがどのトポロジだったかが成果物から分からない。

## 何を測るか

**書いたものを読み直すと元に戻る** (round-trip)。これは性質なので、判別子の
書き出しを外す変異で確実に赤くなる。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rc_basics_lab.config import (
    Anomaly05Config,
    Capacity03Config,
    Chaos04Config,
    Esp02Config,
    ExperimentConfig,
    load_config_as,
)
from rc_basics_lab.config._dump import as_plain_mapping
from rc_basics_lab.experiment.catalog import CATALOG, ExperimentSpec
from rc_basics_lab.meta import collect_meta_for
from rc_basics_lab.overrides import KIND_KEY
from rc_basics_lab.reservoir.deep import DeepESNConfig
from rc_basics_lab.reservoir.esn import ESNConfig
from rc_basics_lab.reservoir.ring import RingConfig
from rc_basics_lab.reservoir.topology import (
    BarabasiAlbertConfig,
    RingTopologyConfig,
    WattsStrogatzConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = (
    ESNConfig(),
    ESNConfig(topology=RingTopologyConfig()),
    ESNConfig(topology=BarabasiAlbertConfig(m=2)),
    ESNConfig(topology=WattsStrogatzConfig()),
    DeepESNConfig(),
    RingConfig(),
)


@pytest.mark.parametrize("model", MODELS, ids=lambda m: repr(m)[:40])
def test_a_reservoir_config_round_trips_through_the_dump(model: object) -> None:
    """落として読み直すと同じ設定に戻る (**判別子が要る**)。"""
    dumped = as_plain_mapping(model)
    section = json.loads(json.dumps({"reservoir": dumped}))
    rebuilt = _rebuild_reservoir(section["reservoir"])
    assert rebuilt == model, f"読み直すと別の設定になりました: {dumped}"


def _rebuild_reservoir(mapping: dict[str, object]) -> object:
    """``meta.json`` の断片から設定を組み直す (``load_config_as`` と同じ経路)。"""
    from rc_basics_lab.config._common import _coerce
    from rc_basics_lab.reservoir.protocol import ReservoirConfig

    return _coerce(mapping, ReservoirConfig, "meta.json.reservoir")


def test_every_config_dump_carries_a_kind_where_a_union_is_involved() -> None:
    """判別子を名乗る dataclass には ``kind`` が書かれている。"""
    dumped = as_plain_mapping(ESNConfig(topology=RingTopologyConfig()))
    assert dumped[KIND_KEY] == "esn"
    topology = dumped["topology"]
    assert isinstance(topology, dict)
    assert topology[KIND_KEY] == "ring", (
        "フィールドが0個の設定は kind が無いと情報がゼロになります"
    )


@pytest.mark.parametrize("spec", CATALOG, ids=lambda s: s.number)
def test_the_committed_meta_json_rebuilds_the_config(spec: ExperimentSpec) -> None:
    """コミット済みの ``meta.json`` から**設定が組み直せる** (D-129)。

    ``kind`` という文字列の有無ではなく round-trip を見る。判別子つき union を
    1つも持たない実験 (05) では ``kind`` が現れないのが正しく、文字列で測ると
    その実験だけ嘘の赤が出る。**測りたいのは「素性が復元できること」**である。
    """
    path = spec.results_dir / "meta.json"
    if not path.is_file():
        pytest.skip(f"{path} がまだありません")
    recorded = json.loads(path.read_text(encoding="utf-8"))["config"]
    rebuilt = _rebuild(recorded, CONFIG_TYPES[spec.number])
    expected: object = load_config_as(spec.config_path, CONFIG_TYPES[spec.number])
    assert rebuilt == expected, (
        f"{path} の設定が読み直せません (make figures-0N で取り直してください)"
    )


CONFIG_TYPES = {
    "01": ExperimentConfig,
    "02": Esp02Config,
    "03": Capacity03Config,
    "04": Chaos04Config,
    "05": Anomaly05Config,
}


@pytest.mark.parametrize("spec", CATALOG, ids=lambda s: s.number)
def test_the_meta_config_block_rebuilds_the_production_config(
    spec: ExperimentSpec,
) -> None:
    """``meta.json`` の ``config`` を読み直すと本番の設定に戻る。

    **記録が記録として機能していることの実測**である。``kind`` が欠けると
    ここで落ちる (union の先頭として組み直され、未知キーになる)。
    """
    number = spec.number
    config: object = load_config_as(spec.config_path, CONFIG_TYPES[number])
    seeds = getattr(config, "seeds", None)
    if seeds is None:
        pytest.skip(f"実験 {number} は seeds を直接持ちません")
    meta = collect_meta_for(config, seeds)
    rebuilt = _rebuild(meta["config"], CONFIG_TYPES[number])
    assert rebuilt == config, f"実験 {number} の meta.json が読み直せません"


def _rebuild(mapping: object, cls: type) -> object:
    from rc_basics_lab.config._common import _build

    return _build(cls, mapping, "meta.json.config")

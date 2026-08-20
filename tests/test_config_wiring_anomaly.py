"""05 の合成源の設定配線テスト (D-13 / D-69 / D-70) —— **全葉が出力を変える**.

01 (``tests/test_config_wiring.py``) / 02 / 03 / 04 と同じ防衛線を、05 の
合成源 (``SyntheticAnomalyConfig``) に張る。ここが測るのは1つだけである:

    ``leaf_paths(SyntheticAnomalyConfig)`` の**全葉**について、値を変えると
    ``generate_synthetic_anomalies`` の出力 (値 + ラベル + ignore + train_end
    + params) が変わる。

T3 が実験1本ぶんの ``Anomaly05Config`` を足すとき、この被覆は
``mackey_glass.*`` を ``DELEGATED_SECTIONS`` で免除せずに済む形になっている
—— 免除が要らないのは、``length`` / ``horizon`` という**上書きされる2葉を
器から取り除いた**ためである (D-69)。04 の委譲免除は「委譲先の別テストが同じ
葉を被覆している」ことが前提であり、05 の経路には使えない (01 の
``test_config_wiring`` が『``length`` は効く』と証明している葉は、05 の経路
では効かない)。

**ネットワークにもファイルにも触れない** (D-60)。唯一の例外は
``config/anomaly05.py`` と ``tasks/anomaly.py`` のソースを AST で読む構造検査
(D-70) で、これは「``MackeyGlassConfig`` を組み立てられる場所が1箇所しかない」
ことを実装の形から確かめるためのものである。
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import yaml
from wiring import apply_case, assert_yaml_has_all_leaves, case, leaf_paths, plain

from rc_basics_lab.config import (
    MackeyGlassConfig,
    SyntheticAnomalyConfig,
    SyntheticMackeyGlassConfig,
    load_config_as,
)
from rc_basics_lab.tasks.anomaly import AnomalySeries, generate_synthetic_anomalies

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "rc_basics_lab"

BASE = SyntheticAnomalyConfig(
    length=4000,
    n_anomalies=3,
    segment_length=60,
    ignore_margin=20,
    mackey_glass=SyntheticMackeyGlassConfig(),
)
"""被覆の基準になる小さい設定 (``tests/test_tasks_anomaly.py`` の ``SMALL`` と同じ)。

1本の生成が実測 0.022 秒なので、全葉 (11 件) を1件ずつ回しても 0.3 秒で済む。
"""

SEED = 20250820
"""全ケースで共有する task ストリームの seed (差分が seed 由来にならないように)。"""

OVERRIDES: dict[str, object] = {
    "length": 4400,
    "n_anomalies": 2,
    "segment_length": 80,
    "ignore_margin": 30,
    "mackey_glass.tau": 18.0,
    "mackey_glass.beta": 0.25,
    "mackey_glass.gamma": 0.12,
    "mackey_glass.exponent": 9,
    "mackey_glass.rk4_step": 0.05,
    "mackey_glass.sample_interval": 8,
    "mackey_glass.integration_burn_in": 800,
}
"""葉ごとの差し替え値。``BASE`` と併せて**必ず生成が成功する**値だけを置く。

``mackey_glass.tau`` / ``rk4_step`` は ``tau / rk4_step`` が整数になる組
(18.0/0.1 = 180、17.0/0.05 = 340) を選んである —— そうでない設定は
``tasks/mackey_glass.py`` の ``delay_steps`` が ``ValueError`` にする。
"""


def _fingerprint(series: AnomalySeries) -> str:
    """``AnomalySeries`` の全成分をまとめた指紋。

    値だけを見ると ``ignore_margin`` のように**マスクしか変えない**葉を
    取りこぼす (04 の 4-C が ``meta`` チャネルを別立てにしたのと同じ事情)。
    ここでは値・ラベル・ignore・train_end・params を1本のダイジェストに畳む。
    """
    digest = hashlib.sha256()
    digest.update(np.asarray(series.values).tobytes())
    digest.update(np.asarray(series.labels).tobytes())
    digest.update(np.asarray(series.ignore).tobytes())
    digest.update(str(series.train_end).encode("utf-8"))
    digest.update(repr(sorted(series.params.items())).encode("utf-8"))
    return digest.hexdigest()


def _generate(cfg: SyntheticAnomalyConfig) -> AnomalySeries:
    return generate_synthetic_anomalies(cfg, np.random.default_rng(SEED))


def test_every_leaf_of_the_synthetic_config_has_an_override_case() -> None:
    """``OVERRIDES`` が全葉と過不足なく一致する (被覆の完全性)。

    ``SyntheticAnomalyConfig`` に葉を足して ``OVERRIDES`` に書き忘れると、
    ここが赤くなる。パラメータ化の元をコード側 (``leaf_paths``) から作っている
    ので、「テストに書いた葉だけを被覆する」同語反復にはならない。
    """
    assert set(OVERRIDES) == leaf_paths(SyntheticAnomalyConfig), (
        "OVERRIDES が全葉と一致しません "
        f"(不足={sorted(leaf_paths(SyntheticAnomalyConfig) - set(OVERRIDES))}, "
        f"余剰={sorted(set(OVERRIDES) - leaf_paths(SyntheticAnomalyConfig))})"
    )


def test_the_synthetic_mackey_glass_config_has_no_length_or_horizon_leaf() -> None:
    """``length`` / ``horizon`` が**葉として存在しない** (D-69)。

    ``generate_synthetic_anomalies`` はこの2つを必ず上書きするので、器に残すと
    「YAML から設定できるのに出力が1バイトも変わらない」死んだ葉になる。
    死葉リストで固定するのではなく、器から取り除くのがこの決定である。
    """
    leaves = leaf_paths(SyntheticAnomalyConfig)
    assert "mackey_glass.length" not in leaves
    assert "mackey_glass.horizon" not in leaves
    assert leaf_paths(SyntheticMackeyGlassConfig) == {
        "tau",
        "beta",
        "gamma",
        "exponent",
        "rk4_step",
        "sample_interval",
        "integration_burn_in",
    }


@pytest.mark.parametrize("leaf", sorted(OVERRIDES))
def test_each_synthetic_leaf_changes_the_generated_series(leaf: str) -> None:
    """全葉について「値を変えたら出力が変わる」ことを実測する (D-69)。

    本リポジトリ最大の失敗モード**「設定したのに効いていない」**への防衛線
    (``tests/wiring.py`` のモジュール docstring)。05 では特に、土台の
    Mackey-Glass 生成パラメータ7葉が合成源の経路でも生きていることを測る
    —— 01 の ``test_each_parameter_changes_output`` は 01 のパイプラインでしか
    測っておらず、05 の経路で効いている保証にはならない。
    """
    baseline = _fingerprint(_generate(BASE))
    changed = apply_case(BASE, case(leaf, OVERRIDES[leaf]))
    assert changed != BASE, f"差し替えが設定に反映されていません: {leaf}"
    assert _fingerprint(_generate(changed)) != baseline, (
        f"{leaf} を変えても合成源の出力が1バイトも変わりません (死んだ設定です。D-69)"
    )


def test_every_leaf_is_settable_from_yaml(tmp_path: Path) -> None:
    """全葉が YAML から往復できる (D-09 の未知キー検査と対になる検査)。

    「dataclass には在るが YAML からは設定できない」パラメータを作らないため
    の検査。効くこと (上のテスト) と設定できること (ここ) の両方が要る。
    """
    written = {
        "length": BASE.length,
        "n_anomalies": BASE.n_anomalies,
        "segment_length": BASE.segment_length,
        "ignore_margin": BASE.ignore_margin,
        "mackey_glass": {
            "tau": 18.0,
            "beta": 0.25,
            "gamma": 0.12,
            "exponent": 9,
            "rk4_step": 0.1,
            "sample_interval": 8,
            "integration_burn_in": 800,
        },
    }
    assert_yaml_has_all_leaves(written, SyntheticAnomalyConfig)
    path = tmp_path / "synthetic.yaml"
    path.write_text(
        yaml.safe_dump(cast("dict[str, object]", plain(written))), encoding="utf-8"
    )
    loaded = load_config_as(path, SyntheticAnomalyConfig)
    assert loaded.mackey_glass.tau == 18.0
    assert loaded.mackey_glass.integration_burn_in == 800


def test_synthetic_mackey_glass_defaults_come_from_experiment01() -> None:
    """7葉の既定値が 01 の ``MackeyGlassConfig`` と一致する (D-69 の但し書き)。

    05 が絞ったのは**葉の集合**であって既定値ではない。数値をリテラルで
    書き写すと、01 の tau を変えたときに 05 の合成源だけ古い値で回り続ける。
    """
    narrowed = SyntheticMackeyGlassConfig()
    full = MackeyGlassConfig()
    for name in leaf_paths(SyntheticMackeyGlassConfig):
        assert getattr(narrowed, name) == getattr(full, name), (
            f"{name} の既定値が 01 の MackeyGlassConfig と食い違っています"
        )


def _names_mackey_glass_config(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id == "MackeyGlassConfig"
    return isinstance(func, ast.Attribute) and func.attr == "MackeyGlassConfig"


def _length_bearing_construction_sites(path: Path) -> list[str]:
    """``MackeyGlassConfig(..., length=...)`` を組み立てている場所を列挙する。

    ``length`` を渡す呼び出しだけを見るのは、系列長を決めて生成パラメータを
    組み立てる行為がここでの関心事であるため (既定値を取るだけの
    ``MackeyGlassConfig()`` は対象外)。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative = str(path.relative_to(SRC))
    sites: list[str] = []

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if (
                isinstance(child, ast.Call)
                and _names_mackey_glass_config(child.func)
                and any(keyword.arg == "length" for keyword in child.keywords)
            ):
                sites.append(f"{relative}::{scope}")
            inner = (
                child.name
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                else scope
            )
            visit(child, inner)

    visit(tree, "<module>")
    return sites


def test_only_to_mackey_glass_builds_the_generation_parameters(

) -> None:
    """合成源の経路で ``MackeyGlassConfig`` を組み立てるのは1箇所だけ (D-70)。

    ``tasks/chaotic.py`` の ``Standardizer.from_training_prefix`` (D-41) と同じ
    「係数を作れる場所を1本に閉じる」流儀。呼び出し側が自前で組み立てられると、
    そこで ``horizon`` や ``sample_interval`` を黙って別の値にする経路が復活し、
    D-69 が取り除いた死葉が別の形で戻る。

    対象は「``SyntheticAnomalyConfig`` / ``SyntheticMackeyGlassConfig`` を
    名指ししているモジュール」= 合成源の経路に限る。01 / 04 が自分の都合で
    ``MackeyGlassConfig`` を組み立てることまで禁じる決定ではない。

    **静的解析の原理的限界**: ``getattr(config, "MackeyGlassConfig")(...)`` の
    ような動的な組み立ては追わない (D-68 で同じ線引きをした)。
    """
    on_path = [
        path
        for path in sorted(SRC.rglob("*.py"))
        if "SyntheticAnomalyConfig" in path.read_text(encoding="utf-8")
        or "SyntheticMackeyGlassConfig" in path.read_text(encoding="utf-8")
    ]
    assert {str(path.relative_to(SRC)) for path in on_path} >= {
        "config/anomaly05.py",
        "tasks/anomaly.py",
    }, "合成源の経路のモジュールが見つかりません (探索条件が壊れています)"

    sites = [
        site for path in on_path for site in _length_bearing_construction_sites(path)
    ]
    assert sites == ["config/anomaly05.py::to_mackey_glass"], (
        "合成源の経路で MackeyGlassConfig を組み立てている場所が "
        f"to_mackey_glass 以外にあります (D-70): {sites}"
    )

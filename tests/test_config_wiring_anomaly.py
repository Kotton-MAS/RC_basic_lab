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
import dataclasses
import hashlib
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import yaml
from wiring import (
    CHANNEL_META,
    WiringCase,
    apply_case,
    assert_yaml_has_all_leaves,
    case,
    leaf_paths,
    plain,
)

from rc_basics_lab.config import (
    Anomaly05Config,
    AnomalyDatasetConfig,
    AnomalyPreprocessConfig,
    AnomalyProtocolSweepConfig,
    AnomalyReservoirConfig,
    AnomalyRidgeConfig,
    AnomalySizeSweepConfig,
    AnomalyThresholdConfig,
    MackeyGlassConfig,
    SyntheticAnomalyConfig,
    SyntheticMackeyGlassConfig,
    load_config_as,
)
from rc_basics_lab.experiment.anomaly import run_anomaly_headline
from rc_basics_lab.experiment.anomaly_rows import (
    AnomalyRow,
    ThresholdSweepRow,
    anomaly_csv_columns,
    anomaly_row_as_dict,
)
from rc_basics_lab.experiment.anomaly_score import (
    ANOMALY_METHODS,
    ESN_RESIDUAL,
    RANDOM_CONTROL,
)
from rc_basics_lab.experiment.anomaly_sources import build_sources
from rc_basics_lab.experiment.anomaly_sweep import (
    run_protocol_sweep,
    run_size_sweep,
    summarize_size_sweep,
)
from rc_basics_lab.tasks.anomaly import AnomalySeries, generate_synthetic_anomalies

if TYPE_CHECKING:  # pragma: no cover - 型検査時のみ必要
    from _typeshed import DataclassInstance

CHANNEL_SWEEP = "sweep"
"""5-C / 5-D の格子専用のチャネル (実験固有)。

``protocol_sweep`` / ``size_sweep`` の葉が変えるのは 5-A の行ではなく
**掃引の行**である (``anomaly_protocol.csv`` / ``anomaly_size.csv`` の行数と、
5-D の劣化点)。5-A の指紋で測ると「変えても変わらない」と出てしまうので、
観測点を掃引側に置く。T3 がこの4葉を置かなかったのはこの観測点が無かった
ためで、T4 が掃引の実装と同時に足した。
"""

CHANNEL_SOURCES = "sources"
"""``dataset.source`` 専用のチャネル (実験固有、``wiring.py`` の想定どおり)。

``source`` を変えると系列源の**型**が変わるが、実データ源はキャッシュが無い
環境では回せない (D-60: pytest はネットワークに触れない)。そこで観測点を
「``build_sources`` が返す源の型」に置く —— 系列そのものの差はキャッシュ
依存になるので、CI で毎回測れる観測点はここしかない。
"""

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
    """``AnomalySeries`` の**データ**をまとめた指紋 (値 / ラベル / ignore / train_end)。

    値だけを見ると ``ignore_margin`` のように**マスクしか変えない**葉を
    取りこぼす (04 の 4-C が ``meta`` チャネルを別立てにしたのと同じ事情) ので、
    4成分すべてを1本のダイジェストに畳む。

    **``params`` は入れない**。``generate_synthetic_anomalies`` は
    ``params["tau"]`` に設定値をそのまま書き写すので、``params`` を指紋に含めると
    「設定を読んで meta に転記しているだけで生成には効いていない」葉が緑で
    通ってしまう。実測 (fixer): ``to_mackey_glass`` を ``tau`` だけ無視する版に
    変異させると、``params`` 込みの指紋では 11 件すべて緑のまま通り、
    データだけの指紋では ``mackey_glass.tau`` の1件が赤くなる。
    """
    digest = hashlib.sha256()
    digest.update(np.asarray(series.values).tobytes())
    digest.update(np.asarray(series.labels).tobytes())
    digest.update(np.asarray(series.ignore).tobytes())
    digest.update(str(series.train_end).encode("utf-8"))
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


def test_only_to_mackey_glass_builds_the_generation_parameters() -> None:
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
        "tasks/synthetic.py",
    }, "合成源の経路のモジュールが見つかりません (探索条件が壊れています)"

    sites = [
        site for path in on_path for site in _length_bearing_construction_sites(path)
    ]
    assert sites == ["tasks/synthetic.py::to_mackey_glass"], (
        "合成源の経路で MackeyGlassConfig を組み立てている場所が "
        f"to_mackey_glass 以外にあります (D-70): {sites}"
    )


# --------------------------------------------------------------------------
# Anomaly05Config (T3) —— 実験1本ぶんの全葉被覆
# --------------------------------------------------------------------------

DELEGATED_SECTIONS: tuple[tuple[str, type], ...] = (
    ("synthetic.", SyntheticAnomalyConfig),
)
"""接頭辞と、その配下が過不足なく一致すべき委譲先の設定クラス。

``synthetic`` は合成源の設定で、被覆はこのファイルの前半
(``test_each_synthetic_leaf_changes_the_generated_series``) が**11葉すべて**
について実測している (D-69)。04 の ``DELEGATED_SECTIONS`` と同じ前提
(委譲先の別テストが同じ葉を被覆している) を満たす —— 05 では委譲先が同じ
ファイルの中にあるので、免除が空振りになる余地がさらに小さい。
"""

REDUCED = Anomaly05Config(
    dataset=AnomalyDatasetConfig(
        series=("s1",),
        max_length=2000,
        train_ratio=0.25,
        calibration_ratio=0.15,
    ),
    synthetic=SyntheticAnomalyConfig(
        length=2000, n_anomalies=3, segment_length=40, ignore_margin=10
    ),
    preprocess=AnomalyPreprocessConfig(
        standardize_steps=200, input_window=8, score_smoothing=4
    ),
    reservoir=AnomalyReservoirConfig(n_units=30, washout=20, n_replicates=1),
    ridge=AnomalyRidgeConfig(alpha_grid=(1e-4, 1e-2, 1.0)),
    threshold=AnomalyThresholdConfig(sweep_points=5),
    protocol_sweep=AnomalyProtocolSweepConfig(
        normalize_grid=("zscore", "minmax"),
        input_window_grid=(4, 8),
        score_smoothing_grid=(2, 4),
    ),
    size_sweep=AnomalySizeSweepConfig(n_units_grid=(20, 25, 30)),
)
"""秒オーダーで回せる縮小設定 (**構造は本番と同じ**)。実測 0.02 秒/回。

本番既定 (``Anomaly05Config()``) は 3系列 x 5レプリケート x 6系統 = 90 行で
実測 3.7 秒。配線テストは葉の数だけ回すので、1回 0.02 秒まで落とす。
"""

ANOMALY_CASES: tuple[WiringCase, ...] = (
    case("name", "another_name", channel=CHANNEL_META),
    case("dataset.source", "mgab", channel=CHANNEL_SOURCES),
    case("dataset.series", ("s1", "s2")),
    case("dataset.max_length", 1500),
    case("dataset.train_ratio", 0.15),
    case("dataset.calibration_ratio", 0.25),
    case("preprocess.normalize", "minmax"),
    case("preprocess.standardize_steps", 350),
    case("preprocess.input_window", 24),
    case("preprocess.score_smoothing", 16),
    case("reservoir.n_units", 50, scope=ESN_RESIDUAL),
    case("reservoir.spectral_radius", 1.1, scope=ESN_RESIDUAL),
    case("reservoir.leak_rate", 0.6, scope=ESN_RESIDUAL),
    case("reservoir.input_scale", 0.9, scope=ESN_RESIDUAL),
    case("reservoir.density", 0.3, scope=ESN_RESIDUAL),
    case("reservoir.washout", 60),
    case("reservoir.n_replicates", 2),
    case("ridge.alpha_grid", (1.0,)),
    case("threshold.target_false_alarm_rate", 0.05),
    case("threshold.report_test_optimal", False),
    case("threshold.sweep_points", 9),
    case("evaluation.report_point_adjust", False),
    case("evaluation.pa_k_grid", (0.0,)),
    case("evaluation.ignore_transition", False),
    case("protocol_sweep.normalize_grid", ("zscore", "robust"), channel=CHANNEL_SWEEP),
    case("protocol_sweep.input_window_grid", (8, 12, 16), channel=CHANNEL_SWEEP),
    case("protocol_sweep.score_smoothing_grid", (1, 4), channel=CHANNEL_SWEEP),
    case("size_sweep.n_units_grid", (30, 60), channel=CHANNEL_SWEEP),
    case("seeds.reservoir", 11, scope=ESN_RESIDUAL),
    case("seeds.task", 12),
    case("seeds.split", 13),
    case("seeds.control", 14, scope=RANDOM_CONTROL),
)
"""1葉1ケース。``scope`` を付けたケースは「その系統の ``auprc`` だけが動く」。

**仕様 §5 の配線表からの訂正が1行ある**: ``seeds.control`` の観測点は
「一様乱数対照の ``auprc`` のみ変わる (他手法は不変)」ではなく
「**一様乱数対照の ``auprc`` だけが変わり、他系統の ``auprc`` は不変**」で
ある。行そのものは全系統で動く —— ``auprc_random`` 列 (と PA の対照列) を
**全行が持ち歩く**設計にしたため (D-61: 図を見ない読者にも基準線が届く)。
"""


def _reduced_result(
    config: Anomaly05Config,
) -> tuple[tuple[AnomalyRow, ...], tuple[ThresholdSweepRow, ...]]:
    results = run_anomaly_headline(config, build_sources(config))
    return results.rows, results.threshold_rows


def _row_fingerprint(config: Anomaly05Config, rows: Sequence[AnomalyRow]) -> str:
    """行 + 列の並びをまとめた指紋 (``wall_time_s`` は除く)。

    列そのものが設定で増減する (``f1_test_optimal`` / PA%K) ので、**列名も
    指紋に入れる** —— 値だけを見ると ``threshold.report_test_optimal`` の
    ような「列の有無」を変える葉を取りこぼす。
    """
    digest = hashlib.sha256()
    digest.update(repr(anomaly_csv_columns(config)).encode("utf-8"))
    for row in rows:
        payload = anomaly_row_as_dict(row)
        payload.pop("wall_time_s")
        digest.update(repr(sorted(payload.items())).encode("utf-8"))
    return digest.hexdigest()


def _auprc_by_method(rows: Sequence[AnomalyRow]) -> dict[str, tuple[float, ...]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.method, []).append(row.auprc)
    return {method: tuple(values) for method, values in grouped.items()}


@lru_cache(maxsize=1)
def _baseline() -> tuple[tuple[AnomalyRow, ...], tuple[ThresholdSweepRow, ...]]:
    return _reduced_result(REDUCED)


def _protocol_fingerprint(config: Anomaly05Config) -> str:
    """5-C の行 (格子点 + 集計) をまとめた指紋。

    行数だけを見ると「格子の値だけ入れ替えた」ケース (``normalize_grid`` を
    ``minmax`` から ``robust`` へ) を取りこぼす —— 行数は同じで中身が変わる。
    """
    digest = hashlib.sha256()
    for row in run_protocol_sweep(config, build_sources(config)):
        digest.update(repr(dataclasses.astuple(row)).encode("utf-8"))
    return digest.hexdigest()


def _size_outcome(config: Anomaly05Config) -> tuple[str, int]:
    """5-D の行の指紋と劣化点 (``n_units_at_90pct``)。

    仕様 §4 T4 の受け入れ基準4 が「行数**と** ``n_units_at_90pct`` が変わる」
    を要求しているので、2つを別々に返して両方を測る。
    """
    rows = run_size_sweep(config, build_sources(config))
    digest = hashlib.sha256()
    for row in rows:
        digest.update(repr(dataclasses.astuple(row)).encode("utf-8"))
    return digest.hexdigest(), summarize_size_sweep(rows).n_units_at_90pct


def _case_named(field: str) -> WiringCase:
    return next(item for item in ANOMALY_CASES if item.field == field)


def test_all_anomaly_config_fields_are_covered() -> None:
    """``Anomaly05Config`` の全葉が ``ANOMALY_CASES`` に登場する。

    ``synthetic`` 節だけは委譲だが、**委譲先の葉集合と過不足なく一致する**
    ことも同時に確かめる (04 の ``test_all_chaos_config_fields_are_covered``
    と同じ形)。確かめずに接頭辞で除外すると、委譲先に無いフィールドをその下に
    足して被覆から逃がせてしまう。
    """
    all_leaves = leaf_paths(Anomaly05Config)
    delegated: set[str] = set()
    for prefix, config_type in DELEGATED_SECTIONS:
        under_prefix = {leaf for leaf in all_leaves if leaf.startswith(prefix)}
        expected_leaves = {
            f"{prefix}{leaf}"
            for leaf in leaf_paths(cast("type[DataclassInstance]", config_type))
        }
        assert under_prefix == expected_leaves, (
            f"{prefix} 配下が {config_type.__name__} と一致していません"
            f" (不足={sorted(expected_leaves - under_prefix)},"
            f" 余分={sorted(under_prefix - expected_leaves)})"
        )
        delegated |= under_prefix

    covered = {item.field for item in ANOMALY_CASES}
    expected = all_leaves - delegated
    assert covered == expected, (
        f"未登録: {sorted(expected - covered)} / 余分: {sorted(covered - expected)}"
    )
    for item in ANOMALY_CASES:
        for path, _ in item.overrides:
            assert path in expected, f"未知のパスです: {path}"


@pytest.mark.parametrize(
    "wiring_case", ANOMALY_CASES, ids=[item.field for item in ANOMALY_CASES]
)
def test_each_parameter_changes_output(wiring_case: WiringCase) -> None:
    """各葉の値変更が 5-A / 5-B の出力を変える (配線の実測)。"""
    changed_config = apply_case(REDUCED, wiring_case)
    assert changed_config != REDUCED, "差し替えが設定に反映されていません"

    if wiring_case.channel == CHANNEL_SWEEP:
        if wiring_case.field.startswith("size_sweep."):
            base_digest, base_point = _size_outcome(REDUCED)
            digest, point = _size_outcome(changed_config)
            assert digest != base_digest, (
                f"{wiring_case.field} を変えても 5-D の行が変わりません"
            )
            assert point != base_point, (
                f"{wiring_case.field} を変えても n_units_at_90pct が変わりません"
            )
            return
        assert _protocol_fingerprint(changed_config) != _protocol_fingerprint(
            REDUCED
        ), f"{wiring_case.field} を変えても 5-C の行が変わりません (配線漏れ)"
        return

    if wiring_case.channel == CHANNEL_SOURCES:
        base_types = {type(item).__name__ for item in build_sources(REDUCED).values()}
        changed_types = {
            type(item).__name__ for item in build_sources(changed_config).values()
        }
        assert base_types != changed_types, (
            f"{wiring_case.field} を変えても系列源が変わりません (配線漏れ)"
        )
        return

    base_rows, base_sweep = _baseline()
    rows, sweep = _reduced_result(changed_config)

    if wiring_case.field == "threshold.sweep_points":
        assert len(sweep) != len(base_sweep), (
            "sweep_points が anomaly_threshold.csv の行数を変えていません"
        )
        return

    if wiring_case.channel == CHANNEL_META:
        assert _row_fingerprint(changed_config, rows) == _row_fingerprint(
            REDUCED, base_rows
        ), "メタ情報のはずが結果行を変えています"
        assert changed_config.name != REDUCED.name
        return

    assert _row_fingerprint(changed_config, rows) != _row_fingerprint(
        REDUCED, base_rows
    ), f"{wiring_case.field} を変えても出力が変わりません (配線漏れ)"

    if wiring_case.scope is not None:
        changed_auprc = _auprc_by_method(rows)
        base_auprc = _auprc_by_method(base_rows)
        assert changed_auprc[wiring_case.scope] != base_auprc[wiring_case.scope], (
            f"{wiring_case.field} が {wiring_case.scope} の auprc を変えていません"
        )
        for method in ANOMALY_METHODS:
            if method == wiring_case.scope:
                continue
            assert changed_auprc[method] == base_auprc[method], (
                f"{wiring_case.field} が {method} の auprc まで変えています"
            )


def test_calibration_ratio_moves_only_the_calibration_size() -> None:
    """``calibration_ratio`` を動かしても ``n_train`` が変わらない (切り分け)。

    ``train_ratio`` のケースで ``n_train`` が変わった原因を ``train_ratio``
    に帰属できるようにするための対照 (01 の ``test_split_ratio_isolation``
    と同じ役割)。
    """
    base_rows, _ = _baseline()
    rows, _ = _reduced_result(
        apply_case(REDUCED, _case_named("dataset.calibration_ratio"))
    )
    assert rows[0].n_train == base_rows[0].n_train
    assert rows[0].n_calibration != base_rows[0].n_calibration


def test_the_false_alarm_rate_moves_the_threshold_but_not_the_auprc() -> None:
    """``target_false_alarm_rate`` は運用点だけを動かす (AUPRC は閾値非依存)。

    仕様 §5 の配線表が ``auprc`` を「**不変**を要求」としている行の実測。
    AUPRC が閾値で動いたら、それは主指標が閾値依存の量に化けている証拠で
    ある (図の縦軸の意味が変わる)。
    """
    base_rows, _ = _baseline()
    rows, _ = _reduced_result(
        apply_case(REDUCED, _case_named("threshold.target_false_alarm_rate"))
    )
    assert [row.threshold for row in rows] != [row.threshold for row in base_rows]
    assert [row.auprc for row in rows] == [row.auprc for row in base_rows]


def test_the_split_seed_moves_the_offset() -> None:
    """``seeds.split`` が分割境界を動かす (``split_offset`` の実測)。"""
    base_rows, _ = _baseline()
    rows, _ = _reduced_result(apply_case(REDUCED, _case_named("seeds.split")))
    assert rows[0].split_offset != base_rows[0].split_offset


def test_every_anomaly_field_round_trips_yaml(tmp_path: Path) -> None:
    """``Anomaly05Config`` の全葉が YAML のキーとして往復する (D-09 と対)。"""
    path = tmp_path / "anomaly05.yaml"
    dumped = cast("Mapping[str, object]", plain(dataclasses.asdict(REDUCED)))
    path.write_text(yaml.safe_dump(dumped, allow_unicode=True), encoding="utf-8")
    assert load_config_as(path, Anomaly05Config) == REDUCED
    assert_yaml_has_all_leaves(
        yaml.safe_load(path.read_text(encoding="utf-8")), Anomaly05Config
    )

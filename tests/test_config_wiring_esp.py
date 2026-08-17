"""02 の設定配線テスト (D-13) —— ``Esp02Config`` の全葉フィールドの被覆.

01 と同じ防衛線を 02 にも張る。ただし**設定クラスは分ける**。02 用フィールドを
``ExperimentConfig`` に相乗りさせると
``tests/test_config_wiring.py::test_each_parameter_changes_output``
(「全フィールドが 01 のパイプライン出力を変える」) を満たせないフィールドが
必ず生まれ、逃がすために例外チャネルを増やすと 01 の検出力そのものが落ちる。

**被覆の3系統**。``Esp02Config`` の葉は効き方が3つに分かれ、それぞれ別の場所で
「実際に効いている」ことが実測される。ここはその**割り当てが漏れていない**ことを
機械的に固定する。

- ``CHANNEL_SEEDS``: 乱数ストリームの基底シード。``esp_stream_seed`` +
  ``make_rng_for`` は T2 の実装なので、ここで**実際に乱数列が変わること**を測る。
- ``CHANNEL_DIAGNOSTIC``: 診断の判定基準 (D-15 の ``cfg``)。効きは
  ``tests/test_diagnostics_esp.py::test_esp_config_fields_change_output`` が
  診断単体のレベルで実測済み。ここでは
  「この葉が確かにその設定クラスのフィールドである」ことを assert して委譲を
  機械的に閉じる (T1 の ``test_all_config_fields_have_a_case`` が、その
  設定クラスの全フィールドにケースがあることを別途強制している)。
- ``CHANNEL_PENDING``: 消費側 (実験層 ``experiment/esp.py``) がまだ存在しない葉。
  **サイクル 2a (T1+T2) の時点では「出力が変わる」を測れない**。ここでは
  YAML → ``Esp02Config`` の経路 (= T2 の成果物) だけを実測し、実験層が生えた
  瞬間に ``test_pending_cases_disappear_once_the_experiment_layer_exists`` が
  赤くなるようにしてある (先送りが黙って居座らないようにするため)。

``washout.base.*`` は 01 の ``ExperimentConfig`` をまるごと内包した部分なので、
被覆は 01 側 (``tests/test_config_wiring.py``) に**委譲**する。委譲先と過不足なく
一致していることは ``test_all_esp_config_fields_are_covered`` が assert する。
"""

from __future__ import annotations

import dataclasses
import pkgutil
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import yaml
from wiring import (
    WiringCase,
    apply_case,
    assert_yaml_has_all_leaves,
    case,
    leaf_paths,
    plain,
)

import rc_basics_lab.experiment as experiment_pkg
from rc_basics_lab.config import (
    ConfigError,
    Esp02Config,
    EspConfig,
    ExperimentConfig,
    LyapunovConfig,
    TimescaleConfig,
    esp_stream_seed,
    load_config,
    load_config_as,
)
from rc_basics_lab.seeds import SeedStream, make_rng_for

if TYPE_CHECKING:  # pragma: no cover - 型検査時のみ必要
    from _typeshed import DataclassInstance

CHANNEL_SEEDS = "seeds"
"""基底シードを変えると、そのストリームの乱数列だけが変わる。"""

CHANNEL_DIAGNOSTIC = "diagnostic"
"""診断の判定基準。効きは tests/test_diagnostics_esp.py が実測する (D-15)。"""

CHANNEL_PENDING = "pending"
"""消費側が T3 / T4 で生える葉。2a では YAML→設定の経路だけを実測する。"""

DELEGATED_PREFIX = "washout.base."
"""01 の ``ExperimentConfig`` を内包した部分。被覆は 01 側に委譲する。"""

KNOWN_EXPERIMENT_MODULES = frozenset(
    {"pipeline", "report", "runner", "split", "state_space", "summary"}
)
"""サイクル1 (01) 時点で存在する ``experiment/`` 配下の公開モジュール集合。

F-02-1-005: 以前の信管は ``find_spec("rc_basics_lab.experiment.esp")`` という
モジュール名1個だけを見ていた。T3 は ``experiment/esp.py`` と
``experiment/esp_pipeline.py`` の2本を作る計画で、実装順によっては
``esp_pipeline.py`` が先に生え、実験層が動き出しているのに信管が沈黙する
期間が生まれ得る。信管をこの既知集合との差分 (= 新しい公開モジュールが
1本でも増えたか) に広げることで、02 の実験層がどの名前で追加されても
(``esp.py`` だろうと ``esp_pipeline.py`` だろうと) 発火するようにする。"""

DIAGNOSTIC_SECTIONS: tuple[tuple[str, type[DataclassInstance]], ...] = (
    ("esp", EspConfig),
    ("lyapunov", LyapunovConfig),
    ("timescale", TimescaleConfig),
)
"""``Esp02Config`` のセクション名と、対応する診断側の設定クラス (D-15)。"""

PENDING_SECTIONS = frozenset(
    {"name", "drive", "reservoir", "decay", "timescale_sweep", "esp_map", "washout"}
)
"""``CHANNEL_PENDING`` を名乗ってよいセクション。

``seeds`` / ``esp`` / ``lyapunov`` / ``timescale`` はサイクル 2a の時点で
効きを実測できる (できる検査を pending へ逃がすのを禁じる)。
"""

_N_BYTES = 32

ESP_SEED_STREAMS: tuple[SeedStream, ...] = (
    SeedStream.RESERVOIR,
    SeedStream.TASK,
    SeedStream.PROBE,
)
"""02 の実験 2-A / 2-B / 2-C が使うストリーム (``SPLIT`` は 2-D 側)。"""


def base_config() -> Esp02Config:
    """既定値そのままの 02 設定 (このファイルは実験を回さないので縮小不要)。"""
    return Esp02Config()


def _seeds_case(field: str, value: int, stream: SeedStream) -> WiringCase:
    """基底シード1本ぶんのケース。``scope`` に変化してよいストリームを書く。"""
    return case(field, value, channel=CHANNEL_SEEDS, scope=stream.value)


def _diagnostic_case(field: str, value: object) -> WiringCase:
    return case(
        field,
        value,
        channel=CHANNEL_DIAGNOSTIC,
        note="効きは test_diagnostics_esp.py::test_esp_config_fields_change_output",
    )


def _pending_case(field: str, value: object, task: str) -> WiringCase:
    return case(
        field,
        value,
        channel=CHANNEL_PENDING,
        note=f"{task}: 消費する実験層がまだ無いため出力での実測は {task} で行う",
    )


ESP_WIRING_CASES: tuple[WiringCase, ...] = (
    _pending_case("name", "02-renamed", "T3"),
    _seeds_case("seeds.reservoir", 100, SeedStream.RESERVOIR),
    _seeds_case("seeds.drive", 101, SeedStream.TASK),
    _seeds_case("seeds.probe", 102, SeedStream.PROBE),
    # --- 駆動入力と2軌道の生成条件 (2-A / 2-B / 2-C 共通) ---
    _pending_case("drive.distribution", "gaussian", "T3"),
    _pending_case("drive.n_steps", 1200, "T3"),
    _pending_case("drive.washout", 300, "T3"),
    _pending_case("drive.n_pairs", 5, "T3"),
    # --- 2-A/2-B/2-C 共有: リザバー構造 (F-02-1-004, セクション横断で1本) ---
    _pending_case("reservoir.input_scale", 2.0, "T3"),
    _pending_case("reservoir.n_units", 40, "T3"),
    _pending_case("reservoir.density", 0.5, "T3"),
    _pending_case("reservoir.n_replicates", 2, "T3"),
    # --- 2-A: ESP の減衰曲線 ---
    _pending_case("decay.rho_grid", (0.6, 1.3), "T3"),
    _pending_case("decay.sigma_u", 0.5, "T3"),
    _pending_case("decay.leak_rate", 0.4, "T3"),
    # --- 2-B: リーク率と実効時定数 ---
    _pending_case("timescale_sweep.leak_rate_grid", (0.2, 0.9), "T3"),
    _pending_case("timescale_sweep.rho", 0.7, "T3"),
    _pending_case("timescale_sweep.sigma_u", 0.9, "T3"),
    # --- 2-C: rho x 入力強度 の ESP 成立領域 ---
    _pending_case("esp_map.rho_grid", (0.8, 1.4), "T3"),
    _pending_case("esp_map.sigma_grid", (0.0, 1.5), "T3"),
    _pending_case("esp_map.leak_rate", 0.6, "T3"),
    # --- 2-D: washout 感度 (base.* は 01 側へ委譲) ---
    _pending_case("washout.grid", (0, 100), "T4"),
    _pending_case("washout.pad_series", False, "T4"),
    # --- 診断の判定基準 (D-15) ---
    _diagnostic_case("esp.abs_tol", 1.0e-8),
    _diagnostic_case("esp.rel_tol", 1.0e-5),
    _diagnostic_case("esp.window", 500),
    _diagnostic_case("esp.fit_skip", 400),
    _diagnostic_case("esp.floor", 1.0e-6),
    _diagnostic_case("lyapunov.method", "jacobian"),
    _diagnostic_case("lyapunov.delta", 1.0e-6),
    _diagnostic_case("lyapunov.renorm_interval", 5),
    _diagnostic_case("lyapunov.max_growth", 1.0e6),
    _diagnostic_case("lyapunov.check_propagator", False),
    _diagnostic_case("lyapunov.propagator_tol", 1.0e-5),
    _diagnostic_case("timescale.max_lag", 100),
)


def _seed_fingerprints(config: Esp02Config) -> dict[SeedStream, bytes]:
    """02 が使う各ストリームの乱数列の先頭バイト列。"""
    return {
        stream: make_rng_for(esp_stream_seed(config.seeds, stream), stream, 0).bytes(
            _N_BYTES
        )
        for stream in ESP_SEED_STREAMS
    }


def _diagnostic_field_names(section: str) -> set[str]:
    """診断セクションの葉フィールド名 (``esp.abs_tol`` -> ``abs_tol``)。"""
    config_type = dict(DIAGNOSTIC_SECTIONS)[section]
    return {item.name for item in fields(config_type)}


def _changed_leaves(base: Esp02Config, changed: Esp02Config) -> set[str]:
    """2つの設定で値が異なる葉フィールドのパス集合。"""
    return {
        leaf
        for leaf in leaf_paths(Esp02Config)
        if _leaf_value(base, leaf) != _leaf_value(changed, leaf)
    }


def _leaf_value(config: object, leaf: str) -> object:
    node: object = config
    for part in leaf.split("."):
        node = getattr(node, part)
    return node


def _round_trip(config: Esp02Config, tmp_path: Path, name: str) -> Esp02Config:
    """設定を YAML へ書き出して読み直す (T2 が実装した経路そのもの)。"""
    path = tmp_path / f"{name}.yaml"
    dumped = cast("Mapping[str, object]", plain(dataclasses.asdict(config)))
    path.write_text(yaml.safe_dump(dumped, allow_unicode=True), encoding="utf-8")
    return load_config_as(path, Esp02Config)


@pytest.mark.parametrize(
    "wiring_case", ESP_WIRING_CASES, ids=[item.field for item in ESP_WIRING_CASES]
)
def test_every_esp_parameter_changes_output(
    wiring_case: WiringCase, tmp_path: Path
) -> None:
    """各パラメータが「効く経路」を実際に持っていることの実測。

    何を「出力」と見なすかはチャネルごとに違う (モジュール docstring 参照)。
    どのチャネルでも共通して、**値が YAML を往復してその葉にだけ届く**ことは
    実測する。これは T2 が実装したローダそのものの検査であり、
    「YAML に書いたのに設定オブジェクトへ届いていない」を殺す。
    """
    base = base_config()
    changed = apply_case(base, wiring_case)
    assert changed != base, "差し替えが設定に反映されていません"

    # 共通: YAML を往復しても値が保たれ、変わったのはその葉だけであること
    assert _round_trip(changed, tmp_path, "changed") == changed
    assert _changed_leaves(base, changed) == {wiring_case.field}, (
        f"{wiring_case.field} の差し替えが他の葉にも波及しています"
    )

    if wiring_case.channel == CHANNEL_SEEDS:
        before = _seed_fingerprints(base)
        after = _seed_fingerprints(changed)
        moved = {
            stream for stream in ESP_SEED_STREAMS if before[stream] != after[stream]
        }
        names = sorted(stream.value for stream in moved)
        assert set(names) == {wiring_case.scope}, (
            f"{wiring_case.field} が動かしたストリーム: {names}"
        )
        return

    if wiring_case.channel == CHANNEL_DIAGNOSTIC:
        section, _, name = wiring_case.field.partition(".")
        assert name in _diagnostic_field_names(section), (
            f"{wiring_case.field} は診断の設定クラスのフィールドではありません"
        )
        return

    assert wiring_case.channel == CHANNEL_PENDING, wiring_case.channel
    assert wiring_case.field.split(".")[0] in PENDING_SECTIONS, (
        f"{wiring_case.field} は 2a の時点で効きを実測できるはずです"
    )
    assert wiring_case.note, "pending の理由が書かれていません"


def test_all_esp_config_fields_are_covered() -> None:
    """``Esp02Config`` の全葉が被覆されている (D-13 guard)。

    02 でパラメータを足したとき、ここに1行足すまでテストが赤になる。
    ``washout.base.*`` だけは 01 側へ委譲するが、**委譲先と過不足なく一致する**
    ことまで assert する。一致を確かめずに接頭辞で除外すると、01 に無い
    フィールドを ``washout.base`` の下に足して被覆から逃がせてしまう。
    """
    all_leaves = leaf_paths(Esp02Config)
    delegated = {leaf for leaf in all_leaves if leaf.startswith(DELEGATED_PREFIX)}
    assert delegated == {
        f"{DELEGATED_PREFIX}{leaf}" for leaf in leaf_paths(ExperimentConfig)
    }, "washout.base 配下が 01 の ExperimentConfig と一致していません"

    covered = {item.field for item in ESP_WIRING_CASES}
    expected = all_leaves - delegated
    assert covered == expected, (
        f"未登録: {sorted(expected - covered)} / 余分: {sorted(covered - expected)}"
    )
    # 差し替えパスの綴りも同じ集合の中にあること (typo で別フィールドを触らない)
    for item in ESP_WIRING_CASES:
        for path, _ in item.overrides:
            assert path in expected, f"未知のパスです: {path}"


def test_diagnostic_sections_cover_the_diagnostic_config_classes() -> None:
    """診断セクションの葉が、診断側の設定クラスの全フィールドと一致する。

    ``CHANNEL_DIAGNOSTIC`` の効きの実測は
    ``tests/test_diagnostics_esp.py::test_esp_config_fields_change_output`` へ
    委譲している。その委譲が閉じているのは、あちら側の
    ``test_all_config_fields_have_a_case`` が「設定クラスの全フィールドに
    ケースがある」ことを強制しており、こちらで「セクションの全葉が設定クラスの
    全フィールドである」ことを固定するため。片方が欠けると委譲に穴が空く。
    """
    for section, config_type in DIAGNOSTIC_SECTIONS:
        leaves = {
            leaf.partition(".")[2]
            for leaf in leaf_paths(Esp02Config)
            if leaf.startswith(f"{section}.")
        }
        assert leaves == {item.name for item in fields(config_type)}, (
            f"{section} セクションが {config_type.__name__} と一致していません"
        )


def test_pending_cases_disappear_once_the_experiment_layer_exists() -> None:
    """実験層が生えたら ``CHANNEL_PENDING`` は許されない (先送りの時限装置)。

    サイクル 2a には ``experiment/esp.py`` (も ``esp_pipeline.py``) も無いため、
    格子や系列長のような「実験を回して初めて効く」葉は出力での実測ができない。
    そこを黙って見逃すと「設定したのに効いていない」が 02 で復活するので、
    実験層が生えた瞬間にこのテストが赤くなるようにしてある。
    T3 では各 pending ケースを実際の出力チャネルへ書き換えること。

    F-02-1-005: 信管は特定のモジュール名1個ではなく、``KNOWN_EXPERIMENT_MODULES``
    (01 時点の公開モジュール集合) を ``pkgutil.iter_modules`` で列挙した実際の
    集合が超えたかどうかで判定する。これにより T3 がどの名前でモジュールを
    追加しても (``esp.py`` でも ``esp_pipeline.py`` でも、実装順に関わらず)
    発火する。
    """
    pending = sorted(
        item.field for item in ESP_WIRING_CASES if item.channel == CHANNEL_PENDING
    )
    current_modules = {
        info.name
        for info in pkgutil.iter_modules(experiment_pkg.__path__)
        if not info.name.startswith("_")
    }
    new_modules = sorted(current_modules - KNOWN_EXPERIMENT_MODULES)
    assert not (new_modules and pending), (
        f"rc_basics_lab.experiment に既知集合を超える新規モジュール "
        f"{new_modules} が追加されているのに未実測の葉が残っています: {pending}"
    )
    # 実測できるチャネルが pending へ逃げていないこと
    assert {field.split(".")[0] for field in pending} <= PENDING_SECTIONS


def test_every_esp_field_round_trips_yaml(tmp_path: Path) -> None:
    """``Esp02Config`` の全フィールドが YAML のキーとして実在し往復する。

    「dataclass には在るが YAML からは設定できない」パラメータを作らないための
    検査 (D-09 の未知キー検査と対になる)。``washout.base.*`` も含めて確かめる
    ので、01 の設定を内包した部分が YAML から届かない事故も落ちる。
    """
    config = Esp02Config()
    path = tmp_path / "roundtrip.yaml"
    dumped = cast("Mapping[str, object]", plain(dataclasses.asdict(config)))
    path.write_text(yaml.safe_dump(dumped, allow_unicode=True), encoding="utf-8")
    assert load_config_as(path, Esp02Config) == config
    assert_yaml_has_all_leaves(
        yaml.safe_load(path.read_text(encoding="utf-8")), Esp02Config
    )


def test_empty_yaml_gives_esp_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config_as(path, Esp02Config) == Esp02Config()


@pytest.mark.parametrize(
    ("yaml_text", "match"),
    [
        pytest.param("n_replicates: 3\n", "n_replicates", id="top_level"),
        pytest.param("esp:\n  abs_tolerance: 1.0\n", "abs_tolerance", id="diagnostic"),
        pytest.param("seeds:\n  task: 1\n", "task", id="seed_stream_name"),
        pytest.param(
            "washout:\n  base:\n    n_replicate: 3\n", "n_replicate", id="nested_01"
        ),
    ],
)
def test_unknown_key_raises_for_esp_config(
    tmp_path: Path, yaml_text: str, match: str
) -> None:
    """02 の YAML も未知キーで即座に落ちる (D-09)。

    ``seeds.task`` を弾くのは、01 の ``SeedConfig`` のキー名をそのまま書いても
    通らないことの確認 (02 の駆動信号のシードは ``seeds.drive``)。似た名前の
    キーが黙って無視されると、シードを変えたつもりで既定値のまま回る。
    """
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_config_as(path, Esp02Config)


def test_load_config_is_load_config_as_for_experiment_config(tmp_path: Path) -> None:
    """``load_config`` は ``load_config_as`` への委譲 (既存呼び出しの互換)。"""
    path = tmp_path / "01.yaml"
    path.write_text("n_replicates: 3\nseeds:\n  reservoir: 11\n", encoding="utf-8")
    assert load_config(path) == load_config_as(path, ExperimentConfig)


def test_esp_config_does_not_leak_into_experiment_config() -> None:
    """01 の ``ExperimentConfig`` に 02 のフィールドが1つも増えていない (D-13)。

    増えた瞬間に 01 の ``test_each_parameter_changes_output`` が
    「01 のパイプライン出力を変えないフィールド」を抱えることになる。
    """
    assert {item.name for item in fields(ExperimentConfig)} == {
        "name",
        "n_replicates",
        "seeds",
        "split",
        "ridge",
        "mackey_glass",
        "delay_parity",
        "esn_mackey_glass",
        "esn_delay_parity",
    }


def test_split_stream_is_rejected_for_esp_seeds() -> None:
    """2-A/2-B/2-C は分割を行わないので ``SPLIT`` は取り出せない (D-14)。"""
    with pytest.raises(ValueError, match="SPLIT"):
        esp_stream_seed(Esp02Config().seeds, SeedStream.SPLIT)

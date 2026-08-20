"""04 の設定配線テスト (D-13) —— ``Chaos04Config`` の全葉フィールドの被覆.

01 (``tests/test_config_wiring.py``) / 02 (``tests/test_config_wiring_esp.py``) /
03 (``tests/test_config_wiring_capacity.py``) と同じ防衛線を 04 にも張る。
**設定クラスは分ける** (D-13): 04 用フィールドを ``ExperimentConfig`` に
相乗りさせると 01 の ``test_each_parameter_changes_output`` を満たせない
フィールドが必ず生まれ、逃がすために例外チャネルを増やすと 01 の検出力そのものが
落ちる。

**被覆の3系統**。

- ``CHANNEL_ROWS``: 値を変えると出力の指紋が変わる。``scope`` を伴うものは
  **その課題の出力だけ**が変わることまで測る (セクション固有の葉が他の課題へ
  漏れていないことの実測)。
- ``CHANNEL_META``: 出力は変えないが ``meta.json`` を変える (``name``)。
- **委譲**: ``base.*`` は 01 の ``ExperimentConfig`` をまるごと内包した部分、
  ``lyapunov.*`` / ``mc.*`` / ``ipc.*`` は診断層の設定 (D-15) をそのまま
  載せた部分なので、効きの実測は各々の持ち主へ委譲する。**委譲先と過不足なく
  一致する**ことまで assert するのが要点で、一致を確かめずに接頭辞で除外すると、
  委譲先に無いフィールドをその下に足して被覆から逃がせてしまう。

「出力」は **4-A の結果行 + 自走の結果**の組である。自走を含めるのは、
``freerun.*`` が 4-A の行を1バイトも変えない (自走にしか効かない) ためで、
4-A の行だけを見ると ``freerun.*`` の配線を実測できない。
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import fields
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import yaml
from wiring import (
    CHANNEL_META,
    CHANNEL_ROWS,
    WiringCase,
    apply_case,
    assert_yaml_has_all_leaves,
    case,
    leaf_paths,
    plain,
)

from rc_basics_lab.config import (
    Chaos04Config,
    ConfigError,
    ESNConfig,
    ExperimentConfig,
    FreeRunConfig,
    IpcConfig,
    LorenzConfig,
    MackeyGlassConfig,
    MackeyGlassStandardizeConfig,
    MaxLyapunovConfig,
    MemoryCapacityConfig,
    RidgeConfig,
    SplitConfig,
    load_config_as,
)
from rc_basics_lab.diagnostics.ipc import count_targets
from rc_basics_lab.experiment.freerun import (
    chaos_task_entries,
    run_free_run,
    run_onestep,
)
from rc_basics_lab.experiment.runner import ResultRow
from rc_basics_lab.meta import collect_meta_for
from rc_basics_lab.tasks.chaotic import TASK_NAME_LORENZ
from rc_basics_lab.tasks.mackey_glass import TASK_NAME as TASK_NAME_MACKEY_GLASS

if TYPE_CHECKING:  # pragma: no cover - 型検査時のみ必要
    from _typeshed import DataclassInstance

DELEGATED_SECTIONS: tuple[tuple[str, type], ...] = (
    ("base.", ExperimentConfig),
    ("lyapunov.", MaxLyapunovConfig),
    ("mc.", MemoryCapacityConfig),
    ("ipc.", IpcConfig),
)
"""接頭辞と、その配下が過不足なく一致すべき委譲先の設定クラス。

``base`` は 01 の ``ExperimentConfig`` をまるごと内包した部分で、
``Narma10Config.base`` (03) / ``WashoutSweepConfig.base`` (02) と同じ形。
被覆は 01 側 (``tests/test_config_wiring.py``) へ委譲する。

``lyapunov`` / ``mc`` / ``ipc`` は 3a・04b-1 の診断設定 (D-15)。効きは
``tests/test_diagnostics_lyapunov.py`` /
``tests/test_diagnostics_memory_capacity.py`` / ``tests/test_diagnostics_ipc.py``
が診断単体のレベルで実測済みで、あちら側の完全性チェックが「設定クラスの全
フィールドにケースがある」ことを強制している。こちらで「セクションの全葉が
その設定クラスの全フィールドである」ことを固定することで、委譲が両側から閉じる。
"""

TASK_LABELS: tuple[str, ...] = (TASK_NAME_LORENZ, TASK_NAME_MACKEY_GLASS)
"""``scope`` に書ける課題名 (``ResultRow.task`` と ``FreeRunOutcome.task``)。"""

_FREE_RUN_REPLICATE = 0
"""指紋を取る自走のレプリケート (縮小設定なので1本で足りる)。"""


def base_config() -> Chaos04Config:
    """秒未満で 4-A と自走を回せる縮小設定 (**構造は本番と同じ**)。

    Lorenz と MG で ``length`` / ``standardize_steps`` を**別の値**にしてある
    のは、課題間で値を取り違える配線を落とすためである。
    """
    return Chaos04Config(
        name="chaos-wiring",
        base=ExperimentConfig(
            name="chaos-wiring-base",
            n_replicates=1,
            split=SplitConfig(washout=30, max_start_offset=10),
            ridge=RidgeConfig(alpha_grid=(1.0e-6, 1.0e-3), n_lags_grid=(2, 4)),
            mackey_glass=MackeyGlassConfig(length=700, integration_burn_in=100),
            esn_mackey_glass=ESNConfig(
                n_units=15, leak_rate=0.5, input_scale=0.5, density=0.5
            ),
        ),
        lorenz=LorenzConfig(
            rk4_step=0.002,
            sample_interval=5,
            integration_burn_in=100,
            length=600,
            horizon=1,
            standardize_steps=150,
        ),
        mackey_glass=MackeyGlassStandardizeConfig(standardize_steps=170),
        freerun=FreeRunConfig(warmup_steps=10, free_run_steps=40),
    )


def _section_case(field: str, value: object, scope: str) -> WiringCase:
    """課題固有の葉。**その課題の出力だけ**が変わることまで測る。"""
    return case(field, value, scope=scope)


CHAOS_WIRING_CASES: tuple[WiringCase, ...] = (
    # name は結果行に出ない純粋なメタ情報。meta.json に載ることを確かめる。
    case("name", "04-renamed", channel=CHANNEL_META),
    # --- Lorenz の生成パラメータ (Lorenz の出力だけが動く) ---
    _section_case("lorenz.rk4_step", 0.001, TASK_NAME_LORENZ),
    _section_case("lorenz.sample_interval", 8, TASK_NAME_LORENZ),
    _section_case("lorenz.integration_burn_in", 150, TASK_NAME_LORENZ),
    _section_case("lorenz.length", 650, TASK_NAME_LORENZ),
    _section_case("lorenz.horizon", 3, TASK_NAME_LORENZ),
    _section_case("lorenz.standardize_steps", 200, TASK_NAME_LORENZ),
    # --- 04 の MG 課題の標準化 (MG の出力だけが動く) ---
    _section_case("mackey_glass.standardize_steps", 220, TASK_NAME_MACKEY_GLASS),
    # --- 自走の実行条件 (課題横断。4-A の行は1バイトも変わらない) ---
    case("freerun.warmup_steps", 25),
    case("freerun.free_run_steps", 60),
)


def _onestep_fingerprint(rows: tuple[ResultRow, ...], task: str | None) -> object:
    """4-A の結果行の指紋 (実測時間の列だけ除く)。"""
    return [
        {
            item.name: getattr(row, item.name)
            for item in fields(ResultRow)
            if item.name != "wall_time_s"
        }
        for row in rows
        if task is None or row.task == task
    ]


def _round4(array: object) -> object:
    """自走の予測を丸めた入れ子リストにする (指紋を JSON 化できる形へ)。"""
    return np.round(np.asarray(array, dtype=np.float64), 10).tolist()


def run_config(config: Chaos04Config, task: str | None = None) -> str:
    """縮小設定で 4-A と自走を回し、出力の指紋を返す。

    ``task`` を渡すとその課題の出力だけを見る。課題固有の葉が他方の課題を
    動かしていないことの確認に使う。
    """
    rows = tuple(run_onestep(config))
    freerun: list[object] = []
    for entry in chaos_task_entries(config):
        if task is not None and entry.name != task:
            continue
        outcome = run_free_run(config, entry, _FREE_RUN_REPLICATE)
        freerun.append(
            {
                "task": outcome.task,
                "switch_index": outcome.switch_index,
                "alpha": outcome.readout.alpha,
                "n_completed": outcome.result.n_completed,
                "diverged": outcome.result.diverged,
                "predictions": _round4(outcome.result.predictions),
                "truth": _round4(outcome.truth),
            }
        )
    return json.dumps(
        {"onestep": _onestep_fingerprint(rows, task), "freerun": freerun},
        sort_keys=True,
        default=str,
    )


@lru_cache(maxsize=1)
def baseline() -> Chaos04Config:
    """基準となる縮小設定 (ケースごとに作り直さない)。"""
    return base_config()


@lru_cache(maxsize=8)
def baseline_fingerprint(task: str | None = None) -> str:
    """基準の出力の指紋 (ケースごとに再計算しない)。"""
    return run_config(baseline(), task)


def _leaf_value(config: object, leaf: str) -> object:
    node: object = config
    for part in leaf.split("."):
        node = getattr(node, part)
    return node


def _changed_leaves(base: Chaos04Config, changed: Chaos04Config) -> set[str]:
    return {
        leaf
        for leaf in leaf_paths(Chaos04Config)
        if _leaf_value(base, leaf) != _leaf_value(changed, leaf)
    }


def _round_trip(config: Chaos04Config, tmp_path: Path, name: str) -> Chaos04Config:
    """設定を YAML へ書き出して読み直す (``load_config_as`` の経路そのもの)。"""
    path = tmp_path / f"{name}.yaml"
    dumped = cast("Mapping[str, object]", plain(dataclasses.asdict(config)))
    path.write_text(yaml.safe_dump(dumped, allow_unicode=True), encoding="utf-8")
    return load_config_as(path, Chaos04Config)


def _meta_fingerprint(config: Chaos04Config) -> str:
    """``meta.json`` に載る設定ダンプの指紋。"""
    meta = collect_meta_for(config, config.base.seeds)
    return json.dumps(plain(meta["config"]), sort_keys=True, default=str)


@pytest.mark.parametrize(
    "wiring_case",
    CHAOS_WIRING_CASES,
    ids=[item.field for item in CHAOS_WIRING_CASES],
)
def test_each_chaos_parameter_changes_output(
    wiring_case: WiringCase, tmp_path: Path
) -> None:
    """各パラメータが「効く経路」を実際に持っていることの実測。

    どのチャネルでも共通して、**値が YAML を往復してその葉にだけ届く**ことは
    実測する。「YAML に書いたのに設定オブジェクトへ届いていない」を殺す。
    """
    base = baseline()
    changed_config = apply_case(base, wiring_case)
    assert changed_config != base, "差し替えが設定に反映されていません"

    assert _round_trip(changed_config, tmp_path, "changed") == changed_config
    assert _changed_leaves(base, changed_config) == {wiring_case.field}, (
        f"{wiring_case.field} の差し替えが他の葉にも波及しています"
    )

    if wiring_case.channel == CHANNEL_META:
        assert run_config(changed_config) == baseline_fingerprint(), (
            "メタ情報のはずが出力を変えています"
        )
        assert _meta_fingerprint(changed_config) != _meta_fingerprint(base)
        return

    assert wiring_case.channel == CHANNEL_ROWS, wiring_case.channel
    assert run_config(changed_config) != baseline_fingerprint(), (
        f"{wiring_case.field} を変えても出力が変わりません (配線漏れ)"
    )
    if wiring_case.scope is not None:
        assert run_config(changed_config, wiring_case.scope) != baseline_fingerprint(
            wiring_case.scope
        )
        for other in TASK_LABELS:
            if other != wiring_case.scope:
                assert run_config(changed_config, other) == baseline_fingerprint(
                    other
                ), f"{wiring_case.field} が {other} の結果まで変えています"


def test_free_run_only_leaves_do_not_change_the_one_step_rows() -> None:
    """``freerun.*`` は自走にしか効かず 4-A の行を1バイトも変えない。

    「出力」を 4-A の行だけにすると ``freerun.*`` の配線が実測できない、という
    このファイルの設計そのものの実測でもある。
    """
    base = baseline()
    base_rows = tuple(run_onestep(base))
    for field, value in (("freerun.warmup_steps", 25), ("freerun.free_run_steps", 60)):
        changed = apply_case(base, case(field, value))
        assert _onestep_fingerprint(
            tuple(run_onestep(changed)), None
        ) == _onestep_fingerprint(base_rows, None), f"{field} が 4-A の行を変えています"


def test_all_chaos_config_fields_are_covered() -> None:
    """``Chaos04Config`` の全葉が被覆されている (D-13 guard)。

    04 でパラメータを足したとき、ここに1行足すまでテストが赤になる。
    ``base.*`` (01 の設定) と ``lyapunov.*`` / ``mc.*`` / ``ipc.*`` (診断設定)
    は委譲するが、**委譲先と過不足なく一致する**ことまで assert する。一致を
    確かめずに接頭辞で除外すると、委譲先に無いフィールドをその下に足して
    被覆から逃がせてしまう。
    """
    all_leaves = leaf_paths(Chaos04Config)
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

    covered = {item.field for item in CHAOS_WIRING_CASES}
    expected = all_leaves - delegated
    assert covered == expected, (
        f"未登録: {sorted(expected - covered)} / 余分: {sorted(covered - expected)}"
    )
    for item in CHAOS_WIRING_CASES:
        for path, _ in item.overrides:
            assert path in expected, f"未知のパスです: {path}"


def test_chaos_ipc_defaults_stay_within_the_existing_bounds() -> None:
    """4-D の目標数が **03 で置いた D-34 の4段の内側**にある (確保軸8)。

    04 で新しい上限を作らない、という決定の実測。``Chaos04Config.ipc`` は
    ``IpcConfig`` そのものなので、``diagnostics/ipc.py`` の ``_validate_config``
    が持つ上限 (``max_targets`` / ``max_degrees`` など) がそのまま効く。
    """
    config = Chaos04Config()
    n_targets = count_targets(config.ipc)
    n_cells = len(config.ipc.max_delay_by_degree) * max(config.ipc.max_delay_by_degree)
    assert n_targets < config.ipc.max_targets
    assert n_cells < config.ipc.max_targets
    assert len(config.ipc.max_delay_by_degree) <= config.ipc.max_degrees


def test_chaos_config_introduces_no_new_capacity_bound() -> None:
    """04 の設定・実験モジュールが新しい容量上限を宣言していない (確保軸8)。

    自走の確保軸3 は 03 の ``validate_state_matrix_bounds`` を再利用し、
    Lorenz の確保軸1・2 は ``tasks/chaotic.py`` に置く。``experiment/freerun.py``
    と ``config/chaos04.py`` に ``_MAX_*`` を足すと、上限の在り処が分散する。
    """
    import rc_basics_lab.config.chaos04 as chaos04_module
    import rc_basics_lab.experiment.freerun as freerun_module

    for module in (chaos04_module, freerun_module):
        offenders = [name for name in vars(module) if name.startswith("_MAX_")]
        assert not offenders, (
            f"{module.__name__} が新しい上限を宣言しています: {offenders}"
        )


def test_every_chaos_field_round_trips_yaml(tmp_path: Path) -> None:
    """``Chaos04Config`` の全フィールドが YAML のキーとして実在し往復する。

    「dataclass には在るが YAML からは設定できない」パラメータを作らないための
    検査 (D-09 の未知キー検査と対になる)。``base.*`` も含めて確かめるので、
    01 の設定を内包した部分が YAML から届かない事故も落ちる。
    """
    config = Chaos04Config()
    path = tmp_path / "roundtrip.yaml"
    dumped = cast("Mapping[str, object]", plain(dataclasses.asdict(config)))
    path.write_text(yaml.safe_dump(dumped, allow_unicode=True), encoding="utf-8")
    assert load_config_as(path, Chaos04Config) == config
    assert_yaml_has_all_leaves(
        yaml.safe_load(path.read_text(encoding="utf-8")), Chaos04Config
    )


def test_empty_yaml_gives_chaos_defaults(tmp_path: Path) -> None:
    """空の YAML は既定値の ``Chaos04Config`` になる。"""
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config_as(path, Chaos04Config) == Chaos04Config()


@pytest.mark.parametrize(
    ("yaml_text", "match"),
    [
        pytest.param("n_replicates: 3\n", "n_replicates", id="top_level"),
        pytest.param("lorenz:\n  sigma: 10.0\n", "sigma", id="lorenz_parameter"),
        pytest.param(
            "mackey_glass:\n  tau: 17.0\n", "tau", id="mg_generation_parameter"
        ),
        pytest.param("lyapunov:\n  reference: 0.9\n", "reference", id="lyapunov_typo"),
        pytest.param("freerun:\n  stats_steps: 100\n", "stats_steps", id="not_yet"),
    ],
)
def test_unknown_chaos_keys_are_rejected(
    yaml_text: str, match: str, tmp_path: Path
) -> None:
    """未知キーは ``ConfigError`` (D-09)。

    ``lorenz.sigma`` が落ちることは D-41 の実測でもある —— Lorenz の系
    パラメータは設定にしないので、YAML に書けば「設定したのに効いていない」
    ではなく即座の失敗になる。``mackey_glass.tau`` が落ちるのは、04 の MG の
    生成パラメータの単一の真実が ``base.mackey_glass`` だからである。
    """
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_config_as(path, Chaos04Config)


def test_production_config_matches_the_committed_yaml() -> None:
    """本番 YAML が既定値から意図した差分しか持たない。

    ``base`` (レプリケート数と実験名) 以外は既定と一致する。既定値を動かした
    のに YAML を更新し忘れる (あるいはその逆) と、``make onestep-04`` の
    再生成結果と ``Chaos04Config()`` を使うテストが静かに食い違う。
    """
    root = Path(__file__).resolve().parents[1]
    config = load_config_as(
        root / "experiments" / "04_chaotic_freerun" / "config.yaml", Chaos04Config
    )
    default = Chaos04Config()
    differing = {
        item.name
        for item in fields(Chaos04Config)
        if getattr(config, item.name) != getattr(default, item.name)
    }
    assert differing == {"base"}
    assert config.base.n_replicates == 10, "シードは10本以上 (受け入れ条件2)"
    assert (
        dataclasses.replace(
            config.base, name=default.base.name, n_replicates=default.base.n_replicates
        )
        == default.base
    )

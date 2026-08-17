"""設定パラメータの配線テスト — 本サイクル最大の失敗モードへの防衛線.

本サイクルは YAML に十数個のパラメータを新設した。最大の失敗モードは
バグではなく**「設定したのに効いていない」**であり、これは出力が正常に見えるため
レビューでは確率的にしか見つからない。ここでは全パラメータについて

1. 値を変えると縮小パイプラインの出力 (``comparison.csv`` 相当の行の指紋) が変わる
   (``test_each_parameter_changes_output``)
2. ``ExperimentConfig`` のフィールドが1つでも上の一覧から漏れたら失敗する
   (``test_all_config_fields_are_covered``) —— 02〜05 でパラメータを足したときに
   自動で強制される
3. 全フィールドが YAML から実際に設定できる (``test_every_field_round_trips_yaml``)

を機械的に確かめる。

**チャネル**: パラメータの効き方は3種類ある。

- ``rows``: 結果行が変わる (ほとんどのパラメータ)
- ``meta``: 結果行は変わらないが ``meta.json`` が変わる (``name`` のような純粋な
  メタ情報。「行が変わらないこと」も併せて固定する)
- ``error``: 値域が1点しかなく、別の値は即座に例外になる (``activation``)

**scope**: 課題別のセクション (``mackey_glass`` / ``delay_parity`` /
``esn_*``) は、担当課題の行だけを変え、**他方の課題の行をバイト単位で変えない**
ことまで確かめる。これにより「片方の課題の設定をもう片方にも使っている」という
配線ミスも落ちる。

**分割比 (``split.*_ratio``) だけは単独で動かせない**。3つの比は合計 1 という
制約 (単体) の上にあり、1つだけ変えると ``make_split`` が正しく ``ValueError``
にする。そこで比のケースは2つを同時に動かし、「どのフィールドが効いたか」は
``n_train`` / ``n_val`` / ``n_test`` の変化で切り分ける (``changed_sizes``)。
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, cast, get_type_hints

import pytest
import yaml

from rc_basics_lab.config import (
    DelayParityConfig,
    ESNConfig,
    ExperimentConfig,
    MackeyGlassConfig,
    RidgeConfig,
    SplitConfig,
    load_config,
)
from rc_basics_lab.experiment.runner import ResultRow, run_experiment
from rc_basics_lab.meta import collect_meta
from rc_basics_lab.seeds import SeedConfig

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

MACKEY_GLASS = "mackey_glass"
DELAY_PARITY = "delay_parity"

CHANNEL_ROWS = "rows"
CHANNEL_META = "meta"
CHANNEL_ERROR = "error"

VOLATILE_COLUMNS = frozenset({"wall_time_s"})
"""指紋から外す列 (実測時間は実行ごとに変わる)。"""

EXPERIMENT_CONFIG_PATH = Path("experiments/01_what_is_rc/config.yaml")
REQUIRED_REPLICATES = 5
"""受け入れ条件3 が要求するシード本数。"""


def base_config() -> ExperimentConfig:
    """秒未満で両課題を1レプリケート回せる縮小設定。

    構造は本番 (``experiments/01_what_is_rc/config.yaml``) と同じで、長さと
    ユニット数だけを削ってある。``n_lags_grid`` の最大値は washout より小さく
    しておき、``ridge.n_lags_grid`` を大きくしたときに ``t0`` が動くことも
    見えるようにしている。
    """
    return ExperimentConfig(
        name="wiring",
        n_replicates=1,
        seeds=SeedConfig(reservoir=0, task=1, split=2),
        split=SplitConfig(
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            washout=20,
            max_start_offset=10,
        ),
        ridge=RidgeConfig(alpha_grid=(1e-4, 1e-1), n_lags_grid=(1, 4)),
        mackey_glass=MackeyGlassConfig(length=300, integration_burn_in=50),
        delay_parity=DelayParityConfig(n_bits=2, delay=1, length=300),
        esn_mackey_glass=ESNConfig(n_units=20, density=0.3),
        esn_delay_parity=ESNConfig(
            n_units=20, density=0.3, leak_rate=1.0, input_scale=1.0
        ),
    )


@dataclass(frozen=True, slots=True)
class WiringCase:
    """1パラメータぶんの配線テスト。

    Attributes:
        field: 検査対象のフィールド (ドット区切り)。coverage の単位。
        overrides: 実際に差し替える (パス, 値) の並び。制約のあるフィールド
            (分割比) だけ2つ以上になる。
        channel: 効き方 (``rows`` / ``meta`` / ``error``)。
        scope: 変化してよい課題名。``None`` なら全課題。
        changed_sizes: 変化していることを追加で要求する分割サイズの列名。
        note: 単独で動かせない等の理由。
    """

    field: str
    overrides: tuple[tuple[str, object], ...]
    channel: str = CHANNEL_ROWS
    scope: str | None = None
    changed_sizes: tuple[str, ...] = ()
    note: str = ""


def case(
    field: str,
    value: object,
    *,
    channel: str = CHANNEL_ROWS,
    scope: str | None = None,
) -> WiringCase:
    """単独で動かせるパラメータのケース。"""
    return WiringCase(
        field=field, overrides=((field, value),), channel=channel, scope=scope
    )


def _esn_cases(section: str, scope: str, leak_rate: float) -> tuple[WiringCase, ...]:
    """ESN セクション1つぶんのケース (2セクションで同じ8フィールドを持つ)。"""
    return (
        case(f"{section}.n_units", 30, scope=scope),
        case(f"{section}.spectral_radius", 0.3, scope=scope),
        case(f"{section}.leak_rate", leak_rate, scope=scope),
        case(f"{section}.input_scale", 2.0, scope=scope),
        case(f"{section}.bias_scale", 1.0, scope=scope),
        case(f"{section}.density", 0.9, scope=scope),
        case(f"{section}.state_noise", 0.05, scope=scope),
        # 活性化関数は tanh 以外を受け付けない。黙って tanh として扱わないこと
        # 自体が配線 (未対応の値が静かに通ると「設定したのに効かない」になる)。
        case(f"{section}.activation", "relu", channel=CHANNEL_ERROR),
    )


WIRING_CASES: tuple[WiringCase, ...] = (
    # name は結果行に出ない純粋なメタ情報。meta.json に載ることを確かめる。
    case("name", "wiring-renamed", channel=CHANNEL_META),
    case("n_replicates", 2),
    case("seeds.reservoir", 100),
    case("seeds.task", 101),
    case("seeds.split", 7),
    WiringCase(
        field="split.train_ratio",
        overrides=(("split.train_ratio", 0.6), ("split.test_ratio", 0.15)),
        changed_sizes=("n_train",),
        note="比の合計は 1 に固定されるため単独では動かせない (test_ratio で調整)",
    ),
    WiringCase(
        field="split.val_ratio",
        overrides=(("split.val_ratio", 0.35), ("split.test_ratio", 0.15)),
        changed_sizes=("n_val",),
        note="同上。train_ratio を据え置くので n_train は不変になるはず",
    ),
    WiringCase(
        field="split.test_ratio",
        overrides=(("split.test_ratio", 0.4), ("split.val_ratio", 0.1)),
        changed_sizes=("n_test",),
        note="同上。n_test は残りの行数として決まる",
    ),
    case("split.washout", 60),
    case("split.max_start_offset", 40),
    case("ridge.alpha_grid", (1e3, 1e4)),
    case("ridge.n_lags_grid", (1, 40)),
    case("mackey_glass.tau", 20.0, scope=MACKEY_GLASS),
    case("mackey_glass.beta", 0.3, scope=MACKEY_GLASS),
    case("mackey_glass.gamma", 0.15, scope=MACKEY_GLASS),
    case("mackey_glass.exponent", 8, scope=MACKEY_GLASS),
    case("mackey_glass.rk4_step", 0.2, scope=MACKEY_GLASS),
    case("mackey_glass.sample_interval", 5, scope=MACKEY_GLASS),
    case("mackey_glass.integration_burn_in", 80, scope=MACKEY_GLASS),
    case("mackey_glass.length", 260, scope=MACKEY_GLASS),
    case("mackey_glass.horizon", 3, scope=MACKEY_GLASS),
    case("delay_parity.n_bits", 3, scope=DELAY_PARITY),
    case("delay_parity.delay", 2, scope=DELAY_PARITY),
    case("delay_parity.length", 260, scope=DELAY_PARITY),
    *_esn_cases("esn_mackey_glass", MACKEY_GLASS, leak_rate=0.9),
    *_esn_cases("esn_delay_parity", DELAY_PARITY, leak_rate=0.4),
)


def _replace_path(instance: object, path: str, value: object) -> object:
    """ドット区切りのパスで frozen dataclass を差し替えた複製を返す。"""
    head, _, rest = path.partition(".")
    new_value = _replace_path(getattr(instance, head), rest, value) if rest else value
    return dataclasses.replace(cast("DataclassInstance", instance), **{head: new_value})


def apply_case(config: ExperimentConfig, wiring_case: WiringCase) -> ExperimentConfig:
    """ケースの差し替えを適用した設定を返す。"""
    changed: object = config
    for path, value in wiring_case.overrides:
        changed = _replace_path(changed, path, value)
    return cast("ExperimentConfig", changed)


def fingerprint(rows: Sequence[ResultRow], task: str | None = None) -> str:
    """結果行の指紋 (実測時間の列だけ除く)。"""
    selected = [row for row in rows if task is None or row.task == task]
    return json.dumps(
        [
            {
                field.name: getattr(row, field.name)
                for field in fields(ResultRow)
                if field.name not in VOLATILE_COLUMNS
            }
            for row in selected
        ],
        sort_keys=True,
    )


def sizes(rows: Sequence[ResultRow]) -> dict[str, set[int]]:
    """分割サイズ (どのフィールドが効いたかの切り分けに使う)。"""
    return {
        name: {int(getattr(row, name)) for row in rows}
        for name in ("n_train", "n_val", "n_test")
    }


@lru_cache(maxsize=1)
def baseline_rows() -> tuple[ResultRow, ...]:
    """基準となる縮小パイプラインの出力 (ケースごとに再計算しない)。"""
    return tuple(run_experiment(base_config()))


def run_case(wiring_case: WiringCase) -> tuple[ResultRow, ...]:
    """ケースを適用して縮小パイプラインを回す。"""
    return tuple(run_experiment(apply_case(base_config(), wiring_case)))


@pytest.mark.parametrize(
    "wiring_case", WIRING_CASES, ids=[item.field for item in WIRING_CASES]
)
def test_each_parameter_changes_output(wiring_case: WiringCase) -> None:
    """各パラメータの値変更がパイプラインの出力を変える (配線の実測)。"""
    base = base_config()
    changed_config = apply_case(base, wiring_case)
    assert changed_config != base, "差し替えが設定に反映されていません"

    if wiring_case.channel == CHANNEL_ERROR:
        with pytest.raises(ValueError):
            run_case(wiring_case)
        return

    base_rows = baseline_rows()
    rows = run_case(wiring_case)

    if wiring_case.channel == CHANNEL_META:
        assert fingerprint(rows) == fingerprint(base_rows), (
            "メタ情報のはずが結果行を変えています"
        )
        assert _meta_fingerprint(changed_config) != _meta_fingerprint(base)
        return

    assert fingerprint(rows) != fingerprint(base_rows), (
        f"{wiring_case.field} を変えても出力が変わりません (配線漏れ)"
    )
    if wiring_case.scope is not None:
        assert fingerprint(rows, wiring_case.scope) != fingerprint(
            base_rows, wiring_case.scope
        )
        for other in (MACKEY_GLASS, DELAY_PARITY):
            if other != wiring_case.scope:
                assert fingerprint(rows, other) == fingerprint(base_rows, other), (
                    f"{wiring_case.field} が {other} の結果まで変えています"
                )
    for name in wiring_case.changed_sizes:
        assert sizes(rows)[name] != sizes(base_rows)[name], (
            f"{wiring_case.field} が {name} を変えていません"
        )


def test_split_ratio_isolation() -> None:
    """比の3フィールドが「どのサイズを動かすか」で切り分けられることの実測.

    ``val_ratio`` / ``test_ratio`` だけを動かしたとき ``n_train`` が変わらない
    ことを示すと、``train_ratio`` のケースで ``n_train`` が変わった原因を
    ``train_ratio`` に帰属できる (単体制約のため単独では動かせないため)。
    """
    base_rows = baseline_rows()
    val_case = next(item for item in WIRING_CASES if item.field == "split.val_ratio")
    rows = run_case(val_case)
    assert sizes(rows)["n_train"] == sizes(base_rows)["n_train"]
    assert sizes(rows)["n_val"] != sizes(base_rows)["n_val"]


def _leaf_paths(cls: type, prefix: str = "") -> set[str]:
    """dataclass の葉フィールドをドット区切りのパスで列挙する。"""
    hints = get_type_hints(cls)
    paths: set[str] = set()
    for field in fields(cast("type[DataclassInstance]", cls)):
        annotation = hints[field.name]
        if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
            paths |= _leaf_paths(annotation, f"{prefix}{field.name}.")
        else:
            paths.add(f"{prefix}{field.name}")
    return paths


def test_all_config_fields_are_covered() -> None:
    """``ExperimentConfig`` の全フィールドが上の parametrize に登場する.

    02〜05 でパラメータを足したとき、ここに1行足すまでテストが赤になる。
    「設定したのに効いていない」を構造的に防ぐのはこのテスト。
    """
    covered = {item.field for item in WIRING_CASES}
    expected = _leaf_paths(ExperimentConfig)
    assert covered == expected, (
        f"未登録: {sorted(expected - covered)} / 余分: {sorted(covered - expected)}"
    )
    # 差し替えパスの綴りも同じ集合の中にあること (typo で別フィールドを触らない)
    for item in WIRING_CASES:
        for path, _ in item.overrides:
            assert path in expected, f"未知のパスです: {path}"


def _plain(value: object) -> object:
    """YAML に安全に書ける値へ落とす (tuple -> list)。"""
    if isinstance(value, tuple | list):
        return [_plain(element) for element in value]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def test_every_field_round_trips_yaml(tmp_path: Path) -> None:
    """全フィールドが YAML のキーとして実在し、読み書きで往復する.

    「dataclass には在るが YAML からは設定できない」パラメータを作らないための
    検査 (D-09 の未知キー検査と対になる)。
    """
    config = base_config()
    path = tmp_path / "roundtrip.yaml"
    dumped = cast("Mapping[str, object]", _plain(dataclasses.asdict(config)))
    path.write_text(yaml.safe_dump(dumped, allow_unicode=True), encoding="utf-8")
    assert load_config(path) == config
    # 葉フィールドがすべて YAML に書き出されていること
    written = yaml.safe_load(path.read_text(encoding="utf-8"))
    for leaf in _leaf_paths(ExperimentConfig):
        node: object = written
        for part in leaf.split("."):
            assert isinstance(node, Mapping), leaf
            assert part in node, f"YAML に現れないフィールドです: {leaf}"
            node = node[part]


def _meta_fingerprint(config: ExperimentConfig) -> str:
    """``meta.json`` に載る設定ダンプの指紋。"""
    meta = collect_meta(config)
    return json.dumps(_plain(meta["config"]), sort_keys=True, default=str)


def test_experiment_config_yaml_matches_the_real_experiment() -> None:
    """本番の設定ファイルが同じローダを通ること (縮小設定だけが通る状態を防ぐ)。"""
    config = load_config(EXPERIMENT_CONFIG_PATH)
    assert config.n_replicates >= REQUIRED_REPLICATES
    assert len(config.ridge.alpha_grid) > 1
    assert len(config.ridge.n_lags_grid) > 1

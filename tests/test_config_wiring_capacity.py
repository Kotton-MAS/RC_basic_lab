"""03 の設定配線テスト (D-13) —— ``Capacity03Config`` の全葉フィールドの被覆.

01 (``tests/test_config_wiring.py``) / 02 (``tests/test_config_wiring_esp.py``)
と同じ防衛線を 03 にも張る。**設定クラスは分ける** (D-13): 03 用フィールドを
``ExperimentConfig`` に相乗りさせると 01 の
``test_each_parameter_changes_output`` を満たせないフィールドが必ず生まれ、
逃がすために例外チャネルを増やすと 01 の検出力そのものが落ちる。

**被覆の5系統**。``Capacity03Config`` の葉は効き方が分かれ、それぞれ別の場所で
「実際に効いている」ことが実測される。ここはその**割り当てが漏れていない**
ことを機械的に固定する。

- ``CHANNEL_ROWS``: 値を変えると縮小した掃引の結果行 (= ``capacity.csv``
  相当) の指紋が変わる。``scope`` を伴うものは、**その実験の行だけ**が変わる
  ことまで測る (セクション固有の葉が他の実験に漏れていないことの実測)。
- ``CHANNEL_META``: 結果行は変えないが ``meta.json`` を変える (``name``)。
- ``CHANNEL_ERROR``: 値域が1点しかなく、別の値は即座に例外になる
  (``drive.distribution``)。黙って既定として扱わないこと自体が配線である。
- ``CHANNEL_SEEDS``: 乱数ストリームの基底シード。そのストリームの乱数列だけが
  変わり、他ストリームは1バイトも動かないことまで測る。``seeds.surrogate``
  は ``SeedStream`` ではなく ``ctx.seed`` へ直接渡る整数なのでここには入れず、
  ``CHANNEL_ROWS`` で「しきい値が動く」ことを測る (D-37)。
- ``CHANNEL_PENDING``: 消費側がまだ存在しない葉。T1 の成果物は「1条件の配線」
  までで、``mc_sweep.*`` のような**どの条件を回すかを決める**葉は掃引が生える
  まで出力で実測できなかった。黙って見逃すと「設定したのに効いていない」が
  復活するので、消費側が生えた瞬間に落ちる信管
  (``test_pending_cases_disappear_once_the_sweeps_exist``) を張ってある。
  T2 で掃引4本が、**T4 で 3-C (``run_narma10``) が生えたので pending は
  1件も残っていない** (``PENDING_SECTIONS`` は空集合)。機構そのものは
  04 以降のために残す。

**委譲**: ``mc.*`` / ``ipc.*`` は 3a の診断設定 (D-15) をそのまま載せた部分、
``narma.base.*`` は 01 の ``ExperimentConfig`` をまるごと内包した部分なので、
効きの実測は各々の持ち主へ委譲する。**委譲先と過不足なく一致する**ことまで
assert するのが要点で、一致を確かめずに接頭辞で除外すると、委譲先に無い
フィールドをその下に足して被覆から逃がせてしまう。
"""

from __future__ import annotations

import dataclasses
import json
import pkgutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import yaml
from wiring import (
    CHANNEL_ERROR,
    CHANNEL_META,
    CHANNEL_ROWS,
    WiringCase,
    apply_case,
    assert_yaml_has_all_leaves,
    case,
    leaf_paths,
    plain,
)

import rc_basics_lab.experiment as experiment_pkg
from rc_basics_lab.config import (
    Capacity03Config,
    CapacityDriveConfig,
    CapacityReservoirConfig,
    ConfigError,
    ConservationConfig,
    ESNConfig,
    ExperimentConfig,
    IpcConfig,
    IpcSweepConfig,
    LengthSweepConfig,
    McSweepConfig,
    MemoryCapacityConfig,
    Narma10Config,
    RidgeConfig,
    SplitConfig,
    load_config_as,
)
from rc_basics_lab.experiment.capacity import (
    CAPACITY_EXPERIMENTS,
    EXPERIMENT_CONSERVATION,
    EXPERIMENT_IPC_SWEEP,
    EXPERIMENT_LENGTH_SWEEP,
    EXPERIMENT_MC_SWEEP,
    EXPERIMENT_NARMA10,
    CapacityOutcome,
    CapacityRow,
    run_conservation_sweep,
    run_ipc_sweep,
    run_length_sweep,
    run_mc_sweep,
)
from rc_basics_lab.experiment.narma import run_narma10
from rc_basics_lab.meta import collect_meta_for
from rc_basics_lab.seeds import SeedStream, make_rng_for

if TYPE_CHECKING:  # pragma: no cover - 型検査時のみ必要
    from _typeshed import DataclassInstance

CHANNEL_SEEDS = "seeds"
"""基底シードを変えると、そのストリームの乱数列だけが変わる。"""

CHANNEL_PENDING = "pending"
"""消費側がまだ無い葉のチャネル (**T4 時点で該当なし**、``PENDING_SECTIONS``)。"""

DELEGATED_SECTIONS: tuple[tuple[str, type], ...] = (
    ("mc.", MemoryCapacityConfig),
    ("ipc.", IpcConfig),
    ("narma.base.", ExperimentConfig),
)
"""接頭辞と、その配下が過不足なく一致すべき委譲先の設定クラス。

``mc`` / ``ipc`` は 3a の診断設定 (D-15)。効きは
``tests/test_diagnostics_memory_capacity.py`` /
``tests/test_diagnostics_ipc.py`` が診断単体のレベルで実測済みで、あちら側の
``test_all_config_fields_have_a_case`` が「設定クラスの全フィールドにケースが
ある」ことを強制している。こちらで「セクションの全葉がその設定クラスの全
フィールドである」ことを固定することで、委譲が両側から閉じる。

``narma.base`` は 01 の ``ExperimentConfig`` をまるごと内包した部分で、
``WashoutSweepConfig.base`` (02) と同じ形。被覆は 01 側へ委譲する。
"""

PENDING_SECTIONS: frozenset[str] = frozenset()
"""``CHANNEL_PENDING`` を名乗ってよいセクション (**現在は空**)。

T2 で掃引 (``run_mc_sweep`` / ``run_ipc_sweep`` / ``run_conservation_sweep``
/ ``run_length_sweep``) が、T4 で 3-C (``run_narma10``) が生えたので、
``Capacity03Config`` の全葉が出力で実測できるようになった
(``narma.length`` は ``3C_narma10`` の行の ``n_steps`` を動かす)。

**空集合のまま残す**のは、04 以降で「消費側がまだ無い葉」を足すときに
この機構ごと書き直さずに済ませるためである。空である事実そのものは
``tests/test_capacity_pipeline.py::test_production_config_matches_the_committed_meta_json``
の比較対象 (このセクションを引いたもの) にも効いている —— pending が空に
なった時点で、``narma`` セクションも本番 ``meta.json`` との突合対象に戻る。
"""

TASK_STAGE_CONSUMERS: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "T2": (
        ("run_mc_sweep", "run_ipc_sweep", "run_conservation_sweep"),
        ("capacity_pipeline",),
    ),
    "T4": (("run_narma10",), ("narma",)),
}
"""段階ごとに「その段階の消費側が実装された」と判定する (関数名, モジュール名)。

02 の信管 (``KNOWN_EXPERIMENT_MODULES``) はモジュールの新設だけを見ていたが、03 の
掃引は T1 で既に存在する ``experiment/capacity.py`` の**中に**生える計画なので、
モジュール名だけでは発火しない。関数名でも判定する。逆に、実装者が掃引を
``experiment/capacity_pipeline.py`` 側に置いた場合はモジュール名の側で発火する
(どちらの実装順でも信管が沈黙しない、というのが F-02-1-005 の教訓)。
"""

EXPERIMENT_LABELS: tuple[str, ...] = CAPACITY_EXPERIMENTS
"""``scope`` に書ける実験名 (``CapacityRow.experiment`` の値)。

``3L_length_sweep`` (``make saturation-03``) も含む。本番の成果物には出ない
実験だが、``length_sweep.*`` を変えても 3-A / 3-B / 3-B' の行が1バイトも
動かないことは同じ scope 検査で測る必要がある。
"""

CAPACITY_SEED_STREAMS: tuple[SeedStream, ...] = (
    SeedStream.RESERVOIR,
    SeedStream.TASK,
)
"""03 が ``make_rng_for`` 経由で使うストリーム。

``PROBE`` は ESP 判定専用 (03 は比較軌道を引かない)、``SPLIT`` は 3-C が 01 の
経路で使う。``seeds.surrogate`` は ``SeedStream`` ではない (``ctx.seed``)。
"""

_N_BYTES = 32


def base_config() -> Capacity03Config:
    """秒未満で4実験 x 2条件を回せる縮小設定 (**構造は本番と同じ**)。

    ``drive.washout`` を最大遅延より大きく取ってあるので、
    ``t0 = max(washout, 最大遅延)`` (D-24) の binding side は washout である。

    格子は各セクション2条件になるように削ってある。1条件だけにすると
    「格子の**点数**を変えたら行が増える」ことしか測れず、点の**値**が届いて
    いるかを分離できない (どちらも指紋を変えるので、値が無視される実装でも
    通ってしまう)。セクションごとに ``n_units`` / ``n_steps`` / ``sigma_u`` を
    別の値にしてあるのは、セクション間で値を取り違える配線を落とすためである。
    """
    return Capacity03Config(
        name="capacity-wiring",
        drive=CapacityDriveConfig(distribution="uniform", washout=40),
        reservoir=CapacityReservoirConfig(input_scale=1.0, density=0.3, n_replicates=1),
        mc_sweep=McSweepConfig(
            rho_grid=(0.5, 0.9),
            leak_rate_grid=(1.0,),
            sigma_u=0.3,
            n_units=12,
            n_steps=1200,
        ),
        ipc_sweep=IpcSweepConfig(
            rho_grid=(0.8,),
            leak_rate_grid=(0.6, 1.0),
            sigma_u=0.35,
            n_units=10,
            n_steps=1100,
        ),
        conservation=ConservationConfig(
            n_units_grid=(9,),
            state_noise_grid=(0.0, 0.02),
            rho=0.95,
            leak_rate=1.0,
            sigma_u=0.4,
            n_steps=1100,
            max_delay_by_degree=(8, 4),
        ),
        length_sweep=LengthSweepConfig(
            n_steps_grid=(1000, 1300),
            rho=0.85,
            leak_rate=0.8,
            sigma_u=0.45,
            n_units=8,
        ),
        mc=MemoryCapacityConfig(max_delay=20, n_surrogates=5),
        ipc=IpcConfig(
            max_delay_by_degree=(8, 4), n_surrogates=5, n_surrogate_targets=2
        ),
        # 3-C も他セクションと違う n_units / n_steps にしてある (セクション間で
        # 値を取り違える配線を落とすため)。base は 01 の ExperimentConfig を
        # 内包した部分で、被覆は 01 側へ委譲する (DELEGATED_SECTIONS)。
        narma=Narma10Config(
            length=900,
            base=ExperimentConfig(
                name="capacity-wiring-narma",
                n_replicates=1,
                split=SplitConfig(washout=50, max_start_offset=20),
                ridge=RidgeConfig(alpha_grid=(1e-2,), n_lags_grid=(2,)),
                esn_mackey_glass=ESNConfig(
                    n_units=7, leak_rate=1.0, input_scale=1.0, density=0.5
                ),
            ),
        ),
    )


def run_narma10_capacity(config: Capacity03Config) -> tuple[CapacityOutcome, ...]:
    """3-C の容量を掃引と同じ形 (``CapacityOutcome`` の並び) で返す。

    3-C は条件を1つしか持たない (掃引ではない) が、``capacity.csv`` に行が
    出る以上ここで一緒に指紋を取る。そうしないと ``narma.length`` の効きも、
    「他のセクションを変えたら 3-C の行まで動いた」も測れない。
    """
    return (run_narma10(config).capacity,)


SWEEPS: tuple[Callable[[Capacity03Config], tuple[CapacityOutcome, ...]], ...] = (
    run_mc_sweep,
    run_ipc_sweep,
    run_conservation_sweep,
    run_length_sweep,
    run_narma10_capacity,
)
"""指紋を取る実験。**5実験すべて**を回す (``scope`` 検査の前提)。

T1 の時点では掃引が無かったため固定の条件を並べていたが、T2 で条件は
``config`` から作られるようになった。セクション固有の葉 (``mc_sweep.*`` /
``ipc_sweep.*`` / ``conservation.*`` / ``length_sweep.*``) は、これで初めて
「値を変えたら**その実験の行だけ**が変わる」を実測できる (仕様 §5 有効性観点
「セクション固有の葉は scope 検査つき (4セクション)」)。

``3L_length_sweep`` は本番の成果物 (``capacity.csv``) に出ない実験だが、
ここでは回す —— ``length_sweep.*`` を変えたときに 3-A の行まで動く配線を
落とすには、両方の行が同じ指紋の中に無ければならない。
"""


def _pending_case(field: str, value: object, task: str, note: str) -> WiringCase:
    return case(field, value, channel=CHANNEL_PENDING, task=task, note=note)


CAPACITY_WIRING_CASES: tuple[WiringCase, ...] = (
    # name は結果行に出ない純粋なメタ情報。meta.json に載ることを確かめる。
    case("name", "03-renamed", channel=CHANNEL_META),
    seeds_case("seeds.reservoir", 100, SeedStream.RESERVOIR),
    seeds_case("seeds.drive", 101, SeedStream.TASK),
    # サロゲートのシードは SeedStream ではなく ctx.seed へ直接渡る (D-37)。
    # 効きは「しきい値が動く = 行が変わる」で測る。
    case("seeds.surrogate", 777),
    # --- 駆動入力の共通条件 ---
    # 一様分布以外は未対応。黙って一様として扱わないこと自体が配線である。
    case("drive.distribution", "gaussian", channel=CHANNEL_ERROR),
    case("drive.washout", 60),
    # --- セクション横断で共有するリザバー構造 (n_units は持たない、D-32) ---
    # 横断共有なので scope は付かない (全実験の行が動くのが正しい)。
    case("reservoir.input_scale", 2.0),
    case("reservoir.density", 0.6),
    case("reservoir.n_replicates", 2),
    # --- 3-A: 線形メモリ容量の掃引軸 ---
    section_case("mc_sweep.rho_grid", (0.6, 1.0), EXPERIMENT_MC_SWEEP),
    section_case("mc_sweep.leak_rate_grid", (0.4,), EXPERIMENT_MC_SWEEP),
    section_case("mc_sweep.sigma_u", 0.5, EXPERIMENT_MC_SWEEP),
    section_case("mc_sweep.n_units", 14, EXPERIMENT_MC_SWEEP),
    section_case("mc_sweep.n_steps", 1400, EXPERIMENT_MC_SWEEP),
    # --- 3-B: IPC の掃引軸 ---
    section_case("ipc_sweep.rho_grid", (0.7,), EXPERIMENT_IPC_SWEEP),
    section_case("ipc_sweep.leak_rate_grid", (0.5, 1.0), EXPERIMENT_IPC_SWEEP),
    section_case("ipc_sweep.sigma_u", 0.5, EXPERIMENT_IPC_SWEEP),
    section_case("ipc_sweep.n_units", 16, EXPERIMENT_IPC_SWEEP),
    section_case("ipc_sweep.n_steps", 1300, EXPERIMENT_IPC_SWEEP),
    # --- 3-B': 保存則。打ち切りとレプリケート数はこの実験だけを上書きする ---
    section_case("conservation.n_units_grid", (8, 12), EXPERIMENT_CONSERVATION),
    section_case(
        "conservation.state_noise_grid", (0.0, 0.05), EXPERIMENT_CONSERVATION
    ),
    section_case("conservation.rho", 0.85, EXPERIMENT_CONSERVATION),
    section_case("conservation.leak_rate", 0.7, EXPERIMENT_CONSERVATION),
    section_case("conservation.sigma_u", 0.55, EXPERIMENT_CONSERVATION),
    section_case("conservation.n_steps", 1500, EXPERIMENT_CONSERVATION),
    section_case("conservation.max_delay_by_degree", (12, 6), EXPERIMENT_CONSERVATION),
    # None なら reservoir.n_replicates を継承する片方向の上書き (仕様 §7 の
    # 縮退規則のノブ)。3-B' の行だけが増える。
    section_case("conservation.n_replicates", 2, EXPERIMENT_CONSERVATION),
    # --- 系列長掃引 (make saturation-03。本番の figures-03 には含めない) ---
    section_case("length_sweep.n_steps_grid", (1100, 1600), EXPERIMENT_LENGTH_SWEEP),
    section_case("length_sweep.rho", 0.6, EXPERIMENT_LENGTH_SWEEP),
    section_case("length_sweep.leak_rate", 0.5, EXPERIMENT_LENGTH_SWEEP),
    section_case("length_sweep.sigma_u", 0.6, EXPERIMENT_LENGTH_SWEEP),
    section_case("length_sweep.n_units", 11, EXPERIMENT_LENGTH_SWEEP),
    # --- 3-C: NARMA10 (系列長は 3-C の行の n_steps を動かす) ---
    section_case("narma.length", 400, EXPERIMENT_NARMA10),
)


VOLATILE_COLUMNS = frozenset(
    {"wall_time_s", "wall_time_state_s", "wall_time_mc_s", "wall_time_ipc_s"}
)
"""指紋から外す列 (実測時間は実行ごとに変わる)。"""


def fingerprint(rows: Sequence[CapacityRow], experiment: str | None = None) -> str:
    """結果行の指紋 (実測時間の列だけ除く)。

    ``experiment`` を渡すとその実験の行だけを見る。セクション固有の葉が
    他の実験の行を動かしていないことの確認に使う。
    """
    selected = [
        row for row in rows if experiment is None or row.experiment == experiment
    ]
    return json.dumps(
        [
            {
                field.name: getattr(row, field.name)
                for field in fields(CapacityRow)
                if field.name not in VOLATILE_COLUMNS
            }
            for row in selected
        ],
        sort_keys=True,
    )


def run_config(config: Capacity03Config) -> tuple[CapacityRow, ...]:
    """縮小設定で4実験の掃引を回して結果行を得る。"""
    return tuple(outcome.row for sweep in SWEEPS for outcome in sweep(config))


@lru_cache(maxsize=1)
def baseline_rows() -> tuple[CapacityRow, ...]:
    """基準となる縮小実験の出力 (ケースごとに再計算しない)。"""
    return run_config(base_config())


def run_case(wiring_case: WiringCase) -> tuple[CapacityRow, ...]:
    """ケースを適用して縮小実験を回す。"""
    return run_config(apply_case(base_config(), wiring_case))


def _seed_fingerprints(config: Capacity03Config) -> dict[SeedStream, bytes]:
    """03 が使う各ストリームの乱数列の先頭バイト列。"""
    seeds = {
        SeedStream.RESERVOIR: config.seeds.reservoir,
        SeedStream.TASK: config.seeds.drive,
    }
    return {
        stream: make_rng_for(seeds[stream], stream, 0).bytes(_N_BYTES)
        for stream in CAPACITY_SEED_STREAMS
    }


def _changed_leaves(base: Capacity03Config, changed: Capacity03Config) -> set[str]:
    """2つの設定で値が異なる葉フィールドのパス集合。"""
    return {
        leaf
        for leaf in leaf_paths(Capacity03Config)
        if _leaf_value(base, leaf) != _leaf_value(changed, leaf)
    }


def _leaf_value(config: object, leaf: str) -> object:
    node: object = config
    for part in leaf.split("."):
        node = getattr(node, part)
    return node


def _round_trip(
    config: Capacity03Config, tmp_path: Path, name: str
) -> Capacity03Config:
    """設定を YAML へ書き出して読み直す (``load_config_as`` の経路そのもの)。"""
    path = tmp_path / f"{name}.yaml"
    dumped = cast("Mapping[str, object]", plain(dataclasses.asdict(config)))
    path.write_text(yaml.safe_dump(dumped, allow_unicode=True), encoding="utf-8")
    return load_config_as(path, Capacity03Config)


def _meta_fingerprint(config: Capacity03Config) -> str:
    """``meta.json`` に載る設定ダンプの指紋。"""
    meta = collect_meta_for(config, config.seeds)
    return json.dumps(plain(meta["config"]), sort_keys=True, default=str)


@pytest.mark.parametrize(
    "wiring_case",
    CAPACITY_WIRING_CASES,
    ids=[item.field for item in CAPACITY_WIRING_CASES],
)
def test_each_capacity_parameter_changes_output(
    wiring_case: WiringCase, tmp_path: Path
) -> None:
    """各パラメータが「効く経路」を実際に持っていることの実測。

    何を「出力」と見なすかはチャネルごとに違う (モジュール docstring 参照)。
    どのチャネルでも共通して、**値が YAML を往復してその葉にだけ届く**ことは
    実測する。「YAML に書いたのに設定オブジェクトへ届いていない」を殺す。
    """
    base = base_config()
    changed_config = apply_case(base, wiring_case)
    assert changed_config != base, "差し替えが設定に反映されていません"

    assert _round_trip(changed_config, tmp_path, "changed") == changed_config
    assert _changed_leaves(base, changed_config) == {wiring_case.field}, (
        f"{wiring_case.field} の差し替えが他の葉にも波及しています"
    )

    if wiring_case.channel == CHANNEL_SEEDS:
        before = _seed_fingerprints(base)
        after = _seed_fingerprints(changed_config)
        moved = {
            stream
            for stream in CAPACITY_SEED_STREAMS
            if before[stream] != after[stream]
        }
        names = sorted(stream.value for stream in moved)
        assert set(names) == {wiring_case.scope}, (
            f"{wiring_case.field} が動かしたストリーム: {names}"
        )
        # シードは結果行も動かす (ストリーム独立性だけで満足しない)
        assert fingerprint(run_case(wiring_case)) != fingerprint(baseline_rows())
        return

    if wiring_case.channel == CHANNEL_PENDING:
        assert wiring_case.field.split(".")[0] in PENDING_SECTIONS, (
            f"{wiring_case.field} は消費側が生えているので効きを実測できるはずです"
        )
        assert wiring_case.note, "pending の理由が書かれていません"
        assert wiring_case.task in TASK_STAGE_CONSUMERS, (
            f"{wiring_case.field} の task が段階表にありません: {wiring_case.task}"
        )
        return

    if wiring_case.channel == CHANNEL_ERROR:
        with pytest.raises(ValueError):
            run_case(wiring_case)
        return

    base_rows = baseline_rows()
    if wiring_case.channel == CHANNEL_META:
        assert fingerprint(run_case(wiring_case)) == fingerprint(base_rows), (
            "メタ情報のはずが結果行を変えています"
        )
        assert _meta_fingerprint(changed_config) != _meta_fingerprint(base)
        return

    assert wiring_case.channel == CHANNEL_ROWS, wiring_case.channel
    rows = run_case(wiring_case)
    assert fingerprint(rows) != fingerprint(base_rows), (
        f"{wiring_case.field} を変えても出力が変わりません (配線漏れ)"
    )
    if wiring_case.scope is not None:
        assert fingerprint(rows, wiring_case.scope) != fingerprint(
            base_rows, wiring_case.scope
        )
        for other in EXPERIMENT_LABELS:
            if other != wiring_case.scope:
                assert fingerprint(rows, other) == fingerprint(base_rows, other), (
                    f"{wiring_case.field} が {other} の結果まで変えています"
                )


def test_all_capacity_config_fields_are_covered() -> None:
    """``Capacity03Config`` の全葉が被覆されている (D-13 guard)。

    03 でパラメータを足したとき、ここに1行足すまでテストが赤になる。
    ``mc.*`` / ``ipc.*`` (3a の診断設定) と ``narma.base.*`` (01 の設定) は
    委譲するが、**委譲先と過不足なく一致する**ことまで assert する。一致を
    確かめずに接頭辞で除外すると、委譲先に無いフィールドをその下に足して
    被覆から逃がせてしまう。
    """
    all_leaves = leaf_paths(Capacity03Config)
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

    covered = {item.field for item in CAPACITY_WIRING_CASES}
    expected = all_leaves - delegated
    assert covered == expected, (
        f"未登録: {sorted(expected - covered)} / 余分: {sorted(covered - expected)}"
    )
    # 差し替えパスの綴りも同じ集合の中にあること (typo で別フィールドを触らない)
    for item in CAPACITY_WIRING_CASES:
        for path, _ in item.overrides:
            assert path in expected, f"未知のパスです: {path}"


def _current_experiment_modules() -> frozenset[str]:
    """``rc_basics_lab.experiment`` 配下の公開モジュール名の実集合。"""
    return frozenset(
        info.name
        for info in pkgutil.iter_modules(experiment_pkg.__path__)
        if not info.name.startswith("_")
    )


def _implemented_consumers(task: str) -> tuple[str, ...]:
    """その段階の消費側で、既に実装されている関数名・モジュール名。"""
    import rc_basics_lab.experiment.capacity as capacity_module

    function_names, module_names = TASK_STAGE_CONSUMERS[task]
    modules = _current_experiment_modules()
    found = [
        name
        for name in function_names
        if hasattr(capacity_module, name) or hasattr(experiment_pkg, name)
    ]
    found.extend(name for name in module_names if name in modules)
    return tuple(found)


def test_pending_cases_disappear_once_the_sweeps_exist() -> None:
    """掃引が生えたら、**その段階の** ``CHANNEL_PENDING`` は許されない。

    T1 の時点では条件の列挙 (掃引) が無いため、格子やレプリケート数のような
    「掃引を回して初めて効く」葉は出力での実測ができない。そこを黙って見逃すと
    「設定したのに効いていない」が 03 で復活するので、消費側が生えた瞬間に
    このテストが赤くなるようにしてある。T2 では各 pending ケースを実際の
    出力チャネル (``CHANNEL_ROWS`` + ``scope``) へ書き換えること。

    段階を分けるのは F-02-2-004 と同じ理由である。「消費側が1つでも増えたら
    段階を問わず全 pending を禁じる」形にすると、T2 が掃引を作った瞬間に
    3b-2 (T4) 担当の ``narma.length`` まで巻き添えで赤くなり、T2 完了〜T4
    着手の間テストが緑にならない。
    """
    for task in TASK_STAGE_CONSUMERS:
        implemented = _implemented_consumers(task)
        if not implemented:
            continue
        stage_pending = sorted(
            item.field
            for item in CAPACITY_WIRING_CASES
            if item.channel == CHANNEL_PENDING and item.task == task
        )
        assert not stage_pending, (
            f"{task} の消費側 {list(implemented)} が実装されているのに、"
            f"{task} の未実測の葉が残っています: {stage_pending}"
        )

    pending = [
        item.field for item in CAPACITY_WIRING_CASES if item.channel == CHANNEL_PENDING
    ]
    assert {field.split(".")[0] for field in pending} <= PENDING_SECTIONS


def test_pending_cases_declare_a_known_task_stage() -> None:
    """pending は必ず既知の段階を名乗る (段階不明の先送りを作らない)。

    ``task`` が段階表に無い pending は、どの信管でも解除されない永久の
    先送りになる。``note`` (自由文) だけで先送りの理由を書く形は
    F-02-2-004 で潰した経路なので、判定に使えるのは ``task`` だけである。
    """
    for item in CAPACITY_WIRING_CASES:
        if item.channel != CHANNEL_PENDING:
            continue
        assert item.task in TASK_STAGE_CONSUMERS, (
            f"{item.field} の task が段階表にありません: {item.task!r}"
        )
        assert item.note, f"{item.field} に pending の理由が書かれていません"


def test_ipc_reservoir_is_smaller_than_mc_reservoir() -> None:
    """IPC の掃引は MC より小さいリザバーで回す (D-32)。

    IPC は目標数が (次数 x 遅延) で増え、必要な系列長も N に対して伸びるため、
    MC と同じ N=200 では予算に収まらない。``n_units`` を**セクションが持つ**
    のはこの非対称のためで、``reservoir`` (横断共有) に持たせると片方に
    不利な値を押し付けることになる。
    """
    config = Capacity03Config()
    assert config.ipc_sweep.n_units < config.mc_sweep.n_units
    assert config.mc_sweep.n_units == 200
    assert config.ipc_sweep.n_units == 50
    # 横断共有のリザバー設定は n_units を持たない (持てば食い違いが生じても
    # 何も落ちない)
    assert "n_units" not in {item.name for item in fields(CapacityReservoirConfig)}
    # 3-B' の N は掃引軸そのもの (上限線 y=N と突き合わせるため)
    assert len(config.conservation.n_units_grid) >= 2


def test_conservation_target_count_stays_within_the_ipc_bound() -> None:
    """3-B' の打ち切りが ``ipc.max_targets`` / ``max_degrees`` の内側にある。

    ``conservation.max_delay_by_degree`` は ``config.ipc`` を上書きするので、
    上書き後の設定が 3a の上限群 (D-34) に収まっていないと本番で
    ``ValueError`` になる。仕様が挙げた実測値 (目標 4,075 本 / heatmap 800
    セル) をここで固定しておくと、打ち切りを深くする変更が予算を割る前に
    落ちる。
    """
    from rc_basics_lab.diagnostics.ipc import count_targets

    config = Capacity03Config()
    overridden = dataclasses.replace(
        config.ipc, max_delay_by_degree=config.conservation.max_delay_by_degree
    )
    n_targets = count_targets(overridden)
    n_cells = len(overridden.max_delay_by_degree) * max(overridden.max_delay_by_degree)
    assert n_targets == 4075
    assert n_cells == 800
    assert n_targets < overridden.max_targets
    assert n_cells < overridden.max_targets
    assert len(overridden.max_delay_by_degree) <= overridden.max_degrees


def test_every_capacity_field_round_trips_yaml(tmp_path: Path) -> None:
    """``Capacity03Config`` の全フィールドが YAML のキーとして実在し往復する。

    「dataclass には在るが YAML からは設定できない」パラメータを作らないための
    検査 (D-09 の未知キー検査と対になる)。``narma.base.*`` も含めて確かめる
    ので、01 の設定を内包した部分が YAML から届かない事故も落ちる。
    """
    config = Capacity03Config()
    path = tmp_path / "roundtrip.yaml"
    dumped = cast("Mapping[str, object]", plain(dataclasses.asdict(config)))
    path.write_text(yaml.safe_dump(dumped, allow_unicode=True), encoding="utf-8")
    assert load_config_as(path, Capacity03Config) == config
    assert_yaml_has_all_leaves(
        yaml.safe_load(path.read_text(encoding="utf-8")), Capacity03Config
    )


def test_empty_yaml_gives_capacity_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config_as(path, Capacity03Config) == Capacity03Config()


@pytest.mark.parametrize(
    ("yaml_text", "match"),
    [
        pytest.param("n_replicates: 3\n", "n_replicates", id="top_level"),
        pytest.param("mc:\n  max_delays: 100\n", "max_delays", id="diagnostic"),
        pytest.param("seeds:\n  probe: 3\n", "probe", id="seed_stream_name"),
        pytest.param(
            "narma:\n  base:\n    n_replicate: 3\n", "n_replicate", id="nested_01"
        ),
        pytest.param(
            "reservoir:\n  n_units: 40\n", "n_units", id="n_units_on_reservoir"
        ),
        pytest.param(
            "mc_sweep:\n  max_delay_by_degree: [10]\n",
            "max_delay_by_degree",
            id="ipc_field_on_mc_sweep",
        ),
    ],
)
def test_unknown_key_raises_for_capacity_config(
    tmp_path: Path, yaml_text: str, match: str
) -> None:
    """03 の YAML も未知キーで即座に落ちる (D-09)。

    ``reservoir.n_units`` を弾くのは D-32 の中心保証である。``n_units`` を
    横断共有の ``reservoir`` に書けてしまうと、MC (N=200) と IPC (N=50) の
    非対称が黙って壊れる。``seeds.probe`` は 02 のキー名で、似た名前のキーが
    無視されると「シードを変えたつもりで既定値のまま回る」が起きる。
    """
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_config_as(path, Capacity03Config)


@pytest.mark.parametrize(
    ("yaml_text", "expected"),
    [
        pytest.param("", None, id="absent"),
        pytest.param("conservation:\n  n_replicates: null\n", None, id="explicit_null"),
        pytest.param("conservation:\n  n_replicates: 1\n", 1, id="overridden"),
    ],
)
def test_optional_replicates_reaches_the_config_from_yaml(
    tmp_path: Path, yaml_text: str, expected: int | None
) -> None:
    """``conservation.n_replicates`` (``int | None``) が YAML から届く。

    ローダは元々 ``X | None`` を「未対応の Union 型」として弾いていた。
    ``None`` を許すのは「セクション側が名乗らなければ横断共有の値を継承する」
    片方向の上書き (``n_replicates_for``) を型で表すためで、既定値そのものを
    こちら側に書かない (書くと二重定義になり、継承元を変えても効かなくなる)。
    """
    path = tmp_path / "optional.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    assert load_config_as(path, Capacity03Config).conservation.n_replicates == expected


def test_optional_replicates_still_rejects_loose_conversions(tmp_path: Path) -> None:
    """``int | None`` でも中身の型検査は緩まない (D-09)。

    ``None`` を素通しする分岐を足したときに「``None`` 以外は何でも通る」に
    退行すると、``"3"`` のような数値らしい文字列が黙って通る。
    """
    path = tmp_path / "loose.yaml"
    path.write_text('conservation:\n  n_replicates: "3"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="整数が必要です"):
        load_config_as(path, Capacity03Config)


def test_capacity_config_does_not_leak_into_experiment_config() -> None:
    """01 の ``ExperimentConfig`` に 03 のフィールドが1つも増えていない (D-13)。

    増えた瞬間に 01 の ``test_each_parameter_changes_output`` が「01 の
    パイプライン出力を変えないフィールド」を抱えることになる。
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

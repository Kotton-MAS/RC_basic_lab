"""しきい値法の比較 (受け入れ条件3) —— 代表条件・モード網羅・本番との突合.

受け入れ条件3 は「しきい値処理の有無で総容量がどれだけ変わるかを記録し、
既定を根拠つきで選ぶ」である。ここで固定するのは4つ:

1. **代表条件が本番の掃引格子の点である** —— 比較のためだけの別条件を作ると、
   表の数字が ``capacity.csv`` のどの行とも突き合わせられなくなる
2. **診断が受理する全モードを回している** —— 診断側にモードが増えたら比較表にも
   出る (``SUPPORTED_THRESHOLD_MODES`` と過不足なく一致)
3. **軌道は1回しか作らない** (仕様 §5 の禁止構造「条件ごとに X を2回作る」)
4. **既定モードの値がコミット済みの本番成果物と一致する** —— ``meta.json`` の
   ``threshold_comparison`` と ``capacity.csv`` の同一条件の行の突合。
   ``docs/design.md`` §11.2 の表はこの ``meta.json`` と機械照合される
   (``tests/test_design_doc.py``) ので、この2件で「散文 -> meta.json ->
   capacity.csv」が1本につながる
"""

from __future__ import annotations

import csv
import dataclasses
import importlib
import json
from pathlib import Path

import pytest
from test_capacity_pipeline import tiny_config

from rc_basics_lab.config import Capacity03Config, load_config_as
from rc_basics_lab.experiment.capacity import EXPERIMENT_IPC_SWEEP
from rc_basics_lab.experiment.capacity_threshold import (
    IPC_THRESHOLD_MODES,
    MC_THRESHOLD_MODES,
    ThresholdComparison,
    comparison_condition,
    run_threshold_comparison,
)
from rc_basics_lab.experiment.report import META_JSON

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "03_capacity"
PRODUCTION_CONFIG = ROOT / "experiments" / "03_capacity" / "config.yaml"

THRESHOLD_MODULE = "rc_basics_lab.experiment.capacity_threshold"
"""差し替え対象のモジュール名 (文字列で持つ理由は §10-1 の罠と同じ)。"""

THRESHOLD_NONE = "none"
"""しきい値を課さないモードの名前 (表の「上限」の行)。"""


def production_config() -> Capacity03Config:
    return load_config_as(PRODUCTION_CONFIG, Capacity03Config)


def _capacity_rows() -> list[dict[str, str]]:
    with (RESULTS / "capacity.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _meta() -> dict[str, object]:
    with (RESULTS / META_JSON).open(encoding="utf-8") as handle:
        loaded: dict[str, object] = json.load(handle)
    return loaded


def _comparison() -> dict[str, object]:
    meta = _meta()
    assert "threshold_comparison" in meta, (
        "meta.json に threshold_comparison がありません (make figures-03 で再生成)"
    )
    comparison: dict[str, object] = meta["threshold_comparison"]  # type: ignore[assignment]
    return comparison


def _rows_of(comparison: dict[str, object], key: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = comparison[key]  # type: ignore[assignment]
    return rows


def _row_with(rows: list[dict[str, object]], mode: str) -> dict[str, object]:
    return next(row for row in rows if row["threshold_mode"] == mode)


# --- 1. 代表条件は本番の掃引格子の点 -----------------------------------------


@pytest.mark.parametrize(
    "config", [production_config(), tiny_config()], ids=["production", "tiny"]
)
def test_comparison_condition_is_a_point_of_the_ipc_sweep_grid(
    config: Capacity03Config,
) -> None:
    """代表条件が 3-B の格子に**実在する**点である (別条件を作っていない)。

    ここが崩れると ``capacity.csv`` に対応する行が無くなり、
    ``test_default_mode_matches_the_committed_capacity_row`` の突合が
    「探したが見つからない」で落ちる —— その前に、崩れた瞬間にここで落とす。
    """
    condition = comparison_condition(config)
    section = config.ipc_sweep
    assert condition.experiment == EXPERIMENT_IPC_SWEEP
    assert condition.rho in section.rho_grid
    assert condition.leak_rate in section.leak_rate_grid
    assert condition.n_units == section.n_units
    assert condition.n_steps == section.n_steps
    assert condition.sigma_u == section.sigma_u
    # 掃引は 3-B でノイズを入れない (state_noise を振るのは 3-B' だけ) ので、
    # 0.0 以外だと格子の点ではなくなる。
    assert condition.state_noise == 0.0
    assert condition.replicate == 0


def test_comparison_condition_follows_the_grid_when_the_grid_moves() -> None:
    """格子を動かすと代表条件も動く (定数を書き写した実装では落ちる)。

    本番格子に合わせた ``rho=0.95`` / ``leak_rate=0.6`` をハードコードしても
    上のテストは通ってしまう (どちらも格子の中に在るため)。中央を選ぶ規則が
    実際に効いていることは、格子を差し替えて確かめるしかない。
    """
    config = production_config()
    moved = dataclasses.replace(
        config,
        ipc_sweep=dataclasses.replace(
            config.ipc_sweep, rho_grid=(0.1, 0.2, 0.3), leak_rate_grid=(0.9,)
        ),
    )
    condition = comparison_condition(moved)
    assert (condition.rho, condition.leak_rate) == (0.2, 0.9)


# --- 2. 診断が受理する全モードを回す -----------------------------------------


def test_threshold_modes_cover_every_mode_the_diagnostics_accept() -> None:
    """比較するモードが ``SUPPORTED_THRESHOLD_MODES`` と過不足なく一致する。

    診断側にモードが増えたのに比較表が古いままだと、「既定を根拠つきで選ぶ」
    (受け入れ条件3) の根拠が選択肢の一部しか見ていないものになる。

    モジュールは ``importlib.import_module`` で引く —— ``from
    rc_basics_lab.diagnostics import ipc`` は ``diagnostics/__init__.py`` が
    再エクスポートしている**関数** ``ipc`` を返す (仕様 §10-1 の罠)。
    """
    mc_module = importlib.import_module("rc_basics_lab.diagnostics.memory_capacity")
    ipc_module = importlib.import_module("rc_basics_lab.diagnostics.ipc")
    assert set(MC_THRESHOLD_MODES) == set(mc_module.SUPPORTED_THRESHOLD_MODES)
    assert set(IPC_THRESHOLD_MODES) == set(ipc_module.SUPPORTED_THRESHOLD_MODES)
    # 表の1行目は「しきい値なし」= 上限であること (design.md §11.2 の読み方)。
    assert MC_THRESHOLD_MODES[0] == THRESHOLD_NONE
    assert IPC_THRESHOLD_MODES[0] == THRESHOLD_NONE
    # MC は chi2 を持たない (次数1しか評価しないため)。比較を診断ごとに分けて
    # いる理由そのものなので、揃った瞬間に気づけるようにしておく。
    assert set(IPC_THRESHOLD_MODES) - set(MC_THRESHOLD_MODES) == {"chi2"}


# --- 3. 軌道は1回だけ作る (仕様 §5 の禁止構造) --------------------------------


def test_states_are_generated_once_for_all_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """しきい値法を3通り試しても ``simulate_condition_trajectory`` は1回。

    差し替えるのは**呼び出し側のモジュール属性** (``capacity_threshold`` の
    名前空間) で、定義元を差し替えると 03 の他の経路まで巻き込む (§10-1)。
    F-3b2-1-001 (M1) で ``simulate_reference_trajectory`` への9引数呼び出しは
    ``capacity.py`` の ``simulate_condition_trajectory`` に一本化された
    (``_validate_condition_bounds`` を呼ぶのもここ)。
    """
    module = importlib.import_module(THRESHOLD_MODULE)
    calls: list[object] = []
    real = module.simulate_condition_trajectory

    def counting(*args: object, **kwargs: object) -> object:
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(f"{THRESHOLD_MODULE}.simulate_condition_trajectory", counting)
    comparison = run_threshold_comparison(tiny_config())
    assert len(calls) == 1
    assert len(comparison.memory_capacity) == len(MC_THRESHOLD_MODES)
    assert len(comparison.ipc) == len(IPC_THRESHOLD_MODES)


# --- 4. しきい値は容量を削る方向にしか働かない -------------------------------


@pytest.fixture(scope="module")
def tiny_comparison() -> ThresholdComparison:
    """縮小設定の比較結果 (モジュール内で使い回す。1回で約1秒)。"""
    return run_threshold_comparison(tiny_config())


def test_threshold_only_removes_capacity(tiny_comparison: ThresholdComparison) -> None:
    """どのモードでも ``total <= total_raw`` で、``none`` は等号になる。

    ``total_raw`` (しきい値**前**) はモードに依存しない量なので、モード間で
    完全に一致することも見る —— 一致しなければ「モードごとに別の X を見た」
    (= 軌道を作り直した / ctx を振り直した) ことになる。
    """
    mc_raw = {row.mc_total_raw for row in tiny_comparison.memory_capacity}
    ipc_raw = {row.ipc_total_raw for row in tiny_comparison.ipc}
    assert len(mc_raw) == 1, mc_raw
    assert len(ipc_raw) == 1, ipc_raw

    for mc_row in tiny_comparison.memory_capacity:
        assert mc_row.mc_total <= mc_row.mc_total_raw + 1.0e-12
        if mc_row.threshold_mode == THRESHOLD_NONE:
            assert mc_row.mc_total == mc_row.mc_total_raw
            assert mc_row.mc_threshold == 0.0
        else:
            assert mc_row.mc_threshold > 0.0
    for ipc_row in tiny_comparison.ipc:
        assert ipc_row.ipc_total <= ipc_row.ipc_total_raw + 1.0e-12
        assert ipc_row.ipc_linear + ipc_row.ipc_nonlinear == pytest.approx(
            ipc_row.ipc_total, rel=1.0e-9
        )
        if ipc_row.threshold_mode == THRESHOLD_NONE:
            assert ipc_row.ipc_total == ipc_row.ipc_total_raw
            assert ipc_row.ipc_threshold_degree1 == 0.0
        else:
            assert ipc_row.ipc_threshold_degree1 > 0.0
            assert ipc_row.n_targets_kept < _row_kept(tiny_comparison, THRESHOLD_NONE)


def _row_kept(comparison: ThresholdComparison, mode: str) -> int:
    return next(
        row.n_targets_kept for row in comparison.ipc if row.threshold_mode == mode
    )


def test_default_modes_are_reported(tiny_comparison: ThresholdComparison) -> None:
    """既定モードが表の中の実在の行を指す (どれが本番を作った行かが分かる)。"""
    assert tiny_comparison.default_mc_mode in {
        row.threshold_mode for row in tiny_comparison.memory_capacity
    }
    assert tiny_comparison.default_ipc_mode in {
        row.threshold_mode for row in tiny_comparison.ipc
    }


# --- 5. コミット済みの本番成果物との突合 -------------------------------------


def test_committed_meta_json_has_the_threshold_comparison() -> None:
    """本番 ``meta.json`` に受け入れ条件3 の一次資料が在る。"""
    comparison = _comparison()
    assert [
        row["threshold_mode"] for row in _rows_of(comparison, "memory_capacity")
    ] == [*MC_THRESHOLD_MODES]
    assert [row["threshold_mode"] for row in _rows_of(comparison, "ipc")] == [
        *IPC_THRESHOLD_MODES
    ]


def test_default_mode_matches_the_committed_capacity_row() -> None:
    """既定モードの総容量が ``capacity.csv`` の同一条件の行と一致する。

    ここが受け入れ条件3 の要 —— 比較表が本番と無関係な数字になっていないこと
    の実測である。同時に ``none`` の総容量が同じ行の ``*_raw`` 列 (しきい値前)
    と一致することも見る (しきい値前はモードに依らない量なので、
    ``threshold_mode='none'`` が本当に「切っていない」ことの裏付けになる)。
    """
    comparison = _comparison()
    condition: dict[str, object] = comparison["condition"]  # type: ignore[assignment]
    row = next(
        record
        for record in _capacity_rows()
        if record["experiment"] == condition["experiment"]
        and float(record["rho"]) == condition["rho"]
        and float(record["leak_rate"]) == condition["leak_rate"]
        and int(record["replicate"]) == condition["replicate"]
    )

    default_mc = _row_with(
        _rows_of(comparison, "memory_capacity"), str(comparison["default_mc_mode"])
    )
    default_ipc = _row_with(
        _rows_of(comparison, "ipc"), str(comparison["default_ipc_mode"])
    )
    for key in ("mc_total", "mc_total_raw", "mc_threshold", "mc_effective_delay"):
        assert default_mc[key] == pytest.approx(float(row[key]), rel=1.0e-9), key
    for key in ("ipc_total", "ipc_total_raw", "ipc_linear", "ipc_nonlinear"):
        assert default_ipc[key] == pytest.approx(float(row[key]), rel=1.0e-9), key
    assert default_ipc["n_targets_kept"] == int(row["n_targets_kept"])

    none_mc = _row_with(_rows_of(comparison, "memory_capacity"), THRESHOLD_NONE)
    none_ipc = _row_with(_rows_of(comparison, "ipc"), THRESHOLD_NONE)
    assert none_mc["mc_total"] == pytest.approx(float(row["mc_total_raw"]), rel=1.0e-9)
    assert none_ipc["ipc_total"] == pytest.approx(
        float(row["ipc_total_raw"]), rel=1.0e-9
    )


# --- 6. 確保より前の上限検査を素通りしない (F-3b2-1-001/HIGH-1) -------------


def test_oversized_ipc_sweep_n_units_is_rejected_before_any_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """代表条件の ``n_units`` が上限超過なら、軌道を作る前に ``ValueError``。

    HIGH-1 (3b-2 reviewer-security): ``run_threshold_comparison`` は
    ``CapacityCondition`` (``comparison_condition``) を組み立てながら
    ``experiment/capacity.py`` の ``_validate_condition_bounds`` を1回も
    呼ばずに素通りしていた。``simulate_condition_trajectory`` に一本化した
    後は掃引3経路とまったく同じ検査を通る。
    """
    capacity_module = importlib.import_module("rc_basics_lab.experiment.capacity")
    called = False

    def fail_if_called(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("simulate_reference_trajectory が呼ばれました")

    monkeypatch.setattr(
        "rc_basics_lab.experiment.capacity.simulate_reference_trajectory",
        fail_if_called,
    )
    config = tiny_config()
    huge = dataclasses.replace(
        config,
        ipc_sweep=dataclasses.replace(
            config.ipc_sweep, n_units=capacity_module._MAX_UNITS + 1
        ),
    )
    with pytest.raises(ValueError, match="n_units"):
        run_threshold_comparison(huge)
    assert not called, "上限検査より前に状態行列の確保が始まっています"


def test_validate_condition_bounds_is_actually_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_validate_condition_bounds`` の呼び出し回数を直接数える (実測、完了条件3)。

    ``ValueError`` が出ることだけを見るテストは「別の経路でたまたま落ちた」
    可能性を排除できない。呼び出し回数そのものを固定する。
    """
    capacity_module = importlib.import_module("rc_basics_lab.experiment.capacity")
    calls: list[object] = []
    real = capacity_module._validate_condition_bounds

    def counting(condition: object) -> None:
        calls.append(condition)
        real(condition)

    monkeypatch.setattr(
        "rc_basics_lab.experiment.capacity._validate_condition_bounds", counting
    )
    run_threshold_comparison(tiny_config())
    assert len(calls) == 1, (
        f"_validate_condition_bounds の呼び出し回数={len(calls)} "
        "(1回だけ CapacityCondition を組んで軌道を作るので1回のはず)"
    )

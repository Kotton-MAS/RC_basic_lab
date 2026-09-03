"""容量測定を**他実装の算法**と突き合わせる (E-1 / D-148).

D-142 の解析解オラクル (完全遅延線で MC = 段数) は「実装が数学どおりか」を
見る。ここが見るのは別のこと —— **文献と同じ土俵か**である。同じ実リザバーの
状態行列に、こちらの経路 (設計行列 + リッジ回帰) と参照実装の経路
(状態行列の SVD で正規直交基底を作り、目標をそこへ射影) の2つを通して、
per-delay の容量が一致することを測る。

参照実装: `kubota0130/ipc <https://github.com/kubota0130/ipc>`_ (MIT,
Copyright (c) 2022 kubota0130)。``docs/series/要件_rc-basics-03.md`` が
IPC のしきい値処理の参照実装として挙げているもの。

**あちらのコードをそのまま動かしてはいない。** 理由は2つ:

- ``cupy`` (GPU) と ``pandas`` に依存しており、CI の実行環境で動かない
- テスト時に外部リポジトリを取りに行くと、CI がネットワークに依存する

代わりに**算法の核だけを numpy へ書き写した** (``_reference_capacity``)。
写したのは ``ipc.singular_value_decomposition`` (中心化した状態行列の SVD と
特異値のしきい値) と ``ipc.get_ipc`` (射影係数の二乗和) の2つで、どちらも
20 行に満たない。**写した以上、写し間違いは一致しないことで露見する** ——
こちらの実装とは行合わせも解き方も違うので、両方が同じ間違いをすることは
考えにくい。

実測 (N=50 / T=20000 / 遅延 30 本): 最大絶対差 4.7e-7、合計 12.869596 対
12.869590。差はこちらがリッジ (alpha=1e-9) を掛けているぶんである。
"""

from __future__ import annotations

import numpy as np
import pytest

from rc_basics_lab.config import Capacity03Config, load_config_as
from rc_basics_lab.diagnostics.base import DiagnosticContext
from rc_basics_lab.diagnostics.memory_capacity import (
    MemoryCapacityConfig,
    memory_capacity,
)
from rc_basics_lab.experiment.capacity import (
    drive_config_for,
    reservoir_config_for,
)
from rc_basics_lab.experiment.esp import simulate_reference_trajectory
from rc_basics_lab.experiment.topology_ladder import _ladder_condition, matched_levels
from rc_basics_lab.types import FloatArray

GOLDEN_CONFIG = "tests/golden/configs/03_capacity.yaml"

WASHOUT = 200
"""捨てる先頭の長さ。参照実装の ``Two`` に相当する。"""

MAX_DELAY = 30
"""突き合わせる遅延の本数。**両実装が同じ目標を見る**ことが要点。"""

TOLERANCE = 2.0e-4
"""許容差。差の出どころは**こちらがリッジ (alpha=1e-9) を掛けている**ぶんで、
系列が短いほど大きく出る。

実測: 本番設定 (N=50 / T=20000) で最大 **4.7e-7**、この検査が使うゴールデン
設定 (T が短い) で最大 **4.4e-5**。容量そのものが 0〜1 の量なので、4.4e-5 は
相対 4e-5 である。許容を 2e-4 に置いたのは、**行合わせを1つずらす変異が
0.1 以上ずれる**ため、その 500 分の1 でも検出力は落ちないからである。
"""


def _states() -> tuple[FloatArray, FloatArray]:
    """実リザバーの状態行列と駆動入力 (合成の遅延線ではない)。

    D-142 のオラクルは完全遅延線という**作った系**を見る。こちらは本番と
    同じ経路で作った状態を見る —— 有限標本・悪条件・中心化の扱いは、作った
    系では出てこない。
    """
    config = load_config_as(GOLDEN_CONFIG, Capacity03Config)
    section = config.topology_ladder
    condition = _ladder_condition(section, 0)
    topology = matched_levels(section.levels, section.n_units)[0]
    trajectory = simulate_reference_trajectory(
        reservoir_config_for(config, condition),
        drive_config_for(config, condition),
        reservoir_seed=config.seeds.reservoir,
        drive_seed=config.seeds.drive,
        rho=section.rho,
        leak_rate=section.leak_rate,
        sigma_u=section.sigma_u,
        replicate=0,
        topology=topology,
        graph_replicate=0,
    )
    return trajectory.states, trajectory.drive


def _reference_capacity(states: FloatArray, drive: FloatArray) -> FloatArray:
    """参照実装の算法で per-delay の容量を出す (kubota0130/ipc, MIT)。

    1. 状態行列を時間方向に中心化して SVD にかけ、特異値が
       ``max(sigma^2) * N * eps`` を超える右特異ベクトルだけを残す
       (``singular_value_decomposition``)
    2. 目標 ``z`` をその正規直交基底へ射影し、係数の二乗和を ``z.z`` で割る
       (``get_coef`` / ``get_ipc``)

    こちらの実装 (設計行列を組んでリッジで解き、決定係数を取る) とは
    **解き方も行合わせも違う**。
    """
    window = states[WASHOUT:].T
    centred = window - window.mean(axis=1, keepdims=True)
    _, sigma, right = np.linalg.svd(centred, full_matrices=False)
    threshold = (sigma**2).max() * centred.shape[0] * np.finfo(centred.dtype).eps
    basis = right[(sigma**2) > threshold]
    n_samples = centred.shape[1]
    capacities: list[float] = []
    for delay in range(1, MAX_DELAY + 1):
        target = drive[WASHOUT - delay : WASHOUT - delay + n_samples, 0]
        target = target - target.mean()
        capacities.append(float(np.sum((basis @ target) ** 2) / float(target @ target)))
    return np.asarray(capacities, dtype=np.float64)


def test_the_per_delay_capacity_matches_the_reference_algorithm() -> None:
    """**同じ状態行列に別の算法を通しても同じ容量が出る** (E-1 / D-148)。

    変異注入: ``memory_capacity`` の遅延を1つずらすと (``D-142`` の
    行合わせの変異) ここも赤くなる —— 参照側は独立に行を合わせているため。
    """
    states, drive = _states()
    ours = memory_capacity(
        states,
        drive,
        ctx=DiagnosticContext(washout=WASHOUT, seed=12345),
        cfg=MemoryCapacityConfig(max_delay=MAX_DELAY, threshold_mode="none"),
    ).arrays["mc_profile"]
    reference = _reference_capacity(states, drive)

    assert ours.shape == reference.shape
    assert ours == pytest.approx(reference, abs=TOLERANCE), (
        "こちらの容量と参照実装の算法が食い違います (最大差 "
        f"{float(np.max(np.abs(ours - reference))):.3e})"
    )
    # 空虚でないこと: 容量がほとんど立っていない状態では一致しても意味がない
    assert float(ours.sum()) > 1.0, f"容量がほとんど立っていません: {ours.sum()}"


def test_the_reference_algorithm_is_not_a_copy_of_ours() -> None:
    """参照側が**こちらの実装を経由していない**ことの確認。

    オラクルがこちらのコードを呼んでいたら、一致は自明で何も測っていない。
    状態行列を壊すと参照側だけが追随することで、独立していることを見る。
    """
    states, drive = _states()
    reference = _reference_capacity(states, drive)
    shuffled = np.array(states)
    rng = np.random.default_rng(0)
    rng.shuffle(shuffled, axis=0)
    broken = _reference_capacity(shuffled, drive)
    assert float(reference.sum()) > float(broken.sum()) + 1.0, (
        "状態行列を壊しても参照側の容量が落ちません (目標だけを見ています)"
    )

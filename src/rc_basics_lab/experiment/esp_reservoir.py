"""1条件ぶんの ``ESNConfig`` を組む層 (D-137).

``esp.py`` が 600 行の上限 (D-63 / D-77) を超えて凍結されているので切り出した。
02 の掃引だけでなく 03 (容量) もここを通るので、**ESN の組み立ての単一の場所**
である。
"""

from __future__ import annotations

from rc_basics_lab.config import ReservoirSweepConfig
from rc_basics_lab.reservoir.esn import ESNConfig
from rc_basics_lab.reservoir.topology import ErdosRenyiConfig, TopologyConfig

BIAS_SCALE = 0.0
"""02 の ESN が使うバイアス幅。**0 に固定する** (実装メモ / Q2)。

``ESNConfig`` の既定は 0.1 だが、定数バイアスは ``[1; u]`` の先頭成分に掛かる
**振幅一定の入力そのもの**であり、``sigma_u = 0`` を「無入力」と呼べなくなる。
実測: ``bias_scale=0.1`` では無入力・rho=1.2 でも2軌道が収束してしまい、
受け入れ条件1 (「無入力で rho>1 なら非収束」) が成立しない。D-17 が入力強度を
駆動信号の標準偏差で定義している以上、その定義に入らない常時入力は 0 にする。
"""


def build_esn_config(
    reservoir: ReservoirSweepConfig,
    rho: float,
    leak_rate: float,
    *,
    state_noise: float = 0.0,
    topology: TopologyConfig | None = None,
) -> ESNConfig:
    """1条件ぶんの ``ESNConfig`` を組む。掃引軸だけが条件ごとに変わる。

    引数は ``ReservoirSweepConfig`` に narrow してある (F-1-005)。本体が読むのは
    ``reservoir`` の4フィールドと ``BIAS_SCALE`` だけで、``Esp02Config`` に型で
    結合すると 03 が ESN 構成の再利用のために丸ごと写経する羽目になる。

    ``state_noise`` は **既定値つきキーワード**である (D-36)。03 の 3-B' は
    「ノイズ下では IPC_total が厳密に N 未満」(受け入れ条件2) を測るために
    状態ノイズを掛ける必要があるが、02 の呼び出しは書き換えない。
    ``state_noise=0`` では ``ESN`` が乱数を1個も引かないため、02 の成果物は
    バイト単位で不変である
    (``tests/test_experiment_capacity.py::test_reference_states_match_esp_simulate_condition``)。
    """
    return ESNConfig(
        n_units=reservoir.n_units,
        spectral_radius=rho,
        leak_rate=leak_rate,
        input_scale=reservoir.input_scale,
        bias_scale=BIAS_SCALE,
        # None なら横断共有の density から Erdos-Renyi を組む (従来どおり)。
        # 渡せばそのトポロジで回る —— 03 の T 依存の掃引が使う (D-137)。
        topology=(
            ErdosRenyiConfig(density=reservoir.density)
            if topology is None
            else topology
        ),
        state_noise=state_noise,
    )


__all__ = ["BIAS_SCALE", "build_esn_config"]

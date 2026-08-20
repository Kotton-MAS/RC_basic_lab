"""実験設定 (YAML) の読み込み.

frozen dataclass 群を単一の真実とし、YAML はそこへ値を流し込むだけにする。
**未知キーは即座に ``ConfigError``** とする (D-09)。本連載は十数個のパラメータを
YAML 化するため、キーのタイプミスが黙って無視されると「設定したのに効いていない」
実験結果が生まれる。

新しいセクションを足すときは dataclass にフィールドを追加するだけでよい
(ローダはフィールド型から再帰的に構築する)。

**実験ごとに設定 dataclass を分ける** (D-13)。``ExperimentConfig`` は 01 専用で、
02 は ``Esp02Config``、03 は ``Capacity03Config`` を使う。ローダ本体は
``load_config_as(path, cls)`` として
共有し、``load_config`` はその 01 向けの別名 (呼び出し互換のため署名を保つ)。
02 のフィールドを ``ExperimentConfig`` に相乗りさせると
``tests/test_config_wiring.py::test_each_parameter_changes_output``
(「全フィールドが 01 のパイプライン出力を変える」) を満たせないフィールドが
生まれ、逃がすための例外チャネルを増やすと配線漏れの検出力そのものが落ちる。

診断の設定 dataclass (``EspConfig`` / ``LyapunovConfig`` / ``TimescaleConfig``
/ ``MemoryCapacityConfig`` / ``IpcConfig``) は ``diagnostics/`` 側に定義し、
ここが import する。``config -> diagnostics``
は許可された向きで、逆向きは D-12 が禁じている。
"""

from __future__ import annotations

from rc_basics_lab.config._common import ConfigError, load_config_as
from rc_basics_lab.config.anomaly05 import SyntheticAnomalyConfig
from rc_basics_lab.config.capacity03 import (
    Capacity03Config,
    CapacityDriveConfig,
    CapacityReservoirConfig,
    CapacitySeedConfig,
    ConservationConfig,
    IpcSweepConfig,
    LengthSweepConfig,
    McSweepConfig,
    Narma10Config,
)
from rc_basics_lab.config.chaos04 import (
    LORENZ_LYAPUNOV_REFERENCE,
    Chaos04Config,
    FreeRunConfig,
    LorenzConfig,
    MackeyGlassStandardizeConfig,
    StabilityConfig,
)
from rc_basics_lab.config.esp02 import (
    DEFAULT_ESP_MAP_RHO_GRID,
    DEFAULT_ESP_MAP_SIGMA_GRID,
    DriveConfig,
    Esp02Config,
    EspDecayConfig,
    EspMapConfig,
    EspSeedConfig,
    ReservoirSweepConfig,
    TimescaleSweepConfig,
    WashoutSweepConfig,
    esp_stream_seed,
)
from rc_basics_lab.config.experiment01 import (
    DEFAULT_ALPHA_GRID,
    TASK_LENGTH_FIELDS,
    DelayParityConfig,
    ExperimentConfig,
    MackeyGlassConfig,
    RidgeConfig,
    SplitConfig,
    load_config,
)
from rc_basics_lab.diagnostics.esp import EspConfig, LyapunovConfig
from rc_basics_lab.diagnostics.ipc import IpcConfig
from rc_basics_lab.diagnostics.lyapunov import MaxLyapunovConfig
from rc_basics_lab.diagnostics.memory_capacity import MemoryCapacityConfig
from rc_basics_lab.diagnostics.timescale import TimescaleConfig
from rc_basics_lab.reservoir.esn import ESNConfig

__all__ = [
    "DEFAULT_ALPHA_GRID",
    "DEFAULT_ESP_MAP_RHO_GRID",
    "DEFAULT_ESP_MAP_SIGMA_GRID",
    "LORENZ_LYAPUNOV_REFERENCE",
    "TASK_LENGTH_FIELDS",
    "Capacity03Config",
    "CapacityDriveConfig",
    "CapacityReservoirConfig",
    "CapacitySeedConfig",
    "Chaos04Config",
    "ConfigError",
    "ConservationConfig",
    "DelayParityConfig",
    "DriveConfig",
    "ESNConfig",
    "Esp02Config",
    "EspConfig",
    "EspDecayConfig",
    "EspMapConfig",
    "EspSeedConfig",
    "ExperimentConfig",
    "FreeRunConfig",
    "IpcConfig",
    "IpcSweepConfig",
    "LengthSweepConfig",
    "LorenzConfig",
    "LyapunovConfig",
    "MackeyGlassConfig",
    "MackeyGlassStandardizeConfig",
    "MaxLyapunovConfig",
    "McSweepConfig",
    "MemoryCapacityConfig",
    "Narma10Config",
    "ReservoirSweepConfig",
    "RidgeConfig",
    "SplitConfig",
    "StabilityConfig",
    "SyntheticAnomalyConfig",
    "TimescaleConfig",
    "TimescaleSweepConfig",
    "WashoutSweepConfig",
    "esp_stream_seed",
    "load_config",
    "load_config_as",
]

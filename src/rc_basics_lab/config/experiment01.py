"""実験01 (What is RC) の設定 dataclass 群 (D-13).

**実験ごとに設定 dataclass を分ける** (D-13) の 01 側。``ExperimentConfig`` は
01 専用で、02 は ``esp02.Esp02Config``、03 は ``capacity03.Capacity03Config``
を使う。ローダ本体 (``_common.load_config_as``) だけを共有する。

``ExperimentConfig`` は 02 の ``WashoutSweepConfig.base`` / 03 の
``Narma10Config.base`` に内包されるため、このモジュールは ``config`` package の
中で最も下流から参照される。**依存は ``_common`` への一方向だけ**で、
``esp02`` / ``capacity03`` をここから import しない (D-49)。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rc_basics_lab.config._common import load_config_as
from rc_basics_lab.reservoir.esn import ESNConfig
from rc_basics_lab.reservoir.protocol import ReservoirConfig
from rc_basics_lab.seeds import SeedConfig

DEFAULT_ALPHA_GRID: tuple[float, ...] = (
    1e-08,
    1e-07,
    1e-06,
    1e-05,
    1e-04,
    1e-03,
    1e-02,
    1e-01,
    1.0,
    10.0,
    100.0,
)
"""既定の ridge alpha 格子 (仕様 T3)。全手法・全タスクがこの単一格子を読む (D-04)。

**literal で書く。``np.logspace(-8, 2, 11)`` で計算してはならない。** 指数側
(``linspace``) は厳密でも ``power(10.0, -5.0)`` の最下位ビットが libm 実装に
依存し、macOS arm64 では ``1e-05``、Linux x86_64 では ``9.999999999999999e-06``
になる。本番 YAML は literal (``1.0e-05``) を書くので、計算した既定と YAML が
Linux でだけ食い違い、``tests/test_config_wiring_chaos.py::
test_production_config_matches_the_committed_yaml`` が CI でだけ落ちていた。

設定の既定値はプラットフォームに依存してはならない。
"""


@dataclass(frozen=True, slots=True)
class MackeyGlassConfig:
    """Mackey-Glass 系列の生成パラメータ (仕様 §3 未確定1 の決定値)。"""

    tau: float = 17.0
    beta: float = 0.2
    gamma: float = 0.1
    exponent: int = 10
    rk4_step: float = 0.1
    sample_interval: int = 10
    integration_burn_in: int = 1000
    length: int = 8000
    horizon: int = 1


@dataclass(frozen=True, slots=True)
class DelayParityConfig:
    """遅延パリティ課題の生成パラメータ (D-07)。"""

    n_bits: int = 2
    delay: int = 1
    length: int = 8000


@dataclass(frozen=True, slots=True)
class RidgeConfig:
    """リッジ回帰の設定。``alpha_grid`` は全手法が共有する単一キー (D-04)。"""

    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID
    n_lags_grid: tuple[int, ...] = (1, 2, 4, 8, 16)


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """時系列を連続区間で切る分割設定 (シャッフルしない)。"""

    train_ratio: float = 0.5
    val_ratio: float = 0.15
    test_ratio: float = 0.35
    washout: int = 200
    max_start_offset: int = 200


def _delay_parity_esn() -> ESNConfig:
    """遅延パリティ用 ESN の既定値 (仕様 T3 / docs/design.md §6)。

    パリティは瞬時的な非線形結合を要求するため、漏れを入れず (leak=1.0)、
    入力を強く駆動する (input_scale=1.0)。MG 用 (``ESNConfig`` の既定値) とは
    別の組であり、**検証分割では調整しない** (D-08)。
    """
    return ESNConfig(
        n_units=200,
        spectral_radius=0.9,
        leak_rate=1.0,
        input_scale=1.0,
        density=0.1,
    )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """実験1本ぶんの設定。

    リザバーの構造ハイパーパラメータは課題ごとに別セクション
    (``esn_mackey_glass`` / ``esn_delay_parity``) に持つ。MG は漏れ積分
    (leak=0.3) が効き、パリティは leak=1.0 が要るため、1つのセクションに
    まとめると片方の課題に不利な値を押し付けることになるため。

    型は ``ReservoirConfig`` である (``ESNConfig`` ではない)。**モデルを足す
    ときに触るのは ``reservoir/`` だけ**で、ここは変えなくてよい。YAML では
    ``kind: esn`` で明示でき、省略すると既定 (ESN) になる。
    """

    name: str = "01_what_is_rc"
    n_replicates: int = 5
    seeds: SeedConfig = field(default_factory=SeedConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    mackey_glass: MackeyGlassConfig = field(default_factory=MackeyGlassConfig)
    delay_parity: DelayParityConfig = field(default_factory=DelayParityConfig)
    esn_mackey_glass: ReservoirConfig = field(default_factory=ESNConfig)
    esn_delay_parity: ReservoirConfig = field(default_factory=_delay_parity_esn)


TASK_LENGTH_FIELDS: Mapping[str, str] = {
    "mackey_glass": "mackey_glass",
    "delay_parity": "delay_parity",
}
"""``build_tasks`` (``experiment/runner.py``) が返す課題名 -> ``ExperimentConfig``
上で対応する、系列長 (``length: int`` 属性) を持つフィールド名。

課題の列挙点は ``build_tasks`` が唯一の真実 (``conventions.md``) だが、washout
感度実験 (D-19) の系列長補償 (``experiment.washout.variant_for``) は
「どの課題がどのフィールドの ``length`` を持つか」を別に知る必要がある。この
対応をここ1か所に集約し、``build_tasks`` に課題を追加してもここへの登録を
忘れると、その課題の系列長は補償されず D-19 の交絡除去が黙って効かなくなる
(``variant_for`` は未登録の課題があれば ``ValueError`` にする)。
"""


def load_config(
    path: Path | str,
    *,
    preset: Path | str | None = None,
    overrides: Sequence[str] = (),
) -> ExperimentConfig:
    """YAML から 01 の ``ExperimentConfig`` を読み込む。

    ``load_config_as(path, ExperimentConfig, ...)`` への委譲。既存の呼び出し
    (``experiments/01_what_is_rc/run.py`` / ``main.py``) を壊さないため、
    ``preset`` / ``overrides`` は**キーワード専用で既定なし**にしてある。

    Raises:
        ConfigError: ファイルが無い / 未知キーがある / 型が合わない場合。
        OverrideError: ``--set`` の書式か経路が不正な場合。
    """
    return load_config_as(path, ExperimentConfig, preset=preset, overrides=overrides)

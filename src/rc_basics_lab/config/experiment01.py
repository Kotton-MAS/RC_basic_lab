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

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from rc_basics_lab.config._common import load_config_as
from rc_basics_lab.reservoir.esn import ESNConfig
from rc_basics_lab.reservoir.protocol import ReservoirConfig
from rc_basics_lab.reservoir.topology import ErdosRenyiConfig
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
class CrossValidationConfig:
    """alpha を交差検証で選ぶ設定 (既定は**使わない**)。

    Attributes:
        n_folds: 折りの数。**0 なら交差検証を使わない** (訓練 + 検証の
            単一分割で選ぶ、従来の挙動)。有効にするなら 2 以上。
        scheme: 折り方。``rolling`` は訓練区間が常に検証区間より前で、未来から
            過去への漏れを構造上作れない。``blocked`` はデータを使い切れるが
            時間の向きを無視する (``readout/cross_validation.py`` の注を参照)。
        embargo: 訓練区間と検証区間のあいだに捨てる行数。**設計行列の
            ``first_valid`` 以上にすること** —— それ未満だと検証行が訓練行と
            同じ入力を含む。``null`` なら ``first_valid`` を自動で使う。

    **既定を変えない理由**: ``results/`` は単一分割で作られており、選び方を
    変えると記事の数値が変わる。交差検証は明示的に有効にする。
    """

    n_folds: int = 0
    scheme: str = "rolling"
    embargo: int | None = None


@dataclass(frozen=True, slots=True)
class RidgeConfig:
    """リッジ回帰の設定。``alpha_grid`` は全手法が共有する単一キー (D-04)。"""

    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID
    n_lags_grid: tuple[int, ...] = (1, 2, 4, 8, 16)
    cv: CrossValidationConfig = field(default_factory=CrossValidationConfig)


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
        topology=ErdosRenyiConfig(density=0.1),
    )


@dataclass(frozen=True, slots=True)
class MackeyGlassTask:
    """Mackey-Glass 課題と、その課題で使うリザバーの組 (D-123)。"""

    KIND: ClassVar[str] = "mackey_glass"

    params: MackeyGlassConfig = field(default_factory=MackeyGlassConfig)
    reservoir: ReservoirConfig = field(default_factory=ESNConfig)


@dataclass(frozen=True, slots=True)
class DelayParityTask:
    """遅延パリティ課題と、その課題で使うリザバーの組 (D-123)。"""

    KIND: ClassVar[str] = "delay_parity"

    params: DelayParityConfig = field(default_factory=DelayParityConfig)
    reservoir: ReservoirConfig = field(default_factory=_delay_parity_esn)


type TaskSpec = MackeyGlassTask | DelayParityTask
"""課題とリザバーの組。**課題を足したら union に1行足す** (D-123)。

``kind`` で選ぶ判別子つき union なので、YAML は課題のリストになる:

.. code-block:: yaml

    tasks:
      - kind: mackey_glass
        params: {length: 8200, horizon: 1}
        reservoir: {leak_rate: 0.3}

``params`` と ``reservoir`` を分けるのは、``MackeyGlassConfig`` を 04 / 05 も
使うためである。課題設定にリザバーを埋めると、リザバーを使わない経路まで
その設定を引きずる。
"""


def with_length(spec: TaskSpec, length: int) -> TaskSpec:
    """系列長だけを差し替えた同じ種類の課題エントリを返す (D-123)。

    02 の washout 感度実験 (D-19) が ``t0`` の増分だけ系列を伸ばすのに使う。

    **``match`` を1つ書く必要はここに残る。** ``dataclasses.replace`` は
    union の上では型が解けず (``params`` が ``Never`` に落ちる)、要素ごとに
    絞ってからでないと書けないため。かつては
    ``config.TASK_LENGTH_FIELDS`` への登録と ``variant_for`` 本体の
    キーワード引数という**実行時にしか気づけない2段**だったので、
    mypy の網羅性検査が見る ``match`` 1つに移ったぶんは前進である
    (足し忘れは型検査で落ちる)。

    Args:
        spec: 課題エントリ。
        length: 新しい系列長 [ステップ]。

    Returns:
        ``params.length`` だけを差し替えた同じ種類のエントリ。
    """
    match spec:
        case MackeyGlassTask():
            return dataclasses.replace(
                spec, params=dataclasses.replace(spec.params, length=length)
            )
        case DelayParityTask():
            return dataclasses.replace(
                spec, params=dataclasses.replace(spec.params, length=length)
            )


def _default_tasks() -> tuple[TaskSpec, ...]:
    """既定の課題列 (**並び順が ``comparison.csv`` の課題の順**)。"""
    return (MackeyGlassTask(), DelayParityTask())


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
    tasks: tuple[TaskSpec, ...] = field(default_factory=_default_tasks)


def require_task[T: TaskSpec](
    config: ExperimentConfig, kind: type[T], used_by: str
) -> T:
    """課題列から1つを型で取り出す (D-123)。**無ければ落とす**。

    ``reservoir.registry.require_esn`` と同じ流儀である。``config.tasks`` は
    YAML で並べ替えも削除もできるので、02〜05 が「MG があるはず」と決め打つと
    課題を1つ外しただけで**別の課題の設定を使って走る**ことになる。

    Args:
        config: 01 の設定 (02〜05 は ``base`` に持つ)。
        kind: 取り出したい ``TaskSpec`` の型。
        used_by: 呼び出し元の説明 (エラーに出す)。

    Returns:
        その型の課題エントリ (**最初の1つ**)。

    Raises:
        ValueError: その課題が ``config.tasks`` に無い場合。
    """
    for spec in config.tasks:
        if isinstance(spec, kind):
            return spec
    present = ", ".join(type(spec).KIND for spec in config.tasks)
    raise ValueError(
        f"{used_by} は課題 {kind.KIND} を要求しますが、tasks にありません "
        f"(あるのは: {present or 'なし'})"
    )


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

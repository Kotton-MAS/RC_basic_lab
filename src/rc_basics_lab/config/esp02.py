"""実験02 (ESP・スペクトル半径・リーク率) の設定 dataclass 群 (D-13).

01 の ``ExperimentConfig`` とはローダ (``_common.load_config_as``) だけを共有し、
フィールドは一切共有しない。``esp`` / ``lyapunov`` / ``timescale`` は診断層の
設定 (D-15) をそのまま載せたもので、``config -> diagnostics`` は許可された向き
(逆向きは D-12 / D-23 が禁じている)。

``experiment01`` への import は ``WashoutSweepConfig.base`` (2-D が 01 の
``run_experiment`` を再利用するための内包) の1本だけで、**一方向**である
(D-49)。``capacity03`` はここから import しない。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rc_basics_lab.config.experiment01 import ExperimentConfig
from rc_basics_lab.diagnostics.esp import EspConfig, LyapunovConfig
from rc_basics_lab.diagnostics.timescale import TimescaleConfig
from rc_basics_lab.seeds import SeedStream

DEFAULT_ESP_MAP_RHO_GRID: tuple[float, ...] = tuple(
    round(float(value), 3) for value in np.linspace(0.4, 1.9, 16)
)
"""実験 2-C の rho 格子 (16点、仕様 §8 Q3)。1.0 の両側を等間隔に挟む。"""

DEFAULT_ESP_MAP_SIGMA_GRID: tuple[float, ...] = (
    0.0,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
)
"""実験 2-C の入力強度 sigma_u 格子 (7点、仕様 §8 Q3)。

sigma=0 は「無入力」で別枠 (図では別パネル)。以降は対数的に広げ、rho>1 でも ESP が
成立する側 (sigma>=1) まで届かせる。強度は**標準偏差**であって振幅ではない (D-17)。
"""


@dataclass(frozen=True, slots=True)
class EspSeedConfig:
    """02 の実験 2-A / 2-B / 2-C が使う3ストリームの基底シード (D-14)。

    01 の ``SeedConfig`` とは別クラスにする。初期状態対は ``probe``
    ストリームから引き、リザバー重み (``reservoir``) や駆動信号 (``drive``) と
    独立に振れる必要があるため。``SeedConfig`` に ``probe`` を足すと 01 の
    配線テストの被覆が破れる (D-13 と同じ理由)。

    Attributes:
        reservoir: リザバー重みの基底シード (``SeedStream.RESERVOIR``)。
        drive: 駆動入力の基底シード (``SeedStream.TASK``)。
        probe: 初期状態対の基底シード (``SeedStream.PROBE``)。
    """

    reservoir: int = 0
    drive: int = 1
    probe: int = 3


def esp_stream_seed(seeds: EspSeedConfig, stream: SeedStream) -> int:
    """02 の設定からストリームの基底シードを取り出す (D-14)。

    他ストリームのシードを一切参照しないことが独立性の根拠なので、
    ``getattr`` ではなく明示的な分岐で書く (``seeds._base_seed`` と同じ流儀)。

    Raises:
        ValueError: ``EspSeedConfig`` が基底シードを持たないストリームの場合。
    """
    match stream:
        case SeedStream.RESERVOIR:
            return seeds.reservoir
        case SeedStream.TASK:
            return seeds.drive
        case SeedStream.PROBE:
            return seeds.probe
        case SeedStream.SPLIT:
            raise ValueError(
                "実験 2-A/2-B/2-C は分割を行いません (SPLIT ストリームは未使用)。"
                " 2-D の分割は washout.base.seeds.split を使ってください"
            )


@dataclass(frozen=True, slots=True)
class DriveConfig:
    """駆動入力と2軌道生成の共通条件 (実験 2-A / 2-B / 2-C)。

    Attributes:
        distribution: 駆動信号の分布。``"uniform"`` (i.i.d. 一様) 以外は未対応で、
            実験層が ``ValueError`` にする (黙って一様として扱わない)。
        n_steps: 生成する系列長 [ステップ]。
        washout: λ と自己相関から外す先頭ステップ数。ESP の距離当てはめには
            使わない (``experiment.esp.ESP_DISTANCE_WASHOUT`` を参照)。
        n_pairs: 参照軌道と比べる第2軌道の本数 (最悪値で判定する, D-16)。
            **既定を 10 にしてある** (T3 の実測で 3 から引き上げ)。無入力で
            rho>1 の ESN は ``+x*`` / ``-x*`` の対をなす吸引子を持つことがあり
            (tanh が奇関数のため)、比較軌道が k 本すべて参照軌道と同じ側へ
            落ちる確率は約 ``2^-k`` ある。実測: ``n_pairs=3`` では特定の
            リザバー draw で rho=1.2 / 1.5・無入力が「収束」と誤判定され、
            受け入れ条件1 が成立しなかった。10 本にすると全レプリケートで
            正しく非収束になり、rho<1 側の判定は変わらない (偽陰性なし)。
    """

    distribution: str = "uniform"
    n_steps: int = 3000
    washout: int = 200
    n_pairs: int = 10


@dataclass(frozen=True, slots=True)
class ReservoirSweepConfig:
    """2-A / 2-B / 2-C が共有するリザバー構造パラメータ (F-02-1-004)。

    ``EspDecayConfig`` / ``TimescaleSweepConfig`` / ``EspMapConfig`` は同一の
    リザバー族を見る図であり (仕様 §8 Q3: N=200 をサイクル1との連続性のため
    連載を通して固定する)、``input_scale`` / ``n_units`` / ``density`` /
    ``n_replicates`` をセクションごとに別々に持たせると、図の間で N が食い
    違っても何も落ちない。``Esp02Config`` 直下でセクション横断に1本だけ持つ。

    これは D-13 の「実験ごとに設定 dataclass を分ける」とは別の軸であり、
    「同じ実験内で複数の図が共有する土台」という位置づけである。この構成に
    より、各セクションが個別に ``n_units`` 等を名乗ることは構造的に禁じられる
    (YAML の未知キー検査 D-09 が ``decay.n_units`` のような書き方を
    ``ConfigError`` にする)。

    掃引軸 (``rho_grid`` / ``leak_rate_grid`` / ``sigma_grid``) はセクション
    固有の性質なので、ここには含めず各セクション側に残す。
    """

    input_scale: float = 1.0
    n_units: int = 200
    density: float = 0.1
    n_replicates: int = 3


@dataclass(frozen=True, slots=True)
class EspDecayConfig:
    """実験 2-A: rho を振ったときの状態距離の減衰曲線。

    無入力 (``sigma_u = 0``) が既定。rho<1 で指数減衰し rho>1 で減衰しないことを
    見る図であり、入力を入れると主張が変わる (受け入れ条件1)。リザバー構造は
    ``Esp02Config.reservoir`` を参照する (F-02-1-004)。
    """

    rho_grid: tuple[float, ...] = (0.5, 0.8, 0.95, 1.2, 1.5)
    sigma_u: float = 0.0
    leak_rate: float = 1.0


@dataclass(frozen=True, slots=True)
class TimescaleSweepConfig:
    """実験 2-B: リーク率を振ったときの実効時定数。

    理論線 ``-1 / log(1 - a)`` と重ねるため、``rho`` は 1 未満に固定して
    リーク率だけを動かす (受け入れ条件4)。最大ラグは診断側の
    ``Esp02Config.timescale.max_lag`` が持つ (二重定義しない)。リザバー構造は
    ``Esp02Config.reservoir`` を参照する (F-02-1-004)。
    """

    leak_rate_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
    rho: float = 0.9
    sigma_u: float = 0.5


@dataclass(frozen=True, slots=True)
class EspMapConfig:
    """実験 2-C: rho x 入力強度 の ESP 成立領域 (記事の目玉)。

    ``input_scale`` は掃引中固定し、動かすのは信号側の ``sigma_grid`` だけ
    (D-17)。同時に動かすと「信号を強くした」のか「重みを大きくした」のかを
    分離できなくなる。``input_scale`` を含むリザバー構造は
    ``Esp02Config.reservoir`` を参照する (F-02-1-004)。
    """

    rho_grid: tuple[float, ...] = DEFAULT_ESP_MAP_RHO_GRID
    sigma_grid: tuple[float, ...] = DEFAULT_ESP_MAP_SIGMA_GRID
    leak_rate: float = 1.0


@dataclass(frozen=True, slots=True)
class WashoutSweepConfig:
    """実験 2-D: washout 長への性能感度 (D-19)。

    実体は ``base`` の washout を差し替えて 01 の ``run_experiment`` を回す
    ループなので、公平性 (D-04 / D-05 / D-08) は既存経路が担保する。

    Attributes:
        grid: 掃引する washout の値 [ステップ]。
        pad_series: 真なら ``length`` を伸ばして訓練/検証/テストの行数を格子
            全体で一定に保つ (washout の効果と訓練データ量の効果の交絡を除く)。
            偽は交絡ありの設計を再現する対比用モード。
        base: 掃引の土台となる 01 の設定。``washout`` 以外は差し替えない。
    """

    grid: tuple[int, ...] = (0, 50, 100, 200, 400, 800)
    pad_series: bool = True
    base: ExperimentConfig = field(default_factory=ExperimentConfig)


def _esp_criteria_for_02() -> EspConfig:
    """02 の実験が使う ESP 判定基準 (``fit_skip`` だけ D-16 の既定から下げる)。

    D-16 の既定 ``fit_skip=50`` は「washout の直後からさらに捨てる」量として
    決めたものだが、02 の実験層は ESP の距離当てはめに washout を掛けない
    (``experiment.esp.ESP_DISTANCE_WASHOUT`` = 0。2-A は過渡そのものを見せる
    図であるため)。当てはめ開始は ``0 + fit_skip`` になるので、無入力
    rho=0.5 の距離が丸めの床に届く t≈46 より十分手前で始める必要がある。
    実測: ``fit_skip=10`` で減衰率と ``log rho`` の相対誤差は
    rho=0.5 → 5.1〜7.7% / rho=0.8 → 2.6〜5.3% / rho=0.95 → 0.2〜0.4%
    (受け入れ条件1 の許容 20% に対して十分な余裕がある)。
    """
    return EspConfig(fit_skip=10)


@dataclass(frozen=True, slots=True)
class Esp02Config:
    """実験02 (ESP・スペクトル半径・リーク率) 1本ぶんの設定 (D-13)。

    ``ExperimentConfig`` とはローダ (``load_config_as``) だけを共有し、
    フィールドは一切共有しない。2-D だけは 01 のパイプラインを再利用するため、
    ``washout.base`` として ``ExperimentConfig`` をまるごと内包する
    (``washout.base.*`` の配線は 01 側の
    ``tests/test_config_wiring.py`` が被覆する)。

    ``esp`` / ``lyapunov`` / ``timescale`` は診断層の設定 (D-15) をそのまま
    載せたもの。YAML から診断の判定基準まで届くのはこの経路だけである。

    ``reservoir`` の理由は ``ReservoirSweepConfig`` を参照する (F-02-1-004)。
    """

    name: str = "02_esp_and_dynamics"
    seeds: EspSeedConfig = field(default_factory=EspSeedConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)
    reservoir: ReservoirSweepConfig = field(default_factory=ReservoirSweepConfig)
    decay: EspDecayConfig = field(default_factory=EspDecayConfig)
    timescale_sweep: TimescaleSweepConfig = field(default_factory=TimescaleSweepConfig)
    esp_map: EspMapConfig = field(default_factory=EspMapConfig)
    washout: WashoutSweepConfig = field(default_factory=WashoutSweepConfig)
    esp: EspConfig = field(default_factory=_esp_criteria_for_02)
    lyapunov: LyapunovConfig = field(default_factory=LyapunovConfig)
    timescale: TimescaleConfig = field(default_factory=TimescaleConfig)

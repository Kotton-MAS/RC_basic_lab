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

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import UnionType
from typing import Protocol, cast, get_args, get_origin, get_type_hints

import numpy as np
import yaml

from rc_basics_lab.diagnostics.esp import EspConfig, LyapunovConfig
from rc_basics_lab.diagnostics.ipc import IpcConfig
from rc_basics_lab.diagnostics.memory_capacity import MemoryCapacityConfig
from rc_basics_lab.diagnostics.timescale import TimescaleConfig
from rc_basics_lab.reservoir.esn import ESNConfig
from rc_basics_lab.seeds import SeedConfig, SeedStream


class _DataclassFactory[T_co](Protocol):
    """dataclass のコンストラクタ。``Any`` を書かずにキーワード構築を型付けする。"""

    def __call__(self, **kwargs: object) -> T_co: ...


class ConfigError(ValueError):
    """設定ファイルの内容が dataclass 群と噛み合わないときに送出される。"""


DEFAULT_ALPHA_GRID: tuple[float, ...] = tuple(
    float(value) for value in np.logspace(-8, 2, 11)
)
"""既定の ridge alpha 格子 (仕様 T3)。全手法・全タスクがこの単一格子を読む (D-04)。"""


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

    ESN の構造ハイパーパラメータは課題ごとに別セクション
    (``esn_mackey_glass`` / ``esn_delay_parity``) に持つ。MG は漏れ積分
    (leak=0.3) が効き、パリティは leak=1.0 が要るため、1つの ``esn`` セクションに
    まとめると片方の課題に不利な値を押し付けることになるため。
    """

    name: str = "01_what_is_rc"
    n_replicates: int = 5
    seeds: SeedConfig = field(default_factory=SeedConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    mackey_glass: MackeyGlassConfig = field(default_factory=MackeyGlassConfig)
    delay_parity: DelayParityConfig = field(default_factory=DelayParityConfig)
    esn_mackey_glass: ESNConfig = field(default_factory=ESNConfig)
    esn_delay_parity: ESNConfig = field(default_factory=_delay_parity_esn)


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


# --- 実験03 (容量: MC / IPC) の設定群 (D-13) ---------------------------------


@dataclass(frozen=True, slots=True)
class CapacitySeedConfig:
    """03 の実験 3-A / 3-B / 3-B' が使う基底シード (D-13 / D-14 と同じ流儀)。

    01 の ``SeedConfig`` / 02 の ``EspSeedConfig`` とは別クラスにする。03 は
    初期状態対 (``PROBE``) を引かない代わりに、しきい値のサロゲート用シードを
    持つ。

    Attributes:
        reservoir: リザバー重みの基底シード (``SeedStream.RESERVOIR``)。
        drive: 駆動入力の基底シード (``SeedStream.TASK``)。
        surrogate: しきい値サロゲートの ``DiagnosticContext.seed`` (D-27 / D-37)。
            **``SeedStream`` ではない**。``make_rng_for`` の ``spawn_key`` 経由
            ではなく、診断へ ``ctx.seed`` としてそのまま渡る整数であり、
            ``seeds.py`` の既存4ストリームの index を1つも動かさない。
            全条件で1個を共有する (共通乱数法、D-37)。
    """

    reservoir: int = 0
    drive: int = 1
    surrogate: int = 4


@dataclass(frozen=True, slots=True)
class CapacityDriveConfig:
    """03 の駆動入力の共通条件 (系列長はセクションが持つ)。

    02 の ``DriveConfig`` を**設定として**再利用しない (仕様 §3.2)。03 は
    比較軌道を引かないので ``n_pairs`` が要らず、``n_steps`` はセクションごとに
    2桁違う (3-A は 2e4、3-B' は 2e5)。**関数** (``simulate_reference_trajectory``)
    は再利用し、値の組み立ては配線層 (``experiment/capacity.py``) が1か所で行う。

    Attributes:
        distribution: 駆動信号の分布。``"uniform"`` 以外は実験層が ``ValueError``
            にする (黙って一様として扱わない)。
        washout: 診断へ渡す ``DiagnosticContext.washout`` [ステップ]。
            容量測定では ``t0 = max(washout, 最大遅延)`` の一方の項になる (D-24)。
    """

    distribution: str = "uniform"
    washout: int = 200


@dataclass(frozen=True, slots=True)
class CapacityReservoirConfig:
    """3-A / 3-B / 3-B' がセクション横断で共有するリザバー構造 (D-32)。

    02 の ``ReservoirSweepConfig`` と違い **``n_units`` を持たない**。03 は
    MC (N=200) と IPC (N=50) で規模が2桁近く違い (D-32: IPC は目標数が
    次数×遅延で増えるため小さいリザバーで回す)、3-B' に至っては ``n_units``
    そのものが掃引軸である。横断で共有できるのはここにある3つだけで、
    ``n_units`` は各セクションが名乗る。

    Attributes:
        input_scale: 入力重みの一様分布の幅 (掃引中は固定、D-17)。
        density: 再帰行列の非零率。
        n_replicates: 各条件のレプリケート数。
    """

    input_scale: float = 1.0
    density: float = 0.1
    n_replicates: int = 3


@dataclass(frozen=True, slots=True)
class McSweepConfig:
    """実験 3-A: rho x リーク率 に対する線形メモリ容量 (受け入れ条件1)。

    ``n_units`` はここが持つ (D-32)。MC の上限は N なので、上限線 y=N を
    引ける規模 (サイクル1・2 と同じ N=200) で回す。
    """

    rho_grid: tuple[float, ...] = (0.5, 0.7, 0.9, 0.95, 1.0, 1.1)
    leak_rate_grid: tuple[float, ...] = (0.3, 0.6, 1.0)
    sigma_u: float = 0.1
    n_units: int = 200
    n_steps: int = 20_000


@dataclass(frozen=True, slots=True)
class IpcSweepConfig:
    """実験 3-B: rho x リーク率 に対する IPC の次数・遅延分解 (受け入れ条件4)。

    ``n_units=50`` は 3-A (N=200) より小さい (D-32)。IPC は目標数が次数と遅延の
    積で増え、必要な系列長も N に対して伸びるため、同じ N では予算に収まらない。
    ``sigma_u`` を 3-A と別に持つのは、次数分解には中程度の駆動が要る一方で
    MC の rho 依存は準線形域を要し、最適な動作点が一致する保証が無いため
    (仕様 §3.2。図をまたいだ絶対値比較はしない)。
    """

    rho_grid: tuple[float, ...] = (0.5, 0.8, 0.95, 1.1)
    leak_rate_grid: tuple[float, ...] = (0.3, 0.6, 1.0)
    sigma_u: float = 0.2
    n_units: int = 50
    n_steps: int = 100_000


@dataclass(frozen=True, slots=True)
class ConservationConfig:
    """実験 3-B': ノイズ下での保存則 IPC_total <= N (受け入れ条件2)。

    ``max_delay_by_degree`` は **3-B' でだけ** ``Capacity03Config.ipc`` を
    ``dataclasses.replace`` で上書きする (片方向。3-A / 3-B は ``ipc`` を
    素のまま使う)。保存則は「打ち切りの外に残った容量」が見えないと N に
    届かないため、この実験だけ遅延を深く取る必要がある。逆向き (3-B' の値を
    ``ipc`` の既定にする) にすると 3-B の掃引まで重くなる。

    Attributes:
        n_units_grid: 上限線 y=N と突き合わせるための N の掃引軸。
        state_noise_grid: tanh 内部に加えるガウスノイズの標準偏差の掃引軸。
            0 を含めるとノイズ無しの基準点が同じ図に乗る。
        rho: スペクトル半径 (固定)。
        leak_rate: リーク率 (固定)。
        sigma_u: 駆動信号の標準偏差 (固定)。
        n_steps: 系列長 [ステップ]。
        max_delay_by_degree: 3-B' でだけ使う次数ごとの遅延の打ち切り。
            既定 ``(200, 60, 20, 10)`` の目標数は 4,075 本 / heatmap 800 セル
            (``count_targets`` で実測) で、``ipc.max_targets`` (200,000) に対し
            49 倍・250 倍の余裕がある。
    """

    n_units_grid: tuple[int, ...] = (25, 50, 100)
    state_noise_grid: tuple[float, ...] = (0.0, 0.01, 0.1)
    rho: float = 0.95
    leak_rate: float = 1.0
    sigma_u: float = 0.2
    n_steps: int = 200_000
    max_delay_by_degree: tuple[int, ...] = (200, 60, 20, 10)


@dataclass(frozen=True, slots=True)
class LengthSweepConfig:
    """系列長 T に対する容量の飽和 (``make saturation-03``、本番には含めない)。

    「容量が足りないのか T が足りないのか」を分けるための補助実験であり、
    ``make figures-03`` の予算 (900 秒) の外で手動実行する。
    """

    n_steps_grid: tuple[int, ...] = (100_000, 200_000, 500_000, 1_000_000)
    rho: float = 0.95
    leak_rate: float = 1.0
    sigma_u: float = 0.2
    n_units: int = 50


@dataclass(frozen=True, slots=True)
class Narma10Config:
    """実験 3-C: 公平な対照下での NARMA10 (D-29 / D-31 / D-39)。

    実体は ``base`` の課題を差し替えて 01 の ``run_task`` を回す経路なので、
    公平性 (D-04 / D-05 / D-08) は既存経路が担保する。係数と入力分布は
    ``tasks/narma.py`` のモジュール定数であり設定にしない (D-29) ので、
    ここが持つのは系列長と 01 側の土台だけである
    (``WashoutSweepConfig.base`` と同じ内包の形)。

    Attributes:
        length: NARMA10 系列の長さ [ステップ]。
        base: 掃引の土台となる 01 の設定 (課題以外は差し替えない)。
    """

    length: int = 8000
    base: ExperimentConfig = field(default_factory=ExperimentConfig)


@dataclass(frozen=True, slots=True)
class Capacity03Config:
    """実験03 (メモリ容量・情報処理容量) 1本ぶんの設定 (D-13)。

    ``ExperimentConfig`` / ``Esp02Config`` とはローダ (``load_config_as``) だけを
    共有し、フィールドは一切共有しない。3-C だけは 01 のパイプラインを再利用
    するため、``narma.base`` として ``ExperimentConfig`` をまるごと内包する
    (``narma.base.*`` の配線は 01 側の ``tests/test_config_wiring.py`` が被覆する。
    ``WashoutSweepConfig.base`` と同じ形)。

    ``mc`` / ``ipc`` は診断層の設定 (D-15) をそのまま載せたもので、YAML から
    容量測定の判定基準まで届くのはこの経路だけである。効きの実測は
    ``tests/test_diagnostics_memory_capacity.py`` /
    ``tests/test_diagnostics_ipc.py`` (3a) に委譲する。

    ``reservoir`` が持つのはセクション横断で共有する3つだけで、``n_units`` は
    セクションが名乗る (D-32。理由は ``CapacityReservoirConfig`` を参照)。
    """

    name: str = "03_capacity"
    seeds: CapacitySeedConfig = field(default_factory=CapacitySeedConfig)
    drive: CapacityDriveConfig = field(default_factory=CapacityDriveConfig)
    reservoir: CapacityReservoirConfig = field(default_factory=CapacityReservoirConfig)
    mc_sweep: McSweepConfig = field(default_factory=McSweepConfig)
    ipc_sweep: IpcSweepConfig = field(default_factory=IpcSweepConfig)
    conservation: ConservationConfig = field(default_factory=ConservationConfig)
    length_sweep: LengthSweepConfig = field(default_factory=LengthSweepConfig)
    mc: MemoryCapacityConfig = field(default_factory=MemoryCapacityConfig)
    ipc: IpcConfig = field(default_factory=IpcConfig)
    narma: Narma10Config = field(default_factory=Narma10Config)


def _fail(location: str, message: str) -> ConfigError:
    return ConfigError(f"{location}: {message}")


def _coerce_scalar(value: object, target: type, location: str) -> object:
    """スカラ値を目標の型へ変換する。暗黙の切り捨てや bool→int は許さない。"""
    if target is bool:
        if isinstance(value, bool):
            return value
        raise _fail(location, f"真偽値が必要です: {value!r}")
    if target is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _fail(location, f"整数が必要です: {value!r}")
        return value
    if target is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _fail(location, f"数値が必要です: {value!r}")
        return float(value)
    if target is str:
        if not isinstance(value, str):
            raise _fail(location, f"文字列が必要です: {value!r}")
        return value
    raise _fail(location, f"未対応の設定型です: {target!r}")


def _coerce_tuple(value: object, annotation: object, location: str) -> object:
    """``tuple[X, ...]`` 型のフィールドを構築する。"""
    args = get_args(annotation)
    if len(args) != 2 or args[1] is not Ellipsis:
        raise _fail(location, f"未対応の tuple 型です: {annotation!r}")
    element_type = args[0]
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise _fail(location, f"リストが必要です: {value!r}")
    return tuple(
        _coerce_scalar(element, element_type, f"{location}[{index}]")
        for index, element in enumerate(value)
    )


def _coerce(value: object, annotation: object, location: str) -> object:
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return _build(annotation, value, location)
    origin = get_origin(annotation)
    if origin is tuple:
        return _coerce_tuple(value, annotation, location)
    if origin is UnionType:
        raise _fail(location, f"未対応の Union 型です: {annotation!r}")
    if isinstance(annotation, type):
        return _coerce_scalar(value, annotation, location)
    raise _fail(location, f"未対応の設定型です: {annotation!r}")


def _build[T](cls: type[T], raw: object, location: str) -> T:
    """dataclass ``cls`` を ``raw`` (マッピング) から構築する。"""
    if not isinstance(raw, Mapping):
        raise _fail(location, f"マッピングが必要です: {raw!r}")
    known = {f.name for f in dataclasses.fields(cast("type", cls))}
    provided = {str(key) for key in raw}
    unknown = sorted(provided - known)
    if unknown:
        raise _fail(
            location,
            f"未知のキーです: {', '.join(unknown)}"
            f" (既知のキー: {', '.join(sorted(known))})",
        )
    hints = get_type_hints(cls)
    kwargs = {
        str(key): _coerce(value, hints[str(key)], f"{location}.{key}")
        for key, value in raw.items()
    }
    factory = cast("_DataclassFactory[T]", cls)
    return factory(**kwargs)


def load_config_as[T](path: Path | str, cls: type[T]) -> T:
    """YAML から任意の設定 dataclass ``cls`` を読み込む (D-13)。

    実験ごとに設定クラスは分かれるが、読み込み規律 (未知キーで即失敗・暗黙の
    型変換をしない・再帰構築) は1か所に置く。02 以降の実験がローダを写経すると
    D-09 の強度が実験ごとに割れるため。

    Args:
        path: YAML ファイルのパス。
        cls: 構築する設定 dataclass。

    Raises:
        ConfigError: ファイルが無い / 未知キーがある / 型が合わない場合。
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path}: YAML の解析に失敗しました: {exc}") from exc
    if raw is None:
        raw = {}
    return _build(cls, raw, str(config_path))


def load_config(path: Path | str) -> ExperimentConfig:
    """YAML から 01 の ``ExperimentConfig`` を読み込む。

    ``load_config_as(path, ExperimentConfig)`` への委譲。既存の呼び出し
    (``experiments/01_what_is_rc/run.py`` / ``main.py``) を壊さないため署名は
    そのまま残す。

    Raises:
        ConfigError: ファイルが無い / 未知キーがある / 型が合わない場合。
    """
    return load_config_as(path, ExperimentConfig)


__all__ = [
    "DEFAULT_ALPHA_GRID",
    "DEFAULT_ESP_MAP_RHO_GRID",
    "DEFAULT_ESP_MAP_SIGMA_GRID",
    "TASK_LENGTH_FIELDS",
    "Capacity03Config",
    "CapacityDriveConfig",
    "CapacityReservoirConfig",
    "CapacitySeedConfig",
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
    "IpcConfig",
    "IpcSweepConfig",
    "LengthSweepConfig",
    "LyapunovConfig",
    "MackeyGlassConfig",
    "McSweepConfig",
    "MemoryCapacityConfig",
    "Narma10Config",
    "ReservoirSweepConfig",
    "RidgeConfig",
    "SplitConfig",
    "TimescaleConfig",
    "TimescaleSweepConfig",
    "WashoutSweepConfig",
    "esp_stream_seed",
    "load_config",
    "load_config_as",
]

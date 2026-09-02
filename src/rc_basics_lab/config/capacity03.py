"""実験03 (メモリ容量・情報処理容量) の設定 dataclass 群 (D-13).

01 の ``ExperimentConfig`` / 02 の ``Esp02Config`` とはローダ
(``_common.load_config_as``) だけを共有し、フィールドは一切共有しない。
``mc`` / ``ipc`` は診断層の設定 (D-15) をそのまま載せたもので、YAML から容量
測定の判定基準まで届くのはこの経路だけである。

``experiment01`` への import は ``Narma10Config.base`` (3-C が 01 の ``run_task``
を再利用するための内包、D-31) の1本だけで、**一方向**である (D-49)。
``esp02`` はここから import しない。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rc_basics_lab.config.experiment01 import ExperimentConfig
from rc_basics_lab.diagnostics.ipc import IpcConfig
from rc_basics_lab.diagnostics.memory_capacity import MemoryCapacityConfig
from rc_basics_lab.reservoir.topology import TopologyConfig


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
    次数 x 遅延で増えるため小さいリザバーで回す)、3-B' に至っては ``n_units``
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
        n_replicates: このセクションだけのレプリケート数。``None`` なら
            ``Capacity03Config.reservoir.n_replicates`` (横断共有) を継承する。
            **``max_delay_by_degree`` とは継承規則が異なる** (F-3b1-1-005)。
            『3-B' 以外へは効かない片方向』という点は同じだが、
            ``max_delay_by_degree`` は非 ``Optional`` で**常に**
            ``config.ipc`` を上書きする (継承の選択肢が無い) のに対し、
            ``n_replicates`` は ``int | None`` で **``None`` のときだけ**
            横断共有値を継承する opt-in の上書きである。この差は意図的:
            打ち切り (``max_delay_by_degree``) は保存則を測るための 3-B' の
            **定義そのもの**なので既定値を持たせて常に効かせる必要があるが、
            レプリケート数は予算超過時にだけ引く**縮退のノブ**であり、既定
            (``None``) では 3-A / 3-B と同じ統計的信頼度を保ちたい。3-B' は
            予算 400 秒と3実験で最も重いので、仕様 §7 リスク1 の縮退規則
            (「予算超過時に許可される調整は ``conservation.n_replicates``
            を 3 → 1 に落とすことだけ」) のノブがここだけに効く必要がある。
            横断共有 (``reservoir.n_replicates``) を 1 に落とすと 3-A / 3-B の
            平均 +- s.d. まで消えるため、縮退の意味が変わる。
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
    n_replicates: int | None = None


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
    topologies: tuple[TopologyConfig, ...] = ()
    """振るトポロジ。**空なら横断共有の density (Erdos-Renyi) 1本だけ** (D-137)。

    「ハブ型は飽和に必要な T が長いかもしれない」を測るための軸である。もし
    そうなら、同じ T で ER と BA を比べた文献は BA を過小評価していることに
    なるので、**トポロジの結論を出す前に**確かめる必要がある。

    既定を空にしてあるので、``make saturation-03`` の既存の成果物は動かない。
    """


@dataclass(frozen=True, slots=True)
class SymmetrySweepConfig:
    """実験 3-S: 駆動入力の対称性と IPC の偶数次 (``make symmetry-03``、D-116)。

    「偶数次の容量が空なのは、入力がゼロ対称で tanh が奇関数だから」という
    仮説を**行の値で**確かめるための補助実験。``make figures-03`` の予算の外で
    手動実行する (``length_sweep`` と同じ扱い)。

    振るのは**平均のずれだけ**で、分布の形 (一様) も標準偏差 (``sigma_u``) も
    変えない。一様のままなので ``orthonormal_basis`` が実測の平均・標準偏差で
    標準化したあとも Legendre 基底は厳密に正規直交で、D-28 を満たしたまま
    測れる (**分布の形を歪めると容量が二重計上され、比較が無意味になる**)。

    Attributes:
        offset_ratio_grid: 駆動入力に加える定数を ``sigma_u`` の倍数で並べたもの。
            ``0.0`` がゼロ対称の基準点。
        rho: スペクトル半径。
        leak_rate: リーク率。
        sigma_u: 駆動信号の標準偏差 (D-17)。
        n_units: リザバーのユニット数 N。
        n_steps: 系列長 [ステップ]。
        n_replicates: レプリケート数。
    """

    offset_ratio_grid: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0)
    rho: float = 0.95
    leak_rate: float = 1.0
    sigma_u: float = 0.2
    n_units: int = 50
    n_steps: int = 30_000
    n_replicates: int = 3


@dataclass(frozen=True, slots=True)
class Narma10Config:
    """実験 3-C: 公平な対照下での NARMA10 (D-29 / D-31 / D-39)。

    実体は ``base`` の課題を差し替えて 01 の ``run_task`` を回す経路なので、
    公平性 (D-04 / D-05 / D-08) は既存経路が担保する。ここが持つのは系列長と
    01 側の土台だけである (D-29。``WashoutSweepConfig.base`` と同じ内包の形)。

    Attributes:
        length: NARMA10 系列の長さ [ステップ]。
        base: 掃引の土台となる 01 の設定 (課題以外は差し替えない)。
        n_lags_sweep: 3-C' のタップ数掃引 (D-95)。**昇順・重複なし**。
            各点は独立に ``plan_replicate`` を通るので ``t0`` も分割も
            その k のものになる (``base.ridge.n_lags_grid`` とは別軸で、
            そちらは 3-C 本体が検証分割で1つ選ぶための候補列である)。
            空なら掃引を回さない。
        n_replicates_sweep: 3-C' のレプリケート数。``None`` なら
            ``base.n_replicates`` を継承する。
    """

    length: int = 8000
    base: ExperimentConfig = field(default_factory=ExperimentConfig)
    n_lags_sweep: tuple[int, ...] = ()
    n_replicates_sweep: int | None = None


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
    symmetry_sweep: SymmetrySweepConfig = field(default_factory=SymmetrySweepConfig)
    mc: MemoryCapacityConfig = field(default_factory=MemoryCapacityConfig)
    ipc: IpcConfig = field(default_factory=IpcConfig)
    narma: Narma10Config = field(default_factory=Narma10Config)

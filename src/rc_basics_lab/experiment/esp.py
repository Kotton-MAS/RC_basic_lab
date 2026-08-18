"""実験 2-A / 2-B / 2-C の配線層 —— ESN と診断層をつなぐ唯一の場所.

``diagnostics/`` は行列を受け取って ``DiagnosticResult`` を返すだけで、どの行列を
渡すかは実験層が決める (01 の ``experiment/state_space.py`` と同じ分業)。
このモジュールは ``reservoir`` と ``diagnostics`` の**両方**を import してよい
唯一の場所であり、D-12 (``diagnostics`` は ``reservoir`` を知らない) はここで
アダプタを持つことで成立している。

3つの実験は「1条件 = (rho, leak_rate, sigma_u, replicate)」という同じ単位に
分解でき、``evaluate_condition`` がその1条件を回す。掃引の違いはどの軸を振るか
だけである。

**入力強度は駆動信号の標準偏差 sigma_u で定義する** (D-17)。一様分布では振幅
``a = sqrt(3) * sigma_u``。掃引中 ``input_scale`` は固定し、動かすのは信号側
だけである (両方動かすと「信号を強くした」のか「重みを大きくした」のかを
分離できない)。CSV には ``sigma_u`` / ``input_amplitude`` /
``input_drive_std`` の3列を出し、指定値と実測値の食い違いが見えるようにする。

**状態行列の計算と、それを使う診断・回帰は別レイヤー** (F-1-005 / F-1-010)。
``simulate_reference_trajectory`` は ``ReservoirSweepConfig`` + ``DriveConfig``
+ 基底シードだけで状態行列 ``X`` を1回計算して返す。02 (ESP) はこれに比較
軌道 n_pairs 本を足す薄い層 (``simulate_condition``) を経由するが、03
(MC/IPC) のように「同じ ``X`` に対して遅延・次数ごとに読み出し回帰だけを
繰り返す」設計は ``simulate_reference_trajectory`` を1条件につき1回呼び、
返ってきた ``states`` を使い回すことで、``ESN.run`` (T に線形) の再実行を
避けられる。02 の ``_sweep`` は1条件=1回の ``evaluate_condition``呼び出しで
十分なため (本番 336 条件・実測 53.65秒)、逐次ループのままで良いが、03 の
想定規模 (delay x degree で 2395 通り) にこのパターンをそのまま複製すると
``ESN.run`` の再実行コストが支配的になる (実測: N=200, T=1e6 で約4.7秒/回。
2395 回なら約3.1時間)。
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, fields

import numpy as np

from rc_basics_lab.config import (
    DriveConfig,
    Esp02Config,
    ReservoirSweepConfig,
    esp_stream_seed,
)
from rc_basics_lab.diagnostics.base import DiagnosticContext, StatePropagator
from rc_basics_lab.diagnostics.esp import conditional_lyapunov, esp_convergence
from rc_basics_lab.diagnostics.timescale import autocorrelation_time
from rc_basics_lab.reservoir.esn import ESN, ESNConfig
from rc_basics_lab.seeds import SeedStream, make_rng_for
from rc_basics_lab.types import FloatArray

logger = logging.getLogger(__name__)

UNIFORM = "uniform"
"""``DriveConfig.distribution`` が受理する唯一の値 (仕様 §8 の前提)。"""

EXPERIMENT_DECAY = "2A_decay"
EXPERIMENT_TIMESCALE = "2B_timescale"
EXPERIMENT_ESP_MAP = "2C_esp_map"

BIAS_SCALE = 0.0
"""02 の ESN が使うバイアス幅。**0 に固定する** (実装メモ / Q2)。

``ESNConfig`` の既定は 0.1 だが、定数バイアスは ``[1; u]`` の先頭成分に掛かる
**振幅一定の入力そのもの**であり、``sigma_u = 0`` を「無入力」と呼べなくなる。
実測: ``bias_scale=0.1`` では無入力・rho=1.2 でも2軌道が収束してしまい、
受け入れ条件1 (「無入力で rho>1 なら非収束」) が成立しない。D-17 が入力強度を
駆動信号の標準偏差で定義している以上、その定義に入らない常時入力は 0 にする。
"""

ESP_DISTANCE_WASHOUT = 0
"""``esp_convergence`` に渡す ctx の washout。**0 に固定する** (実装メモ)。

``esp_convergence`` で washout が効くのは減衰率の当てはめ開始位置だけであり、
判定 (末尾 window の中央値) には効かない。一方 2-A は**過渡そのものを見せる
図**であって、距離が丸めの床 (``EspConfig.floor``) に届く前に当てはめを始め
ないと減衰率が測れない。実測: 無入力 rho=0.5 の距離は t≈46 で 1e-14 を割る
ため、当てはめ開始が ``washout(200) + fit_skip(50) = 250`` だと当てはめ点が
0 点になり ``decay_rate_per_step`` が ``nan`` になる (rho=0.8 も同様)。
過渡を捨てるのは λ と自己相関の側の要求なので、そちらの ctx にだけ
``drive.washout`` を渡す。当てはめ開始の微調整は ``esp.fit_skip`` が担う。
"""

BOUNDARY_LAMBDA = 0.01
"""λ の符号と ESP 判定の整合を問わない境界帯の幅 (仕様 T3 受け入れ基準)。"""

STRONG_DRIVE_SIGMA = 0.5
"""この強度以上の駆動では λ の符号と ESP 判定が完全に一致する (D-20 予定)。

実測 (本番格子 369 行のうち境界近傍を除く 332 件): ``sigma_u >= 0.5`` の
158 条件で不一致 0 件。不一致 27 件はすべて ``sigma_u <= 0.2`` かつ
``rho in [1.1, 1.6]`` に限局し、向きも「λ<0 なのに非収束」の一方向だけである。
"""

_N_INPUTS = 1
"""駆動入力の次元 (i.i.d. 一様乱数の1系列)。"""

_SQRT3 = math.sqrt(3.0)
"""一様分布 ``U[-a, a]`` の標準偏差 ``a / sqrt(3)`` からの換算係数 (D-17)。"""


def make_drive(
    sigma: float,
    n_steps: int,
    rng: np.random.Generator,
    *,
    distribution: str = UNIFORM,
) -> FloatArray:
    """i.i.d. 駆動入力 ``(n_steps, 1)`` を作る。**強度は標準偏差** (D-17)。

    Args:
        sigma: 駆動信号の標準偏差 sigma_u。``0.0`` は厳密なゼロ系列。
        n_steps: 系列長 [ステップ]。
        rng: 乱数生成器 (``SeedStream.TASK`` から引く)。
        distribution: 分布名。``"uniform"`` 以外は ``ValueError``
            (黙って一様として扱わない。設定値が効かない実験を作らないため)。

    Returns:
        駆動入力 ``(n_steps, 1)``。一様分布なので振幅は ``sqrt(3) * sigma``。

    Raises:
        ValueError: ``distribution`` が未対応、``sigma`` が負、
            ``n_steps`` が 1 未満の場合。
    """
    if distribution != UNIFORM:
        raise ValueError(
            f"未対応の駆動信号の分布です (現在は {UNIFORM!r} のみ): {distribution!r}"
        )
    if sigma < 0.0:
        raise ValueError(f"sigma_u は 0 以上である必要があります: {sigma!r}")
    if n_steps < 1:
        raise ValueError(f"n_steps は 1 以上である必要があります: {n_steps!r}")
    if sigma == 0.0:
        zeros: FloatArray = np.zeros((n_steps, _N_INPUTS), dtype=np.float64)
        return zeros
    amplitude = _SQRT3 * sigma
    drive: FloatArray = rng.uniform(-amplitude, amplitude, (n_steps, _N_INPUTS))
    return drive


def make_initial_states(
    n_units: int, n_pairs: int, rng: np.random.Generator
) -> tuple[FloatArray, ...]:
    """``U[-1, 1]^N`` から ``n_pairs + 1`` 本の初期状態を独立に引く (D-16)。

    先頭が参照軌道の初期状態、残りが比較軌道の初期状態。**片方をゼロ状態に
    しない**。無入力ではゼロが不動点なので、片方を 0 にすると「2軌道の分離」
    ではなく「単一軌道の原点への収束」を測ることになり ESP と別の量に化ける。

    Raises:
        ValueError: ``n_units`` が 1 未満、または ``n_pairs`` が 1 未満の場合。
    """
    if n_units < 1:
        raise ValueError(f"n_units は 1 以上である必要があります: {n_units!r}")
    if n_pairs < 1:
        raise ValueError(f"n_pairs は 1 以上である必要があります: {n_pairs!r}")
    return tuple(
        np.asarray(rng.uniform(-1.0, 1.0, n_units), dtype=np.float64)
        for _ in range(n_pairs + 1)
    )


def esn_propagator(esn: ESN, u: FloatArray) -> StatePropagator:
    """``X[t]`` から ``X[t+1]`` を返す伝播器を作る (D-18)。

    ``ESN.run`` の規約により ``X[t]`` は ``u[t]`` を**処理した後**の状態なので、
    1ステップ進めるのに使う入力は ``u[t + 1]`` である。``u[t]`` を渡すと λ が
    "それらしい値" で出てレビューでは気づけないため、``conditional_lyapunov``
    は既定でこの一致を実行時に検査する (``check_propagator``)。
    """

    def propagate(x: FloatArray, t: int) -> FloatArray:
        return esn.step(x, u[t + 1])

    return propagate


@dataclass(frozen=True, slots=True)
class EspRow:
    """``esp_diagnostics.csv`` の1行。**宣言順が CSV の列順の単一の真実**。

    ``input_scale`` / ``n_units`` / ``density`` は ``Esp02Config.reservoir``
    (``ReservoirSweepConfig``) 由来で、セクション固有の YAML キーではない
    (F-02-1-004)。``washout`` は λ と自己相関に効く値であり、ESP の距離当て
    はめには ``ESP_DISTANCE_WASHOUT`` (=0) が使われる点に注意。
    """

    experiment: str
    replicate: int
    seed_reservoir: int
    seed_drive: int
    seed_probe: int
    rho: float
    leak_rate: float
    input_scale: float
    sigma_u: float
    input_amplitude: float
    input_drive_std: float
    n_units: int
    density: float
    n_steps: int
    washout: int
    window: int
    n_pairs: int
    d_initial: float
    d_tail: float
    converged: int
    decay_rate_per_step: float
    lyapunov_per_step: float
    lyapunov_per_time: float
    tau_1e: float
    tau_censored: float
    tau_integrated: float
    wall_time_s: float


ESP_CSV_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(EspRow))
"""``esp_diagnostics.csv`` の列順 (``EspRow`` の宣言順が単一の真実)。"""


def build_esn_config(
    reservoir: ReservoirSweepConfig,
    rho: float,
    leak_rate: float,
    *,
    state_noise: float = 0.0,
) -> ESNConfig:
    """1条件ぶんの ``ESNConfig`` を組む。掃引軸だけが条件ごとに変わる。

    引数は ``Esp02Config`` 全体ではなく ``ReservoirSweepConfig`` に narrow して
    ある (F-1-005)。本体が読むのは ``reservoir`` の4フィールドと ``BIAS_SCALE``
    だけなので、``Esp02Config`` に型で結合すると 03 (MC/IPC) が ESN 構成を
    再利用するために ``Esp02Config`` を丸ごと写経する羽目になる。

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
        density=reservoir.density,
        state_noise=state_noise,
    )


@dataclass(frozen=True, slots=True)
class ReferenceTrajectory:
    """参照軌道1本と、それを作った ESN・駆動入力 (F-1-005)。

    ``simulate_condition`` (ESP) は比較軌道 n_pairs 本も併せて作るが、03
    (MC/IPC) の読み出し回帰には参照軌道1本があれば足りる。比較軌道を作らない
    ぶん ``simulate_condition`` の ``n_pairs + 1`` 倍の計算 (本番 n_pairs=10 で
    11 倍) を避けられる。
    """

    esn: ESN
    drive: FloatArray
    states: FloatArray


def simulate_reference_trajectory(
    reservoir: ReservoirSweepConfig,
    drive_config: DriveConfig,
    *,
    reservoir_seed: int,
    drive_seed: int,
    rho: float,
    leak_rate: float,
    sigma_u: float,
    replicate: int,
    x0: FloatArray | None = None,
    state_noise: float = 0.0,
) -> ReferenceTrajectory:
    """参照軌道1本を作る (``Esp02Config`` を要らない。F-1-005)。

    ``ReservoirSweepConfig`` + ``DriveConfig`` + 基底シード
    (``reservoir_seed`` / ``drive_seed``) だけで呼べるので、03 (MC/IPC) は
    ``Esp02Config`` の3ストリーム配線 (``EspSeedConfig`` / D-14) を写経せずに
    この関数を再利用できる。比較軌道の初期状態対 (``SeedStream.PROBE``) は
    ESP 判定専用なのでここでは作らない。``x0`` を省略すると ``ESN.run`` の
    既定 (零ベクトル) になる —— MC/IPC の読み出し回帰は washout で過渡を
    捨てる前提なので、初期状態をどこから引くかは主要な関心事ではない。

    ``simulate_condition`` はこの関数へ ``config.seeds.reservoir`` /
    ``config.seeds.drive`` (D-14 の3ストリームのうち2本) と ESP 用の
    ``x0`` (``SeedStream.PROBE`` から引いた初期状態) をそのまま渡す薄い層に
    なっており、既存の成果物 (``results/``) はバイト単位で不変である。

    Args:
        reservoir: リザバー構造 (掃引軸を除く4フィールド)。
        drive_config: 駆動入力の共通条件。
        reservoir_seed: リザバー重みストリームの基底シード。
        drive_seed: 駆動入力ストリームの基底シード。
        rho: スペクトル半径。
        leak_rate: リーク率。
        sigma_u: 駆動信号の標準偏差 (D-17)。
        replicate: レプリケート番号 (0 始まり)。
        x0: 初期状態 ``(N,)``。``None`` なら ``ESN.run`` の既定 (零ベクトル)。
        state_noise: tanh 内部に加えるガウスノイズの標準偏差 (D-36)。**既定値
            つきキーワード**なので 02 の呼び出しは書き換えない。0 のとき
            ``ESN`` は乱数を1個も引かず、既存の成果物はバイト単位で不変である。

    Returns:
        ESN・駆動入力・状態系列 ``(T, N)``。
    """
    drive_rng = make_rng_for(drive_seed, SeedStream.TASK, replicate)
    u = make_drive(
        sigma_u,
        drive_config.n_steps,
        drive_rng,
        distribution=drive_config.distribution,
    )
    reservoir_rng = make_rng_for(reservoir_seed, SeedStream.RESERVOIR, replicate)
    esn = ESN(
        build_esn_config(reservoir, rho, leak_rate, state_noise=state_noise),
        reservoir_rng,
        n_inputs=_N_INPUTS,
    )
    # 重み生成に使った Generator をそのまま状態ノイズにも渡す (reservoir
    # ストリームの続き。01 の ``runner.plan_replicate`` と同じ形)。**常に**
    # 渡すのは、rng を省く分岐が残っていると state_noise > 0 を設定した瞬間に
    # ``ESN.run`` が ValueError になる配線漏れが復活するため (D-36)。
    # state_noise=0 では1個も引かれないので既存の結果は変わらない。
    return ReferenceTrajectory(
        esn=esn, drive=u, states=esn.run(u, x0=x0, rng=reservoir_rng)
    )


@dataclass(frozen=True, slots=True)
class Trajectories:
    """1条件ぶんの軌道と、伝播器を作るのに要る材料。

    ``esp_convergence`` は ``(states, companions)`` の純関数なので、判定基準
    (``EspConfig``) を変えるだけの掃引 (2-C の閾値感度。``experiment/threshold.py``)
    は軌道を作り直す必要が無い。その再利用のために ``evaluate_condition`` から
    シミュレーション部分だけを切り出したのがこの型である。
    """

    esn: ESN
    drive: FloatArray
    states: FloatArray
    companions: tuple[FloatArray, ...]


def simulate_condition(
    config: Esp02Config,
    *,
    rho: float,
    leak_rate: float,
    sigma_u: float,
    replicate: int,
) -> Trajectories:
    """1条件 (rho, leak_rate, sigma_u, replicate) の軌道を作る。

    参照軌道の生成は ``simulate_reference_trajectory`` に委譲する (F-1-005)。
    ここで足すのは ESP 専用の比較軌道 n_pairs 本と、その初期状態対 (D-14 の
    ``PROBE`` ストリーム) だけである。乱数は3ストリームに分ける (D-14)。
    リザバー重み ``RESERVOIR`` / 駆動信号 ``TASK`` / 初期状態対 ``PROBE`` が
    独立なので、「初期状態だけを振ったときに判定が変わるか」を重みを固定した
    まま測れる。
    """
    probe_rng = make_rng_for(
        esp_stream_seed(config.seeds, SeedStream.PROBE), SeedStream.PROBE, replicate
    )
    initial_states = make_initial_states(
        config.reservoir.n_units, config.drive.n_pairs, probe_rng
    )
    reference = simulate_reference_trajectory(
        config.reservoir,
        config.drive,
        reservoir_seed=esp_stream_seed(config.seeds, SeedStream.RESERVOIR),
        drive_seed=esp_stream_seed(config.seeds, SeedStream.TASK),
        rho=rho,
        leak_rate=leak_rate,
        sigma_u=sigma_u,
        replicate=replicate,
        x0=initial_states[0],
    )
    return Trajectories(
        esn=reference.esn,
        drive=reference.drive,
        states=reference.states,
        companions=tuple(
            reference.esn.run(reference.drive, x0=x0) for x0 in initial_states[1:]
        ),
    )


@dataclass(frozen=True, slots=True)
class ConditionOutcome:
    """1条件ぶんの結果。行に加えて図が必要とする曲線を持つ。

    仕様 §4 T3 は ``evaluate_condition -> EspRow`` と書いていたが、2-A の図は
    距離曲線そのものが主役であり、行だけ返すと図のために全条件をもう一度
    回すことになる (実測 77 秒が倍になる)。``row`` が ``EspRow`` である以上
    CSV 列順の単一の真実は変わらない。
    """

    row: EspRow
    distance: FloatArray
    acf: FloatArray


def evaluate_condition(
    config: Esp02Config,
    *,
    experiment: str,
    rho: float,
    leak_rate: float,
    sigma_u: float,
    replicate: int,
) -> ConditionOutcome:
    """1条件 (rho, leak_rate, sigma_u, replicate) を回して3診断を取る。

    乱数は3ストリームに分ける (D-14)。リザバー重み ``RESERVOIR`` / 駆動信号
    ``TASK`` / 初期状態対 ``PROBE`` が独立なので、「初期状態だけを振ったときに
    判定が変わるか」を重みを固定したまま測れる。

    Args:
        config: 02 の設定。
        experiment: CSV の ``experiment`` 列 (``EXPERIMENT_*``)。
        rho: スペクトル半径。
        leak_rate: リーク率。
        sigma_u: 駆動信号の標準偏差 (D-17)。
        replicate: レプリケート番号 (0 始まり)。

    Returns:
        ``EspRow`` と、2-A / 2-B の図が使う距離曲線・自己相関曲線。
    """
    started = time.perf_counter()
    drive_config = config.drive
    reservoir_config = config.reservoir

    trajectories = simulate_condition(
        config, rho=rho, leak_rate=leak_rate, sigma_u=sigma_u, replicate=replicate
    )
    u = trajectories.drive
    states = trajectories.states

    # ESP の距離当てはめは過渡そのものが測定対象なので washout を 0 にする。
    # λ と自己相関は過渡を捨てたいので drive.washout を使う (両者は別の要求)。
    distance_ctx = DiagnosticContext(
        washout=ESP_DISTANCE_WASHOUT, companion_states=trajectories.companions
    )
    dynamics_ctx = DiagnosticContext(
        washout=drive_config.washout,
        propagator=esn_propagator(trajectories.esn, u),
    )
    esp = esp_convergence(states, ctx=distance_ctx, cfg=config.esp)
    lyapunov = conditional_lyapunov(states, ctx=dynamics_ctx, cfg=config.lyapunov)
    timescale = autocorrelation_time(states, ctx=dynamics_ctx, cfg=config.timescale)

    row = EspRow(
        experiment=experiment,
        replicate=replicate,
        seed_reservoir=config.seeds.reservoir,
        seed_drive=config.seeds.drive,
        seed_probe=config.seeds.probe,
        rho=rho,
        leak_rate=leak_rate,
        input_scale=reservoir_config.input_scale,
        sigma_u=sigma_u,
        input_amplitude=_SQRT3 * sigma_u,
        input_drive_std=float(np.std(u)),
        n_units=reservoir_config.n_units,
        density=reservoir_config.density,
        n_steps=drive_config.n_steps,
        washout=drive_config.washout,
        window=config.esp.window,
        n_pairs=drive_config.n_pairs,
        d_initial=esp.scalars["d_initial"],
        d_tail=esp.scalars["d_tail"],
        converged=int(esp.scalars["converged"]),
        decay_rate_per_step=esp.scalars["decay_rate_per_step"],
        lyapunov_per_step=lyapunov.scalars["lyapunov_per_step"],
        lyapunov_per_time=lyapunov.scalars["lyapunov_per_time"],
        tau_1e=timescale.scalars["tau_1e"],
        tau_censored=timescale.scalars["tau_censored"],
        tau_integrated=timescale.scalars["tau_integrated"],
        wall_time_s=time.perf_counter() - started,
    )
    logger.debug(
        "experiment=%s rep=%d rho=%.3f leak=%.2f sigma_u=%.3f "
        "converged=%d lambda=%+.4f tau_1e=%.3f (%.3fs)",
        row.experiment,
        row.replicate,
        row.rho,
        row.leak_rate,
        row.sigma_u,
        row.converged,
        row.lyapunov_per_step,
        row.tau_censored,
        row.wall_time_s,
    )
    return ConditionOutcome(
        row=row, distance=esp.arrays["distance"], acf=timescale.arrays["acf"]
    )


def run_decay_sweep(config: Esp02Config) -> tuple[ConditionOutcome, ...]:
    """実験 2-A: rho を振って状態距離の減衰曲線を得る (受け入れ条件1)。"""
    section = config.decay
    return _sweep(
        config,
        EXPERIMENT_DECAY,
        tuple((rho, section.leak_rate, section.sigma_u) for rho in section.rho_grid),
    )


def run_timescale_sweep(config: Esp02Config) -> tuple[ConditionOutcome, ...]:
    """実験 2-B: リーク率を振って実効時定数を得る (受け入れ条件4)。"""
    section = config.timescale_sweep
    return _sweep(
        config,
        EXPERIMENT_TIMESCALE,
        tuple(
            (section.rho, leak_rate, section.sigma_u)
            for leak_rate in section.leak_rate_grid
        ),
    )


def run_esp_map(config: Esp02Config) -> tuple[ConditionOutcome, ...]:
    """実験 2-C: rho x 入力強度 の ESP 成立領域 (受け入れ条件2。記事の目玉)。"""
    section = config.esp_map
    return _sweep(
        config,
        EXPERIMENT_ESP_MAP,
        tuple(
            (rho, section.leak_rate, sigma_u)
            for rho in section.rho_grid
            for sigma_u in section.sigma_grid
        ),
    )


def _sweep(
    config: Esp02Config,
    experiment: str,
    axes: tuple[tuple[float, float, float], ...],
) -> tuple[ConditionOutcome, ...]:
    """``(rho, leak_rate, sigma_u)`` の並びをレプリケートぶん回す共通ループ。"""
    n_replicates = config.reservoir.n_replicates
    if n_replicates < 1:
        raise ValueError(
            f"reservoir.n_replicates は 1 以上である必要があります: {n_replicates}"
        )
    started = time.perf_counter()
    outcomes = tuple(
        evaluate_condition(
            config,
            experiment=experiment,
            rho=rho,
            leak_rate=leak_rate,
            sigma_u=sigma_u,
            replicate=replicate,
        )
        for replicate in range(n_replicates)
        for rho, leak_rate, sigma_u in axes
    )
    logger.info(
        "experiment=%s 条件数=%d (軸=%d x レプリケート=%d) wall_time=%.2fs",
        experiment,
        len(outcomes),
        len(axes),
        n_replicates,
        time.perf_counter() - started,
    )
    return outcomes


@dataclass(frozen=True, slots=True)
class EspResults:
    """3実験ぶんの結果 (図はそれぞれ別の実験の結果だけを見る)。"""

    decay: tuple[ConditionOutcome, ...]
    timescale: tuple[ConditionOutcome, ...]
    esp_map: tuple[ConditionOutcome, ...]

    @property
    def outcomes(self) -> tuple[ConditionOutcome, ...]:
        """3実験ぶんを宣言順につなげたもの。"""
        return self.decay + self.timescale + self.esp_map

    @property
    def rows(self) -> tuple[EspRow, ...]:
        """``esp_diagnostics.csv`` に書く行 (3実験ぶん)。"""
        return tuple(outcome.row for outcome in self.outcomes)


def run_esp_experiment(config: Esp02Config) -> EspResults:
    """実験 2-A / 2-B / 2-C をこの順に回す。"""
    return EspResults(
        decay=run_decay_sweep(config),
        timescale=run_timescale_sweep(config),
        esp_map=run_esp_map(config),
    )


@dataclass(frozen=True, slots=True)
class VerdictAgreement:
    """λ の符号と ESP 判定の整合の要約 (``meta.json`` に載せる)。

    **整合の要求は非対称である** (D-20 予定)。条件付き Lyapunov 指数は参照軌道
    まわりの**局所**量なので、多安定性 (複数の吸引子が共存する状態) を原理的に
    検出できない。tanh は奇関数なので ``x*`` が不動点なら ``-x*`` も不動点であり、
    どちらも局所安定 (λ<0) でありながら初期状態によって行き先が割れる
    (= ESP 不成立)。したがって

    - ``λ > 0`` なのに収束 (**偽の ESP**) は起きてはならない (実測 0 件)
    - ``λ < 0`` なのに非収束は起きうる (実測 27 件。すべて弱駆動・臨界超え)

    どこで起きたかを追えるように sigma_u と rho の分布まで残す。
    """

    boundary_lambda: float
    strong_drive_sigma: float
    n_rows: int
    n_near_boundary: int
    n_compared: int
    n_false_esp: int
    n_local_but_not_global: int
    disagreement_rate: float
    n_compared_strong_drive: int
    n_disagreement_strong_drive: int
    disagreement_by_sigma: tuple[tuple[float, int], ...]
    disagreement_by_rho: tuple[tuple[float, int], ...]

    def to_summary(self) -> dict[str, object]:
        """``meta.json`` に載せるプレーンな dict。"""
        return {
            "boundary_lambda": self.boundary_lambda,
            "strong_drive_sigma": self.strong_drive_sigma,
            "n_rows": self.n_rows,
            "n_near_boundary": self.n_near_boundary,
            "n_compared": self.n_compared,
            "n_false_esp": self.n_false_esp,
            "n_local_but_not_global": self.n_local_but_not_global,
            "disagreement_rate": self.disagreement_rate,
            "n_compared_strong_drive": self.n_compared_strong_drive,
            "n_disagreement_strong_drive": self.n_disagreement_strong_drive,
            "disagreement_by_sigma": [
                {"sigma_u": sigma, "n": count}
                for sigma, count in self.disagreement_by_sigma
            ],
            "disagreement_by_rho": [
                {"rho": rho, "n": count} for rho, count in self.disagreement_by_rho
            ],
        }


def _is_disagreement(row: EspRow) -> bool:
    """境界帯の外で λ の符号と ESP 判定が食い違っているか。"""
    return (row.lyapunov_per_step < 0.0) != (row.converged == 1)


def summarize_verdict_agreement(
    rows: tuple[EspRow, ...],
    *,
    boundary_lambda: float = BOUNDARY_LAMBDA,
    strong_drive_sigma: float = STRONG_DRIVE_SIGMA,
) -> VerdictAgreement:
    """λ の符号と ESP 判定の整合を集計する (受け入れ条件3 の一次資料)。"""
    compared = [
        row
        for row in rows
        if math.isfinite(row.lyapunov_per_step)
        and abs(row.lyapunov_per_step) > boundary_lambda
    ]
    disagreements = [row for row in compared if _is_disagreement(row)]
    strong = [row for row in compared if row.sigma_u >= strong_drive_sigma]
    return VerdictAgreement(
        boundary_lambda=boundary_lambda,
        strong_drive_sigma=strong_drive_sigma,
        n_rows=len(rows),
        n_near_boundary=len(rows) - len(compared),
        n_compared=len(compared),
        n_false_esp=sum(1 for row in disagreements if row.lyapunov_per_step > 0.0),
        n_local_but_not_global=sum(
            1 for row in disagreements if row.lyapunov_per_step < 0.0
        ),
        disagreement_rate=len(disagreements) / len(compared) if compared else 0.0,
        n_compared_strong_drive=len(strong),
        n_disagreement_strong_drive=sum(1 for row in strong if _is_disagreement(row)),
        disagreement_by_sigma=tuple(
            sorted(Counter(row.sigma_u for row in disagreements).items())
        ),
        disagreement_by_rho=tuple(
            sorted(Counter(row.rho for row in disagreements).items())
        ),
    )


def esp_defaults(config: Esp02Config) -> dict[str, object]:
    """02 の判定基準と、YAML に出てこない実装側の固定値 (``meta.json`` 用)。

    T5 が ``docs/design.md`` §9 に既定値の根拠を書くときの一次資料になる。
    YAML から設定できる値 (``config`` 全体) は ``meta.json`` の ``config``
    に別途載るので、ここには**コード側にしか無い値**を明示的に並べる。
    """
    return {
        "bias_scale": BIAS_SCALE,
        "esp_distance_washout": ESP_DISTANCE_WASHOUT,
        "boundary_lambda": BOUNDARY_LAMBDA,
        "strong_drive_sigma": STRONG_DRIVE_SIGMA,
        "drive_distribution": config.drive.distribution,
        "input_amplitude_per_sigma": _SQRT3,
        "esp": {
            "abs_tol": config.esp.abs_tol,
            "rel_tol": config.esp.rel_tol,
            "window": config.esp.window,
            "fit_skip": config.esp.fit_skip,
            "floor": config.esp.floor,
        },
        "lyapunov": {
            "method": config.lyapunov.method,
            "delta": config.lyapunov.delta,
            "renorm_interval": config.lyapunov.renorm_interval,
            "max_growth": config.lyapunov.max_growth,
            "check_propagator": config.lyapunov.check_propagator,
            "propagator_tol": config.lyapunov.propagator_tol,
        },
        "timescale": {"max_lag": config.timescale.max_lag},
    }


__all__ = [
    "BIAS_SCALE",
    "BOUNDARY_LAMBDA",
    "ESP_CSV_COLUMNS",
    "ESP_DISTANCE_WASHOUT",
    "EXPERIMENT_DECAY",
    "EXPERIMENT_ESP_MAP",
    "EXPERIMENT_TIMESCALE",
    "STRONG_DRIVE_SIGMA",
    "UNIFORM",
    "ConditionOutcome",
    "EspResults",
    "EspRow",
    "ReferenceTrajectory",
    "Trajectories",
    "VerdictAgreement",
    "build_esn_config",
    "esn_propagator",
    "esp_defaults",
    "evaluate_condition",
    "make_drive",
    "make_initial_states",
    "run_decay_sweep",
    "run_esp_experiment",
    "run_esp_map",
    "run_timescale_sweep",
    "simulate_condition",
    "simulate_reference_trajectory",
    "summarize_verdict_agreement",
]
